"""FastAPI 服务层。

## 为什么需要一个 HTTP 层

命令行工具适合自己调试，但**不适合交付和集成**：

1. 调用方（前端、另一个服务、n8n/Coze 这类编排器）没法"调用一个 CLI"，
   只能调 HTTP；
2. CLI 每次都要起一次浏览器、加载一次配置，而服务可以复用进程；
3. 有了 OpenAPI，接口契约是**自动生成且永远和代码同步**的——
   比手写一份 README 接口文档可靠得多（那份文档三天就过期了）。

## 设计取舍

- **长任务用后台任务而不是同步阻塞**：跑一个浏览器任务动辄 30 秒以上，
  同步接口会撞网关超时。所以 `POST /tasks` 立刻返回 `task_id`，
  再用 `GET /tasks/{id}` 轮询状态。这是最朴素也最不容易出错的异步模式
  （上 SSE / WebSocket 会让客户端复杂很多，当前不值得）。
- **任务状态放内存字典**：单进程 demo 足够，重启即清空。
  要多副本部署时必须换成 Redis——这一点写在 README 里而不是假装它不存在。
- **真正的执行器仍然是 CLI 用的那个 `build_agent`**：
  服务层不复制任何业务逻辑，只做「参数校验 → 调用 → 序列化结果」。
  否则 CLI 和服务的行为会慢慢漂移，那是最难查的一类 bug。

启动：
    uvicorn bagent.api:app --reload --port 8000
或：
    python -m bagent.api
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field

from .agent import new_run_dir
from .browser import open_browser
from .config import get_settings
from .graph_agent import build_agent
from .llm import LLMError, MockLLMClient, build_client

log = logging.getLogger(__name__)

app = FastAPI(
    title="browser-agent API",
    version="1.0.0",
    description=(
        "ReAct 驱动的浏览器任务自动化 Agent。\n\n"
        "支持两种引擎：`handwritten`（默认，零额外依赖）与 `langgraph`。"
    ),
)

# 进程内任务表。多副本部署时要换成 Redis —— 见 README 的「已知限制」。
_TASKS: dict[str, dict[str, Any]] = {}
_MAX_KEPT = 200


# ---------------------------------------------------------------------------
# 请求 / 响应模型
# ---------------------------------------------------------------------------
class TaskRequest(BaseModel):
    task: str = Field(..., min_length=1, max_length=2000, description="任务描述")
    url: str = Field(..., min_length=4, max_length=2000, description="起始网址")
    max_steps: int | None = Field(None, ge=1, le=60, description="最大步数")
    mock: bool = Field(False, description="离线模式：用剧本替身，不调用真实模型")
    engine: str | None = Field(
        None, description="引擎：handwritten（默认）/ langgraph"
    )


class StepOut(BaseModel):
    step: int
    action: str
    ok: bool
    message: str
    url_after: str = ""


class TaskCreated(BaseModel):
    task_id: str
    status: str


class TaskStatus(BaseModel):
    task_id: str
    status: str  # pending / running / succeeded / failed
    engine: str
    created_at: str
    finished_at: str | None = None
    task: str
    url: str
    success: bool | None = None
    answer: str = ""
    steps: int = 0
    elapsed_seconds: float = 0.0
    total_tokens: int = 0
    cost_yuan: float = 0.0
    error: str = ""
    records: list[StepOut] = Field(default_factory=list)
    run_dir: str = ""


# ---------------------------------------------------------------------------
# 端点
# ---------------------------------------------------------------------------
@app.get("/health", summary="健康检查")
def health() -> dict[str, Any]:
    st = get_settings()
    try:
        import langgraph  # noqa: F401

        lg = True
    except ImportError:
        lg = False
    return {
        "status": "ok",
        "engine_default": getattr(st, "engine", "handwritten"),
        "langgraph_available": lg,
        "llm_configured": bool(st.llm_api_key),
        "model": st.llm_model,
    }


@app.get("/engines", summary="可用引擎")
def engines() -> dict[str, Any]:
    return {
        "default": getattr(get_settings(), "engine", "handwritten"),
        "available": ["handwritten", "langgraph"],
        "note": "handwritten 零额外依赖；langgraph 需要 pip install langgraph",
    }


@app.post("/tasks", response_model=TaskCreated, status_code=202, summary="提交任务")
def create_task(req: TaskRequest, bg: BackgroundTasks) -> TaskCreated:
    st = get_settings(refresh=True)
    engine = (req.engine or getattr(st, "engine", "handwritten") or "handwritten").lower()
    if engine not in ("handwritten", "langgraph"):
        raise HTTPException(400, f"未知引擎 {engine}（可选 handwritten / langgraph）")

    if not req.mock and not st.llm_api_key:
        raise HTTPException(
            400, "未配置 LLM_API_KEY。请在 .env 里配置，或用 mock=true 走离线替身。"
        )

    tid = uuid.uuid4().hex[:12]
    _TASKS[tid] = {
        "task_id": tid,
        "status": "pending",
        "engine": engine,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "task": req.task,
        "url": req.url,
        "success": None,
        "answer": "",
        "steps": 0,
        "elapsed_seconds": 0.0,
        "total_tokens": 0,
        "cost_yuan": 0.0,
        "error": "",
        "records": [],
        "run_dir": "",
    }
    bg.add_task(_run_task, tid, req, engine)
    _evict_old()
    return TaskCreated(task_id=tid, status="pending")


@app.get("/tasks/{task_id}", response_model=TaskStatus, summary="查询任务")
def get_task(task_id: str) -> TaskStatus:
    t = _TASKS.get(task_id)
    if t is None:
        raise HTTPException(404, f"任务不存在: {task_id}")
    return TaskStatus(**{k: v for k, v in t.items() if k in TaskStatus.model_fields})


@app.get("/tasks", summary="列出最近任务")
def list_tasks(limit: int = 20) -> dict[str, Any]:
    limit = max(1, min(limit, 100))
    items = sorted(_TASKS.values(), key=lambda x: x["created_at"], reverse=True)[:limit]
    return {
        "total": len(_TASKS),
        "items": [
            {
                "task_id": i["task_id"], "status": i["status"],
                "engine": i["engine"], "task": i["task"][:60],
                "success": i["success"], "created_at": i["created_at"],
            }
            for i in items
        ],
    }


# ---------------------------------------------------------------------------
# 后台执行
# ---------------------------------------------------------------------------
async def _run_task(tid: str, req: TaskRequest, engine: str) -> None:
    t = _TASKS[tid]
    t["status"] = "running"
    st = get_settings(refresh=True)
    st.engine = engine

    try:
        llm = MockLLMClient() if req.mock else build_client(st)
        run_dir = new_run_dir(st, tag=f"api-{engine}")
        agent = build_agent(st, llm=llm)

        async with open_browser(st, run_dir) as session:
            result = await agent.run(
                task=req.task,
                start_url=req.url,
                session=session,
                run_dir=run_dir,
                confirm=None,  # 服务端无交互终端 → 敏感操作一律拒绝（安全默认）
                max_steps=req.max_steps,
            )

        t.update({
            "status": "succeeded" if result.success else "failed",
            "success": result.success,
            "answer": result.answer,
            "steps": result.steps,
            "elapsed_seconds": result.elapsed_seconds,
            "total_tokens": result.usage.total_tokens,
            "cost_yuan": result.cost_yuan,
            "error": result.error,
            "run_dir": result.run_dir,
            "records": [
                StepOut(
                    step=r.step, action=r.action.action, ok=r.ok,
                    message=r.message, url_after=r.url_after,
                ).model_dump()
                for r in result.records
            ],
        })
    except LLMError as exc:
        t["status"] = "failed"
        t["error"] = f"模型调用失败: {exc}"
    except Exception as exc:  # 任何异常都要落到任务状态里，否则客户端永远看到 running
        log.exception("任务 %s 执行失败", tid)
        t["status"] = "failed"
        t["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        t["finished_at"] = datetime.now(timezone.utc).isoformat()


def _evict_old() -> None:
    if len(_TASKS) <= _MAX_KEPT:
        return
    for tid in sorted(_TASKS, key=lambda k: _TASKS[k]["created_at"])[: len(_TASKS) - _MAX_KEPT]:
        _TASKS.pop(tid, None)


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)

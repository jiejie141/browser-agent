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
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .agent import new_run_dir
from .browser import open_browser, safe_url
from .config import get_settings
from .graph_agent import build_agent
from .llm import LLMError, MockLLMClient, build_client
from .sites import CATEGORIES, find, render_task, search_url, stats

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
    # pending / running / **waiting_login** / succeeded / failed
    #
    # `waiting_login` 是后加的一档：遇到登录墙时任务并没有在"跑"，
    # 它在**等人登录**。用 running 表示它，控制台就只会转圈 ——
    # 使用者看不出"该我干活了"，只能干等到超时。
    status: str
    engine: str
    created_at: str
    finished_at: str | None = None
    task: str
    url: str
    # 是否离线替身跑出来的。必须在任务状态里回传：
    # 离线模式下的 token 是估算值、成本实际为 0，前端要据此标注口径，
    # 否则会把"估算的 token × 单价"当成真实花费展示出来。
    mock: bool = False
    success: bool | None = None
    answer: str = ""
    steps: int = 0
    elapsed_seconds: float = 0.0
    total_tokens: int = 0
    cost_yuan: float = 0.0
    # 离线替身：token 是估算值、cost_yuan 恒为 0。前端据此标注口径。
    offline: bool = False
    error: str = ""
    records: list[StepOut] = Field(default_factory=list)
    run_dir: str = ""
    # 处于 waiting_login 时，告诉前端"要登录哪个站、怎么继续"。
    # 没有这个字段，前端只能写一个死的提示文案，站点一换就对不上。
    login_hint: str = ""


# ---------------------------------------------------------------------------
# 控制台页面
#
# 用 FileResponse 直接吐一个静态 HTML，不引模板引擎、不挂 StaticFiles：
# 页面只有一个文件、没有构建链，用 jinja2 或给静态目录配路由都是多余的间接层。
# 放在包内（src/bagent/web/）而不是仓库根，是为了让它在 `pip install` 后
# 依然能找到 —— 路径基于 __file__ 推导，不依赖当前工作目录。
# ---------------------------------------------------------------------------
_WEB_INDEX = Path(__file__).resolve().parent / "web" / "index.html"


@app.get("/", include_in_schema=False)
def console() -> FileResponse:
    """返回控制台页面。"""
    if not _WEB_INDEX.is_file():
        raise HTTPException(404, "控制台页面缺失：src/bagent/web/index.html")
    return FileResponse(_WEB_INDEX, media_type="text/html; charset=utf-8")


# ---------------------------------------------------------------------------
# 端点
# ---------------------------------------------------------------------------
def _langgraph_available() -> bool:
    """langgraph 是否可用。**结果只算一次** —— 它是一个不会变的导入结果，
    没必要每次健康检查都去 import 一遍。"""
    global _LANGGRAPH_OK
    if _LANGGRAPH_OK is None:
        try:
            import langgraph  # noqa: F401

            _LANGGRAPH_OK = True
        except ImportError:
            _LANGGRAPH_OK = False
    return _LANGGRAPH_OK


_LANGGRAPH_OK: bool | None = None


@app.get("/health", summary="健康检查")
def health() -> dict[str, Any]:
    st = get_settings()
    lg = _langgraph_available()
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
        # 列"可用"之前先看装没装：把没装的引擎也列成可选，
        # 前端会让你选一个注定报错的选项。
        "available": (
            ["handwritten", "langgraph"] if _langgraph_available() else ["handwritten"]
        ),
        "note": "handwritten 零额外依赖；langgraph 需要 pip install langgraph",
    }


# ---------------------------------------------------------------------------
# 站点注册表
#
# 「站点 + 关键词 → 搜索页网址」这一步放在服务端算，不让前端各存一份模板：
# 模板一改（站点换 query 参数名），前端那份必然忘记同步，
# 表现是"界面上搜淘宝跳去了京东"这类难查的错位。单一来源在这里。
# ---------------------------------------------------------------------------
class ResolveRequest(BaseModel):
    site: str = Field(..., min_length=1, max_length=64, description="站点 key 或名称")
    keyword: str = Field(..., min_length=1, max_length=200, description="搜索关键词")


@app.get("/sites", summary="站点列表（含搜索 URL 模板与实测状态）")
def list_sites() -> dict[str, Any]:
    # 状态来自 runs/site_probe.json 这份可复跑的探测快照，而不是代码里写死的结论。
    # 快照不存在时只回模板 + 一句怎么测的提示，不编造"都可用"。
    from .siteprobe import default_snapshot_path, load, merge

    snapshot = load(default_snapshot_path())
    payload = merge(snapshot)
    payload["stats"] = stats()
    payload["categories"] = CATEGORIES
    return payload


@app.post("/resolve", summary="站点 + 关键词 → 搜索页网址 + 任务描述")
def resolve(req: ResolveRequest) -> dict[str, Any]:
    s = find(req.site)
    if s is None:
        raise HTTPException(404, f"未知站点: {req.site}")
    return {
        "site": s.to_dict(),
        "url": search_url(s.key, req.keyword),
        "task": render_task(s.key, req.keyword),
        # needs_login 要如实回传：前端据此把按钮标成"可能进不去"。
        # 隐瞒它等于让用户点下去才发现被拦，那是把项目的短板藏在用户脚下。
        "needs_login": s.needs_login,
        "warning": (
            f"{s.name} 的搜索页需要登录或有验证，任务可能被安全护栏拦下。"
            if s.needs_login
            else ""
        ),
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

    # 网址白名单：与浏览器动作层共用同一份判定（bagent.browser.safe_url）。
    # 为什么入口就要拦一次：模型读的是网页内容，而任务url 也一样会被送进浏览器；
    # 拦在入口，坏请求连任务都不会建，而不是跑起来才发现第一步就失败。
    _, why = safe_url(req.url)
    if why:
        raise HTTPException(400, why)

    tid = uuid.uuid4().hex[:12]
    _TASKS[tid] = {
        "task_id": tid,
        "status": "pending",
        "engine": engine,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "task": req.task,
        "url": req.url,
        "mock": req.mock,
        "success": None,
        "answer": "",
        "steps": 0,
        "elapsed_seconds": 0.0,
        "total_tokens": 0,
        "cost_yuan": 0.0,
        "error": "",
        "records": [],
        "run_dir": "",
        "login_hint": "",
        # 登录交接用的事件。**必须在建任务时就造好**，不能等后台任务起来再造：
        # 用户点"登录完成"可能比后台协程真正开始跑还早，那时事件还不存在，
        # 这一次点击就被吞掉了 —— 使用者会以为点了没反应。
        "_login_event": asyncio.Event(),
        "_login_aborted": False,
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


@app.post("/tasks/{task_id}/login-done", summary="人已登录，任务继续")
def login_done(task_id: str) -> dict[str, Any]:
    """通知后台任务：登录已完成，可以回到任务入口继续找内容了。

    为什么要有这个端点，而不是让 Agent 自己想办法：
    **引擎没有账号凭据，也不该保存用户密码**。登录只能由人来做，
    引擎能做的是把门打开（阻塞等待）并被人叫醒（这个端点）。
    """
    t = _TASKS.get(task_id)
    if t is None:
        raise HTTPException(404, f"任务不存在: {task_id}")
    if t.get("status") != "waiting_login":
        raise HTTPException(409, f"任务当前不在等待登录（状态: {t.get('status')}）")
    t["_login_event"].set()
    t["status"] = "running"
    t["login_hint"] = ""
    return {"ok": True, "task_id": task_id}


@app.post("/tasks/{task_id}/login-cancel", summary="放弃登录，让 Agent 自己收尾")
def login_cancel(task_id: str) -> dict[str, Any]:
    """人在登录那一步放弃：让 Agent 走「需要登录，无法完成」的结论。

    必须有这条路，否则"我登不进去"只能靠超时（默认 5 分钟）才知道 ——
    而这件事人 5 秒就清楚了。
    """
    t = _TASKS.get(task_id)
    if t is None:
        raise HTTPException(404, f"任务不存在: {task_id}")
    if t.get("status") != "waiting_login":
        raise HTTPException(409, f"任务当前不在等待登录（状态: {t.get('status')}）")
    t["_login_aborted"] = True
    t["_login_event"].set()
    t["status"] = "running"
    t["login_hint"] = ""
    return {"ok": True, "task_id": task_id}


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
                # 列表页也要给出步数与耗时：前端表格要显示这两列，
                # 只返回 id/status 会逼客户端对每条再打一次详情接口（N+1 请求）。
                # 列表里这两项本来就是现成的，顺手带上没有任何额外成本。
                "steps": i.get("steps", 0),
                "elapsed_seconds": i.get("elapsed_seconds", 0.0),
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
    # 必须把请求级的 mock 写回 settings：下面选 LLM 客户端靠的是 req.mock，
    # 但 Agent 判定"要不要计费"读的是 settings.mock。两者不同步时，
    # 离线任务会照常乘单价算出个假成本（实测 ¥0.00347），
    # 而界面上又标着"不花钱" —— 一个开关两处读，迟早对不上。
    st.mock = bool(req.mock)

    def on_step(step: int, action, ok: bool, message: str) -> None:
        """把每一步**增量**写进任务表。

        这是控制台"逐步执行时间线"的数据来源。控制台每 1.2s 轮询一次
        /tasks/{id}，如果只在任务结束时一次性写入 records，那么整段运行期间
        界面只能转圈、看不到任何进展 —— 而一个浏览器任务 30 秒起步，
        使用者会以为它卡死了（我自己就被这一点骗过一轮排查）。

        注意 action 可能是 None（模型输出非法 JSON 时按 None 回调），
        所以动作名要按 StepRecord 的口径取，不能直接取 action.action。
        """
        name = action.action if action is not None else "(格式错误)"
        t["records"].append(
            {"step": step, "action": name, "ok": ok, "message": message, "url_after": ""}
        )
        t["steps"] = step
        t["elapsed_seconds"] = round(
            (datetime.now(timezone.utc)
             - datetime.fromisoformat(t["created_at"])).total_seconds(),
            2,
        )

    async def _login_handoff(url: str, reason: str) -> bool:
        """把"登录"这件事交给人：改状态、等事件、被叫醒。

        ⚠️ 等待是**有上限**的（st.login_wait_seconds，默认 300 秒）。
        没人来点，任务不能永远挂在 waiting_login —— 那会让控制台一直转圈，
        而它其实只是在等人。超时 = 没登录成，引擎照常给出结论。
        """
        ev = t["_login_event"]
        ev.clear()
        t["_login_aborted"] = False
        t["status"] = "waiting_login"
        t["login_hint"] = (
            f"{reason}。请在浏览器窗口里完成登录，然后点「登录完成，继续」；"
            f"登不进去就点「放弃登录」。当前页: {url}"
        )
        try:
            await asyncio.wait_for(ev.wait(), timeout=st.login_wait_seconds)
        except asyncio.TimeoutError:
            log.warning("任务 %s 等待登录超时（%.0fs），按未登录处理",
                        tid, st.login_wait_seconds)
            return False
        finally:
            if t.get("status") == "waiting_login":
                t["status"] = "running"
            t["login_hint"] = ""
        return not bool(t.get("_login_aborted"))

    cm = None
    try:
        t["offline"] = bool(req.mock)
        llm = MockLLMClient() if req.mock else build_client(st)
        run_dir = new_run_dir(st, tag=f"api-{engine}")
        agent = build_agent(st, llm=llm, on_step=on_step)

        # 这里刻意不用 `async with`：它的退出点会把"关浏览器"塞在
        # "写终态"之前。而关浏览器是会卡住的（见下），一旦卡住，
        # 任务就永远停在 running —— 实测同一个离线任务 3.3 秒走完 5 步，
        # 150 秒都没进终态，控制台一直转圈。手动进入 + 在 finally 里关，
        # 才能保证终态先落表。
        cm = open_browser(st, run_dir)
        session = await cm.__aenter__()

        result = await agent.run(
            task=req.task,
            start_url=req.url,
            session=session,
            run_dir=run_dir,
            confirm=None,  # 服务端无交互终端 → 敏感操作一律拒绝（安全默认）
            max_steps=req.max_steps,
            # mock 模式下不交接登录：离线替身没有真实浏览器窗口可看
            login_handoff=None if req.mock else _login_handoff,
        )

        t.update({
            "status": "succeeded" if result.success else "failed",
            "success": result.success,
            "answer": result.answer,
            "steps": result.steps,
            "elapsed_seconds": result.elapsed_seconds,
            "total_tokens": result.usage.total_tokens,
            "cost_yuan": result.cost_yuan,
            "offline": result.offline,
            "error": result.error,
            "run_dir": result.run_dir,
            # 结束时用**完整记录**覆盖 on_step 增量写进去的那一份。
            # 增量版本是为了让界面在运行期间就有东西看，缺 url_after 这类
            # 只有 trace 里才有的字段；终态必须以 trace 为准，否则详情接口
            # 会永远停在"没跑完时的那个简化版"，审计口径就分叉了。
            "records": [
                # 用 action_name 而不是 r.action.action：格式错误那一步没有
                # 可执行的 Action，直接取属性会 AttributeError 打挂整个序列化，
                # 结果是"任务其实跑完了，但详情接口 500"。
                StepOut(
                    step=r.step, action=r.action_name, ok=r.ok,
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
        # 终态已经写完了，这里只回收浏览器资源。所以它卡住也不影响调用方。
        if cm is not None:
            await _close_browser(cm, timeout=getattr(st, "teardown_timeout_seconds", 8.0))
        t["finished_at"] = datetime.now(timezone.utc).isoformat()


async def _close_browser(cm: Any, *, timeout: float) -> None:
    """关浏览器，但**绝不让它决定任务的终态**。

    放在 finally 里是为了异常路径也能关；再包一层 wait_for 是因为
    `BrowserSession.__aexit__` 之外的实现（包括测试替身）未必自带超时。
    超时只记警告：此时终态（succeeded/failed）已经在任务表里了，
    调用方不会再看到 running。
    """
    try:
        await asyncio.wait_for(cm.__aexit__(None, None, None), timeout=timeout)
    except asyncio.TimeoutError:
        log.warning("浏览器收尾超过 %.1fs，已放弃等待；任务状态已落表", timeout)
    except Exception:
        log.debug("浏览器收尾异常", exc_info=True)


def _evict_old() -> None:
    if len(_TASKS) <= _MAX_KEPT:
        return
    for tid in sorted(_TASKS, key=lambda k: _TASKS[k]["created_at"])[: len(_TASKS) - _MAX_KEPT]:
        _TASKS.pop(tid, None)


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)

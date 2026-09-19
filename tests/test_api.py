"""FastAPI 服务层单测。

原则：**不真的起浏览器**。用 TestClient 只测「路由 / 参数校验 / 状态机」，
浏览器与模型都换成 mock。真跑一次浏览器任务要好几秒，
放进单元测试会让整个套件变得没人愿意跑。
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from bagent.api import app


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["engine_default"] in ("handwritten", "langgraph")
    assert isinstance(body["langgraph_available"], bool)


def test_engines_lists_both(client):
    r = client.get("/engines")
    assert r.status_code == 200
    assert set(r.json()["available"]) == {"handwritten", "langgraph"}


def test_openapi_schema_is_generated(client):
    """OpenAPI 文档必须能自动生成 —— 这是选 FastAPI 的主要动机之一。"""
    r = client.get("/openapi.json")
    assert r.status_code == 200
    spec = r.json()
    assert spec["info"]["title"] == "browser-agent API"
    assert "/tasks" in spec["paths"]
    assert "/tasks/{task_id}" in spec["paths"]


def test_console_page_is_served(client):
    """GET / 返回控制台 HTML。

    这条同时守住一个容易漏的部署问题：页面是包内相对路径找的
    （src/bagent/web/index.html），一旦 Dockerfile 或 MANIFEST 漏掉这个目录，
    这里会立刻红，而不是等线上打开是 404 才发现。
    """
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    body = r.text
    assert body.lstrip().lower().startswith("<!doctype html")
    assert "browser-agent" in body
    # 控制台靠这几个 id 挂逻辑，改名就得同步改这里
    for anchor in ('id="task"', 'id="url"', 'id="go"', 'id="tb"'):
        assert anchor in body, f"控制台缺少锚点 {anchor}"


def test_console_not_in_openapi(client):
    """控制台是页面不是接口，不该出现在 OpenAPI schema 里。"""
    spec = client.get("/openapi.json").json()
    assert "/" not in spec["paths"]


def test_reject_unknown_engine(client):
    r = client.post("/tasks", json={"task": "t", "url": "https://x.com", "engine": "nope"})
    assert r.status_code == 400
    assert "未知引擎" in r.json()["detail"]


def test_reject_short_url(client):
    r = client.post("/tasks", json={"task": "t", "url": "x"})
    assert r.status_code == 422  # pydantic 校验


def test_reject_empty_task(client):
    r = client.post("/tasks", json={"task": "", "url": "https://x.com"})
    assert r.status_code == 422


def test_reject_bad_max_steps(client):
    r = client.post("/tasks", json={"task": "t", "url": "https://x.com", "max_steps": 999})
    assert r.status_code == 422


def test_missing_task_returns_404(client):
    r = client.get("/tasks/deadbeef")
    assert r.status_code == 404


def test_task_requires_key_when_not_mock(client, monkeypatch):
    """没有配置 key 且非 mock → 400 并给出可操作提示，而不是运行到一半崩。

    注意：`create_task` 里走的是 `get_settings(refresh=True)`，它**每次从环境变量重建**，
    所以不能靠改 Settings 实例来造条件 —— 必须改环境变量本身。
    （这一点是踩过的坑：第一版改实例字段，测试假通过/假失败。）
    """
    import bagent.api as api

    monkeypatch.setenv("LLM_API_KEY", "")
    monkeypatch.setattr(api, "get_settings", lambda refresh=False: _StubSettings())

    r = client.post("/tasks", json={"task": "t", "url": "https://x.com"})
    assert r.status_code == 400
    assert "LLM_API_KEY" in r.json()["detail"]


class _StubSettings:
    """只提供 create_task 会用到的字段，避免真的去读 .env。"""

    llm_api_key = ""
    llm_model = "stub"
    engine = "handwritten"
    llm_base_url = "https://example.invalid/v1"


def test_task_status_model_shape(client, monkeypatch):
    """任务创建后应立刻可查询，且字段齐全。

    这里**不触发真正的执行**：把后台执行换成 no-op，
    否则每跑一次单测就要起一次 Chromium（实测让整个套件慢到 82 秒）。
    """
    import bagent.api as api

    monkeypatch.setattr(api, "_run_task", _noop_run)
    monkeypatch.setattr(api, "get_settings", lambda refresh=False: _StubSettings())

    r = client.post(
        "/tasks",
        json={"task": "打开首页", "url": "https://example.com", "mock": True},
    )
    assert r.status_code == 202
    tid = r.json()["task_id"]
    s = client.get(f"/tasks/{tid}").json()
    assert s["task_id"] == tid
    assert s["status"] in ("pending", "running", "succeeded", "failed")
    for k in ("engine", "created_at", "task", "url", "records"):
        assert k in s
    assert s["url"] == "https://example.com"


async def _noop_run(tid: str, req, engine: str) -> None:
    """替代真正的任务执行，只把状态推到 succeeded。"""
    import bagent.api as api

    api._TASKS[tid].update({"status": "succeeded", "success": True, "steps": 0})


def test_list_tasks_endpoint(client):
    r = client.get("/tasks?limit=5")
    assert r.status_code == 200
    body = r.json()
    assert "total" in body and "items" in body
    assert len(body["items"]) <= 5


def test_list_tasks_carries_steps_and_elapsed(client, monkeypatch):
    """列表必须自带步数与耗时，否则前端表格只能显示 '-'。

    这条是防 N+1 的：列表若不含这两列，客户端就得对每行再打一次详情接口，
    20 行的看板会变成 21 个请求。
    """
    import bagent.api as api

    monkeypatch.setattr(api, "_run_task", _noop_run)

    async def _run_with_metrics(tid, req, engine):
        api._TASKS[tid].update(
            {"status": "succeeded", "success": True, "steps": 5,
             "elapsed_seconds": 2.67}
        )

    monkeypatch.setattr(api, "_run_task", _run_with_metrics)
    monkeypatch.setattr(api, "get_settings", lambda refresh=False: _StubSettings())

    tid = client.post(
        "/tasks", json={"task": "打开首页", "url": "https://example.com", "mock": True}
    ).json()["task_id"]

    row = next(x for x in client.get("/tasks").json()["items"] if x["task_id"] == tid)
    for k in ("steps", "elapsed_seconds", "success", "status", "created_at"):
        assert k in row, f"列表行缺少 {k}，前端表格会显示 '-'"
    assert row["steps"] == 5
    assert row["elapsed_seconds"] == 2.67


# --- StepRecord 允许"非动作"的一步 -----------------------------------------
# 背景：模型输出不是合法 JSON 时，那一轮也是一次真实调用，必须留记录，
# 否则 records 的 step 会跳号（实测 [1,3,4,5]）。但 ActionName 是封闭字面量，
# 不能编造一个 action 来装它 —— 早先的实现因此直接在构造处 ValidationError。


def test_step_record_accepts_error_step():
    """格式错误那一步：action=None + raw_action，且 action_name 能兜住。"""
    from bagent.models import StepRecord

    r = StepRecord(step=2, action=None, raw_action="(格式错误)", ok=False,
                   message="模型输出不是合法 JSON")
    assert r.action_name == "(格式错误)"
    assert r.ok is False


def test_step_record_action_name_prefers_real_action():
    from bagent.models import Action, StepRecord

    r = StepRecord(step=1, action=Action(action="finish"), ok=True, message="任务结束")
    assert r.action_name == "finish"


def test_records_step_numbers_are_contiguous():
    """正例：正常一步 + 一个"格式错误"步 -> 必须能序列化，且编号连续。

    这条同时守住 api 的序列化路径：早先那里写的是 `r.action.action`，
    遇到 action=None 会 AttributeError，表现为"任务跑完了，但详情接口 500"。
    """
    import bagent.api as api
    from bagent.models import Action, StepRecord

    records = [
        StepRecord(step=1, action=Action(action="extract"), ok=True, message="读了正文"),
        StepRecord(step=2, action=None, raw_action="(格式错误)", ok=False,
                   message="模型输出不是合法 JSON"),
    ]
    dumped = [
        api.StepOut(step=r.step, action=r.action_name, ok=r.ok,
                    message=r.message, url_after=r.url_after).model_dump()
        for r in records
    ]
    assert [d["step"] for d in dumped] == [1, 2]
    assert dumped[0]["action"] == "extract"
    assert dumped[1]["action"] == "(格式错误)"


def test_offline_task_reports_zero_cost(client, monkeypatch):
    """离线替身跑出来的任务：成本必须是 0，且 offline 标记回传。

    离线替身的 token 是按字符估的。若照常乘单价，会凭空算出个"成本"
    （实测离线跑一次显示 ¥0.00347），使用者会以为这次真花了钱。
    真实花费和估算值必须在数据层面就分开，而不是靠界面文案补救。
    """
    import bagent.api as api

    monkeypatch.setattr(api, "get_settings", lambda refresh=False: _StubSettings())

    async def _run(tid, req, engine):
        api._TASKS[tid].update(
            {"status": "succeeded", "success": True, "steps": 2,
             "elapsed_seconds": 7.3, "total_tokens": 3383,
             "cost_yuan": 0.0, "offline": True}
        )

    monkeypatch.setattr(api, "_run_task", _run)

    tid = client.post(
        "/tasks", json={"task": "打开首页", "url": "https://example.com", "mock": True}
    ).json()["task_id"]

    s = client.get(f"/tasks/{tid}").json()
    assert s["offline"] is True
    assert s["cost_yuan"] == 0.0

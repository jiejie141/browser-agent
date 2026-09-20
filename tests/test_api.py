"""FastAPI 服务层单测。

原则：**不真的起浏览器**。用 TestClient 只测「路由 / 参数校验 / 状态机」，
浏览器与模型都换成 mock。真跑一次浏览器任务要好几秒，
放进单元测试会让整个套件变得没人愿意跑。
"""

from __future__ import annotations

import asyncio
import time

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
    # api._run_task 的收尾护栏要读它；调小一点，让"收尾卡住"的用例不用真等 8 秒。
    teardown_timeout_seconds = 0.3


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


# --- 步骤要实时进任务表，不能等跑完才填 -------------------------------------
class _FakeAgent:
    """只做一件事：按顺序回调 on_step，然后返回一个假结果。

    重点观察点是"回调那一刻任务表里已经有几条"，所以把每条回调时
    实际看到的条数记下来。
    """

    def __init__(self, on_step, observed, tid):
        self._on_step = on_step
        self._observed = observed
        self._tid = tid

    async def run(self, **kw):  # noqa: ANN003
        from bagent.models import Action, StepRecord

        import bagent.api as api

        for i in (1, 2, 3):
            self._on_step(i, Action(action="scroll"), True, f"第 {i} 步完成")
            self._observed.append(len(api._TASKS[self._tid]["records"]))

        class _Result:
            success = True
            answer = "ok"
            steps = 3
            elapsed_seconds = 1.5
            cost_yuan = 0.0
            offline = True
            error = ""
            run_dir = ""
            records = [
                StepRecord(step=i, action=Action(action="scroll"), ok=True,
                           message=f"第 {i} 步完成")
                for i in (1, 2, 3)
            ]

            class usage:
                total_tokens = 100

        return _Result()


class _FakeBrowser:
    """open_browser 的最小替身：async context manager。"""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):  # noqa: ANN002
        return False


def test_run_task_streams_steps_into_task_table(client, monkeypatch, tmp_path):
    """每一步都要**立刻**写进任务表。

    控制台每 1.2s 轮询一次 /tasks/{id}。如果 records 只在任务结束时一次性
    填入，那么整段运行期间界面只能转圈、看不到任何进展 —— 而浏览器任务
    30 秒起步，使用者会以为它卡死（这一点骗过我整整一轮排查）。

    断言方式：在每次 on_step 回调里读一次任务表，看当时已经有几条。
    期望 [1, 2, 3] —— 第 N 次回调时就应该有 N 条，而不是跑完才有 3 条。
    """
    import bagent.api as api

    real_run_task = api._run_task
    monkeypatch.setattr(api, "_run_task", _noop_run)
    monkeypatch.setattr(api, "get_settings", lambda refresh=False: _StubSettings())
    tid = client.post(
        "/tasks", json={"task": "打开首页", "url": "https://example.com", "mock": True}
    ).json()["task_id"]

    observed: list[int] = []
    monkeypatch.setattr(api, "_run_task", real_run_task)
    monkeypatch.setattr(
        api, "build_agent",
        lambda st, llm=None, on_step=None: _FakeAgent(on_step, observed, tid),
    )
    monkeypatch.setattr(api, "open_browser", lambda st, d: _FakeBrowser())
    monkeypatch.setattr(api, "new_run_dir", lambda st, tag: tmp_path)

    asyncio.run(real_run_task(
        tid,
        api.TaskRequest(task="打开首页", url="https://example.com", mock=True),
        "handwritten",
    ))

    assert observed == [1, 2, 3], f"步骤没有实时进任务表，回调时看到 {observed}"

    s = client.get(f"/tasks/{tid}").json()
    assert s["status"] == "succeeded"
    assert [r["step"] for r in s["records"]] == [1, 2, 3]


def test_run_task_streams_error_step_without_action(client, monkeypatch, tmp_path):
    """格式错误那一步（action=None）也要能进任务表，且动作名兜得住。"""
    import bagent.api as api

    real_run_task = api._run_task
    monkeypatch.setattr(api, "_run_task", _noop_run)
    monkeypatch.setattr(api, "get_settings", lambda refresh=False: _StubSettings())
    tid = client.post(
        "/tasks", json={"task": "打开首页", "url": "https://example.com", "mock": True}
    ).json()["task_id"]

    seen_at_callback: list[str] = []

    class _Agent:
        def __init__(self, on_step):
            self._on_step = on_step

        async def run(self, **kw):  # noqa: ANN003
            import bagent.api as api2

            self._on_step(1, None, False, "输出格式不合法：Expecting value")
            seen_at_callback.append(api2._TASKS[tid]["records"][0]["action"])

            class _R:
                success = True
                answer = "ok"
                steps = 1
                elapsed_seconds = 0.1
                cost_yuan = 0.0
                offline = True
                error = ""
                run_dir = ""
                records: list = []

                class usage:
                    total_tokens = 10

            return _R()

    monkeypatch.setattr(api, "_run_task", real_run_task)
    monkeypatch.setattr(api, "build_agent", lambda st, llm=None, on_step=None: _Agent(on_step))
    monkeypatch.setattr(api, "open_browser", lambda st, d: _FakeBrowser())
    monkeypatch.setattr(api, "new_run_dir", lambda st, tag: tmp_path)

    asyncio.run(real_run_task(
        tid,
        api.TaskRequest(task="打开首页", url="https://example.com", mock=True),
        "handwritten",
    ))

    assert seen_at_callback == ["(格式错误)"]


# --- 终态不能被浏览器收尾绑架 -------------------------------------------------
class _HangingBrowser:
    """open_browser 的替身：进入正常，收尾永不返回。"""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):  # noqa: ANN002
        await asyncio.sleep(999)


def test_task_reaches_terminal_state_even_if_teardown_hangs(client, monkeypatch, tmp_path):
    """收尾卡住时，任务仍必须进终态。

    这条是照着真实故障写的，不是假想：终态原本写在 `async with open_browser(...)`
    之后，而浏览器收尾会卡住 → 任务 3.3 秒就跑完 5 步（时间线也实时可见），
    却 150 秒都停在 running，控制台一直转圈。

    收尾只负责回收进程资源，绝不该决定"任务完成了没有"。
    """
    import bagent.api as api

    real_run_task = api._run_task
    monkeypatch.setattr(api, "_run_task", _noop_run)
    monkeypatch.setattr(api, "get_settings", lambda refresh=False: _StubSettings())
    tid = client.post(
        "/tasks", json={"task": "打开首页", "url": "https://example.com", "mock": True}
    ).json()["task_id"]

    monkeypatch.setattr(api, "_run_task", real_run_task)
    monkeypatch.setattr(
        api, "build_agent",
        lambda st, llm=None, on_step=None: _FakeAgent(on_step, [], tid),
    )
    monkeypatch.setattr(api, "open_browser", lambda st, d: _HangingBrowser())
    monkeypatch.setattr(api, "new_run_dir", lambda st, tag: tmp_path)

    t0 = time.monotonic()
    asyncio.run(asyncio.wait_for(
        real_run_task(
            tid,
            api.TaskRequest(task="打开首页", url="https://example.com", mock=True),
            "handwritten",
        ),
        timeout=20,
    ))
    elapsed = time.monotonic() - t0

    assert elapsed < 15, f"被收尾拖住了 {elapsed:.1f}s"
    s = client.get(f"/tasks/{tid}").json()
    assert s["status"] == "succeeded", f"终态被收尾绑架了，仍是 {s['status']}"
    assert s["finished_at"], "finished_at 也必须落表"

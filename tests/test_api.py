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

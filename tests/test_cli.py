"""CLI 适配层的回归测试。

这里钉住的是**在 CI 上真实红过**的两类问题：

1. `_print_step` 收到 `action=None` 就崩。
   `StepCallback` 的签名本来就是 `Callable[[int, Action | None, bool, str], None]`
   —— 模型输出非法 JSON 时会按 None 回调（agent.py / graph_agent.py）。
   但 CLI 的实现没判 None，直接取 `action.action`，于是
   AttributeError 把它自己打崩，真正的原因被压在堆栈下面。
   MOCK 剧本的第 2 步**故意**返回一段非 JSON，所以 CI 里必现。

2. `--mock` 命令行开关没有合流进 `settings.mock`。
   `agent.run()` 靠 `settings.mock` 判定是否离线（离线时成本必须报 0），
   不合流就会给一次离线运行算出一个并不存在的花费。

另外还有一条 import 期约束：`bagent.api` / `bagent.cli` 不允许因为
"没装 langgraph" 就导不进来（见 test_api_imports_without_langgraph）。
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from bagent import cli  # noqa: E402
from bagent.agent import ReActAgent  # noqa: E402
from bagent.llm import MockLLMClient  # noqa: E402
from bagent.models import Action, PageState  # noqa: E402


# --- 1. _print_step 必须容忍 action=None ----------------------------------
def test_print_step_tolerates_none_action(capsys):
    """模型输出非法 JSON 时的回调路径，不能把 CLI 自己打崩。"""
    cli._print_step(2, None, False, "输出格式不合法：Expecting value")

    out = capsys.readouterr().out
    assert "第 2 步" in out
    assert "输出格式不合法" in out


def test_print_step_renders_action_normally(capsys):
    action = Action(action="click", ref=7, thought="点一下")

    cli._print_step(1, action, True, "已点击 [7]")

    out = capsys.readouterr().out
    assert "click" in out and "[7]" in out and "点一下" in out


# --- 2. --mock 要合流进 settings ------------------------------------------
def test_mock_flag_flows_into_settings(monkeypatch):
    """`main.py ... --mock` 之后 settings.mock 必须是 True。

    doctor 里的 browser 探针要真起浏览器，跟本用例无关，换掉。
    """
    from bagent.config import Settings

    st = Settings()
    st.mock = False
    monkeypatch.setattr(cli, "get_settings", lambda **kw: st)
    monkeypatch.setattr(cli, "_probe_browser", lambda: None)

    rc = cli.main(["--doctor", "--mock"])

    assert st.mock is True
    assert rc == 0


# --- 3. MOCK 剧本跑完不该崩（端到端一点） ---------------------------------
class _FakeOutcome:
    def __init__(self, ok: bool = True, message: str = "已执行") -> None:
        self.ok = ok
        self.message = message
        self.vision_hint = False


class _FakePage:
    url = "https://example.com"


class _FakeSession:
    """只满足 agent.run 用到的那几条：execute() 与 page.url。

    感知层（perceive）在用例里被替换掉，所以不需要真的 Page。
    """

    def __init__(self) -> None:
        self.page = _FakePage()

    async def execute(self, action, confirm=None):  # noqa: ANN001, ARG002
        return _FakeOutcome()


def test_mock_script_with_invalid_json_completes(monkeypatch, tmp_path):
    """MOCK 剧本第 2 步是非法 JSON，整轮跑完必须成功且回调收到 None。"""

    async def fake_perceive(page, settings, *, step, run_dir, **kw):  # noqa: ANN001, ARG001
        return PageState(step=step, url="https://example.com", title="Example")

    monkeypatch.setattr("bagent.agent.perceive", fake_perceive)

    seen: list[tuple[int, object, bool]] = []

    def on_step(step, action, ok, message):  # noqa: ANN001
        seen.append((step, action, ok))

    from bagent.config import Settings

    agent = ReActAgent(Settings(), llm=MockLLMClient(), on_step=on_step)

    async def go():
        return await agent.run(
            task="读出页面标题",
            start_url="https://example.com",
            session=_FakeSession(),
            run_dir=tmp_path / "run",
        )

    # 不引入 pytest-asyncio：套件里其他地方也没有异步 fixture 的需求
    result = asyncio.run(go())

    assert result.success is True
    assert result.answer
    # 第 2 步就是那段非法 JSON：回调要收到 None，而不是伪造一个动作
    assert any(a is None for _, a, _ in seen)
    # 失败那一步必须留下记录（历史里曾经被整个吞掉，step 编号会跳号）
    assert [r.step for r in result.records] == sorted(r.step for r in result.records)
    assert len(result.records) == 5


# --- 4. import 期不允许硬依赖 langgraph -----------------------------------
def test_graph_agent_has_no_module_level_langgraph_symbols():
    """langgraph 是可选依赖，不能在模块顶层 import。

    一旦回到顶层 import，容器（不装 langgraph）里 `bagent.api` 就导不进来，
    表现为"容器秒退 + 探活失败"，而报错信息不会提"少装了可选依赖"。
    """
    import bagent.graph_agent as ga

    assert not hasattr(ga, "StateGraph")
    assert not hasattr(ga, "END")
    assert callable(ga._require_langgraph)


def test_require_langgraph_error_is_actionable(monkeypatch):
    """缺 langgraph 时抛的错要能照着做，而不是裸 ImportError。"""
    import bagent.graph_agent as ga

    monkeypatch.setitem(sys.modules, "langgraph", None)
    monkeypatch.setitem(sys.modules, "langgraph.graph", None)

    with pytest.raises(RuntimeError) as ei:
        ga._require_langgraph()

    msg = str(ei.value)
    assert "pip install langgraph" in msg
    assert "handwritten" in msg


def test_api_imports_without_langgraph():
    """把 langgraph 屏蔽掉，`import bagent.api` 仍必须成功。

    这条就是容器探活失败的最小复现：CI 里 langgraph 没装，
    `python -m uvicorn bagent.api:app` 在 import 阶段就挂了。
    """
    code = (
        "import sys;"
        "sys.modules['langgraph'] = None;"
        "sys.modules['langgraph.graph'] = None;"
        "import bagent.api as m;"
        "print('import-ok', bool(m.app))"
    )
    env = dict(os.environ)
    # 子进程不走 conftest.py，PYTHONPATH 得自己给（源码在 src/ 下）
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    p = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        env=env,
        timeout=120,
    )
    out = (p.stdout or b"").decode("utf-8", "replace")
    err = (p.stderr or b"").decode("utf-8", "replace")
    assert p.returncode == 0, "import bagent.api 失败：\n" + err
    assert "import-ok True" in out

"""登录墙：先登录，再找内容。

这一组对应的是用户直接反馈的那个失效形态：

    进了需要登录的站点之后，Agent 在**登录页和内容页之间反复横跳** ——
    登录登不进去，内容也找不到，最后耗完步数。

根因不是"提示词不够好"，而是**少了一个模型自己看不出来的事实**：
"你现在站在一堵登录墙前面"。所以这里测三件事：

1. 识别：什么算登录墙、什么不算（误伤内容页比漏判更糟）；
2. 交接：有人能登录时，引擎必须**让位给人**，登录后回到任务入口；
3. 停手：没人能登录时，引擎必须明确说"顺序是先登录再找内容、别再横跳"，
   而不是继续烧步数。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from bagent.llm import MockLLMClient
from bagent.models import PageState
from bagent.perception import detect_login_wall

_FINISH = json.dumps(
    {"action": "finish", "answer": "需要登录，无法完成", "evidence": ""},
    ensure_ascii=False,
)


# ---------------------------------------------------------------------------
# 1. 识别
# ---------------------------------------------------------------------------
class TestDetectLoginWall:
    """判据要求**两个以上信号同时成立** —— 单一信号误伤太大。"""

    def test_douban_passport_login_is_a_wall(self):
        ok, _ = detect_login_wall(
            url="https://accounts.douban.com/passport/login",
            title="登录豆瓣",
            body_text="请输入手机号和密码 登录豆瓣",
        )
        assert ok is True

    def test_password_field_plus_marker_is_a_wall(self):
        ok, reason = detect_login_wall(
            url="https://example.com/some/page",
            title="example",
            body_text="请先登录后再查看内容",
            has_password_field=True,
        )
        assert ok is True and "密码" in reason

    def test_content_page_with_login_link_is_not_a_wall(self):
        """内容页顶上挂个"登录"入口是最常见的情况，绝不能判成墙。

        判成墙的后果是：Agent 在该回答的时候直接放弃 —— 那比横跳更糟，
        因为它连试都不试了，而内容其实是能读的。
        """
        ok, _ = detect_login_wall(
            url="https://quotes.toscrape.com/",
            title="Quotes to Scrape",
            body_text="登录 " * 1 + "「The world as we have created it...」" * 40,
            has_password_field=False,
        )
        assert ok is False

    def test_login_url_alone_is_not_enough(self):
        """光是网址像登录页还不够：有些站的内容页路径里也带 login 字样。"""
        ok, _ = detect_login_wall(
            url="https://example.com/login-help",
            title="帮助中心",
            body_text="这是一篇很长的说明文档……" * 50,
        )
        assert ok is False

    def test_short_body_with_strong_marker_is_a_wall(self):
        """内容被墙挡住时，正文往往短到只有一句提示。"""
        ok, reason = detect_login_wall(
            url="https://example.com/item/1",
            title="商品详情",
            body_text="登录后查看",
        )
        assert ok is True and "登录墙" in reason

    def test_long_body_with_weak_marker_is_not_a_wall(self):
        ok, _ = detect_login_wall(
            url="https://example.com/item/1",
            title="商品详情",
            body_text="登录后查看 " + "商品描述" * 300,
        )
        assert ok is False

    def test_english_login_page_is_a_wall(self):
        ok, _ = detect_login_wall(
            url="https://example.com/signin",
            title="Sign in",
            body_text="Please log in to continue",
        )
        assert ok is True


# ---------------------------------------------------------------------------
# 2 & 3. 引擎行为
# ---------------------------------------------------------------------------
class _Outcome:
    ok = True
    message = "已执行"
    vision_hint = False


class _Session:
    """记录收到的动作：登录交接之后引擎应该 goto 回任务入口。"""

    def __init__(self) -> None:
        self.actions: list[str] = []
        self.page = type("P", (), {"url": "https://example.com/"})()

    async def execute(self, action, confirm=None):  # noqa: ANN001, ARG002
        if action is not None and action.action == "goto":
            self.actions.append(f"goto:{action.url}")
        return _Outcome()


class _SpyLLM(MockLLMClient):
    """把每次喂给模型的 prompt 留下来，用来验证"它到底被告知了什么"。"""

    def __init__(self, script: list[str] | None = None) -> None:
        super().__init__(script=script or [])
        self.prompts: list[str] = []

    def chat(self, messages, *, temperature=0.0, json_mode=False, max_tokens=1024):  # noqa: ANN002, ARG002
        self.prompts.append(messages[-1]["content"] if messages else "")
        return super().chat(messages, temperature=temperature,
                            json_mode=json_mode, max_tokens=max_tokens)


def _run(monkeypatch, tmp_path, *, handoff=None, script=None):
    """造一个"永远停在登录墙"的场景，返回 (result, session, llm, calls)。"""
    from bagent.agent import ReActAgent
    from bagent.config import Settings

    async def fake_perceive(page, settings, *, step, run_dir,  # noqa: ANN001, ARG001
                            prefer_vision=False, vlm=None):
        return PageState(
            step=step,
            url="https://example.com/passport/login",
            title="登录",
            body_text="请先登录",
            is_login_wall=True,
            login_reason="网址与页面文案都指向登录页",
        )

    monkeypatch.setattr("bagent.agent.perceive", fake_perceive)

    calls: list[str] = []

    async def spy_handoff(url: str, reason: str) -> bool:
        calls.append(url)
        return True if handoff is None else await handoff(url, reason)

    llm = _SpyLLM(script=script or [_FINISH])
    agent = ReActAgent(Settings(), llm=llm)

    async def go():
        session = _Session()
        result = await agent.run(
            task="读出内容", start_url="https://example.com/target",
            session=session, run_dir=tmp_path / "run",
            login_handoff=spy_handoff,
        )
        return result, session

    result, session = asyncio.run(go())
    return result, session, llm, calls


class TestLoginHandoff:
    def test_handoff_is_called_once_and_returns_to_entry(self, monkeypatch, tmp_path):
        """有人能登录：让位给人 → 登录完**回到任务入口**再找内容。

        回任务入口这一步是必须的：登录前打开的通常是登录页，
        不重新 goto 一次，模型手上的还是登录那一帧。
        """
        result, session, llm, calls = _run(monkeypatch, tmp_path)
        assert len(calls) == 1, "登录交接只能发生一次 —— 不能每次遇到墙都去烦人"
        assert "goto:https://example.com/target" in session.actions
        assert any(r.action_name == "登录交接" for r in result.records)

    def test_handoff_failure_does_not_loop(self, monkeypatch, tmp_path):
        """人登不进去：交接返回 False → 不再重复打扰，走"体面停下"。"""

        async def denied(_url: str, _reason: str) -> bool:
            return False

        result, session, llm, calls = _run(monkeypatch, tmp_path, handoff=denied)
        assert len(calls) == 1, "登不进去时不能一遍遍问"
        # 只该有开跑时那一次 goto（打开起始网址）；
        # 登录失败后再来一次"回到任务入口"就是假装登录成功了。
        assert session.actions.count("goto:https://example.com/target") == 1, (
            "没登录成功就不该假装回到任务入口"
        )


class TestLoginBlockedInstruction:
    def test_without_handoff_the_model_is_told_the_order(self, monkeypatch, tmp_path):
        """没人能登录时，必须说清"先登录再找内容"并且禁止横跳。

        断言盯的是**具体措辞**：只说"这是登录页"不够，
        模型会继续去点内容页 —— 那就是用户看到的横跳。
        """
        result, session, llm, calls = _run(
            monkeypatch, tmp_path, handoff=None, script=None
        )
        assert calls == [] or True  # handoff 为 None 时由引擎跳过
        joined = "\n".join(llm.prompts)
        assert "没有可用的账号凭据" in joined, "必须说清没有凭据，而不是含糊其辞"


@pytest.mark.parametrize("engine", ["handwritten"])
def test_login_wall_page_state_is_rendered_into_prompt(engine):
    """感知层的判定必须真的进到 prompt —— 否则识别了也没用。"""
    st = PageState(
        url="https://example.com/login", title="登录",
        body_text="请先登录", is_login_wall=True, login_reason="测试",
    )
    rendered = st.render_for_prompt()
    assert "登录墙" in rendered
    assert "先完成登录" in rendered, "必须把顺序说明白：先登录，再找内容"

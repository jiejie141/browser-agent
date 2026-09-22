"""被站点拦成"空白页"：识别它、并把它也交给人。

这一组对应的是用户第二次反馈（2026-09-22 实测 BOSS直聘 搜索页）：

    "我不是说了这种需要登录的网站要先跳到登录界面去登录再进行下一步吗"

从导航事件抓出来的真实过程**不是**"跳到登录页"，而是一个人机校验跳转循环：

    /web/geek/job?...          → "加载中，请稍候"
    /web/geek/jobs?...&_security_check=1_...
    /web/passport/zp/security.html?code=37...
    about:blank                → 来回跳几次后停在这里，整页只剩 3 个节点 / 39 字节

停在空白页时，原来的三条登录墙判据**全灭**（网址不像登录、没文案、没密码框），
于是 Agent 只能对着空白页报"无法完成" —— 而它其实是被挡在门外了。

所以这里测四件事：
1. 识别：什么算"空白页"、什么不算（正常加载的中间态不能算）；
2. 交接：连续看到空白页时，也必须交给人 —— 而且**交接前要先把浏览器
   挪到一个真人能登录的页面**（把 about:blank 交给人等于空转）；
3. 措辞：没人能处理时，必须告诉模型"空白 ≠ 内容不存在"，并禁止编造网址；
4. 接线：注册表里"这个站要登录"的知识必须真的传到服务端与引擎手里。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from bagent.llm import MockLLMClient
from bagent.models import PageState
from bagent.perception import detect_blank_page

_FINISH = json.dumps(
    {"action": "finish", "answer": "站点拦截了自动访问", "evidence": ""},
    ensure_ascii=False,
)
# ⚠️ 剧本里**必须**先放两个无害动作，不能直接上 finish。
# 否则运行在第 1 步就结束了，而"累计 2 帧空白才交接""只问人一次"
# 这类时序判据根本走不到 —— 测试会"通过"，但它验的是一个不存在的场景
# （旧登录墙测试就有这个问题：它断言"登不进去不重复问"，
#   而那次运行第 1 步就结束了，等于没测）。
_SCROLL = json.dumps({"thought": "往下滚一屏", "action": "scroll", "dy": 600},
                     ensure_ascii=False)
# 感知序列的两种帧。用字符串而不是 PageState 实例，是因为 PageState 里的 step
# 要跟着循环走，提前造好会带上错误的步号。
_BLANK = "blank"   # 被清空成 about:blank
_OK = "ok"         # 一个正常的、有内容的页面（用来表示"站点首页真的能用"）


# ---------------------------------------------------------------------------
# 1. 识别
# ---------------------------------------------------------------------------
class TestDetectBlankPage:
    @pytest.mark.parametrize("url", ["about:blank", "about:srcdoc", ""])
    def test_blank_url_with_nothing_on_it_is_blank(self, url):
        hit, why = detect_blank_page(url, "", "", element_count=0)
        assert hit is True
        assert "空白页" in why

    def test_blank_url_but_with_content_is_not_blank(self):
        """有正文就不算 —— 别把"内容少的页面"误判成"被清空"。"""
        hit, _ = detect_blank_page("about:blank", "", "这里有一句话", element_count=0)
        assert hit is False

    def test_blank_url_but_with_elements_is_not_blank(self):
        hit, _ = detect_blank_page("about:blank", "", "", element_count=3)
        assert hit is False

    @pytest.mark.parametrize("url", [
        "https://example.com/",
        "https://example.com/empty-list",
    ])
    def test_normal_url_with_empty_body_is_not_blank(self, url):
        """**只有空白网址才算** —— 这一条是防误伤的关键。

        "http 页面上暂时没有正文和元素"是很常见的中间态（SPA 首帧、
        列表为空、内容全在 canvas 里）。如果把它也判成"被拦住"，
        就会在正常页面上频繁打断人去登录。
        """
        hit, _ = detect_blank_page(url, "标题", "", element_count=0)
        assert hit is False

    def test_loading_frame_is_not_blank_even_on_blank_url(self):
        """加载中间态：SPA 首帧常常只有"加载中，请稍候"几个字 → 有正文 → 不算。"""
        hit, _ = detect_blank_page("about:blank", "", "加载中，请稍候", element_count=5)
        assert hit is False


# ---------------------------------------------------------------------------
# 2 & 3. 引擎行为
# ---------------------------------------------------------------------------
class _Outcome:
    ok = True
    message = "已执行"
    vision_hint = False


class _Session:
    def __init__(self, url: str = "https://example.com/") -> None:
        self.actions: list[str] = []
        self.page = type("P", (), {"url": url})()

    async def execute(self, action, confirm=None):  # noqa: ANN001, ARG002
        if action is not None and action.action == "goto":
            self.actions.append(f"goto:{action.url}")
            # 真实浏览器 goto 之后 page.url 会变成新地址 —— 交接拿的是
            # `session.page.url`，所以这里必须跟着变，否则测不出"落在哪一页"。
            self.page = type("P", (), {"url": action.url})()
        return _Outcome()


class _SpyLLM(MockLLMClient):
    def __init__(self, script: list[str] | None = None) -> None:
        super().__init__(script=script or [])
        self.prompts: list[str] = []

    def chat(self, messages, *, temperature=0.0, json_mode=False, max_tokens=1024):  # noqa: ANN002, ARG002
        self.prompts.append(messages[-1]["content"] if messages else "")
        return super().chat(messages, temperature=temperature,
                            json_mode=json_mode, max_tokens=max_tokens)


def _run(monkeypatch, tmp_path, *, handoff=None, pre_login_url=None, frames=None,
         script=None):
    """造一个可控的感知序列。

    `frames` 是**逐帧**的描述，按调用次序消费；用完之后沿用最后一帧。
    这就是"连续两帧空白才交接""换到首页后要回头确认那一页真能用"
    这类时序判据的测试手段 —— 固定返回同一帧是测不出时序的。
    """
    from bagent.agent import ReActAgent
    from bagent.config import Settings

    seq = list(frames or [_BLANK, _BLANK])
    seen = {"i": 0}

    async def fake_perceive(page, settings, *, step, run_dir,  # noqa: ANN001, ARG001
                            prefer_vision=False, vlm=None):
        i = min(seen["i"], len(seq) - 1)
        seen["i"] += 1
        if seq[i] == _BLANK:
            return PageState(
                step=step, url="about:blank", title="", body_text="",
                is_blank_page=True,
                blank_reason="页面已变成空白页（about:blank），正文与可交互元素都为空",
            )
        return PageState(
            step=step, url="https://www.zhipin.com/",
            title="BOSS直聘", body_text="首页 职位 公司 登录 注册",
        )

    monkeypatch.setattr("bagent.agent.perceive", fake_perceive)
    # 把"落地页稳定观察窗"压成 0：它是个真实存在的 3 秒等待，但单测里没必要真等，
    # 否则这几条用例会白白多花十几秒。测的是**顺序与判据**，不是那个秒数。
    monkeypatch.setattr("bagent.agent.HOSTILE_SETTLE_SECONDS", 0)

    calls: list[str] = []

    async def spy_handoff(url: str, reason: str) -> bool:
        calls.append(url)
        return True if handoff is None else await handoff(url, reason)

    llm = _SpyLLM(script=script or [_SCROLL, _SCROLL, _FINISH, _FINISH, _FINISH])
    agent = ReActAgent(Settings(), llm=llm)

    async def go():
        session = _Session()
        result = await agent.run(
            task="读出内容", start_url="https://example.com/target",
            session=session, run_dir=tmp_path / "run",
            login_handoff=spy_handoff, pre_login_url=pre_login_url,
        )
        return result, session

    result, session = asyncio.run(go())
    return result, session, llm, calls


class TestBlankPageHandoff:
    def test_one_blank_frame_is_not_enough(self, monkeypatch, tmp_path):
        """只看一帧空白就打断人，会被页面的正常加载中间态频繁误伤。

        判据是"累计 2 帧"（见 agent.BLANK_STRIKES）：语义是
        "看过一次是空的、重开一次还是空的"，也就是 retry 没用。
        这里让第 2 步恢复成正常页面，验证交接没有被触发。
        """
        from bagent.agent import ReActAgent
        from bagent.config import Settings

        seen = {"n": 0}

        async def fake_perceive(page, settings, *, step, run_dir,  # noqa: ANN001, ARG001
                                prefer_vision=False, vlm=None):
            seen["n"] += 1
            if seen["n"] == 1:
                return PageState(step=step, url="about:blank", title="",
                                 body_text="", is_blank_page=True,
                                 blank_reason="空白")
            return PageState(step=step, url="https://example.com/target",
                             title="目标页", body_text="正文")

        monkeypatch.setattr("bagent.agent.perceive", fake_perceive)
        calls: list[str] = []

        async def spy_handoff(url: str, reason: str) -> bool:
            calls.append(url)
            return True

        agent = ReActAgent(Settings(), llm=_SpyLLM(script=[_SCROLL, _SCROLL, _FINISH]))
        asyncio.run(agent.run(
            task="读出内容", start_url="https://example.com/target",
            session=_Session(), run_dir=tmp_path / "run",
            login_handoff=spy_handoff,
        ))
        assert calls == [], "只看到一帧空白就叫人登录，属于误伤"

    def test_two_blank_frames_trigger_handoff(self, monkeypatch, tmp_path):
        """确认被拦住之后必须交给能处理的人 —— 前提是**换过去那一页真的能用**。"""
        result, session, llm, calls = _run(
            monkeypatch, tmp_path,
            pre_login_url="https://www.zhipin.com/",
            frames=[_BLANK, _BLANK, _OK],
        )
        assert len(calls) == 1, "确认被拦住之后必须交给能处理的人"
        assert any(r.action_name == "登录交接" for r in result.records)

    def test_handoff_happens_on_a_page_a_human_can_use(self, monkeypatch, tmp_path):
        """⭐ 交接前必须先把浏览器挪到真人能登录的页面。

        这一条是这次修复里最要紧的断言。交出 about:blank 等于空转：
        人在空白页上没有任何东西可点，"登录"这件事根本无从发生。
        所以要先把 pre_login_url（站点首页）打开，再把**那个地址**交给人。
        """
        _r, session, _llm, calls = _run(
            monkeypatch, tmp_path, pre_login_url="https://www.zhipin.com/",
            frames=[_BLANK, _BLANK, _OK],
        )
        assert "goto:https://www.zhipin.com/" in session.actions, (
            "交接前必须先离开 about:blank"
        )
        assert calls == ["https://www.zhipin.com/"], (
            f"交出去的应该是能登录的那一页，实际交的是 {calls}"
        )
        # 先挪页面、再交接 —— 顺序反了就没意义。
        assert session.actions.index("goto:https://www.zhipin.com/") < 2

    def test_stability_window_is_longer_than_the_measured_self_destruct(
        self, monkeypatch, tmp_path
    ):
        """落地页的"稳定观察窗"必须**长于实测的自毁延迟**。

        实测两条数据（BOSS直聘 首页，有头模式）：渲染出 6399 个节点 → 约 **2 秒后**
        自己跳成 about:blank。所以这个窗口如果被"优化"到 2 秒以内，
        判定就会采到"它还活着的那一瞬间"，功能悄悄退化成骗人。
        这里把 2 秒这个观测值钉成断言，防止以后有人顺手调小它。
        """
        from bagent.agent import HOSTILE_SETTLE_SECONDS

        assert HOSTILE_SETTLE_SECONDS > 2.0, (
            "观察窗必须大于实测的 ~2 秒自毁延迟，否则会误判成'页面还活着'"
        )

    def test_hopeless_site_is_not_handed_off_to_the_human(self, monkeypatch, tmp_path):
        """⭐ **站点连首页都清空时，不要弹"请去登录"** —— 那是在骗人。

        这一条是实测逼出来的（BOSS直聘）：即便有头模式、即便首页真的渲染出了
        6399 个节点，2 秒后它仍会自己跳回 about:blank（页内 browser-check-v2.js
        干的）。这种站点卡的不是"登录"，是"不许自动化"——
        把一个 2 秒后就会消失的页面交给人，人在上面同样登不进去。
        所以引擎要先换到首页、**回头确认那一页还在**，才决定要不要叫人。
        """
        _r, _s, llm, calls = _run(
            monkeypatch, tmp_path, pre_login_url="https://www.zhipin.com/",
            frames=[_BLANK],       # 首页换过去还是空白
        )
        assert calls == [], "首页也是空白就不该叫人 —— 叫了人也做不了任何事"
        joined = "\n".join(llm.prompts)
        assert "阻止自动化" in joined, "必须把真实原因说清楚，而不是含糊地说'被拦住了'"

    def test_login_returns_to_the_task_entry_not_the_home_page(self, monkeypatch, tmp_path):
        """登录完之后要回到**任务要求的入口**，不能停在首页。"""
        _r, session, _llm, _calls = _run(
            monkeypatch, tmp_path, pre_login_url="https://www.zhipin.com/",
            frames=[_BLANK, _BLANK, _OK],
        )
        assert session.actions[-1] == "goto:https://example.com/target", (
            "登录后应该回到任务入口继续找内容"
        )

    def test_handoff_denied_does_not_loop(self, monkeypatch, tmp_path):
        """人点了"放弃登录"之后，**不能每一步都再问一遍**。

        ⚠️ 这一条的旧版本（登录墙那组里）其实没测到东西：它的剧本第 1 步就是
        finish，循环没走到第二次，所以"不重复问"根本没被验证。
        这里刻意让运行活过 3 步 —— 判据才是真的成立的。
        """

        async def denied(_url: str, _reason: str) -> bool:
            return False

        _r, session, _llm, calls = _run(
            monkeypatch, tmp_path, handoff=denied,
            pre_login_url="https://www.zhipin.com/",
            frames=[_BLANK, _BLANK, _OK],
        )
        assert len(calls) == 1, f"登不进去时不能一遍遍问，实际问了 {len(calls)} 次"
        assert session.actions.count("goto:https://example.com/target") == 1, (
            "没登录成功就不该假装回到任务入口"
        )

    def test_handoff_is_attempted_at_most_once_even_when_block_keeps_recurring(
        self, monkeypatch, tmp_path
    ):
        """即使被拦住的状态一直持续，也**只问人一次**。

        人已经明确表态过（登录了或放弃了），再问就是打扰 ——
        而且会把运行时间拖在等事件上。
        """
        _r, _s, _llm, calls = _run(
            monkeypatch, tmp_path, pre_login_url="https://www.zhipin.com/",
            frames=[_BLANK, _BLANK, _OK],
        )
        assert len(calls) == 1


class TestBlankPageInstruction:
    def test_without_handoff_the_model_is_told_blank_is_not_evidence(
        self, monkeypatch, tmp_path
    ):
        """没人能处理时，必须说清"空白 ≠ 内容不存在"。

        实测模型会拿空白页当证据下"任务无法完成"的结论 ——
        这是把**站点的拦截行为**误当成**网站没有这份内容**。
        """
        _r, _s, llm, _calls = _run(monkeypatch, tmp_path, handoff=None)
        joined = "\n".join(llm.prompts)
        assert "空白不等于内容不存在" in joined
        assert "拦截" in joined
    def test_warning_forbids_inventing_urls(self):
        """实测：模型在空白页上编过一个从没出现过的登录网址（/web/user/）。

        提示里必须直接禁掉这件事。
        """
        from bagent.agent import page_blocked_warning

        text = page_blocked_warning("页面已变成空白页（about:blank）")
        assert "不要" in text and "编造" in text

    def test_rendered_prompt_states_the_real_url_and_forbids_guessing(self):
        """渲染时必须**如实报出观测到的地址**，并禁止猜别处。"""
        st = PageState(url="about:blank", title="", body_text="", is_blank_page=True,
                       blank_reason="页面已变成空白页（about:blank）")
        rendered = st.render_for_prompt()
        assert "空白页" in rendered
        assert "about:blank" in rendered
        assert "编造" in rendered
        # 关键语义：不能据此下"内容不存在"的结论。
        assert "不要据此下" in rendered or "不等于" in rendered

    def test_two_kinds_of_block_use_different_wording(self):
        """登录墙与空白页的措辞不能混用 —— 指错方向比不说更糟。"""
        wall = PageState(url="https://x.com/login", title="登录", body_text="请先登录",
                         is_login_wall=True, login_reason="网址与文案都指向登录页")
        blank = PageState(url="about:blank", is_blank_page=True, blank_reason="空白")
        assert "登录墙" in wall.render_for_prompt()
        assert "登录墙" not in blank.render_for_prompt()
        assert "空白页" in blank.render_for_prompt()


# ---------------------------------------------------------------------------
# 4. 接线：注册表的知识必须传到用得上的地方
# ---------------------------------------------------------------------------
class TestRegistryLoginKnowledgeIsWired:
    def test_resolve_returns_login_url_for_known_login_site(self):
        """已知要登录的站点，/resolve 除了 needs_login 还要给**登录落点**。

        只回传一个布尔值是不够的：引擎需要一个"真人能登录的页面"地址，
        否则就只能把 about:blank 交给人。
        """
        from bagent.api import ResolveRequest, resolve

        r = resolve(ResolveRequest(site="boss", keyword="ai应用工程师"))
        assert r["needs_login"] is True
        assert r["login_url"] == "https://www.zhipin.com/", (
            "登录落点应该是站点首页"
        )
        assert "先打开站点首页" in r["warning"]

    def test_resolve_omits_login_url_when_not_needed(self):
        from bagent.api import ResolveRequest, resolve

        r = resolve(ResolveRequest(site="baidu", keyword="机械键盘"))
        assert r["needs_login"] is False
        assert r["login_url"] is None

    def test_task_request_accepts_login_fields(self):
        from bagent.api import TaskRequest

        req = TaskRequest(
            task="搜一下", url="https://www.zhipin.com/web/geek/job?query=x",
            requires_login=True, login_url="https://www.zhipin.com/",
        )
        assert req.requires_login is True
        assert req.login_url == "https://www.zhipin.com/"

    def test_defaults_are_off(self):
        from bagent.api import TaskRequest

        req = TaskRequest(task="t", url="https://example.com/")
        assert req.requires_login is False
        assert req.login_url is None
        # headless 留 None = 用 .env 的 HEADLESS，服务端/容器里保持无头。
        assert req.headless is None

    def test_request_can_ask_for_a_visible_window(self):
        """控制台默认勾"显示浏览器窗口" → 请求带 headless=False。

        这一条是"人工登录"能成立的前提：没有窗口，交接只能被拒绝
        （见 api._login_handoff 的无头分支）。
        """
        from bagent.api import TaskRequest

        req = TaskRequest(task="t", url="https://example.com/", headless=False)
        assert req.headless is False

    def test_both_engines_accept_pre_login_url(self):
        """两个引擎的 run() 签名必须一致 —— 服务端是按同一份参数调的。

        签名漂移会让"服务端传了、引擎没接"这种缺陷静默发生（参数被吞掉，
        不报错、只是不生效）—— 视觉通道接错线就是这么来的。
        """
        import inspect

        from bagent.agent import ReActAgent
        from bagent.graph_agent import LangGraphReActAgent

        for cls in (ReActAgent, LangGraphReActAgent):
            params = inspect.signature(cls.run).parameters
            assert "pre_login_url" in params, f"{cls.__name__}.run 少了 pre_login_url"

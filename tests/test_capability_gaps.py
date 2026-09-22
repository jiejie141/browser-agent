"""补齐 README「能力边界」里那几条之后，钉住它们**真的生效**。

## 这一组为什么刻意用真实浏览器（本地 set_content，不联网）

iframe 和 Shadow DOM 这两条，用假 page 对象**测不出来** ——
假对象里 `document.querySelectorAll` 的行为是我们自己写的，等于拿自己的
假设去验证自己的假设。README 第八节记过这个教训（"测试写着却没生效"）：

    `test_handoff_failure_does_not_loop` 断言"不会循环"，但脚本第 1 步就
    finish 了，循环根本没走到第 2 步 —— 测试是绿的，功能是坏的。

所以这里每一条都起一个真的 chromium（headless，`set_content` 不联网），
断言"页面上那个按钮**真的被抽到了、真的被点到了**"。
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from bagent.browser import BrowserSession
from bagent.perception import detect_captcha, extract_body_text, extract_elements

pytest.importorskip("playwright", reason="需要真实浏览器来验收 iframe / Shadow DOM")

from playwright.async_api import async_playwright  # noqa: E402


# ---------------------------------------------------------------------------
# 页面：主文档 + iframe + 影子 DOM，各放一个按钮
# ---------------------------------------------------------------------------
_PAGE = """
<!doctype html><html><body>
<button id="main-btn">主页面按钮</button>
<iframe id="f1" width="300" height="150" srcdoc="
  &lt;button id='inner'&gt;iframe里的按钮&lt;/button&gt;
  &lt;script&gt;document.getElementById('inner').onclick = () =&gt; {
     window.__clicked = true; }&lt;/script&gt;
"></iframe>
<my-widget id="w1"></my-widget>
<script>
  class W extends HTMLElement {
    constructor() {
      super();
      const r = this.attachShadow({mode: 'open'});
      r.innerHTML = "<button id='shadow-btn'>影子DOM里的按钮</button>";
      r.getElementById('shadow-btn').onclick = () => { window.__shadow_clicked = true; };
    }
  }
  customElements.define('my-widget', W);
</script>
</body></html>
"""

# 正文全在 iframe 里、主文档几乎没有字的页面
_FRAME_BODY_PAGE = """
<!doctype html><html><body><div>加载中</div>
<iframe id="f1" width="400" height="300" srcdoc="
  &lt;div&gt;这是装在 iframe 里的第一段正文。&lt;/div&gt;
  &lt;div&gt;这是装在 iframe 里的第二段正文。&lt;/div&gt;
"></iframe>
</body></html>
"""

_CAPTCHA_PAGE = """
<!doctype html><html><body>
<div class="geetest_captcha" style="width:200px;height:60px">请完成安全验证</div>
</body></html>
"""

_HIDDEN_CAPTCHA_PAGE = """
<!doctype html><html><body>
<div class="hidden-captcha" style="display:none"></div>
<p>这是一段完全正常的正文，随便写点什么凑字数。</p>
</body></html>
"""


def _run_with_page(html: str, body) -> None:
    """起一个真浏览器 → set_content → 跑 body(page) → 关掉。"""
    async def main():
        pw = await async_playwright().start()
        br = await pw.chromium.launch(headless=True)
        try:
            ctx = await br.new_context(
                viewport={"width": 1440, "height": 900}, locale="zh-CN"
            )
            page = await ctx.new_page()
            await page.set_content(html)
            return await body(page)
        finally:
            await br.close()
            await pw.stop()

    return asyncio.run(main())


def _session(page) -> BrowserSession:
    """绕开 __aenter__ 造一个会话：这里只想测动作层，不想再起一个浏览器。"""
    sess = BrowserSession.__new__(BrowserSession)
    sess.settings = SimpleNamespace(step_timeout_seconds=5)
    sess.run_dir = Path(".")
    sess._profile_dir = None
    sess.page = page
    return sess


def _child_frames(page):
    return [f for f in page.frames if f is not page.main_frame]


# ---------------------------------------------------------------------------
# 缺陷一：iframe 里的内容看不见
# ---------------------------------------------------------------------------
class TestIframeVisibility:
    def test_iframe_element_is_extracted(self):
        async def body(page):
            els = await extract_elements(page)
            texts = [e.text for e in els]
            assert "iframe里的按钮" in texts, f"iframe 里的按钮没被抽到: {texts}"

        _run_with_page(_PAGE, body)

    def test_iframe_element_is_marked_with_frame_index(self):
        async def body(page):
            els = await extract_elements(page)
            inner = [e for e in els if e.text == "iframe里的按钮"]
            assert inner and inner[0].frame > 0, (
                "来自 iframe 的元素必须带 frame 序号，否则复盘时说不清它在哪个文档里"
            )
            main = [e for e in els if e.text == "主页面按钮"]
            assert main and main[0].frame == 0

        _run_with_page(_PAGE, body)

    def test_clicking_an_iframe_element_really_clicks_it(self):
        """最要紧的一条：编号不能只是"看得见"，还得**点得到**。"""

        async def body(page):
            sess = _session(page)
            els = await extract_elements(page)
            ref = [e for e in els if e.text == "iframe里的按钮"][0].ref
            out = await sess._click(ref, None)
            assert out.ok, f"点击 iframe 里的元素失败了: {out.message}"

            child = _child_frames(page)[0]
            assert await child.evaluate("() => !!window.__clicked"), (
                "点击被认为成功了，但 iframe 里的 onclick 没被触发 —— "
                "典型的「编号打在主文档上、点了个空」"
            )

        _run_with_page(_PAGE, body)

    def test_body_text_falls_back_to_iframe(self):
        async def body(page):
            text = await extract_body_text(page)
            assert "iframe 里的第一段正文" in text
            assert "iframe" in text, (
                "拼接 iframe 正文时必须**显式标注来源**，否则模型会把两个文档的"
                "文字当成同一段证据来引用"
            )

        _run_with_page(_FRAME_BODY_PAGE, body)


# ---------------------------------------------------------------------------
# 缺陷二：Shadow DOM 不解析
# ---------------------------------------------------------------------------
class TestShadowDom:
    def test_shadow_element_is_extracted(self):
        async def body(page):
            els = await extract_elements(page)
            texts = [e.text for e in els]
            assert "影子DOM里的按钮" in texts, f"影子 DOM 里的按钮没被抽到: {texts}"

        _run_with_page(_PAGE, body)

    def test_clicking_a_shadow_element_really_clicks_it(self):
        """Playwright 的 CSS 选择器默认穿透开放 shadowRoot，这里把这件事钉住。"""

        async def body(page):
            sess = _session(page)
            els = await extract_elements(page)
            ref = [e for e in els if e.text == "影子DOM里的按钮"][0].ref
            out = await sess._click(ref, None)
            assert out.ok, f"点击影子 DOM 里的元素失败了: {out.message}"
            assert await page.evaluate("() => !!window.__shadow_clicked")

        _run_with_page(_PAGE, body)


# ---------------------------------------------------------------------------
# 缺陷三：没有持久化会话
# ---------------------------------------------------------------------------
def _profile_settings(tmp_path):
    from bagent.config import Settings

    settings = Settings()
    settings.headless = True
    settings.step_timeout_seconds = 10
    settings.teardown_timeout_seconds = 8.0
    settings.persistent_profile_dir = str(tmp_path / "profile")
    return settings


def test_persistent_profile_keeps_cookies_across_runs(tmp_path):
    """登录一次、之后不用再登 —— 这条能力能不能成立，看 Cookie 还在不在。"""
    settings = _profile_settings(tmp_path)

    async def main():
        async with BrowserSession(settings, tmp_path) as s:
            # 走的是持久化那条路：不再有 browser 对象，context 就是根
            assert s._browser is None, "配了持久化目录却还在用 launch()"
            await s._context.add_cookies(
                [
                    {
                        "name": "sid",
                        "value": "abc123",
                        "domain": "example.com",
                        "path": "/",
                        "expires": int(time.time()) + 86400,
                    }
                ]
            )

        async with BrowserSession(settings, tmp_path) as s2:
            got = {c["name"]: c["value"] for c in await s2._context.cookies()}
            assert got.get("sid") == "abc123", (
                f"换了新的运行，Cookie 没留下: {got}（持久化没生效）"
            )

    asyncio.run(main())


def test_session_cookies_are_still_dropped(tmp_path):
    """一条**必须写进 README 的限制**，不是能修的 bug。

    Chromium 只把**有过期时间**的 Cookie 写进磁盘；会话 Cookie（expires=-1）
    只活在内存里，关掉浏览器就没了。所以"配了持久化目录"**不等于**
    "所有登录态都留得住" —— 站点下发的是不是持久 Cookie 决定成败。

    这条测试把现状钉住：哪天它变红了，说明 Chromium 改了行为，
    README 里那句限制就该跟着改。
    """
    settings = _profile_settings(tmp_path)

    async def main():
        async with BrowserSession(settings, tmp_path) as s:
            await s._context.add_cookies(
                [{"name": "sess", "value": "x", "domain": "example.com", "path": "/"}]
            )
        async with BrowserSession(settings, tmp_path) as s2:
            got = {c["name"] for c in await s2._context.cookies()}
            assert "sess" not in got, (
                "会话 Cookie 居然留下了 —— 那 README 里「会话 Cookie 不落盘」"
                "这句限制就该删掉了"
            )

    asyncio.run(main())


def test_persistent_profile_is_off_by_default():
    """默认必须是关的：它会把登录态长期留在磁盘上。"""
    from bagent.config import Settings

    st = Settings()
    assert (st.persistent_profile_dir or "").strip() == "", (
        ".env 里配了 PERSISTENT_PROFILE_DIR —— 这是「把账号凭据留在磁盘上」的开关，"
        "默认必须是关闭的"
    )


# ---------------------------------------------------------------------------
# 缺陷四：验证码
# ---------------------------------------------------------------------------
class TestCaptchaDetection:
    def test_widget_alone_convinces(self):
        ok, why = detect_captcha(has_captcha_widget=True)
        assert ok and "验证码控件" in why

    def test_strong_text_alone_convinces(self):
        ok, _ = detect_captcha(body_text="请拖动滑块完成拼图")
        assert ok, "「拖动滑块」这种话只可能出现在验证码上"

    def test_url_plus_text_convinces(self):
        ok, _ = detect_captcha(url="https://x.com/captcha", body_text="图形验证码")
        assert ok

    def test_url_alone_does_not_convict(self):
        # /account/verify-email 这类正常页面也带 verify —— 单凭网址定案会误伤
        assert not detect_captcha(url="https://x.com/account/verify-email")[0]

    def test_the_word_captcha_alone_does_not_convict(self):
        # 侧栏/登录框里常有"验证码"三个字，光凭它就把整页判成验证码会误伤
        assert not detect_captcha(body_text="登录 验证码登录")[0]

    def test_normal_page_is_clean(self):
        assert not detect_captcha(
            url="https://example.com/list", title="列表页", body_text="共 20 条结果"
        )[0]

    def test_real_page_with_widget_is_flagged(self):
        async def body(page):
            import tempfile

            from bagent.config import get_settings
            from bagent.perception import perceive

            # run_dir 不能写 Path(".") —— 感知会给"没抽到元素"的帧截图，
            # 那会在仓库根留下一个 shots/ 目录（真踩过一次）。
            with tempfile.TemporaryDirectory() as td:
                st = await perceive(page, get_settings(), step=1, run_dir=Path(td))
            assert st.is_captcha, "页面上真挂着一个可见的验证控件却没被认出来"

        _run_with_page(_CAPTCHA_PAGE, body)

    def test_hidden_captcha_container_does_not_false_positive(self):
        """容器预先埋在 DOM 里（display:none）是很常见的写法，不能误判。"""

        async def body(page):
            import tempfile

            from bagent.config import get_settings
            from bagent.perception import perceive

            with tempfile.TemporaryDirectory() as td:
                st = await perceive(page, get_settings(), step=1, run_dir=Path(td))
            assert not st.is_captcha, "把隐藏的验证容器当成了验证码，会白白叫人来滑一下"

        _run_with_page(_HIDDEN_CAPTCHA_PAGE, body)


# ---------------------------------------------------------------------------
# 缺陷五：视觉通道的代码路径
# ---------------------------------------------------------------------------
class TestVisionChannelRequestShape:
    """供应商有没有视觉模型是**外部事实**，会变；
    但"我们发出的请求对不对"是**自己的代码**，必须钉死。

    没有真实视觉模型时，这一段就是唯一能证明 VLM 通道不是死代码的证据。
    """

    def _client(self, captured: list):
        from bagent.llm import LLMClient

        c = LLMClient(api_key="k", base_url="http://127.0.0.1:1/v1", model="vm")

        class _Msg:
            content = "图上有一个搜索框和一个按钮"

        class _Usage:
            prompt_tokens = 11
            completion_tokens = 7

        class _Choice:
            message = _Msg()

        class _Resp:
            usage = _Usage()
            choices = [_Choice()]

        class _Completions:
            def create(self, **kwargs):
                captured.append(kwargs)
                return _Resp()

        class _Chat:
            completions = _Completions()

        class _Fake:
            chat = _Chat()

        c._client = _Fake()
        return c

    def test_vision_request_is_multimodal(self, tmp_path):
        captured: list = []
        c = self._client(captured)
        img = tmp_path / "shot.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\nnot-a-real-png")

        text = c.chat_vision("描述一下", img)
        assert text == "图上有一个搜索框和一个按钮"

        assert captured, "根本没发出请求"
        content = captured[0]["messages"][0]["content"]
        assert isinstance(content, list), "多模态请求必须是 content 数组，不是字符串"
        kinds = [p.get("type") for p in content]
        assert kinds == ["text", "image_url"], kinds

        url = content[1]["image_url"]["url"]
        assert url.startswith("data:image/png;base64,"), (
            f"图片必须以 data URL 形式内联发送，实际是: {url[:40]}"
        )
        assert len(url) > len("data:image/png;base64,"), "图片内容不能是空的"

    def test_vision_usage_is_counted(self, tmp_path):
        captured: list = []
        c = self._client(captured)
        img = tmp_path / "shot.png"
        img.write_bytes(b"png")
        c.chat_vision("描述一下", img)
        assert c.usage.calls == 1 and c.usage.prompt_tokens == 11, (
            "视觉通道的用量没进成本统计 —— 成本面板会少算一截"
        )


# ---------------------------------------------------------------------------
# 缺陷四（引擎侧）：撞到验证码时，引擎到底怎么做
# ---------------------------------------------------------------------------
_CAPTCHA_STATE = dict(
    url="https://example.com/list",
    title="列表页",
    body_text="请完成安全验证",
    is_captcha=True,
    captcha_reason="页面文案明确要求完成人机验证（滑块 / 点选）",
)

_SCROLL_ACTION = '{"thought": "往下滚", "action": "scroll", "dy": 600}'
_FINISH_ACTION = '{"thought": "收尾", "action": "finish", "answer": "遇到验证码"}'


class TestCaptchaHandoff:
    """识别出来只是第一步，**引擎拿它怎么办**才是这次要补的东西。"""

    def _run(self, monkeypatch, tmp_path, *, handoff, with_handoff=True):
        import json  # noqa: F401
        from bagent.agent import ReActAgent
        from bagent.config import Settings
        from bagent.models import PageState
        from test_blocked_page import _Session, _SpyLLM

        async def fake_perceive(page, settings, *, step, run_dir,  # noqa: ANN001
                                prefer_vision=False, vlm=None):
            return PageState(step=step, **_CAPTCHA_STATE)

        monkeypatch.setattr("bagent.agent.perceive", fake_perceive)

        calls: list[tuple[str, str]] = []

        async def spy(url: str, reason: str) -> bool:
            calls.append((url, reason))
            return await handoff(url, reason)

        llm = _SpyLLM(
            script=[_SCROLL_ACTION, _SCROLL_ACTION, _FINISH_ACTION, _FINISH_ACTION]
        )
        agent = ReActAgent(Settings(), llm=llm)

        async def go():
            # 起始网址刻意**不等于**验证码所在的那一页：这样"引擎有没有多余地
            # 跳一次"才看得出来（开局那次 goto 会记下来，之后不该再出现）。
            session = _Session(url="https://example.com/list")
            result = await agent.run(
                task="读出内容", start_url="https://example.com/target",
                session=session, run_dir=tmp_path / "run",
                login_handoff=spy if with_handoff else None,
            )
            return result, session

        result, session = asyncio.run(go())
        return result, session, llm, calls

    def test_captcha_is_handed_off_on_the_current_page(self, monkeypatch, tmp_path):
        """⭐ 验证码**就在当前这一页上** —— 交接前绝不能跳走。

        跳到站点首页会把要滑的那个滑块弄没了，人打开窗口什么都看不到。
        （登录墙那条路径才需要先挪到首页，两者不能混用。）
        """

        async def solved(_url: str, _reason: str) -> bool:
            return True

        _r, session, _llm, calls = self._run(monkeypatch, tmp_path, handoff=solved)
        assert len(calls) == 1, f"撞到验证码就该叫人一次，实际 {calls}"
        assert calls[0][0] == "https://example.com/list", (
            f"交接的必须是当前这一页，实际交的是 {calls[0][0]}"
        )
        assert session.actions == ["goto:https://example.com/target"], (
            "除了开局那一次打开，交接前不该再有跳转（尤其不能跳去站点首页，"
            f"那会把要滑的滑块弄没了），实际做了 {session.actions}"
        )

    def test_solved_captcha_does_not_reopen_the_entry(self, monkeypatch, tmp_path):
        """人滑完之后**留在当前页**继续 —— 重新加载等于重新校验。"""

        async def solved(_url: str, _reason: str) -> bool:
            return True

        result, session, _llm, _c = self._run(monkeypatch, tmp_path, handoff=solved)
        assert session.actions == ["goto:https://example.com/target"], (
            "刚滑完就重新 goto，会把这次验证作废（页面重新加载 = 重新校验）："
            f"{session.actions}"
        )
        kinds = [r.action_name for r in result.records]
        assert "验证码交接" in kinds, f"trace 里要能看出是哪种交接: {kinds}"

    def test_no_handoff_tells_the_model_it_cannot_solve_it(self, monkeypatch, tmp_path):
        """没人能来滑的时候，提示词必须说清"解不了"，而不是笼统的"被拦住了"。"""

        async def never(_url: str, _reason: str) -> bool:
            return False

        _r, _s, llm, _c = self._run(
            monkeypatch, tmp_path, handoff=never, with_handoff=False
        )
        joined = "\n".join(llm.prompts)
        assert "解不了" in joined, "没说清引擎解不了，模型会继续去点那个滑块"
        assert "验证码" in joined


# ---------------------------------------------------------------------------
# 2026-09-22 审查尾账的钉住测试（fake frame 即可，不需要真浏览器 ——
# 这里测的是 Python 侧的降级策略，不是 JS 的扫描行为）：
# ① 单个 frame 回写编号失败只丢该 frame，不再整页"全盲"。
# ---------------------------------------------------------------------------


class _FakeFrame:
    """最小 fake：collect 返回造好的候选，apply 可配置为抛异常。"""

    def __init__(self, cands, fail_apply=False):
        self._cands = cands
        self._fail_apply = fail_apply

    async def evaluate(self, js, arg=None):
        from bagent.perception import _APPLY_JS, _COLLECT_JS

        if js == _COLLECT_JS:
            return self._cands
        if js == _APPLY_JS:
            if self._fail_apply:
                raise RuntimeError("frame detached during apply")
            return None
        raise AssertionError(f"unexpected evaluate: {js[:40]}")


class _FakePage:
    def __init__(self, frames):
        self.main_frame = frames[0]
        self.frames = frames


def _cand(idx, text):
    return {"idx": idx, "tag": "button", "text": text, "aria": "",
            "placeholder": "", "region": "main"}


def test_frame_apply_failure_drops_only_that_frame():
    """钉住审查发现的问题：任何一个 iframe 回写失败，旧实现直接 return []，
    主文档已成功编号的元素也被一起丢掉，页面瞬间"全盲"。
    现在必须只丢失败的那个 frame。"""
    main = _FakeFrame([_cand(0, "主文档按钮")])
    broken = _FakeFrame([_cand(0, "iframe按钮")], fail_apply=True)
    page = _FakePage([main, broken])

    elements = asyncio.run(extract_elements(page))

    texts = [e.text for e in elements]
    assert "主文档按钮" in texts, f"主文档元素不该被 iframe 的失败连坐: {texts}"
    assert "iframe按钮" not in texts, f"编号没写上的元素不能往外报: {texts}"


def test_captcha_and_login_wall_render_mutually_exclusive():
    """验证码与登录墙同时命中时，prompt 只能给**一条**指令（验证码优先，
    与引擎的阻塞判定优先级一致）。两段一起渲染等于对模型说
    "去登录"和"别动，交给人"两句互相矛盾的话。"""
    from bagent.models import PageState

    state = PageState(
        url="https://example.com/login", title="安全验证",
        elements=[], body_text="请完成滑块验证后登录",
        is_captcha=True, captcha_reason="可见验证码组件",
        is_login_wall=True, login_reason="URL+文案+密码框",
    )
    rendered = state.render_for_prompt()
    assert "验证码" in rendered
    assert "登录墙" not in rendered, "两段同时渲染会给模型互相矛盾的指令"

    only_login = PageState(
        url="https://example.com/login", title="登录",
        elements=[], body_text="请登录",
        is_login_wall=True, login_reason="URL 命中",
    )
    assert "登录墙" in only_login.render_for_prompt()

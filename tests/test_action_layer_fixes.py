"""动作层与解析层的三处修复。

这三组不是"顺手加的测试"，每一组都对应 README「能力边界」里白纸黑字
写着的一条已知缺陷：

1. 点击超时只回一句 `Timeout 30000ms exceeded` → 模型不知道为什么失败，
   只会再点一次（实测 12 次点击 4 次超时，每次 30 秒）。
2. 敏感操作护栏在 `confirm is None` 时被整个跳过 → API 服务模式下护栏失效。
3. `ref` / `dy` 被模型写串 → 整步判非法，最贵一次连犯 4 步、21 步全部作废。

外加一条观测补齐（解析失败时保留模型原文），它不修 bug，修的是排障能力。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bagent.browser import BrowserSession, translate_click_error
from bagent.llm import MockLLMClient, build_vlm_client
from bagent.models import Action, PageState, StepRecord, parse_action


# ---------------------------------------------------------------------------
# 1. 点击超时的报错翻译
# ---------------------------------------------------------------------------
class TestTranslateClickError:
    """翻译的目标是"模型能据以改动作"，不是"把英文换成中文"。

    所以每条断言都盯着两件事：说清了**原因**、并且**禁止重试同一个元素**。
    只满足第一条是不够的 —— 模型知道"点不动"之后最常见的反应还是再点一次。
    """

    def test_not_stable_is_translated_to_animating(self):
        """轮播 / 动画元素：等不到稳定。这是必应首页那个真实案例。"""
        exc = Exception(
            "Timeout 30000ms exceeded.\nCall log:\n"
            "  - waiting for element to be visible, enabled and stable\n"
            "  - element is not stable"
        )
        msg = translate_click_error(exc, 37)
        assert "一直在动" in msg, "必须说清原因是在动，而不是笼统的超时"
        assert "不要" in msg or "换做法" in msg, "必须拦住'再点一次'这个默认反应"

    def test_intercepts_pointer_events_is_translated_to_overlay(self):
        exc = Exception("<div class=\"modal\"></div> intercepts pointer events")
        msg = translate_click_error(exc, 5)
        assert "挡住" in msg

    def test_not_visible_is_translated_to_hidden(self):
        exc = Exception("element is not visible")
        msg = translate_click_error(exc, 9)
        assert "不可见" in msg

    def test_unknown_error_still_forbids_retry(self):
        """兜底分支也必须带上'别再点它'——未知原因时更不能鼓励重试。"""
        msg = translate_click_error(Exception("something weird"), 1)
        assert "不要重复点" in msg

    def test_original_error_is_not_dropped(self):
        """原始报错首行要保留：翻译可能译错，人还得有据可查。"""
        msg = translate_click_error(Exception("Timeout 30000ms exceeded. xyz"), 2)
        assert "30000" in msg


# ---------------------------------------------------------------------------
# 2. 敏感操作护栏：没人能确认 = 不放行
# ---------------------------------------------------------------------------
class _FakeLocator:
    def __init__(self, text: str, click_exc: Exception | None = None) -> None:
        self._text = text
        self._click_exc = click_exc
        self.clicked = False

    async def count(self) -> int:
        return 1

    @property
    def first(self) -> "_FakeLocator":
        # 真实 Playwright 的 locator.first 返回另一个 locator；
        # 替身里指向自己即可，省掉一层无意义的包装。
        return self

    async def inner_text(self) -> str:
        return self._text

    async def scroll_into_view_if_needed(self) -> None:
        return None

    async def click(self, timeout: int = 0) -> None:  # noqa: ARG002
        if self._click_exc is not None:
            raise self._click_exc
        self.clicked = True


class _FakeClickPage:
    def __init__(self, label: str, click_exc: Exception | None = None) -> None:
        self.locator_obj = _FakeLocator(label, click_exc)
        self.url = "https://example.com/"

    def locator(self, selector: str) -> _FakeLocator:  # noqa: ARG002
        return self.locator_obj


def _click_session(label: str, click_exc: Exception | None = None) -> BrowserSession:
    """绕开 __aenter__ 造一个会话：这里只想测动作层的判定，不想真起浏览器。"""
    sess = BrowserSession.__new__(BrowserSession)
    sess.settings = SimpleNamespace(step_timeout_seconds=1)
    sess.run_dir = Path(".")
    sess.page = _FakeClickPage(label, click_exc)
    return sess


class TestSensitiveGuard:
    """旧代码是 `if _is_sensitive(label) and confirm is not None:`。

    那个 `and confirm is not None` 的语义是"没人确认就放行" ——
    而 API 服务恰恰传的是 None（它想表达"无交互终端，一律拒绝"）。
    注释和代码正好相反，所以这里把默认值钉死为**拒绝**。
    """

    def test_no_confirm_callback_denies(self):
        sess = _click_session("立即购买")
        out = asyncio.run(sess._click(1, None))
        assert out.ok is False
        assert "敏感操作" in out.message
        assert sess.page.locator_obj.clicked is False, "绝不能真的点下去"

    def test_confirm_returning_false_denies(self):
        async def deny(_p: str) -> bool:
            return False

        sess = _click_session("确认下单")
        out = asyncio.run(sess._click(1, deny))
        assert out.ok is False
        assert sess.page.locator_obj.clicked is False

    def test_confirm_returning_true_allows(self):
        async def allow(_p: str) -> bool:
            return True

        sess = _click_session("删除")
        out = asyncio.run(sess._click(1, allow))
        assert out.ok is True, "人工确认过就该放行 —— 别把护栏做成一律拦死"
        assert sess.page.locator_obj.clicked is True

    def test_normal_label_not_blocked(self):
        sess = _click_session("下一页")
        out = asyncio.run(sess._click(1, None))
        assert out.ok is True, "普通标签不能被护栏误伤"

    def test_click_timeout_is_translated_end_to_end(self):
        """串联：真的点超时时，模型收到的是翻译后的话，不是原始 Timeout。"""
        sess = _click_session(
            "热点新闻标题",
            click_exc=Exception("Timeout 30000ms exceeded. element is not stable"),
        )
        out = asyncio.run(sess._click(37, None))
        assert out.ok is False
        assert "一直在动" in out.message
        assert out.vision_hint is True, "点不动时应提示下一帧走视觉通道"


# ---------------------------------------------------------------------------
# 3. ref / dy 字段写串
# ---------------------------------------------------------------------------
class TestParseActionFieldNormalization:
    """分界线：无歧义的直接掰正，有歧义的绝不猜但报错要说清正确写法。"""

    def test_numeric_string_ref_is_accepted(self):
        action, err = parse_action('{"action": "click", "ref": "12"}')
        assert err == ""
        assert action is not None and action.ref == 12

    def test_none_like_ref_means_no_ref(self):
        """`"None"` 的意思是"这一步没有编号"，无歧义，不该烧掉一步。"""
        for raw in ('{"action": "scroll", "ref": "None"}',
                    '{"action": "scroll", "ref": ""}',
                    '{"action": "scroll", "ref": "null"}'):
            action, err = parse_action(raw)
            assert err == "", f"{raw} 应该被接受，实际报错: {err}"
            assert action is not None and action.ref is None

    def test_dy_string_is_coerced(self):
        action, err = parse_action('{"action": "scroll", "dy": "-300"}')
        assert err == "" and action is not None and action.dy == -300

    def test_dy_float_string_is_coerced(self):
        action, err = parse_action('{"action": "scroll", "dy": "500.0"}')
        assert err == "" and action is not None and action.dy == 500

    def test_ref_written_as_dy_is_rejected_with_teaching_message(self):
        """`ref="dy=500"`：**不猜**成 scroll(dy=500)，但要把正确写法讲出来。

        猜了等于引擎替模型补动作参数；不猜又必须让它知道该怎么改，
        否则实测的后果就是连犯 4 步、21 步作废。
        """
        action, err = parse_action('{"action": "scroll", "ref": "dy=500"}')
        assert action is None
        assert "dy" in err and "scroll" in err, f"报错必须给出正确写法，实际: {err}"
        assert "500" in err, "要把模型自己写的那个值回显出来，它才知道错的是哪个"

    def test_dy_written_as_other_field_is_rejected(self):
        action, err = parse_action('{"action": "scroll", "dy": "dx=1"}')
        assert action is None and "dy" in err

    def test_garbage_ref_is_rejected_clearly(self):
        action, err = parse_action('{"action": "click", "ref": "第三个按钮"}')
        assert action is None and "不是整数" in err

    def test_bool_ref_is_rejected(self):
        action, err = parse_action('{"action": "click", "ref": true}')
        assert action is None

    def test_normal_action_still_parses(self):
        action, err = parse_action(
            '{"thought": "点它", "action": "click", "ref": 3}'
        )
        assert err == "" and isinstance(action, Action) and action.ref == 3

    def test_error_message_carries_raw_output(self):
        """报错要带原文：它会被喂回模型的下一轮 prompt，模型得先看见自己吐了什么。"""
        _, err = parse_action('{"action": "click", "ref": "abc"}')
        assert "模型原始输出" in err


# ---------------------------------------------------------------------------
# 4. 解析失败时保留模型原文
# ---------------------------------------------------------------------------
_BAD_JSON = "我觉得应该先点搜索框，但是我不太确定编号"
_FINISH = json.dumps(
    {"action": "finish", "answer": "页面上写着：价格 £57.25", "evidence": "£57.25"},
    ensure_ascii=False,
)


def _run_script(monkeypatch, tmp_path, script: list[str]):
    from bagent.agent import ReActAgent
    from bagent.config import Settings

    async def fake_perceive(page, settings, *, step, run_dir,  # noqa: ANN001, ARG001
                            prefer_vision=False, vlm=None):
        return PageState(
            step=step, url="https://books.toscrape.com/", title="All products",
            body_text="价格 £57.25 书名 Our Band Could Be Your Life",
        )

    monkeypatch.setattr("bagent.agent.perceive", fake_perceive)

    class _Outcome:
        ok = True
        message = "已执行"
        vision_hint = False

    class _Page:
        url = "https://books.toscrape.com/"

    class _Session:
        page = _Page()

        async def execute(self, action, confirm=None):  # noqa: ANN001, ARG002
            return _Outcome()

    agent = ReActAgent(Settings(), llm=MockLLMClient(script=script))

    async def go():
        return await agent.run(
            task="读出最高价", start_url="https://books.toscrape.com/",
            session=_Session(), run_dir=tmp_path / "run",
        )

    return asyncio.run(go())


class TestRawOutputRetention:
    def test_bad_json_step_keeps_raw_output(self, monkeypatch, tmp_path):
        """复盘时能回答"它到底错成什么样"，而不只是"这一步错了"。"""
        result = _run_script(monkeypatch, tmp_path, [_BAD_JSON, _FINISH])
        bad = [r for r in result.records if r.action is None]
        assert bad, "非法输出那一步必须留下记录"
        assert _BAD_JSON in bad[0].raw_output, "原始文本必须留档"
        assert bad[0].raw_action == "(格式错误)", "给渲染用的是稳定的占位标签"

    def test_normal_step_leaves_raw_output_empty(self, monkeypatch, tmp_path):
        """正常步骤刻意留空：每步都存原文会让 trace 膨胀，且信息是重复的。"""
        result = _run_script(monkeypatch, tmp_path, [_FINISH])
        assert all(r.raw_output == "" for r in result.records)

    def test_step_record_has_the_field(self):
        assert StepRecord(step=1, raw_output="x").raw_output == "x"


# ---------------------------------------------------------------------------
# 5. 视觉通道接线
# ---------------------------------------------------------------------------
class TestVlmWiring:
    """`perceive()` 判定 `vlm_enabled and vlm is not None`。

    引擎不传 vlm → 第二个条件恒假 → 截图照拍、描述从来没跑过。
    这是最难发现的缺陷类型：没有异常、没有报错，功能只是没接上。
    """

    def test_no_vlm_config_returns_none(self):
        from bagent.config import Settings

        st = Settings()
        st.vlm_api_key = ""
        st.vlm_model = ""
        assert build_vlm_client(st) is None

    def test_vlm_base_url_falls_back_to_main(self):
        from bagent.config import Settings

        st = Settings()
        st.vlm_api_key = "sk-test"
        st.vlm_model = "some-vl"
        st.vlm_base_url = ""
        st.llm_base_url = "https://example.invalid/v1"
        client = build_vlm_client(st)
        assert client is not None
        assert str(client._client.base_url).startswith("https://example.invalid")

    def test_mock_vision_returns_string(self):
        """替身也必须返回字符串：调用方会直接把它拼进正文。"""
        assert MockLLMClient().chat_vision("描述一下", "x.png") == ""

    def test_both_engines_pass_vlm_to_perceive(self):
        """源码级断言：有人以后写新引擎忘了传，这条会红。"""
        import inspect

        from bagent import agent as agent_mod
        from bagent import graph_agent as graph_mod

        assert "vlm=self.vlm" in inspect.getsource(agent_mod.ReActAgent.run)
        assert "vlm=self.vlm" in inspect.getsource(graph_mod.LangGraphReActAgent._node_perceive)

    def test_engine_builds_vlm_client_when_enabled(self):
        from bagent.agent import ReActAgent
        from bagent.config import Settings

        st = Settings()
        st.vlm_api_key = "sk-test"
        st.vlm_model = "some-vl"
        st.vlm_base_url = "https://example.invalid/v1"
        agent = ReActAgent(st, llm=MockLLMClient())
        assert agent.vlm is not None, "配了 VLM 就该有客户端，否则降级通道永远不通"

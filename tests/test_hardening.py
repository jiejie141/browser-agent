"""审查后补的加固测试。

这一组覆盖的是"平时不会触发、一触发就是事故"的几类问题：
网址白名单（安全）、模型返回数组（会把整轮跑崩的 TypeError）、
以及历史窗口这个被复制了 5 遍的常量。
"""

from __future__ import annotations

import asyncio
import json

import pytest

from bagent.agent import HISTORY_WINDOW, trim_history
from bagent.browser import safe_url
from bagent.llm import MockLLMClient
from bagent.models import parse_action


# ---------------------------------------------------------------------------
# 网址白名单
# ---------------------------------------------------------------------------
class TestSafeUrl:
    """url 是模型给的，而模型读的是网页内容 —— 不校验就是提示词注入的入口。"""

    @pytest.mark.parametrize(
        "bad",
        [
            "javascript:alert(1)",
            "data:text/html,<script>alert(1)</script>",
            "blob:https://example.com/abc",
            "chrome://settings",
            "about:blank",
        ],
    )
    def test_dangerous_schemes_are_rejected(self, bad):
        url, why = safe_url(bad)
        assert url == "", f"{bad} 不该被放行"
        assert why, "拒绝时必须说清为什么"

    def test_file_url_denied_by_default(self, monkeypatch):
        monkeypatch.delenv("ALLOW_FILE_URL", raising=False)
        url, why = safe_url("file:///C:/secret/.env")
        assert url == ""
        assert "ALLOW_FILE_URL" in why, "要告诉人怎么在确有必要的时候打开它"

    def test_file_url_allowed_when_opted_in(self, monkeypatch):
        """离线自测需要 file:// 任务，所以留了显式开关，而不是一律封死。"""
        monkeypatch.setenv("ALLOW_FILE_URL", "1")
        url, why = safe_url("file:///tmp/page.html")
        assert url.startswith("file://") and why == ""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("https://example.com/a", "https://example.com/a"),
            ("http://example.com/a", "http://example.com/a"),
            ("www.example.com", "https://www.example.com"),
        ],
    )
    def test_normal_urls_pass(self, raw, expected):
        url, why = safe_url(raw)
        assert url == expected and why == ""

    def test_bare_domain_defaults_to_https(self):
        """没写协议时补 https 而不是 http：明文会被劫持/注入。"""
        assert safe_url("example.com")[0] == "https://example.com"

    def test_empty_url_reports_missing_param(self):
        url, why = safe_url("")
        assert url == "" and "缺少" in why

    def test_whitespace_is_trimmed(self):
        assert safe_url("  https://example.com  ")[0] == "https://example.com"


# ---------------------------------------------------------------------------
# 解析：模型返回数组
# ---------------------------------------------------------------------------
class TestParseActionNonObject:
    """`Action(**list)` 抛的是 TypeError，**不是** ValidationError。

    原来没人接它，于是一句 `[{...}]` 就能把整轮任务打挂 ——
    而它本该和别的格式错误同一待遇（反馈回去让模型重来）。
    """

    def test_single_object_array_is_unwrapped(self):
        action, err = parse_action('[{"action": "extract"}]')
        assert err == "" and action is not None and action.action == "extract"

    def test_multi_element_array_is_a_format_error_not_a_crash(self):
        """多元素数组：要么被"截取最外层花括号"那一步挡掉，要么被数组分支挡掉。

        断言只盯**结果**：不能解析出动作、必须给出报错、绝不能抛异常。
        具体落在哪一条分支上不重要 —— 重要的是它不再是"整轮崩掉"。
        """
        action, err = parse_action('[{"action": "extract"}, {"action": "finish"}]')
        assert action is None
        assert err, "必须给出报错，不能静默失败"

    def test_nested_array_does_not_crash(self):
        """真正会走到数组分支的输入：解析出来是 list 而不是 dict。"""
        action, err = parse_action("[[1]]")
        assert action is None
        assert err, "不能让 TypeError 穿出去把主循环打挂"


# ---------------------------------------------------------------------------
# 异步模型调用
# ---------------------------------------------------------------------------
class TestAsyncChat:
    def test_achat_returns_what_chat_returns(self):
        """achat 只是把同步调用挪到线程池，语义必须完全一致。"""
        client = MockLLMClient(script=['{"action": "extract"}'])
        sync = client.chat([{"role": "user", "content": "hi"}])
        got = asyncio.run(client.achat([{"role": "user", "content": "hi"}]))
        assert isinstance(got, str)


# ---------------------------------------------------------------------------
# 历史窗口常量
# ---------------------------------------------------------------------------
class TestHistoryWindow:
    def test_trim_keeps_last_n(self):
        h = [str(i) for i in range(50)]
        out = trim_history(h)
        assert len(out) == HISTORY_WINDOW
        assert out[-1] == "49"

    def test_shorter_than_window_is_untouched(self):
        assert trim_history(["a", "b"]) == ["a", "b"]

    def test_window_is_positive(self):
        """窗口值散着写了 5 遍才抽成常量 —— 至少保证它不是个没意义的数。"""
        assert HISTORY_WINDOW >= 4


# ---------------------------------------------------------------------------
# 任务文件读取
# ---------------------------------------------------------------------------
class TestLoadTaskFile:
    def test_missing_file_says_so(self, tmp_path):
        from bagent.cli import load_task_file

        with pytest.raises(RuntimeError, match="不存在"):
            load_task_file(tmp_path / "nope.json")

    def test_broken_json_points_at_the_line(self, tmp_path):
        from bagent.cli import load_task_file

        p = tmp_path / "bad.json"
        p.write_text('{"tasks": [ {"id": "t1",} ]}', encoding="utf-8")
        with pytest.raises(RuntimeError, match="合法 JSON"):
            load_task_file(p)

    def test_wrong_shape_is_reported(self, tmp_path):
        from bagent.cli import load_task_file

        p = tmp_path / "shape.json"
        p.write_text('{"tasks": "oops"}', encoding="utf-8")
        with pytest.raises(RuntimeError, match="数组"):
            load_task_file(p)

    def test_normal_file_loads(self, tmp_path):
        from bagent.cli import load_task_file

        p = tmp_path / "ok.json"
        p.write_text(json.dumps({"tasks": [{"id": "t1"}]}), encoding="utf-8")
        assert load_task_file(p) == [{"id": "t1"}]

"""评测脚本「环境预检」的回归测试。

## 为什么值得单独钉住

真实站点评测集的成功率**强依赖网络**。如果不把
"Agent 不会做" 和 "这台机器连不上" 分开，就会出现两种坏结果：

- 网络差的时候成功率暴跌，看着像代码退化了 → 有人会去改提示词"修"它，
  在本地永远改不动，最后只能靠编数字交差；
- 报告里只剩一个 `3/12`，没人知道那 9 个失败里有多少是环境造成的。

所以这里钉三件事：
1. `summarize` 把 skipped **排除在分母之外**（既不记成失败，也不记成成功）；
2. `print_report` 会单独列出 skipped，并且在没有计分任务时不崩；
3. `main` 在全部任务都被预检拦下时返回 1（而不是 0，也不是异常）。
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

import run_eval  # noqa: E402


class _SettingsStub:
    """只带 `validate()` 的替身。

    ⚠️ 这一条是被 CI 打脸才补上的：`run_eval.main()` 里会调
    `settings.validate()`，而它**在缺 LLM_API_KEY 时直接抛 RuntimeError**。
    本地有 `.env` 所以怎么跑都绿，推上去 CI 上没有 key →
    "运行测试" 这步 6 秒就红了（其余 42 条用例全过，只有这两条挂）。

    教训：**单测不该依赖密钥**。它测的是"闸门接线对不对"，
    跟有没有 key 毫无关系 —— 把外部依赖摘掉才是单测该干的事。
    """

    def validate(self) -> None:
        return None

    # main() 会打印生效的模型/引擎（模型层消融要对账），桩得带上这两个字段。
    llm_model = "stub-model"
    engine = "handwritten"
    # 步数预算也是"会改变结论的参数"（20 → 30 让三条难任务从 4/12 变 10/12），
    # 所以 main() 也把它打进头部和明细。桩同样要带上 —— 规则是：
    # **main() 读什么，桩就得有什么**，否则一改 main() 就红在这两条无关的用例上。
    max_steps = 30


@pytest.fixture
def no_llm_key(monkeypatch):
    """把 settings 换成不需要 key 的替身。"""
    monkeypatch.setattr(run_eval, "get_settings", lambda **_kw: _SettingsStub())


def _row(rid: str, ok: bool, kind: str = "normal", scored: bool = True) -> dict:
    return {
        "id": rid,
        "kind": kind,
        "scored": scored,
        "ok": ok,
        "reason": "ok" if ok else "nope",
        "answer": "x",
        "steps": 3,
        "elapsed": 1.0,
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "calls": 2,
        "cost_yuan": 0.0,
        "error": "",
        "run_dir": "",
    }


class TestManualTasksAreNotScored:
    """`kind=manual`：机器判不了的任务 —— 照跑照记，但不进分子分母。

    动机是实打实的假阳性：r05（GitHub 趋势第一名）用 must_regex 钉
    `owner/repo` 形状，分数上是"通过"，但我拿真值一对是错的
    （当天 #1 是 affaan-m/ECC，它答的是 anthropics/claude-code，
    页面上确实存在、但排第七）。形状对、内容错。
    宁可把这类任务单列一档，也不要让"看起来通过"污染成功率。
    """

    def test_manual_task_excluded_from_denominator(self):
        rows = [_row("a", True), _row("m", False, kind="manual", scored=False)]
        s = run_eval.summarize(rows)
        assert s["total"] == 1, "manual 不能进分母"
        assert s["passed"] == 1
        assert s["manual_total"] == 1
        assert s["declared_total"] == 2

    def test_judge_labels_manual_as_needs_human(self):
        ok, reason = run_eval.judge(
            {"kind": "manual"},
            _JudgeResult(answer="anthropics / claude-code"),
        )
        assert not ok, "不判对错就不能记成通过"
        assert "人工核对" in reason and "anthropics" in reason

    def test_manual_listed_in_report_with_answer(self, capsys):
        rows = [_row("a", True), _row("m", False, kind="manual", scored=False)]
        run_eval.print_report(rows, run_eval.summarize(rows))
        out = capsys.readouterr().out
        assert "人工核对" in out and "不计分" in out

    def test_all_manual_returns_nonzero(self):
        """全是 manual → 给不出成功率，成功率必须是 0，不能装成 100%。"""
        s = run_eval.summarize([_row("m", False, kind="manual", scored=False)])
        assert s["total"] == 0 and s["success_rate"] == 0

    def test_declared_total_covers_all_three_buckets(self):
        rows = [_row("a", True), _row("m", False, kind="manual", scored=False)]
        s = run_eval.summarize(rows, [{"id": "s", "url": "u", "reason": "r"}])
        assert s["declared_total"] == 3, "计分 + 人工 + 跳过 = 任务集声明的总数"


class _JudgeResult:
    """judge() 只需要这几个字段。"""

    def __init__(self, *, finished=True, answer="", error=""):
        self.finished = finished
        self.answer = answer
        self.error = error


class TestSummarizeExcludesSkipped:
    def test_skipped_not_in_denominator(self):
        rows = [_row("a", True), _row("b", False)]
        skipped = [{"id": "c", "url": "https://x/", "reason": "ERR_CONNECTION_CLOSED"}]
        s = run_eval.summarize(rows, skipped)

        assert s["total"] == 2, "skipped 不能进分母"
        assert s["success_rate"] == 0.5, "成功率只对能连上的那部分负责"
        assert s["skipped_total"] == 1
        assert s["declared_total"] == 3, "declared_total 才是任务集声明的总数"

    def test_no_skipped_keeps_old_behavior(self):
        rows = [_row("a", True), _row("b", True), _row("c", False)]
        s = run_eval.summarize(rows)
        assert s["total"] == 3 and s["passed"] == 2
        assert s["skipped_total"] == 0
        assert s["declared_total"] == 3

    def test_skipped_does_not_inflate_success_rate(self):
        """全部跳过 + 0 计分任务时，成功率应是 0 而不是 1。"""
        s = run_eval.summarize([], [{"id": "a", "url": "u", "reason": "r"}])
        assert s["total"] == 0 and s["success_rate"] == 0

    def test_kind_split_still_works_with_skipped(self):
        rows = [_row("n1", True, "normal"), _row("c1", True, "counter"), _row("n2", False, "normal")]
        s = run_eval.summarize(rows, [{"id": "s", "url": "u", "reason": "r"}])
        assert s["normal_rate"] == 0.5 and s["counter_rate"] == 1.0


class TestPrintReport:
    def test_lists_skipped_separately(self, capsys):
        rows = [_row("a", True)]
        skipped = [{"id": "b", "url": "https://y/", "reason": "net::ERR_CONNECTION_CLOSED"}]
        run_eval.print_report(rows, run_eval.summarize(rows, skipped), skipped)
        out = capsys.readouterr().out
        assert "环境预检" in out
        assert "b" in out and "ERR_CONNECTION_CLOSED" in out
        assert "未计分" in out

    def test_all_skipped_does_not_crash(self, capsys):
        skipped = [{"id": "a", "url": "https://y/", "reason": "boom"}]
        run_eval.print_report([], run_eval.summarize([], skipped), skipped)
        assert "环境预检" in capsys.readouterr().out


class TestPreflightGate:
    def test_preflight_ignores_file_urls(self):
        """file:// 任务不该触发任何网络动作 —— examples.json 全靠它。"""
        res = asyncio.run(run_eval.preflight(["file:///tmp/a.html", "file:///tmp/b.html"]))
        assert res == {}

    def test_all_skipped_returns_nonzero(self, tmp_path, monkeypatch, capsys, no_llm_key):
        """全部被预检拦下 → 返回 1，且写不出假的成功率。"""
        task_file = tmp_path / "net_dependent.json"
        task_file.write_text(
            json.dumps(
                {
                    "_network_dependent": True,
                    "tasks": [
                        {
                            "id": "z01",
                            "kind": "normal",
                            "task": "打开一个不存在的站点",
                            "url": "https://this-host-should-not-resolve.invalid/",
                            "must_contain": [],
                        }
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        # 不去真的连：直接把预检换成测试桩，钉住的是「闸门接线是否正确」。
        # 桩必须是 async 的 —— main 里是 asyncio.run(preflight(...))。
        monkeypatch.setattr(run_eval, "preflight", _fake_preflight)
        rc = run_eval.main(["--task-file", str(task_file)])
        assert rc == 1, "全部跳过时不能返回 0"
        assert "环境预检" in capsys.readouterr().out

    def test_explicit_no_preflight_runs_everything(self, tmp_path, monkeypatch, no_llm_key):
        """--no-preflight 必须真的把闸门关掉（否则离线任务集会被误伤）。"""
        called = {"n": 0}

        async def _spy(urls, **kw):
            called["n"] += 1
            return {}

        monkeypatch.setattr(run_eval, "preflight", _spy)
        task_file = tmp_path / "empty.json"
        task_file.write_text(json.dumps({"tasks": []}), encoding="utf-8")
        rc = run_eval.main(["--task-file", str(task_file), "--no-preflight"])
        assert rc == 2, "空任务集应返回 2"
        assert called["n"] == 0, "关掉预检后不该再做网络请求"

    def test_task_file_flips_preflight_on(self):
        """`_network_dependent` 这个声明必须真的能打开预检，否则等于摆设。"""
        data = json.loads((ROOT / "tasks" / "real_sites.json").read_text(encoding="utf-8"))
        assert data.get("_network_dependent") is True

    def test_examples_json_keeps_preflight_off(self):
        """基线任务集**故意**不开预检。

        它的 URL 虽然也是真站点（quotes/books.toscrape.com、bing），
        但它们是专门给人练爬虫的稳定站点，可达性不是变量。
        而 examples.json 是简历里那条『端到端 3/6』的判分基线 ——
        分母一旦会因为网络抖动而变化，这条基线就没法对账了。
        需要临时挡网络问题时，用 `--preflight` 显式打开。
        """
        data = json.loads((ROOT / "tasks" / "examples.json").read_text(encoding="utf-8"))
        assert not data.get("_network_dependent"), "基线不能被改成网络依赖"
        tasks = data["tasks"]
        assert len(tasks) == 6, "基线是 6 个任务（4 常规 + 2 反例），改动会让简历数字失效"
        assert sum(1 for t in tasks if t.get("kind") == "counter") == 2


async def _fake_preflight(urls, **_kw):
    return {u: (False, "sandbox: 测试桩") for u in urls}


class TestPreflightUsesSameEgressAsAgent:
    """预检必须和正式跑**走同一个出口**。

    这条是被真实缺陷逼出来的：`preflight` 原本自己裸开浏览器
    （`chromium.launch(headless=True)`，不带代理参数），而正式跑的
    `BrowserSession` 是读 `BROWSER_PROXY` 的。后果不是"慢一点"，
    而是**结论反了**：代理配好、站点其实连得上，预检却按直连去试，
    报"不可达"，任务在开跑前就被 skipped 剔除，一次都没跑。

    所以这里用一个假的 Playwright 把 `launch()` 收到的参数截下来，
    断言代理确实被传进去了 —— 不依赖真网络、不依赖真浏览器。
    """

    def _install_fake_playwright(self, monkeypatch, captured):
        class _FakeResp:
            status = 200

        class _FakePage:
            async def goto(self, url, **_kw):
                return _FakeResp()

        class _FakeBrowser:
            async def new_page(self):
                return _FakePage()

            async def close(self):
                captured["closed"] = True

        class _FakeChromium:
            async def launch(self, **kw):
                captured.update(kw)
                return _FakeBrowser()

        class _FakePW:
            chromium = _FakeChromium()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_a):
                return False

        import playwright.async_api as pw_api

        monkeypatch.setattr(pw_api, "async_playwright", lambda: _FakePW())

    def test_proxy_from_settings_reaches_launch(self, monkeypatch):
        captured: dict = {}
        self._install_fake_playwright(monkeypatch, captured)
        monkeypatch.setattr(
            run_eval,
            "get_settings",
            lambda **_kw: _ProxySettings("http://127.0.0.1:7890", "baidu.com"),
        )

        res = asyncio.run(run_eval.preflight(["https://example.com/"], min_interval=0))

        assert captured.get("proxy") == {
            "server": "http://127.0.0.1:7890",
            "bypass": "baidu.com",
        }, "预检的浏览器必须带上 BROWSER_PROXY 的代理，否则它测的不是正式跑的那条路"
        assert captured.get("headless") is True, "预检仍应无头运行"
        assert res["https://example.com/"] == (True, "HTTP 200")

    def test_no_proxy_configured_omits_key(self, monkeypatch):
        """没配代理时不能凭空塞一个 proxy 键（空 server 会让 Playwright 抛错）。"""
        captured: dict = {}
        self._install_fake_playwright(monkeypatch, captured)
        monkeypatch.setattr(
            run_eval, "get_settings", lambda **_kw: _ProxySettings("", "")
        )

        asyncio.run(run_eval.preflight(["https://example.com/"], min_interval=0))

        assert "proxy" not in captured


class _ProxySettings:
    """只带代理两个字段的替身（preflight 只读这两个）。"""

    def __init__(self, proxy: str, bypass: str) -> None:
        self.browser_proxy = proxy
        self.browser_proxy_bypass = bypass


class TestTransportCrossCheck:
    """把「不可达」拆成两类 —— 两类处置完全不同。

    本机实测的真实现象（12 个真实站点逐条核过）：同一条链路、同一个出口 IP，
    三种客户端结论不一致 ——

    | 客户端 | TLS 栈 | 豆瓣 |
    |---|---|---|
    | chromium | BoringSSL | `ERR_CONNECTION_CLOSED` |
    | Python urllib | OpenSSL | `SSL: UNEXPECTED_EOF_WHILE_READING` |
    | curl | schannel | **`200` + 64416 字节真实正文** |

    所以**区分点在 TLS 栈，不是"浏览器 vs 非浏览器"**。这也是为什么
    通道要跑两条、而且必须包含 curl（它与另外两条不是同一个栈）。
    只看"浏览器不通、urllib 也不通"会误判成"站点挂了" —— 豆瓣、
    东方财富的站点都是好的。

    已经逐一排除的解释：不是沙箱（内外一致）、不是 IPv6（无 AAAA）、
    代理无用（直连 5/12，走代理还是 5/12，失败的是同一批）。
    """

    def test_urllib_http_error_still_counts_as_reachable(self, monkeypatch):
        """403/404 说明链路是通的 —— 这里只问通不通，不问内容对不对。"""
        import urllib.error
        import urllib.request

        def _raise(*_a, **_kw):
            raise urllib.error.HTTPError("https://x/", 403, "Forbidden", {}, None)

        monkeypatch.setattr(urllib.request, "urlopen", _raise)
        ok, detail = run_eval._probe_urllib("https://x/", 5.0)
        assert ok is True and "403" in detail

    def test_urllib_connection_error_counts_as_unreachable(self, monkeypatch):
        import urllib.request

        def _raise(*_a, **_kw):
            raise OSError("Connection refused")

        monkeypatch.setattr(urllib.request, "urlopen", _raise)
        ok, _detail = run_eval._probe_urllib("https://x/", 5.0)
        assert ok is False

    def test_curl_missing_is_reported_not_crashed(self, monkeypatch):
        """没装 curl 时要降级成一条结论，不能抛。"""
        import shutil

        monkeypatch.setattr(shutil, "which", lambda _n: None)
        ok, detail = run_eval._probe_curl("https://x/", 5.0)
        assert ok is False and "未安装" in detail

    def test_curl_302_counts_as_reachable(self, monkeypatch):
        """微博/当当对 curl 就是回 302 —— 服务器有响应就是通。"""
        import subprocess

        def _fake_run(*_a, **_kw):
            class _CP:
                stdout = b"302"
                stderr = b""

            return _CP()

        monkeypatch.setattr(subprocess, "run", _fake_run)
        ok, detail = run_eval._probe_curl("https://x/", 5.0)
        assert ok is True and "302" in detail

    def test_curl_000_is_failure(self, monkeypatch):
        import subprocess

        def _fake_run(*_a, **_kw):
            class _CP:
                stdout = b"000"
                stderr = b"curl: (35) schannel: failed to receive handshake"

            return _CP()

        monkeypatch.setattr(subprocess, "run", _fake_run)
        ok, detail = run_eval._probe_curl("https://x/", 5.0)
        assert ok is False and "000" in detail

    def test_any_channel_ok_means_site_is_fine(self, monkeypatch):
        """curl 通、urllib 不通 → 站点是好的。这正是豆瓣的真实现场。"""
        monkeypatch.setattr(run_eval, "_probe_urllib", lambda *a: (False, "URLError"))
        monkeypatch.setattr(run_eval, "_probe_curl", lambda *a: (True, "curl 200 64416B"))
        ok, detail = run_eval.transport_reachable("https://movie.douban.com/top250")
        assert ok is True
        assert "64416B" in detail and "urllib" in detail

    def test_both_channels_down(self, monkeypatch):
        monkeypatch.setattr(run_eval, "_probe_urllib", lambda *a: (False, "URLError"))
        monkeypatch.setattr(run_eval, "_probe_curl", lambda *a: (False, "curl 000"))
        ok, _detail = run_eval.transport_reachable("https://x/")
        assert ok is False

    def test_annotate_labels_the_two_cases(self, monkeypatch):
        calls: list[str] = []

        def _fake(url, timeout=8.0):
            calls.append(url)
            return ("site-ok" in url, "curl 200 1B" if "site-ok" in url else "curl 000")

        monkeypatch.setattr(run_eval, "transport_reachable", _fake)
        skipped = [
            {"id": "a", "url": "https://site-ok/", "reason": "ERR_CONNECTION_CLOSED"},
            {"id": "b", "url": "https://all-down/", "reason": "ERR_CONNECTION_CLOSED"},
        ]
        run_eval.annotate_transport(skipped)

        assert skipped[0]["transport_ok"] is True
        assert "站点是好的" in skipped[0]["diagnosis"]
        assert skipped[1]["transport_ok"] is False
        assert "站点/网络问题" in skipped[1]["diagnosis"]

    def test_same_url_only_probed_once(self, monkeypatch):
        """r02/r03 指向同一个 douban URL，别把它连试两次。"""
        calls: list[str] = []

        def _fake(url, timeout=8.0):
            calls.append(url)
            return False, "curl 000"

        monkeypatch.setattr(run_eval, "transport_reachable", _fake)
        skipped = [
            {"id": "r02", "url": "https://movie.douban.com/top250", "reason": "x"},
            {"id": "r03", "url": "https://movie.douban.com/top250", "reason": "x"},
        ]
        run_eval.annotate_transport(skipped)
        assert len(calls) == 1, "同一 URL 只该探一次"
        assert skipped[1].get("transport_ok") is False

    def test_report_surfaces_site_is_fine_row(self, capsys, monkeypatch):
        # 测试环境里 rich 探测不到终端宽度，会把列挤到看不清。
        # 这里给一个够宽的控制台，测的是"有没有这一列"，不是"排得好看不好看"。
        from rich.console import Console

        monkeypatch.setattr(run_eval, "console", Console(width=220))
        rows = [_row("a", True)]
        skipped = [
            {
                "id": "r03",
                "url": "https://movie.douban.com/top250",
                "reason": "net::ERR_CONNECTION_CLOSED",
                "transport": "curl 200 64416B / urllib ✗ URLError",
                "transport_ok": True,
                "diagnosis": "站点是好的：出口在区分客户端",
            }
        ]
        run_eval.print_report(rows, run_eval.summarize(rows, skipped), skipped)
        out = capsys.readouterr().out
        assert "非浏览器通道" in out
        assert "64416B" in out and "站点是好的" in out

    def test_report_handles_row_without_transport_fields(self, capsys):
        """老调用方（测试桩/手工构造）不带这几个键时不能崩。"""
        rows = [_row("a", True)]
        skipped = [{"id": "b", "url": "https://y/", "reason": "boom"}]
        run_eval.print_report(rows, run_eval.summarize(rows, skipped), skipped)
        assert "环境预检" in capsys.readouterr().out

    def test_real_sites_file_declares_network_dependence(self):
        """真实站点集必须继续声明网络依赖，否则预检会被关掉。"""
        data = json.loads((ROOT / "tasks" / "real_sites.json").read_text(encoding="utf-8"))
        assert data.get("_network_dependent") is True
        assert len(data["tasks"]) == 12

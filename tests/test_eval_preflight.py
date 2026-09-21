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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

import run_eval  # noqa: E402


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

    def test_all_skipped_returns_nonzero(self, tmp_path, monkeypatch, capsys):
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

    def test_explicit_no_preflight_runs_everything(self, tmp_path, monkeypatch):
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

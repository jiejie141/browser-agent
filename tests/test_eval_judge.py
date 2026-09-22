"""判分逻辑的回归测试。

判分写错了比 Agent 写错了更危险：Agent 出错只是答错一道题，
判分出错会让你**以为自己答对了**，然后拿这个数字去对账、去写简历。
这里钉住的都是真实跑出来的假阳性。

## 案例一：r12（京东登录墙）——「熔断」被当成了「拒答」

原判分：`if not result.finished: return True, "未完成（合理放弃）"`。

看起来宽容得合理，实际上给"崩了"发免费通行证。真实轨迹里 Agent
认出了登录页（thought 写着"当前页面是登录页面"），但它没放弃 ——
它往登录框敲 `testuser` / `testpassword`，连点登录，再点，然后开始
原地 scroll，最后被"原地打转"熔断。`finished=False`、`answer=None`，
按旧规矩 → 通过。这是判分的错：反例任务考的是**识别并说出来**。

## 案例二：r01（百度搜索）——「答非所问」被当成了「已完成」

`must_contain` 为空时任何非空答案都算过。真实跑出来：让它搜「机械键盘」
读第一条结果，它交回"一天11枚金牌！这是中国队的金牌速度"（百度首页的
热搜新闻）。内容是真的，题是答错的。

## 案例三：r05（GitHub 趋势）—— 答案写不死，但形状写得死

第一名仓库天天换，`must_contain` 没法写。但 `owner/repo` 这个形状是稳的，
于是有 `must_regex`。注意它必须容忍 `vercel-labs / json-render` 这种带空格的写法
（模型输出里真的会带空格，我第一版正则就是这么挂的）。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

import run_eval  # noqa: E402


class _Result:
    """最小化的运行结果替身。只需要 judge 用到的那几个字段。"""

    def __init__(self, *, finished=True, answer="", error=""):
        self.finished = finished
        self.answer = answer
        self.error = error


# ===========================================================================
# 反例任务：必须说出理由
# ===========================================================================


class TestCounterMustSayItOutLoud:
    def test_fused_without_explanation_fails(self):
        """真实复现 r12：认出了登录页，却去敲假账号，最后熔断。"""
        r = _Result(
            finished=False,
            answer="",
            error="检测到原地打转：动作 scroll#None 重复 5 次且无进展，已熔断",
        )
        ok, reason = run_eval.judge({"kind": "counter"}, r)
        assert not ok, "熔断不是拒答——它没说出『我做不到』"
        assert "熔断" in reason or "没说明原因" in reason

    def test_unfinished_without_error_also_fails(self):
        ok, reason = run_eval.judge({"kind": "counter"}, _Result(finished=False))
        assert not ok
        assert "没说明原因" in reason

    def test_explicit_refusal_passes(self):
        """真实复现 r11：小红书，明确说明要登录所以拿不到。"""
        r = _Result(finished=True, answer="无法直接从当前页面获取推荐流里的第一条笔记标题，需要登录后才能查看。")
        ok, reason = run_eval.judge({"kind": "counter"}, r)
        assert ok and "主动说明" in reason

    def test_baseline_counter_tasks_still_pass(self):
        """基线 t05/t06 就是靠『主动说明无法完成』过的，收紧后不能误伤。

        这条卡住的是"为了诚实把基线一起搞崩" —— 那会让简历里的 3/6 失效。
        """
        for answer in ("账号没有收藏项", "当前页面没有提供下单或付款的按钮，任务无法完成。"):
            ok, _ = run_eval.judge({"kind": "counter"}, _Result(finished=True, answer=answer))
            assert ok, answer

    def test_refusal_after_thrashing_still_counts(self):
        """先打转、最后一步明确说出做不到 —— 也算过。

        放宽到只要 `answer` 里有拒答措辞就通过，与是否熔断无关：
        要考的是"最终有没有说清楚"，不是"过程漂不漂亮"。
        """
        r = _Result(finished=False, answer="该站点需要登录，无法完成", error="熔断")
        ok, _ = run_eval.judge({"kind": "counter"}, r)
        assert ok

    def test_counter_giving_a_real_answer_fails(self):
        ok, reason = run_eval.judge(
            {"kind": "counter"}, _Result(finished=True, answer="第一个商品是 XX 键盘，299 元")
        )
        assert not ok and "给出了结论" in reason


# ===========================================================================
# normal 任务：must_regex
# ===========================================================================


class TestNormalJudging:
    def test_empty_constraints_accept_any_nonempty(self):
        ok, reason = run_eval.judge({"kind": "normal"}, _Result(answer="随便一句"))
        assert ok and reason == "已完成"

    def test_empty_answer_fails(self):
        ok, _ = run_eval.judge({"kind": "normal"}, _Result(answer=""))
        assert not ok

    def test_baidu_off_topic_answer_now_fails(self):
        """真实复现 r01：读成了首页热搜新闻。加了 must_contain 就该拦住。"""
        task = {"kind": "normal", "must_contain": ["键盘"]}
        ok, reason = run_eval.judge(task, _Result(answer="一天11枚金牌！这是中国队的金牌速度"))
        assert not ok, "没搜就答，不该通过"
        assert "未匹配预期" in reason

    def test_baidu_on_topic_answer_passes(self):
        task = {"kind": "normal", "must_contain": ["键盘"]}
        ok, _ = run_eval.judge(task, _Result(answer="机械键盘推荐：2026 年最值得买的 10 款"))
        assert ok

    def test_github_shape_regex_accepts_spaced_owner_repo(self):
        """模型真的会输出 `vercel-labs / json-render`（斜杠两边带空格）。"""
        task = {"kind": "normal", "must_regex": "[A-Za-z0-9._-]+\\s*/\\s*[A-Za-z0-9._-]+"}
        ok, reason = run_eval.judge(task, _Result(answer="vercel-labs / json-render"))
        assert ok, reason

    def test_github_shape_regex_accepts_tight_owner_repo(self):
        task = {"kind": "normal", "must_regex": "[A-Za-z0-9._-]+\\s*/\\s*[A-Za-z0-9._-]+"}
        assert run_eval.judge(task, _Result(answer="openai/whisper"))[0]

    def test_shape_regex_rejects_prose(self):
        task = {"kind": "normal", "must_regex": "[A-Za-z0-9._-]+\\s*/\\s*[A-Za-z0-9._-]+"}
        ok, _ = run_eval.judge(task, _Result(answer="今天的第一名是一个非常热门的项目"))
        assert not ok

    def test_regex_may_be_a_bare_string(self):
        """任务文件里手写 `"must_regex": "\\d+"` 也该能用，不用强迫写成数组。"""
        ok, _ = run_eval.judge({"kind": "normal", "must_regex": "\\d+"}, _Result(answer="买 3 个"))
        assert ok

    def test_bad_regex_reports_instead_of_silently_failing(self):
        """任务文件写错正则必须报出来 —— 静默失败会让人以为模型不行。"""
        ok, reason = run_eval.judge({"kind": "normal", "must_regex": "([unclosed"}, _Result(answer="x"))
        assert not ok
        assert "不是合法正则" in reason

    def test_unfinished_normal_reports_error(self):
        ok, reason = run_eval.judge(
            {"kind": "normal"}, _Result(finished=False, error="检测到原地打转，已熔断")
        )
        assert not ok and "熔断" in reason

    def test_refusal_on_normal_task_fails(self):
        """⚠️ 正常任务给了拒答 = 没完成。这条规则是**被一次真实退化逼出来的**。

        t04 的 `must_contain` 是空的，原来"答案非空就通过"，于是加了振荡警告
        之后 t04 有 3/4 次直接答"无法找到搜索结果"，分数却照样 4/4 ——
        判分假阳性把一次真实退化整个盖住了。
        对照真值：改前那一轮 t04 的答案都是页面上的真内容
        （"ReAct 是一种新颖的方法，通过在语言模型中融合推理和行动…"）。
        """
        ok, reason = run_eval.judge({"kind": "normal"}, _Result(answer="这个页面上没有搜索结果"))
        assert not ok, "拒答不是正常任务的有效结论"
        assert "拒答" in reason

    def test_refusal_is_still_correct_for_counter_task(self):
        """counter 任务里"拒答"正是**正确**行为 —— 别把上面那条规则误伤过来。"""
        ok, reason = run_eval.judge(
            {"kind": "counter"}, _Result(answer="无法完成，页面上没有下单按钮")
        )
        assert ok and reason == "主动说明无法完成"

    def test_correct_answer_that_mentions_a_refusal_word_is_flagged(self):
        """⚠️ 已知代价，故意钉住：拒答判定是**子串匹配**，正常答案里带"没有"也会被判失败。

        这是一处真的假阴性风险。之所以接受它：t04 这类正常任务的正确标题
        通常不含这些词，而这条规则拦住的是"答不出来却记成成功"——后者更贵。
        哪天真撞上了，应该改这里（做更精确的拒答判定），而不是把口径放回去。
        """
        task = {"kind": "normal", "must_contain": ["ReAct"]}
        ok, reason = run_eval.judge(
            task, _Result(answer="第一条标题是 ReAct，页面上没有更多结果")
        )
        assert not ok and "拒答" in reason


# ===========================================================================
# "拒答"词表被三处用到，口径必须一致
# ===========================================================================


class TestRefusalMarkersStayConsistent:
    """同一个词表三处各管一段，任何一处漂移都会产出**自相矛盾的报告**。

    | 用处 | 位置 | 拿它干什么 |
    |---|---|---|
    | 引擎免检 | `bagent/grounding.py` | 拒答类结论免检证据 |
    | 判分 | `eval/run_eval.py` | 反例算过 / **正常任务算失败** |
    | 标可疑 | `scripts/show_run_answers.py` | 摊开答案时标出可疑行 |

    漂移的后果不是"数字难看"，而是报告自己跟自己打架：
    引擎免检、判分却判失败；或者这里标可疑、那边判通过。
    这种报告比一个低分危险得多 —— 低分至少是真的。
    """

    def test_all_three_users_share_one_definition(self):
        """不再是"三份抄得一样"，而是**只有一份**（在 `bagent/grounding.py`）。

        2026-09-22 加这条测试时，它立刻红了一次 —— 实测两边就是长歪的：
        判分这边 13 个词，引擎那边 17 个。所以修法不是"把 13 个改成 17 个"，
        而是把另外两处**改成引用同一份**。
        """
        import importlib.util

        from bagent.grounding import REFUSAL_MARKERS as grounding_markers

        spec = importlib.util.spec_from_file_location(
            "show_run_answers", ROOT / "scripts" / "show_run_answers.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        assert run_eval.REFUSAL_MARKERS is grounding_markers, (
            "判分处自己抄了一份词表，会发生'引擎免检、判分判失败'"
        )
        assert mod.REFUSAL_MARKERS is grounding_markers, (
            "show_run_answers 自己抄了一份词表，会发生'这里标可疑、那边判通过'"
        )


# ===========================================================================
# 多次重复跑：为什么要报 pass@k / pass^k，而不是一个百分数
# ===========================================================================


class TestRepeatStats:
    """`--repeat 3` 的汇总。

    动机是实测出来的：收紧判分后接连跑两次基线，`t05/t06` 第一次是
    "主动说明无法完成"（过）、第二次变成"点同一个按钮 5 次熔断"（不过），
    而 `t04` 反过来先不过后过。**单次跑出来的 2/6 和 3/6 都能自圆其说** ——
    这种数字拿去写简历，被要求现场复现就露馅。所以要报分布。
    """

    def _row(self, rid, ok, kind="normal"):
        return {"id": rid, "kind": kind, "ok": ok, "scored": True, "steps": 1, "elapsed": 1.0,
                "total_tokens": 1, "cost_yuan": 0.0, "reason": "", "answer": "", "error": "",
                "run_dir": ""}

    def test_flaky_task_has_passa_but_not_passk(self):
        rows = [
            self._row("t05", True, "counter"),
            self._row("t05", False, "counter"),
            self._row("t05", False, "counter"),
        ]
        per = run_eval.repeat_stats(rows, 3)
        assert len(per) == 1
        p = per[0]
        assert p["passes"] == 1
        assert p["pass_at_k"] is True, "至少成功过一次 = 有能力"
        assert p["pass_all_k"] is False, "三次全过才算稳定"

    def test_stable_task_has_both(self):
        rows = [self._row("t01", True) for _ in range(3)]
        p = run_eval.repeat_stats(rows, 3)[0]
        assert p["pass_at_k"] and p["pass_all_k"]

    def test_never_passing_task_has_neither(self):
        rows = [self._row("t03", False) for _ in range(3)]
        p = run_eval.repeat_stats(rows, 3)[0]
        assert not p["pass_at_k"] and not p["pass_all_k"]

    def test_repeat_one_returns_nothing(self):
        """`--repeat` 默认 1 时不该多输出一张表。"""
        assert run_eval.repeat_stats([self._row("a", True)], 1) == []

    def test_manual_tasks_marked_unscored_in_repeat_table(self):
        rows = [dict(self._row("m", False, "manual"), scored=False) for _ in range(3)]
        p = run_eval.repeat_stats(rows, 3)[0]
        assert p["scored"] is False
        assert "passes" not in p, "不判分的任务不该有通过次数"

    def test_grouping_keeps_task_order(self):
        rows = [
            self._row("t01", True), self._row("t02", False),
            self._row("t01", True), self._row("t02", True),
        ]
        ids = [p["id"] for p in run_eval.repeat_stats(rows, 2)]
        assert ids == ["t01", "t02"], "按首次出现顺序输出，别按 dict 乱序"

    def test_report_shows_both_metrics(self, capsys):
        rows = [self._row("t01", True), self._row("t01", False)]
        stats = run_eval.summarize(rows)
        run_eval.print_report(rows, stats, repeat=2)
        out = capsys.readouterr().out
        assert "pass@2" in out and "pass^2" in out
        assert "不能" in out, "要明确提醒按次累计的成功率不能读成单轮成功率"


# ===========================================================================
# 任务集与判分口径的一致性
# ===========================================================================


class TestTaskSetsMatchJudge:
    def test_task_ids_are_unique_in_every_task_file(self):
        """任务 id 必须唯一。

        这不是洁癖：手工编辑 JSON 时插块很容易留下重名（我自己就留了一个
        r06），而重名的后果很隐蔽 —— 汇总里"总数"对不上任务文件的行数，
        明细 JSON 里两条覆盖成一条，排查时根本想不到是文件写重了。
        """
        import json

        for name in ("examples.json", "real_sites.json", "demo_sites.json"):
            data = json.loads((ROOT / "tasks" / name).read_text(encoding="utf-8"))
            ids = [t["id"] for t in data["tasks"]]
            assert len(ids) == len(set(ids)), f"{name} 里有重复 id: {ids}"

    def test_every_auto_scored_task_passes_with_a_plausible_answer(self):
        """给每个自动判分的任务编一个"像样"的答案，必须都能过。

        防的是"任务文件和判分器对不上"：比如把 must_regex 写成
        must_contain 那套不可能命中的字符串，任务就永远过不了，
        而看报告只会以为模型不行。
        """
        import json

        data = json.loads((ROOT / "tasks" / "real_sites.json").read_text(encoding="utf-8"))
        seen = 0
        for t in data["tasks"]:
            if t.get("kind") in ("counter", "manual"):
                continue
            needle = (t.get("must_contain") or [None])[0]
            answer = f"结论：{needle}" if needle else "正常给出了一条结论"
            ok, reason = run_eval.judge(t, _Result(answer=answer))
            assert ok, f"{t['id']} 用合理答案都过不了：{reason}"
            seen += 1
        assert seen >= 4, "自动判分的任务太少，这层评测就没意义了"

    def test_auto_scored_tasks_all_have_a_concrete_anchor(self):
        """自动判分的任务**必须**写出可核对的判据。

        这条卡的是最阴的那种回归：有人为了"提高通过率"把 must_contain 删空，
        任务就变成"只要答了就算对"。真实站点上这么干过一次，
        结果 Agent 把百度首页的热搜新闻当成搜索结果交上来，也判成通过。
        """
        import json

        data = json.loads((ROOT / "tasks" / "real_sites.json").read_text(encoding="utf-8"))
        for t in data["tasks"]:
            if t.get("kind") in ("counter", "manual"):
                continue
            assert t.get("must_contain") or t.get("must_regex"), (
                f"{t['id']} 是自动判分任务却没写判据 —— 请补锚点，或改成 kind=manual"
            )

    def test_counter_tasks_expect_no_completion_flag(self):
        import json

        data = json.loads((ROOT / "tasks" / "real_sites.json").read_text(encoding="utf-8"))
        counters = [t for t in data["tasks"] if t.get("kind") == "counter"]
        assert counters, "真实站点集里必须保留反例任务"
        assert all(t.get("expect_no_completion") for t in counters)

    def test_manual_tasks_say_why_they_cannot_be_scored(self):
        """每个 manual 任务都要写清"为什么判不了"。

        否则下一个人会以为这是漏写的 task，顺手给它加个 must_contain
        让它"变绿" —— 那正好退回到我们刚爬出来的坑。
        """
        import json

        data = json.loads((ROOT / "tasks" / "real_sites.json").read_text(encoding="utf-8"))
        manual = [t for t in data["tasks"] if t.get("kind") == "manual"]
        assert manual, "至少要保留 manual 这一类，它是判分诚实的兜底"
        for t in manual:
            assert t.get("_judge_note"), f"{t['id']} 标了 manual 却没写原因"

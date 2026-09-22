"""证据锚定（grounding）的回归测试。

## 这个模块是为什么存在的

评测里 t02 **连续三轮全败**，逐条读 trace 后是同一个根因：
**动作之后凭假设收尾。**

t02 点了 Next 翻到第 2 页，下一步直接 finish。它 thought 里写
「第一条名言是 'This life is what you make it...'，作者是 Elie Wiesel」——
引的句子确实在第 2 页上，但**作者配错了**（首句是 Marilyn Monroe），
而且它把结论写进了 thought、`answer` 字段留成了 null。它在"回忆"，不在"抄"。

而引擎原来对 finish 的 answer 不做任何核对 —— "答错"和"答对"长得一模一样。
所以新增 `check_grounding`：finish 必须带一段**逐字从当前页面复制**的
evidence，且 answer 里那些**在页面上确实找得到**的关键信息必须都出现在 evidence 里。

## ⚠️ 这套规则的边界：它管"有没有依据"，不管"对不对"

这是本文件里最容易被误读的一条，所以专门有用例钉住（见
`test_grounding_does_not_judge_correctness`）：
**一个有页面依据的错误结论，照样能通过证据核对。**

grounding 只能回答"结论是不是从页面上读出来的"，回答不了"它是不是题目的答案"——
后者是评测判分的事。把两者混为一谈，就会拿 grounded=True 冒充"答对了"。

（t03 就是反方向的教训：Agent 答的 £57.25 其实**是对的**，是任务文件的真值
凭印象写错了，判分给了假阴性。真值必须换通道核 —— 见 tasks/examples.json 的
`_truth_source`。）
"""

from __future__ import annotations

import pytest

from bagent.grounding import (
    MAX_GROUNDING_RETRIES,
    MIN_EVIDENCE_CHARS,
    GroundingVerdict,
    check_grounding,
    extract_terms,
    is_refusal,
    normalize,
)


# ---------------------------------------------------------------------------
# normalize / is_refusal：两个基础件
# ---------------------------------------------------------------------------
class TestNormalize:
    def test_strips_all_whitespace(self):
        """页面正文里换行缩进到处都是，模型复制过来空格数必然对不上。
        所以比对前把空白全去掉 —— 只证明"这段话在页面上"，不追求标点级严格。"""
        assert normalize("A  Light\n  in\t the") == normalize("A Light in the")

    def test_fullwidth_punctuation_folds(self):
        assert normalize("他说：“好”。") == normalize('他说:"好".')

    def test_none_is_safe(self):
        assert normalize("") == ""
        assert normalize(None) == ""  # type: ignore[arg-type]


class TestIsRefusal:
    @pytest.mark.parametrize(
        "text",
        [
            "该站点没有登录功能，无法完成",
            "页面上找不到下单按钮",
            "需要登录后才能查看",
            "这个任务我做不到",
        ],
    )
    def test_refusal_markers_hit(self, text):
        assert is_refusal(text)

    def test_normal_answer_is_not_refusal(self):
        assert not is_refusal("作者是 Albert Einstein")


# ---------------------------------------------------------------------------
# extract_terms：抽"关键信息"，并丢掉题面措辞
# ---------------------------------------------------------------------------
class TestExtractTerms:
    def test_drops_words_that_came_from_the_task(self):
        """题面里的词（"作者""第一条"）参与核对只会误伤。"""
        terms = extract_terms("第一句名言的作者是 Albert Einstein", task="读出第一句名言的作者")
        assert "einstein" in [t.lower() for t in terms]
        assert "第一句名言" not in terms
        assert "作者" not in terms

    def test_filters_to_terms_present_on_page(self):
        """只留页面上找得到的词 —— 那些才是"对页面内容的断言"。"""
        terms = extract_terms(
            "作者是 Albert Einstein，不是 Elie Wiesel",
            page_text="“The world as we have created it...” — Albert Einstein",
        )
        joined = " ".join(t.lower() for t in terms)
        assert "einstein" in joined
        assert "wiesel" not in joined, "页面上没有的名字属于幻觉，不据此拒收"

    def test_page_filter_off_returns_everything(self):
        terms = extract_terms("作者是 Albert Einstein", page_text=None)
        assert any("einstein" in t.lower() for t in terms)

    def test_respects_max_terms_checked(self):
        answer = " ".join(f"term{i}value" for i in range(40))
        assert len(extract_terms(answer, page_text=None)) <= 12


# ---------------------------------------------------------------------------
# 测试用页面：直接照抄真实站点（2026-09-21 用换通道方式核过）
# ---------------------------------------------------------------------------
QUOTES_PAGE_1 = (
    "“The world as we have created it is a process of our thinking. "
    "It cannot be changed without changing our thinking.” by Albert Einstein (about)"
)

# quotes.toscrape.com/page/2/ 的第一句 —— t02 的真值锚点
QUOTES_PAGE_2 = (
    "“This life is what you make it. No matter what, you're going to make mistakes. "
    "But it's the way you recover from the mistakes that matters.” by Marilyn Monroe (about)"
)

# books.toscrape.com 首页全部 20 本的真实价格（按页面顺序）。
# 最高价是 £57.25 的 Our Band Could Be Your Life...，不是 £59.99。
BOOKS_PAGE = "\n".join(
    [
        "A Light in the Attic £51.77",
        "Tipping the Velvet £53.74",
        "Soumission £50.10",
        "Sharp Objects £47.82",
        "Sapiens: A Brief History of Humankind £54.23",
        "The Requiem Red £22.65",
        "The Dirty Little Secrets of Getting Your Dream Job £33.34",
        "The Coming Woman £17.93",
        "The Boys in the Boat £22.60",
        "The Black Maria £52.15",
        "Starving Hearts (Triangular Trade Trilogy, #1) £13.99",
        "Shakespeare's Sonnets £20.66",
        "Set Me Free £17.46",
        "Scott Pilgrim's Precious Little Life (Scott Pilgrim #1) £52.29",
        "Rip it Up and Start Again £35.02",
        "Our Band Could Be Your Life: Scenes from the American Indie Underground £57.25",
        "Olio £23.88",
        "Mesaerion: The Best Science Fiction Stories 1800-1849 £37.59",
        "Libertarianism for Beginners £51.33",
        "It's Only the Himalayas £45.17",
    ]
)


class TestCheckGroundingPasses:
    def test_verbatim_evidence_with_matching_answer_passes(self):
        v = check_grounding(
            answer="作者是 Albert Einstein",
            evidence="by Albert Einstein",
            page_text=QUOTES_PAGE_1,
        )
        assert v.ok and v.grounded

    def test_whitespace_and_case_differences_do_not_block(self):
        """模型复制时改了空格/大小写不算错 —— 证明的是"这段话在页面上"。"""
        v = check_grounding(
            answer="这句话出自第一条",
            evidence="the world as we have created it",
            page_text=QUOTES_PAGE_1,
        )
        assert v.ok and v.grounded

    def test_chinese_paraphrase_answer_is_not_blocked(self):
        """纯中文改写的答案里没有数字/英文 → 不该被"逐字"规则卡住。"""
        page = "本页共收录 10 句名言，全部来自不同作者。"
        v = check_grounding(
            answer="这一页收录的是名言，来源各不相同",
            evidence="本页共收录 10 句名言，全部来自不同作者。",
            page_text=page,
        )
        assert v.ok and v.grounded


class TestCheckGroundingBlocksTheRealBug:
    """把 t02 那条真实错答做成可复现的用例。"""

    def test_t02_wrong_author_is_rejected(self):
        """t02：引了第 2 页的句子，却把作者报成页面上根本没有的人。

        旧引擎：写什么收什么 → 判成"完成"。
        新引擎：answer 里的英文名在当前页面上一个都找不到 → 退回重读。
        """
        v = check_grounding(
            answer="第 2 页第一句名言的作者是 Elie Wiesel",
            evidence="This life is what you make it.",
            page_text=QUOTES_PAGE_2,
        )
        assert not v.ok
        assert not v.grounded

    def test_t02_empty_answer_is_rejected(self):
        """t02 的另一半：结论写进了 thought，answer 字段是 null。

        旧引擎：`finished=True` 就记成功，结论为空也照收 —— 评测里表现为
        "完成但结论为空"，看着像判分严，其实是引擎根本没拦住空结论。
        """
        v = check_grounding(answer="", evidence="", page_text=QUOTES_PAGE_2)
        assert not v.ok and "answer 是空的" in v.reason

    def test_t02_correct_author_with_evidence_passes(self):
        """照着页面抄就该能过 —— 规则不是来卡死模型的。"""
        v = check_grounding(
            answer="第 2 页第一句名言的作者是 Marilyn Monroe",
            evidence="by Marilyn Monroe",
            page_text=QUOTES_PAGE_2,
        )
        assert v.ok and v.grounded


class TestT03TruthWasWrongNotTheAgent:
    """t03 的"失败"是判分真值写错了，不是能力缺口。

    这一组用例的作用是**把这个结论钉进测试**：以后有人看到 t03 曾经"全败"
    想再去"修 Agent"，会先撞到这组用例 —— 它告诉你真值是 £57.25。
    """

    def test_homepage_max_is_57_25_not_59_99(self):
        assert "57.25" in BOOKS_PAGE
        assert "59.99" not in BOOKS_PAGE, "首页根本没有 £59.99，那是凭印象写错的真值"

    def test_agent_answer_57_25_is_grounded(self):
        v = check_grounding(
            answer="售价最高的书是《Our Band Could Be ...》，价格为£57.25。",
            evidence="Our Band Could Be Your Life: Scenes from the American Indie Underground £57.25",
            page_text=BOOKS_PAGE,
        )
        assert v.ok and v.grounded

    def test_grounding_does_not_judge_correctness(self):
        """⚠️ 边界：有页面依据的**错误**结论照样通过。

        "选错了书但证据抄得很忠实"这种情况，grounding 拦不住 ——
        因为它的职责只是"结论有没有落在页面上"，不是"是不是题目的答案"。
        谁能判对错？评测的 judge（must_contain / must_regex）。
        把这两件事分开，才不会拿 grounded 冒充"答对了"。
        """
        v = check_grounding(
            answer="售价最高的书是 A Light in the Attic，£51.77",
            evidence="A Light in the Attic £51.77",
            page_text=BOOKS_PAGE,
        )
        assert v.ok, "证据确实抄对了，grounding 无从判断它选错了书"
        assert v.grounded


class TestCheckGroundingFailures:
    def test_empty_answer_fails(self):
        v = check_grounding(answer="", evidence="x", page_text="some page content here")
        assert not v.ok and "answer 是空的" in v.reason

    def test_missing_evidence_fails(self):
        v = check_grounding(answer="作者是 Einstein", evidence="", page_text=QUOTES_PAGE_1)
        assert not v.ok and "缺少 evidence" in v.reason

    def test_evidence_too_short_fails(self):
        v = check_grounding(answer="作者是 Einstein", evidence="是", page_text=QUOTES_PAGE_1)
        assert not v.ok and "太短" in v.reason

    def test_evidence_not_on_page_fails(self):
        """凭记忆写的证据：页面上找不到 → 退回，逼它回页面重新读。"""
        v = check_grounding(
            answer="作者是 Albert Einstein",
            evidence="这是我在别处记得的一句话，页面上其实没有",
            page_text=QUOTES_PAGE_1,
        )
        assert not v.ok and "找不到" in v.reason

    def test_answer_term_is_not_on_current_page_fails(self):
        """换页之后拿旧页的答案收尾：结论里的名字在新页面上找不到。"""
        v = check_grounding(
            answer="作者是 Elie Wiesel",
            evidence="This life is what you make it.",
            page_text=QUOTES_PAGE_2,
        )
        assert not v.ok

    def test_citing_a_but_answering_b_fails(self):
        """引 A 答 B：证据里没有结论里的那个人名。"""
        page = QUOTES_PAGE_1 + "\nby Marilyn Monroe"
        v = check_grounding(
            answer="作者是 Albert Einstein 而不是 Marilyn Monroe",
            evidence="by Marilyn Monroe",
            page_text=page,
        )
        assert not v.ok
        assert "einstein" in " ".join(v.missing).lower()


class TestCheckGroundingExemptions:
    """两类免检：都要 checked=False 显式标记，不能假装核对过了。"""

    def test_refusal_is_exempt_but_not_grounded(self):
        v = check_grounding(
            answer="该站点没有登录功能，无法完成",
            evidence="",
            page_text=QUOTES_PAGE_1,
        )
        assert v.ok
        assert not v.checked
        assert not v.grounded, "免检 ≠ 有证据，不能给自己发绿灯"

    def test_near_empty_page_is_exempt_not_grounded(self):
        v = check_grounding(answer="作者是 Einstein", evidence="", page_text="")
        assert v.ok and not v.checked and not v.grounded

    def test_verdict_grounded_requires_checked_and_ok(self):
        assert GroundingVerdict(True, "ok", checked=True).grounded
        assert not GroundingVerdict(True, "免检", checked=False).grounded
        assert not GroundingVerdict(False, "fail", checked=True).grounded


class TestRetryBudget:
    def test_retry_budget_is_small_and_positive(self):
        """退回上限要存在且不大：我们要测的是"模型能不能自纠"，
        不是"能不能把它卡死" —— 卡死只会让评测从"答错"变成"没答案"。"""
        assert 1 <= MAX_GROUNDING_RETRIES <= 3

    def test_min_evidence_chars_allows_short_title(self):
        """阈值不能太大：合法证据可能就是一个短标题或一个价格。"""
        assert MIN_EVIDENCE_CHARS <= 8


# ---------------------------------------------------------------------------
# 引擎层的接线：退回 / 自纠 / 上限后采纳
# ---------------------------------------------------------------------------
# 上面测的是纯函数。这一组测的是**接线**：check_grounding 真的被 ReActAgent
# 用上了、退回真的会再跑一轮、重试用尽真的会采纳而不是卡死。
# 不测这里，就可能出现"函数是对的、主循环根本没调它"这种最隐蔽的假绿。
import asyncio  # noqa: E402
import json  # noqa: E402

from bagent.agent import ReActAgent  # noqa: E402
from bagent.llm import MockLLMClient  # noqa: E402
from bagent.models import PageState  # noqa: E402

# 场景页：一个能读出最高价的页面（对应 t03 的"照着页面抄"版本）
SCENE_PAGE = "全部商品\nA Light in the Attic £51.77\n本页最高价 £57.25"

_FINISH_NO_EVIDENCE = json.dumps(
    {"thought": "读完了，直接给结论", "action": "finish", "answer": "最高价是 £57.25"},
    ensure_ascii=False,
)
_FINISH_GOOD = json.dumps(
    {
        "thought": "把支撑结论的原文抄下来",
        "action": "finish",
        "answer": "最高价是 £57.25",
        "evidence": "本页最高价 £57.25",
    },
    ensure_ascii=False,
)


class _FakeOutcome:
    def __init__(self) -> None:
        self.ok = True
        self.message = "已执行"
        self.vision_hint = False


class _FakePage:
    url = "https://books.toscrape.com/"


class _FakeSession:
    def __init__(self) -> None:
        self.page = _FakePage()

    async def execute(self, action, confirm=None):  # noqa: ANN001, ARG002
        return _FakeOutcome()


def _run_script(monkeypatch, tmp_path, script: list[str]):
    """用剧本化的离线模型跑一遍，返回 RunResult。"""

    async def fake_perceive(page, settings, *, step, run_dir, prefer_vision=False, vlm=None):  # noqa: ANN001, ARG001
        return PageState(
            step=step, url="https://books.toscrape.com/", title="All products",
            body_text=SCENE_PAGE,
        )

    monkeypatch.setattr("bagent.agent.perceive", fake_perceive)

    from bagent.config import Settings

    agent = ReActAgent(Settings(), llm=MockLLMClient(script=script))

    async def go():
        return await agent.run(
            task="读出最高价",
            start_url="https://books.toscrape.com/",
            session=_FakeSession(),
            run_dir=tmp_path / "run",
        )

    return asyncio.run(go())


class TestAgentRetryWiring:
    def test_finish_without_evidence_is_retried_then_adopted(self, monkeypatch, tmp_path):
        """先交一个没证据的 finish → 被退回 → 补齐证据后收尾。

        这就是 t02 那条陷阱的正面解：引擎逼它回到页面上抄一遍，
        而不是把 thought 里的结论直接当成 answer 收下。
        """
        result = _run_script(monkeypatch, tmp_path, [_FINISH_NO_EVIDENCE, _FINISH_GOOD])

        assert result.finished, "补齐证据后必须能收尾，不能卡死"
        assert result.grounded is True, "抄对了证据 → grounded 必须为真"
        assert result.grounding_retries == 1, "被退回一次"

    def test_retries_exhausted_still_finishes_but_not_grounded(self, monkeypatch, tmp_path):
        """连着交 N 次没证据的结论 → 用尽重试后**采纳**，但 grounded 标 False。

        为什么不是卡死：卡死会把"答错"变成"没答案"，那是把问题藏起来。
        正确做法是收下答案、如实标记"这条结论没有证据支撑"，
        让报告里能一眼看出是"没核对"还是"核对过了"。
        """
        script = [_FINISH_NO_EVIDENCE] * (MAX_GROUNDING_RETRIES + 1)
        result = _run_script(monkeypatch, tmp_path, script)

        assert result.finished
        assert result.grounded is False, "没通过核对就不能标成有证据"
        assert result.grounding_retries == MAX_GROUNDING_RETRIES
        assert result.answer, "答案仍要保留，不能因为没证据就丢掉"

    def test_step_record_marks_rejected_finish_as_failed(self, monkeypatch, tmp_path):
        """被退回的那一步必须留痕（ok=False），否则 trace 里看不出发生过退回。"""
        result = _run_script(monkeypatch, tmp_path, [_FINISH_NO_EVIDENCE, _FINISH_GOOD])

        rejected = [r for r in result.records if not r.ok]
        assert len(rejected) == 1
        assert "退回" in rejected[0].message

    def test_grounding_defaults_are_off_for_plain_run(self, monkeypatch, tmp_path):
        """没走到证据核对那一支时，字段要有安全默认值（不能是 None）。"""
        from bagent.agent import RunResult
        from bagent.models import Usage

        r = RunResult(
            task="t", start_url="u", success=False, finished=False, answer="",
            steps=0, elapsed_seconds=0.0, usage=Usage(), cost_yuan=0.0,
        )
        assert r.grounded is False
        assert r.grounding_note == ""
        assert r.grounding_retries == 0


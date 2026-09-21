# -*- coding: utf-8 -*-
"""finish 的证据锚定：让结论必须落在**页面上**，而不是模型脑子里。

## 为什么需要（一条真实轨迹逼出来的）

评测里 t02 **连续三轮全败**，逐条读 trace 后找到根因：
**动作之后凭假设收尾。**

- **t02**：点了「Next」切到第 2 页，下一步直接 finish。它 thought 里写
  「第一条名言是 'This life is what you make it...'，作者是 Elie Wiesel」——
  这句确实是第 2 页上的，但**作者配错了**（首句是 Marilyn Monroe），
  而且它把结论写在 thought 里、`answer` 字段留成了 null。
  它是"记得"这句话，不是"照着抄"。

共同点：**结论里的关键信息一条都不是从页面上抄下来的**，
而引擎原来对 `finish` 的 answer 不做任何核对，写什么就收什么。
于是"答错了"和"答对了"在引擎眼里长得一模一样。

## ⚠️ 另一半真相：t03 不是能力缺口，是**判分真值写错了**

t02 和 t03 当时都被记成"连续三轮回合全败"，一开始被当成同一个能力缺口。
但动手改之前先去核了真值 —— 用**换通道**的方式（不走浏览器、不经过模型，
直接 HTTP 抓 books.toscrape.com 首页的 `price_color` 再排序）：

    20 本书，最高价 £57.25 = "Our Band Could Be Your Life..."
    首页上根本不存在 £59.99；A Light in the Attic 只有 £51.77

而任务文件里写的 `must_contain` 是 `["A Light in the Attic", "59.99"]`。
**Agent 一直答的是对的（£57.25），是判分把它判错了** —— 假阴性。

教训有两条，比代码本身重要：

1. **"全败"不等于"能力缺口"。** 先怀疑判分，再怀疑模型。
   把一个假阴性当成能力问题去"修"，改的提示词永远修不好它 ——
   最后只会为了让它变绿去编数据。这跟把假阳性当成能力问题是同一个坑的两面。
2. **真值必须换通道核对。** 任务文件里的 expected 也是人写的，也会凭记忆写错 ——
   而"凭记忆写 expected"恰恰就是本模块要治的那个病。
   t03 的 expected 已按上面这次核对结果改正。

## 做法：finish 必须带一段逐字复制的证据

规则见 `check_grounding()`。核心是三条：

1. `evidence` 必须**逐字**出现在**当前页面**的正文里
   —— 强制"复制"而非"回忆"，也强制它回到当前这一帧去核对；
2. `answer` 里那些**在页面上确实找得到**的关键信息（人名 / 数字 / 标题），
   必须**全部出现在 evidence 里** —— 堵住"引 A 答 B"；
3. 证据核对不过就**退回重做**，而不是直接采纳。

## 为什么"只核对页面上找得到的词"这一条很关键

直觉写法是"answer 里的词必须都在 evidence 里"。那样会把
「这句话」「任务已完成」这类**措辞**也拿去比对，必然误伤 ——
它们本来就不该出现在页面原文里。

所以先做一次过滤：**只挑那些真的出现在当前页面上的词**。
- 出现在页面上 = 模型在对页面内容下断言 → 那它就必须拿出原文；
- 页面上根本没有 = 要么是措辞（"这句话"），要么是幻觉（另行记录，不据此拒收）。

这个过滤也让规则天然对中文口语化答案友好（纯中文改写里没有数字/英文时不会被卡）。

## 免检的两类（都要显式记录，不能假装核对过）

- **拒答类结论**：它断言的是"页面上没有 X"，**没法逐字引用一个不存在的东西**。
  硬要它引用只会逼模型编一段假证据 —— 那比不核对更糟。
- **页面正文近乎为空**：没有可核对的原文。这种情况返回 `checked=False`，
  如实标记"未核对"，而不是默认通过。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# 证据短到这个程度就没有信息量了（"是"、"9.7"），挡的是"随便填一个词交差"。
# 注意不能设太大：合法证据可能就是一个短标题或一个价格。
MIN_EVIDENCE_CHARS = 6

# 退回重做的上限。超过就**采纳当前答案并把 grounded 标成 False** ——
# 我们要测的是"模型能不能自纠"，不是"能不能把它卡死"。
# 卡死只会让评测从"答错"变成"没答案"，那是把问题藏起来，不是解决。
MAX_GROUNDING_RETRIES = 2

# 一次最多核对几个词。长答案可能抽出几十个词，全核对会把 prompt 反馈撑爆，
# 而且第 13 个之后的词基本验证不了新东西。
MAX_TERMS_CHECKED = 12

# 拒答措辞。与 eval/run_eval.py 的 REFUSAL_MARKERS 是**同一个口径**
# （那边判分、这边免检，两侧不一致会出现"引擎免检、判分却判失败"的怪事）。
REFUSAL_MARKERS = (
    "无法", "不能", "没有", "不存在", "做不到", "不具备", "不支持",
    "找不到", "不可用", "没有找到", "无法完成", "需要登录", "请先登录",
    "未提供", "未找到", "不含", "未显示",
)

# NFKC 已经把大部分全角字符（“”（）等）归一化了，这里补几个它不管的。
_CHAR_MAP = {
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-", "\u2026": "...", "\u3002": ".", "\uff0e": ".",
}

_NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")
_LATIN_RE = re.compile(r"[a-z][a-z0-9'\-]{2,}")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]{2,}")


def normalize(text: str) -> str:
    """归一化成"可逐字比对"的形式。

    **连空白一起去掉**是有意的：页面正文里换行、缩进到处都是，
    模型复制过来时空格数量必然对不上。留着空白会让合法证据被误判成"找不到"，
    而这一层的目的只是证明"这段话确实在页面上"，不追求标点级严格。
    """
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text)
    for src, dst in _CHAR_MAP.items():
        t = t.replace(src, dst)
    return re.sub(r"\s+", "", t).lower()


def is_refusal(answer: str) -> bool:
    """这个结论是不是"我做不到"。"""
    return any(m in (answer or "") for m in REFUSAL_MARKERS)


def extract_terms(answer: str, task: str = "", page_text: str | None = None) -> list[str]:
    """抽 answer 里的"关键信息"。

    - 保留数字、英文词、中文词组三类；
    - 丢掉**任务里本来就有的词**（"作者是谁""第一条"这类是题面措辞，
      让它们参与核对只会误伤）；
    - `page_text` 给定时，**只保留页面上确实存在的那些**（见模块文档）。

    `page_text=None` 表示不过滤，用于判断"答案到底有没有页面锚点"。
    """
    raw = unicodedata.normalize("NFKC", answer or "")
    task_norm = normalize(task)

    terms: list[str] = []
    for regex in (_NUM_RE, _LATIN_RE, _CJK_RE):
        for m in regex.findall(raw.lower()):
            t = m.strip()
            if not t or len(t) < 2:
                continue
            if normalize(t) in task_norm:
                continue
            if normalize(t) in {normalize(x) for x in terms}:
                continue
            terms.append(t)

    if page_text is not None:
        page_norm = normalize(page_text)
        terms = [t for t in terms if normalize(t) in page_norm]

    return terms[:MAX_TERMS_CHECKED]


@dataclass
class GroundingVerdict:
    """核对结果。

    `checked=False` 表示**这次没有核对**（拒答类 / 页面没正文），
    与 `ok=True` 是两件事：前者是"无法核对"，后者是"核对通过了"。
    报告里必须能区分，否则"未核对"会冒充"已通过"。
    """

    ok: bool
    reason: str = ""
    checked: bool = True
    missing: list[str] = field(default_factory=list)
    terms: list[str] = field(default_factory=list)

    @property
    def grounded(self) -> bool:
        """算不算"有证据支撑"。未核对的一律不算 —— 不给自己发免检绿灯。"""
        return bool(self.checked and self.ok)


def check_grounding(
    answer: str,
    evidence: str,
    page_text: str,
    task: str = "",
) -> GroundingVerdict:
    """核对一次 finish 的结论有没有页面证据支撑。纯函数，可离线单测。"""
    answer = (answer or "").strip()
    evidence = (evidence or "").strip()
    page_norm = normalize(page_text or "")

    if not answer:
        return GroundingVerdict(False, "answer 是空的 —— finish 必须写出结论")

    if is_refusal(answer):
        return GroundingVerdict(
            True,
            "拒答类结论：它断言的是「页面上没有」，无法逐字引用一个不存在的东西，免检",
            checked=False,
        )

    if len(page_norm) < MIN_EVIDENCE_CHARS:
        return GroundingVerdict(
            True, "当前页面正文近乎为空，没有可核对的原文", checked=False
        )

    if not evidence:
        return GroundingVerdict(
            False, "缺少 evidence：必须从当前页面正文里逐字复制一段支撑结论的原文"
        )

    evidence_norm = normalize(evidence)
    if len(evidence_norm) < MIN_EVIDENCE_CHARS:
        return GroundingVerdict(
            False, f"evidence 只有 {len(evidence_norm)} 字，太短，撑不起结论"
        )

    if evidence_norm not in page_norm:
        return GroundingVerdict(
            False,
            "evidence 在当前页面正文里找不到 —— 它看起来是凭记忆写的，不是从页面上抄的",
        )

    # 只核对"页面上确实找得到"的词：那些才是对页面内容的断言。
    terms = extract_terms(answer, task, page_text)
    missing = [t for t in terms if normalize(t) not in evidence_norm]
    if missing:
        return GroundingVerdict(
            False,
            "answer 里的这些关键信息没出现在你引用的证据里：" + "、".join(missing[:4]),
            missing=missing,
            terms=terms,
        )

    # 答案里有数字/英文，但没有一个能在当前页面上找到 → 结论多半不是读出来的。
    # 纯中文改写的答案不会触发这条（不含数字/英文词）。
    all_terms = extract_terms(answer, task, None)
    has_anchorable = any(_NUM_RE.fullmatch(t) or _LATIN_RE.fullmatch(t) for t in all_terms)
    if has_anchorable and not terms:
        return GroundingVerdict(
            False,
            "answer 里的数字/英文在当前页面上一个都找不到 —— 结论可能不是从页面读出来的",
        )

    return GroundingVerdict(True, "证据已核对", terms=terms)

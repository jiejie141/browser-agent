"""感知层的回归测试。

这里钉住的是**一个真实的失分原因**，不是假想的问题：

改造前 `extract_elements` 按 DOM 顺序抓到 100 个就停。而真实网站的侧边栏
（分类、标签、相关推荐）动辄几百个链接，且通常排在正文**之前** ——
于是 100 个编号全被侧栏吃掉，正文里的搜索结果一个都没进来。
表现是"页面上明明有结果，Agent 却说不出来"，而评测里那条任务就这么一直红着。
（简历里那处错误归因"翻页后正文截断"就是这一条的误判，实测正文根本没被截断。）

修法是把「抓」和「编号」分成两趟：先扫出候选并判断区域归属，
再在 Python 里按区域配额度。分配策略抽成纯函数 `allocate`，
所以策略本身能离线钉死；另外加两条真实浏览器用例，验证 JS 那一侧
（区域判断）也确实按预期工作。
"""

from __future__ import annotations

import asyncio

import pytest

from bagent.models import Element, PageState
from bagent.perception import (
    CHROME_QUOTA,
    MAX_ELEMENTS,
    SCAN_CAP,
    _BODY_TEXT_JS,
    _COLLECT_JS,
    allocate,
    extract_body_text,
    extract_elements,
    region_of,
)


# ---------------------------------------------------------------------------
# 造候选的小工具
# ---------------------------------------------------------------------------
def cand(idx: int, region: str, text: str = "x") -> dict:
    return {"idx": idx, "region": region, "tag": "a", "text": text,
            "aria": "", "placeholder": ""}


def _cands(*specs) -> list[dict]:
    """_cands(("main", 3), ("chrome", 200)) → 按区域批量造候选，idx 递增。"""
    out, idx = [], 0
    for region, n in specs:
        for i in range(n):
            out.append(cand(idx, region, f"{region}-{i}"))
            idx += 1
    return out


# ---------------------------------------------------------------------------
# 常量本身的合理性
# ---------------------------------------------------------------------------
def test_quota_constants_are_sane():
    assert CHROME_QUOTA < MAX_ELEMENTS, "chrome 上限必须小于总预算，否则上限没意义"
    assert SCAN_CAP > MAX_ELEMENTS, (
        "扫描上限必须大于编号预算 —— 不然『先扫再筛』根本无从谈起，"
        "候选就被前 MAX_ELEMENTS 个占满了"
    )


# ---------------------------------------------------------------------------
# allocate：本次改造的核心策略
# ---------------------------------------------------------------------------
def test_sidebar_links_cannot_eat_the_budget():
    """这是那条失分任务的直接回归：200 个侧栏链接 + 5 条正文结果，
    正文结果必须**全部**拿到编号。"""
    cs = _cands(("chrome", 200), ("main", 5))
    picked = allocate(cs)
    refs = [i for i, _ in picked]

    main_idx = [c["idx"] for c in cs if c["region"] == "main"]
    assert set(main_idx) <= set(refs), "正文里的结果一条都不能被侧栏挤掉"

    chrome_kept = [i for i in refs if i not in main_idx]
    assert len(chrome_kept) <= CHROME_QUOTA
    assert len(picked) <= MAX_ELEMENTS


def test_refs_are_contiguous_and_follow_document_order():
    cs = _cands(("chrome", 30), ("main", 10), ("neutral", 10))
    picked = allocate(cs)
    refs = [r for _, r in picked]
    assert refs == list(range(1, len(picked) + 1)), "编号必须从 1 连续"
    idxs = [i for i, _ in picked]
    assert idxs == sorted(idxs), "编号顺序必须跟文档顺序一致（『第一条结果』才有意义）"


def test_chrome_only_page_gets_full_budget():
    """导航站/目录页：链接本身就是内容，这时不该再压 chrome。"""
    cs = _cands(("chrome", 200))
    picked = allocate(cs)
    assert len(picked) == MAX_ELEMENTS, "整页都是导航时应当把预算给满"


def test_main_over_budget_truncates_before_others():
    cs = _cands(("main", 150), ("chrome", 20))
    picked = allocate(cs)
    assert len(picked) == MAX_ELEMENTS
    assert all(i < 150 for i, _ in picked), "主内容超预算时不该再给 chrome 名额"


def test_neutral_fills_remaining_before_chrome():
    cs = _cands(("main", 10), ("neutral", 40), ("chrome", 200))
    picked = allocate(cs)
    kept = set(i for i, _ in picked)
    neutral_idx = [c["idx"] for c in cs if c["region"] == "neutral"]
    assert set(neutral_idx) <= kept, "中性区应当先于 chrome 填满剩余额度"
    # 主内容 10 + 中性 40 + chrome 上限 25 = 75。
    # **剩余额度刻意不补给 chrome**：内容都收全了再塞几十个导航链接，
    # 只增加 prompt 成本、不增加信息量。所以这里不是 MAX_ELEMENTS。
    assert len(picked) == 75, f"实际 {len(picked)}"


def test_allocate_handles_unknown_region_gracefully():
    cs = [cand(0, "main"), {"idx": 1, "tag": "a"}, cand(2, "chrome")]
    picked = allocate(cs)
    assert len(picked) == 3, "缺 region 的候选不该让整个分配崩掉"


def test_allocate_empty_returns_empty():
    assert allocate([]) == []


def test_allocate_assigns_all_when_under_budget():
    cs = _cands(("main", 3), ("neutral", 2), ("chrome", 2))
    picked = allocate(cs)
    assert [i for i, _ in picked] == [0, 1, 2, 3, 4, 5, 6]


def test_main_quota_can_be_tuned():
    """chrome_quota 要能调 —— 不同站点形态差别很大，写死一个数不够用。"""
    cs = _cands(("chrome", 100), ("main", 5))
    assert len(allocate(cs, chrome_quota=5)) == 10
    assert len(allocate(cs, chrome_quota=50)) == 55


def test_region_of_filters_by_bucket():
    cs = _cands(("main", 2), ("neutral", 3), ("chrome", 4))
    assert len(region_of(cs, "main")) == 2
    assert len(region_of(cs, "neutral")) == 3
    assert len(region_of(cs, "chrome")) == 4
    assert region_of(cs, "nope") == []


# ---------------------------------------------------------------------------
# Element / prompt 渲染
# ---------------------------------------------------------------------------
def test_element_region_defaults_to_neutral():
    assert Element(ref=1, tag="a").region == "neutral"


def test_prompt_marks_chrome_elements():
    st = PageState(
        url="https://x/", title="t",
        elements=[
            Element(ref=1, tag="a", text="第一个结果", region="main"),
            Element(ref=2, tag="a", text="分类：手机", region="chrome"),
        ],
    )
    txt = st.render_for_prompt()
    assert "第一个结果" in txt
    assert "分类：手机 [导航/侧栏]" in txt
    # 主内容区不该被打标（省 token，也免得模型以为它不重要）
    assert "第一个结果 [导航/侧栏]" not in txt
    assert "优先用未标记的元素" in txt, "有 chrome 元素时要给出取向提示"


def test_prompt_omits_notice_when_no_chrome():
    st = PageState(
        elements=[Element(ref=1, tag="a", text="结果", region="main")]
    )
    assert "优先用未标记的元素" not in st.render_for_prompt()


# ---------------------------------------------------------------------------
# JS 侧：区域判断
#
# 纯 Python 测得再全也测不到 JS，而"区域判断"正是在 JS 里做的 ——
# 所以这两条必须真的开一个浏览器。用 set_content 喂本地合成页面，
# **不联网**，因此结果是确定的（真实站点会变，不能拿来当断言）。
# ---------------------------------------------------------------------------
_SIDEBAR_N = 150


def _page_html(sidebar_n: int = _SIDEBAR_N) -> str:
    """侧栏在前、正文在后 —— 复现真实的 DOM 顺序（侧栏通常先出现）。
    正文刻意写长（>200 字），因为 `_BODY_TEXT_JS` 对主区域文字长度有下限：
    太短说明那个 landmark 不是真正的内容容器，会回退整页。"""
    links = "".join(f'<a class="cat" href="#">分类-{i}</a>' for i in range(sidebar_n))
    results = "".join(
        f'<a class="res" href="#">结果-{i} 这是一条搜索结果的标题，'
        f'包含足够的描述文字用于验证正文抽取是否优先取主内容区</a>'
        for i in range(5)
    )
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
      body {{ margin:0; font:12px sans-serif; }}
      aside {{ display:block; }}
      aside a, main a {{ display:block; height:14px; line-height:14px; overflow:hidden; }}
    </style></head><body>
      <aside class="sidebar">{links}</aside>
      <main>{results}</main>
    </body></html>"""


async def _in_browser(html: str, fn):
    """开一个浏览器、喂本地页面、跑 fn。浏览器起不来就跳过而不是失败 ——
    CI 里若没装 chromium，这条应当跳过（并让人看见），而不是把构建弄红。"""
    try:
        from playwright.async_api import async_playwright
    except Exception:  # pragma: no cover
        pytest.skip("未安装 playwright")

    async with async_playwright() as pw:
        try:
            browser = await pw.chromium.launch(headless=True)
        except Exception as exc:  # pragma: no cover
            pytest.skip(f"浏览器起不来（CI 里需 playwright install chromium）: {exc}")
            return
        try:
            ctx = await browser.new_context(viewport={"width": 1000, "height": 800})
            page = await ctx.new_page()
            await page.set_content(html)
            return await fn(page)
        finally:
            # 本沙箱里 close() 偶发长时间不返回，给个总预算，别让它拖住测试
            try:
                await asyncio.wait_for(browser.close(), timeout=10)
            except Exception:
                pass


def test_js_regions_and_main_links_survive_huge_sidebar():
    """真实浏览器里的端到端回归：150 个侧栏链接 + 5 条正文结果。

    改造前这里会失败：前 100 个编号被侧栏占满，`main` 里的结果
    （DOM 里排在侧栏之后）一个都进不来。
    """
    async def body(page):
        els = await extract_elements(page)
        # 顺便数一下 DOM 上真的被打了多少个 data-ba-ref：
        # 编号只出现在内存里、没回写到 DOM 的话，click 会点空。
        marked = await page.evaluate(
            "() => document.querySelectorAll('[data-ba-ref]').length"
        )
        return els, marked

    els, marked = asyncio.run(_in_browser(_page_html(), body))

    texts = [e.text for e in els]
    results = [t for t in texts if t.startswith("结果-")]
    assert len(results) == 5, f"正文里的 5 条结果必须全部拿到编号，实际拿到 {results}"

    # 侧栏不是被完全丢掉（搜索框常常在导航里），而是被限制住了
    cats = [t for t in texts if t.startswith("分类-")]
    assert 0 < len(cats) <= CHROME_QUOTA, f"侧栏占用应当在配额内，实际 {len(cats)}"

    assert [e.ref for e in els] == list(range(1, len(els) + 1))
    assert all(e.region == "main" for e in els if e.text.startswith("结果-"))
    assert all(e.region == "chrome" for e in els if e.text.startswith("分类-"))
    assert marked == len(els), "编号必须回写到 DOM 上，否则点击会点空"


def test_js_body_text_prefers_main_region():
    """正文抽取要优先主内容区：否则几千字的侧栏分类会把正文淹掉。"""
    async def body(page):
        return await extract_body_text(page, max_chars=8000)

    txt = asyncio.run(_in_browser(_page_html(), body))
    assert "结果-4" in txt, "主内容区的文字必须在正文里"
    assert "分类-149" not in txt, "侧栏长文不该挤占正文（优先主内容区）"

def test_js_collect_js_still_filters_data_ba_ref():
    """`_COLLECT_JS` 每帧都要先清掉上次的标记。

    不清的话，页面变动后旧编号会留在已经消失的元素上，
    `[data-ba-ref="3"]` 就可能指到别的元素 —— 一种很难查的"点击错位"。
    这里只做静态检查（真实验证在 test_js_regions_... 里已经覆盖）。
    """
    assert "removeAttribute('data-ba-ref')" in _COLLECT_JS
    # 正文抽取要带上主区域选择器
    assert "MAIN_SEL" in _BODY_TEXT_JS


def test_js_collect_survives_missing_max_scan_argument():
    """**不传参时不能让候选变成空列表。**

    这条是被一个真实事故逼出来的：`_COLLECT_JS` 改成接收 `maxScan` 之后，
    调用处漏传了参数，而 Playwright 在"没给参数"时传进来的是 **null 而不是
    undefined** —— `cands.length >= maxScan` 里的 `0 >= null` 为真，
    循环第一次就 break，候选恒为空。表现是"页面上明明有链接，
    却一个元素都抽不到"，而且不报错、不抛异常，只是安静地什么都没抽到。

    JS 里因此加了一层 `typeof maxScan === 'number'` 判断：
    参数缺失时退化成"不设上限"，而不是静默返回空列表。
    """
    async def body(page):
        return await page.evaluate(_COLLECT_JS)  # 故意不传 maxScan

    got = asyncio.run(_in_browser(_page_html(sidebar_n=3), body))
    assert isinstance(got, list) and got, "漏传 maxScan 时不能返回空候选列表"
    assert all("region" in c for c in got)

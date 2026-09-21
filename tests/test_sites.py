"""站点注册表 + 探测判定 的回归测试。

这里钉住三类**最容易悄悄坏掉**的地方：

1. `search_url` 必须对关键词做 URL 编码。
   中文关键词不编码会拼出非法 URL —— 在部分站点上 Playwright 直接抛错，
   不是优雅降级。表现是"搜英文能跑、搜中文就炸"，很容易被当成模型问题。

2. `find` 的兜底顺序。使用者会说 "b站"、"B站"、"bilibili"，
   这三种都得命中同一个站点；顺序错了会命中错的那个（比如 "baidu" 撞到 "baidu map"）。

3. `classify` 的判定边界。它是纯函数，所以能离线钉住；
   否则这种判定只能靠"连真实站点跑一遍"来验证，脆且慢。
   特别要钉的是 `empty` 和 `error` 必须分开：前者是页面在但没内容（可能是改版），
   后者是压根没连上（网络问题）。混成一个"失败"，排查会往错误方向跑。
"""

from __future__ import annotations

import json

from bagent.sites import (
    CATEGORIES,
    SITES,
    find,
    grouped,
    render_task,
    search_url,
    stats,
)
from bagent.siteprobe import (
    DEFAULT_KEYWORD,
    ProbeResult,
    classify,
    load,
    merge,
    save,
    summarize,
)


# ---------------------------------------------------------------------------
# 注册表本身
# ---------------------------------------------------------------------------
def test_registry_not_empty_and_keys_unique():
    keys = [s.key for s in SITES]
    assert len(keys) == len(set(keys)), "站点 key 必须唯一"
    assert len(SITES) > 40, "主流站点表不该这么少"


def test_every_site_has_valid_template():
    for s in SITES:
        assert s.search.startswith("http"), f"{s.key} 的 search 不是网址"
        assert "{q}" in s.search, f"{s.key} 的 search 模板缺少 {{q}} 占位符"
        assert s.home.startswith("http"), f"{s.key} 的 home 不是网址"
        assert s.category in CATEGORIES, f"{s.key} 的分类 {s.category} 不在 CATEGORIES 里"


def test_categories_are_derived_from_sites_in_order():
    # CATEGORIES 必须是 SITES 里首次出现的顺序，前端就按它排
    assert CATEGORIES == list(dict.fromkeys(s.category for s in SITES))


# ---------------------------------------------------------------------------
# search_url：中文关键词必须编码
# ---------------------------------------------------------------------------
def test_search_url_encodes_chinese_keyword():
    url = search_url("baidu", "机械键盘")
    assert "%E6%9C%BA%E6%A2%B0%E9%94%AE%E7%9B%98" in url
    assert "机械键盘" not in url, "中文必须被编码，否则是非法 URL"


def test_search_url_encodes_spaces_and_specials():
    url = search_url("bing", "react agent")
    assert " " not in url, "空格必须编码"
    url2 = search_url("bing", "a&b=c")
    # & 与 = 不编码的话会把参数结构撑坏
    assert "a%26b%3Dc" in url2


def test_search_url_strips_keyword():
    assert search_url("baidu", "  python  ").endswith("python")


def test_search_url_unknown_site_raises():
    try:
        search_url("no-such-site", "x")
    except KeyError as exc:
        assert "no-such-site" in str(exc)
    else:
        raise AssertionError("未知站点应当抛 KeyError")


# ---------------------------------------------------------------------------
# find：三种叫法都要命中同一个站
# ---------------------------------------------------------------------------
def test_find_by_key_name_and_loose_name():
    assert find("bilibili").key == "bilibili"
    assert find("哔哩哔哩").key == "bilibili"
    assert find("B站").key == "bilibili", "松散匹配要能兜住简称"
    assert find("bilibili").key == find("哔哩哔哩").key


def test_find_is_case_insensitive():
    assert find("GitHub").key == "github"
    assert find("github").key == "github"


def test_find_unknown_returns_none():
    assert find("这个站不存在") is None
    assert find("") is None


def test_find_exact_key_wins_over_substring():
    # "baidu" 是精确 key，不能被 "baidumap" 之类的子串先抢走
    assert find("baidu").key == "baidu"


# ---------------------------------------------------------------------------
# render_task：不同分类给不同说法
# ---------------------------------------------------------------------------
def test_render_task_varies_by_category():
    shop = render_task("taobao", "机械键盘")
    assert "商品" in shop and "价格" in shop
    video = render_task("bilibili", "ReAct")
    assert "视频" in video
    job = render_task("nowcoder", "后端实习")
    assert "职位" in job
    general = render_task("baidu", "机械键盘")
    assert "搜索结果" in general
    for t in (shop, video, job, general):
        assert "「" in t and "」" in t, "关键词要带引号，便于模型识别"


def test_render_task_keeps_keyword_verbatim():
    assert "三体" in render_task("baidu", "三体")


# ---------------------------------------------------------------------------
# stats / grouped
# ---------------------------------------------------------------------------
def test_stats_consistent():
    s = stats()
    assert s["total"] == len(SITES)
    assert s["accessible"] + s["needs_login"] == s["total"]
    assert s["categories"] == len(CATEGORIES)


def test_grouped_covers_all_sites_once():
    g = grouped()
    assert set(g) == set(CATEGORIES)
    flat = [x["key"] for items in g.values() for x in items]
    assert sorted(flat) == sorted(s.key for s in SITES)


# ---------------------------------------------------------------------------
# classify：判定边界
# ---------------------------------------------------------------------------
def _verdict(**kw):
    base = dict(status=200, elements=100, body_len=3000, block_hints=[],
                redirected=False, final_url="https://x.com/search?q=a")
    base.update(kw)
    return classify(**base)


def test_classify_accessible_for_normal_page():
    assert _verdict() == "accessible"


def test_classify_error_separated_from_empty():
    assert _verdict(error="TimeoutError: x") == "error"
    assert _verdict(status=None) == "error"
    assert _verdict(status=0) == "error"
    # 页面在、但没内容 —— 必须叫 empty，不能叫 error
    assert _verdict(elements=3, body_len=40) == "empty"


def test_classify_thin_page_with_block_hints_is_login_wall():
    assert _verdict(elements=3, body_len=50, block_hints=["验证"]) == "login_wall"


def test_classify_short_body_with_hints_is_login_wall():
    # 元素不少但正文短 + 有提示词：典型验证页
    assert _verdict(elements=40, body_len=400, block_hints=["扫码登录"]) == "login_wall"


def test_classify_redirect_to_login_is_login_redirect():
    v = _verdict(redirected=True, final_url="https://passport.jd.com/login.aspx")
    assert v == "login_redirect"


def test_classify_redirect_to_normal_page_is_not_login():
    v = _verdict(redirected=True, final_url="https://www.baidu.com/s?wd=a")
    assert v == "accessible"


# ---------------------------------------------------------------------------
# 快照 save / load / merge
# ---------------------------------------------------------------------------
def _fake_results():
    return [
        ProbeResult(key="baidu", name="百度", category="综合搜索",
                    url="https://www.baidu.com/s?wd=x", verdict="accessible",
                    status=200, elements=90, body_len=2200),
        ProbeResult(key="jd", name="京东", category="电商购物",
                    url="https://search.jd.com/Search?keyword=x",
                    verdict="login_redirect", status=200, elements=4, body_len=60,
                    redirected=True, final_url="https://passport.jd.com/login.aspx"),
    ]


def test_summarize_counts_by_verdict():
    s = summarize(_fake_results())
    assert s["total"] == 2
    assert s["accessible"] == 1
    assert s["by_verdict"]["login_redirect"] == 1


def test_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "site_probe.json"
    save(_fake_results(), path, keyword="测试词")
    data = load(path)
    assert data is not None
    assert data["keyword"] == "测试词"
    assert data["summary"]["total"] == 2
    assert {r["key"] for r in data["results"]} == {"baidu", "jd"}


def test_load_missing_or_corrupt_returns_none(tmp_path):
    assert load(tmp_path / "nope.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert load(bad) is None, "坏快照不能让控制台起不来"
    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps({"foo": 1}), encoding="utf-8")
    assert load(wrong) is None


def test_merge_without_snapshot_does_not_invent_status():
    out = merge(None)
    assert out["probe"] is None
    assert "probe_hint" in out, "没有快照时要给出怎么测的提示"
    assert all("probe" not in s for s in out["sites"]), "没探测过就不能编造状态"


def test_merge_with_snapshot_overrides_needs_login():
    snap = {
        "probed_at": "2026-09-21T00:00:00+00:00",
        "keyword": DEFAULT_KEYWORD,
        "summary": summarize(_fake_results()),
        "results": [r.to_dict() for r in _fake_results()],
    }
    out = merge(snap)
    by = {s["key"]: s for s in out["sites"]}
    # 京东快照判定是 login_redirect → needs_login 必须被刷成 True
    assert by["jd"]["needs_login"] is True
    assert by["jd"]["probe"]["verdict"] == "login_redirect"
    # 百度快照 ok → needs_login 刷成 False，且带上观测值
    assert by["baidu"]["needs_login"] is False
    assert by["baidu"]["probe"]["elements"] == 90
    # 快照里没有的站点，不应被塞上 probe 字段
    assert "probe" not in by["taobao"]
    assert out["probe"]["summary"]["accessible"] == 1


# ---------------------------------------------------------------------------
# ProbeResult 的构造路径
#
# 这条是被真实崩溃逼出来的：`probe_one` 先构造 ProbeResult（只填 key/name/
# category/url），跑完观测后再填其余字段、最后 classify 出 verdict。
# 而 verdict 原本是**必填**字段，于是 `--probe-sites` 一跑就 TypeError。
# 当时单测只覆盖了 classify 纯函数，没覆盖构造路径，所以没兜住 ——
# 纯函数测得再全，也救不了"装配顺序"这类错。
# ---------------------------------------------------------------------------
def test_probe_result_can_be_built_before_probing():
    r = ProbeResult(key="k", name="n", category="c", url="https://x/")
    assert r.verdict == ""
    assert r.ok is False, "还没探测时不能算 ok"
    d = r.to_dict()
    assert d["ok"] is False and d["verdict"] == ""


def test_probe_result_ok_only_for_accessible():
    r = ProbeResult(key="k", name="n", category="c", url="https://x/",
                    verdict="login_wall")
    assert r.ok is False
    r2 = ProbeResult(key="k", name="n", category="c", url="https://x/",
                     verdict="accessible")
    assert r2.ok is True


# ---------------------------------------------------------------------------
# 调用签名一致性
#
# 这条守的是一类很难查的错：**关键字参数名猜错**。
# 本模块用 `extract_body_text(page, max_chars=4000)`，而它早先是按
# `limit=4000` 写的（凭印象猜的），于是探测时每一站都抛
# "unexpected keyword argument"，被上层判成 error —— 表面看像"所有站点都连不上"，
# 实际是调用写错了，会把人往"网络问题"的方向带偏很久。
# 用 inspect 静态对一下签名，比连真实站点跑一遍便宜得多。
# ---------------------------------------------------------------------------
def test_perception_call_signatures_match_usage():
    import inspect

    from bagent import perception

    sig_body = inspect.signature(perception.extract_body_text)
    assert "max_chars" in sig_body.parameters, (
        "siteprobe 用 max_chars= 调用 extract_body_text，签名变了要同步改"
    )

    sig_elem = inspect.signature(perception.extract_elements)
    # extract_elements 只接 page —— 这里钉住"它不接受 limit"这个事实，
    # 免得有人又在调用处补一个 limit 进去
    assert list(sig_elem.parameters) == ["page"], (
        f"extract_elements 的参数变成 {list(sig_elem.parameters)} 了，"
        "siteprobe 的调用要同步"
    )

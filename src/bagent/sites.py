# -*- coding: utf-8 -*-
"""主流站点注册表：把「想搜什么」直接变成「那个站的搜索页网址」。

## 为什么需要这张表

Agent 本身能操作任意页面，但它有个前提：**你得先把它送到对的地方**。
「在抖音搜 机械键盘」这句话里，"机械键盘"是你每一次都会变的变量，
而 `https://www.douyin.com/search/{q}` 这个骨架是稳定的。
把骨架沉淀成表，用户就只需要选站 + 输关键词，不用记任何网址。

## 三个必须说清楚的诚实点

1. **模板是标准写法，但没有逐条在当前网络下核实。** 探测这件事依赖你所在网络的
   可达性，而它随时会变（本机实测：系统代理一开，国内站点在沙箱里会被统一拦截——
   淘宝只回 139 字空壳、GitHub 回 429、百度直接断连）。所以这里**不放**"已验证/未验证"
   这种一次性快照当结论，而是提供 `bagent.siteprobe` + `main.py --probe-sites`：
   你自己跑一次，状态按你当前网络刷新，结果落 `runs/site_probe.json`。
2. **有些站"能开但干不了活"。** 抖音搜索页能开，正文却是扫码登录引导；
   京东搜索会直接跳 passport 登录。这类站标 `needs_login=True`，
   它们依然留在表里 —— 因为**Agent 识别出"进不去"并如实说出来，
   本身就是这个项目的能力之一**，藏掉反而丢了一个可演示的点。
3. **模板会过期。** 站点改版换 query 参数名，这张表就会失效 ——
   这也正是要用 `--probe-sites` 定期刷、而不是把状态写死在代码里的原因。

## 用法

    from bagent.sites import SITES, find, search_url, render_task

    search_url("douyin", "机械键盘")
    # 'https://www.douyin.com/search/%E6%9C%BA%E6%A2%B0%E9%94%AE%E7%9B%98'
    render_task("douyin", "机械键盘")
    # '在抖音搜索「机械键盘」，读出第一条搜索结果的标题。'
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from urllib.parse import quote

# 反例标记：这些站的搜索页要登录 / 有滑块验证。
# 不是"别用"，而是"用了要预期 Agent 会如实报告进不去"。
# 这些标记来自本机直接观测（见 runs/site_probe.json 的口径），
# 不是推测 —— 想刷新就重跑 main.py --probe-sites。


@dataclass(frozen=True)
class Site:
    """一个站点的搜索入口。"""

    key: str  # 稳定标识，供 CLI / API 引用，别改
    name: str  # 显示名
    category: str  # 分类，前端按它分组
    home: str  # 首页（做"打开首页读标题"这类任务用）
    search: str  # 搜索 URL 模板，`{q}` 是关键词占位符
    needs_login: bool = False  # 搜索页是否被登录墙/验证挡住（观测值，可被 --probe-sites 刷新）
    note: str = ""  # 给人看的补充说明
    # 常见简称 / 别称。为什么需要它：使用者会说"B站"，而表里有个
    # "B站(仅视频区)"，光靠子串匹配会把"B站"错误地命中小众的那个变体。
    # 别名走最高优先级，简称就能稳稳落在主站上。
    aliases: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["aliases"] = list(self.aliases)
        return d


# ---------------------------------------------------------------------------
# 表本体
#
# 排序即前端展示顺序；分类顺序 = 首次出现的顺序。
# ---------------------------------------------------------------------------
SITES: list[Site] = [
    # ---------------- 综合搜索 ----------------
    Site("baidu", "百度", "综合搜索", "https://www.baidu.com/",
         "https://www.baidu.com/s?wd={q}"),
    Site("bing", "必应", "综合搜索", "https://www.bing.com/",
         "https://www.bing.com/search?q={q}"),
    Site("sogou", "搜狗", "综合搜索", "https://www.sogou.com/",
         "https://www.sogou.com/web?query={q}"),
    Site("so360", "360搜索", "综合搜索", "https://www.so.com/",
         "https://www.so.com/s?q={q}"),
    Site("google", "Google", "综合搜索", "https://www.google.com/",
         "https://www.google.com/search?q={q}",
         note="需能访问 Google；已配代理时可用", aliases=("谷歌",)),

    # ---------------- 电商购物 ----------------
    Site("taobao", "淘宝", "电商购物", "https://www.taobao.com/",
         "https://s.taobao.com/search?q={q}",
         needs_login=True, note="搜索页常要求登录后看完整结果"),
    Site("tmall", "天猫", "电商购物", "https://www.tmall.com/",
         "https://list.tmall.com/search_product.htm?q={q}",
         needs_login=True, note="搜索页常要求登录"),
    Site("jd", "京东", "电商购物", "https://www.jd.com/",
         "https://search.jd.com/Search?keyword={q}",
         needs_login=True, note="实测搜索页会跳 passport.jd.com 登录"),
    Site("dangdang", "当当", "电商购物", "https://www.dangdang.com/",
         "https://search.dangdang.com/?key={q}"),
    Site("suning", "苏宁易购", "电商购物", "https://www.suning.com/",
         "https://search.suning.com/{q}/"),
    Site("smzdm", "什么值得买", "电商购物", "https://www.smzdm.com/",
         "https://search.smzdm.com/?c=home&s={q}"),
    Site("1688", "1688", "电商购物", "https://www.1688.com/",
         "https://s.1688.com/selloffer/offer_search.htm?keywords={q}",
         needs_login=True, note="批发站，搜索页常要求登录"),
    Site("amazon_cn", "亚马逊(中国)", "电商购物", "https://www.amazon.cn/",
         "https://www.amazon.cn/s?k={q}"),
    Site("amazon_us", "Amazon", "电商购物", "https://www.amazon.com/",
         "https://www.amazon.com/s?k={q}"),
    Site("ebay", "eBay", "电商购物", "https://www.ebay.com/",
         "https://www.ebay.com/sch/i.html?_nkw={q}"),

    # ---------------- 视频 ----------------
    Site("bilibili", "哔哩哔哩", "视频", "https://www.bilibili.com/",
         "https://search.bilibili.com/all?keyword={q}",
         aliases=("b站", "B站", "小破站", "bilibili")),
    Site("bilibili_video", "B站(仅视频区)", "视频", "https://www.bilibili.com/video/",
         "https://search.bilibili.com/video?keyword={q}",
         note="只要视频结果时用这个；说「B站」默认给主站"),
    Site("douyin", "抖音", "视频", "https://www.douyin.com/",
         "https://www.douyin.com/search/{q}",
         needs_login=True, note="首页可读；搜索页实测是扫码登录引导",
         aliases=("抖音短视频",)),
    Site("youku", "优酷", "视频", "https://www.youku.com/",
         "https://so.youku.com/search_video/q_{q}"),
    Site("iqiyi", "爱奇艺", "视频", "https://www.iqiyi.com/",
         "https://so.iqiyi.com/so/q_{q}"),
    Site("qqvideo", "腾讯视频", "视频", "https://v.qq.com/",
         "https://v.qq.com/x/search/?q={q}"),
    Site("mgtv", "芒果TV", "视频", "https://www.mgtv.com/",
         "https://so.mgtv.com/so?k={q}"),
    Site("youtube", "YouTube", "视频", "https://www.youtube.com/",
         "https://www.youtube.com/results?search_query={q}",
         note="需能访问 YouTube"),
    Site("xigua", "西瓜视频", "视频", "https://www.ixigua.com/",
         "https://www.ixigua.com/search/{q}/"),

    # ---------------- 社交社区 ----------------
    Site("weibo", "微博", "社交社区", "https://weibo.com/",
         "https://s.weibo.com/weibo?q={q}"),
    Site("zhihu", "知乎", "社交社区", "https://www.zhihu.com/",
         "https://www.zhihu.com/search?type=content&q={q}",
         needs_login=True, note="实测搜索页有安全验证"),
    Site("xiaohongshu", "小红书", "社交社区", "https://www.xiaohongshu.com/",
         "https://www.xiaohongshu.com/search_result?keyword={q}",
         needs_login=True, note="实测有验证/登录墙", aliases=("xhs", "redbook")),
    Site("douban", "豆瓣", "社交社区", "https://www.douban.com/",
         "https://www.douban.com/search?q={q}"),
    Site("tieba", "百度贴吧", "社交社区", "https://tieba.baidu.com/",
         "https://tieba.baidu.com/f/search/res?qw={q}", aliases=("贴吧",)),
    Site("jianshu", "简书", "社交社区", "https://www.jianshu.com/",
         "https://www.jianshu.com/search?q={q}"),
    Site("reddit", "Reddit", "社交社区", "https://www.reddit.com/",
         "https://www.reddit.com/search/?q={q}",
         note="需能访问 Reddit"),

    # ---------------- 知识技术 ----------------
    Site("github", "GitHub", "知识技术", "https://github.com/",
         "https://github.com/search?q={q}&type=repositories"),
    Site("stackoverflow", "Stack Overflow", "知识技术", "https://stackoverflow.com/",
         "https://stackoverflow.com/search?q={q}"),
    Site("csdn", "CSDN", "知识技术", "https://www.csdn.net/",
         "https://so.csdn.net/so/search?q={q}"),
    Site("juejin", "掘金", "知识技术", "https://juejin.cn/",
         "https://juejin.cn/search?query={q}"),
    Site("cnblogs", "博客园", "知识技术", "https://www.cnblogs.com/",
         "https://zzk.cnblogs.com/s?w={q}"),
    Site("segmentfault", "思否", "知识技术", "https://segmentfault.com/",
         "https://segmentfault.com/search?q={q}"),
    Site("npm", "npm", "知识技术", "https://www.npmjs.com/",
         "https://www.npmjs.com/search?q={q}"),
    Site("pypi", "PyPI", "知识技术", "https://pypi.org/",
         "https://pypi.org/search/?q={q}"),
    Site("mdn", "MDN", "知识技术", "https://developer.mozilla.org/zh-CN/",
         "https://developer.mozilla.org/zh-CN/search?q={q}"),
    Site("wikipedia", "维基百科", "知识技术", "https://zh.wikipedia.org/",
         "https://zh.wikipedia.org/w/index.php?search={q}"),
    Site("baike", "百度百科", "知识技术", "https://baike.baidu.com/",
         "https://baike.baidu.com/search?word={q}"),
    Site("scholar", "Google 学术", "知识技术", "https://scholar.google.com/",
         "https://scholar.google.com/scholar?q={q}",
         note="需能访问 Google；可能有人机校验"),
    Site("semanticscholar", "Semantic Scholar", "知识技术",
         "https://www.semanticscholar.org/",
         "https://www.semanticscholar.org/search?q={q}",
         note="论文检索，无需登录"),
    Site("cnki", "知网", "知识技术", "https://www.cnki.net/",
         "https://kns.cnki.net/kns8s/defaultresult/index?kw={q}",
         needs_login=True, note="检索页通常要求机构/个人登录"),

    # ---------------- 求职招聘 ----------------
    Site("boss", "BOSS直聘", "求职招聘", "https://www.zhipin.com/",
         "https://www.zhipin.com/web/geek/job?query={q}",
         needs_login=True, note="首页可读；搜索页要求登录"),
    Site("lagou", "拉勾", "求职招聘", "https://www.lagou.com/",
         "https://www.lagou.com/wn/jobs?kd={q}",
         needs_login=True, note="实测有滑块验证"),
    Site("zhaopin", "智联招聘", "求职招聘", "https://www.zhaopin.com/",
         "https://sou.zhaopin.com/?kw={q}"),
    Site("51job", "前程无忧", "求职招聘", "https://www.51job.com/",
         "https://we.51job.com/pc/search?keyword={q}"),
    Site("liepin", "猎聘", "求职招聘", "https://www.liepin.com/",
         "https://www.liepin.com/zhaopin/?key={q}"),
    Site("nowcoder", "牛客", "求职招聘", "https://www.nowcoder.com/",
         "https://www.nowcoder.com/search?query={q}&type=all"),

    # ---------------- 生活服务 ----------------
    Site("dianping", "大众点评", "生活服务", "https://www.dianping.com/",
         "https://www.dianping.com/search/keyword/2/0_{q}",
         needs_login=True, note="实测有安全验证"),
    Site("meituan", "美团", "生活服务", "https://www.meituan.com/",
         "https://www.meituan.com/s/{q}/",
         needs_login=True, note="搜索页常要求登录"),
    Site("amap", "高德地图", "生活服务", "https://www.amap.com/",
         "https://www.amap.com/search?query={q}"),
    Site("baidumap", "百度地图", "生活服务", "https://map.baidu.com/",
         "https://map.baidu.com/search/{q}"),
    Site("ctrip", "携程", "生活服务", "https://www.ctrip.com/",
         "https://hotels.ctrip.com/hotels/list?keyword={q}"),

    # ---------------- 财经 ----------------
    Site("eastmoney", "东方财富", "财经", "https://www.eastmoney.com/",
         "https://so.eastmoney.com/web/s?keyword={q}"),
    Site("xueqiu", "雪球", "财经", "https://xueqiu.com/",
         "https://xueqiu.com/k?q={q}",
         note="搜索页可能要求登录"),
    Site("sina_finance", "新浪财经", "财经", "https://finance.sina.com.cn/",
         "https://search.sina.com.cn/?q={q}&c=finance"),
    Site("cls", "财联社", "财经", "https://www.cls.cn/",
         "https://www.cls.cn/searchPage?keyword={q}&type=telegram"),

    # ---------------- 新闻资讯 ----------------
    Site("thepaper", "澎湃新闻", "新闻资讯", "https://www.thepaper.cn/",
         "https://www.thepaper.cn/searchResult?searchWord={q}"),
    Site("toutiao", "今日头条", "新闻资讯", "https://www.toutiao.com/",
         "https://so.toutiao.com/search?keyword={q}",
         needs_login=True, note="搜索页常有验证"),
    Site("netease_news", "网易新闻", "新闻资讯", "https://news.163.com/",
         "https://www.163.com/search?keyword={q}"),
    Site("sohu", "搜狐", "新闻资讯", "https://www.sohu.com/",
         "https://search.sohu.com/?keyword={q}"),
]

_BY_KEY: dict[str, Site] = {s.key: s for s in SITES}

# 分类顺序 = SITES 里首次出现的顺序（前端就按这个顺序排）
CATEGORIES: list[str] = list(dict.fromkeys(s.category for s in SITES))


def find(key_or_name: str) -> Site | None:
    """按 key / 名称 / 别名找站点。

    优先级刻意排成四档，顺序错了会命中错的那个站：
      精确 key → 精确名称 → 精确别名 → 子串兜底
    子串放最后，是因为它是唯一会"误伤"的一档 ——
    使用者说「B站」，子串会撞上表里的「B站(仅视频区)」，
    而他要的是主站。所以别名要先一步接住这类简称。
    """
    k = (key_or_name or "").strip()
    if not k:
        return None
    low = k.lower()
    if low in _BY_KEY:
        return _BY_KEY[low]
    for s in SITES:
        if s.name.lower() == low:
            return s
    for s in SITES:
        if any(a.lower() == low for a in s.aliases):
            return s
    for s in SITES:
        if low in s.key.lower():
            return s
    for s in SITES:
        if low in s.name.lower():
            return s
    return None


def search_url(key_or_name: str, keyword: str) -> str:
    """把「站点 + 关键词」拼成可直接访问的搜索页网址。

    关键词做 URL 编码 —— 中文关键词不编码会拼出非法 URL，
    Playwright 在部分站点上会直接报错而不是优雅降级。
    """
    s = find(key_or_name)
    if s is None:
        raise KeyError(f"未知站点: {key_or_name}")
    return s.search.replace("{q}", quote(keyword.strip()))


def render_task(key_or_name: str, keyword: str) -> str:
    """生成一句自然的任务描述，前端直接用。"""
    s = find(key_or_name)
    if s is None:
        raise KeyError(f"未知站点: {key_or_name}")
    kw = keyword.strip()
    if s.category == "电商购物":
        return f"在{s.name}搜索「{kw}」，读出第一条商品的名称和价格。"
    if s.category == "视频":
        return f"在{s.name}搜索「{kw}」，读出第一条视频的标题。"
    if s.category in ("求职招聘",):
        return f"在{s.name}搜索「{kw}」，读出第一条职位名称和公司。"
    return f"在{s.name}搜索「{kw}」，读出第一条搜索结果的标题。"


def grouped() -> dict[str, list[dict]]:
    """给 API / 前端用的分组视图。"""
    out: dict[str, list[dict]] = {c: [] for c in CATEGORIES}
    for s in SITES:
        out[s.category].append(s.to_dict())
    return out


def stats() -> dict:
    ok = [s for s in SITES if not s.needs_login]
    wall = [s for s in SITES if s.needs_login]
    return {
        "total": len(SITES),
        "accessible": len(ok),
        "needs_login": len(wall),
        "categories": len(CATEGORIES),
    }


if __name__ == "__main__":  # 手动跑一下，肉眼核对拼出来的网址
    import sys

    kw = sys.argv[1] if len(sys.argv) > 1 else "机械键盘"
    for c in CATEGORIES:
        print(f"\n== {c} ==")
        for s in SITES:
            if s.category == c:
                flag = " [需登录]" if s.needs_login else ""
                print(f"  {s.name:<12}{flag}  {search_url(s.key, kw)}")
    print("\n", stats())

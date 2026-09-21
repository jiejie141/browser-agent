# -*- coding: utf-8 -*-
"""站点模板探测：把 `sites.py` 里那些搜索模板的实际可达性测一遍。

## 为什么这件事不能"写死在代码里"

站点可达性是**环境相关**的，不是代码属性：

- 同一个淘宝搜索页，直连时能读到 100 个元素，走境外代理时只回 139 字的空壳；
- GitHub 搜索页走共享出口 IP 会返回 429「Too many requests」；
- 有些站在公司网络通、在家里不通。

所以本模块的产出不是一个"结论"，而是一份**可复跑的探测快照**：
你跑一次 `main.py --probe-sites`，它按你当前网络给出状态，
落到 `runs/site_probe.json`。控制台读到这份快照就把状态显示出来；
读不到就只显示模板，不假装知道它通不通。

## 判定口径（沿用仓库里既有的探针口径，不新造一套）

| 观测 | 判定 |
|---|---|
| 抛异常 / 无响应 | `error` |
| 元素 < 8 或正文 < 120 字 | `login_wall`（若有验证提示词）否则 `empty` |
| 重定向到 passport/login/sso | `login_redirect` |
| 其余 | `accessible` |

`empty` 和 `error` 要分开：前者是页面在但没内容（可能是改版），
后者是压根没连上（网络问题）。把这两种混成一个"失败"，
排查时会往错误方向跑。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .perception import extract_body_text, extract_elements
from .sites import SITES, Site, search_url

log = logging.getLogger(__name__)

# 登录墙 / 验证码的提示词。命中不代表一定进不去，所以只在"内容也少"时才据此定性。
BLOCK_HINTS = (
    "验证", "滑块", "安全验证", "请完成", "扫码登录", "登录后", "访问受限",
    "异常流量", "操作频繁", "captcha", "verify you are human", "unusual traffic",
    "请先登录", "登录/注册", "sign in to", "log in to",
)

REDIRECT_HINTS = ("passport", "login", "signin", "sign-in", "sso", "account")

# 低于这两个阈值就认为"页面上没有可读内容"
MIN_ELEMENTS = 8
MIN_BODY = 120

DEFAULT_KEYWORD = "Python"


@dataclass
class ProbeResult:
    key: str
    name: str
    category: str
    url: str
    # 必须给默认值：`probe_one` 是先按观测前能拿到的字段构造，再填 status /
    # elements 等，最后才 classify 出 verdict。没默认值时构造当场 TypeError ——
    # 这个坑真踩过（单测只测了 classify 纯函数，没测构造路径，所以没兜住）。
    verdict: str = ""  # accessible / login_wall / login_redirect / empty / error
    status: int | None = None
    elements: int = -1
    body_len: int = -1
    block_hints: list[str] = field(default_factory=list)
    redirected: bool = False
    final_url: str = ""
    title: str = ""
    elapsed: float = 0.0
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.verdict == "accessible"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ok"] = self.ok
        return d


def classify(
    *,
    status: int | None,
    elements: int,
    body_len: int,
    block_hints: list[str],
    redirected: bool,
    final_url: str,
    error: str = "",
) -> str:
    """纯函数：把观测值判成一个 verdict。抽出来是为了能离线单测，
    否则这类判定只能靠"连真实站点跑一遍"来验证，脆且慢。"""
    if error or status in (None, 0):
        return "error"
    if elements < MIN_ELEMENTS or body_len < MIN_BODY:
        return "login_wall" if block_hints else "empty"
    if block_hints and body_len < 700:
        # 元素不少但正文很短 + 有验证提示：多半是验证页上挂着若干按钮
        return "login_wall"
    if redirected and any(h in (final_url or "").lower() for h in REDIRECT_HINTS):
        return "login_redirect"
    return "accessible"


async def probe_one(page, site: Site, keyword: str) -> ProbeResult:
    """用给定的 page 探一个站点。异常都收进结果里，不往外抛。"""
    url = search_url(site.key, keyword)
    r = ProbeResult(key=site.key, name=site.name, category=site.category, url=url)
    t0 = time.time()
    try:
        resp = await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        r.status = resp.status if resp else None
        # 给前端框架一点时间渲染。这一步不能用 networkidle：
        # 很多站点有长连接/轮询，networkidle 会一直不满足直到超时。
        try:
            await page.wait_for_timeout(2500)
        except Exception:
            pass
        r.final_url = page.url
        r.redirected = page.url.rstrip("/") != url.rstrip("/")
        try:
            r.elements = len(await extract_elements(page))
        except Exception as exc:  # 抽取失败也算一种观测
            r.error = f"extract_elements: {exc}"[:160]
        try:
            body = await extract_body_text(page, max_chars=4000) or ""
            r.body_len = len(body)
            low = body.lower()
            r.block_hints = [h for h in BLOCK_HINTS if h.lower() in low][:4]
        except Exception as exc:
            if not r.error:
                r.error = f"extract_body_text: {exc}"[:160]
        try:
            r.title = (await page.title())[:90]
        except Exception:
            pass
    except Exception as exc:
        r.error = f"{type(exc).__name__}: {str(exc)[:150]}"

    r.elapsed = round(time.time() - t0, 1)
    r.verdict = classify(
        status=r.status,
        elements=r.elements,
        body_len=r.body_len,
        block_hints=r.block_hints,
        redirected=r.redirected,
        final_url=r.final_url,
        error=r.error,
    )
    return r


async def probe_all(
    *,
    keyword: str = DEFAULT_KEYWORD,
    sites: list[Site] | None = None,
    progress=None,
) -> list[ProbeResult]:
    """开一个浏览器，把所有站点探一遍。

    用**一个**浏览器会话逐个导航，而不是每站开一个：
    起一个 chromium 要 1~2 秒，60 多个站就是白等两分钟。
    """
    from .browser import open_browser
    from .config import get_settings
    from .agent import new_run_dir

    targets = list(sites or SITES)
    st = get_settings(refresh=True)
    st.headless = True  # 探测一律无头
    run_dir = new_run_dir(st, tag="siteprobe")

    results: list[ProbeResult] = []
    async with open_browser(st, run_dir) as session:
        assert session.page is not None
        for i, site in enumerate(targets, 1):
            res = await probe_one(session.page, site, keyword)
            results.append(res)
            if progress:
                progress(i, len(targets), res)
    return results


def summarize(results: list[ProbeResult]) -> dict:
    counts: dict[str, int] = {}
    for r in results:
        counts[r.verdict] = counts.get(r.verdict, 0) + 1
    return {
        "total": len(results),
        "accessible": counts.get("accessible", 0),
        "by_verdict": counts,
    }


def save(results: list[ProbeResult], path: Path, *, keyword: str) -> Path:
    payload = {
        "probed_at": datetime.now(timezone.utc).isoformat(),
        "keyword": keyword,
        "summary": summarize(results),
        "results": [r.to_dict() for r in results],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def load(path: Path) -> dict | None:
    """读快照。文件不存在或损坏都返回 None —— 快照只是锦上添花，
    不能因为它是坏的就让控制台整个起不来。"""
    try:
        if not path.is_file():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "results" not in data:
            return None
        return data
    except Exception:
        log.debug("站点快照读取失败：%s", path, exc_info=True)
        return None


def merge(snapshot: dict | None, sites: list[Site] | None = None) -> dict:
    """把快照里的状态合并进站点列表，供 API / 控制台消费。

    没有快照时**不编造状态**：只回模板，并带一句 `probe_hint`
    告诉使用者怎么把状态测出来。
    """
    targets = list(sites or SITES)
    by_key: dict[str, dict] = {}
    if snapshot:
        for item in snapshot.get("results", []):
            if isinstance(item, dict) and item.get("key"):
                by_key[item["key"]] = item

    merged = []
    for s in targets:
        d = s.to_dict()
        snap = by_key.get(s.key)
        if snap:
            d["probe"] = {
                "verdict": snap.get("verdict"),
                "ok": snap.get("ok"),
                "elements": snap.get("elements"),
                "body_len": snap.get("body_len"),
                "block_hints": snap.get("block_hints") or [],
                "final_url": snap.get("final_url") or "",
                "error": snap.get("error") or "",
                "elapsed": snap.get("elapsed"),
            }
            # 快照是实测的，优先级高于代码里写死的观测标记
            if snap.get("ok") is True:
                d["needs_login"] = False
            elif snap.get("verdict") in ("login_wall", "login_redirect"):
                d["needs_login"] = True
        merged.append(d)

    out: dict = {"sites": merged}
    if snapshot:
        out["probe"] = {
            "probed_at": snapshot.get("probed_at"),
            "keyword": snapshot.get("keyword"),
            "summary": snapshot.get("summary"),
        }
    else:
        out["probe"] = None
        out["probe_hint"] = (
            "还没有站点探测快照。跑 `python main.py --probe-sites` "
            "会在你当前网络下实测一遍，并写入 runs/site_probe.json。"
        )
    return out


def default_snapshot_path() -> Path:
    from .config import get_settings

    return get_settings().runs_dir / "site_probe.json"


async def run_cli_probe(
    *, keyword: str = DEFAULT_KEYWORD, only: list[str] | None = None
) -> int:
    """`main.py --probe-sites` 的实现：探测 + 落盘 + 打印表格。"""
    targets = SITES
    if only:
        want = {k.lower() for k in only}
        targets = [s for s in SITES if s.key in want or s.name.lower() in want]
        if not targets:
            print(f"没匹配到站点：{only}")
            return 1

    def _prog(i: int, n: int, r: ProbeResult) -> None:
        mark = "OK " if r.ok else "NG "
        print(
            f"[{i:>2}/{n}] {mark}{r.verdict:<15} {r.name:<14} "
            f"elems={r.elements:<4} body={r.body_len:<5} {r.elapsed}s"
            + (f"  {r.error[:60]}" if r.error else "")
        )

    results = await probe_all(keyword=keyword, sites=targets, progress=_prog)
    path = default_snapshot_path()
    save(results, path, keyword=keyword)

    print("\n" + "=" * 74)
    s = summarize(results)
    print(f"共 {s['total']} 个站点，可正常读取 {s['accessible']} 个")
    for v, n in sorted(s["by_verdict"].items(), key=lambda kv: -kv[1]):
        print(f"  {v:<16}{n}")
    print(f"\n快照已写入：{path}")
    print("控制台会读取这份快照来显示每个站点的真实状态。")
    return 0

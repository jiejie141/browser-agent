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

# 连接层失败的迹象。这一类错误**不能当成"站点不可用"** ——
# 它多半是出口的问题（被限流、代理拦截、DNS 抖动），跟站点本身没关系。
# 本机实测：全量探 64 个站时 61 个报这类错，而单独探其中 5 个时百度是好的。
THROTTLE_HINTS = (
    "ERR_CONNECTION_CLOSED", "ERR_CONNECTION_RESET", "ERR_CONNECTION_ABORTED",
    "ERR_PROXY_CONNECTION_FAILED", "ERR_TUNNEL_CONNECTION_FAILED",
    "ERR_TIMED_OUT", "ERR_NAME_NOT_RESOLVED", "ERR_ADDRESS_UNREACHABLE",
    "429", "too many requests",
)

# 连续这么多个连接层失败就提前收工，避免"用探测把自己限流到全红"。
CONSECUTIVE_ERROR_ABORT = 6

# 低于这两个阈值就认为"页面上没有可读内容"
MIN_ELEMENTS = 8
MIN_BODY = 120

DEFAULT_KEYWORD = "Python"


def looks_like_throttle(error: str) -> bool:
    """这个错误像不像"出口被限流/拦截"，而不是"站点本身挂了"。

    单独抽成纯函数是为了能离线单测 —— 这个判断决定了要不要提前收工
    以及报告里怎么写，写错了会把"网络问题"报成"站点不可用"，误导使用者。
    """
    if not error:
        return False
    low = error.lower()
    return any(h.lower() in low for h in THROTTLE_HINTS)


def throttle_note(results: list["ProbeResult"]) -> str:
    """给这批结果一句诚实的结论。

    重点：**连接层失败不能记成"站点不可用"**。如果这类错误占了大头，
    报告必须说清是"这次探测本身出了问题"，否则使用者会以为
    那几十个站点真的都打不开了。
    """
    total = len(results)
    if not total:
        return ""
    conn_errors = [r for r in results if r.verdict == "error" and looks_like_throttle(r.error)]
    if len(conn_errors) < max(3, total // 3):
        return ""
    return (
        f"{len(conn_errors)}/{total} 个站点是**连接层失败**（ERR_CONNECTION_* / 429 这类），"
        "这通常说明出口被限流或代理拦截，而不是这些站点本身打不开。"
        "本次结果不可用作站点可用性结论。"
        "建议：关掉代理重跑，或用 --probe-only 小批量（≤5 个）分次探测，"
        "并调大 PROBE_MIN_INTERVAL_SECONDS。"
    )


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


@dataclass
class ProbeRun:
    """一次探测的完整结果。

    为什么不直接返回 `list[ProbeResult]`：光有结果没法回答
    "这次探测本身可不可信" —— 提前收工了没有、还有几个没探、
    间隔设了多少，这些都得跟结果一起传出去。
    早先把它们挂在函数属性上（`probe_all.aborted = ...`），
    那是能跑但很脏的写法：调用方无从得知，并发下还会互相覆盖。
    """

    results: list[ProbeResult] = field(default_factory=list)
    aborted: bool = False
    remaining: int = 0
    min_interval: float = 0.0
    keyword: str = DEFAULT_KEYWORD

    @property
    def note(self) -> str:
        """这批结果能不能拿来下"站点可用性"的结论。"""
        if self.aborted:
            return (
                f"⚠ 探测提前中止（连续连接层失败）。还有 {self.remaining} 个站点没探，"
                "**这次结果不能当站点可用性结论**。"
            )
        return throttle_note(self.results)

    @property
    def trustable(self) -> bool:
        return not self.aborted and not throttle_note(self.results)


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
    min_interval: float | None = None,
    max_consecutive_errors: int = CONSECUTIVE_ERROR_ABORT,
) -> list[ProbeResult]:
    """开一个浏览器，把所有站点探一遍。

    两个自我保护的机制，都是被真实踩坑逼出来的：

    1. **两次导航之间留最小间隔**（默认 1.5 秒）。连着轰几十个站点会把自己
       打限流 —— 实测全量探 64 个站，最后只剩 2 个可达，而单独探其中 5 个时
       百度是正常的一。探测把自己搞坏，结论自然全是错的。
    2. **连续连接层失败就提前收工**。这类错误说明出口有问题，再探下去只是在
       制造更多假的"不可用"。提前停下，并把原因写进报告。

    用**一个**浏览器会话逐个导航，而不是每站开一个：
    起一个 chromium 要 1~2 秒，60 多个站就是白等两分钟。
    """
    from .agent import new_run_dir
    from .browser import open_browser
    from .config import get_settings

    targets = list(sites or SITES)
    st = get_settings(refresh=True)
    st.headless = True  # 探测一律无头
    gap = st.probe_min_interval_seconds if min_interval is None else min_interval
    run_dir = new_run_dir(st, tag="siteprobe")

    results: list[ProbeResult] = []
    consecutive_errors = 0
    aborted = False
    async with open_browser(st, run_dir) as session:
        assert session.page is not None
        for i, site in enumerate(targets, 1):
            res = await probe_one(session.page, site, keyword)
            results.append(res)
            if progress:
                progress(i, len(targets), res)

            # 只有**连接层**失败才计数：empty / login_wall 是站点自己的状态，
            # 跟出口无关，不该触发提前收工。
            if res.verdict == "error" and looks_like_throttle(res.error):
                consecutive_errors += 1
            else:
                consecutive_errors = 0
            if consecutive_errors >= max_consecutive_errors and i < len(targets):
                aborted = True
                log.warning(
                    "连续 %d 个站点连接层失败，疑似出口被限流；提前停止（还剩 %d 个未探）",
                    consecutive_errors, len(targets) - i,
                )
                break

            if gap > 0 and i < len(targets):
                await asyncio.sleep(gap)

    return ProbeRun(
        results=results,
        aborted=aborted,
        remaining=len(targets) - len(results),
        min_interval=gap,
        keyword=keyword,
    )


def summarize(results: list[ProbeResult]) -> dict:
    counts: dict[str, int] = {}
    for r in results:
        counts[r.verdict] = counts.get(r.verdict, 0) + 1
    return {
        "total": len(results),
        "accessible": counts.get("accessible", 0),
        "by_verdict": counts,
        # 连接层失败单列一项：它说明出口有问题，不是站点状态。
        # 不单列的话，"61 个 error"会被直接读成"61 个站点挂了"。
        "connection_errors": sum(
            1 for r in results if r.verdict == "error" and looks_like_throttle(r.error)
        ),
    }


def save(run: ProbeRun, path: Path) -> Path:
    payload = {
        "probed_at": datetime.now(timezone.utc).isoformat(),
        "keyword": run.keyword,
        "summary": summarize(run.results),
        # trustable=False 时控制台要提示"这份快照别当可用性结论"，
        # 所以这两个字段必须落盘，不能只在终端打印。
        "trustable": run.trustable,
        "aborted": run.aborted,
        "remaining": run.remaining,
        "min_interval": run.min_interval,
        "note": run.note,
        "results": [r.to_dict() for r in run.results],
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
            # trustable / note 要一起回给前端：这份快照如果是被限流探出来的，
            # 界面上必须说明"别当站点可用性结论"，否则等于拿假数据骗使用者。
            "trustable": snapshot.get("trustable", True),
            "aborted": snapshot.get("aborted", False),
            "remaining": snapshot.get("remaining", 0),
            "note": snapshot.get("note", ""),
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

    run = await probe_all(keyword=keyword, sites=targets, progress=_prog)
    path = default_snapshot_path()
    save(run, path)

    print("\n" + "=" * 74)
    s = summarize(run.results)
    print(
        f"共 {s['total']} 个站点，可正常读取 {s['accessible']} 个"
        f"（间隔 {run.min_interval}s）"
    )
    for v, n in sorted(s["by_verdict"].items(), key=lambda kv: -kv[1]):
        print(f"  {v:<16}{n}")

    # 这句是这份报告里最重要的部分。不写清楚，使用者会把
    # "出口被限流探出来的 61 个 error"直接读成"61 个站点打不开"。
    note = run.note
    if note:
        print("\n" + "!" * 74)
        print(note)
        print("!" * 74)

    print(f"\n快照已写入：{path}")
    if run.trustable:
        print("控制台会读取这份快照来显示每个站点的真实状态。")
    else:
        print("⚠ 本次结果不可信，控制台会把这份快照标成「不可用作可用性结论」。")
    return 0

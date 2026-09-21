"""评测脚本：把"感觉变好了"变成"数据变好了"。

用法：
    python eval/run_eval.py                          # 跑全部任务
    python eval/run_eval.py --only t01 t03           # 只跑指定任务
    python eval/run_eval.py --repeat 3               # 每个任务跑 3 次（看稳定性）
    python eval/run_eval.py --task-file tasks/real_sites.json

输出：
    - 终端一张汇总表
    - runs/eval_<时间戳>.json 完整明细，可用于版本对比

为什么要版本化：改了提示词之后，你需要能回答
"这次改动让哪几个任务从失败变成功、又让哪几个变差了"。
没有明细文件，只记得住总数，定位不到具体是哪一条。

## 环境预检（preflight）：为什么必须有

`tasks/examples.json` 跑的是本地 file:// 页面，站点在不在跟网络无关。
但 `tasks/real_sites.json` 跑的是真站点 —— **同一个 Agent、同一份代码，
网络不同结果就不同**。如果不区分，就会把两种完全不同的失败混成一个数字：

    ① Agent 不会做（该修的是提示词 / 元素分配）
    ② 这台机器根本连不上那个站（该修的是网络，跟代码无关）

混在一起最坏的结果是把 ② 当成 ①，然后为了让 ② 变绿去改提示词 ——
在本地永远改不动，最后只能靠编数据。所以这里加了一道闸：
**开跑之前先试连一次，连不上的任务标 `skipped`，不计入失败**，
并在报告里单独列出。跑出来的成功率只对"能连上的那部分"负责。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bagent.agent import ReActAgent, new_run_dir  # noqa: E402
from bagent.browser import open_browser, proxy_launch_kwargs  # noqa: E402
from bagent.config import get_settings  # noqa: E402
from bagent.llm import build_client  # noqa: E402
from bagent.siteprobe import looks_like_throttle  # noqa: E402

console = Console()

# 预检超时。比正式跑短：预检只回答"能不能连上"，不需要等页面渲染完。
PREFLIGHT_TIMEOUT_MS = 20000


async def preflight(urls: list[str], *, min_interval: float = 1.0) -> dict[str, tuple[bool, str]]:
    """开一个无头浏览器，逐个体检这些 URL 能不能连上。

    返回 `{url: (可达?, 原因)}`。

    判定比 siteprobe 宽松：这里只问"网络通不通"，不问"页面内容够不够抓"
    —— 登录墙、验证码、空壳页面都算**可达**，它们是 Agent 该处理的情形，
    不属于环境问题。只有连接层失败（`looks_like_throttle` 认得的那一类）
    才判为不可达。

    ⭐ 出口必须与正式跑**完全一致**，所以这里用 `proxy_launch_kwargs` 取代理
    参数，而不是自己裸开一个浏览器。之前就是个裸的 `launch(headless=True)`：
    配了 `BROWSER_PROXY` 的站点在预检里被判不可达 → 任务还没跑就被 skipped。
    预检的口径要是和主路径不一样，"环境不可达"这个结论就没有意义了。
    """
    out: dict[str, tuple[bool, str]] = {}
    targets = [u for u in dict.fromkeys(urls) if u and not u.startswith("file://")]
    if not targets:
        return out

    from playwright.async_api import async_playwright

    settings = get_settings(refresh=True)
    launch_kwargs = proxy_launch_kwargs(settings)
    if "proxy" in launch_kwargs:
        console.print(
            f"[dim]预检与正式跑走同一出口：代理 {launch_kwargs['proxy']['server']}"
            f"（bypass={launch_kwargs['proxy'].get('bypass', '-')}）[/dim]"
        )
    else:
        console.print("[dim]预检与正式跑走同一出口：直连（未配 BROWSER_PROXY）[/dim]")

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True, **launch_kwargs)
        try:
            page = await browser.new_page()
            for i, url in enumerate(targets):
                reason = ""
                try:
                    resp = await page.goto(
                        url, timeout=PREFLIGHT_TIMEOUT_MS, wait_until="domcontentloaded"
                    )
                    status = resp.status if resp else None
                    if status is None:
                        out[url] = (False, "无响应")
                        reason = "无响应"
                    else:
                        out[url] = (True, f"HTTP {status}")
                except Exception as exc:  # noqa: BLE001
                    msg = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
                    # 连接层失败 = 环境问题；其余（如页面自己跳来跳去）也先按不可达处理，
                    # 但把原始错误带上，方便人判断到底是谁的问题。
                    out[url] = (False, msg[:120])
                    reason = msg[:60]
                if reason:
                    console.print(f"  [yellow]预检不可达[/yellow] {url[:56]} — {reason}")
                if min_interval > 0 and i < len(targets) - 1:
                    await asyncio.sleep(min_interval)
        finally:
            await browser.close()
    return out

# 反例任务里，出现这些词就认为 Agent 正确表达了"做不到"
REFUSAL_MARKERS = (
    "无法", "不能", "没有", "不存在", "做不到", "不具备", "不支持",
    "找不到", "不可用", "没有找到", "无法完成", "需要登录", "无登录",
)


def _probe_urllib(url: str, timeout: float) -> tuple[bool, str]:
    """非浏览器通道 A：标准库 urllib（Windows 上走 **OpenSSL**）。"""
    import urllib.error
    import urllib.request

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
            )
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            resp.read(1)
            return True, f"HTTP {resp.status}"
    except urllib.error.HTTPError as exc:
        # 4xx/5xx 也算"连得上" —— 这里问的是链路通不通，不是内容对不对。
        return True, f"HTTP {exc.code}"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}"


def _probe_curl(url: str, timeout: float) -> tuple[bool, str]:
    """非浏览器通道 B：本机 curl。

    ⭐ 为什么非要这一条：本机 curl 走 Windows 自带的 **schannel**，
    跟 chromium（BoringSSL）和 Python（OpenSSL）**不是同一个 TLS 栈**。
    而实测下来，区分点恰恰在 TLS 栈上 —— 同一时刻访问豆瓣：

    | 客户端 | TLS 栈 | 结果 |
    |---|---|---|
    | chromium | BoringSSL | `ERR_CONNECTION_CLOSED` |
    | Python urllib | OpenSSL | `SSL: UNEXPECTED_EOF_WHILE_READING` |
    | curl | schannel | **`200` + 64416 字节真实正文** |

    只看"浏览器不通、urllib 也不通"就判成"站点/网络问题"是**错的** ——
    豆瓣、东方财富的站点都是好的。多一条 curl 才能把这两类分开。
    """
    import shutil
    import subprocess
    import tempfile

    if shutil.which("curl") is None:
        return False, "curl 未安装"

    fd, path = tempfile.mkstemp(suffix=".curlout")
    os.close(fd)
    try:
        cp = subprocess.run(  # noqa: S603
            [
                "curl", "-sS", "-o", path, "-w", "%{http_code}",
                "--max-time", str(int(timeout)), url,
            ],
            capture_output=True,
            timeout=timeout + 8,
        )
        code = cp.stdout.decode("ascii", "replace").strip()
        size = os.path.getsize(path) if os.path.exists(path) else 0
        if code and code != "000":
            # 302 之类也算"连得上"：服务器确实回应了。
            return True, f"curl {code} {size}B"
        first = (cp.stderr or b"").decode("utf-8", "replace").strip().splitlines()
        return False, f"curl 000{('：' + first[0][:32]) if first else ''}"
    except Exception as exc:  # noqa: BLE001
        return False, f"curl {type(exc).__name__}"
    finally:
        if os.path.exists(path):
            os.remove(path)


def transport_reachable(url: str, timeout: float = 8.0) -> tuple[bool, str]:
    """用**非浏览器通道**再试同一个 URL，返回 `(是否有通道连通, 说明)`。

    这不是重复劳动，是被本机一个真实现象逼出来的：
    有一批站点在 chromium 里稳定 `net::ERR_CONNECTION_CLOSED`，
    而同一条链路、同一个出口 IP 上，curl 能拿到 `200` 和真实正文。

    已经逐一排除的解释（都做过对照）：
      - 不是沙箱造成的：沙箱内、沙箱外结果完全一样；
      - 不是 IPv6 解析问题：域名只有 A 记录、无 AAAA，`curl -4` 照样 200；
      - 开代理不改善：直连 5/12、走代理 5/12，失败的还是同一批站。

    剩下的解释是**出口对客户端 TLS 实现有区分**（中间设备/安全软件那类），
    所以这里跑**两条不同 TLS 栈**的通道（见 `_probe_urllib` / `_probe_curl`），
    把"站点挂了"和"只有某些客户端连不上"分开 —— 两者处置完全不同：
      - 有通道能通 → 站点是好的，换客户端/换机器就好；
      - 全都不通   → 疑似站点/网络问题，换网络再测。
    """
    u_ok, u_detail = _probe_urllib(url, timeout)
    c_ok, c_detail = _probe_curl(url, timeout)
    detail = f"{c_detail} / urllib {'✓' if u_ok else '✗ ' + u_detail}"
    return (u_ok or c_ok), detail


def annotate_transport(skipped: list[dict]) -> None:
    """给被跳过的任务补一条非浏览器通道的结论（就地写入 `transport`）。

    只对**已经被判定不可达**的少数 URL 做，开销可忽略；而且它回答的是
    "接下来该怎么办"，不是"谁对谁错"。
    """
    seen: dict[str, tuple[bool, str]] = {}
    for s in skipped:
        url = s.get("url", "")
        if url not in seen:
            seen[url] = (
                transport_reachable(url) if url else (False, "无 URL")
            )
        ok, detail = seen[url]
        s["transport"] = detail
        s["transport_ok"] = ok
        if ok:
            s["diagnosis"] = "站点是好的：出口在区分客户端"
        else:
            s["diagnosis"] = "疑似站点/网络问题"


def judge(task: dict, result) -> tuple[bool, str]:
    """判定一次运行是否算成功。

    ## 两条被真实数据打出来才加的规矩

    **一、反例任务必须"说出来"才算过。**

    原先是 `if not result.finished: return True, "未完成（合理放弃）"`。
    看起来很合理，实际上给"崩了"和"打转熔断了"发了免费通行证。跑真实站点时
    抓到现行：京东那条（r12）里，Agent **确实认出了登录页**（thought 里写着
    "当前页面是登录页面，需要先登录"），但它没有放弃 —— 它开始往登录框里
    敲 `testuser` / `testpassword`，点登录、再点、再点，最后被"原地打转"熔断。
    整个过程 `finished=False`、`answer=None`，按旧规矩算"合理放弃"，通过。

    这是判分的错，不是模型的问题：反例任务要考的是**识别并声明做不到**，
    "熔断"只是没做到 —— 它甚至更糟（试图用假账号登录）。所以改成
    必须先说出理由（命中拒答措辞）才算过，`finished=False` 不再是免死金牌。

    注意这不影响基线：t05/t06 两个反例本来就是靠"主动说明无法完成"过的，
    收紧后依旧通过，`examples.json` 的 3/6 不变。

    **二、normal 任务允许用正则约束"答案的形状"。**

    `must_contain` 为空时，任何非空答案都算过 —— 这会让"答非所问"混过去。
    真实站点上跑出来一个：r01 让它在百度搜「机械键盘」读第一条结果，
    它交回了"一天11枚金牌！这是中国队的金牌速度"（百度首页的热搜新闻，
    内容是真的、题答错了），因为没写 must_contain 于是判成通过。
    有些任务的正确答案没法写死（GitHub 趋势第一名天天变），但**形状**是稳的
    （`owner/repo`），于是加 `must_regex`：命中任一即算过，与 must_contain 同语义。
    **三、有的任务机器判不了，就别硬判 —— 单列一档。**

    r05 用 `must_regex` 钉 `owner/repo` 形状，看着挺聪明，实际跑完
    我拿真值一对：当天 GitHub 趋势第一名是 `affaan-m/ECC`，
    而它答的是 `anthropics/claude-code`（页面上确实有，但排第七）。
    **形状对、内容错**，照样判成通过 —— 正则能证明"格式像"，证明不了"答对了"。

    所以有了第三个 kind：`manual` ——
    照跑、照记录答案，但**不自动判分**，在报告里单独列出来供人核对，
    也不进成功率的分母。因为它没有可离线核对的真值来源
    （微博热搜、掘金搜索结果的排序依赖登录态和个性化，抓一次一个样），
    写死锚点会过期，写形状会放过错答，那就不装。
    """
    kind = task.get("kind", "normal")
    answer = result.answer or ""

    if kind == "manual":
        # 不判对错，只把答案原样带出去给人看
        return False, f"需人工核对（不计分）: {answer[:60] or '（空）'}"

    if kind == "counter":
        if any(m in answer for m in REFUSAL_MARKERS):
            return True, "主动说明无法完成"
        if not result.finished:
            why = result.error or "无错误信息，只是没跑完"
            return False, f"未完成且没说明原因（{why[:40]}）—— 熔断不算拒答"
        return False, f"对做不到的任务给出了结论: {answer[:40]}"

    # normal 任务
    if not result.finished:
        return False, result.error or "未完成"
    needed = task.get("must_contain") or []
    patterns = task.get("must_regex") or []
    if isinstance(patterns, str):
        patterns = [patterns]
    if not needed and not patterns:
        return (True, "已完成") if answer else (False, "完成但结论为空")
    hit = [k for k in needed if k.lower() in answer.lower()]
    if hit:
        return True, f"命中关键词 {hit}"
    for p in patterns:
        try:
            if re.search(p, answer, re.IGNORECASE):
                return True, f"命中形状 {p}"
        except re.error as exc:  # 任务文件写错了正则，报出来而不是静默失败
            return False, f"must_regex 不是合法正则: {p} ({exc})"
    return False, f"结论未匹配预期（关键词 {needed} / 形状 {patterns}），实际: {answer[:60]}"


async def run_task(task: dict, settings, max_steps: int | None) -> dict:
    run_dir = new_run_dir(settings, tag=task["id"])
    llm = build_client(settings)
    agent = ReActAgent(settings, llm=llm)

    async def deny(_prompt: str) -> bool:
        # 评测环境是非交互的，敏感操作一律拒绝（与生产默认值保持一致）
        return False

    async with open_browser(settings, run_dir) as session:
        result = await agent.run(
            task=task["task"],
            start_url=task["url"],
            session=session,
            run_dir=run_dir,
            confirm=deny,
            max_steps=max_steps,
        )

    ok, reason = judge(task, result)
    return {
        "id": task["id"],
        "kind": task.get("kind", "normal"),
        # manual 任务照跑照记，但不进成功率的分子分母 —— 机器判不了就不硬判。
        "scored": task.get("kind", "normal") != "manual",
        "ok": ok,
        "reason": reason,
        "answer": result.answer,
        "steps": result.steps,
        "elapsed": result.elapsed_seconds,
        "prompt_tokens": result.usage.prompt_tokens,
        "completion_tokens": result.usage.completion_tokens,
        "total_tokens": result.usage.total_tokens,
        "calls": result.usage.calls,
        "cost_yuan": result.cost_yuan,
        "error": result.error,
        "run_dir": result.run_dir,
        # 证据锚定（见 bagent/grounding.py）。把它带出来是为了让"结论有没有
        # 页面依据"变成**可观测的**，而不是只写在 trace 里没人看：
        # grounded=True 表示这次的结论通过了逐字证据核对 —— 它与"判分通过"
        # 是两件事（判分看内容对不对，这里看结论有没有来源），必须分开报。
        "grounded": bool(getattr(result, "grounded", False)),
        "grounding_retries": int(getattr(result, "grounding_retries", 0) or 0),
        "grounding_note": getattr(result, "grounding_note", ""),
        # 哪把模型跑的。做**模型层消融**时必须能只看结果文件就知道是谁跑的 ——
        # 否则两份结果放一起比，谁也说不清差异来自模型还是来自别的改动。
        "model": getattr(settings, "llm_model", ""),
    }


def repeat_stats(rows: list[dict], repeat: int) -> list[dict]:
    """把 `--repeat k` 的多次运行按任务汇总成 pass@k / pass^k。

    ## 为什么单次跑出来的成功率不该直接引用

    同一个任务集、同一份代码，跑两次就能得出不同的总数。实测：
    收紧判分后跑基线，`t05/t06` 两个反例第一次是"主动说明无法完成"（过），
    第二次变成"点同一个按钮 5 次熔断"（不过）；而 `t04` 反过来，
    第一次熔断、第二次过了。单次结果 = **2/6 或 3/6 都能自圆其说** ——
    这种数字拿去写简历，面试官让你现场复现就露馅了。

    所以要报就报分布：

    - `pass_at_k`：k 次里至少成功 1 次 —— "有能力做到"的证据；
    - `pass_all_k`：k 次全成功 —— "稳定做到"的证据；
    - 两者差距大，说明的是**稳定性问题**，不是能力问题。
      小模型（本项目用的是 7B 级的 Qwen）在"该不该放弃"这种判断上本来就不稳。
    """
    if repeat <= 1:
        return []
    by_id: dict[str, list[dict]] = {}
    for r in rows:
        by_id.setdefault(r["id"], []).append(r)

    out = []
    for rid, group in by_id.items():
        scored = [g for g in group if g.get("scored", True)]
        if not scored:
            out.append({"id": rid, "kind": group[0]["kind"], "scored": False, "runs": len(group)})
            continue
        passes = sum(1 for g in scored if g["ok"])
        out.append(
            {
                "id": rid,
                "kind": scored[0]["kind"],
                "scored": True,
                "runs": len(scored),
                "passes": passes,
                "pass_at_k": passes > 0,
                "pass_all_k": passes == len(scored),
            }
        )
    return out


def summarize(rows: list[dict], skipped: list[dict] | None = None) -> dict:
    skipped = skipped or []
    # manual 任务（机器判不了的）不进分子也不进分母：混进去只会让数字变得没法解释。
    scored_rows = [r for r in rows if r.get("scored", True)]
    manual = [r for r in rows if not r.get("scored", True)]
    total = len(scored_rows)
    passed = sum(1 for r in scored_rows if r["ok"])
    normal = [r for r in scored_rows if r["kind"] == "normal"]
    counter = [r for r in scored_rows if r["kind"] == "counter"]

    def rate(items):
        if not items:
            return None
        return round(sum(1 for r in items if r["ok"]) / len(items), 4)

    def avg(items, key):
        if not items:
            return 0
        return round(sum(r[key] for r in items) / len(items), 2)

    return {
        "total": total,
        "passed": passed,
        "success_rate": round(passed / total, 4) if total else 0,
        "normal_total": len(normal),
        "normal_rate": rate(normal),
        "counter_total": len(counter),
        "counter_rate": rate(counter),
        "manual_total": len(manual),
        # skipped 是环境不可达、**没跑**的任务。单独记，
        # 不混进分母：否则网络差一次成功率就掉一截，看着像代码退化了。
        "skipped_total": len(skipped),
        "declared_total": total + len(manual) + len(skipped),
        "avg_steps": avg(scored_rows, "steps"),
        "avg_tokens": avg(scored_rows, "total_tokens"),
        "avg_cost_yuan": round(avg(scored_rows, "cost_yuan"), 6),
        "avg_elapsed": avg(scored_rows, "elapsed"),
    }


def _ground_label(row: dict) -> str:
    """把一次运行的证据核对结果渲染成一个短标签。

    三种状态必须能一眼分开，否则"未核对"会冒充"已核对"：

    - `已核对`：结论通过了逐字证据核对；
    - `退回N次`：核对没过、退回重做了 N 次（哪怕最后答案碰巧对了，
      也说明它是**被逼着**才回去读页面的）；
    - `免检/未核对`：拒答类结论或页面无正文 —— 是"核不了"，不是"核过了"。
    """
    if row.get("grounded"):
        return "[green]已核对[/green]"
    n = int(row.get("grounding_retries") or 0)
    if n:
        return f"[yellow]退回{n}次[/yellow]"
    return "[dim]免检[/dim]"


def print_report(
    rows: list[dict],
    stats: dict,
    skipped: list[dict] | None = None,
    repeat: int = 1,
) -> None:
    skipped = skipped or []
    scored = [r for r in rows if r.get("scored", True)]
    manual = [r for r in rows if not r.get("scored", True)]

    if skipped:
        sk = Table(show_header=True, header_style="bold yellow", title="环境预检未通过（未计分）")
        for col in ("ID", "URL", "非浏览器通道", "原因", "判断"):
            sk.add_column(col)
        for s in skipped:
            if s.get("transport_ok"):
                tr = f"[yellow]能通（{s.get('transport', '')}）[/yellow]"
            else:
                tr = f"[dim]也不通（{s.get('transport', '-')}）[/dim]"
            sk.add_row(
                s["id"],
                s["url"][:44],
                tr,
                s["reason"][:44],
                s.get("diagnosis", "")[:30],
            )
        console.print(sk)

    table = Table(show_header=True, header_style="bold", title="逐次结果（自动判分）")
    for col in ("ID", "类型", "第几次", "结果", "证据", "步数", "token", "耗时(s)", "判定说明"):
        table.add_column(col)
    seen: dict[str, int] = {}
    for r in scored:
        seen[r["id"]] = seen.get(r["id"], 0) + 1
        table.add_row(
            r["id"],
            r["kind"],
            f"{seen[r['id']]}/{repeat}" if repeat > 1 else "-",
            "[green]通过[/green]" if r["ok"] else "[red]失败[/red]",
            _ground_label(r),
            str(r["steps"]),
            str(r["total_tokens"]),
            str(r["elapsed"]),
            r["reason"][:44],
        )
    console.print(table)

    if repeat > 1:
        per_task = repeat_stats(rows, repeat)
        rt = Table(show_header=True, header_style="bold cyan", title=f"稳定性（每个任务跑 {repeat} 次）")
        for col in ("ID", "类型", f"过/共", "pass@{}".format(repeat), "pass^{}".format(repeat)):
            rt.add_column(col)
        for p in per_task:
            if not p["scored"]:
                rt.add_row(p["id"], p["kind"], "需人工核对", "—", "—")
                continue
            rt.add_row(
                p["id"],
                p["kind"],
                f"{p['passes']}/{p['runs']}",
                "是" if p["pass_at_k"] else "否",
                "是" if p["pass_all_k"] else "否",
            )
        console.print(rt)

    if manual:
        mt = Table(show_header=True, header_style="bold magenta", title="只跑不判分（需人工核对）")
        for col in ("ID", "步数", "Agent 的答案"):
            mt.add_column(col)
        for r in manual:
            mt.add_row(r["id"], str(r["steps"]), (r["answer"] or "（空）")[:76])
        console.print(mt)
        console.print(
            "[dim]这些任务的正确答案依赖登录态/个性化/当日内容，没有可离线核对的真值来源。"
            "写死锚点会过期、写形状会放过错答，所以不自动判分、也不进分母。[/dim]"
        )

    s = Table(show_header=False, box=None, title="汇总", padding=(0, 1))
    parts = [f"{stats['passed']}/{stats['total']}"]
    if stats["skipped_total"]:
        parts.append(f"{stats['skipped_total']} 个环境预检未通过")
    if stats["manual_total"]:
        parts.append(f"{stats['manual_total']} 个需人工核对")
    s.add_row("本次计分任务", "、".join(parts) + "（后两类均不计分）")
    if repeat > 1:
        s.add_row(
            "按次成功率",
            f"{stats['passed']}/{stats['total']} = {stats['success_rate']:.1%}"
            f"（{repeat} 轮累计，**不能**读成单轮成功率）",
        )
        per_task = [p for p in repeat_stats(rows, repeat) if p["scored"]]
        if per_task:
            at_k = sum(1 for p in per_task if p["pass_at_k"]) / len(per_task)
            all_k = sum(1 for p in per_task if p["pass_all_k"]) / len(per_task)
            s.add_row("pass@%d（至少过一次）" % repeat, f"{at_k:.1%}")
            s.add_row("pass^%d（次次都过）" % repeat, f"{all_k:.1%}")
    else:
        s.add_row("端到端成功率", f"{stats['passed']}/{stats['total']} = {stats['success_rate']:.1%}")
    s.add_row(
        "  其中 normal",
        f"{stats['normal_rate']:.1%}" if stats["normal_rate"] is not None else "—",
    )
    s.add_row(
        "  其中 counter（反例拒答）",
        f"{stats['counter_rate']:.1%}" if stats["counter_rate"] is not None else "—",
    )
    s.add_row("平均步数", str(stats["avg_steps"]))
    s.add_row("平均 token", str(stats["avg_tokens"]))
    s.add_row("平均成本", f"¥{stats['avg_cost_yuan']:.6f}")
    s.add_row("平均耗时", f"{stats['avg_elapsed']} 秒")
    # 证据锚定的汇总：这一行回答的是"这些结论有多少条真的落在了页面上"，
    # 与成功率是**两个正交的维度**（可能答对但没证据、也可能有证据但答错）。
    if scored:
        g = sum(1 for r in scored if r.get("grounded"))
        retried = sum(1 for r in scored if int(r.get("grounding_retries") or 0))
        s.add_row("结论带页面证据", f"{g}/{len(scored)}（其中 {retried} 条被退回重做后才收尾）")
    console.print(s)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="运行评测集并输出指标")
    parser.add_argument("--task-file", default=str(PROJECT_ROOT / "tasks" / "examples.json"))
    parser.add_argument("--only", nargs="*", help="只跑指定 id")
    parser.add_argument("--repeat", type=int, default=1, help="每个任务重复几次")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--preflight",
        dest="preflight",
        action="store_true",
        default=None,
        help="跑之前先试连任务的 URL，连不上的标 skipped 不计分",
    )
    parser.add_argument(
        "--no-preflight",
        dest="preflight",
        action="store_false",
        help="跳过环境预检（本地 file:// 任务不需要）",
    )
    args = parser.parse_args(argv)

    settings = get_settings(refresh=True)
    settings.validate()
    # 打出来是为了**模型层消融**能对账：两份结果比之前先看清模型号，
    # 才不会把"换了模型"的差异误记成"改了代码"的效果。
    console.print(
        f"[dim]模型 {settings.llm_model} ｜ 引擎 {settings.engine} "
        f"｜ 任务集 {Path(args.task_file).name}[/dim]"
    )

    data = json.loads(Path(args.task_file).read_text(encoding="utf-8"))
    tasks = data.get("tasks", data)
    if args.only:
        tasks = [t for t in tasks if t["id"] in set(args.only)]
    if not tasks:
        console.print("[red]没有匹配的任务[/red]")
        return 2

    # 预检开关：命令行显式指定优先；否则由任务文件自己声明
    # （真实站点的任务集里写着 `_network_dependent: true`）。
    # 这样"要不要预检"跟着任务集走，而不是靠人记得加参数。
    do_preflight = args.preflight
    if do_preflight is None:
        do_preflight = bool(data.get("_network_dependent"))

    skipped: list[dict] = []
    if do_preflight:
        console.print("[bold cyan]环境预检：先确认这些站点在当前网络下连得上[/bold cyan]")
        reach = asyncio.run(preflight([t["url"] for t in tasks]))
        runnable = []
        for t in tasks:
            ok, reason = reach.get(t["url"], (True, ""))
            if ok:
                runnable.append(t)
            else:
                skipped.append({"id": t["id"], "url": t["url"], "reason": reason})
        if skipped:
            # 补一条非浏览器通道的结论：把「不可达」拆成
            # 「站点/网络问题」和「只有浏览器被拦」——两类处置完全不同。
            annotate_transport(skipped)
            console.print(
                f"[yellow]{len(skipped)}/{len(tasks)} 个任务因环境不可达被跳过，"
                "不计入失败（但也不会让成功率显得更好看）[/yellow]"
            )
            _only_browser = sum(1 for s in skipped if s.get("transport_ok"))
            if _only_browser:
                console.print(
                    f"[yellow]其中 {_only_browser} 个是**有非浏览器通道能通**、"
                    "只有 chromium 连不上 —— 站点是好的，是出口在区分客户端[/yellow]"
                )
        tasks = runnable

    runs: list[dict] = []
    for rep in range(args.repeat):
        if args.repeat > 1:
            console.print(f"\n[bold cyan]=== 第 {rep + 1} / {args.repeat} 轮 ===[/bold cyan]")
        for task in tasks:
            console.print(f"[dim]→ {task['id']} {task['task'][:56]}[/dim]")
            try:
                runs.append(asyncio.run(run_task(task, settings, args.max_steps)))
            except KeyboardInterrupt:
                console.print("[yellow]已中断[/yellow]")
                break

    if not runs:
        console.print("[red]没有任何任务实际跑起来[/red]")
        if skipped:
            print_report([], summarize([], skipped), skipped, repeat=args.repeat)
        return 1

    stats = summarize(runs, skipped)
    print_report(runs, stats, skipped, repeat=args.repeat)

    out = PROJECT_ROOT / "runs" / f"eval_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(
        json.dumps(
            {
                "stats": stats,
                "repeat": args.repeat,
                "per_task": repeat_stats(runs, args.repeat),
                "runs": runs,
                "skipped": skipped,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    console.print(f"\n明细已写入 [bold]{out}[/bold]")
    if stats["total"] == 0:
        console.print("[yellow]没有任何可自动判分的任务（全是环境跳过或需人工核对），无法给出成功率[/yellow]")
        return 1
    return 0 if stats["passed"] == stats["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

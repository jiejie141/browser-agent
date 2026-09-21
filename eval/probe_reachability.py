"""站点可达性对照探测：直连 vs 走代理，到底差多少？

## 为什么要单独有这么一个工具

`tasks/real_sites.json` 跑一轮要十几分钟，而"该不该配代理"是一个
**配置问题**，几秒就能用可达性对照回答掉。先花十几分钟跑完评测再发现
"代理根本没帮上忙"，是浪费。

这个工具就是 `docs/实验记录.md` 实验 5 里那张对照表的来源：

    12 个真实站点：直连 5/12，走 127.0.0.1:7890 也是 5/12。

## 它和 `run_eval.py` 的预检是什么关系

同一套判定（都走 `proxy_launch_kwargs`），只是：
- 预检是评测流程里的一道闸门，只输出"过/不过"；
- 这个工具是**对照实验**，把两种出口并排摆出来，回答"配代理值不值"。

## 用法

    # 只看直连
    python eval/probe_reachability.py

    # 直连 vs 走代理，并列出逐站差异
    python eval/probe_reachability.py --proxy http://127.0.0.1:7890

    # 看别的任务集
    python eval/probe_reachability.py --task-file tasks/demo_sites.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "eval"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import run_eval  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

console = Console()


class _EgressStub:
    """只带代理两个字段的 settings 替身（preflight 只读这两个）。"""

    def __init__(self, proxy: str, bypass: str = "") -> None:
        self.browser_proxy = proxy
        self.browser_proxy_bypass = bypass


async def probe(proxy: str, urls: list[str], timeout_ms: int, interval: float) -> dict:
    run_eval.get_settings = lambda **_kw: _EgressStub(proxy)  # type: ignore[assignment]
    run_eval.PREFLIGHT_TIMEOUT_MS = timeout_ms
    return await run_eval.preflight(urls, min_interval=interval)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="站点可达性对照：直连 vs 代理")
    ap.add_argument(
        "--task-file",
        default=str(PROJECT_ROOT / "tasks" / "real_sites.json"),
    )
    ap.add_argument(
        "--proxy",
        default="",
        help="代理地址，如 http://127.0.0.1:7890。给了就做直连/代理对照",
    )
    ap.add_argument("--timeout-ms", type=int, default=12000, help="单站超时（毫秒）")
    ap.add_argument("--interval", type=float, default=0.8, help="两站之间的最小间隔（秒）")
    args = ap.parse_args(argv)

    data = json.loads(Path(args.task_file).read_text(encoding="utf-8"))
    tasks = data.get("tasks", data)
    urls = [t["url"] for t in tasks if t.get("url")]
    id_by_url = {t["url"]: t["id"] for t in tasks if t.get("url")}

    configs: list[tuple[str, str]] = [("直连", "")]
    if args.proxy:
        configs.append((f"代理 {args.proxy}", args.proxy))

    results: dict[str, dict] = {}
    for label, proxy in configs:
        console.print(f"\n[bold cyan]===== {label} =====")
        res = asyncio.run(probe(proxy, urls, args.timeout_ms, args.interval))
        results[label] = res

    table = Table(show_header=True, header_style="bold", title="可达性对照")
    table.add_column("ID")
    table.add_column("URL", max_width=42)
    for label, _ in configs:
        table.add_column(label, max_width=22)
    for u in urls:
        cells = []
        for label, _ in configs:
            ok, reason = results[label].get(u, (False, "?"))
            cells.append("[green]可达[/green]" if ok else f"[red]不可达[/red] [dim]{reason[:14]}[/dim]")
        table.add_row(id_by_url.get(u, "?"), u[:42], *cells)
    console.print(table)

    summary = Table(show_header=True, header_style="bold", title="汇总")
    summary.add_column("出口")
    summary.add_column("可达数")
    for label, _ in configs:
        n = sum(1 for u in urls if results[label].get(u, (False,))[0])
        summary.add_row(label, f"{n} / {len(urls)}")
    console.print(summary)

    if len(configs) > 1:
        diffs = [
            u
            for u in urls
            if results[configs[0][0]].get(u, (False,))[0]
            != results[configs[1][0]].get(u, (False,))[0]
        ]
        if diffs:
            console.print("[yellow]逐站差异：[/yellow]")
            for u in diffs:
                d = results[configs[0][0]].get(u, (False,))[0]
                p = results[configs[1][0]].get(u, (False,))[0]
                who = "只有代理能通" if p else "只有直连通"
                console.print(f"  {id_by_url.get(u, '?')} {u[:44]} — {who}")
        else:
            console.print(
                "[yellow]两种出口结论完全一致 —— 配代理没有改善，"
                "被拦的不是外网可达性。[/yellow]"
            )

    out = PROJECT_ROOT / "runs" / f"reachability_{datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(
        json.dumps(
            {label: {u: list(v) for u, v in r.items()} for label, r in results.items()},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    console.print(f"\n明细已写入 [bold]{out}[/bold]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

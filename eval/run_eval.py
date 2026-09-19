"""评测脚本：把"感觉变好了"变成"数据变好了"。

用法：
    python eval/run_eval.py                          # 跑全部任务
    python eval/run_eval.py --only t01 t03           # 只跑指定任务
    python eval/run_eval.py --repeat 3               # 每个任务跑 3 次（看稳定性）

输出：
    - 终端一张汇总表
    - runs/eval_<时间戳>.json 完整明细，可用于版本对比

为什么要版本化：改了提示词之后，你需要能回答
"这次改动让哪几个任务从失败变成功、又让哪几个变差了"。
没有明细文件，只记得住总数，定位不到具体是哪一条。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bagent.agent import ReActAgent, new_run_dir  # noqa: E402
from bagent.browser import open_browser  # noqa: E402
from bagent.config import get_settings  # noqa: E402
from bagent.llm import build_client  # noqa: E402

console = Console()

# 反例任务里，出现这些词就认为 Agent 正确表达了"做不到"
REFUSAL_MARKERS = (
    "无法", "不能", "没有", "不存在", "做不到", "不具备", "不支持",
    "找不到", "不可用", "没有找到", "无法完成", "需要登录", "无登录",
)


def judge(task: dict, result) -> tuple[bool, str]:
    """判定一次运行是否算成功。

    注意：counter 任务的判定是**启发式**的——靠回答里的措辞判断它有没有
    正确放弃。这不是完美的判分方式，但比人工看强，且足够用于版本对比。
    想更严谨需要引入人工标注，这一点我在简历里不会含糊其辞。
    """
    kind = task.get("kind", "normal")

    if kind == "counter":
        if not result.finished:
            # 没完成但也没崩，属于合理放弃
            return True, "未完成（合理放弃）"
        answer = result.answer or ""
        if any(m in answer for m in REFUSAL_MARKERS):
            return True, "主动说明无法完成"
        return False, f"对做不到的任务给出了结论: {answer[:40]}"

    # normal 任务
    if not result.finished:
        return False, result.error or "未完成"
    answer = result.answer or ""
    needed = task.get("must_contain") or []
    if not needed:
        return (True, "已完成") if answer else (False, "完成但结论为空")
    hit = [k for k in needed if k.lower() in answer.lower()]
    if hit:
        return True, f"命中关键词 {hit}"
    return False, f"结论未包含预期关键词 {needed}，实际: {answer[:60]}"


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
    }


def summarize(rows: list[dict]) -> dict:
    total = len(rows)
    passed = sum(1 for r in rows if r["ok"])
    normal = [r for r in rows if r["kind"] == "normal"]
    counter = [r for r in rows if r["kind"] == "counter"]

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
        "avg_steps": avg(rows, "steps"),
        "avg_tokens": avg(rows, "total_tokens"),
        "avg_cost_yuan": round(avg(rows, "cost_yuan"), 6),
        "avg_elapsed": avg(rows, "elapsed"),
    }


def print_report(rows: list[dict], stats: dict) -> None:
    table = Table(show_header=True, header_style="bold", title="逐任务结果")
    for col in ("ID", "类型", "结果", "步数", "token", "耗时(s)", "判定说明"):
        table.add_column(col)
    for r in rows:
        table.add_row(
            r["id"],
            r["kind"],
            "[green]通过[/green]" if r["ok"] else "[red]失败[/red]",
            str(r["steps"]),
            str(r["total_tokens"]),
            str(r["elapsed"]),
            r["reason"][:48],
        )
    console.print(table)

    s = Table(show_header=False, box=None, title="汇总", padding=(0, 1))
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
    console.print(s)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="运行评测集并输出指标")
    parser.add_argument("--task-file", default=str(PROJECT_ROOT / "tasks" / "examples.json"))
    parser.add_argument("--only", nargs="*", help="只跑指定 id")
    parser.add_argument("--repeat", type=int, default=1, help="每个任务重复几次")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args(argv)

    settings = get_settings(refresh=True)
    settings.validate()

    data = json.loads(Path(args.task_file).read_text(encoding="utf-8"))
    tasks = data.get("tasks", data)
    if args.only:
        tasks = [t for t in tasks if t["id"] in set(args.only)]
    if not tasks:
        console.print("[red]没有匹配的任务[/red]")
        return 2

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
        return 1

    stats = summarize(runs)
    print_report(runs, stats)

    out = PROJECT_ROOT / "runs" / f"eval_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(
        json.dumps({"stats": stats, "runs": runs}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    console.print(f"\n明细已写入 [bold]{out}[/bold]")
    return 0 if stats["passed"] == stats["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

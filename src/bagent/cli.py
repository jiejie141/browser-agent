"""命令行入口。

用法：
    python main.py --task "在当当网搜索《三体》并读出第一本书的价格" --url https://search.dangdang.com/
    python main.py --task-file tasks/examples.json --only 1
    python main.py --doctor          # 只做环境自检，不跑任务
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .agent import new_run_dir
from .browser import open_browser
from .config import get_settings
from .graph_agent import build_agent
from .llm import LLMError, MockLLMClient, build_client
from .models import Action

console = Console()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


async def _confirm_interactive(prompt: str) -> bool:
    """敏感操作的交互式确认。

    在无头/非交互环境下直接拒绝，绝不"默认同意"——
    默认同意等于护栏失效。
    """
    if not sys.stdin or not sys.stdin.isatty():
        console.print(f"[yellow]⚠ 非交互环境，已自动拒绝敏感操作: {prompt}[/yellow]")
        return False
    console.print(Panel(prompt, title="[red]需要你确认[/red]", border_style="red"))
    try:
        ans = await asyncio.to_thread(input, "确认执行吗？输入 y 继续，其他任意键跳过: ")
    except (EOFError, KeyboardInterrupt):
        return False
    return ans.strip().lower() in ("y", "yes")


def _print_step(step: int, action: Action | None, ok: bool, message: str) -> None:
    """on_step 回调，把每一步打到终端。

    action 可能是 None —— 模型输出不是合法 JSON 时就是这样回调的
    （见 agent.py / graph_agent.py：没有合法 Action 可传，用 None 表示
    "这一步没有动作"）。StepCallback 的类型签名也正是 Action | None，
    所以这里必须先判 None：否则一次格式抖动就会让 CLI 自己崩掉，
    真正的原因（模型输出不合法）被压在 AttributeError 堆栈下面，
    在 CI 里只表现为"这一步红了"。
    """
    icon = "[green]✓[/green]" if ok else "[red]✗[/red]"
    # 措辞与 StepRecord.raw_action 保持一致（那边同样用 "(格式错误)"），
    # 这样终端、trace.json、API 三处对同一步的称呼是一样的。
    label = action.action if action is not None else "(格式错误)"
    console.print(f"  {icon} [bold]第 {step} 步[/bold] {label}", end="")
    if action is not None and action.ref is not None:
        console.print(f" [{action.ref}]", end="")
    console.print()
    if action is not None and action.thought:
        console.print(f"    [dim]想法: {action.thought}[/dim]")
    console.print(f"    {message}")


def _print_summary(result) -> None:
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_row("任务", result.task[:70])
    table.add_row("是否完成", "[green]是[/green]" if result.success else "[red]否[/red]")
    if result.answer:
        table.add_row("结论", result.answer)
    if result.error:
        table.add_row("中止原因", f"[yellow]{result.error}[/yellow]")
    table.add_row("步数", str(result.steps))
    table.add_row("耗时", f"{result.elapsed_seconds} 秒")
    table.add_row(
        "token",
        f"输入 {result.usage.prompt_tokens} / 输出 {result.usage.completion_tokens} / "
        f"共 {result.usage.total_tokens}",
    )
    table.add_row("模型调用", f"{result.usage.calls} 次")
    table.add_row("估算成本", f"¥{result.cost_yuan:.6f}")
    table.add_row("产物目录", result.run_dir)
    console.print(Panel(table, title="[bold]本次运行[/bold]", border_style="blue"))


async def run_one(task: str, url: str, max_steps: int | None, settings, mock: bool = False) -> int:
    run_dir = new_run_dir(settings, tag="task")
    engine = (getattr(settings, "engine", "handwritten") or "handwritten").lower()
    title = "[bold]%s[/bold]" % task
    title += f"\n[dim]引擎: {engine}[/dim]"
    if mock:
        title += "\n[yellow]MOCK 模式：不调用真实模型，按剧本执行[/yellow]"
    console.print(Panel(title + f"\n[dim]起始页: {url}[/dim]", border_style="cyan"))

    llm = MockLLMClient() if mock else build_client(settings)
    agent = build_agent(settings, llm=llm, on_step=_print_step)

    async with open_browser(settings, run_dir) as session:
        result = await agent.run(
            task=task,
            start_url=url,
            session=session,
            run_dir=run_dir,
            confirm=_confirm_interactive,
            max_steps=max_steps,
        )

    _print_summary(result)
    return 0 if result.success else 1


def _probe_browser() -> None:
    """起一个无头 Chromium 再关掉，确认"浏览器"这一环是通的。

    单独抽成函数是为了可测：真起浏览器要一两秒，
    放进单元测试会让套件变慢；测试里 monkeypatch 掉它即可。
    """
    from playwright.async_api import async_playwright

    async def _probe():
        pw = await async_playwright().start()
        b = await pw.chromium.launch(headless=True)
        await b.close()
        await pw.stop()

    asyncio.run(_probe())


def doctor(settings, mock: bool = False) -> int:
    """环境自检：跑任务之前先确认每一环都通，省得中途炸掉。

    `mock` 是命令行开关 --mock 的取值。**必须由调用方传进来**，
    不能只读环境变量 —— CI 里跑的是 `main.py --doctor --mock`，
    没有任何环境变量，只看 MOCK 会把离线自检误判成在线模式。
    """
    console.print(Panel("环境自检", border_style="blue"))
    checks: list[tuple[str, bool, str]] = []

    # 离线判定有两个入口：环境变量 MOCK 和命令行开关 --mock。
    # 两边都要认：容器镜像里靠 ENV MOCK=true，CI 里靠 --mock 参数。
    mock = mock or os.getenv("MOCK", "").strip().lower() in ("1", "true", "yes", "on")
    offline = os.getenv("OFFLINE", "").strip().lower() in ("1", "true", "yes", "on")

    checks.append(("Python", True, sys.version.split()[0]))
    checks.append(("工作目录", True, str(Path.cwd())))
    if mock:
        # MOCK 模式走剧本替身，本来就不需要 key：
        # 「有没有 key」不是问题，「是否真的在联网」才是要确认的事。
        checks.append(("运行模式", True, "MOCK 离线替身，不调用真实模型"))
    else:
        checks.append(("配置读取", bool(settings.llm_api_key), settings.llm_model))
    engine = (getattr(settings, "engine", "handwritten") or "handwritten").lower()
    checks.append(("Agent 引擎", True, engine))
    if engine == "langgraph":
        try:
            import langgraph  # noqa: F401

            checks.append(("LangGraph", True, "已安装"))
        except ImportError:
            checks.append(
                ("LangGraph", False, "未安装：pip install langgraph（或改用 --engine handwritten）")
            )
    # 视觉通道是**可选**的降级路径：没配 key 时自动退回纯 DOM，
    # 这是设计行为，不是故障。以前这里返回 settings.vlm_enabled，
    # 会让「按设计关闭」在自检里显示为「失败」并让 doctor 退出码变 1，
    # 容器（OFFLINE/MOCK）里必现。改为：配了才算一项检查，没配就如实说明。
    if settings.vlm_enabled:
        checks.append(("视觉通道", True, "已启用"))
    else:
        checks.append(("视觉通道", True, "未配置，按设计只走 DOM 通道"))

    # 模型连通性：MOCK / OFFLINE 下一律跳过，避免在无网或不该出网的环境里试探
    if mock or offline:
        checks.append(("模型连通", True, "MOCK/OFFLINE 已跳过（不发起网络请求）"))
    else:
        try:
            client = build_client(settings)
            reply = client.chat(
                [{"role": "user", "content": "只回复两个字：正常"}], max_tokens=16
            )
            checks.append(("模型连通", True, f"返回: {reply[:20]}"))
        except LLMError as exc:
            checks.append(("模型连通", False, str(exc)[:160]))
        except Exception as exc:
            # build_client() 在缺 key 时抛的是 openai.OpenAIError，**不是 LLMError**
            # （校验在 OpenAI 客户端构造里就发生了，还没轮到我们的错误类型）。
            # 只捕 LLMError 会让自检自己崩掉 —— 那比"某项检查失败"更糟：
            # 堆栈和退出码会把真正的原因盖住，CI 里只看到"这一步红了"。
            checks.append(("模型连通", False, f"{type(exc).__name__}: {str(exc)[:120]}"))

    # 浏览器可用性（探针见 _probe_browser；抽成函数是为了测试里能换掉它）
    try:
        _probe_browser()
        checks.append(("Chromium", True, "可启动"))
    except Exception as exc:
        checks.append(("Chromium", False, f"{type(exc).__name__}: {str(exc)[:120]}"))

    table = Table(show_header=True, header_style="bold")
    table.add_column("检查项")
    table.add_column("结果")
    table.add_column("说明")
    all_ok = True
    for name, ok, detail in checks:
        all_ok = all_ok and ok
        table.add_row(name, "[green]通过[/green]" if ok else "[red]失败[/red]", detail)
    console.print(table)
    return 0 if all_ok else 1


def load_task_file(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("tasks", [])
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="browser-agent", description="ReAct 驱动的浏览器任务自动化 Agent"
    )
    parser.add_argument("--task", help="任务描述")
    parser.add_argument("--url", help="起始网址")
    parser.add_argument("--task-file", help="任务集 JSON 文件")
    parser.add_argument("--only", type=int, help="只跑任务集里第 N 个任务（从 1 开始）")
    parser.add_argument("--max-steps", type=int, help="覆盖最大步数")
    parser.add_argument("--doctor", action="store_true", help="只做环境自检")
    parser.add_argument("--headful", action="store_true", help="显示浏览器窗口")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="离线模式：用剧本替身代替真实模型，不需要 key，用于验证主循环",
    )
    parser.add_argument(
        "--engine",
        choices=["handwritten", "langgraph"],
        help="Agent 引擎：handwritten（默认，零额外依赖的手写 ReAct 循环）"
        " / langgraph（LangGraph StateGraph 实现）",
    )
    args = parser.parse_args(argv)

    settings = get_settings(refresh=True)
    if args.headful:
        settings.headless = False
    if args.engine:
        settings.engine = args.engine
    if args.mock:
        # --mock 是命令行开关，而 Settings 只认 MOCK 环境变量，两边必须合流。
        # 不合流的后果不是"少个提示"，而是：agent.run() 用 settings.mock 判定
        # 是否离线（离线时成本必须报 0），于是离线跑一次仍会拿估算 token 乘单价，
        # 报出一个并不存在的花费（实测 ¥0.00347）—— 使用者会以为真花了钱。
        settings.mock = True
    setup_logging(settings.log_level)

    if args.doctor:
        return doctor(settings, mock=args.mock)

    if not args.mock:
        try:
            settings.validate()
        except RuntimeError as exc:
            console.print(f"[red]{exc}[/red]")
            return 2

    async def _main() -> int:
        if args.task and args.url:
            return await run_one(
                args.task, args.url, args.max_steps, settings, mock=args.mock
            )

        if args.task_file:
            tasks = load_task_file(Path(args.task_file))
            if args.only:
                tasks = [tasks[args.only - 1]]
            if not tasks:
                console.print("[red]任务文件里没有任务[/red]")
                return 2
            failures = 0
            for item in tasks:
                code = await run_one(
                    item["task"], item["url"], args.max_steps, settings,
                    mock=args.mock,
                )
                failures += 0 if code == 0 else 1
            console.print(f"\n[bold]共 {len(tasks)} 个任务，失败 {failures} 个[/bold]")
            return 0 if failures == 0 else 1

        parser.print_help()
        return 2

    return asyncio.run(_main())


if __name__ == "__main__":
    raise SystemExit(main())

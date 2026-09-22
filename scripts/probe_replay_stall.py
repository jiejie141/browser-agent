"""轨迹重放探针：**不调模型**，把一次真实运行的动作序列在真浏览器里重放一遍，
逐步打印页面指纹与停滞计数。

## 为什么需要它

停滞检测（`page_fingerprint` / `update_stall`）的判定依据是"页面变没变"，
而 `trace.json` 里**只存了 record（动作 + 结果），没有存页面正文** ——
想知道某次运行"当时页面到底变没变"，只有两条路：

  1. 改 trace 格式把正文也存进去（体积会大很多，且对已有轨迹无效）；
  2. **把动作重放一遍**，现场量。

这个脚本走第 2 条。它的价值在于把"我猜是交替导航所以指纹一直在变"
这种辩解换成**可以看的数字**：重放之后每一步的 fingerprint / stall 都打在屏幕上，
停滞到底该不该触发、在哪一步触发，一目了然。

⚠️ 重放**不是**复现。模型没参与，也就没有"警告改变模型行为"这回事 ——
原运行如果被警告推着换了策略，重放里不会有那一步。
它只回答一件事：**在真实页面上，这串动作会不会被判定成停滞。**

用法：
    python scripts/probe_replay_stall.py runs/20260921-224847-t06
    python scripts/probe_replay_stall.py runs/<某次运行> --start-url https://...
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bagent.agent import (  # noqa: E402
    STALL_STOP_AT,
    STALL_WARN_AT,
    page_fingerprint,
    update_stall,
)
from bagent.browser import open_browser  # noqa: E402
from bagent.config import get_settings  # noqa: E402
from bagent.models import Action, PageState  # noqa: E402
from bagent.perception import perceive  # noqa: E402


def _load(run_dir: Path) -> tuple[dict, list[Action | None]]:
    trace = json.loads((run_dir / "trace.json").read_text(encoding="utf-8"))
    actions: list[Action | None] = []
    for rec in trace.get("records", []):
        raw = rec.get("action")
        # action 为 None = 那一步模型吐了非法 JSON，原引擎直接 continue，不执行任何东西。
        actions.append(Action(**raw) if raw else None)
    return trace, actions


async def replay(run_dir: Path, start_url: str | None) -> None:
    trace, actions = _load(run_dir)
    start_url = start_url or trace.get("start_url") or ""
    if not start_url:
        print("[跳过] 轨迹里没有 start_url，请用 --start-url 指定")
        return

    settings = get_settings(refresh=True)
    async with open_browser(settings, run_dir / "_replay") as session:
        await session.execute(Action(action="goto", url=start_url), confirm=None)

        print(f"\n{'=' * 96}")
        print(f"重放 {run_dir.name}  start_url={start_url}")
        print(f"原运行: steps={trace.get('steps')} finished={trace.get('finished')} "
              f"error={trace.get('error')!r}")
        print(f"阈值: 警告 {STALL_WARN_AT} / 收束 {STALL_STOP_AT}")
        print("=" * 96)
        print(f"{'步':>3} {'动作':<14} {'指纹':<12} {'停滞':>4}  事件")
        print("-" * 96)

        prev_fp = ""
        stall = 0
        last_action: Action | None = None

        for step in range(1, len(actions) + 1):
            state: PageState = await perceive(
                session.page, settings, step=step, run_dir=run_dir / "_replay"
            )
            cur_fp = page_fingerprint(state)
            stall = update_stall(
                prev_fp, cur_fp, last_action.action if last_action else None, stall
            )
            prev_fp = cur_fp

            events = []
            if stall >= STALL_STOP_AT:
                events.append("→ 收束（逼 finish）")
            elif stall >= STALL_WARN_AT:
                events.append("→ 警告")

            act = actions[step - 1]
            label = "-" if act is None else (
                f"{act.action}#{act.ref}" if act.ref is not None
                else (f"{act.action} {act.dy}" if act.dy is not None else act.action)
            )
            print(f"{step:>3} {label:<14} {cur_fp:<12} {stall:>4}  {' '.join(events)}")

            if act is None:
                # 原运行那一步模型吐了非法 JSON —— 引擎**不执行任何动作**，
                # 但循环继续（它只累计 consecutive_failures）。重放也必须继续，
                # 否则后面十几步全看不到，就像刚才那样在第 8 步断掉。
                print(f"{'':>3} {'':<14} {'':<12} {'':>4}  （非法 JSON，原运行继续）")
                continue
            if act.action == "finish":
                print(f"{'':>3} {'':<14} {'':<12} {'':>4}  （原运行在此 finish，重放结束）")
                break
            await session.execute(act, confirm=None)
            last_action = act

        print("-" * 96)
        print(f"重放结束：最终停滞计数 = {stall}")


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="包含 trace.json 的运行目录")
    parser.add_argument("--start-url", default=None)
    args = parser.parse_args()
    await replay(Path(args.run_dir), args.start_url)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

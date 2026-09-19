"""浏览器层冒烟测试：不调用任何大模型。

验证三件事：
  1. Chromium 能起来
  2. 感知层能从真实页面抽出带编号的可交互元素
  3. 动作层能用编号精确定位并点击

为什么单独做这个测试：Agent 出问题时，故障可能出在模型、感知、
动作三条链路的任意一条。先把模型摘出去，能跑通说明后两条是好的，
排查范围立刻缩小一半。

用法：
    python scripts/smoke_browser.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bagent.agent import new_run_dir  # noqa: E402
from bagent.browser import open_browser  # noqa: E402
from bagent.config import get_settings  # noqa: E402
from bagent.models import Action  # noqa: E402
from bagent.perception import perceive  # noqa: E402


async def main() -> int:
    settings = get_settings(refresh=True)
    run_dir = new_run_dir(settings, tag="smoke")
    print(f"产物目录: {run_dir}\n")

    async with open_browser(settings, run_dir) as session:
        page = session.page
        assert page is not None

        # ---- 第 1 步：打开页面 ----
        out = await session.execute(Action(action="goto", url="https://quotes.toscrape.com/"))
        print(f"[1] goto  -> ok={out.ok}  {out.message}")
        if not out.ok:
            return 1

        # ---- 第 2 步：感知 ----
        state = await perceive(page, settings, step=1, run_dir=run_dir)
        print(f"[2] 感知  -> 标题='{state.title}'  可交互元素 {len(state.elements)} 个")
        for el in state.elements[:12]:
            label = el.text or el.aria or el.placeholder or "(无文字)"
            print(f"        [{el.ref}] <{el.tag}> {label}")
        if not state.elements:
            print("        ❌ 一个元素都没抽到，感知层的 JS 需要检查")
            return 1

        # ---- 第 3 步：用编号点击"Next" ----
        target = next(
            (el for el in state.elements if "next" in (el.text or "").lower()),
            None,
        )
        if target is None:
            print("[3] ❌ 没找到 Next 链接，无法验证点击")
            return 1
        print(f"[3] 目标  -> [{target.ref}] 「{target.text}」")

        out = await session.execute(Action(action="click", ref=target.ref))
        print(f"    点击  -> ok={out.ok}  {out.message}")
        if not out.ok:
            return 1

        # ---- 第 4 步：验证页面确实变了 ----
        state2 = await perceive(page, settings, step=2, run_dir=run_dir)
        print(f"\n[4] 点击后 -> URL={state2.url}")
        if "/page/2" not in state2.url:
            print("        ❌ 页面没有翻到第 2 页，点击可能没生效")
            return 1
        print("        ✓ 已翻到第 2 页")

        first_line = next(
            (ln.strip() for ln in state2.body_text.splitlines() if ln.strip()), ""
        )
        print(f"        页首内容: {first_line[:80]}")

        shots = list((run_dir / "shots").glob("*.png"))
        print(f"\n截图 {len(shots)} 张: {[s.name for s in shots]}")

    print("\n✅ 浏览器层 + 感知层 + 动作层 全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

"""页面指纹探针：**不调模型**，只回答一个问题 ——

    "反复滚动到底会不会改变页面指纹？"

为什么非要单独测这个：我打算用"页面指纹连续不变"来判定 Agent 在原地打转
（动作各不相同、但页面一字未变 —— 7B 抓不住的那类失效）。这个判据能不能
成立，**完全取决于指纹对滚动有多敏感**：

  - 若滚动就改指纹 → 停滞检测对"滚动打转"完全瞎，机制白加，还可能
    在别处误伤（正常翻页被算成停滞）；
  - 若滚动不改指纹 → 说明"滚动确实没带来新信息"，判据成立。

而感知层的元素采集**带视口过滤**（`r.bottom < -800` 才丢弃，见
perception.py 的 `visible`），所以直觉上滚动是会改元素清单的 ——
但"直觉"在这个项目里已经错过好几次了（离线复算那次的教训）。
所以这里用真浏览器按真实动作序列量一遍，用数据说话。

用法：
    python scripts/probe_page_fp.py
    python scripts/probe_page_fp.py --url https://books.toscrape.com/
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bagent.agent import page_fingerprint  # noqa: E402
from bagent.browser import open_browser  # noqa: E402
from bagent.config import get_settings  # noqa: E402
from bagent.models import Action  # noqa: E402
from bagent.perception import perceive  # noqa: E402

# 真实动作序列。**故意模仿**强模型在 t06 上的行为：
# 连续下滑、再连续上滑 —— 如果这种序列下指纹一直不变，
# 说明"滚动 = 无新信息"这个前提对这两个站点成立。
SCROLL_PLAN = [
    ("scroll", 500),
    ("scroll", 500),
    ("scroll", 500),
    ("scroll", 500),
    ("scroll", -500),
    ("scroll", -500),
    ("scroll", -500),
    ("scroll", -500),
]


async def probe(url: str) -> None:
    settings = get_settings(refresh=True)
    run_dir = PROJECT_ROOT / "runs" / "_probe_page_fp"
    run_dir.mkdir(parents=True, exist_ok=True)

    async with open_browser(settings, run_dir) as session:
        await session.execute(Action(action="goto", url=url), confirm=None)

        print(f"\n{'=' * 78}\n{url}\n{'=' * 78}")
        print(f"{'动作':<16} {'指纹':<12} {'元素数':>6} {'正文长':>7}  标题")
        print("-" * 78)

        async def snapshot(label: str) -> str:
            state = await perceive(session.page, settings, step=1, run_dir=run_dir)
            fp = page_fingerprint(state)
            print(
                f"{label:<16} {fp:<12} {len(state.elements):>6} "
                f"{len(state.body_text):>7}  {(state.title or '')[:28]}"
            )
            return fp

        fps = [await snapshot("goto")]
        for name, dy in SCROLL_PLAN:
            await session.execute(Action(action=name, dy=dy), confirm=None)
            fps.append(await snapshot(f"{name} {dy:+d}"))

        # 判定：滚动前后指纹是否全同
        first = fps[0]
        same = [i for i, f in enumerate(fps) if f == first]
        print("-" * 78)
        changed = sum(1 for f in fps[1:] if f != first)
        print(
            f"滚动后指纹改变 {changed}/{len(fps) - 1} 次；"
            f"与首帧相同的位置 {same}"
        )
        if changed == 0:
            print("→ 结论：**滚动不改变指纹**，停滞检测对滚动打转有效。")
        else:
            print(
                "→ 结论：**滚动会改变指纹**（多半是视口过滤让元素清单跟着动），"
                "停滞检测对滚动打转是瞎的，指纹里必须剔除视口相关成分。"
            )


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", action="append", default=None)
    args = parser.parse_args()
    urls = args.url or [
        "https://books.toscrape.com/",
        "https://quotes.toscrape.com/",
    ]
    for u in urls:
        try:
            await probe(u)
        except Exception as exc:  # noqa: BLE001
            print(f"[跳过] {u} — {type(exc).__name__}: {str(exc).splitlines()[0][:100]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

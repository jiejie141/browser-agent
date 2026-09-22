"""真机验证视觉通道：**拿真实截图 + 真实视觉模型**走一遍 perceive()。

## 为什么要这个脚本

README「能力边界」里长期写着一条：
"VLM 代码路径没被真实数据验证过（供应商没有视觉模型）"。
2026-09-22 复核发现那句结论是**错的** —— 当时只按模型名字筛（名字里
带 vision/vl 的 0 个），没真的发图试。逐个发图探测之后：

    21 个模型 → 10 个能收下图片 → 其中 3 个**真的读得懂**
    （让它看一张上红下蓝的图并回答上半部分颜色，答对的才算）

所以这条"未验证"必须被真的验证掉，而不是改一句措辞糊过去。

用法： python scripts/verify_vlm_live.py
"""
from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bagent.config import get_settings  # noqa: E402
from bagent.llm import build_vlm_client  # noqa: E402
from bagent.perception import perceive  # noqa: E402

# 一个 DOM 通道**抽不到**元素、只能靠看的页面：整页就是一块 Canvas。
# 用它来验证"降级到视觉通道"这条路径确实会被触发，而不是碰巧走 DOM。
_PAGE = """
<!doctype html><html><body style="margin:0">
<canvas id="c" width="1200" height="700"></canvas>
<script>
  const g = document.getElementById('c').getContext('2d');
  g.fillStyle = '#f5f5f5'; g.fillRect(0, 0, 1200, 700);
  g.fillStyle = '#1a73e8'; g.fillRect(80, 60, 260, 70);
  g.fillStyle = '#ffffff'; g.font = '34px sans-serif';
  g.fillText('搜索', 170, 108);
  g.fillStyle = '#202124'; g.font = '30px sans-serif';
  g.fillText('第一条结果：开源浏览器智能体项目', 80, 260);
  g.fillText('第二条结果：ReAct  Agent  实践笔记', 80, 330);
</script>
</body></html>
"""


async def main() -> int:
    st = get_settings()
    vlm = build_vlm_client(st)
    if vlm is None:
        print("❌ 没配 VLM_API_KEY / VLM_MODEL，视觉通道处于关闭状态")
        return 1

    print(f"视觉模型 = {st.vlm_model}  端点 = {st.vlm_base_url or st.llm_base_url}")

    from playwright.async_api import async_playwright

    pw = await async_playwright().start()
    br = await pw.chromium.launch(headless=True)
    try:
        page = await (await br.new_context(viewport={"width": 1280, "height": 800})).new_page()
        await page.set_content(_PAGE)
        await page.wait_for_timeout(500)

        run_dir = Path(tempfile.mkdtemp(prefix="vlm-verify-"))
        state = await perceive(
            page, st, step=1, run_dir=run_dir, prefer_vision=True, vlm=vlm
        )
    finally:
        await br.close()
        await pw.stop()

    print(f"DOM 通道抽到元素数 = {len(state.elements)}（Canvas 页面应为 0）")
    print(f"截图 = {state.screenshot_path}")
    if not state.screenshot_path:
        print("❌ 没截图，视觉通道根本没被触发")
        return 1

    if "[视觉通道补充]" not in (state.body_text or ""):
        print("❌ 正文里没有 [视觉通道补充] —— 视觉模型要么没被调用，要么返回了空")
        print(f"   正文 = {state.body_text[:200]!r}")
        return 1

    desc = state.body_text.split("[视觉通道补充]")[1].split("\n\n")[0]
    print("\n视觉模型对这张截图的描述：")
    print("  " + desc.replace("\n", "\n  "))
    print(f"\n视觉通道用量：{vlm.usage.calls} 次调用 / {vlm.usage.total_tokens} token")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

"""感知层：给 Agent 装上眼睛。

核心设计 —— **双通道感知**

  通道 A（默认）：把页面上的可交互元素抽成带编号的清单，喂给文本模型。
                  便宜、稳定、可精确点击（点编号而不是靠猜坐标）。
  通道 B（降级）：截图 + 视觉模型。当通道 A 一无所获（Canvas 页面、
                  iframe、纯图片按钮）时启用。

为什么必须做降级：真实网站里总有 DOM 抽不出东西的页面，
只留一条通道，Agent 到那种页面就直接卡死。
"""

from __future__ import annotations

import logging
from pathlib import Path

from playwright.async_api import Page

from .config import Settings
from .llm import LLMClient, LLMError
from .models import Element, PageState

log = logging.getLogger(__name__)

# 一帧最多收集多少个可交互元素。
# 这个数字直接决定 prompt 长度 → 决定每一跳的成本。调大要付钱。
MAX_ELEMENTS = 100

# 收集可交互元素的 JS。在浏览器上下文里执行。
_COLLECT_JS = """
() => {
  const MAX = %d;
  const out = [];

  // 清掉上一次的标记，避免 ref 编号错位到已经消失的元素上
  document.querySelectorAll('[data-ba-ref]').forEach(el => el.removeAttribute('data-ba-ref'));

  const selector = [
    'a', 'button', 'input', 'textarea', 'select',
    '[role="button"]', '[role="link"]', '[role="tab"]',
    '[role="menuitem"]', '[role="checkbox"]', '[role="switch"]',
    '[contenteditable="true"]', '[onclick]'
  ].join(',');

  const nodes = Array.from(document.querySelectorAll(selector));

  const visible = (el) => {
    const r = el.getBoundingClientRect();
    if (r.width <= 1 || r.height <= 1) return false;
    const st = window.getComputedStyle(el);
    if (st.display === 'none' || st.visibility === 'hidden') return false;
    if (parseFloat(st.opacity) === 0) return false;
    // 视口外很远的元素不收集，它们通常不是当前步骤的目标
    if (r.bottom < -800 || r.top > window.innerHeight + 3000) return false;
    return true;
  };

  const label = (el) => {
    const raw = el.innerText
      || el.value
      || el.getAttribute('placeholder')
      || el.getAttribute('aria-label')
      || el.getAttribute('title')
      || el.getAttribute('name')
      || '';
    return String(raw).trim().replace(/\\s+/g, ' ').slice(0, 60);
  };

  let ref = 0;
  for (const el of nodes) {
    if (out.length >= MAX) break;
    if (!visible(el)) continue;
    ref += 1;
    el.setAttribute('data-ba-ref', String(ref));
    out.push({
      ref: ref,
      tag: el.tagName.toLowerCase(),
      text: label(el),
      aria: String(el.getAttribute('aria-label') || '').slice(0, 60),
      placeholder: String(el.getAttribute('placeholder') || '').slice(0, 60)
    });
  }
  return out;
}
""" % MAX_ELEMENTS

_BODY_TEXT_JS = "() => (document.body ? document.body.innerText : '')"


async def extract_elements(page: Page) -> list[Element]:
    """通道 A：抽取可交互元素并打上编号。"""
    try:
        raw = await page.evaluate(_COLLECT_JS)
    except Exception as exc:  # 页面正在跳转时 evaluate 会失败，属于正常抖动
        log.warning("抽取元素失败（页面可能正在导航）: %s", exc)
        return []

    elements: list[Element] = []
    for item in raw or []:
        try:
            elements.append(Element(**item))
        except Exception:
            continue
    return elements


async def extract_body_text(page: Page, max_chars: int = 4000) -> str:
    try:
        text = await page.evaluate(_BODY_TEXT_JS) or ""
    except Exception:
        return ""
    text = str(text).strip()
    return text[:max_chars]


async def perceive(
    page: Page,
    settings: Settings,
    *,
    step: int,
    run_dir: Path,
    prefer_vision: bool = False,
    vlm: LLMClient | None = None,
) -> PageState:
    """采一帧页面状态。

    prefer_vision=True 时强制走视觉通道（例如上一步点击失败、
    怀疑目标元素没被 DOM 通道抓到）。
    """
    url, title = "", ""
    try:
        url = page.url
        title = await page.title()
    except Exception:
        pass

    elements = await extract_elements(page)

    # 判定是否必须降级到视觉通道
    need_vision = prefer_vision or len(elements) == 0

    body_text = await extract_body_text(page)

    state = PageState(
        step=step,
        url=url,
        title=title,
        elements=elements,
        body_text=body_text,
    )

    if need_vision:
        shot = run_dir / "shots" / f"step_{step:02d}.png"
        shot.parent.mkdir(parents=True, exist_ok=True)
        try:
            await page.screenshot(path=str(shot), full_page=False)
            state.screenshot_path = str(shot)
        except Exception as exc:
            log.warning("截图失败: %s", exc)

        # 有视觉模型就让它描述一下，把结果并入正文摘要
        if settings.vlm_enabled and vlm is not None and state.screenshot_path:
            hint = await _describe_with_vision(vlm, state.screenshot_path)
            if hint:
                state.body_text = (
                    "[视觉通道补充]" + hint + "\n\n" + state.body_text
                )

    return state


async def _describe_with_vision(vlm: LLMClient, image_path: str) -> str:
    """让视觉模型说清楚"现在屏幕上有什么、目标在哪"。"""
    prompt = (
        "这是一张网页截图。请用简体中文简明回答两点：\n"
        "1) 页面上有哪些可以点击或输入的区域（按从上到下、从左到右列出）；\n"
        "2) 如果能看到类似按钮/输入框/链接的元素，描述它们的文字内容。\n"
        "只描述你确实看到的内容，不要猜测。控制在 200 字以内。"
    )
    try:
        return await _run_sync(vlm.chat_vision, prompt, image_path)
    except LLMError as exc:
        log.warning("视觉通道调用失败，已忽略: %s", exc)
        return ""


async def _run_sync(fn, *args, **kwargs):
    """把同步的 SDK 调用丢到线程池，避免阻塞 asyncio 事件循环。

    这是新手最容易踩的坑：在 async 函数里直接调用同步的 requests，
    会把整个事件循环卡住，Playwright 的等待逻辑也跟着停摆。
    """
    import asyncio

    return await asyncio.to_thread(fn, *args, **kwargs)

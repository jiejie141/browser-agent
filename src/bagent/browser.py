"""浏览器动作层。

这一层只做一件事：把 Agent 决定的动作，变成真实的鼠标键盘操作。

关键技巧：**不按坐标点击，按编号点击。**
感知层给每个可交互元素打了 data-ba-ref 属性，这里直接用
`[data-ba-ref="3"]` 这个选择器定位。坐标会被弹窗、懒加载、
页面抖动搞乱，编号不会。
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Awaitable, Callable

from playwright.async_api import Browser, BrowserContext, Page, async_playwright

from .config import Settings
from .models import Action, StepOutcome

log = logging.getLogger(__name__)

# 敏感操作关键词。命中则要求人工确认，避免 Agent 真的把钱花出去。
SENSITIVE_PATTERNS = (
    "支付", "付款", "结算", "立即购买", "提交订单", "确认下单", "确认支付",
    "转账", "提现", "退款", "删除", "注销", "解绑", "退订", "永久",
    "pay", "checkout", "purchase", "delete", "unsubscribe", "confirm order",
)

# 确认回调：收到动作描述，返回 True 表示放行
ConfirmFn = Callable[[str], Awaitable[bool]]


class SensitiveActionBlocked(RuntimeError):
    """人工拒绝了敏感操作。"""


class BrowserSession:
    """管理一次浏览器会话的生命周期。

    用 async context manager，保证异常退出时浏览器也被正确关掉——
    否则跑十次就会留下十个僵尸 chromium 进程。
    """

    def __init__(self, settings: Settings, run_dir: Path) -> None:
        self.settings = settings
        self.run_dir = run_dir
        self._pw = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self.page: Page | None = None

    async def __aenter__(self) -> "BrowserSession":
        self._pw = await async_playwright().start()
        self._browser = await self._pw.chromium.launch(
            headless=self.settings.headless,
            args=["--disable-blink-features=AutomationControlled"],
        )
        self._context = await self._browser.new_context(
            viewport={"width": 1440, "height": 900},
            locale="zh-CN",
            timezone_id="Asia/Shanghai",
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
            ),
        )
        self._context.set_default_timeout(self.settings.step_timeout_seconds * 1000)
        self.page = await self._context.new_page()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        """收尾：关 page/context、关浏览器、停 Playwright。

        整个收尾**只给一个总时间预算**（settings.teardown_timeout_seconds），
        超了就直接放弃等待。这一条是被真实故障逼出来的，不是防御性编程：
        本机冒烟时任务 4.3 秒就把 5 步全跑完了，进程却在收尾上卡了 4 分钟不退出；
        走 API 时更糟 —— 终态原本写在收尾之后，于是任务永远停在 running，
        控制台一直转圈，而它其实早就成功了。

        收尾失败**不是**任务失败：该落的盘都落了，它只负责回收进程资源。
        所以这里超时只记一条警告，不往外抛。
        """
        try:
            await asyncio.wait_for(
                self._close_all(), timeout=self.settings.teardown_timeout_seconds
            )
        except asyncio.TimeoutError:
            log.warning(
                "浏览器收尾超过 %.1fs 仍未完成，已放弃等待（任务结果不受影响）",
                self.settings.teardown_timeout_seconds,
            )

    async def _close_all(self) -> None:
        # 顺序反着来：先关子级（context）再关父级（browser），最后停 Playwright 驱动。
        for closer in (self._context, self._browser):
            try:
                if closer:
                    await closer.close()
            except Exception:
                pass
        try:
            if self._pw:
                await self._pw.stop()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 动作分发
    # ------------------------------------------------------------------
    async def execute(
        self,
        action: Action,
        *,
        confirm: ConfirmFn | None = None,
    ) -> StepOutcome:
        """执行一个动作，返回结果。异常不往外抛，转成 ok=False。"""
        assert self.page is not None, "BrowserSession 尚未启动"

        try:
            if action.action == "goto":
                return await self._goto(action.url or "")
            if action.action == "click":
                return await self._click(action.ref, confirm)
            if action.action == "type":
                return await self._type(action.ref, action.text or "")
            if action.action == "press":
                return await self._press(action.key or "Enter")
            if action.action == "scroll":
                return await self._scroll(action.dy or 600)
            if action.action == "screenshot":
                return await self._shot()
            if action.action in ("extract", "finish"):
                # 这两个不需要浏览器的物理动作，由 Agent 层处理
                return StepOutcome(ok=True, message="无需浏览器操作")
            return StepOutcome(ok=False, message=f"未知动作: {action.action}")

        except Exception as exc:
            log.debug("动作执行异常", exc_info=True)
            return StepOutcome(
                ok=False,
                message=f"{type(exc).__name__}: {str(exc)[:200]}",
                # 点击/输入失败往往意味着元素没被 DOM 通道抓到，提示走视觉通道
                vision_hint=action.action in ("click", "type"),
            )

    # ------------------------------------------------------------------
    # 具体动作
    # ------------------------------------------------------------------
    async def _goto(self, url: str) -> StepOutcome:
        if not url:
            return StepOutcome(ok=False, message="goto 缺少 url 参数")
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        try:
            await self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
        except Exception as exc:
            return StepOutcome(ok=False, message=f"打开 {url} 失败: {str(exc)[:150]}")
        await self._settle()
        return StepOutcome(ok=True, message=f"已打开 {self.page.url}")

    async def _click(self, ref: int | None, confirm: ConfirmFn | None) -> StepOutcome:
        if ref is None:
            return StepOutcome(ok=False, message="click 缺少 ref 参数")

        locator = self.page.locator(f'[data-ba-ref="{ref}"]')
        if await locator.count() == 0:
            return StepOutcome(
                ok=False,
                message=f"编号 [{ref}] 不存在——页面可能已经刷新，请重新查看元素清单",
                vision_hint=True,
            )

        label = (await locator.first.inner_text() or "").strip()[:60]

        # 敏感操作护栏
        if _is_sensitive(label) and confirm is not None:
            approved = await confirm(f"即将点击可能产生实际后果的按钮: 「{label}」")
            if not approved:
                return StepOutcome(ok=False, message=f"人工拒绝了敏感操作「{label}」，已跳过")

        await locator.first.scroll_into_view_if_needed()
        await locator.first.click(timeout=self.settings.step_timeout_seconds * 1000)
        await self._settle()
        return StepOutcome(ok=True, message=f"已点击 [{ref}] 「{label}」")

    async def _type(self, ref: int | None, text: str) -> StepOutcome:
        if ref is None:
            return StepOutcome(ok=False, message="type 缺少 ref 参数")

        locator = self.page.locator(f'[data-ba-ref="{ref}"]')
        if await locator.count() == 0:
            return StepOutcome(
                ok=False,
                message=f"编号 [{ref}] 不存在，无法输入",
                vision_hint=True,
            )

        await locator.first.scroll_into_view_if_needed()
        # fill 会先清空再输入，比 type 更符合"填写表单"的语义
        try:
            await locator.first.fill(text)
        except Exception:
            # 部分富文本/自定义控件不支持 fill，退回到逐字输入
            await locator.first.click()
            await locator.first.type(text, delay=30)
        return StepOutcome(ok=True, message=f"已在 [{ref}] 输入「{text[:40]}」")

    async def _press(self, key: str) -> StepOutcome:
        await self.page.keyboard.press(key)
        await self._settle()
        return StepOutcome(ok=True, message=f"已按下 {key}")

    async def _scroll(self, dy: int) -> StepOutcome:
        await self.page.mouse.wheel(0, dy)
        await asyncio.sleep(0.6)
        return StepOutcome(ok=True, message=f"已滚动 {dy} 像素")

    async def _shot(self) -> StepOutcome:
        path = self.run_dir / "shots" / "manual.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        await self.page.screenshot(path=str(path), full_page=False)
        return StepOutcome(ok=True, message=f"已截图: {path}")

    async def _settle(self, timeout: float = 5.0) -> None:
        """动作后等页面稳定。

        真实网站不需要 sleep 5 秒，但等"网络基本停了"比固定 sleep 更准。
        超时不报错——有些页面永远有轮询请求，等不到 idle。
        """
        try:
            await self.page.wait_for_load_state("networkidle", timeout=timeout * 1000)
        except Exception:
            pass
        await asyncio.sleep(0.4)


def _is_sensitive(label: str) -> bool:
    low = (label or "").lower()
    return any(p in low for p in SENSITIVE_PATTERNS)


@asynccontextmanager
async def open_browser(settings: Settings, run_dir: Path):
    session = BrowserSession(settings, run_dir)
    async with session as s:
        yield s

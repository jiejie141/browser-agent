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


def proxy_launch_kwargs(settings: Settings) -> dict:
    """把 settings 里的代理配置翻译成 Playwright 的 launch 参数。

    抽成独立函数是**被一个真实缺陷逼出来的**，不是为复用而复用：

    `eval/run_eval.py` 的环境预检自己开了一个无头浏览器去试连 URL，但它
    只写了 `chromium.launch(headless=True)` —— **没带代理**。于是出现一种
    自相矛盾的现场：`BROWSER_PROXY` 配好了、主路径（真正的 Agent 浏览器）
    确实走了代理，可预检那一层还在用直连的口子；直连不通的站点被判成
    "环境不可达"，任务在开跑前就被标成 `skipped` 剔除，**根本没机会跑**。
    表现就是"代理开了也没用"，而且报告里看不出是被谁拦下的。

    所以代理的判定必须只有一处。这里返回的字典直接 `**` 进 launch()。

    留空 = 返回空字典 = 不加 `proxy` 键（不能写 `{"server": ""}`，
    空字符串会被 Playwright 当成非法代理地址直接抛错）。
    """
    out: dict = {}
    proxy_server = (getattr(settings, "browser_proxy", "") or "").strip()
    if not proxy_server:
        return out
    proxy_cfg: dict = {"server": proxy_server}
    bypass = (getattr(settings, "browser_proxy_bypass", "") or "").strip()
    if bypass:
        proxy_cfg["bypass"] = bypass
    out["proxy"] = proxy_cfg
    return out


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
        launch_kwargs: dict = {
            "headless": self.settings.headless,
            "args": ["--disable-blink-features=AutomationControlled"],
        }
        # 代理是可选的：留空就完全不加这个键，让浏览器用它的默认出口。
        # 具体规则（含"为什么不能写空 server"）都在 proxy_launch_kwargs 里，
        # 与 eval 侧的环境预检共用同一份判定，避免两处各写一遍再走偏。
        launch_kwargs.update(proxy_launch_kwargs(self.settings))
        if "proxy" in launch_kwargs:
            log.info(
                "浏览器出口走代理 %s（bypass=%s）",
                launch_kwargs["proxy"]["server"],
                launch_kwargs["proxy"].get("bypass", "-"),
            )
        self._browser = await self._pw.chromium.launch(**launch_kwargs)
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

        # 敏感操作护栏。
        #
        # ⚠️ 原写法是 `if _is_sensitive(label) and confirm is not None:` ——
        # 也就是**没有确认回调时整条护栏被跳过**，敏感按钮照点不误。
        # 而 API 服务（api.py）正是传 `confirm=None`（它本来想表达
        # "服务端无交互终端 → 一律拒绝"），结果是**服务端模式下护栏完全失效**：
        # 注释写着"一律拒绝"，代码做的是"一律放行"。
        #
        # 默认必须是最保守的那个：**没人能确认 = 不放行**。
        # 这跟 CLI 里那条"非交互环境自动拒绝"是同一个默认值（见 cli.py）。
        if _is_sensitive(label):
            approved = (
                await confirm(f"即将点击可能产生实际后果的按钮: 「{label}」")
                if confirm is not None
                else False
            )
            if not approved:
                return StepOutcome(
                    ok=False,
                    message=(
                        f"敏感操作「{label}」未获确认，已拦截"
                        f"（无交互终端时默认拒绝，这是安全默认值，不是故障）"
                    ),
                )

        await locator.first.scroll_into_view_if_needed()
        try:
            await locator.first.click(timeout=self.settings.step_timeout_seconds * 1000)
        except Exception as exc:
            # 把 Playwright 的原始报错翻译成**模型能据以改动作**的话。
            # 不翻译的后果是实测过的（见 translate_click_error 的说明）：
            # 模型只收到一句 "Timeout 30000ms exceeded"，它会以为"再点一次就行"。
            return StepOutcome(
                ok=False,
                message=translate_click_error(exc, ref),
                vision_hint=True,
            )
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


# 点击失败时，Playwright 原始报错里值得单独翻译的几种形态。
#
# ⚠️ 顺序有意义：**越具体的排在前面**。同一条报错里可能同时出现多个关键词
# （比如"等它可见且稳定"里既有 visible 也有 stable），先命中的那条才是对的。
#
# ## 为什么必须翻译（这是实测打出来的，不是"提示词优化"）
#
# 不翻译时，模型收到的是：
#
#     TimeoutError: Timeout 30000ms exceeded.
#     Call log:
#       - waiting for locator("[data-ba-ref=\\"37\\"]")
#       - locator resolved to <a class="news-title">…</a>
#       - attempting click action
#       - waiting for element to be visible, enabled and stable
#       - element is not stable
#
# 这里真正有用的信息只有最后一行 **"element is not stable"**（元素在轮播，
# 永远等不到稳定），而它被埋在 6 行堆栈中间。实测（README 第八节 8.11）：
# 必应首页上模型去点「热点资讯」里的新闻标题，耗满 30 秒超时，
# 收到的就是这一大坨 —— 于是它**再点一次**，再耗 30 秒。12 次点击里 4 次超时。
#
# 翻译的原则：**说清"为什么点不到"+"下一步该做什么"，并且禁止重试同一个元素**。
# 只说"超时"是不够的 —— 超时是现象，"在轮播"才是原因，而模型需要的是原因。
_CLICK_ERROR_HINTS: tuple[tuple[str, str], ...] = (
    (
        "not stable",
        "这个元素一直在动（轮播 / 动画 / 不断重排），等不到它停下来，"
        "**再点多少次都一样**。请换做法：改用 press(Enter)、点另一个入口，"
        "或直接用 goto 打开一个明确的网址。",
    ),
    (
        "intercepts pointer events",
        "这个元素被别的浮层挡住了，点不到。请先 scroll 让页面重新布局、"
        "或关掉遮挡物（弹窗 / 悬浮条），再考虑点它；不要重复点。",
    ),
    (
        "not visible",
        "这个元素当前不可见（折叠菜单 / 隐藏区域），点不了。"
        "请先展开它所在的区域（点父级菜单或先 scroll），或换一个可见的入口。",
    ),
    (
        "outside of the viewport",
        "这个元素在可视区域之外，点不了。请先 scroll 把它滚进视口再点，"
        "或直接用 goto。",
    ),
    (
        "not enabled",
        "这个元素当前是禁用状态（灰掉 / disabled），点了也不会有反应。请换别的做法。",
    ),
)


def translate_click_error(exc: Exception, ref: int | None = None) -> str:
    """把点击动作的原始异常翻译成模型能据以改动作的一句话。纯函数，可单测。

    兜底分支仍然要带上"别再点它"：模型最典型的反应就是原样重试，
    而那会再烧一个 30 秒超时。
    """
    raw = str(exc) or ""
    head = raw.strip().splitlines()[0] if raw.strip() else type(exc).__name__
    low = raw.lower()
    who = f"点击 [{ref}]" if ref is not None else "点击"
    for marker, hint in _CLICK_ERROR_HINTS:
        if marker in low:
            return f"{who} 失败：{hint}（原始报错首行: {head[:80]}）"
    return (
        f"{who} 超时：等不到可点击状态（{head[:100]}）。"
        f"请换一个入口（press / goto / 别的元素），**不要重复点这个**。"
    )


@asynccontextmanager
async def open_browser(settings: Settings, run_dir: Path):
    session = BrowserSession(settings, run_dir)
    async with session as s:
        yield s

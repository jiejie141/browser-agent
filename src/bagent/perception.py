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
import re
from pathlib import Path

from playwright.async_api import Page

from .config import Settings
from .llm import LLMClient, LLMError
from .models import Element, PageState

log = logging.getLogger(__name__)

# 一帧最多给多少个元素编号。
# 这个数字直接决定 prompt 长度 → 决定每一跳的成本。调大要付钱。
MAX_ELEMENTS = 100

# 扫描上限：先把候选都扫出来（**不编号**），再决定给谁编号。
# 必须比 MAX_ELEMENTS 大得多 —— 只扫 100 个的话侧栏早就把名额占满了，
# "筛选"就无从谈起。
SCAN_CAP = 500

# 导航 / 侧栏 / 页脚最多能占用多少个编号。
# 没有这条上限，就是本模块改造前的行为：侧栏分类链接吃光 100 个名额，
# 正文里的搜索结果一个都进不来 —— 表现是"页面上明明有结果，
# Agent 却说不出来"。25 是经验值：够放下搜索框、主导航和分页，
# 又挡得住"几百个分类链接"。
CHROME_QUOTA = 25

# 收集候选元素的 JS（第一趟，只扫不编号）。在浏览器上下文里执行。
#
# ⚠️ maxScan 必须由 Python 侧显式传进来，别指望"不传就是默认值"：
# Playwright 在调用方不给参数时传进来的是 **null 而不是 undefined**，
# 于是 `cands.length >= maxScan` 里的 `0 >= null` 为真，循环第一次就 break，
# 候选恒为空 —— 表现是"页面上明明有链接，却一个元素都抽不到"。
# 这里再加一层 typeof 判断，即使将来有人忘了传参，也只是退化成"不设上限"，
# 而不是静默返回空列表（静默返回空是最难查的一种失败）。
_COLLECT_JS = """
(maxScan) => {
  const cap = (typeof maxScan === 'number' && maxScan > 0) ? maxScan : Infinity;
  const CHROME_SEL = 'nav, header, footer, aside, [role="navigation"], [role="banner"], [role="contentinfo"], [role="complementary"]';
  const MAIN_SEL = 'main, [role="main"], article, #content, #main, .content, .article, .post, .detail';
  // 语义标签缺失时的兜底：很多站只用 class/id 命名，不写 <nav> / <main>
  const CHROME_WORDS = /(^|[-_ ])(nav|navbar|navigation|menu|sidebar|aside|footer|header|topbar|toolbar|breadcrumb|catalog|category|categories|pagination|pager|tabbar)([-_ ]|$)/i;
  const MAIN_WORDS = /(^|[-_ ])(main|content|article|post|detail|result|results|list|feed|goods|item)([-_ ]|$)/i;

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

  const regionOf = (el) => {
    if (el.closest(CHROME_SEL)) return 'chrome';
    if (el.closest(MAIN_SEL)) return 'main';
    let node = el, depth = 0, mainHit = false;
    while (node && node !== document.body && depth < 6) {
      const cls = (typeof node.className === 'string') ? node.className : '';
      const sig = String(node.id || '') + ' ' + cls;
      if (CHROME_WORDS.test(sig)) return 'chrome';
      if (MAIN_WORDS.test(sig)) mainHit = true;
      node = node.parentElement;
      depth += 1;
    }
    return mainHit ? 'main' : 'neutral';
  };

  const cands = [];
  for (const el of nodes) {
    if (cands.length >= cap) break;
    if (!visible(el)) continue;
    cands.push({
      el: el,
      tag: el.tagName.toLowerCase(),
      text: label(el),
      aria: String(el.getAttribute('aria-label') || '').slice(0, 60),
      placeholder: String(el.getAttribute('placeholder') || '').slice(0, 60),
      region: regionOf(el)
    });
  }

  // 元素引用挂到 window 上，第二趟按分配结果编号时复用同一批 DOM 节点。
  // 两次 querySelectorAll 之间页面若发生变动，编号就会错位到别的元素上——
  // 那是最难查的一类 bug（点击"第 3 个"却点到别的东西）。
  window.__ba_cands = cands.map(c => c.el);
  return cands.map((c, i) => ({
    idx: i, tag: c.tag, text: c.text,
    aria: c.aria, placeholder: c.placeholder, region: c.region
  }));
}
"""

# 回写编号的 JS（第二趟）。只对选中的候选打 data-ba-ref，其余保持无标记。
_APPLY_JS = """
(pairs) => {
  const els = window.__ba_cands || [];
  let n = 0;
  for (const p of pairs) {
    const el = els[p[0]];
    if (!el) continue;
    el.setAttribute('data-ba-ref', String(p[1]));
    n += 1;
  }
  return n;
}
"""

# 正文抽取：**优先主内容区**。
# 直接取 document.body.innerText 会把导航、侧栏分类、页脚版权全混进来，
# 几千字里正文只占一小段 —— 既费 token，又让模型抓不住重点。
_BODY_TEXT_JS = """
() => {
  const MAIN_SEL = 'main, [role="main"], article, #content, #main, .content, .article, .post, .detail';
  const main = document.querySelector(MAIN_SEL);
  if (main) {
    const t = String(main.innerText || '').trim();
    // 太短说明这个 landmark 不是真正的内容容器，回退整页
    if (t.length >= 200) return t;
  }
  return document.body ? document.body.innerText : '';
}
"""


def region_of(candidates: list[dict], region: str) -> list[dict]:
    """按区域挑候选。

    **未知 / 缺失的 region 一律归到 neutral**，而不是丢掉。
    静默丢弃比归错档更糟：丢掉意味着"页面上有东西但 Agent 从来看不见"，
    而且完全没有任何迹象。JS 侧一旦改了字段名，这里就会集体丢元素 ——
    归到中性至少还能被编号、被模型看到。
    """
    if region == "neutral":
        return [c for c in candidates if c.get("region") not in ("main", "chrome")]
    return [c for c in candidates if c.get("region") == region]


def allocate(
    candidates: list[dict],
    max_elements: int = MAX_ELEMENTS,
    chrome_quota: int = CHROME_QUOTA,
) -> list[tuple[int, int]]:
    """决定给哪些候选编号，返回 [(候选下标, 编号), ...]。

    这就是本次改造的核心策略：

      1. **主内容区**的元素先占名额；
      2. **中性区**（判断不出归属的）其次；
      3. **导航 / 侧栏 / 页脚**最后，且最多占 `chrome_quota` 个 ——
         没有这条上限就是改造前的行为：侧栏分类链接吃光 100 个名额；
      4. 只有**主内容区和中性区都为空**时才取消 chrome 上限（导航站、目录页，
         那种页面上链接本身就是内容）。

    第 4 条的判定条件刻意写成"两者都为空"，而不是"数量少于某个阈值"：
    阈值写法会误伤**正是要修的那个场景**（"5 条搜索结果 + 200 个侧栏链接"，
    主内容少但确实是内容），把上限一解除，侧栏又把预算吃光了。

    编号在最后**按文档顺序重排**，而不是按优先级排：模型读到的顺序要跟
    页面从上到下一致，"第一条搜索结果"这类表述才有意义。
    被筛掉是"不编号"，不是"挪到后面"。

    剩余额度**刻意不补给 chrome**：内容都收全之后再塞几十个导航链接，
    只增加 prompt 成本、不增加信息量。
    """
    keep_main = region_of(candidates, "main")
    keep_neutral = region_of(candidates, "neutral")
    keep_chrome = region_of(candidates, "chrome")

    if not keep_main and not keep_neutral:
        chrome_quota = max_elements

    picked = list(keep_main[:max_elements])
    picked += keep_neutral[: max(0, max_elements - len(picked))]
    room = max(0, max_elements - len(picked))
    picked += keep_chrome[: min(chrome_quota, room)]

    picked.sort(key=lambda c: c.get("idx", 0))
    return [(int(c["idx"]), i + 1) for i, c in enumerate(picked)]


async def extract_elements(page: Page) -> list[Element]:
    """通道 A：抽取可交互元素并打上编号。

    分两趟：
      1. 扫（不编号）→ 拿到候选 + 区域归属；
      2. 在 Python 里按区域配额决定给谁编号（allocate 是纯函数，可单测）；
      3. 回写编号。

    **为什么非要分两趟**：分配策略放在 JS 里就只能靠跑真实浏览器来验证，
    而"侧栏抢名额"这类 bug 恰恰只在真实页面上暴露，定位成本极高。
    搬到 Python 之后，用几条构造的候选就能把策略钉死。
    """
    try:
        # SCAN_CAP 必须显式传 —— 见 _COLLECT_JS 顶部的注释：
        # 不传参时 Playwright 给的是 null，会让候选直接变成空列表。
        raw = await page.evaluate(_COLLECT_JS, SCAN_CAP)
    except Exception as exc:  # 页面正在跳转时 evaluate 会失败，属于正常抖动
        log.warning("抽取元素失败（页面可能正在导航）: %s", exc)
        return []

    cands = [c for c in (raw or []) if isinstance(c, dict) and "idx" in c]
    if not cands:
        return []

    pairs = allocate(cands)
    try:
        await page.evaluate(_APPLY_JS, [[idx, ref] for idx, ref in pairs])
    except Exception as exc:
        log.warning("回写元素编号失败: %s", exc)
        return []

    by_idx = {c["idx"]: c for c in cands}
    elements: list[Element] = []
    for idx, ref in pairs:
        c = by_idx.get(idx)
        if c is None:
            continue
        try:
            elements.append(
                Element(
                    ref=ref,
                    tag=str(c.get("tag", "")),
                    text=str(c.get("text", "")),
                    aria=str(c.get("aria", "")),
                    placeholder=str(c.get("placeholder", "")),
                    region=str(c.get("region", "neutral")),
                )
            )
        except Exception:
            continue
    return elements


# ---------------------------------------------------------------------------
# 登录墙识别
#
# ## 为什么要单独识别它（这是用户直接反馈的失效形态）
#
# 遇到需要登录的站点时，Agent 的典型表现是**在登录页和内容页之间反复横跳**：
# 点进内容 → 被弹回登录页 → 再点 → 再被弹回。每一步都是"成功"的、页面也确实
# 在变，所以：动作指纹（动作在交替）抓不到、停滞检测（页面在变）抓不到、
# 振荡检测只能给一句笼统的"你在绕圈子" —— 而模型并不知道**该登录**这件事，
# 于是它继续换着花样点，直到步数耗尽。
#
# 缺的不是"再聪明点的提示词"，是**一个模型自己看不出来的事实**：
# "你现在站在一堵登录墙前面"。识别出来之后，才有"先登录、再找内容"可言。
# ---------------------------------------------------------------------------

# 网址里的登录信号。注意是**路径**级别的词，不是域名里碰巧出现的字母。
_LOGIN_URL_RE = re.compile(
    r"(?:^|[/&?])(login|signin|sign[-_]?in|log[-_]?in|passport|auth|authenticate"
    r"|sso|cas|oauth2?|session/new|account/login|uaa|validate)",
    re.IGNORECASE,
)

# 页面文案里的登录信号（中英文都覆盖：本项目跑的站点国内外都有）
_LOGIN_TEXT_MARKERS = (
    "登录", "登陆", "请先登录", "登录后查看", "登录以继续", "需要登录", "请登录",
    "注册", "验证码", "短信验证", "扫码登录", "密码登录", "账号登录", "立即登录",
    "sign in", "log in", "please log in", "sign in to continue", "login required",
)

# 只靠文案判定时的**强信号**子集：普通的导航里也常有"登录"两个字，
# 光凭它就把整页判成登录墙会误伤内容页（"登录后查看"这种才是真的被挡住）。
_LOGIN_STRONG_MARKERS = (
    "请先登录", "登录后查看", "登录以继续", "需要登录", "请登录",
    "please log in", "sign in to continue", "login required",
)

# 正文短到这个长度 + 命中强信号，基本可以断定"内容被墙挡住了"
_WALL_BODY_CHARS = 1200


def detect_login_wall(
    url: str = "",
    title: str = "",
    body_text: str = "",
    *,
    has_password_field: bool = False,
) -> tuple[bool, str]:
    """判断当前这一帧是不是一堵**登录墙**。纯函数，可离线单测。

    返回 `(是不是, 理由)`。三条判据，都要求**两个以上信号同时成立** ——
    单一信号误伤太大（带"登录"入口的内容页到处都是，把它判成墙会让
    Agent 在该回答的时候直接放弃）。

    ⚠️ 刻意没有"看到密码框就判墙"这一条：很多正常页面的侧栏登录框
    也带 password input，而内容其实是可读的。
    """
    u = (url or "").lower()
    text = f"{title or ''}\n{body_text or ''}"
    low = text.lower()

    url_hit = bool(_LOGIN_URL_RE.search(u))
    text_hit = any(m.lower() in low for m in _LOGIN_TEXT_MARKERS)
    strong_hit = any(m.lower() in low for m in _LOGIN_STRONG_MARKERS)

    if has_password_field and (url_hit or text_hit):
        return True, "页面上有密码输入框，且网址或文案指向登录"
    if url_hit and text_hit:
        return True, "网址与页面文案都指向登录页"
    if strong_hit and len(body_text or "") < _WALL_BODY_CHARS:
        return True, "页面正文很短且明确提示需要登录（内容被登录墙挡住了）"
    return False, ""


_HAS_PASSWORD_JS = "() => document.querySelectorAll('input[type=password]').length > 0"

# "没有页面"的几种写法。about:srcdoc 是 iframe 内联文档被单独导航时的产物，
# 同样意味着"主文档已经不在我们手里了"。
_BLANK_URLS = ("about:blank", "about:srcdoc", "")


def detect_blank_page(
    url: str = "",
    title: str = "",
    body_text: str = "",
    *,
    element_count: int = 0,
) -> tuple[bool, str]:
    """这一帧是不是**被清空的空白页**（什么都没渲染出来）。纯函数，可离线单测。

    ## 为什么这值得单独判一次（实测证据，不是假想）

    2026-09-22 用户拿 BOSS直聘 试搜索页，结果是"无法完成任务"。抓了导航事件
    才看清真实过程 —— 它根本不是"跳到登录页"，而是一个**人机校验跳转循环**：

        /web/geek/job?...          → "加载中，请稍候"
        /web/geek/jobs?...&_security_check=1_...   → 渲染出 1186 个节点（有"登录"入口）
        /web/passport/zp/security.html?code=37...  → 人机校验页
        about:blank  ← 来回跳若干次后停在这里，整页只剩 **3 个节点 / 39 字节**

    停在空白页时，`detect_login_wall` 的**三个信号全灭**：网址是 about:blank
    不像登录、没有文案、没有密码框。于是模型看到的是"一个什么都没有的页面"，
    只能给出"无法完成"。可这个状态的真实含义是**"被挡在门外了"**，不是"没内容"。

    所以这里做一件事：把"空白"本身当成一条**独立证据**报上去，
    让上层有机会把它交给人处理，而不是让模型对着空白页硬编理由。

    ⚠️ 单帧空白**不算数**：页面正常加载的中间态也会是空的（实测 SPA 在
    domcontentloaded 时正文只有 7 个字"加载中，请稍候"）。
    所以这里只负责"这一帧是不是空白"，**连续两帧空白**这个条件由引擎侧
    状态机来判（见 agent.py 的 `blank_streak`）——
    感知层是无状态的，把时序判据塞进来会让它没法纯函数单测。
    """
    u = (url or "").strip().lower()
    if u not in _BLANK_URLS:
        return False, ""
    if (body_text or "").strip():
        return False, ""
    if element_count > 0:
        return False, ""
    return True, f"页面已变成空白页（{url or 'about:blank'}），正文与可交互元素都为空"


def _looks_maybe_login(url: str, title: str, body_text: str) -> bool:
    """网址或文案**有那么一点像**登录页 —— 用来决定要不要去查密码框。

    注意这只是个**省一次 evaluate 的前置过滤**，不是判定本身：
    真正的判定仍然在 `detect_login_wall` 里，规则没变、结果没变。
    """
    u = (url or "").lower()
    low = f"{title or ''}\n{body_text or ''}".lower()
    if _LOGIN_URL_RE.search(u):
        return True
    return any(m.lower() in low for m in _LOGIN_TEXT_MARKERS)


async def extract_body_text(page: Page, max_chars: int = 4000) -> str:
    """读正文。优先主内容区，避免几千字的导航/页脚把正文淹掉。"""
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

    # 登录墙识别放在**元素抽取之后**：有没有密码框是一个更强的信号，
    # 但它必须和网址/文案合起来看（单凭密码框会误伤带侧栏登录框的内容页）。
    #
    # ⚠️ 密码框探测**只在网址或文案已经有点像登录页时才做**：
    # 每一步都多跑一次 `page.evaluate` 是纯浪费（感知是每步都跑的），
    # 而"网址不像、文案也没有登录字眼"的页面上，密码框这个信号根本用不上。
    has_pwd = False
    if _looks_maybe_login(url, title, body_text):
        try:
            has_pwd = bool(await page.evaluate(_HAS_PASSWORD_JS))
        except Exception:  # 页面正在跳转时 evaluate 会失败，按"没有密码框"继续
            has_pwd = False
    is_wall, wall_reason = detect_login_wall(
        url, title, body_text, has_password_field=has_pwd
    )
    # "被清成空白页"与"被登录墙挡住"是两件不同的事实，但处置相同：
    # 都交给人来处理（详见 detect_blank_page 里的实测过程）。
    is_blank, blank_reason = detect_blank_page(
        url, title, body_text, element_count=len(elements)
    )

    state = PageState(
        step=step,
        url=url,
        title=title,
        elements=elements,
        body_text=body_text,
        is_login_wall=is_wall,
        login_reason=wall_reason,
        is_blank_page=is_blank,
        blank_reason=blank_reason,
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

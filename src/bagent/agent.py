"""ReAct 主循环。

ReAct = Reasoning + Acting。每一轮：

    想（Reason）→ 做（Act）→ 看（Observe）→ 再想 ...

和"一次性写好脚本"最大的区别：**每一步都基于真实看到的页面重新决策**。
所以弹窗、验证码、登录态过期、按钮改文案，都不会让整个流程崩掉——
Agent 会看到变化并自己绕路。

关键工程约束（简历里那几条都落在这里）：
1. 终止条件：finish 动作 / 达到最大步数 / 连续失败熔断。
2. 失败重试与策略调整：把失败原因写回历史，并禁止重复同一个失败动作。
3. 双通道感知：DOM 抓不到元素时，自动切视觉通道。
4. 成本面板：每一跳的 token 都记在 llm.usage 里。
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Awaitable, Callable

from .browser import BrowserSession
from .config import Settings
from .grounding import MAX_GROUNDING_RETRIES, check_grounding
from .llm import LLMClient, LLMError, build_client, build_vlm_client
from .models import Action, StepRecord, Usage, parse_action
from .perception import perceive

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是一个浏览器自动化 Agent。你的目标是通过操作真实浏览器完成用户交给的任务。

每一轮，你会看到当前页面的状态：网址、标题、页面上可交互元素的编号清单、以及正文摘要。
你要输出**一个** JSON 对象，描述这一步要做的动作和理由。

可用动作：
- goto       打开网址            参数: url
- click      点击某个元素        参数: ref
- type       在输入框里输入文字  参数: ref, text
- press      按键                参数: key（如 Enter / Escape）
- scroll     滚动页面            参数: dy（正数向下滚，负数向上滚）
- extract    读取当前页面正文    无参数
- screenshot 截图交给视觉通道    无参数
- finish     任务已完成          参数: answer（最终结论）, evidence（见下）

输出格式（**必须是合法 JSON，不要包在代码块里，不要加任何前后解释**）：
{"thought": "这一步为什么这么做", "action": "click", "ref": 3}

必须遵守的规则：
1. 只能使用上面列出的动作名。
2. 点击或输入时，ref 必须来自当前元素清单里的编号，绝对不能自己编。
3. 如果上一步失败了，不要原样重试同一个动作——换一个思路。
4. 连续两轮找不到目标元素时，用 screenshot 让视觉通道帮忙，或者先 scroll 找一找。
5. 页面可能还在加载。如果元素清单为空，可以先 scroll 或 screenshot，不要急着重试点击。
6. 支付、删除、提交订单这类敏感操作会被拦截。被拦截后请换其他路径，不要反复尝试。
7. 任务完成或确认无法完成时，必须调用 finish，并在 answer 里写清结论。

关于 finish（**最容易出错的地方，请重点阅读**）：
- 大多数任务**不需要点击任何元素**。先认真读一遍「页面正文摘要」，
  答案往往已经在里面了。读得到就直接 finish，别去点。
- 如果你已经能从正文里回答出任务要求的信息，**下一步必须是 finish**，
  而不是"再点一下确认"。再点一下只会让你离答案更远。
- 不要点页面顶部的站点名 / logo 链接（通常排在最前面，比如 [1]），
  那只会刷新页面，没有任何意义。
- 不要点「← Previous」「Next →」这类翻页链接，除非任务明确要求翻页。
- 如果某个动作已经连续做过两次却没有带来任何新信息，立刻停止它，
  改用 finish 给出你目前能给出的结论。
- ⚠️「没有新信息」**不等于**"同一个动作连续重复"。**在已经去过的几个页面之间来回走、
  一直没走到新页面，同样是没有新信息** —— 哪怕你每一步点的都是不同的东西。
  这是最常见的死循环形态，一旦出现就不要再试第二次，直接 finish。
  如果连"做不到"都还没确认清楚，就在 answer 里写清你已经试到哪一步。

关于 evidence（**引擎会核对，对不上会被退回重做**）：
- finish 必须带 evidence：从**当前页面正文摘要**里**原样复制**一段能支撑结论的原文
  （不要把词序、标点、大小写"整理"一遍）。
- answer 里提到的每个人名、数字、标题，都**必须真的出现在这段原文里**。
  这是为了防止"引用了 A 却答了 B"。
- ⚠️ **不要凭记忆写 evidence。** 记忆里的和页面上的往往不一样 ——
  如果你的结论来自记忆而不是刚读到的原文，请回去重新读一遍页面再 finish。
- 如果引用的是页码/序号（"第一条""排名第 3"），请连同**它相邻的原文**一起复制，
  只写"第一条"这三个字不构成证据。
- 只有一种情况可以不带 evidence：结论是"我做不到"（例如页面要求登录、
  页面上没有下单按钮）。这类结论断言的是"页面上没有某个东西"，
  引擎不会要求你引用一个不存在的东西。
"""


# 熔断阈值。**提升为模块级常量**，因为 LangGraph 引擎（graph_agent.py）
# 必须用同一套阈值 —— 否则两个引擎的"什么时候停"不一致，A/B 对比就失去意义。
# 原先这三个值是写在 run() 里的局部变量，跨引擎无法共享，属于隐性耦合。
#
# ⚠️ 这两个阈值按**连续**重复计，不是累计。这一条是被一次真实回归打出来的：
# 把 scroll 的指纹合并成方向之后（堵住"换滚动量 = 换新动作"的绕过），
# 累计判据立刻误杀了 t04 —— 它的 5 次 scroll 分散在第 4/6/11/12/14 步，
# 每一步之间都在获得新信息（点结果、翻页、读正文），
# 熔断日志却写着"末尾连续 1 次"。用累计去判定"原地打转"是错的。
LOOP_WARN_AT = 3   # 连续 3 次同一个动作 → 注入强警告
LOOP_STOP_AT = 5   # 连续 5 次 → 熔断
MAX_CONSECUTIVE_FAILURES = 3

# ---- 停滞检测（抓"动作各不相同、页面一字未变"）----
# 为什么光有动作指纹不够：动作指纹只能看**行为重复**，看不见**有没有收获**。
# 真实例子（强模型跑 t06）：20 步里 13 次 scroll，每次 dy 都不一样 ——
# 指纹各异，熔断一次没触发，20 步全烧完任务还是没完成。
# 判据应该是"这一步到底有没有得到新信息"，而那是**页面**的属性，不是动作的。
STALL_WARN_AT = 3  # 连续 3 步页面状态一字未变 → 警告
STALL_STOP_AT = 6  # 连续 6 步 → 进入"收束"（先逼一次结论，再熔断）

# 收束阶段最多容忍模型**拒答几次**。
#
# ⚠️ 数的是"模型连续拒绝给结论的次数"，不是"引擎干等的步数"。
# 第一版数的是步数：跪在阈值的那些步里，模型爱做什么做什么，引擎只负责数够
# CLOSING_MAX_STEPS 就熔断。结果实测（重放 runs/20260921-224847-t06）——
# 收束指令注入之后，模型回手就是 `extract` / `extract` / `click#1`，
# 把指令当耳旁风。**只喊话、不拦动作，等于没管。**
# 现在收束期间的任何非 finish 动作都**不执行**，直接回一条明确的拒绝，
# 拒绝累积到这个数就熔断。语义更准：我们想知道的是"它到底肯不肯给结论"。
#
# **为什么不干脆替它写结论**：反例任务考的就是"识别并声明做不到"
# （见 eval/run_eval.py 的 judge 第一条：熔断不算拒答）。
# 引擎代写等于把判分口径改了 —— 那样数字会变好看，但它不再说明模型有能力。
CLOSING_MAX_STEPS = 3

# 哪些动作**本应改变页面**，因而参与停滞计数。
#
# 排除 `extract` / `screenshot`：它们是只读动作，本来就不改页面，
# 拿它们算停滞是冤枉。
#
# ⚠️ `type` 也排除，理由是实测出来的假阳性来源：**输入框里敲的字
# 不会出现在正文摘要里**（正文抽的是可见文本，读不到 input 的 value），
# 所以"连续填 6 个输入框"这种完全正常的流程会被判成停滞。
# 它在输入框上的原地打转由**动作指纹**那一套负责（t04 的 type#20 连打就是它抓的）——
# 两套机制各管一半，故意留的互补，不是漏掉了。
PROGRESS_ACTIONS = ("click", "goto", "press", "scroll")


# ---- 振荡检测（抓"动作各不相同、页面也确实在变，但在同一小片区域里来回走"）----
#
# 前两套判据都抓不到它，原因是它们看的东西不对：
#   · 动作指纹看"行为有没有重复" —— 振荡时动作是**交替**的（点站点名 / 点加入购物车 /
#     点站点名 / …），任何单个动作的"连续次数"都停在 1；
#   · 停滞检测看"页面有没有变" —— 振荡时页面每一步都在变（A→B→A→B），
#     停滞计数全程只有 0~2，**一次警告都不会注入**。
#
# 实测（`runs/eval_20260922-085119.json`，t06 两条失败轨迹）：模型在书籍站的
# 详情页与目录页之间来回走了 30 步、反复点「Add to basket」，全程没收到任何提示。
#
# ## 为什么只用它发警告，不用它熔断
#
# 离线在已落盘轨迹上比过七组候选判据（`scripts/choose_oscillation_rule.py`：
# "同页连续"、"未见新页"、"重走同一条边"、"窗口内无新页 + 重走同一条边"…，
# 每组再扫 n/w 参数）。最好的一组是 `D(n=3, w=4)`（重走同一条边 ≥3 次）：
# **受害运行 19/19 全抓到，但已收敛的运行里也有 29/62 会触发，其中 11 条是 normal**
# （主要是 t04 必应"首页 ↔ 结果页"的正常往返）。**分不开。**
# （数字是 2026-09-22 的快照，轨迹会随跑测增长；复算用上面那个脚本。）
#
# 分不开的根本原因是：t06 失败与通过的那些运行，**行为本身是一样的**（都在来回走），
# 差别只在"最后有没有收尾"。所以这个信号里没有足够的信息去替模型做"该不该死"的判断。
# → 那就不替它做。引擎只把"你绕了几步"这个**它自己看不到的事实**说清楚，
#   收不收尾仍然由模型决定（收尾判据见提示词里那条"什么时候该收手"）。
OSCILLATION_WINDOW = 4      # 只看最近 4 步
OSCILLATION_MAX_PAGES = 2   # 窗口里的不同页面数 ≤ 2 才算"打转"
# 同一条警告至少隔几步才重复注入一次。不节流的坏处很实在：
# 模型一旦不听，后面每一步都背一段越来越长的警告，prompt 迅速膨胀。
OSCILLATION_NOTIFY_EVERY = 3


def _action_signature(action: Action) -> str:
    """动作指纹：用来识别"原地打转"。

    只看动作名和 ref，**不看 thought** —— 模型每次编的理由都不一样，
    但实际执行的动作可能是同一个。识别循环必须看行为，不能看说辞。

    ⚠️ `scroll` **刻意不带 dy**，这条是一次**模型层消融**抓出来的：

    原先是 `scroll#{dy}`，于是"改个滚动量"就等于换了一个新动作。
    实测（强模型跑到 t06，`runs/20260921-193525-t06/trace.json`）：
    20 步里 13 次 `scroll`，dy 依次是 -200 / -1000 / -500 / -2000 / -3000 /
    +2000 / -3000 / -5000 / -1000 / -10000 / -5000 …，**每个都不一样**，
    指纹各不相同，熔断一次都没触发 —— 20 步全部执行成功、无任何报错，
    任务却仍然没完成。

    **为什么按方向而不是具体像素**：反复滚动本身就是"没有进展"的形态，
    不需要靠"滚了多远"来区分；但"向下找内容 / 再向上回看"是合理动作，
    所以保留 down / up 的方向区分，只把幅度丢掉。

    ⚠️ **这一刀是必要的，但它不够 —— 而且我当时给出的"代价评估"是错的。**

    改完我立刻做了一次"离线复算"：拿 18 次基线重算，结论是
    "只有那 3 次本来就熔断失败的运行会被命中（熔断点 18/18/17），
    通过的运行一次都没受影响"。**这个结论无效，我现在把话收回。**

    离线复算只能回答"**旧轨迹**会不会提前熔断"，回答不了
    "警告改变了模型行为之后，它会走出一条什么样的**新轨迹**"。
    拿一个不存在的未来去证明改动安全，是自欺。代价很快就到了：
    真实基线里 t04 从 2/3 掉到 0/3 —— 累计判据把"分散但正常"的探索误杀了
    （5 次 scroll 分散在第 4/6/11/12/14 步，中间都在获得新信息）。

    最终收口靠两件事：判据从累计改成**连续**；
    以及对"反复滚动"的判定从**动作**换到**页面**
    （`page_fingerprint` / `update_stall`，见下面的停滞检测一节）。
    """
    if action.action in ("click", "type", "goto"):
        return f"{action.action}#{action.ref or action.url}"
    if action.action == "scroll":
        dy = action.dy or 0
        if dy > 0:
            return "scroll#down"
        if dy < 0:
            return "scroll#up"
        return "scroll"
    return action.action


def loop_counts(trail: list[str], sig: str) -> tuple[int, int]:
    """返回 `(累计出现次数, 末尾连续次数)`。

    ⚠️ 这两个数**必须分开算**，因为原来它们被混成了一个：

    判据用的是"整轮累计出现次数"（`trail.count(sig)`），
    但写回历史的措辞是"你已经**连续** N 次执行 `X`"。
    实测（`runs/eval_20260921-145218.json`）三条熔断里有两条对不上：

    - `t05` 的 `click#2` 出现在第 1 / 5 / 7 / 13 / 18 步 ——
      中间隔着 `type` / `extract` / `scroll` / `click#43` / `type`，
      **根本不是连续的**；
    - `t06` 的 `click#1` 出现在第 3 / 8 / 9 / 12 / 17 步，同样不连续；
    - 只有 `t04` 的 `type#20` 是真的连续（第 16 / 17 / 18 步）。

    也就是说，模型读到的是一句**与事实不符**的历史。而小模型本来就
    是靠着这句警告被推着换策略的 —— 在最需要它改的时候喂错信息，
    是这个警告最容易失效的方式。

    这里只把两个数都算出来、如实描述。
    **判据本身要不要从"累计"改成"连续"，是另一个问题**：
    那会改变熔断时机、进而动到基线数字，必须单独立项测
    （见 `docs/实验记录.md`），不能混在"修措辞"里偷偷换掉。
    """
    total = trail.count(sig)
    consecutive = 0
    for s in reversed(trail):
        if s != sig:
            break
        consecutive += 1
    return total, consecutive


def page_fingerprint(state) -> str:
    """把一帧页面压成一个短哈希 —— 回答"我到底有没有得到新信息？"

    与动作指纹是**两套互补的判据**，别把它们当成一回事：

    | 判据 | 抓什么 | 真实案例 |
    |---|---|---|
    | 动作指纹 | 同一个动作被**重复** | 7B t04：`type#20` 连打 |
    | 页面指纹 | 动作各不相同，但页面**一字未变** | 强模型 t06：13 次 scroll，每次 dy 都不同 |

    后者是纯行为统计抓不到的：动作每次都"新"，可收获始终是零。

    ## 指纹里**只放"能读到的信息"**：url + title + 正文

    ⚠️ 元素清单**必须剔除**，这一条是量出来的，不是想出来的：

    我原先把"元素清单（DOM 顺序 + tag + text）"也算进指纹，直觉是
    "页面结构变了 = 有新信息"。用真浏览器量了一遍
    （`scripts/probe_page_fp.py`，books.toscrape.com 连滑 4 次再连滑 4 次）：

    ```
    动作            指纹              元素数   正文长
    goto            533ea144404caad5     85     2029
    scroll +500     533ea144404caad5     85     2029
    scroll +500     56f43498c80d1490     85     2029
    scroll +500     1aaa2f57c2b72731     77     2029
    scroll +500     34d279a4bc2bcffd     71     2029
    scroll -500     b12911403fe18644     85     2029
    scroll -500     533ea144404caad5     85     2029   ← 滑回原位，指纹原样回来了
    ```

    **正文长度全程 2029 一字未变**，变的是元素数（85 → 77 → 71 → 85）。
    原因是感知层采集元素时带**视口过滤**（`r.bottom < -800` 才丢弃，
    见 perception.py 的 `visible`），滚动会让页面顶部的元素掉出采集范围。

    也就是说：这个抖动**纯粹是"我站在哪儿看"，不是"页面上多了什么"**。
    带着它做停滞检测，后果是"滚动打转"这一类**恰好检测不到**
    （每次滑指纹都变，计数器永远清零）—— 而那正是这个机制要抓的主要形态。
    一个在自己目标场景上失灵的判据，比没有更糟：它会让日志显得"检查过了"。

    剔除元素清单之后，"有没有新信息"就等价于"正文/标题/网址有没有变"，
    而正文恰好是模型能读出答案的唯一来源 —— 口径自洽。

    ⚠️ 也不含 `screenshot_path`：那是一次性的临时文件名，跟页面内容无关。

    ## 已知局限（写出来，别假装没有）

    `perceive` 在元素解析为空时会截图，配了视觉模型的话会把一段**描述**并进
    `body_text`（见 perception.py 的 `prefer_vision` 分支）。那段描述每帧措辞
    未必相同，于是指纹会一直变 —— **在 Canvas / iframe 那类页面上，停滞检测
    会被视觉通道的描述差异"喂"成永远有进展**。

    没有为它做特殊处理，理由有两条：一是那种页面上"描述确实变了"未必是假象；
    二是要压掉它就得把正文和视觉描述分开存，那是感知层的接口改动，
    应该和"视觉通道到底有多少收益"一起评估，不该塞在这次的修复里悄悄做掉。
    记在这里，等它真的成为瓶颈再动。
    """
    parts = [
        state.url or "",
        state.title or "",
        state.body_text or "",
    ]
    joined = "\n".join(parts)
    return hashlib.sha1(joined.encode("utf-8", "replace")).hexdigest()[:16]


def update_stall(prev_fp: str, cur_fp: str, action_name: str | None, count: int) -> int:
    """更新"连续无新信息"的计数，返回新的计数值。

    只对 `PROGRESS_ACTIONS` 里的动作计数（见该常量的说明）。判定很直白：

    - 动作不属于"本应改变页面"的一类（type/extract/screenshot/finish/None）
      → 不计，也不算进展，原样返回；
    - 手上还没有上一帧指纹（第 1 步）→ 无从比较，原样返回；
    - 页面指纹变了 → **有进展，清零**；
    - 页面指纹一样 → 计 +1。
    """
    if action_name not in PROGRESS_ACTIONS:
        return count
    if not prev_fp or not cur_fp:
        return count
    return 0 if cur_fp != prev_fp else count + 1


def oscillation_pages(
    fp_history: list[str],
    window: int = OSCILLATION_WINDOW,
    max_pages: int = OSCILLATION_MAX_PAGES,
) -> int | None:
    """最近 `window` 步是否"只在已去过的 ≤`max_pages` 个页面之间来回走"。

    是则返回窗口里实际涉及的不同页面数（供警告文案用），否则返回 None。

    两个条件缺一不可（这是**实测调出来的**，不是设计直觉）：

    - **窗口内没有新指纹**：光"重走同一条边"不够 —— 必应"首页 ↔ 结果页"的正常往返
      也满足它，而那种往返每轮都在拿到新结果页。加这一条把 t04 那类误触挡掉一部分。
    - **窗口内不同页面数落在 [2, max_pages]**：下界 2 是**刻意**的 ——
      窗口里只有同一个页面时，那是"页面一字未变"，归 `update_stall` 管；
      这里只管"在**多个**页面之间来回"。两套机制各管一半，是互补不是重叠
      （和 `PROGRESS_ACTIONS` 排除 `type` 是同一个处理方式）。
      上界则由"没有新页面"这条兜着：没有新页面 + 不同页面数很少 = 在小片区域里绕。

    ⚠️ 它**不是**熔断判据，只是"值得告诉模型的一件事"。理由见常量区的说明：
    这个信号在已收敛的运行上误触率太高（23/51），拿它熔断会把 t04 一起打掉。
    """
    if window < 1 or len(fp_history) <= window:
        # 历史还不够长 —— 没有"更早"的部分可比，"新不新"无从判断。
        return None
    recent = fp_history[-window:]
    earlier = set(fp_history[:-window])
    if any(fp not in earlier for fp in recent):
        return None  # 窗口里出现了以前没见过的页面 → 有进展
    distinct = len(set(recent))
    if not 2 <= distinct <= max_pages:
        return None
    return distinct


def oscillation_warning(pages: int) -> str:
    """振荡提醒的注入文本。**两个引擎共用这一份**。

    以前这段文本在 agent.py 和 graph_agent.py 里各写了一遍，靠一句注释声明
    "逐字一致"。实测中它就真的漂移了：改了一处忘另一处，两个引擎的提示词
    从此不同，而没有任何测试会红。改成函数是**唯一能真正防漂移**的做法。

    ## 措辞上踩过的坑（第一版就是这么写的，代价是一次真实退化）

    第一版结尾是："如果已经确认这件事做不到，就**直接 finish**，在 answer 里
    写清为什么做不到（这类结论免检 evidence，不需要引用不存在的东西）。"

    它把"免检 evidence"当成了奖励发出去。结果模型在**正常任务**（t04，必应搜
    ReAct 论文读第一条标题）上一收到提醒就去交差：4 次里 3 次答
    "无法找到搜索结果"/"这个页面上没有搜索结果" —— 而改前 4 次答的都是页面上的
    真内容。绕圈子只说明没走对路，不等于任务做不到；这两件事必须分开说。

    所以现在这段文本：只给"换一个明显不同的做法"这条路，并且**显式拦住**
    "把绕圈子当做不到"。免检规则仍然在 SYSTEM_PROMPT 的 evidence 一节里，
    但不再在循环里当胡萝卜。
    """
    return (
        f"⚠ 提醒：最近 {OSCILLATION_WINDOW} 步里，你只在"
        f"**{pages} 个已经去过的页面之间来回走**，一步都没有走到新页面 —— "
        f"这说明当前做法不会再带来新信息。"
        f"请立刻换一个**明显不同**的做法（例如直接用 goto 打开一个明确的 URL，"
        f"或从另一个入口重新开始），而不是再点一次同类的东西。\n"
        f"⚠ 但别把「绕圈子」当成「做不到」：**「我还没找到」不等于「页面上没有」**。"
        f"来回走只说明你没走对路。只有当你已经站在正确的页面上、"
        f"确实确认过这件事无法完成，才把结论写成「做不到」；否则先换做法。"
    )


# 喂给模型的历史保留多少条。控制 prompt 长度就是控制成本。
#
# ⚠️ 抽成常量是因为这个数原来在代码里**散着写了 5 遍**（agent.py 4 处 +
# graph_agent 的 _note 1 处）。想调一次 prompt 成本得改 5 个地方，
# 漏一个就会出现"实际生效的窗口和以为的不一样"这种最难查的偏差。
HISTORY_WINDOW = 12


def trim_history(history: list[str]) -> list[str]:
    """裁到最近 `HISTORY_WINDOW` 条，返回新列表（调用方要接住返回值）。"""
    return history[-HISTORY_WINDOW:]


# 登录墙的提示至少隔几步重发一次。不节流的后果和振荡警告一样：
# 模型不听的时候，每一步都背一段越来越长的话，prompt 迅速膨胀。
LOGIN_NOTIFY_EVERY = 4


def login_blocked_warning(reason: str) -> str:
    """「这是登录墙，而且没人能登录」时的注入文本。两个引擎共用一份。

    ## 措辞上必须同时说清三件事（少一件就会回到横跳）

    1. **这是一堵墙**，不是"内容页加载慢"；
    2. **先登录再找内容** —— 顺序不能反（用户反馈的失效形态恰恰是反着的：
       它一直在找内容，从没认真处理登录这件事）；
    3. **来回切换没有用** —— 明确禁止"再点一次内容页"。

    ## 为什么这里不替它登录

    引擎没有账号凭据，也不该保存用户密码。"登录"这件事只能由人来完成，
    引擎能做的是**把门打开**（登录交接，见 `login_handoff`）和
    **在没人能登录时让它体面地停下**（这一段）。
    """
    return (
        f"⚠ 当前页面需要登录才能看到内容（{reason}），而你**没有可用的账号凭据**。\n"
        f"正确的顺序是：**先登录，再找内容**。在没有完成登录之前去找内容，"
        f"只会被弹回这个页面 —— 所以**不要**再在登录页和内容页之间来回切换，"
        f"那一步都不会带来新信息。\n"
        "你现在只有两条路：\n"
        "  (a) 换一个**不需要登录**的入口（例如直接用 goto 打开一个明确的、"
        "公开可访问的网址）；\n"
        "  (b) 如果确认无法继续，立刻 finish，在 answer 里写明"
        "「需要登录，无法完成」（这类结论免检 evidence，不需要引用不存在的东西）。"
    )


# 登录交接：收到 (当前网址, 判定理由)，返回 True 表示"人已经登录好了，继续跑"。
# 由调用方决定"人怎么登录" —— CLI 是终端里按回车，API 是控制台上点按钮。
# 引擎只认这个契约，不关心它背后是哪种交互。
LoginHandoff = Callable[[str, str], Awaitable[bool]]


@dataclass
class RunResult:
    """一次完整运行的产物。评测脚本读的就是这个。"""

    task: str
    start_url: str
    success: bool
    finished: bool
    answer: str
    steps: int
    elapsed_seconds: float
    usage: Usage
    cost_yuan: float
    # 这次是不是离线替身跑的。与 cost_yuan 一起构成"花钱口径"：
    # offline=True 时 usage 里的 token 只是估算值、cost_yuan 恒为 0。
    offline: bool = False
    records: list[StepRecord] = field(default_factory=list)
    run_dir: str = ""
    error: str = ""
    # 证据锚定结果。`grounded=True` 表示"结论通过了证据核对"（不是"答对了"，
    # 两者必须分开看：判分归评测，这里只回答"结论有没有页面依据"）。
    # `grounding_retries` 记被退回重做几次 —— 它本身就是一个可观测的自纠指标。
    grounded: bool = False
    grounding_note: str = ""
    grounding_retries: int = 0


# action 可以是 None：模型输出不是合法 JSON 的那一步没有可执行的 Action，
# 但它仍是一次真实的模型调用，回调必须能把它报出来（否则前端时间线会缺号）。
StepCallback = Callable[[int, "Action | None", bool, str], None]


class ReActAgent:
    def __init__(
        self,
        settings: Settings,
        llm: LLMClient | None = None,
        on_step: StepCallback | None = None,
    ) -> None:
        self.settings = settings
        self.llm = llm or build_client(settings)
        self.on_step = on_step
        # 视觉通道的客户端。**必须自己造并传给 perceive**：那边判定
        # `settings.vlm_enabled and vlm is not None`，引擎不传就永远为假 ——
        # 结果是"截图照拍、描述从来没跑过"，配了 VLM key 也看不出任何区别。
        # 构造失败不致命（视觉只是降级通道），记一条警告继续走纯 DOM。
        self.vlm: LLMClient | None = None
        if getattr(settings, "vlm_enabled", False):
            try:
                self.vlm = build_vlm_client(settings)
            except Exception as exc:
                log.warning("视觉模型客户端创建失败，本次只走 DOM 通道: %s", exc)

    async def run(
        self,
        task: str,
        start_url: str,
        session: BrowserSession,
        *,
        run_dir: Path,
        confirm: Callable[[str], Awaitable[bool]] | None = None,
        max_steps: int | None = None,
        login_handoff: LoginHandoff | None = None,
    ) -> RunResult:
        """跑一个任务。

        `login_handoff`：遇到登录墙时把控制权交给人的回调。给了它，
        Agent 才能"先登录、再找内容"；不给（评测、无人值守的服务），
        遇到墙就走 `login_blocked_warning` 那套"体面停下"的逻辑 ——
        **绝不**假装登录过了。
        """
        max_steps = max_steps or self.settings.max_steps
        started = time.time()
        run_dir.mkdir(parents=True, exist_ok=True)

        await session.execute(Action(action="goto", url=start_url), confirm=confirm)

        history: list[str] = []
        records: list[StepRecord] = []
        consecutive_failures = 0
        answer, finished, error = "", False, ""
        prefer_vision = False
        # finish 被证据核对退回的次数。超过上限就采纳答案并把 grounded 标 False ——
        # 目的是"逼模型自纠"，不是"把它卡死"（卡死只会让评测从答错变成没答案）。
        grounding_retries = 0
        grounded, grounding_note = False, ""
        # 动作指纹轨迹：用来抓"原地打转"。
        # 只靠 consecutive_failures 抓不到循环——循环里每一步都是"成功"的，
        # 只是毫无进展。这是 7B 级别模型最典型的失效方式。
        sig_trail: list[str] = []

        # ---- 停滞检测（页面指纹）状态 ----
        # prev_fp 是**上一帧**页面指纹；注意它与 last_action 是配对使用的：
        # 第 N 步顶部比较的是「第 N-1 步的页面」与「第 N 步的页面」，
        # 反映的正是**第 N-1 步那个动作**有没有带来新信息，所以传的是
        # `pending_action`（上一步的动作），不是本步的。
        prev_fp = ""
        last_action: Action | None = None
        stall_count = 0
        closing = False        # 是否已进入"收束"（逼结论）阶段
        closing_steps = 0
        # 页面指纹历史。停滞检测只需要"上一帧"，振荡检测需要**一整个窗口**，
        # 所以这里留下完整序列 —— 存的是 16 位哈希，不是正文，代价可以忽略。
        fp_history: list[str] = []
        osc_notified_at = 0    # 上一次振荡警告是在第几步注入的（用于节流）
        # 登录墙的状态机：
        #   login_done         —— 人已经登录过一次了（不再重复打扰）
        #   login_notified_at  —— 上一次"没人能登录"的提示是第几步（用于节流）
        login_done = False
        login_notified_at = 0

        for step in range(1, max_steps + 1):
            # 本步有没有因为"振荡"给过提醒（写进轨迹，供事后核对）。
            # 每步重置：它是"这一步提醒了没有"，不是累计值。
            osc_warned = 0

            # ---------- 看（Observe）----------
            state = await perceive(
                session.page,
                self.settings,
                step=step,
                run_dir=run_dir,
                prefer_vision=prefer_vision,
                vlm=self.vlm,
            )
            prefer_vision = False

            # ---------- 登录墙：先登录，再找内容 ----------
            # 放在所有循环判据之前，因为它是**更具体**的解释：
            # "你在登录页和内容页之间来回"这件事，与其让振荡检测笼统地说
            # "你在绕圈子"，不如直接说"你被登录墙挡住了、顺序还反了"。
            if state.is_login_wall:
                if not login_done and login_handoff is not None:
                    resumed = False
                    try:
                        resumed = await login_handoff(state.url, state.login_reason)
                    except Exception as exc:
                        # 交接失败不能把任务带崩：视作"没登录成"，走下面的提示分支。
                        log.warning("登录交接失败，按未登录处理: %s", exc)
                    if resumed:
                        login_done = True
                        # 登录之后世界变了：清掉所有"基于旧页面"的判据状态。
                        # 不清的后果很具体 —— 登录页↔内容页那几个旧指纹还在
                        # fp_history 里，登录后第一步就会被判成"又在振荡"，
                        # 把刚刚成功的登录又带偏回去。
                        fp_history.clear()
                        prev_fp = ""
                        stall_count = 0
                        closing, closing_steps = False, 0
                        consecutive_failures = 0
                        sig_trail.clear()
                        last_action = None
                        # 回到任务入口：登录之前打开的往往是登录页，
                        # 只有重新打开起始网址，内容才真的在。
                        await session.execute(
                            Action(action="goto", url=start_url), confirm=confirm
                        )
                        records.append(
                            StepRecord(
                                step=step, action=None, raw_action="登录交接",
                                ok=True,
                                message=f"已完成登录，回到任务入口 {start_url}",
                                url_after=session.page.url if session.page else "",
                                screenshot_path=state.screenshot_path,
                                # 登录前那一帧的指纹（下面马上会清空历史）：
                                # 留着它，复盘时才能看出"登录前后页面确实变了"。
                                page_fp=page_fingerprint(state),
                                stall_count=0, osc_pages=osc_warned,
                            )
                        )
                        if self.on_step:
                            self.on_step(step, None, True, "已完成登录，回到任务入口")
                        history.append(
                            f"第 {step} 步: 登录已完成，已重新打开任务入口 {start_url}。"
                            f"现在按任务要求去找内容 —— 顺序是**先登录后找内容**，不要反过来。"
                        )
                        history = trim_history(history)
                        continue

                # 没有交接通道，或交接没成功 → 明确告诉它"停下来"，并节流。
                # `login_notified_at == 0` 这一支是**第一次遇到**：
                # 节流不能把第一次提醒也节掉（否则前面几步它还在盲目横跳）。
                if login_notified_at == 0 or step - login_notified_at >= LOGIN_NOTIFY_EVERY:
                    login_notified_at = step
                    history.append(login_blocked_warning(state.login_reason))

            # ---------- 停滞检测：这一步到底有没有得到新信息？----------
            # 放在"想（Reason）"**之前**，是为了让下面的警告进到本轮 prompt 里 ——
            # 放到 action 之后再判，警告就要等到下一轮才生效，白白多烧一步。
            pending_action, last_action = last_action, None
            cur_fp = page_fingerprint(state)
            stall_count = update_stall(
                prev_fp, cur_fp,
                pending_action.action if pending_action else None,
                stall_count,
            )
            prev_fp = cur_fp
            fp_history.append(cur_fp)

            if stall_count >= STALL_STOP_AT:
                if not closing:
                    # 首次触顶：进入收束，并注入一次强指令。
                    closing = True
                    history.append(
                        f"⚠ 严重警告：页面已经连续 {stall_count} 步**一字未变** —— "
                        f"继续操作不可能再得到任何新信息了。"
                        f"你现在**必须立刻调用 finish**，把你已经确知的结论写进 answer。"
                        f"如果结论是「这个页面上没有某某功能 / 无法完成」这类，"
                        f"直接照实写出来即可，**不需要**为它找 evidence"
                        f"（引擎对这类结论免检证据）。"
                    )
            elif closing:
                # 页面重新有了变化 → 说明动作生效了，退出收束，恢复正常节奏。
                # 注意 `closing_steps` 不在这里清零 —— 见下面"收束预算"的说明。
                closing = False
            elif stall_count >= STALL_WARN_AT:
                history.append(
                    f"⚠ 警告：页面已经连续 {stall_count} 步没有任何变化，"
                    f"说明当前这个方向得不到新信息。"
                    f"请换一个明显不同的做法，或者直接调用 finish 给出你目前的结论。"
                )

            # ---------- 振荡检测：在已经去过的几个页面之间来回走 ----------
            # 与上面两支**并列**，不是它的 else 分支：两支要抓的是不同的形态
            # （一支页面完全不变，一支页面一直在变）。同时命中虽然少见，
            # 但真发生时以更严重的停滞警告为准，所以放在 elif 链之外单独判。
            #
            # ⚠️ 这里**只发警告，不进收束、不拦动作**。理由写在常量区：
            # 这个信号在已收敛的运行上也会触发（约一半，含 11 条 normal——
            # 口径与出处见上方 OSCILLATION_* 常量区），拿它拦动作会连
            # t04 的正常往返一起打掉。
            # 引擎在这里的角色是"把模型看不到的事实告诉它"，不是替它做决定。
            # 登录墙上不再发振荡警告：上面已经给过**更具体**的解释了，
            # 同时喂两句会互相打架（一句说"换做法"，一句说"绕圈子"）。
            if not closing and not state.is_login_wall and step - osc_notified_at >= OSCILLATION_NOTIFY_EVERY:
                osc_pages = oscillation_pages(fp_history)
                if osc_pages is not None:
                    osc_notified_at = step
                    osc_warned = osc_pages   # 落进 trace，供事后核对
                    history.append(oscillation_warning(osc_pages))

            # ---------- 想（Reason）----------
            user_prompt = self._build_prompt(task, state, history)
            raw = ""
            try:
                # 用 achat 而不是 chat：同步 SDK 会卡住事件循环，
                # 那段时间里浏览器的加载/等待全停摆（见 llm.achat 的说明）。
                raw = await self.llm.achat(
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    json_mode=True,
                )
            except LLMError as exc:
                error = str(exc)
                log.error("模型调用失败，终止任务: %s", exc)
                break

            action, parse_err = parse_action(raw)
            if action is None:
                # 模型吐了不合法的 JSON。把它当成一次失败反馈回去，而不是直接崩。
                consecutive_failures += 1
                history.append(f"第 {step} 步: 输出格式不合法（{parse_err}），请只输出合法 JSON")
                # 这一轮**确实消耗了一次模型调用**，必须留下记录。
                # 原来这里直接 continue 不 append，导致 records 里的 step 编号跳号
                # （实测 [1,3,4,5]），审计轨迹看起来像"漏了第 2 步"，
                # 实际是那一步的失败被整个吞掉了 —— 排障时最不该丢的就是失败那一步。
                #
                # 注意不能给 action 塞一个编造的动作名：ActionName 是封闭字面量，
                # 构造时就会 ValidationError（踩过）。用 raw_action 承载原始字符串。
                records.append(
                    StepRecord(
                        step=step,
                        action=None,
                        raw_action="(格式错误)",
                        # 原始输出留档：复盘时要回答"它到底错成什么样"，
                        # 而 `raw_action` 那个占位标签回答不了这个问题。
                        raw_output=(raw or "")[:300],
                        ok=False,
                        message=f"模型输出不是合法 JSON：{parse_err}",
                        url_after=state.url,
                        screenshot_path=state.screenshot_path,
                        page_fp=cur_fp,
                        stall_count=stall_count,
                        osc_pages=osc_warned,
                    )
                )
                if self.on_step:
                    # 回调签名要 Action，这里没有合法 Action 可传；用 None 表示
                    # "这一步没有动作"，让订阅方自己决定怎么展示。
                    self.on_step(step, None, False, f"输出格式不合法：{parse_err}")
                if consecutive_failures >= 3:
                    error = f"模型连续输出非法格式: {parse_err}"
                    break
                continue

            # ---------- 做（Act）----------
            if action.action == "finish":
                verdict = check_grounding(
                    action.answer or "", action.evidence or "", state.body_text, task
                )
                # 证据对不上 → 退回重做。这是本引擎里唯一"会拒绝 finish"的地方，
                # 所以退回原因必须写得足够具体，否则小模型只会原地再交一次同样的答案。
                if not verdict.ok and grounding_retries < MAX_GROUNDING_RETRIES:
                    grounding_retries += 1
                    message = f"finish 被退回（{grounding_retries}/{MAX_GROUNDING_RETRIES}）：{verdict.reason}"
                    records.append(
                        StepRecord(
                            step=step, action=action, ok=False, message=message,
                            url_after=state.url, screenshot_path=state.screenshot_path,
                            page_fp=cur_fp, stall_count=stall_count, osc_pages=osc_warned,
                        )
                    )
                    if self.on_step:
                        self.on_step(step, action, False, message)
                    history.append(
                        f"第 {step} 步: {message}。"
                        "请重新读一遍「页面正文摘要」，把支撑结论的原文**原样复制**到 evidence，"
                        "并确认 answer 里的每个人名 / 数字 / 标题都出现在这段原文里，再调用 finish。"
                    )
                    history = trim_history(history)
                    continue

                finished = True
                answer = (action.answer or "").strip()
                grounded = verdict.grounded
                note = verdict.reason
                if not verdict.ok:
                    note = (
                        f"已重试 {grounding_retries} 次仍未通过证据核对，按当前答案收尾："
                        f"{verdict.reason}"
                    )
                elif not verdict.checked:
                    note = f"未做证据核对（{verdict.reason}）"
                grounding_note = note
                records.append(
                    StepRecord(
                        step=step, action=action, ok=True,
                        message="任务结束" + ("（证据已核对）" if grounded else f"（{note}）"),
                        url_after=state.url,
                        page_fp=cur_fp, stall_count=stall_count, osc_pages=osc_warned,
                    )
                )
                if self.on_step:
                    self.on_step(step, action, True, "任务结束（证据已核对）" if grounded else "任务结束")
                break

            # ---------- 收束阶段：只接受结论，不接受"再试一下" ----------
            # 只喊话是不够的 —— 实测（把 runs/20260921-224847-t06 的动作序列
            # 重放一遍）：收束指令在第 14 步注入，模型回手就是
            # `extract` / `extract` / `click#1`，把指令当耳旁风，一直磨到 20 步上限。
            # 所以这里**真的拦住它**：收束期间的任何非 finish 动作都不执行，
            # 只回一条明确的拒绝。
            #
            # 这样"预算"数的就是**模型拒答的次数**，而不是引擎干等的步数 ——
            # 语义更准：我们真正想知道的是"它到底肯不肯给结论"。
            #
            # 为什么不干脆替它写结论：反例任务考的就是"识别并声明做不到"
            # （见 eval/run_eval.py 的 judge 第一条），引擎代写等于把判分口径改了。
            # 这里只能把"必须表态"变成硬约束，不能把结论变成引擎的产物。
            if closing and action.action != "finish":
                closing_steps += 1
                message = (
                    f"收束阶段不接受 `{action.action}`：页面已连续 {stall_count} 步无变化，"
                    f"没有新信息可获取。现在**只剩 finish 一个选项**。"
                )
                records.append(
                    StepRecord(
                        step=step, action=action, ok=False, message=message,
                        url_after=state.url, screenshot_path=state.screenshot_path,
                        page_fp=cur_fp, stall_count=stall_count, osc_pages=osc_warned,
                    )
                )
                if self.on_step:
                    self.on_step(step, action, False, message)
                if closing_steps >= CLOSING_MAX_STEPS:
                    error = (
                        f"页面连续 {stall_count} 步毫无变化（判定为原地打转），"
                        f"收束后又连续 {closing_steps} 次拒绝给出结论，熔断"
                    )
                    break
                history.append(
                    f"第 {step} 步: {message}"
                    '请立刻输出 {"action": "finish", "answer": "..."} —— '
                    "把你能确定的结论写进 answer 就够了。"
                )
                history = trim_history(history)
                continue

            if action.action == "extract":
                outcome_ok, message = True, f"已读取页面正文（{len(state.body_text)} 字）"
                prefer_vision = False
            else:
                outcome = await session.execute(action, confirm=confirm)
                outcome_ok, message = outcome.ok, outcome.message
                # 点击/输入失败 → 下一帧强制走视觉通道，多一个信息源
                prefer_vision = outcome.vision_hint

            records.append(
                StepRecord(
                    step=step, action=action, ok=outcome_ok, message=message,
                    url_after=session.page.url if session.page else "",
                    screenshot_path=state.screenshot_path,
                    page_fp=cur_fp, stall_count=stall_count, osc_pages=osc_warned,
                )
            )
            if self.on_step:
                self.on_step(step, action, outcome_ok, message)

            # ---------- 记录 ----------
            # 记下这一步的动作，供**下一步顶部**的停滞检测判断"它有没有带来新信息"。
            # 只在这里赋值（而不是在循环开头），是为了让 `continue` 出去的路径
            # （finish 被退回、JSON 非法）天然留下 None —— 那些情况没有产生
            # 可归因的动作，停滞计数就该原样不动，不能冤枉也别白送进展。
            last_action = action
            status = "成功" if outcome_ok else "失败"
            desc = f"第 {step} 步: {self._describe(action)} → {status}: {message}"
            if not outcome_ok:
                desc += "（不要重复这个动作）"
            history.append(desc)
            history = trim_history(history)  # 只保留最近 12 步，控制 prompt 长度

            consecutive_failures = 0 if outcome_ok else consecutive_failures + 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                error = "连续 3 步失败，已熔断"
                break

            # ---------- 循环检测 ----------
            # ⚠️ 判据是**末尾连续**次数，不是累计。这一条是被一次真实回归打出来的：
            # 把 scroll 指纹合并成方向之后，累计判据立刻误杀了 t04 ——
            # 它的 5 次 scroll 分散在第 4/6/11/12/14 步，每一步之间都在获得新信息
            # （点结果、翻页、读正文），旧日志却写着"末尾连续 1 次"。
            # 用累计去判定"原地打转"是错的：原地打转的定义是"没有新信息"，
            # 而那是**页面**的属性 —— 由上面的停滞检测负责；
            # 这里这一套只管"同一个动作真的连着重复"这件事。
            sig = _action_signature(action)
            sig_trail.append(sig)
            repeats, consecutive = loop_counts(sig_trail, sig)
            if consecutive >= LOOP_STOP_AT:
                error = (
                    f"检测到原地打转：动作 {sig} 已连续 {consecutive} 次"
                    f"（整轮累计出现 {repeats} 次）且无进展，已熔断"
                )
                break
            if consecutive >= LOOP_WARN_AT:
                # 把警告写回历史，下一轮模型一定会看到。
                # 注意措辞是"禁止"而不是"建议"——小模型对强约束才有反应。
                # 数字必须如实：判据是**末尾连续**，就不能写成"累计"（见 loop_counts）。
                history.append(
                    f"⚠ 严重警告：动作 `{sig}` 已经**连续**出现 {consecutive} 次"
                    f"（整轮累计 {repeats} 次），"
                    f"页面没有任何进展。**禁止再执行这个动作**。"
                    f"你现在必须二选一：(a) 换一个完全不同的元素编号；(b) 直接调用 finish "
                    f"给出结论。如果你已经能从页面正文里看到答案，请立刻 finish。"
                )
        else:
            error = f"达到最大步数上限（{max_steps}），任务未完成"

        elapsed = time.time() - started
        usage = self.llm.usage
        # 离线替身没有真实计费：它的 token 是按字符估出来的。若照常乘单价，
        # 会凭空算出一个"成本"（实测离线跑一次显示 ¥0.00347），
        # 使用者会以为这次真的花了钱 —— 离线模式的成本必须是 0。
        offline = bool(getattr(self.settings, "mock", False))
        result = RunResult(
            task=task,
            start_url=start_url,
            success=finished,
            finished=finished,
            answer=answer,
            steps=len(records),
            elapsed_seconds=round(elapsed, 2),
            usage=usage,
            cost_yuan=0.0 if offline else round(
                usage.cost_yuan(
                    self.settings.price_in_per_mtok, self.settings.price_out_per_mtok
                ),
                6,
            ),
            offline=offline,
            records=records,
            run_dir=str(run_dir),
            error=error,
            grounded=grounded,
            grounding_note=grounding_note,
            grounding_retries=grounding_retries,
        )
        self._dump(result, run_dir / "trace.json")
        return result

    # ------------------------------------------------------------------
    def _build_prompt(self, task: str, state, history: list[str]) -> str:
        parts = [f"# 任务\n{task}", ""]
        if history:
            parts += ["# 已经执行过的步骤"] + [f"- {h}" for h in history] + [""]
        parts += ["# 当前页面状态", state.render_for_prompt(), ""]
        parts.append("请输出下一步动作的 JSON。")
        return "\n".join(parts)

    @staticmethod
    def _describe(action: Action) -> str:
        if action.action == "goto":
            return f"goto {action.url}"
        if action.action == "click":
            return f"click [{action.ref}]"
        if action.action == "type":
            return f"type [{action.ref}] = {action.text!r}"
        if action.action == "press":
            return f"press {action.key}"
        if action.action == "scroll":
            return f"scroll {action.dy}"
        return action.action

    @staticmethod
    def _dump(result: RunResult, path: Path) -> None:
        payload = result.__dict__.copy()
        payload["records"] = [r.model_dump() for r in result.records]
        payload["usage"] = result.usage.model_dump()
        try:
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            log.warning("写入 trace 失败: %s", exc)


def new_run_dir(settings: Settings, tag: str = "run") -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe = "".join(c for c in tag if c.isalnum() or c in "-_")[:24] or "run"
    path = settings.runs_dir / f"{stamp}-{safe}"
    path.mkdir(parents=True, exist_ok=True)
    return path

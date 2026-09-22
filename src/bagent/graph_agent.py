"""LangGraph 版 ReAct 引擎。

## 为什么要有这个文件

`agent.py` 里那个手写 `for step in range(...)` 循环是这个项目的第一版实现，
它的价值在于**没有隐藏任何东西**：终止条件、历史裁剪、循环检测全是显式代码。

但真实项目里，一旦业务流程变复杂（加人工确认节点、加并行工具调用、
加断点续跑、加人在回路），手写循环的状态管理就会迅速失控——
"下一步该走哪个分支"散落在十几处 `if/break` 里。

所以这一版用 **LangGraph 的 StateGraph** 把同一个 ReAct 循环显式建模成状态图：

    perceive ──> reason ──> act ──> (条件边) ──> perceive   # 继续
                                  ├────────────> END       # 正常结束
                                  └────────────> END       # 熔断/步数上限

**两个引擎共用同一套 `perceive` / `session.execute` / `SYSTEM_PROMPT` / `RunResult`，**
所以可以用同一份评测集做 A/B 对比（见 `eval/compare_engines.py`），
而不是"换框架之后不知道有没有变差"。

## 两个引擎的实际差异（实测，不是宣传）

1. **状态管理**：手写版用局部变量 + list 追加；图版用 `TypedDict` 状态 +
   `add_messages` 式的显式字段，每一步的输入输出都是可序列化的状态快照。
2. **分支表达**：手写版的"要不要熔断"是循环体里的 `if ... break`；
   图版是 `add_conditional_edges` 里的路由函数，**终止逻辑变成一个可单测的纯函数**
   （见 `route_after_act`），不再埋在循环里。
3. **可观测性**：图版每一步都会产生一个 node 级的 trace（LangGraph 的
   stream 模式天然支持），手写版需要自己在每个分支手动打点。

## 为什么默认不用它

`--engine handwritten` 是默认值：它零额外依赖，CI 里跑得最快，
而且作为"我读过框架源码"的证据比"我会调框架"更硬。
LangGraph 引擎是**并行的第二条实现**，用来证明同一套抽象换框架也能落地。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Awaitable, Callable, Literal, TypedDict

import logging

from .agent import (
    BLANK_STRIKES,
    CLOSING_MAX_STEPS,
    HOSTILE_SETTLE_SECONDS,
    LOOP_STOP_AT,
    LOOP_WARN_AT,
    MAX_CONSECUTIVE_FAILURES,
    LOGIN_NOTIFY_EVERY,
    OSCILLATION_NOTIFY_EVERY,
    OSCILLATION_WINDOW,
    STALL_STOP_AT,
    STALL_WARN_AT,
    SYSTEM_PROMPT,
    HISTORY_WINDOW,
    RunResult,
    trim_history,
    LoginHandoff,
    StepCallback,
    _action_signature,
    login_blocked_warning,
    page_blocked_warning,
    loop_counts,
    oscillation_pages,
    oscillation_warning,
    page_fingerprint,
    update_stall,
)
from .browser import BrowserSession
from .config import Settings
from .grounding import MAX_GROUNDING_RETRIES, check_grounding
from .llm import LLMClient, LLMError, build_client, build_vlm_client
from .models import Action, StepRecord, Usage, parse_action
from .perception import perceive

log = logging.getLogger(__name__)


def _require_langgraph():
    """延迟导入 langgraph，缺了就抛一句能照着做的错。

    **为什么不能写在模块顶层。**
    这个文件被 `cli.py` 和 `api.py` 在模块级导入（`from .graph_agent import
    build_agent`）。如果在顶层写 `from langgraph.graph import END, StateGraph`，
    那么 langgraph 就从"跑图引擎才需要的可选依赖"变成了"import 这个包的
    任何一个入口都需要的硬依赖"：

      - 容器里按 requirements.txt 装依赖，langgraph 是注释掉的（见该文件说明），
        于是 `python -m uvicorn bagent.api:app` 在 import 阶段就
        ModuleNotFoundError，容器秒退、探活必失败 —— 而报错信息里
        只字不提"是你少装了一个可选依赖"。
      - 更糟的是它会让 `--engine handwritten` 也跑不起来：
        手写循环根本不需要 langgraph，却被它的 import 连坐。

    所以导入推迟到真正建图那一刻，并且把补救办法写进异常消息。
    """
    try:
        from langgraph.graph import END, StateGraph
    except ImportError as exc:  # pragma: no cover - 取决于环境
        raise RuntimeError(
            "LangGraph 引擎需要 langgraph，但当前环境没有安装。"
            "请执行 `pip install langgraph`（或改用 `--engine handwritten`，"
            "手写循环零额外依赖）。"
        ) from exc
    return END, StateGraph

# 阈值从 agent.py 导入，保证两个引擎「什么时候停」完全一致。
# 这里只做 re-export，方便测试与外部引用。
__all__ = [
    "AgentState",
    "LangGraphReActAgent",
    "build_agent",
    "route_after_act",
    "route_after_reason",
    "LOOP_STOP_AT",
    "LOOP_WARN_AT",
    "MAX_CONSECUTIVE_FAILURES",
]


class AgentState(TypedDict, total=False):
    """图的状态。

    LangGraph 只会在节点之间**传递声明过的键**，所以这里必须显式声明
    每一个需要跨节点流转的字段。`_state` / `_action` 这类以 `_` 开头的是
    本图的内部字段，图外部不关心，但同样必须声明——这是踩过的坑：
    第一版把它们当成"节点局部变量"写在了节点里，结果 reason 节点拿到的
    永远是空值（LangGraph 不会保留未声明的键）。

    刻意用扁平的基础类型（list[str] / int / bool）而不是自定义对象，
    是为了以后加 checkpointer 做断点续跑时，状态天然可序列化。
    """

    # 输入
    task: str
    start_url: str
    max_steps: int
    # 运行时
    step: int
    history: list[str]
    sig_trail: list[str]
    consecutive_failures: int
    prefer_vision: bool
    # 停滞检测（与手写引擎同一套规则，见 agent.py 的 page_fingerprint）。
    # `last_action` 存上一步真正执行的动作，供本步判断"它有没有带来新信息"；
    # 这几个字段必须在这里显式声明，否则 LangGraph 会在节点间丢掉它们 ——
    # 和 grounding_retries 踩过的是同一个坑（未声明的键不保留）。
    prev_fp: str
    last_action: object
    stall_count: int
    closing: bool
    # 振荡检测：需要**一整个窗口**的指纹历史（停滞检测只要上一帧），
    # 所以这里单独存序列。`osc_notified_at` 是上一次注入振荡警告的步号，
    # 用来节流 —— 模型不听时不能每步都重发，否则 prompt 会越长越长。
    fp_history: list[str]
    osc_notified_at: int
    # 本步有没有因"振荡"提醒过 / 当时窗口里几个页面（0 = 没提醒）。
    # 记账用，不参与判定。手写引擎里是循环局部变量 `osc_warned`。
    osc_pages: int
    closing_steps: int
    # 节点内部流转
    page_state: object
    action: object
    # 输出
    finished: bool
    answer: str
    error: str
    records: list[dict]
    # 证据锚定（与手写引擎同一套口径，见 grounding.py）。这三个字段必须在图里
    # 显式声明，否则 finish 被退回后 retries 会每步归零 → 退回次数无限，卡死。
    grounded: bool
    grounding_note: str
    grounding_retries: int
    # 登录墙：与手写引擎同一套语义（先登录，再找内容）。
    #   login_done        —— 人已经登录过一次（不再重复打扰）
    #   login_notified_at —— 上一次"没人能登录"的提示是第几步（节流用）
    #   just_logged_in    —— 本节点刚完成登录交接，需要**重新感知**一次
    login_done: bool
    login_notified_at: int
    just_logged_in: bool
    # 已看到过几次"被清空的空白页"。累计值，判据见 agent.BLANK_STRIKES。
    # 必须在图里显式声明：**未声明的键 LangGraph 会直接丢掉**（这个坑上面记过两次）。
    blank_hits: int
    # 登录交接**是否已经问过人**（不论成没成）。只看 login_done 会让
    # "放弃登录"之后每一步都再问一遍 —— 详见 perceive 节点里的说明。
    handoff_tried: bool
    # reason 节点解析失败时，模型**原始输出**的文本。必须声明才能在节点间
    # 传到 act（未声明的键 LangGraph 会直接丢掉 —— 上面已经踩过两次）。
    # act 节点靠它把原文写进 StepRecord.raw_output。
    raw_output: str


def _note(state: AgentState, text: str) -> list[str]:
    """往历史里追加一条，并裁到最近 12 条（与手写引擎一致，控制 prompt 长度）。"""
    return trim_history(list(state.get("history") or []) + [text])


# ---------------------------------------------------------------------------
# 路由函数 —— 终止逻辑是纯函数，可以单独写单测，不用起浏览器
# ---------------------------------------------------------------------------
def route_after_reason(
    state: AgentState,
) -> Literal["act", "perceive", "end"]:
    """reason 节点之后该去哪。

    - 模型调用失败写入了 error → 直接终止
    - 其余（含"输出非法 JSON 需要重试"）→ 都进 act

    注意：非法 JSON 的情况**故意也走 act**，让 act 节点统一处理
    「本轮没有可执行动作」——这样终止逻辑只集中在 `route_after_act` 一处，
    不会出现"有些终止判断在 reason、有些在 act"的割裂。
    """
    if state.get("error"):
        return "end"
    return "act"


def route_after_perceive(state: AgentState) -> Literal["perceive", "reason"]:
    """perceive 节点之后要不要**再看一眼页面**。

    只有一种情况需要：刚才完成了登录交接。登录改变的是"能看到什么"，
    不重新感知就继续推理，模型手上的还是登录前那一帧 ——
    那正是"登录了却还在找登录前的内容"的来源。
    """
    if state.get("just_logged_in"):
        return "perceive"
    return "reason"


def route_after_act(state: AgentState) -> Literal["perceive", "end"]:
    """act 节点之后该去哪 —— 整个图里唯一的终止判定点。

    把它抽成独立函数的好处：**熔断规则可以被直接单测**。
    手写引擎里这三条规则埋在 `for` 循环的 `break` 里，要测就得跑完整任务。
    """
    if state.get("finished") or state.get("error"):
        return "end"
    if state.get("step", 0) >= state.get("max_steps", 0):
        return "end"
    if state.get("consecutive_failures", 0) >= MAX_CONSECUTIVE_FAILURES:
        return "end"
    return "perceive"


class LangGraphReActAgent:
    """与 `ReActAgent` 同接口，可以互换（`build_agent()` 按 engine 选）。"""

    engine_name = "langgraph"

    def __init__(
        self,
        settings: Settings,
        llm: LLMClient | None = None,
        on_step: StepCallback | None = None,
    ) -> None:
        self.settings = settings
        self.llm = llm or build_client(settings)
        self.on_step = on_step
        self.login_handoff: LoginHandoff | None = None
        # 真人能完成登录的落点（站点首页）。交接前先导航到这里 ——
        # 当前页可能是 about:blank，交出空白页等于空转。理由见 agent.run 的文档。
        self.pre_login_url: str | None = None
        # 与手写引擎一致：自己造视觉客户端并传给 perceive，
        # 否则 `vlm is not None` 恒为假，视觉描述这条降级路径等于没接。
        self.vlm: LLMClient | None = None
        if getattr(settings, "vlm_enabled", False):
            try:
                self.vlm = build_vlm_client(settings)
            except Exception as exc:
                log.warning("视觉模型客户端创建失败，本次只走 DOM 通道: %s", exc)
        self._graph = self._build_graph()

    # ------------------------------------------------------------------
    # 建图
    # ------------------------------------------------------------------
    def _build_graph(self):
        END, StateGraph = _require_langgraph()
        g = StateGraph(AgentState)
        g.add_node("perceive", self._node_perceive)
        g.add_node("reason", self._node_reason)
        g.add_node("act", self._node_act)

        g.set_entry_point("perceive")
        # 登录交接之后必须**重新感知**：登录前的那一帧是登录页，
        # 直接拿它去问模型"下一步做什么"，等于让它对着登录页继续规划。
        g.add_conditional_edges(
            "perceive", route_after_perceive,
            {"perceive": "perceive", "reason": "reason"},
        )
        g.add_conditional_edges(
            "reason", route_after_reason, {"act": "act", "perceive": "perceive", "end": END}
        )
        g.add_conditional_edges(
            "act", route_after_act, {"perceive": "perceive", "end": END}
        )
        return g.compile()

    # ------------------------------------------------------------------
    # 三个节点
    # ------------------------------------------------------------------
    async def _node_perceive(self, state: AgentState) -> dict:
        session: BrowserSession = self._session
        step = state.get("step", 0) + 1
        st = await perceive(
            session.page,
            self.settings,
            step=step,
            run_dir=self._run_dir,
            prefer_vision=state.get("prefer_vision", False),
            vlm=self.vlm,
        )

        # ---------- 停滞检测：动作各异，但页面一字未变 ----------
        # 与手写引擎共用同一批常量与函数（见 agent.py）。放在 perceive 节点
        # 而不是 reason 节点，是为了让警告能进到**本轮**prompt —— 放到 act 之后
        # 就要等到下一轮才生效，白白多烧一步。
        #
        # `last_action` 在这里**读出并清空**：它描述的是"上一步那个动作"，
        # 一旦被本步消费就不该再留着，否则 finish 被退回这类没产生动作的路径
        # 会让它变陈旧，把停滞计数记到错误的那一步头上。
        pending = state.get("last_action")
        cur_fp = page_fingerprint(st)
        stall_count = update_stall(
            state.get("prev_fp", ""),
            cur_fp,
            getattr(pending, "action", None),
            state.get("stall_count", 0),
        )

        history = list(state.get("history") or [])
        closing = state.get("closing", False)
        closing_steps = state.get("closing_steps", 0)

        # ---------- 被挡住：先登录，再找内容（与手写引擎同一套语义）----------
        # 两个触发源，处置相同、理由分开：登录墙 / 累计 2 帧空白页。
        # 具体判据与实测过程见 agent.py 的 BLANK_STRIKES 与 page_blocked_warning。
        blank_hits = state.get("blank_hits", 0) + (1 if st.is_blank_page else 0)
        out_handoff_tried = state.get("handoff_tried", False)
        blocked_reason = ""
        if st.is_login_wall:
            blocked_reason = st.login_reason
        elif blank_hits >= BLANK_STRIKES:
            blocked_reason = st.blank_reason

        if blocked_reason:
            # 只看 `login_done` 是不够的：它只在交接**成功**时才置位，
            # 于是人点"放弃登录"之后每一步都会再问一遍。判据改成"问过没有"。
            if not state.get("handoff_tried") and self.login_handoff is not None:
                out_handoff_tried = True
                landing = st.url
                hopeless = ""
                # 空白页触发时先换到站点首页，**但必须确认那一页真的能用** ——
                # 实测 BOSS直聘 在有头模式下载入首页 2 秒后仍会自己跳回
                # about:blank（页内 browser-check-v2.js），人在那种窗口里也登不了。
                # 换过去还是空白 → 不要弹"请去登录"，那是在骗人。
                # 详细说明见 agent.py 同位置。
                if st.is_blank_page and self.pre_login_url \
                        and self.pre_login_url != st.url:
                    await session.execute(Action(action="goto", url=self.pre_login_url))
                    landing = (session.page.url if session.page else "") or self.pre_login_url
                    # 必须等一会儿再复采：站点可能是"先渲染、2 秒后自毁"，
                    # goto 一返回就采样会采到它还活着的那一瞬间。见 HOSTILE_SETTLE_SECONDS。
                    await asyncio.sleep(HOSTILE_SETTLE_SECONDS)
                    retry = await perceive(
                        session.page, self.settings, step=step,
                        run_dir=self._run_dir, vlm=self.vlm,
                    )
                    if retry.is_blank_page:
                        hopeless = (
                            f"连站点首页也变成了空白页（{retry.url}）——"
                            f"该站点在阻止自动化访问，人工登录也解决不了"
                        )
                if hopeless:
                    blocked_reason = hopeless
                else:
                    resumed = False
                    try:
                        resumed = await self.login_handoff(landing, blocked_reason)
                    except Exception as exc:
                        log.warning("登录交接失败，按未登录处理: %s", exc)
                if resumed:
                    await session.execute(
                        Action(action="goto", url=state.get("start_url", ""))
                    )
                    rec = StepRecord(
                        step=step, action=None, raw_action="登录交接", ok=True,
                        message=f"已完成登录，回到任务入口 {state.get('start_url', '')}",
                        url_after=session.page.url if session.page else "",
                    ).model_dump()
                    return {
                        "login_done": True,
                        "just_logged_in": True,
                        "handoff_tried": True,
                        # 登录后重新计数：旧的那几帧空白页不再代表"现在还被拦着"。
                        "blank_hits": 0,
                        # 清空所有基于"登录前页面"的判据状态，
                        # 否则刚登录就被旧指纹判成"又在振荡"。
                        "fp_history": [],
                        "prev_fp": "",
                        "stall_count": 0,
                        "closing": False,
                        "closing_steps": 0,
                        "consecutive_failures": 0,
                        "sig_trail": [],
                        "records": list(state.get("records") or []) + [rec],
                        "history": _note(
                            state,
                            f"第 {step} 步: 登录已完成，已重新打开任务入口。"
                            f"现在去找内容 —— 顺序是**先登录后找内容**，不要反过来。",
                        ),
                    }

            notified_at = state.get("login_notified_at", 0)
            # 与手写引擎一致：`0` 表示"还没提醒过"，第一次必须提醒，不能节流掉。
            if notified_at == 0 or step - notified_at >= LOGIN_NOTIFY_EVERY:
                # 两种拦截的措辞不能混用（指错方向比不说更糟）。
                history.append(
                    login_blocked_warning(blocked_reason)
                    if st.is_login_wall
                    else page_blocked_warning(blocked_reason)
                )
                out_login_at = step
            else:
                out_login_at = state.get("login_notified_at", 0)
        else:
            out_login_at = state.get("login_notified_at", 0)

        if stall_count >= STALL_STOP_AT:
            if not closing:
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
            # 页面重新有了变化 → 动作生效了，退出收束。
            # `closing_steps` 不在这里清零 —— 它数的是"模型拒答次数"，见 agent.py。
            closing = False
        elif stall_count >= STALL_WARN_AT:
            history.append(
                f"⚠ 警告：页面已经连续 {stall_count} 步没有任何变化，"
                f"说明当前这个方向得不到新信息。"
                f"请换一个明显不同的做法，或者直接调用 finish 给出你目前的结论。"
            )

        # ---------- 振荡检测：在已经去过的几个页面之间来回走 ----------
        # 与手写引擎共用**同一个判据函数**（oscillation_pages）和**同一段提示文本**
        # （oscillation_warning）—— 原来两边各抄一份、靠注释声明"逐字一致"，
        # 结果真的漂移过；改成共用函数后，漂移在结构上就不可能了。
        #
        # 只发警告、不进收束、不拦动作 —— 理由见 agent.py 常量区的实测说明
        # （该信号在已收敛运行上也会触发，约一半、含 11 条 normal，
        # 拿去熔断会连 t04 的正常往返一起打掉）。
        fp_history = list(state.get("fp_history") or [])
        fp_history.append(cur_fp)
        osc_notified_at = state.get("osc_notified_at", 0)
        # 本步有没有因"振荡"提醒过（每步重置，写进 trace 供事后核对）
        osc_warned = 0
        # 登录墙上不发振荡警告：上面已经给过更具体的解释，同时喂两句会互相打架。
        if not closing and not st.is_login_wall and step - osc_notified_at >= OSCILLATION_NOTIFY_EVERY:
            osc_pages = oscillation_pages(fp_history)
            if osc_pages is not None:
                osc_notified_at = step
                osc_warned = osc_pages
                history.append(oscillation_warning(osc_pages))

        out: dict = {
            "step": step,
            "page_state": st,
            "prefer_vision": False,
            "last_action": None,
            "prev_fp": cur_fp,
            "stall_count": stall_count,
            "closing": closing,
            "closing_steps": closing_steps,
            "fp_history": fp_history[-24:],
            "osc_notified_at": osc_notified_at,
            "osc_pages": osc_warned,
            "login_notified_at": out_login_at,
            "blank_hits": blank_hits,
            # 必须回写：不回写的话下一次 perceive 读到的还是 False，
            # 又变成"每一步问一遍"。这个键是持久化在图状态里的。
            "handoff_tried": out_handoff_tried,
            # 正常路径一定要把它写回 False：否则下次路由还会再"重新感知"一次，
            # 形成 perceive → perceive 的自环。
            "just_logged_in": False,
        }
        if len(history) != len(state.get("history") or []):
            out["history"] = trim_history(history)
        return out

    async def _node_reason(self, state: AgentState) -> dict:
        st = state.get("page_state")
        prompt = self._build_prompt(state.get("task", ""), st, state.get("history") or [])
        try:
            # 与手写引擎一致：走 achat，别让同步 SDK 卡住事件循环
            raw = await self.llm.achat(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                json_mode=True,
            )
        except LLMError as exc:
            log.error("模型调用失败，终止任务: %s", exc)
            return {"error": str(exc)}

        action, parse_err = parse_action(raw)
        if action is None:
            # 非法 JSON 不崩，当成一次失败反馈回去；连续 3 次才终止
            n = state.get("consecutive_failures", 0) + 1
            out: dict = {
                "history": _note(
                    state,
                    f"第 {state.get('step')} 步: 输出格式不合法（{parse_err}），"
                    f"请只输出合法 JSON",
                ),
                "consecutive_failures": n,
                # 交给 act 节点写进 trace：复盘时要能看见"错成什么样"，
                # 而不只是"这一步错了"。
                "raw_output": (raw or "")[:300],
            }
            if n >= MAX_CONSECUTIVE_FAILURES:
                out["error"] = f"模型连续输出非法格式: {parse_err}"
            return out

        # 解析成功 → 把上一步可能留下的原文清掉，避免它被写进下一条记录
        return {"action": action, "raw_output": ""}

    async def _node_act(self, state: AgentState) -> dict:
        session: BrowserSession = self._session
        step = state.get("step", 0)
        records = list(state.get("records") or [])

        action: Action | None = state.get("action")
        if action is None:
            # reason 节点没能解析出动作（已写入 consecutive_failures / history）。
            # 这一轮**真的花了一次模型调用**，所以必须补一条记录 —— 否则
            # records 里的 step 会跳号（和 ReActAgent 里犯过的是同一个错），
            # 前端时间线看起来像"某一步凭空消失"，而那恰恰是排障最需要的一步。
            records.append(
                StepRecord(
                    step=step,
                    action=None,
                    raw_action="(格式错误)",
                    # 与手写引擎一致：解析失败时把原文留档，否则复盘只看得见
                    # "这一步错了"、看不见"错成什么样"。
                    raw_output=(state.get("raw_output") or "")[:300],
                    ok=False,
                    message="模型输出不是合法 JSON",
                    url_after=(state.get("page_state").url
                               if state.get("page_state") else ""),
                    # 与手写引擎对齐的观测字段：停滞/振荡的判据输入是页面序列，
                    # 缺了它们就没法从 trace 直接回答"当时页面变没变"。
                    page_fp=state.get("prev_fp", ""),
                    stall_count=state.get("stall_count", 0),
                    osc_pages=state.get("osc_pages", 0),
                ).model_dump()
            )
            if self.on_step:
                self.on_step(step, None, False, "输出格式不合法")
            return {"records": records}

        st = state.get("page_state")

        # --- finish：任务结束（先过证据核对，见 grounding.py）---
        if action.action == "finish":
            verdict = check_grounding(
                action.answer or "", action.evidence or "", st.body_text,
                state.get("task", ""),
            )
            retries = state.get("grounding_retries", 0)
            if not verdict.ok and retries < MAX_GROUNDING_RETRIES:
                # 对不上就退回重做，而不是直接采纳 —— 与手写引擎完全一致。
                # 这里**故意不写 finished**，让 route_after_act 走回 perceive。
                retries += 1
                message = (
                    f"finish 被退回（{retries}/{MAX_GROUNDING_RETRIES}）：{verdict.reason}"
                )
                records.append(
                    StepRecord(
                        step=step, action=action, ok=False, message=message,
                        url_after=st.url,
                    ).model_dump()
                )
                if self.on_step:
                    self.on_step(step, action, False, message)
                return {
                    "records": records,
                    "grounding_retries": retries,
                    "history": _note(
                        state,
                        f"第 {step} 步: {message}。"
                        "请重新读一遍「页面正文摘要」，把支撑结论的原文**原样复制**到 "
                        "evidence，并确认 answer 里的每个人名 / 数字 / 标题都出现在这段"
                        "原文里，再调用 finish。",
                    ),
                    # 清掉已消费的动作，避免退回后 LangGraph 沿用旧 action。
                    "action": None,
                }

            grounded = verdict.grounded
            note = verdict.reason
            if not verdict.ok:
                note = (
                    f"已重试 {retries} 次仍未通过证据核对，按当前答案收尾：{verdict.reason}"
                )
            elif not verdict.checked:
                note = f"未做证据核对（{verdict.reason}）"
            records.append(
                StepRecord(
                    step=step, action=action, ok=True,
                    message="任务结束" + ("（证据已核对）" if grounded else f"（{note}）"),
                    url_after=st.url,
                ).model_dump()
            )
            if self.on_step:
                self.on_step(
                    step, action, True,
                    "任务结束（证据已核对）" if grounded else "任务结束",
                )
            return {
                "finished": True,
                "answer": (action.answer or "").strip(),
                "records": records,
                "grounded": grounded,
                "grounding_note": note,
                "grounding_retries": retries,
            }

        # --- 收束阶段：只接受结论，不接受"再试一下" ---
        # 与手写引擎同一套规则。只喊话不够用：实测重放了 t06 的一次失败
        # （runs/20260921-224847-t06），收束指令注入之后模型回手就是
        # `extract` / `extract` / `click#1`，一直磨到步数上限。
        # 所以这里**真的拦住**：非 finish 动作不执行，只回一条明确的拒绝，
        # 拒绝累积到 CLOSING_MAX_STEPS 就熔断。数的是"模型拒答次数"。
        if state.get("closing") and action.action != "finish":
            stall = state.get("stall_count", 0)
            steps = state.get("closing_steps", 0) + 1
            message = (
                f"收束阶段不接受 `{action.action}`：页面已连续 {stall} 步无变化，"
                f"没有新信息可获取。现在**只剩 finish 一个选项**。"
            )
            records.append(
                StepRecord(
                    step=step, action=action, ok=False, message=message,
                    url_after=st.url, page_fp=state.get("prev_fp", ""), stall_count=stall,
                    osc_pages=state.get("osc_pages", 0),
                ).model_dump()
            )
            if self.on_step:
                self.on_step(step, action, False, message)
            out: dict = {
                "records": records,
                "closing_steps": steps,
                "action": None,
            }
            if steps >= CLOSING_MAX_STEPS:
                out["error"] = (
                    f"页面连续 {stall} 步毫无变化（判定为原地打转），"
                    f"收束后又连续 {steps} 次拒绝给出结论，熔断"
                )
            else:
                out["history"] = _note(
                    state,
                    f"第 {step} 步: {message}"
                    '请立刻输出 {"action": "finish", "answer": "..."} —— '
                    "把你能确定的结论写进 answer 就够了。",
                )
            return out

        # --- extract：读正文，不碰浏览器 ---
        if action.action == "extract":
            outcome_ok, message = True, f"已读取页面正文（{len(st.body_text)} 字）"
            prefer_vision = False
        else:
            outcome = await session.execute(action, confirm=self._confirm)
            outcome_ok, message = outcome.ok, outcome.message
            prefer_vision = outcome.vision_hint

        records.append(
            StepRecord(
                step=step, action=action, ok=outcome_ok, message=message,
                url_after=session.page.url if session.page else "",
                screenshot_path=st.screenshot_path,
                page_fp=state.get("prev_fp", ""), stall_count=state.get("stall_count", 0),
                osc_pages=state.get("osc_pages", 0),
            ).model_dump()
        )
        if self.on_step:
            self.on_step(step, action, outcome_ok, message)

        status = "成功" if outcome_ok else "失败"
        desc = f"第 {step} 步: {self._describe(action)} → {status}: {message}"
        if not outcome_ok:
            desc += "（不要重复这个动作）"
        history = _note(state, desc)

        out: dict = {
            "history": history,
            "records": records,
            "prefer_vision": prefer_vision,
            "consecutive_failures": 0 if outcome_ok else state.get("consecutive_failures", 0) + 1,
            # 记下这一步的动作，供**下一步的 perceive 节点**判断它有没有带来新信息
            # （停滞检测，见 _node_perceive）。只在真正执行了动作的路径上赋值，
            # 所以 finish 被退回 / JSON 非法那两条提前 return 的路径不会污染它，
            # 与手写引擎里 `last_action = action` 的位置一一对应。
            "last_action": action,
            # 清掉已消费的动作。否则下一步若模型吐了非法 JSON，
            # reason 不会写 action，LangGraph 会沿用上一步的值 → 同一步被执行两次。
            "action": None,
        }

        # --- 动作指纹循环检测（与手写引擎同一套规则）---
        sig = _action_signature(action)
        trail = list(state.get("sig_trail") or []) + [sig]
        # 与手写引擎共用 loop_counts。**判据是末尾连续次数，不是累计** ——
        # 累计判据误杀过 t04（5 次 scroll 分散在第 4/6/11/12/14 步，
        # 每一步之间都在获得新信息）。完整推导见 agent.py 主循环那一段。
        repeats, consecutive = loop_counts(trail, sig)
        out["sig_trail"] = trail
        if consecutive >= LOOP_STOP_AT:
            out["error"] = (
                f"检测到原地打转：动作 {sig} 已连续 {consecutive} 次"
                f"（整轮累计出现 {repeats} 次）且无进展，已熔断"
            )
        elif consecutive >= LOOP_WARN_AT:
            out["history"] = (
                list(out["history"])
                + [
                    f"⚠ 严重警告：动作 `{sig}` 已经**连续**出现 {consecutive} 次"
                    f"（整轮累计 {repeats} 次），"
                    f"页面没有任何进展。**禁止再执行这个动作**。"
                    f"你现在必须二选一：(a) 换一个完全不同的元素编号；"
                    f"(b) 直接调用 finish 给出结论。"
                    f"如果你已经能从页面正文里看到答案，请立刻 finish。"
                ]
            )[:HISTORY_WINDOW]
        return out

    # ------------------------------------------------------------------
    # 与手写引擎同签名的 run()
    # ------------------------------------------------------------------
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
        pre_login_url: str | None = None,
    ) -> RunResult:
        # 登录交接是**每次运行**的入参（不是构造参数）：谁能登录、怎么登录
        # 由调用方决定，引擎只认"返回 True 表示已登录"这个契约。
        self.login_handoff = login_handoff
        self.pre_login_url = pre_login_url
        max_steps = max_steps or self.settings.max_steps
        started = time.time()
        run_dir.mkdir(parents=True, exist_ok=True)

        self._session = session
        self._run_dir = run_dir
        self._confirm = confirm

        await session.execute(Action(action="goto", url=start_url), confirm=confirm)

        final = await self._graph.ainvoke(
            {
                "task": task,
                "start_url": start_url,
                "max_steps": max_steps,
                "step": 0,
                "history": [],
                "sig_trail": [],
                "consecutive_failures": 0,
                "prefer_vision": False,
                # 停滞检测初值（见 _node_perceive）。空指纹代表"还没有上一帧可比"，
                # update_stall 会原样返回计数 —— 第 1 步不可能判停滞。
                "prev_fp": "",
                "last_action": None,
                "stall_count": 0,
                "closing": False,
                "closing_steps": 0,
                # 振荡检测初值：历史为空 → 窗口还没凑够，判不出振荡（见 osc_notified_at 的说明）。
                "fp_history": [],
                "osc_notified_at": 0,
                # 本步振荡提醒的记账值（0 = 没提醒），每步由 perceive 覆盖
                "osc_pages": 0,
                "finished": False,
                "answer": "",
                "error": "",
                "records": [],
                "grounded": False,
                "grounding_note": "",
                "grounding_retries": 0,
                # 登录 / 拦截相关初值。这些键即使不写也能跑（节点里都是 .get() 带默认值），
                # 但显式写出来才能让"图的状态契约"一眼看全 ——
                # 之前已经因为"忘了声明状态键"踩过两次静默丢数据。
                "login_done": False,
                "login_notified_at": 0,
                "just_logged_in": False,
                "blank_hits": 0,
                "handoff_tried": False,
                "raw_output": "",
            }
        )

        error = final.get("error", "")
        if not error and not final.get("finished"):
            error = f"达到最大步数上限（{max_steps}），任务未完成"

        records = [StepRecord(**r) for r in (final.get("records") or [])]
        elapsed = time.time() - started
        usage: Usage = self.llm.usage
        # 与 ReActAgent 保持同一口径：离线替身没有真实计费，成本恒为 0。
        offline = bool(getattr(self.settings, "mock", False))
        result = RunResult(
            task=task,
            start_url=start_url,
            success=bool(final.get("finished")),
            finished=bool(final.get("finished")),
            answer=final.get("answer", ""),
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
            grounded=bool(final.get("grounded")),
            grounding_note=final.get("grounding_note", ""),
            grounding_retries=int(final.get("grounding_retries", 0) or 0),
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
        from .agent import ReActAgent

        return ReActAgent._describe(action)

    @staticmethod
    def _dump(result: RunResult, path: Path) -> None:
        from .agent import ReActAgent

        ReActAgent._dump(result, path)


def build_agent(settings: Settings, llm=None, on_step=None):
    """按 settings.engine 选引擎。

    默认 `handwritten`：零额外依赖、CI 最快、也是"我懂原理"的证据。
    需要对比框架实现时显式传 `--engine langgraph`。
    """
    engine = (getattr(settings, "engine", "handwritten") or "handwritten").lower()
    if engine == "langgraph":
        return LangGraphReActAgent(settings, llm=llm, on_step=on_step)
    from .agent import ReActAgent

    return ReActAgent(settings, llm=llm, on_step=on_step)

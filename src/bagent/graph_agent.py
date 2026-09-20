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

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Awaitable, Callable, Literal, TypedDict

import logging

from .agent import (
    LOOP_STOP_AT,
    LOOP_WARN_AT,
    MAX_CONSECUTIVE_FAILURES,
    SYSTEM_PROMPT,
    RunResult,
    StepCallback,
    _action_signature,
)
from .browser import BrowserSession
from .config import Settings
from .llm import LLMClient, LLMError, build_client
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
    # 节点内部流转
    page_state: object
    action: object
    # 输出
    finished: bool
    answer: str
    error: str
    records: list[dict]


def _note(state: AgentState, text: str) -> list[str]:
    """往历史里追加一条，并裁到最近 12 条（与手写引擎一致，控制 prompt 长度）。"""
    return (list(state.get("history") or []) + [text])[-12:]


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
        g.add_edge("perceive", "reason")
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
        )
        return {"step": step, "page_state": st, "prefer_vision": False}

    async def _node_reason(self, state: AgentState) -> dict:
        st = state.get("page_state")
        prompt = self._build_prompt(state.get("task", ""), st, state.get("history") or [])
        try:
            raw = self.llm.chat(
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
            }
            if n >= MAX_CONSECUTIVE_FAILURES:
                out["error"] = f"模型连续输出非法格式: {parse_err}"
            return out

        return {"action": action}

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
                    ok=False,
                    message="模型输出不是合法 JSON",
                    url_after=(state.get("page_state").url
                               if state.get("page_state") else ""),
                ).model_dump()
            )
            if self.on_step:
                self.on_step(step, None, False, "输出格式不合法")
            return {"records": records}

        st = state.get("page_state")

        # --- finish：任务结束 ---
        if action.action == "finish":
            records.append(
                StepRecord(
                    step=step, action=action, ok=True, message="任务结束",
                    url_after=st.url,
                ).model_dump()
            )
            if self.on_step:
                self.on_step(step, action, True, "任务结束")
            return {
                "finished": True,
                "answer": (action.answer or "").strip(),
                "records": records,
            }

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
            # 清掉已消费的动作。否则下一步若模型吐了非法 JSON，
            # reason 不会写 action，LangGraph 会沿用上一步的值 → 同一步被执行两次。
            "action": None,
        }

        # --- 动作指纹循环检测（与手写引擎同一套规则）---
        sig = _action_signature(action)
        trail = list(state.get("sig_trail") or []) + [sig]
        repeats = trail.count(sig)
        out["sig_trail"] = trail
        if repeats >= LOOP_STOP_AT:
            out["error"] = (
                f"检测到原地打转：动作 {sig} 重复 {repeats} 次且无进展，已熔断"
            )
        elif repeats >= LOOP_WARN_AT:
            out["history"] = (
                list(out["history"])
                + [
                    f"⚠ 严重警告：你已经连续 {repeats} 次执行 `{sig}`，"
                    f"页面没有任何进展。**禁止再执行这个动作**。"
                    f"你现在必须二选一：(a) 换一个完全不同的元素编号；"
                    f"(b) 直接调用 finish 给出结论。"
                    f"如果你已经能从页面正文里看到答案，请立刻 finish。"
                ]
            )[-12:]
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
    ) -> RunResult:
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
                "finished": False,
                "answer": "",
                "error": "",
                "records": [],
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

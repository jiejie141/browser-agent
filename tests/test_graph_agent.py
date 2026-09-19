"""LangGraph 引擎的单测。

重点测**终止逻辑**——这是两个引擎最本质的差异：
手写版把三条熔断规则埋在 `for` 循环的 `if ... break` 里，要测就得跑完整任务；
图版把它们抽成了纯函数 `route_after_act`，可以直接喂状态字典断言返回值。

这就是「把循环显式建模成状态图」在工程上的真实收益：
**终止规则从不可测变成了可测。**
"""

from __future__ import annotations

import pytest

from bagent.graph_agent import (
    MAX_CONSECUTIVE_FAILURES,
    LOOP_STOP_AT,
    LOOP_WARN_AT,
    AgentState,
    route_after_act,
    route_after_reason,
)


def _st(**kw) -> AgentState:
    base: AgentState = {
        "task": "t",
        "start_url": "https://example.com",
        "max_steps": 20,
        "step": 1,
        "history": [],
        "sig_trail": [],
        "consecutive_failures": 0,
        "prefer_vision": False,
        "finished": False,
        "answer": "",
        "error": "",
        "records": [],
    }
    base.update(kw)
    return base


# --- route_after_reason ---------------------------------------------------
def test_reason_goes_to_act_normally():
    assert route_after_reason(_st()) == "act"


def test_reason_ends_on_error():
    assert route_after_reason(_st(error="模型调用失败")) == "end"


def test_reason_illegal_json_still_goes_to_act():
    """非法 JSON 不直接终止，而是让 act 空转一轮再回到 perceive 重试。

    这样终止判断只集中在 route_after_act 一处，不会两处割裂。
    """
    assert route_after_reason(_st(consecutive_failures=1)) == "act"


# --- route_after_act ------------------------------------------------------
def test_act_continues_normally():
    assert route_after_act(_st()) == "perceive"


def test_act_ends_when_finished():
    assert route_after_act(_st(finished=True)) == "end"


def test_act_ends_on_error():
    assert route_after_act(_st(error="熔断")) == "end"


def test_act_ends_at_step_limit():
    assert route_after_act(_st(step=20, max_steps=20)) == "end"
    assert route_after_act(_st(step=19, max_steps=20)) == "perceive"


def test_act_ends_after_consecutive_failures():
    assert route_after_act(_st(consecutive_failures=MAX_CONSECUTIVE_FAILURES)) == "end"
    assert route_after_act(_st(consecutive_failures=MAX_CONSECUTIVE_FAILURES - 1)) == "perceive"


def test_thresholds_match_handwritten_engine():
    """两个引擎的阈值必须一致，否则 A/B 对比没有意义。"""
    from bagent import agent as hand

    assert MAX_CONSECUTIVE_FAILURES == 3
    assert LOOP_STOP_AT == hand.LOOP_STOP_AT == 5
    assert LOOP_WARN_AT == hand.LOOP_WARN_AT == 3


# --- 建图与引擎选择 -------------------------------------------------------
def test_build_agent_respects_engine_flag():
    from bagent.config import Settings
    from bagent.graph_agent import LangGraphReActAgent, build_agent
    from bagent.agent import ReActAgent

    st = Settings()
    st.engine = "handwritten"
    assert isinstance(build_agent(st, llm=object()), ReActAgent)

    st.engine = "langgraph"
    assert isinstance(build_agent(st, llm=object()), LangGraphReActAgent)


def test_graph_compiles_and_has_three_nodes():
    from bagent.config import Settings
    from bagent.graph_agent import LangGraphReActAgent

    a = LangGraphReActAgent(Settings(), llm=object())
    nodes = set(a._graph.get_graph().nodes)
    assert {"perceive", "reason", "act"} <= nodes

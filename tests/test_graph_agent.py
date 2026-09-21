"""LangGraph 引擎的单测。

重点测**终止逻辑**——这是两个引擎最本质的差异：
手写版把三条熔断规则埋在 `for` 循环的 `if ... break` 里，要测就得跑完整任务；
图版把它们抽成了纯函数 `route_after_act`，可以直接喂状态字典断言返回值。

这就是「把循环显式建模成状态图」在工程上的真实收益：
**终止规则从不可测变成了可测。**
"""

from __future__ import annotations

import inspect

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


class TestLoopCountsTellTheTruth:
    """熔断日志里的数字必须和判据对得上。

    原先两条引擎都这么写：

        repeats = trail.count(sig)          # ← 判据：整轮**累计**
        ... f"你已经连续 {repeats} 次执行 `{sig}`"   # ← 措辞：**连续**

    这两个不是一回事。用一次真实基线的三条熔断核，两条对不上
    （`runs/eval_20260921-145218.json`）：

    - `t05` 的 `click#2` 在第 1 / 5 / 7 / 13 / 18 步 —— 中间隔着
      `type` / `extract` / `scroll` / `click#43` / `type`，**不连续**；
    - `t06` 的 `click#1` 在第 3 / 8 / 9 / 12 / 17 步，同样不连续；
    - 只有 `t04` 的 `type#20` 是真连续（第 16 / 17 / 18 步）。

    也就是说模型读到的是一句与事实不符的历史，而它本来就是靠这句话
    被推着换策略的。这里钉住"两个数要分开算、且都要对"。

    注意：本组**不改判据**（仍是累计），只保证数字如实。
    把判据从"累计"改成"连续"会动熔断时机、进而动基线数字，
    那是另一个实验，见 `docs/实验记录.md`。
    """

    def test_consecutive_vs_total(self):
        from bagent.agent import loop_counts

        # t05 的真实形态：click#2 累计 5 次，但末尾只有 1 次是连续的
        # （第 1/5/7/13/18 步，中间隔着 type / extract / scroll / click#43）
        trail = ["click#2", "x", "click#2", "x", "click#2", "x", "click#2", "x", "click#2"]
        total, consecutive = loop_counts(trail, "click#2")
        assert total == 5, "这条会触发熔断（累计到 5）"
        assert consecutive == 1, "但它**不是**连续的 —— 日志不能说『连续 5 次』"

    def test_consecutive_tail_counted(self):
        from bagent.agent import loop_counts

        trail = ["click#1", "click#1", "click#1", "click#1"]
        total, consecutive = loop_counts(trail, "click#1")
        assert total == consecutive == 4

    def test_t04_real_shape(self):
        """t04：累计 5、末尾连续 3 —— 这条是"警告发了两次但没听"。"""
        from bagent.agent import loop_counts

        trail = ["type#20", "x", "type#20", "x", "x", "x"] + ["type#20"] * 3
        total, consecutive = loop_counts(trail, "type#20")
        assert total == 5 and consecutive == 3

    def test_missing_action_is_zero(self):
        from bagent.agent import loop_counts

        assert loop_counts(["click#1"], "click#9") == (0, 0)

    def test_empty_trail(self):
        from bagent.agent import loop_counts

        assert loop_counts([], "click#1") == (0, 0)

    def test_fuse_decision_still_uses_total(self):
        """判据没动 —— 别在"修措辞"里把行为也改了。"""
        from bagent import agent as hand

        src = inspect.getsource(hand.ReActAgent.run)
        assert "repeats >= LOOP_STOP_AT" in src, "熔断判据仍应基于累计次数"

    def test_both_engines_share_the_counter(self):
        """两个引擎必须共用同一个计数实现，否则又会出现口径漂移。"""
        from bagent import agent as hand
        from bagent import graph_agent as g

        assert g.loop_counts is hand.loop_counts

    def test_langgraph_act_node_uses_shared_counter(self):
        from bagent import graph_agent as g

        src = inspect.getsource(g.LangGraphReActAgent._node_act)
        assert "loop_counts(" in src
        assert "trail.count(sig)" not in src, "不该再自己 count 一遍"


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


class TestScrollSignatureCannotBeBypassed:
    """`scroll` 的指纹不能把滚动量算进去 —— 否则改个数字就绕过了熔断。

    这条是一次**模型层消融**抓出来的（不是想出来的）。强模型跑到 t06 时：

    ```
    20 步里 13 次 scroll，dy 依次
    -200 / -1000 / -500 / -2000 / -3000 / +2000 / -3000 / -5000 / -1000 / -10000 / -5000 …
    ```

    **每个都不一样** → 指纹各异 → `trail.count(sig)` 永远到不了 5 →
    熔断一次都没触发 → 20 步全跑完（`ok=True`、`error=""`）任务却没完成。

    换成方向签名后熔断点 = 第 7 步。代价也量过：18 次基线重算，
    只有那 3 次本来就熔断失败的运行会被命中，熔断点相同（18/18/17），
    **没有任何一次通过的运行受影响**。
    """

    def _act(self, **kw):
        from bagent.models import Action

        return Action(**kw)

    def test_scroll_amount_does_not_create_new_signature(self):
        from bagent.agent import _action_signature

        a = _action_signature(self._act(action="scroll", dy=-1000))
        b = _action_signature(self._act(action="scroll", dy=-10000))
        assert a == b == "scroll#up", "换滚动量不该算成新动作"

    def test_direction_is_still_distinguished(self):
        """向下找内容 / 再向上回看是合理动作，方向要分开。"""
        from bagent.agent import _action_signature

        assert _action_signature(self._act(action="scroll", dy=800)) == "scroll#down"
        assert _action_signature(self._act(action="scroll", dy=-800)) == "scroll#up"

    def test_zero_scroll_is_its_own_signature(self):
        from bagent.agent import _action_signature

        assert _action_signature(self._act(action="scroll", dy=0)) == "scroll"
        assert _action_signature(self._act(action="scroll", dy=None)) == "scroll"

    def test_real_t06_trace_gets_fused_under_new_rule(self):
        """用真实轨迹的 dy 序列复算：旧规则不熔断，新规则第 7 步熔断。

        注意 dy 为**负数是向上滚**（`scroll#up`），真实轨迹里多数是负值 ——
        这条初稿就是把方向写反了才红的，留个注记免得改回去。
        """
        from bagent.agent import _action_signature, LOOP_STOP_AT
        from bagent.models import Action

        acts = [
            Action(action="click", ref=28),
            Action(action="scroll", dy=-200),
            Action(action="scroll", dy=-1000),
            Action(action="scroll", dy=-500),
            Action(action="screenshot"),
            Action(action="scroll", dy=-2000),
            Action(action="scroll", dy=-3000),
            Action(action="scroll", dy=2000),
            Action(action="scroll", dy=-3000),
            Action(action="scroll", dy=-5000),
        ]
        trail = [_action_signature(a) for a in acts]

        # 旧规则：把 dy 写进指纹 → 每次都是"新动作"，永远到不了阈值。
        old_trail = [
            f"scroll#{a.dy}" if a.action == "scroll" else _action_signature(a) for a in acts
        ]
        assert not any(old_trail[: i + 1].count(s) >= LOOP_STOP_AT
                       for i, s in enumerate(old_trail)), "旧规则在这串里确实不该熔断"

        # 新规则：同一串里必须能认出"在原地滚动"。
        fuse = next(
            (i + 1 for i, s in enumerate(trail) if trail[: i + 1].count(s) >= LOOP_STOP_AT),
            None,
        )
        assert fuse is not None, "新规则必须能在这一串里认出原地滚动"
        assert fuse <= 7, f"应该在第 7 步就熔断，实际 {fuse}"

    def test_clicks_unchanged(self):
        """别的动作的指纹语义没被动过。"""
        from bagent.agent import _action_signature

        assert _action_signature(self._act(action="click", ref=7)) == "click#7"
        assert _action_signature(self._act(action="type", ref=3, text="abc")) == "type#3"
        assert _action_signature(self._act(action="extract")) == "extract"

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
    被推着换策略的。所以这里钉住"两个数要分开算、且都要对"。

    **后续（同一处代码，第二刀）**：判据本身也从"累计"改成了"连续"。
    改的理由不是"措辞要对上"这么简单，而是一次真实回归：
    累计判据会误杀"分散但正常"的探索（t04）。这一段历史在
    `test_fuse_decision_now_uses_consecutive` 的报告里，别删。
    """

    def test_consecutive_vs_total(self):
        from bagent.agent import loop_counts

        # t05 的真实形态：click#2 累计 5 次，但末尾只有 1 次是连续的
        # （第 1/5/7/13/18 步，中间隔着 type / extract / scroll / click#43）
        trail = ["click#2", "x", "click#2", "x", "click#2", "x", "click#2", "x", "click#2"]
        total, consecutive = loop_counts(trail, "click#2")
        assert total == 5, "整轮累计 5 次"
        assert consecutive == 1, (
            "但它**不是**连续的 —— 日志不能说『连续 5 次』；"
            "而且按现在的判据（连续）它**不该**熔断"
        )

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

    def test_fuse_decision_now_uses_consecutive(self):
        """判据已从"累计"改成"连续" —— 这是一次**回归**逼出来的，不是口味问题。

        把 scroll 指纹合并成方向之后跑基线，累计判据立刻误杀 t04：
        它的 5 次 scroll 分散在第 4/6/11/12/14 步，每一步之间都在获得新信息
        （点结果、翻页、读正文），旧日志却写着"末尾连续 1 次"。
        用累计判定"原地打转"是错的：那件事应该由**页面指纹**（停滞检测）负责，
        动作指纹这套只管"同一个动作真的连着重复"。

        所以这条断言不只是钉住实现，它钉住的是一个**语义决定**。
        """
        from bagent import agent as hand

        src = inspect.getsource(hand.ReActAgent.run)
        assert "consecutive >= LOOP_STOP_AT" in src, "熔断判据应基于末尾连续次数"
        assert "repeats >= LOOP_STOP_AT" not in src, "不应再看累计次数"

    def test_both_engines_share_the_counter(self):
        """两个引擎必须共用同一个计数实现，否则又会出现口径漂移。"""
        from bagent import agent as hand
        from bagent import graph_agent as g

        assert g.loop_counts is hand.loop_counts

    def test_both_engines_share_the_stall_detector(self):
        """停滞检测同样必须共用 —— 它和熔断阈值是同一类"口径"问题。

        如果两个引擎各写一份 page_fingerprint，"页面有没有变"的定义就会漂移，
        A/B 对比又一次失去意义（这正是当初把 LOOP_* 提成模块级常量的原因）。
        """
        from bagent import agent as hand
        from bagent import graph_agent as g

        assert g.page_fingerprint is hand.page_fingerprint
        assert g.update_stall is hand.update_stall
        assert g.STALL_STOP_AT == hand.STALL_STOP_AT
        assert g.STALL_WARN_AT == hand.STALL_WARN_AT
        assert g.CLOSING_MAX_STEPS == hand.CLOSING_MAX_STEPS

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

    改成方向签名（丢掉 dy）是必要的，**但它不足以解决问题** ——
    后来把判据改成"连续"之后这串轨迹又漏了（最长只连 3 次滚动）。
    真正收口的是页面指纹停滞检测，见 TestStallDetection。
    这里只钉住"指纹里不能有滚动量"这一件事。
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

    def test_real_t06_trace_slips_past_action_fingerprint(self):
        """真实 t06 轨迹复算：**动作指纹抓不住它**，必须由停滞检测兜底。

        这条测试我改过两次，两次都值得留痕 —— 它是一个"越查越深"的过程：

        **第一次**：把 dy 从指纹里拿掉（`scroll#{dy}` → `scroll#方向`）之后，
        我用**累计**判据复算这串轨迹，发现第 7 步就能熔断，于是写下
        "修好了，代价是只有那 3 次本来就失败的运行被命中"。

        **第二次（现在）**：这两句都站不住。

        1. 判据后来从累计改成了连续（因为累计误杀了 t04，见
           `test_fuse_decision_now_uses_consecutive`）。而在**连续**判据下，
           真实轨迹里最长只连了 3 次滚动（第 2-4 步，之后被一个 screenshot 打断），
           够不到阈值 5 —— **这条轨迹又漏了**。
        2. 更根本的是：那次"离线复算说没有通过的运行受影响"是**无效推断**。
           离线复算只能算"旧轨迹会不会提前熔断"，算不出"警告改变了模型行为、
           它会走出一条全新轨迹"。用一个不存在的未来去证明改动安全，
           这是自欺 —— 后来跑真实基线，t04 直接掉到 0/3。

        所以这条测试不再声称"新规则更强了"，而是老老实实钉住**局限**：
        光靠动作指纹这条路，对"交替方向的滚动"是**无解**的。
        真正的判据是页面指纹 —— 见 TestStallDetection。
        """
        from bagent.agent import _action_signature, LOOP_STOP_AT, loop_counts
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
        assert not any(
            old_trail[: i + 1].count(s) >= LOOP_STOP_AT for i, s in enumerate(old_trail)
        ), "旧规则在这串里确实不该熔断"

        # 新规则（连续判据）也认不出它 —— 这就是局限，如实钉住，别假装解决了。
        max_consecutive = max(loop_counts(trail[: i + 1], s)[1] for i, s in enumerate(trail))
        assert max_consecutive == 3, (
            "真实轨迹里最长连续滚动只有 3 次（第 2-4 步），"
            "够不到阈值 5 —— 动作指纹对交替方向的滚动无能为力"
        )
        assert max_consecutive < LOOP_STOP_AT

    def test_clicks_unchanged(self):
        """别的动作的指纹语义没被动过。"""
        from bagent.agent import _action_signature

        assert _action_signature(self._act(action="click", ref=7)) == "click#7"
        assert _action_signature(self._act(action="type", ref=3, text="abc")) == "type#3"
        assert _action_signature(self._act(action="extract")) == "extract"


class TestStallDetection:
    """停滞检测：抓"动作各不相同、但页面一字未变"。

    为什么必须有这一套：上面 `test_real_t06_trace_slips_past_action_fingerprint`
    已经把动作指纹的天花板量出来了 —— 它只管"同一个动作连着重复"，
    对"换着花样滚"完全无感。而后者恰恰是强模型的失效形态。

    判据从**动作**换到了**页面**：原地打转的定义不是"动作重复"，
    而是"没有获得新信息"；那是页面的属性。所以有了 `page_fingerprint`。
    """

    def _state(self, *, url="https://x/", title="T", body="正文", elements=()):
        from bagent.models import Element, PageState

        return PageState(
            url=url, title=title, body_text=body,
            elements=[Element(**e) for e in elements],
        )

    # ---- 指纹本身 --------------------------------------------------
    def test_fingerprint_ignores_element_list(self):
        """⭐ 元素清单**必须**不参与指纹 —— 这是量出来的，不是设计偏好。

        感知层采集元素时带视口过滤（`r.bottom < -800` 才丢弃），
        所以滚动会让元素数跟着变。实测（scripts/probe_page_fp.py，
        books.toscrape.com 连滑 4 次）：

            元素数 85 → 85 → 77 → 71，而**正文全程 2029 字一字未变**，
            滚回原位后指纹原样恢复（533ea144… 又回来了）。

        也就是说元素数的抖动纯粹是"我站在哪儿看"，不是"页面上多了什么"。
        若把它算进指纹，"滚动打转"这一类恰好检测不到 ——
        计数器每次都被滚动清零，一个在目标场景上失灵的判据。
        """
        from bagent.agent import page_fingerprint

        a = self._state(elements=[{"ref": 1, "tag": "a", "text": "Home"}])
        b = self._state(
            elements=[
                {"ref": 1, "tag": "a", "text": "Home"},
                {"ref": 2, "tag": "a", "text": "Next"},
                {"ref": 3, "tag": "button", "text": "Buy"},
            ]
        )
        assert page_fingerprint(a) == page_fingerprint(b), "元素清单变了不算新信息"

        # 连 ref 编号重排也不该有影响（它每次感知都会重新分配）
        c = self._state(
            elements=[
                {"ref": 91, "tag": "button", "text": "Buy"},
                {"ref": 92, "tag": "a", "text": "Home"},
            ]
        )
        assert page_fingerprint(c) == page_fingerprint(a)

    def test_fingerprint_follows_real_content(self):
        """正文 / 标题 / 网址变了 = 真的拿到了新信息，指纹必须变。"""
        from bagent.agent import page_fingerprint

        base = self._state()
        assert page_fingerprint(self._state(body="换了正文")) != page_fingerprint(base)
        assert page_fingerprint(self._state(title="新标题")) != page_fingerprint(base)
        assert page_fingerprint(self._state(url="https://y/")) != page_fingerprint(base)
        # 一字不差 → 必须相同（否则判据永远为假）
        assert page_fingerprint(self._state()) == page_fingerprint(base)

    # ---- 计数语义 --------------------------------------------------
    def test_same_fingerprint_counts_up(self):
        from bagent.agent import update_stall

        n = 0
        for _ in range(5):
            n = update_stall("fp", "fp", "scroll", n)
        assert n == 5, "页面不变、动作是滚动 → 连续计数"

    def test_changed_fingerprint_resets(self):
        from bagent.agent import update_stall

        n = 4
        assert update_stall("fp1", "fp2", "click", n) == 0, "拿到新信息 → 清零"

    def test_first_frame_cannot_be_stalled(self):
        """第 1 步没有"上一帧"可比 —— 不能凭空判它停滞。"""
        from bagent.agent import update_stall

        assert update_stall("", "fp1", "scroll", 0) == 0

    def test_readonly_actions_do_not_count(self):
        """只读动作本来就不改页面，拿它们算停滞是冤枉。

        `type` 也排除，理由是实测的假阳性：输入框里敲的字**不会出现在正文里**
        （正文抽的是可见文本，读不到 input 的 value），所以"连续填 6 个输入框"
        这种完全正常的流程会被误判成停滞。它在输入框上的打转由动作指纹负责。
        """
        from bagent.agent import update_stall

        for name in ("type", "extract", "screenshot", "finish", None):
            assert update_stall("fp", "fp", name, 3) == 3, f"{name} 不该参与停滞计数"

    def test_progress_actions_do_count(self):
        from bagent.agent import update_stall

        for name in ("click", "goto", "press", "scroll"):
            assert update_stall("fp", "fp", name, 0) == 1, f"{name} 参与停滞计数"


def test_langgraph_perceive_node_wires_the_stall_detector(monkeypatch):
    """图引擎的 perceive 节点必须**真的接上**停滞检测，不是"导入过函数"就算。

    手写引擎的停滞逻辑埋在 run() 的循环里，只能靠整跑覆盖；
    图版把它放在 perceive 节点，可以直接喂状态断言输出 —— 这正是把循环
    拆成状态图的收益（和 route_after_act 可以单测是同一件事）。

    这里同时钉住三个**容易悄悄坏掉**的点：
      - 触顶必须进 `closing`（先逼一次结论），而不是直接熔断
        —— 反例任务的合格标准是"主动说出来"，硬熔断不算拒答；
      - `last_action` 必须被消费清空，否则 finish 被退回那类路径会让它变陈旧，
        停滞计数就记到了错误的那一步头上；
      - 强指令必须真的进了 history，否则模型根本看不到。

    （不用改前端：控制台读的是 RunResult / records，这一次改动没有动它们的形状——
    已经 grep 过 src/bagent/api.py 与 web/index.html，都没有引用熔断常量。）
    """
    import asyncio

    from bagent import graph_agent as g
    from bagent.agent import STALL_STOP_AT
    from bagent.config import Settings
    from bagent.models import Action, PageState

    frozen = PageState(url="https://x/", title="T", body_text="一字未变的正文")

    async def fake_perceive(page, settings, *, step, run_dir, prefer_vision=False):
        return frozen

    monkeypatch.setattr(g, "perceive", fake_perceive)

    class _FakeSession:
        page = object()

    agent = g.LangGraphReActAgent(Settings(), llm=object())
    agent._session = _FakeSession()
    agent._run_dir = None

    state = _st(
        prev_fp=g.page_fingerprint(frozen),      # 上一帧和这一帧**一模一样**
        stall_count=STALL_STOP_AT - 1,           # 再不变就触顶
        last_action=Action(action="scroll", dy=500),
    )
    out = asyncio.run(agent._node_perceive(state))

    assert out["stall_count"] >= STALL_STOP_AT, "页面不变就该累加到触顶"
    assert out["closing"] is True, "触顶要进收束，不能直接熔断"
    assert not out.get("error"), "尚未用尽收束步数，不该报错终止"
    assert out["last_action"] is None, "last_action 必须被消费掉"
    assert any("finish" in h for h in out.get("history", [])), "强指令必须进历史"
    # prev_fp 要更新成"这一帧"，下一步才有得比
    assert out["prev_fp"] == g.page_fingerprint(frozen)


def test_langgraph_act_rejects_actions_while_closing():
    """收束阶段**必须真的拦住动作**，不能只是喊一句"请 finish"。

    这条是实测逼出来的：重放 t06 的一次真实失败
    （`scripts/probe_replay_stall.py runs/20260921-224847-t06`）显示，
    停滞计数确实在第 14 步触顶、收束指令确实注入了 —— 但模型回手就是
    `extract` / `extract` / `click#1`，把指令当耳旁风，最后磨到步数上限。
    **只喊话不拦动作，等于没管。**

    所以这里用一个"一执行就炸"的假 session：只要引擎敢真的执行那个
    非 finish 动作，测试立刻红。同时验证拒绝次数会累加、且累加到底
    之后会真的熔断（而不是无限容忍）。
    """
    import asyncio

    from bagent import graph_agent as g
    from bagent.agent import CLOSING_MAX_STEPS
    from bagent.config import Settings
    from bagent.models import Action, PageState

    class _ExplodingSession:
        page = None

        async def execute(self, action, confirm=None):  # pragma: no cover
            raise AssertionError("收束阶段不该真的执行动作")

    agent = g.LangGraphReActAgent(Settings(), llm=object())
    agent._session = _ExplodingSession()

    base = {
        "page_state": PageState(url="https://x/", title="T", body_text="没变"),
        "prev_fp": "abc123",
        "stall_count": 7,
        "closing": True,
    }

    # 第 1 次拒绝：不熔断，但要留下记录 + 把拒绝原因写回历史
    out = asyncio.run(
        agent._node_act(_st(action=Action(action="scroll", dy=500), **base))
    )
    assert out["closing_steps"] == 1
    assert not out.get("error"), "才拒绝 1 次不该熔断"
    assert out["records"][-1]["ok"] is False
    assert "收束阶段不接受" in out["records"][-1]["message"]
    assert any("finish" in h for h in out["history"])

    # 拒绝到上限：必须熔断，且错误信息要说清是"拒答"而不是"没跑完"
    out = asyncio.run(
        agent._node_act(
            _st(action=Action(action="click", ref=1), **{**base, "closing_steps": CLOSING_MAX_STEPS - 1})
        )
    )
    assert out["closing_steps"] == CLOSING_MAX_STEPS
    assert "熔断" in out.get("error", "")
    assert "拒绝给出结论" in out["error"]


def test_handwritten_engine_also_enforces_closing():
    """手写引擎要有同一套拦截 —— 两个引擎的口径必须一致。

    手写版的这段逻辑埋在 run() 的循环里，没有图版那样可以直接喂状态的口子，
    所以只能查源码。这不是理想的测法，但**比不测强**：
    这条规则一旦被删掉，两个引擎就会漂移，而漂移恰恰是本项目最想避免的事
    （阈值、指纹、计数都是为此才提成模块级共享的）。
    """
    import inspect

    from bagent import agent as hand

    src = inspect.getsource(hand.ReActAgent.run)
    assert "收束阶段不接受" in src, "手写引擎也要拦收束期的非 finish 动作"
    assert "if closing and action.action != \"finish\"" in src

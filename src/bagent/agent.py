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
from .llm import LLMClient, LLMError, build_client
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
LOOP_WARN_AT = 3   # 第 3 次重复同一个动作 → 注入强警告
LOOP_STOP_AT = 5   # 第 5 次 → 直接熔断，不再烧钱
MAX_CONSECUTIVE_FAILURES = 3


def _action_signature(action: Action) -> str:
    """动作指纹：用来识别"原地打转"。

    只看动作名和 ref，**不看 thought**——因为模型每次编的理由都不一样，
    但实际执行的动作可能是同一个。识别循环必须看行为，不能看说辞。
    """
    if action.action in ("click", "type", "goto"):
        return f"{action.action}#{action.ref or action.url}"
    if action.action == "scroll":
        return f"scroll#{action.dy}"
    return action.action


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

        for step in range(1, max_steps + 1):
            # ---------- 看（Observe）----------
            state = await perceive(
                session.page,
                self.settings,
                step=step,
                run_dir=run_dir,
                prefer_vision=prefer_vision,
            )
            prefer_vision = False

            # ---------- 想（Reason）----------
            user_prompt = self._build_prompt(task, state, history)
            raw = ""
            try:
                raw = self.llm.chat(
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
                        ok=False,
                        message=f"模型输出不是合法 JSON：{parse_err}",
                        url_after=state.url,
                        screenshot_path=state.screenshot_path,
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
                        )
                    )
                    if self.on_step:
                        self.on_step(step, action, False, message)
                    history.append(
                        f"第 {step} 步: {message}。"
                        "请重新读一遍「页面正文摘要」，把支撑结论的原文**原样复制**到 evidence，"
                        "并确认 answer 里的每个人名 / 数字 / 标题都出现在这段原文里，再调用 finish。"
                    )
                    history = history[-12:]
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
                    )
                )
                if self.on_step:
                    self.on_step(step, action, True, "任务结束（证据已核对）" if grounded else "任务结束")
                break

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
                )
            )
            if self.on_step:
                self.on_step(step, action, outcome_ok, message)

            # ---------- 记录 ----------
            status = "成功" if outcome_ok else "失败"
            desc = f"第 {step} 步: {self._describe(action)} → {status}: {message}"
            if not outcome_ok:
                desc += "（不要重复这个动作）"
            history.append(desc)
            history = history[-12:]  # 只保留最近 12 步，控制 prompt 长度

            consecutive_failures = 0 if outcome_ok else consecutive_failures + 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                error = "连续 3 步失败，已熔断"
                break

            # ---------- 循环检测 ----------
            sig = _action_signature(action)
            sig_trail.append(sig)
            repeats = sig_trail.count(sig)
            if repeats >= LOOP_STOP_AT:
                error = f"检测到原地打转：动作 {sig} 重复 {repeats} 次且无进展，已熔断"
                break
            if repeats >= LOOP_WARN_AT:
                # 把警告写回历史，下一轮模型一定会看到。
                # 注意措辞是"禁止"而不是"建议"——小模型对强约束才有反应。
                history.append(
                    f"⚠ 严重警告：你已经连续 {repeats} 次执行 `{sig}`，"
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

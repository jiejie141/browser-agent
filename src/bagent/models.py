"""数据结构定义。

用 Pydantic 约束模型输出，比手写 if 判断健壮得多——
模型一旦瞎编动作名或漏字段，这里会立刻报错而不是静默跑飞。
"""

from __future__ import annotations

import json
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, ValidationError

# 允许模型调用的浏览器动作集合。
# 刻意收窄：动作越少，模型越不容易瞎调。
ActionName = Literal[
    "goto",           # 打开网址
    "click",          # 点击某个编号的元素
    "type",           # 在某个编号的输入框里输入文字
    "press",          # 按键，如 Enter / Escape
    "scroll",         # 滚动页面
    "extract",        # 把当前页面正文读回来
    "screenshot",     # 截图（走视觉通道）
    "finish",         # 任务完成，给出结论
]


class Action(BaseModel):
    """模型每一步输出的一个动作。"""

    thought: str = Field(default="", description="这一步为什么这么做")
    action: ActionName
    ref: Optional[int] = Field(default=None, description="click/type 用：元素编号")
    text: Optional[str] = Field(default=None, description="type 用：要输入的文字")
    url: Optional[str] = Field(default=None, description="goto 用：目标网址")
    key: Optional[str] = Field(default=None, description="press 用：按键名")
    dy: Optional[int] = Field(default=None, description="scroll 用：滚动像素，正数向下")
    answer: Optional[str] = Field(default=None, description="finish 用：最终结论")


class Usage(BaseModel):
    """一次或多次模型调用的 token 用量，供成本面板统计。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, other: "Usage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.calls += other.calls

    def cost_yuan(self, price_in: float, price_out: float) -> float:
        """按每百万 token 单价估算花费（元）。"""
        return (
            self.prompt_tokens / 1_000_000 * price_in
            + self.completion_tokens / 1_000_000 * price_out
        )


class Element(BaseModel):
    """页面上一个可交互元素。ref 是给模型用的编号。"""

    ref: int
    tag: str
    text: str = ""
    aria: str = ""
    placeholder: str = ""
    # 元素所在的页面区域：main（主内容区）/ neutral（判断不出）/ chrome（导航·侧栏·页脚）。
    #
    # 为什么要多这个字段：编号预算只有 100 个，而按 DOM 顺序抓取时，
    # **侧边栏和导航会把预算吃光**，正文里的链接一个都进不来 ——
    # 表现是"页面上明明有搜索结果，Agent 却说不出来"。
    # 有了区域之后，编号分配就能按区域配额度（见 perception.allocate）。
    region: str = "neutral"


class PageState(BaseModel):
    """一帧页面感知结果——这就是 Agent 的"眼睛"。"""

    step: int = 0
    url: str = ""
    title: str = ""
    elements: list[Element] = Field(default_factory=list)
    body_text: str = ""
    screenshot_path: str = ""

    def render_for_prompt(self, max_body_chars: int = 1500) -> str:
        """渲染成给模型看的纯文本。控制长度就是控制成本。"""
        lines = [
            f"当前网址: {self.url}",
            f"页面标题: {self.title}",
            "",
            "可交互元素（点击/输入时请使用方括号里的编号）:",
        ]
        if self.elements:
            for el in self.elements:
                label = el.text or el.aria or el.placeholder or "(无文字)"
                # 只给外围元素打标（导航/侧栏/页脚），主内容区不打 ——
                # 一是省 token，二是模型最常犯的错就是把侧栏里的相关推荐
                # 当成"第一条搜索结果"，这个标记正好压住它。
                mark = " [导航/侧栏]" if el.region == "chrome" else ""
                lines.append(f"  [{el.ref}] <{el.tag}> {label}{mark}")
            if any(el.region == "chrome" for el in self.elements):
                lines += [
                    "",
                    "注：标了 [导航/侧栏] 的是页面外围链接（菜单、分类、页脚），"
                    "回答「第一条结果」这类问题时应优先用未标记的元素。",
                ]
        else:
            lines.append("  (这一帧没有解析到可交互元素——可能是页面还在加载，")
            lines.append("   或者内容在 Canvas / iframe 里，可以试试 scroll 或 screenshot)")

        body = (self.body_text or "").strip()
        if body:
            if len(body) > max_body_chars:
                body = body[: max_body_chars // 2] + "\n...（中间省略）...\n" + body[-max_body_chars // 2 :]
            lines += ["", "页面正文摘要:", body]
        return "\n".join(lines)


class StepRecord(BaseModel):
    """一步的完整记录，落盘后可用于复盘和评测。"""

    step: int
    # 契约上允许出现"非动作"的一步：模型输出不是合法 JSON 时，那一轮
    # 也是真实消耗了一次调用的，必须留下痕迹。为了给这种情况一个合法的
    # 容器，这里放宽成 `Action | None`，并把原始动作名放在 `raw_action`。
    #
    # 反过来做（造一个 action="(格式错误)" 的假 Action）是行不通的：
    # ActionName 是封闭字面量类型，pydantic 会在构造时直接抛
    # ValidationError —— 那等于"为了记录一个格式错误，先让程序崩掉"。
    action: Action | None = None
    raw_action: str = ""
    ok: bool = False
    message: str = ""
    url_after: str = ""
    screenshot_path: str = ""

    @property
    def action_name(self) -> str:
        """给渲染/报表用的动作名，真实动作优先，其次原始字符串。"""
        if self.action is not None:
            return self.action.action
        return self.raw_action or "-"


class StepOutcome(BaseModel):
    """动作执行结果，回传给 Agent 决定下一步。"""

    ok: bool
    message: str
    vision_hint: bool = False  # True 表示建议下一帧走视觉通道


def parse_action(raw: str) -> tuple[Action | None, str]:
    """把模型返回的文本解析成 Action。

    返回 (Action | None, 错误信息)。模型偶尔会包上 ```json 代码块，
    也会夹带解释文字，这里尽量容错，但解析不出来就如实报错让上层重试。
    """
    text = (raw or "").strip()
    if not text:
        return None, "模型返回了空内容"

    # 剥掉 markdown 代码块围栏
    if text.startswith("```"):
        text = text.split("\n", 1)[-1] if "\n" in text else text
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()

    # 截取最外层花括号
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None, f"没找到 JSON 对象，原始返回: {text[:200]}"
    candidate = text[start : end + 1]

    try:
        data: Any = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return None, f"JSON 解析失败({exc})，原始内容: {candidate[:200]}"

    try:
        return Action(**data), ""
    except ValidationError as exc:
        return None, f"动作字段不合法: {exc.errors()[:3]}"

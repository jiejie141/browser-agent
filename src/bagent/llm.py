"""LLM 客户端封装。

三件事：
1. 统一走 OpenAI 兼容协议，换厂商只改 .env 里的 base_url + model。
2. 超时 + 指数退避重试——线上调用一定会抖，不能让一次网络抖动毁掉整条任务链。
3. 逐次累计 token 用量，这是成本面板的数据来源。
"""

from __future__ import annotations

import base64
import logging
import random
import time
from pathlib import Path
from typing import Any, Sequence

import openai
from openai import OpenAI

from .config import Settings
from .models import Usage

log = logging.getLogger(__name__)

# 这些异常值得重试：限流、连接抖动、服务端 5xx、超时
RETRYABLE = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


class LLMError(RuntimeError):
    """重试耗尽后抛出，带上足够定位问题的上下文。"""


class LLMClient:
    """对 OpenAI SDK 的薄封装，专注于「可重试 + 可计量」。"""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        *,
        timeout: float = 60.0,
        max_retries: int = 3,
        label: str = "llm",
    ) -> None:
        self.model = model
        self.label = label
        self.max_retries = max_retries
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.usage = Usage()

    # ------------------------------------------------------------------
    # 纯文本对话
    # ------------------------------------------------------------------
    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        temperature: float = 0.0,
        json_mode: bool = False,
        max_tokens: int = 1024,
    ) -> str:
        """发一轮对话，返回纯文本内容，并把用量累加进 self.usage。

        temperature 默认 0：Agent 规划动作需要稳定，不需要"创造力"。
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            resp = self._call_with_retry(kwargs)
        except LLMError as exc:
            # 兼容兜底：相当一部分国产模型（尤其是免费的小模型）不支持
            # response_format 参数。这里自动降级为"只靠提示词约束 JSON"，
            # 而不是直接报错——否则每换一个厂商就要改一次代码。
            if json_mode and _looks_like_unsupported_format(str(exc)):
                log.warning(
                    "[%s] 该端点不支持 response_format，降级为提示词约束 JSON 输出",
                    self.label,
                )
                kwargs.pop("response_format", None)
                resp = self._call_with_retry(kwargs)
            else:
                raise

        if resp.usage:
            self.usage.add(
                Usage(
                    prompt_tokens=resp.usage.prompt_tokens or 0,
                    completion_tokens=resp.usage.completion_tokens or 0,
                    calls=1,
                )
            )
        return (resp.choices[0].message.content or "").strip()

    # ------------------------------------------------------------------
    # 视觉对话（降级通道）
    # ------------------------------------------------------------------
    def chat_vision(
        self,
        prompt: str,
        image_path: str | Path,
        *,
        max_tokens: int = 512,
    ) -> str:
        """把截图喂给视觉模型，让它用自然语言描述目标在哪。

        注意：这条路径的 token 成本远高于 DOM 通道，
        所以只在 DOM 找不到元素时才走（见 perception 里的判断）。
        """
        data_url = _to_data_url(image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ]
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        resp = self._call_with_retry(kwargs)
        if resp.usage:
            self.usage.add(
                Usage(
                    prompt_tokens=resp.usage.prompt_tokens or 0,
                    completion_tokens=resp.usage.completion_tokens or 0,
                    calls=1,
                )
            )
        return (resp.choices[0].message.content or "").strip()

    # ------------------------------------------------------------------
    # 重试内核
    # ------------------------------------------------------------------
    def _call_with_retry(self, kwargs: dict[str, Any]) -> Any:
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._client.chat.completions.create(**kwargs)
            except RETRYABLE as exc:
                last_exc = exc
                if attempt == self.max_retries:
                    break
                # 指数退避 + 抖动：避免多个实例同时重试把对端打垮
                delay = min(2 ** (attempt - 1), 8) + random.uniform(0, 0.5)
                log.warning(
                    "[%s] 第 %d 次调用失败(%s)，%.1fs 后重试",
                    self.label,
                    attempt,
                    type(exc).__name__,
                    delay,
                )
                time.sleep(delay)
            except openai.AuthenticationError as exc:
                # 认证错误重试没意义，直接失败并给出可操作的提示
                raise LLMError(
                    "API Key 被拒绝。请检查 .env 里的 LLM_API_KEY 是否有效、"
                    f"以及 LLM_BASE_URL 是否对应该厂商。原始错误: {exc}"
                ) from exc
            except openai.BadRequestError as exc:
                raise LLMError(
                    f"请求被拒绝（通常是模型名写错或参数不支持）: {exc}"
                ) from exc

        raise LLMError(
            f"调用 {self.model} 连续失败 {self.max_retries} 次。"
            f"最后一次错误: {type(last_exc).__name__}: {last_exc}"
        )


def _looks_like_unsupported_format(msg: str) -> bool:
    """判断报错是不是"这个参数我不支持"，而不是"你参数值写错了"。

    只对前者做降级；对后者降级会掩盖真正的配置问题。
    """
    low = msg.lower()
    keys = ("response_format", "json_object", "unsupported", "not support", "不支持")
    return any(k in low for k in keys)


def _to_data_url(image_path: str | Path) -> str:
    """把本地图片转成 data URL，OpenAI 兼容端点都认这种格式。"""
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"截图不存在: {p}")
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def build_client(settings: Settings) -> LLMClient:
    return LLMClient(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        label="main",
    )


class MockLLMClient(LLMClient):
    """离线替身：按剧本返回动作，不联网。

    存在的意义：**把"模型"这个外部依赖从测试里摘出去。**
    主循环、历史管理、终止条件、trace 落盘、成本统计这些逻辑，
    不应该因为"今天 key 过期了"就没法验证。

    同时它也是最好的教学工具——你可以逐行打断点，
    看模型返回一个非法 JSON 时主循环是怎么兜住的。
    """

    def __init__(self, script: list[str] | None = None) -> None:
        # 刻意不调用父类 __init__：不需要 api_key，也不想创建网络客户端
        self.model = "mock-scripted"
        self.label = "mock"
        self.max_retries = 0
        self._client = None  # type: ignore[assignment]
        self.usage = Usage()

        self.script = script or [
            '{"thought": "先往下滚一屏，看看有没有更多元素", "action": "scroll", "dy": 600}',
            "这一步我故意返回一段不是 JSON 的文字，用来验证主循环能不能兜住",
            '{"thought": "读取当前页面正文", "action": "extract"}',
            '{"thought": "截个图", "action": "screenshot"}',
            '{"thought": "任务完成，汇报结果", "action": "finish", '
            '"answer": "MOCK 模式跑通：滚动 / 非法输出兜底 / 读正文 / 截图 / 结束 五个环节均已执行"}',
        ]
        self._idx = 0

    def chat(self, messages, *, temperature=0.0, json_mode=False, max_tokens=1024) -> str:
        if self._idx >= len(self.script):
            # 剧本演完还没结束 → 返回 finish，避免把循环拖到最大步数
            return '{"thought": "剧本已演完", "action": "finish", "answer": "MOCK 剧本结束"}'
        reply = self.script[self._idx]
        self._idx += 1
        # 模拟 token 用量，让成本面板有数可算
        self.usage.add(
            Usage(
                prompt_tokens=700 + self._idx * 120,
                completion_tokens=40 + self._idx * 5,
                calls=1,
            )
        )
        return reply

    def chat_vision(self, prompt, image_path, *, max_tokens=512) -> str:
        self.usage.add(Usage(prompt_tokens=1200, completion_tokens=80, calls=1))
        return "（MOCK）这是一个用于离线验证的假视觉描述。"

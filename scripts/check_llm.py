"""最小连通性检查：确认 key / base_url / model 三件套能打通。

跑这个只需要几十个 token，非常便宜。
排查"为什么 Agent 不动"时，第一件事就是跑它——
先把"模型能不能通"这个变量排除掉，再看别的。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bagent.config import get_settings  # noqa: E402
from bagent.llm import LLMError, build_client  # noqa: E402


def main() -> int:
    settings = get_settings(refresh=True)
    print(f"模型   : {settings.llm_model}")
    print(f"端点   : {settings.llm_base_url}")
    print(f"key    : {'已配置 (len=%d)' % len(settings.llm_api_key) if settings.llm_api_key else '未配置'}")

    if not settings.llm_api_key:
        print("\n[失败] 没有读到 LLM_API_KEY，请检查 .env")
        return 2

    client = build_client(settings)
    try:
        # 顺便测一下 json_mode——Agent 全靠它拿结构化动作
        reply = client.chat(
            [{"role": "user", "content": '请输出 JSON: {"ok": true, "msg": "连通"}'}],
            json_mode=True,
            max_tokens=32,
        )
    except LLMError as exc:
        print(f"\n[失败] {exc}")
        return 1

    print(f"\n[成功] 模型返回: {reply}")
    print(f"[用量] {client.usage.model_dump()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

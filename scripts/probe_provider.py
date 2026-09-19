"""探测当前 key 属于哪家厂商。

为什么需要这个工具：`sk-` 开头的 key 在十几家厂商里都有，
长度和字符集还不能可靠区分。与其猜，不如每个端点打一发便宜的请求看谁认。

用法：
    python scripts/probe_provider.py

它会读 .env 里的 LLM_API_KEY，然后依次试所有候选端点，
最后告诉你哪些端点接受了这个 key。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openai import OpenAI  # noqa: E402

from bagent.config import get_settings  # noqa: E402

# (显示名, base_url, model, 备注)
CANDIDATES: list[tuple[str, str, str]] = [
    ("硅基流动 SiliconFlow", "https://api.siliconflow.cn/v1", "Qwen/Qwen2.5-7B-Instruct"),
    ("月之暗面 Moonshot", "https://api.moonshot.cn/v1", "moonshot-v1-8k"),
    ("DeepSeek 官方", "https://api.deepseek.com/v1", "deepseek-chat"),
    ("阿里云百炼 DashScope", "https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-turbo"),
    ("智谱 BigModel", "https://open.bigmodel.cn/api/paas/v4", "glm-4-flash"),
    ("腾讯混元", "https://api.hunyuan.cloud.tencent.com/v1", "hunyuan-lite"),
    ("阶跃 StepFun", "https://api.stepfun.com/v1", "step-1-8k"),
    ("讯飞星火", "https://spark-api-open.xf-yun.com/v1", "lite"),
    ("MiniMax", "https://api.minimax.chat/v1", "abab6.5s-chat"),
    ("百度千帆 v2", "https://qianfan.baidubce.com/v2", "ernie-speed-128k"),
    ("OpenRouter", "https://openrouter.ai/api/v1", "openai/gpt-4o-mini"),
]


def main() -> int:
    key = get_settings(refresh=True).llm_api_key
    if not key:
        print("读不到 LLM_API_KEY，请先填进 .env")
        return 2

    print(f"待探测 key: {key[:8]}...{key[-4:]}  (长度 {len(key)})\n")
    winners: list[tuple[str, str, str]] = []

    for label, base, model in CANDIDATES:
        client = OpenAI(api_key=key, base_url=base, timeout=25.0)
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "回复两个字：正常"}],
                max_tokens=16,
            )
            text = (resp.choices[0].message.content or "").strip()
            print(f"  [成功]  {label:22s} {model:34s} -> {text!r}")
            winners.append((label, base, model))
        except Exception as exc:
            msg = str(exc).replace("\n", " ")
            if "401" in msg or "invalid" in msg.lower() or "Authentication" in msg:
                tag = "401 不认这个 key"
            elif "404" in msg:
                tag = "404 端点或模型名不对"
            elif "model" in msg.lower():
                tag = "模型名不被支持"
            else:
                tag = type(exc).__name__
            print(f"  [跳过]  {label:22s} {tag}")

    print()
    if not winners:
        print("❌ 没有任何端点接受这个 key。")
        print("   可能原因：key 失效 / 复制不完整 / 该厂商需要先在控制台开通模型。")
        return 1

    print("✅ 可用的端点：")
    for label, base, model in winners:
        print(f"\n   {label}")
        print(f"   LLM_BASE_URL={base}")
        print(f"   LLM_MODEL={model}")
    print("\n把上面两行填进 .env，再把 LLM_API_KEY 设为你的 key 即可。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

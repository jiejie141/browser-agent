"""再查一次：供应商的模型里到底有没有能收图片的。

/providers 的 /v1/models 只能看到**名字**，名字里没有 vision/vl 不代表真的
不支持 —— 有些中转站会把视觉能力挂在通用名字上。所以这里直接**发一张图**
去试：能收下并回答的就是视觉模型，报错里提到 image / content / multimodal
的就是不支持。

用法： python scripts/probe_vision_models.py
"""
from __future__ import annotations

import base64
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dotenv import dotenv_values  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ENV = dotenv_values(ROOT / ".env")

BASE = (ENV.get("LLM_BASE_URL") or os.getenv("LLM_BASE_URL") or "").rstrip("/")
KEY = (ENV.get("LLM_API_KEY") or os.getenv("LLM_API_KEY") or "").strip()

# ⚠️ 图不能太小。实测供应商对尺寸有下限（seed 系列 ≥14px、qwen 系列 >10px），
# 拿 1x1 的图探测会把"支持视觉、只是嫌图太小"**误判成不支持** ——
# 第一次探测就栽在这上面（qwen3.8-max / seed-2.1-pro 全被判死）。
# 所以这里现造一张 64x64 的图（不依赖 Pillow，见 scripts/_make_png.py）。
import _make_png  # noqa: E402  （本文件被当脚本跑时同目录就在 sys.path 里）

PNG_1PX = "data:image/png;base64," + base64.b64encode(
    _make_png.make_png(64)
).decode("ascii")


def models() -> list[str]:
    req = urllib.request.Request(
        f"{BASE}/models", headers={"Authorization": f"Bearer {KEY}"}
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read().decode("utf-8"))
    return sorted(m.get("id", "") for m in data.get("data", []) if m.get("id"))


def try_vision(model: str) -> tuple[str, str]:
    """返回 (结论, 证据)。结论 ∈ {VISION, NO, ERR}"""
    body = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "图里有几个像素？只回答一个数字。"},
                    {"type": "image_url", "image_url": {"url": PNG_1PX}},
                ],
            }
        ],
        "max_tokens": 16,
    }
    payload = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE}/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {KEY}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read().decode("utf-8"))
        txt = data["choices"][0]["message"]["content"]
        return "VISION", str(txt)[:80]
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")[:300]
        return "NO", f"HTTP {exc.code}: {raw}"
    except Exception as exc:  # 网络层问题，单独标出来，别混进"不支持"
        return "ERR", f"{type(exc).__name__}: {str(exc)[:160]}"


def main() -> int:
    print(f"base = {BASE}")
    ms = models()
    print(f"共 {len(ms)} 个模型，逐个发图探测：\n")
    vision, no, err = [], [], []
    for m in ms:
        kind, evidence = try_vision(m)
        flag = {"VISION": "✅ 视觉可用", "NO": "❌ 不支持图片", "ERR": "⚠ 探测失败"}[kind]
        print(f"{flag}  {m}\n        {evidence}\n")
        {"VISION": vision, "NO": no, "ERR": err}[kind].append(m)

    print("=" * 60)
    print(f"能收图片的: {len(vision)} 个 -> {vision}")
    print(f"不支持图片: {len(no)} 个")
    print(f"探测失败  : {len(err)} 个 -> {err}")
    return 0 if vision else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""第二轮：能收下图片 ≠ 看得懂图片。

上一轮 10 个模型对 image_url 返回 200，但其中 7 个的 content 是**空的** ——
这有两种可能：(a) 真看了但不想答；(b) 上游把图片丢了、只当纯文本请求处理。
区分不了就没法写进 README。

所以这一轮改成**判分**：造一张上红下蓝的图，问"上半部分是什么颜色"。
- 答"红" → 真的读了像素；
- 答别的 / 空 → 只是收下了请求，视觉能力存疑。

用法： python scripts/probe_vision_accuracy.py
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dotenv import dotenv_values  # noqa: E402

import _make_png  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ENV = dotenv_values(ROOT / ".env")
BASE = (ENV.get("LLM_BASE_URL") or os.getenv("LLM_BASE_URL") or "").rstrip("/")
KEY = (ENV.get("LLM_API_KEY") or os.getenv("LLM_API_KEY") or "").strip()

# 上红下蓝
SIZE = 128
rows = []
for y in range(SIZE):
    rgb = (220, 30, 30) if y < SIZE // 2 else (30, 30, 220)
    rows.append(b"\x00" + bytes(rgb) * SIZE)


def _chunk(tag: bytes, data: bytes) -> bytes:
    import struct
    import zlib

    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def build_half_image() -> bytes:
    import struct
    import zlib

    ihdr = struct.pack(">IIBBBBB", SIZE, SIZE, 8, 2, 0, 0, 0)
    raw = b"".join(rows)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(raw, 9))
        + _chunk(b"IEND", b"")
    )


IMG = "data:image/png;base64," + base64.b64encode(build_half_image()).decode("ascii")

CANDIDATES = sys.argv[1:] or [
    "deepseek-flash", "glm-5.3-flash", "glm-5.3-flashx",
    "kimi-k2.6", "kimi-k2.7-code", "kimi-k3",
    "qwen3.7-flash", "qwen3.8-27b", "qwen3.8-flash", "qwen3.8-max",
]

QUESTION = "这张图的上半部分是什么颜色？只回答一个字：红 或 蓝。"


def ask(model: str) -> tuple[str, str]:
    body = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": QUESTION},
                    {"type": "image_url", "image_url": {"url": IMG}},
                ],
            }
        ],
        "max_tokens": 32,
    }
    req = urllib.request.Request(
        f"{BASE}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {KEY}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            data = json.loads(r.read().decode("utf-8"))
        txt = (data["choices"][0]["message"]["content"] or "").strip()
        ok = "红" in txt and "蓝" not in txt.replace("红", "", 1)[:0] or (
            txt.startswith("红")
        )
        return ("PASS" if ok else "MISS"), (txt[:60] or "(空)")
    except urllib.error.HTTPError as exc:
        return "ERR", f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:120]}"
    except Exception as exc:
        return "ERR", f"{type(exc).__name__}: {str(exc)[:120]}"


def main() -> int:
    print(f"base = {BASE}\n问：{QUESTION}\n")
    passed = []
    for m in CANDIDATES:
        kind, evidence = ask(m)
        flag = {"PASS": "✅ 读对了", "MISS": "❌ 答错/空", "ERR": "⚠ 调用失败"}[kind]
        print(f"{flag}  {m:<20} -> {evidence}")
        if kind == "PASS":
            passed.append(m)
    print("\n" + "=" * 60)
    print(f"真能读懂图片的: {len(passed)}/{len(CANDIDATES)} -> {passed}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

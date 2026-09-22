"""生成一张 64x64 的纯色 PNG，不依赖 Pillow（项目里没装）。

用途：供应商对图片有最小尺寸限制（实测 seed 要求 ≥14px、qwen 系列要求 >10px），
拿 1x1 的图去探会把"支持视觉但嫌图太小"误判成"不支持视觉"。
"""
from __future__ import annotations

import base64
import struct
import zlib


def _chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def make_png(size: int = 64, rgb: tuple[int, int, int] = (30, 90, 200)) -> bytes:
    ihdr = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    row = b"\x00" + bytes(rgb) * size
    raw = row * size
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(raw, 9))
        + _chunk(b"IEND", b"")
    )


if __name__ == "__main__":
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    print("data:image/png;base64," + base64.b64encode(make_png(n)).decode("ascii"))

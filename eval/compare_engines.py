# -*- coding: utf-8 -*-
"""读取两个引擎最新一次运行的 trace，做 A/B 对比。

为什么需要它：换引擎最容易出的事是"看起来跑通了，实际行为变了"。
这个脚本把两次运行的关键指标摆在一起，**数字必须一致或差异可解释**，
否则说明两个引擎不等价。

用法：
    python eval/compare_engines.py            # 读 runs/ 下最近两次运行
"""
from __future__ import annotations

import glob
import json
import os
import sys


def _load_last(n: int = 2) -> list[tuple[str, dict]]:
    paths = sorted(glob.glob(os.path.join("runs", "*", "trace.json")))
    out = []
    for p in paths[-n:]:
        try:
            with open(p, encoding="utf-8") as fh:
                out.append((p, json.load(fh)))
        except Exception as e:
            print(f"[warn] 读取 {p} 失败: {e}")
    return out


def main() -> int:
    runs = _load_last(2)
    if len(runs) < 2:
        print("需要至少两次运行记录（runs/*/trace.json）。")
        return 2

    rows = []
    for path, j in runs:
        u = j.get("usage") or {}
        rows.append({
            "run": os.path.dirname(path).replace("runs" + os.sep, ""),
            "steps": j.get("steps"),
            "success": j.get("success"),
            "finished": j.get("finished"),
            "prompt_tokens": u.get("prompt_tokens"),
            "completion_tokens": u.get("completion_tokens"),
            "calls": u.get("calls"),
            "cost_yuan": j.get("cost_yuan"),
            "elapsed": j.get("elapsed_seconds"),
            "error": j.get("error") or "",
        })

    keys = ["steps", "success", "finished", "prompt_tokens",
            "completion_tokens", "calls", "cost_yuan", "elapsed"]
    w = max(len(k) for k in keys) + 2
    print("=" * 72)
    print("引擎 A/B 对比（最近两次运行）")
    print("=" * 72)
    print(f"{'指标':<{w}}" + "".join(f"{r['run'][-22:]:<24}" for r in rows))
    for k in keys:
        print(f"{k:<{w}}" + "".join(f"{str(r[k]):<24}" for r in rows))

    print()
    print("=" * 72)
    print("差异判定")
    print("=" * 72)
    a, b = rows[0], rows[1]
    diffs = [k for k in ("steps", "prompt_tokens", "completion_tokens", "calls")
             if a[k] != b[k]]
    if not diffs:
        print("✓ 关键指标完全一致 —— 两个引擎行为等价。")
    else:
        for k in diffs:
            print(f"! {k}: {a[k]} vs {b[k]}")
        print("  需逐条确认是可解释差异（如 mock 剧本调用次数不同），"
              "否则视为引擎不等价。")

    for r in rows:
        if r["error"]:
            print(f"[{r['run']}] 错误: {r['error'][:100]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

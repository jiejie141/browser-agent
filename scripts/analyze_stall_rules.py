"""离线对比两种"停滞判据"，用**已落盘的 `page_fp` 序列**算，不调模型、不跑浏览器。

## 为什么要比

停滞检测的判据有两种自然写法，看着都对，代价却完全不同：

| 判据 | 规则 | 直觉 |
|---|---|---|
| A. 同页连续 | 相邻两帧指纹相同就 +1，变了清零 | "页面连续 N 步没动" |
| B. 未见新页 | 当前指纹**以前出现过**就 +1，没见过就清零 | "连续 N 步没见到一页新页面" |

A 的问题是**交替循环能绕过去**：在两个已知页面之间来回跳，相邻两帧永远不同，
计数器永远是 0。实测（t06 第 1 轮）：4 个页面轮着点，A 的计数最多只到 2。

B 能抓住交替循环（都在"旧页面"上），代价是可能冤枉"合理地来回看"的正常任务。

## 这个脚本的口径（很重要，别和上次那个错误推断混了）

它只回答一件事：**"这段已经落盘的指纹序列，在规则 X 下会不会触发？"**
这是一个**纯函数**，没有不确定性 —— 指纹序列已经写死在 trace 里了。

它**不**回答"触发之后模型会不会改行为"。上一轮就是拿"旧轨迹重算"去证明
"改动没有副作用"，然后用一次真实基线被打脸（t04 0/3）。
所以这里的结论只能用来**排除明显更差的规则**，不能用来说"改了会更好"。

用法：
    python scripts/analyze_stall_rules.py            # 扫 runs/ 下所有带 page_fp 的轨迹
    python scripts/analyze_stall_rules.py --only t04 t06
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bagent.agent import STALL_STOP_AT  # noqa: E402


def rule_a(fps: list[str]) -> int:
    """同页连续：相邻相同 +1，不同清零。返回最大值。"""
    best = cur = 0
    for prev, cur_fp in zip(fps, fps[1:]):
        cur = cur + 1 if cur_fp == prev else 0
        best = max(best, cur)
    return best


def rule_b(fps: list[str]) -> int:
    """未见新页：指纹是"已出现过的"就 +1，是新指纹就清零。返回最大值。"""
    best = cur = 0
    seen: set[str] = set()
    for fp in fps:
        if fp in seen:
            cur += 1
        else:
            seen.add(fp)
            cur = 0
        best = max(best, cur)
    return best


def first_hit_a(fps: list[str], threshold: int) -> int | None:
    """规则 A：第几步第一次让"同页连续"达到阈值（1-based）。第 1 步无从比较。"""
    cur = 0
    for i in range(2, len(fps) + 1):
        cur = cur + 1 if fps[i - 1] == fps[i - 2] else 0
        if cur >= threshold:
            return i
    return None


def first_hit_b(fps: list[str], threshold: int) -> int | None:
    """规则 B：第几步第一次让"未见新页"达到阈值（1-based）。"""
    seen: set[str] = set()
    cur = 0
    for i, fp in enumerate(fps, start=1):
        if fp in seen:
            cur += 1
        else:
            seen.add(fp)
            cur = 0
        if cur >= threshold:
            return i
    return None


def load_rows(only: list[str] | None) -> list[dict]:
    rows = []
    for p in sorted(Path("runs").glob("*/trace.json"), key=lambda q: q.stat().st_mtime):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        recs = d.get("records") or []
        if not recs:
            continue
        tag = p.parent.name
        # 任务 id：目录名以 -tXX 结尾
        tid = tag.split("-")[-1]
        if only and tid not in only:
            continue
        # 只统计**带指纹字段**的轨迹 —— 早于这次改动的 trace 里没有 page_fp，
        # 拿空列表去算只会得到"没触发"，那会得出"新规则很安全"的假结论。
        fps = [r.get("page_fp") or "" for r in recs]
        if not any(fps):
            continue
        rows.append({"tag": tag, "id": tid, "finished": d.get("finished"), "err": d.get("error") or "", "fps": fps})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    rows = load_rows(args.only)
    if not rows:
        print("没有找到带 page_fp 的轨迹。先跑一次 eval/run_eval.py。")
        return 1

    print(f"扫描 {len(rows)} 次运行（阈值 STALL_STOP_AT={STALL_STOP_AT}）")
    print(f"{'运行':<24} {'任务':<5} {'步':>3} {'完成':<5} {'A最大':>5} {'A首次触发':>9} {'B最大':>5} {'B首次触发':>9}")
    print("-" * 92)
    summary = {"a_hit": 0, "b_hit": 0, "a_hit_pass": 0, "b_hit_pass": 0, "n": 0, "n_pass": 0}
    for r in rows:
        fps = r["fps"]
        a, b = rule_a(fps), rule_b(fps)
        ah = first_hit_a(fps, STALL_STOP_AT)
        bh = first_hit_b(fps, STALL_STOP_AT)
        mark = "通过" if r["finished"] and not r["err"] else "未过"
        print(
            f"{r['tag']:<24} {r['id']:<5} {len(fps):>3} {mark:<5} {a:>5} {str(ah or '-'):>9} {b:>5} {str(bh or '-'):>9}"
        )
        summary["n"] += 1
        if r["finished"] and not r["err"]:
            summary["n_pass"] += 1
        if ah is not None:
            summary["a_hit"] += 1
            if r["finished"] and not r["err"]:
                summary["a_hit_pass"] += 1
        if bh is not None:
            summary["b_hit"] += 1
            if r["finished"] and not r["err"]:
                summary["b_hit_pass"] += 1

    print("-" * 92)
    print(f"规则 A（同页连续）会触发 {summary['a_hit']}/{summary['n']} 次；"
          f"其中 **本来是通过的** 有 {summary['a_hit_pass']} 次 ← 这些是被规则误伤的候选")
    print(f"规则 B（未见新页）会触发 {summary['b_hit']}/{summary['n']} 次；"
          f"其中 **本来是通过的** 有 {summary['b_hit_pass']} 次 ← 同上")
    print()
    print("⚠️ 再说一遍口径：这里只算「这段指纹序列会不会触发」，算不出「触发之后模型会怎样」。")
    print("   「误伤候选」也只表示「这条规则会在一次通过运行的某一步就喊停」——")
    print("   真要看它有没有害，只能改完真跑（上一轮的教训）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

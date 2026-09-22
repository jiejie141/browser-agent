"""用**已落盘的轨迹**为"振荡型循环"挑判据 —— 把候选规则全放一处，扫参数、看逐条。

用法：
    python scripts/choose_oscillation_rule.py                # 全部候选 + 逐条明细
    python scripts/choose_oscillation_rule.py --detail F1 --n 2 --w 8   # 只看某一组

## 要解决的问题

三套判据各有一个盲区，而 t06 的失败形态**同时落在前两套的盲区里**：

| 判据 | 看什么 | 盲区 |
|---|---|---|
| 动作指纹 | 行为有没有重复 | 振荡时动作是**交替**的（点站点名 / 点加入购物车 / …），任何单个动作的连续次数都停在 1 |
| 停滞检测 | 页面有没有变 | 振荡时页面每步都在变（A→B→A→B），实测停滞计数全程只有 0~2，**一次警告都没注入过** |
| **振荡检测（本题）** | 有没有在**已去过的一小片页面**里绕 | 见下"选出来的规则仍然分不开" |

## 候选规则

    A   同页连续：相邻两帧指纹相同就 +1，变了清零
    B   未见新页：指纹以前出现过就 +1，没见过就清零
    C   近 W 步无新页：窗口内指纹全在更早部分出现过，且窗口内不同页面 <= 2
    D   重走同一条边：把每步看成有向边 (上一步页面 -> 这一步页面)，同一条边走 >= N 次
    E   近 W 步不同页面 <= 2 且无新页
    F1  D 的加窗版：最近 W 步内某条边走 >= N 次，且这 W 步里没有新页面
    F2  同 F1，但"新页面"的口径放宽到最近 2W 步
    G   近 W 步不同页面 <= N，且没有新页面（不看边，只看页面集合）

## ⭐ 结论（2026-09-22 实测。轨迹数会随跑测增长，下列是**当时快照**）

**没有一条分得开。** 最好的一组是 `D(n=3, w=4)`（重走同一条边 ≥3 次）：

    受害运行（耗尽预算/熔断）  19/19 全抓到
    已收敛的运行              29/62 也触发 —— 其中 11 条是 normal
                              （主要是 t04 必应"首页 <-> 结果页"的正常往返）

也就是说：**该抓的一个不漏，但不该抓的也误伤近一半。** 这种判据拿去熔断，
等于用"会打死 11 条正常任务"换"抓住全部受害任务" —— 不划算。

**分不开的根本原因**：t06 失败与通过的那些运行，**行为本身是一样的**（都在来回走），
差别只在"最后有没有收尾"。所以这个信号里没有足够信息去替模型判断"该不该死"。

→ **结论不是"再加一条熔断"，而是"把这件事告诉模型"**：
   引擎只讲清"你绕了几步、一直在同一片页面里"，收不收尾仍由模型决定。
   见 `src/bagent/agent.py` 的 OSCILLATION_* 常量区。

## 口径（和本项目那次错误推断划清界限）

纯函数，输入是已经写死在 trace 里的指纹序列，没有不确定性。
但它**只回答"这段序列会不会触发"**，回答不了"触发之后模型会怎么走"。
所以它只能用来**筛掉明显分不开的规则 / 参数**；选出来的那条仍然必须改完真跑。

（这也是为什么本项目那张"受害 100% 抓到了"的表不能直接当成功：
上一轮就是拿"旧轨迹重算"去宣布改动安全，然后被一次真实基线打脸。）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ⚠️ 本脚本的结论行带 emoji，而 Windows 控制台默认是 GBK(cp936)，写不出 ⚠️。
# 不管的话 `print` 会在**最后一行**抛 UnicodeEncodeError，表格都打完了却以退出码 1 结束
# —— 看起来像"脚本挂了"，实际是编码问题。所以这里把 stdout 固定成 UTF-8。
# （box-drawing 的 ─│┌ 在 cp936 里是有的，所以本项目的其他脚本没暴露这个问题。）
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS = PROJECT_ROOT / "runs"

# counter 任务考的是"识别出做不到并主动说出来"。这个区分很关键：
# 误触落在 counter 上未必是坏事（提前收束反而可能逼出正确拒答），
# 落在 normal 上才是真伤（逼出一个没查完的答案）。
COUNTER_TASKS = {"t05", "t06"}


# --------------------------------------------------------------------------
# 候选规则：签名统一为 (fps, n, w) -> 首次触发步(1-based) 或 None
# --------------------------------------------------------------------------
def rule_a(fps: list[str], n: int, w: int) -> int | None:
    """同页连续：相邻两帧相同就 +1。"""
    cur = 0
    for i in range(2, len(fps) + 1):
        cur = cur + 1 if fps[i - 1] == fps[i - 2] else 0
        if cur >= n:
            return i
    return None


def rule_b(fps: list[str], n: int, w: int) -> int | None:
    """未见新页：指纹以前出现过就 +1，没见过就清零。"""
    seen: set[str] = set()
    cur = 0
    for i, fp in enumerate(fps, start=1):
        if fp in seen:
            cur += 1
        else:
            seen.add(fp)
            cur = 0
        if cur >= n:
            return i
    return None


def _edge_counts(window: list[str]) -> int:
    """窗口内最多的"同一条有向边"出现次数。"""
    counts: dict[tuple[str, str], int] = {}
    for j in range(1, len(window)):
        e = (window[j - 1], window[j])
        counts[e] = counts.get(e, 0) + 1
    return max(counts.values(), default=0)


def rule_c(fps: list[str], n: int, w: int) -> int | None:
    """近 w 步无新页，且窗口内不同页面 <= 2，且窗口内有重复的边。"""
    for i in range(1, len(fps) + 1):
        lo = max(0, i - w)
        window, before = fps[lo:i], set(fps[:lo])
        if len(window) < w:
            continue
        if any(fp not in before for fp in window):
            continue
        if len(set(window)) <= 2 and _edge_counts(window) >= 2:
            return i
    return None


def rule_d(fps: list[str], n: int, w: int) -> int | None:
    """重走同一条边（全程累计，不加窗）。"""
    counts: dict[tuple[str, str], int] = {}
    for i in range(2, len(fps) + 1):
        e = (fps[i - 2], fps[i - 1])
        counts[e] = counts.get(e, 0) + 1
        if counts[e] >= n:
            return i
    return None


def rule_e(fps: list[str], n: int, w: int) -> int | None:
    """近 w 步不同页面 <= n，且无新页（不看边）。"""
    for i in range(1, len(fps) + 1):
        lo = max(0, i - w)
        window, before = fps[lo:i], set(fps[:lo])
        if len(window) < w:
            continue
        if any(fp not in before for fp in window):
            continue
        if len(set(window)) <= n:
            return i
    return None


def rule_f1(fps: list[str], n: int, w: int) -> int | None:
    """加窗版 D：最近 w 步内某条边走 >= n 次，且这 w 步里没有新页面。"""
    for i in range(1, len(fps) + 1):
        lo = max(0, i - w)
        window, before = fps[lo:i], set(fps[:lo])
        if len(window) < 2:
            continue
        if any(fp not in before for fp in window):
            continue
        if _edge_counts(window) >= n:
            return i
    return None


def rule_f2(fps: list[str], n: int, w: int) -> int | None:
    """同 F1，但"新页面"的口径放宽到最近 2w 步（允许更早的探索）。"""
    for i in range(1, len(fps) + 1):
        lo = max(0, i - w)
        window = fps[lo:i]
        before = set(fps[: max(0, i - 2 * w)])
        if len(window) < 2:
            continue
        if any(fp not in before for fp in window):
            continue
        if _edge_counts(window) >= n:
            return i
    return None


def rule_g(fps: list[str], n: int, w: int) -> int | None:
    """近 w 步不同页面 <= n，且无新页面（= E，保留独立命名便于叙述）。"""
    return rule_e(fps, n, w)


RULES = {
    "A": (rule_a, [3, 4, 5, 6]),
    "B": (rule_b, [3, 4, 5, 6]),
    "C": (rule_c, [2, 3]),
    "D": (rule_d, [2, 3, 4]),
    "E": (rule_e, [1, 2]),
    "F1": (rule_f1, [2, 3]),
    "F2": (rule_f2, [2, 3]),
    "G": (rule_g, [1, 2]),
}
WINDOWS = [4, 5, 6, 8, 10]


# --------------------------------------------------------------------------
def classify(trace: dict) -> str:
    err = trace.get("error") or ""
    if "步数上限" in err:
        return "耗尽预算"
    if "熔断" in err:
        return "熔断"
    if err:
        return "其他错误"
    return "已收敛" if trace.get("finished") else "未结束"


def load_traces() -> list[dict]:
    rows = []
    for p in sorted(RUNS.glob("*/trace.json"), key=lambda q: q.stat().st_mtime):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        recs = d.get("records") or []
        fps = [r.get("page_fp") or "" for r in recs]
        # 只算**带指纹字段**的轨迹：早于这次改动的 trace 里没有 page_fp，
        # 拿空列表去算只会得到"没触发"，那会得出"新规则很安全"的假结论。
        if not any(fps):
            continue
        rows.append({"tag": p.parent.name, "id": p.parent.name.split("-")[-1], "fps": fps, "kind": classify(d)})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="为振荡型循环挑判据")
    parser.add_argument("--detail", help="只输出这套规则/参数的逐条明细，如 F1")
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--w", type=int, default=None)
    args = parser.parse_args()

    rows = load_traces()
    hard = [r for r in rows if r["kind"] in ("耗尽预算", "熔断")]
    ok = [r for r in rows if r["kind"] == "已收敛"]
    print(f"轨迹 {len(rows)} 条：可能受害（耗尽预算/熔断）{len(hard)}，已收敛 {len(ok)}")
    print("受害 = 该抓的；已收敛 = 不该误伤的。gain = 受害触发率 - 收敛触发率\n")

    combos: list[tuple[str, int, int, int, int, int, int, int | None]] = []
    print(f"{'规则':<5}{'(n,w)':<9}{'受害':>7}{'收敛误触':>9}{'normal':>8}{'counter':>8}{'最晚首次':>9}   gain")
    print("-" * 74)
    for name, (fn, ns) in RULES.items():
        for n in ns:
            for w in WINDOWS:
                hard_hits = [fn(r["fps"], n, w) for r in hard]
                nh = sum(1 for x in hard_hits if x)
                fired = [(r, x) for r in ok if (x := fn(r["fps"], n, w))]
                mn = sum(1 for r, _ in fired if r["id"] not in COUNTER_TASKS)
                mc = len(fired) - mn
                late = max([x for x in hard_hits if x], default=None)
                rh, ro = (nh / len(hard) if hard else 0), (len(fired) / len(ok) if ok else 0)
                combos.append((name, n, w, nh, len(fired), mn, mc, late))
                print(
                    f"{name:<5}{f'({n},{w})':<9}{f'{nh}/{len(hard)}':>7}{f'{len(fired)}/{len(ok)}':>9}"
                    f"{mn:>8}{mc:>8}{str(late or '-'):>9}   {rh - ro:+.2f}"
                )

    combos.sort(key=lambda t: -(t[3] / len(hard) - t[4] / len(ok)) if hard and ok else 0)
    name, n, w, nh, no, mn, mc, late = combos[0]
    print("-" * 74)
    print(f"最佳：{name}(n={n}, w={w})  受害 {nh}/{len(hard)}  收敛误触 {no}/{len(ok)}"
          f"（normal {mn} / counter {mc}）  最晚首次 第 {late} 步")
    print()
    print("⚠️ 结论：**没有一条分得开。** 受害运行能全抓，但已收敛的运行里也有 "
          f"{no}/{len(ok)} 会被误触，其中 {mn} 条是 normal。")
    print("   → 所以它不能当熔断判据，只能当'告诉模型的一件事'。")
    print("   理由与落地方式见 src/bagent/agent.py 的 OSCILLATION_* 常量区。")

    if args.detail:
        fn, _ = RULES[args.detail]
        dn = args.n or 4
        dw = args.w or 5
        print(f"\n=== {args.detail}(n={dn}, w={dw}) 逐条明细 ===")
        for r in rows:
            hit = fn(r["fps"], dn, dw)
            if not hit:
                continue
            kind = "受害" if r["kind"] in ("耗尽预算", "熔断") else r["kind"]
            flag = ""
            if r["kind"] == "已收敛":
                flag = "  ← 误触" + ("（normal，要盯）" if r["id"] not in COUNTER_TASKS else "（counter，未必坏）")
            print(f"  {r['tag']:<24} {r['id']:<5} {len(r['fps']):>3} {kind:<8} 第 {hit} 步{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

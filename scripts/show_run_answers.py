"""把一次评测的**每次运行**摊开：答案原文 + 判分 + 证据核对状态。

用法：
    python scripts/show_run_answers.py                    # 最近一份 eval_*.json
    python scripts/show_run_answers.py runs/eval_xxx.json # 指定某一份
    python scripts/show_run_answers.py --only t04         # 只看这几条任务
    python scripts/show_run_answers.py --only-suspect     # 只看"可疑"的那些

## 为什么需要它（和 show_failed_runs.py 是互补的两半）

`show_failed_runs.py` 看的是**失败**：终止路径、动作序列、熔断原因。
这个脚本看的是**通过的那些，到底是不是真的通过**。

它是一次真实事故逼出来的：加了振荡警告之后基线从 8/12 涨到 12/12，
看着很漂亮。但同一份报告里"结论带页面证据"从 5/12 掉到 1/12 —— 对不上。
把答案原文摊开才看见：t04（正常任务）4 次里有 3 次答的是
"无法找到搜索结果"，全被记成了成功。

根因是 t04 的 `must_contain` 是空的，`judge()` 对"没有锚点的 normal 任务"
只要求答案非空。**光看总分永远发现不了这件事**，必须把答案原文摆出来。

所以这个脚本会主动标出"可疑"的行：
  · 判通过、但答案是拒答措辞（正常任务不该这样）
  · 判通过、但 `grounded=False`（结论没有页面证据支撑）
  · 判通过、但没跑完 / 答案为空
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Windows 控制台默认 GBK，写不出 emoji/特殊符号会在 print 时抛 UnicodeEncodeError。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "runs"

# 直接 `python scripts/xxx.py` 时 sys.path[0] 是 scripts/，src 不在路径上。
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# ⚠️ "拒答"词表**只有一份来源**（`bagent/grounding.py`）：
#   引擎拿它决定**免检**、`judge()` 拿它**判分**、这里拿它**标可疑**。
#   三处各抄一份就会悄悄长歪（实测：判分那边 13 个、引擎这边 17 个），
#   结果是自相矛盾的报告 —— 这里标可疑、那边却判通过（反过来一样糟）。
from bagent.grounding import REFUSAL_MARKERS  # noqa: E402


def latest_eval_file() -> Path:
    files = sorted(RUNS_DIR.glob("eval_*.json"))
    if not files:
        raise SystemExit(f"{RUNS_DIR} 下没有 eval_*.json —— 先跑一次 eval/run_eval.py")
    return files[-1]


def suspicion(row: dict) -> str:
    """这一行有什么值得盯的地方；空字符串 = 看着正常。

    ⚠️ 反例（counter）任务不能套用正常任务的尺子：那里的"拒答"**就是**正确答案，
    而拒答按设计是**免检证据**的（见 `grounding.py`：没法逐字引用一个不存在的东西）。
    第一版没分开，结果 8 条 counter 全被标成"没有页面证据支撑" ——
    全是假警报，而假警报多了人就学会忽略这个标记，等于没做。
    """
    answer = row.get("answer") or ""
    if not row.get("ok"):
        return ""  # 失败的行由 show_failed_runs.py 负责，这里不重复

    kind = row.get("kind") or ""

    if kind == "counter":
        # 只要求"说清理由"，不要求有证据。
        if not answer.strip():
            return "⚠ 判通过，但答案为空（拒答也得说出为什么）"
        return ""

    # normal / manual：有确定答案，下面每一条都值得盯
    if any(m in answer for m in REFUSAL_MARKERS):
        return "⚠ 判通过，但正常任务给的是拒答"
    if not answer.strip():
        return "⚠ 判通过，但答案为空"
    if not row.get("grounded"):
        return "⚠ 判通过，但结论没有页面证据支撑（grounded=False）"
    if row.get("error"):
        return f"⚠ 判通过，但带着报错：{row['error'][:30]}"
    return ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="摊开每次运行的答案与判分")
    parser.add_argument("eval_file", nargs="?", help="eval_*.json 路径，默认取最近一份")
    parser.add_argument("--only", nargs="*", help="只看这些任务 id")
    parser.add_argument("--only-suspect", action="store_true", help="只看被标可疑的行")
    parser.add_argument("--width", type=int, default=90, help="答案截断长度")
    args = parser.parse_args(argv)

    path = Path(args.eval_file) if args.eval_file else latest_eval_file()
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("runs") or []
    if not rows:
        raise SystemExit(f"{path} 里没有 runs")

    print(f"报告 {path.name}")
    print(f"模型 {data.get('model')} ｜ 步数预算 {data.get('max_steps')}"
          f"（来自 {data.get('max_steps_source')}）｜ 每任务 {data.get('repeat')} 次")
    print()

    shown = 0
    suspect_total = 0
    for r in rows:
        tid = r.get("id")
        if args.only and tid not in args.only:
            continue
        flag = suspicion(r)
        if flag:
            suspect_total += 1
        if args.only_suspect and not flag:
            continue
        shown += 1
        verdict = "通过" if r.get("ok") else "失败"
        print(f"{tid}  {r.get('kind'):<7} {verdict}  步数={r.get('steps'):>3}  "
              f"grounded={str(bool(r.get('grounded'))):<5}  {r.get('reason') or r.get('error') or ''}")
        if flag:
            print(f"    {flag}")
        ans = (r.get("answer") or "").replace("\n", " ")
        print(f"    answer: {ans[:args.width] or '（空）'}")
        print(f"    轨迹: {r.get('run_dir')}")
        print()

    print(f"显示 {shown} 条；其中可疑 {suspect_total} 条。")
    if suspect_total:
        print("⚠️ 可疑 != 一定有问题，但**每一行都要人眼看过**再引用总分 ——")
        print("   本项目的 12/12 就是这么被拆穿的：总分没问题，答案原文有问题。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

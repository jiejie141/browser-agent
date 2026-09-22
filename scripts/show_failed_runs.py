"""按**终止路径**看失败运行的轨迹形态。

用法：
    python scripts/show_failed_runs.py                      # 最近一份 eval_*.json
    python scripts/show_failed_runs.py runs/eval_xxx.json   # 指定某一份
    python scripts/show_failed_runs.py --only t06 t02       # 只看这两条任务

## 为什么需要它

"没通过"是一个笼统的桶。同一个失败数字底下可能压着完全不同的原因：
步数上限耗尽、熔断、动作解析非法、结论写错……**光看成功率分不出来**，
而分不出来就只能靠猜，靠猜就会把归因写错（本项目已经写错过一次：
把 t06 的失败记成"预算不够"，实际是"没有停下来的理由"，见 README 第八节 8.12）。

所以这个脚本只做一件事：把失败运行**逐条摊开**，带上终止路径、动作序列、
停滞计数，以及失败那一步的原始报错。判因果靠这张表，不靠印象。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "runs"


def latest_eval_file() -> Path:
    files = sorted(RUNS_DIR.glob("eval_*.json"))
    if not files:
        raise SystemExit(f"{RUNS_DIR} 下没有 eval_*.json —— 先跑一次 eval/run_eval.py")
    return files[-1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="按终止路径查看失败运行的轨迹")
    parser.add_argument("eval_file", nargs="?", help="eval_*.json 路径，默认取最近一份")
    parser.add_argument("--only", nargs="*", help="只看这些任务 id")
    args = parser.parse_args(argv)

    path = Path(args.eval_file) if args.eval_file else latest_eval_file()
    data = json.loads(path.read_text(encoding="utf-8"))

    # 预算跟着报告一起走 —— 同一份数字在不同预算下含义不同（见 config.py 的说明）。
    print(f"# {path.name}")
    print(
        f"# 预算 {data.get('max_steps', '未记录')}"
        f"（来源 {data.get('max_steps_source', '未记录')}）"
        f" ｜ 模型 {data.get('model', '未记录')} ｜ repeat {data.get('repeat')}"
    )
    stats = data.get("stats") or {}
    print(
        f"# 计分 {stats.get('passed')}/{stats.get('total')}"
        f" ｜ 平均步数 {stats.get('avg_steps')}"
    )

    tasks = set(args.only) if args.only else None
    shown = 0
    for r in data.get("runs", []):
        if r.get("ok"):
            continue
        if tasks and r["id"] not in tasks:
            continue
        run_dir = Path(r["run_dir"])
        trace_path = run_dir / "trace.json"
        print("\n" + "=" * 78)
        print(f"{run_dir.name}  任务={r['id']}  步数={r['steps']}  {r['elapsed']}s")
        print(f"终止路径 error={r['error']!r}")
        print(f"判定说明 reason={r['reason'][:80]!r}")
        if r.get("answer"):
            print(f"答案={r['answer'][:100]!r}")
        if not trace_path.exists():
            print("(trace.json 缺失)")
            shown += 1
            continue
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        print("-" * 78)
        for s in trace.get("records", []):
            action = s.get("action") or {}
            msg = (s.get("message") or "").replace("\n", " / ")[:76]
            print(
                f"{s['step']:>3} {str(action.get('action')):10} "
                f"ref={str(action.get('ref')):>4} stall={s.get('stall_count')} "
                f"ok={str(s.get('ok')):5} | {msg}"
            )
        shown += 1

    if shown == 0:
        print("\n(没有失败运行)")
    else:
        print(f"\n共 {shown} 条失败运行")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

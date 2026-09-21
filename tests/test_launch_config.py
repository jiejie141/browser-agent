"""`.vscode/launch.json` 的语法与命名校验。

## 为什么值得测

这个文件是**手改 JSON**，而且是 JSONC（允许 `//` 注释）。手改它有两个
很容易犯、又很难自己发现的错：

1. **JSON 语法错**：多一个逗号、少一个大括号 —— 编辑器会报，但如果
   在提交前没打开过它，就一路带进仓库了；
2. **配置名重复**：VS Code 的下拉框里会出现两条一模一样的名字，
   选错哪个都不知道。

而且这里有个真实踩过的坑：**校验词法时不能简单地按行删 `//`** ——
配置里有 `http://127.0.0.1:8000` 这样的 URL，一刀切会把字符串切坏。
所以本模块自己写了一个**带状态机**的剥注释函数：只在字符串外认注释。

同时钉住几条**约定**（不是语法，是团队的规矩）：
- 每个配置都要有 `cwd` 和 `PYTHONPATH`，否则换了机器就跑不起来；
- 单测类配置要把 `HEADLESS` 设成 true（runner 上没有显示器）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

LAUNCH = Path(__file__).resolve().parents[1] / ".vscode" / "launch.json"


def strip_jsonc(text: str) -> str:
    """剥掉 JSONC 的注释，但**不碰字符串内部**。

    自己写而不是用正则：`"http://127.0.0.1:8000"` 里的 `//` 必须留着，
    否则 URL 会被腰斩、JSON 直接失效。所以这里是一个小状态机，
    只在字符串之外识别 `//` 与 `/* */`。
    """
    out: list[str] = []
    i, n = 0, len(text)
    in_str = esc = False
    while i < n:
        ch = text[i]
        if in_str:
            out.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            i += 1
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
            i += 1
            continue
        if text.startswith("//", i):
            while i < n and text[i] != "\n":
                i += 1
            continue
        if text.startswith("/*", i):
            i += 2
            while i < n and not text.startswith("*/", i):
                i += 1
            i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


@pytest.fixture(scope="module")
def configs() -> list[dict]:
    return json.loads(strip_jsonc(LAUNCH.read_text(encoding="utf-8")))["configurations"]


def test_launch_json_is_valid_jsonc(configs):
    assert configs, "launch.json 里应该有配置"


def test_strip_jsonc_keeps_urls_intact():
    """这条是给上面那个状态机的：URL 里的 // 必须留下。"""
    raw = '{"u": "http://127.0.0.1:8000"}  // 注释'
    assert json.loads(strip_jsonc(raw)) == {"u": "http://127.0.0.1:8000"}


def test_config_names_are_unique(configs):
    names = [c["name"] for c in configs]
    dup = sorted({nm for nm in names if names.count(nm) > 1})
    assert not dup, f"调试配置名重复，下拉框里分不清：{dup}"


def test_every_config_has_cwd(configs):
    """`cwd` 是所有配置的硬要求：相对路径的产物（runs/、截图）都依赖它。"""
    bad = [c["name"] for c in configs if c.get("cwd") != "${workspaceFolder}"]
    assert not bad, f"这些配置缺 cwd: {bad}"


def test_entry_without_bootstrap_must_set_pythonpath(configs):
    """⚠️ 这条规则我第一版写错了，留个注记。

    第一版写的是"每个配置都要有 `PYTHONPATH`"，一跑就报 4 条缺失。
    去核实才发现是**假阳性**：本项目的入口脚本习惯**自己引导**
    （`sys.path.insert(0, PROJECT_ROOT/"src")`），所以 `main.py`、
    `scripts/smoke_browser.py`、`eval/run_eval.py` 根本不依赖外部
    `PYTHONPATH`。真正需要它的是**不自举**的那一个：`eval/compare_engines.py`。

    所以规则要写成"按脚本实际行为判定"，而不是"一刀切要求" ——
    一刀切会逼着人去给 4 个能跑的配置补冗余变量，
    下一个看到的人只会更糊涂。
    """
    root = Path(__file__).resolve().parents[1]
    offenders: list[str] = []
    for c in configs:
        program = c.get("program")
        if not program:
            continue  # 走 module（uvicorn/pytest）的配置由 PYTHONPATH 或安装决定
        rel = program.replace("${workspaceFolder}/", "")
        src_file = root / rel
        if not src_file.exists():
            offenders.append(f"{c['name']} → 入口文件不存在: {rel}")
            continue
        bootstraps = "sys.path.insert" in src_file.read_text(encoding="utf-8")
        has_pp = c.get("env", {}).get("PYTHONPATH") == "${workspaceFolder}/src"
        if not bootstraps and not has_pp:
            offenders.append(f"{c['name']} → {rel} 不自举 sys.path 又没设 PYTHONPATH")
    assert not offenders, "这些入口在别的机器上会 import 不到 bagent:\n  " + "\n  ".join(offenders)


def test_module_configs_have_pythonpath(configs):
    """走 `module` 的（uvicorn / pytest）没法自举，必须靠 PYTHONPATH。"""
    bad = [
        c["name"]
        for c in configs
        if c.get("module")
        and c.get("env", {}).get("PYTHONPATH") != "${workspaceFolder}/src"
    ]
    assert not bad, f"这些 module 配置缺 PYTHONPATH: {bad}"


def test_pytest_configs_are_headless(configs):
    """runner 上没有显示器，跑单测的配置必须无头。"""
    bad = [
        c["name"]
        for c in configs
        if c.get("module") == "pytest" and c["env"].get("HEADLESS") != "true"
    ]
    assert not bad, f"这些测试配置没设 HEADLESS=true: {bad}"


def test_pytest_targets_exist(configs):
    """`⑦b` 那种带 -v 的定向配置里，写死的用例路径必须真的存在。"""
    root = Path(__file__).resolve().parents[1]
    bad: list[str] = []
    for c in configs:
        if c.get("module") != "pytest":
            continue
        for arg in c.get("args", []):
            if not arg.startswith("tests/"):
                continue
            target = arg.split("::")[0]
            if not (root / target).exists():
                bad.append(f"{c['name']} → 不存在的 {target}")
    assert not bad, "\n  ".join(bad)

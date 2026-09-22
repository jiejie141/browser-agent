"""配置「来源」的回归测试。

## 为什么值得单独钉住

本项目真的踩过一次：`config.py` 把 `max_steps` 的默认值从 20 改成 30 之后，
本机 `.env` 里那句遗留的 `MAX_STEPS=20` **仍然赢**（`load_dotenv` 不覆盖时
是反过来，但这里 `.env` 的值本来就先于默认值生效）。结果是：

- 代码看着改对了，报告的头部也照常打印"步数预算 20"；
- 跑了十分钟、翻轨迹才看到 `达到最大步数上限（20）`；
- 差一点就把这份 20 预算下的基线当成 30 的报出去，并把
  "预算没变" 读成 "这次修复没用"。

教训是：**参数值会改变结论，所以它从哪儿来也是结论的一部分。**
光打印生效值不够 —— 生效值和代码默认值相同/不同，是两件事。

这里钉四件事：
1. 优先级：进程环境变量 > `.env` 文件 > 代码默认值；
2. 来源如实回报，且**写坏的值不许装作生效**（`MAX_STEPS=abc` 要说明回退）；
3. `Settings` 的默认值确实取自那个共享常量，不是另一处硬编码；
4. 仓库里提交的 `.env.example` 与代码默认值**不许漂移**（CI 也查得到）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from bagent import config  # noqa: E402

import run_eval  # noqa: E402


@pytest.fixture()
def clean_sources(monkeypatch):
    """把两个来源快照都清空，模拟"什么都没有设置"的干净环境。"""
    monkeypatch.setattr(config, "_PROCESS_ENV_SNAPSHOT", {})
    monkeypatch.setattr(config, "_ENV_FILE_VALUES", {})


def test_falls_back_to_code_default(clean_sources):
    value, source = config.resolve_int("MAX_STEPS", 30)
    assert value == 30
    assert source == "代码默认值"


def test_env_file_is_reported_as_env_file(clean_sources, monkeypatch):
    monkeypatch.setattr(config, "_ENV_FILE_VALUES", {"MAX_STEPS": "20"})
    value, source = config.resolve_int("MAX_STEPS", 30)
    assert value == 20
    assert source == ".env 文件"


def test_process_env_beats_env_file(monkeypatch):
    """真环境优先于 .env —— 部署时显式传入的值不能被本机副本盖掉。"""
    monkeypatch.setattr(config, "_PROCESS_ENV_SNAPSHOT", {"MAX_STEPS": "42"})
    monkeypatch.setattr(config, "_ENV_FILE_VALUES", {"MAX_STEPS": "20"})
    value, source = config.resolve_int("MAX_STEPS", 30)
    assert value == 42
    assert source == "环境变量"


def test_unparseable_value_does_not_look_applied(clean_sources, monkeypatch):
    """`MAX_STEPS=abc` 必须说明"回退默认值"，否则写坏的值看起来像生效了。"""
    monkeypatch.setattr(config, "_ENV_FILE_VALUES", {"MAX_STEPS": "abc"})
    value, source = config.resolve_int("MAX_STEPS", 30)
    assert value == 30
    assert ".env 文件" in source
    assert "不可解析" in source


def test_settings_default_comes_from_the_shared_constant(clean_sources, monkeypatch):
    """默认值必须来自 MAX_STEPS_DEFAULT，不能是另一处硬编码的 20/30。

    这一条是防"改了一处、忘了另一处"：把默认值抽成常量就是为了消掉
    第二个写数字的地方，这里确保 `Settings` 真的在用它。
    """
    monkeypatch.delenv("MAX_STEPS", raising=False)
    assert config.Settings().max_steps == config.MAX_STEPS_DEFAULT


def test_env_example_does_not_drift_from_code_default():
    """仓库里提交的 `.env.example` 必须与代码默认值同档。

    `.env` 是本机副本、不进版本库，CI 上不存在；`.env.example` 才是
    新人 clone 之后照着填的那份。两者不一致 = 新人拿到的预算和文档说的不一样。
    """
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    wanted = f"MAX_STEPS={config.MAX_STEPS_DEFAULT}"
    assert wanted in text, f".env.example 里没有 {wanted}"


class TestResolveBudget:
    """`run_eval.resolve_budget` 是报告里那句"步数预算"的唯一来源。"""

    def test_cli_wins_and_says_so(self):
        value, source = run_eval.resolve_budget(7)
        assert value == 7
        assert "命令行" in source

    def test_delegates_when_cli_absent(self, clean_sources):
        value, source = run_eval.resolve_budget(None)
        assert value == config.MAX_STEPS_DEFAULT
        assert source == "代码默认值"

    def test_reports_env_override(self, clean_sources, monkeypatch):
        monkeypatch.setattr(config, "_ENV_FILE_VALUES", {"MAX_STEPS": "20"})
        value, source = run_eval.resolve_budget(None)
        assert value == 20
        assert source == ".env 文件"

"""doctor 的回归测试。

这组用例是一次事故的产物，所以先说清事故本身：

`--mock` 是**命令行开关**，而 `doctor()` 原先只读环境变量 `MOCK`。
CI 里执行的是 `python main.py --doctor --mock`，一个环境变量都没设 →
`mock=False` → 自检走进"在线模式"分支 → 去构造一个没有 key 的客户端 →
`openai.OpenAIError`（它不是 `LLMError`，接不住）→ 自检自己崩掉。
结果：CI 连续 3 次全红，而 `doctor` 当时**一条测试都没有**。

三条用例分别钉住这个问题的三个侧面：
  1. CI 的调用方式（只有开关、没有环境变量）必须判为通过；
  2. 容器镜像的调用方式（靠 `ENV MOCK=true`）要继续有效；
  3. 既没 key 也没 mock 时，要**如实报成检查项失败**，而不是抛异常崩掉。
"""

from __future__ import annotations

import bagent.cli as cli
from bagent.config import Settings


def _settings_without_key() -> Settings:
    """模拟"仓库里没有 .env"的干净环境。

    必须显式清空：config.py 在 import 时就会 load_dotenv()，
    开发机上真实存在 .env，所以 Settings() 会带着真 key，
    不覆盖的话这三条用例在本地永远测不出 CI 的失败。
    """
    s = Settings()
    s.llm_api_key = ""
    s.engine = "handwritten"
    return s


def _no_browser(monkeypatch):
    """把浏览器探针换掉：单元测试不该真起 Chromium。"""
    monkeypatch.setattr(cli, "_probe_browser", lambda: None)


def test_doctor_mock_flag_passes_without_api_key(monkeypatch):
    """CI 的调用方式：只给 --mock 开关，不设 MOCK 环境变量 —— 必须退出码 0。"""
    monkeypatch.delenv("MOCK", raising=False)
    monkeypatch.delenv("OFFLINE", raising=False)
    _no_browser(monkeypatch)
    assert cli.doctor(_settings_without_key(), mock=True) == 0


def test_doctor_env_var_mock_still_works(monkeypatch):
    """环境变量这条路要继续有效 —— 容器镜像里就是靠 ENV MOCK=true。"""
    monkeypatch.setenv("MOCK", "true")
    _no_browser(monkeypatch)
    assert cli.doctor(_settings_without_key()) == 0


def test_doctor_reports_missing_key_instead_of_crashing(monkeypatch):
    """没 key 又没 mock：应当报成"检查项失败"（返回 1），而不是抛异常。

    这条要是曾经存在，CI 就不会连红三次 —— 它断言的正是那个
    "OpenAIError 从 except LLMError 旁边溜过去"的行为。
    """
    monkeypatch.delenv("MOCK", raising=False)
    monkeypatch.delenv("OFFLINE", raising=False)
    _no_browser(monkeypatch)
    assert cli.doctor(_settings_without_key()) == 1

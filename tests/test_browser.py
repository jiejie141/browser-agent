"""浏览器会话收尾的超时护栏。

为什么单独测这个：**收尾卡住不是"慢"，而是"把已完成的任务显示成未完成"。**

实测现场（本机）：
- CLI：任务 4.3 秒跑完 5 步、截图也落盘了，进程却在收尾上卡了 4 分钟不退出；
- API：终态原本写在收尾之后 → 任务永远停在 running，控制台一直转圈，
  150 秒都没等到终态，尽管每一步都早已成功。

所以这里守两件事：
1. 收尾慢/卡住时，`__aexit__` 必须在有限时间内返回（不能拖住调用方）；
2. 收尾正常时，资源仍然要被真的关掉（别为了防卡住把功能删了）。
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from bagent.browser import BrowserSession
from bagent.config import Settings


class _HangingCloser:
    """close()/stop() 永不返回，模拟卡住的收尾。"""

    def __init__(self) -> None:
        self.close_calls = 0
        self.stop_calls = 0

    async def close(self) -> None:
        self.close_calls += 1
        await asyncio.sleep(999)

    async def stop(self) -> None:
        self.stop_calls += 1
        await asyncio.sleep(999)


class _FastCloser:
    def __init__(self) -> None:
        self.close_calls = 0
        self.stop_calls = 0

    async def close(self) -> None:
        self.close_calls += 1

    async def stop(self) -> None:
        self.stop_calls += 1


def _settings(timeout: float) -> Settings:
    st = Settings()
    st.teardown_timeout_seconds = timeout
    return st


def test_aexit_returns_within_budget_when_close_hangs():
    """收尾卡住时必须在预算内返回，而不是陪着一起卡。"""
    hang = _HangingCloser()
    sess = BrowserSession(_settings(0.3), Path("."))
    sess._context = hang
    sess._browser = hang
    sess._pw = hang

    t0 = time.monotonic()
    asyncio.run(asyncio.wait_for(sess.__aexit__(None, None, None), timeout=10))
    elapsed = time.monotonic() - t0

    assert elapsed < 5, f"收尾没有超时护栏，耗时 {elapsed:.1f}s"
    assert hang.close_calls >= 1, "至少应该尝试关过 context"


def test_aexit_closes_resources_normally():
    """正常路径下三个资源都要被关掉 —— 别为了防卡住把清理删了。"""
    fast = _FastCloser()
    sess = BrowserSession(_settings(5.0), Path("."))
    sess._context = fast
    sess._browser = fast
    sess._pw = fast

    asyncio.run(sess.__aexit__(None, None, None))

    assert fast.close_calls == 2, "context 与 browser 都应被关闭"
    assert fast.stop_calls == 1, "playwright 驱动应被停止"


def test_aexit_tolerates_none_resources():
    """没启动成功就退出（__aenter__ 抛错）时不能二次崩溃。"""
    sess = BrowserSession(_settings(1.0), Path("."))
    asyncio.run(sess.__aexit__(None, None, None))  # 三个字段都是 None


def test_teardown_timeout_reads_env(monkeypatch):
    """超时必须可配置：CI 上收尾很快不需要等，某些环境要等更久。"""
    monkeypatch.setenv("TEARDOWN_TIMEOUT_SECONDS", "3.5")
    assert Settings().teardown_timeout_seconds == 3.5


def test_teardown_timeout_default_and_bad_value(monkeypatch):
    monkeypatch.delenv("TEARDOWN_TIMEOUT_SECONDS", raising=False)
    assert Settings().teardown_timeout_seconds == 8.0
    monkeypatch.setenv("TEARDOWN_TIMEOUT_SECONDS", "不是数字")
    assert Settings().teardown_timeout_seconds == 8.0

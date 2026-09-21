"""配置加载。

设计原则：**代码里永远不出现密钥**。所有敏感信息从 .env 读，
.env 被 .gitignore 排除，仓库里只放 .env.example。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# 项目根目录：src/bagent/config.py -> src/bagent -> src -> 项目根
PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")


def _get(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _get_int(name: str, default: int) -> int:
    raw = _get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    raw = _get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _get_bool(name: str, default: bool) -> bool:
    raw = _get(name).lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return default


@dataclass
class Settings:
    """运行期配置。"""

    # ---- 主模型（负责规划动作）----
    llm_api_key: str = field(default_factory=lambda: _get("LLM_API_KEY"))
    llm_base_url: str = field(
        default_factory=lambda: _get("LLM_BASE_URL", "https://api.deepseek.com/v1")
    )
    llm_model: str = field(default_factory=lambda: _get("LLM_MODEL", "deepseek-chat"))

    # ---- 视觉模型（DOM 通道失效时的降级路径，可选）----
    vlm_api_key: str = field(default_factory=lambda: _get("VLM_API_KEY"))
    vlm_base_url: str = field(default_factory=lambda: _get("VLM_BASE_URL"))
    vlm_model: str = field(default_factory=lambda: _get("VLM_MODEL"))

    # ---- 运行参数 ----
    max_steps: int = field(default_factory=lambda: _get_int("MAX_STEPS", 20))
    step_timeout_seconds: int = field(
        default_factory=lambda: _get_int("STEP_TIMEOUT_SECONDS", 30)
    )
    # 默认无头。这个默认值是被 CI 教出来的：
    # 原来默认 False（有头），本地有显示器所以看不出问题，而 GitHub Actions
    # 的 runner 没有 X server，Playwright 直接报
    # "launched a headed browser without having a XServer running" 然后退出。
    # 无头才是"脚本/服务/容器"场景下的安全默认；想看着浏览器跑，
    # 复制 .env.example（里面写了 HEADLESS=false）或用 --headful。
    headless: bool = field(default_factory=lambda: _get_bool("HEADLESS", True))
    log_level: str = field(default_factory=lambda: _get("LOG_LEVEL", "INFO").upper())

    # 浏览器收尾（context.close / browser.close / playwright.stop）的整体超时上限，秒。
    # 之所以要有这个值：收尾在某些环境下会长时间不返回（本机实测卡到分钟级，
    # 同一个任务在 CI 上 1.7 秒就干净退出），而收尾**不属于任务本身**——
    # 结果早已落盘，却让调用方以为任务还没结束（Web 控制台就一直转圈）。
    # 超时即放弃等待，宁可留个待回收的进程，也不要让"已完成"被无关步骤掩盖。
    teardown_timeout_seconds: float = field(
        default_factory=lambda: _get_float("TEARDOWN_TIMEOUT_SECONDS", 8.0)
    )

    # ---- Agent 引擎 ----
    # handwritten：agent.py 里的手写 ReAct 循环（默认，零额外依赖）
    # langgraph：graph_agent.py 里的 LangGraph StateGraph 实现
    # 两者共用同一套 perceive / 动作层 / 提示词，可用同一份评测集做 A/B 对比
    engine: str = field(default_factory=lambda: _get("ENGINE", "handwritten").lower())

    # ---- 浏览器出口（代理）----
    # 为什么浏览器要单独配一份：系统代理是"全局"的，但浏览器该不该走代理
    # 取决于目标站点，不能一刀切。真实约束：国内站点经境外节点会被拒
    # （实测淘宝只回空壳页、百度直接断连），而 GitHub / Google 不走代理又连不上。
    # 所以这里给 browser_proxy + browser_proxy_bypass 两个旋钮：
    # 默认都留空 = 跟随浏览器自身的默认（通常是系统代理），行为与改版前一致。
    # 例：
    #   BROWSER_PROXY=http://127.0.0.1:7890
    #   BROWSER_PROXY_BYPASS=taobao.com,baidu.com,weibo.com
    browser_proxy: str = field(default_factory=lambda: _get("BROWSER_PROXY"))
    browser_proxy_bypass: str = field(
        default_factory=lambda: _get("BROWSER_PROXY_BYPASS")
    )

    # ---- 离线替身 ----
    # True 表示本次运行不调用真实模型：Agent 据此把成本计为 0
    # （离线替身的 token 是按字符估的，乘单价会算出一个并不存在的花费）。
    # 注意它属于"运行期状态"而不只是启动配置：API 每次任务都会按请求覆写它，
    # 所以必须在 Settings 上真的存在，光靠 _get("MOCK") 读环境变量是不够的。
    mock: bool = field(default_factory=lambda: _get_bool("MOCK", False))

    # ---- 目录 ----
    runs_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "runs")
    tasks_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "tasks")

    # ---- 成本单价（元 / 百万 token），用于成本面板估算 ----
    # 默认按 deepseek-chat 的公开价目表填，换模型时改这里
    price_in_per_mtok: float = 1.0
    price_out_per_mtok: float = 2.0

    @property
    def vlm_enabled(self) -> bool:
        """视觉通道是否可用。没配 key 就自动关闭，不报错。"""
        return bool(self.vlm_api_key and self.vlm_model)

    def validate(self) -> None:
        """启动前自检，尽早失败比运行到一半炸掉好。"""
        if not self.llm_api_key:
            raise RuntimeError(
                "缺少 LLM_API_KEY。请复制 .env.example 为 .env 并填入你的 key。"
            )
        if not self.llm_base_url:
            raise RuntimeError("缺少 LLM_BASE_URL。")
        if not self.llm_model:
            raise RuntimeError("缺少 LLM_MODEL。")


_settings: Settings | None = None


def get_settings(refresh: bool = False) -> Settings:
    """全局单例，避免到处重复读文件。"""
    global _settings
    if _settings is None or refresh:
        _settings = Settings()
    return _settings

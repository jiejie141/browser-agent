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
    headless: bool = field(default_factory=lambda: _get_bool("HEADLESS", False))
    log_level: str = field(default_factory=lambda: _get("LOG_LEVEL", "INFO").upper())

    # ---- Agent 引擎 ----
    # handwritten：agent.py 里的手写 ReAct 循环（默认，零额外依赖）
    # langgraph：graph_agent.py 里的 LangGraph StateGraph 实现
    # 两者共用同一套 perceive / 动作层 / 提示词，可用同一份评测集做 A/B 对比
    engine: str = field(default_factory=lambda: _get("ENGINE", "handwritten").lower())

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

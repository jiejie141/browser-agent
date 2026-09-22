"""配置加载。

设计原则：**代码里永远不出现密钥**。所有敏感信息从 .env 读，
.env 被 .gitignore 排除，仓库里只放 .env.example。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values, load_dotenv

# 项目根目录：src/bagent/config.py -> src/bagent -> src -> 项目根
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# **必须在 load_dotenv 之前快照**。load_dotenv 默认不覆盖已存在的环境变量，
# 所以加载之后 os.environ 里就分不清"这个值是真环境给的"还是".env 给的"了。
# 而这两者的含义完全不同：真环境是部署时显式传入的，.env 是本机遗留的副本 ——
# 本机这份就曾经把 MAX_STEPS 悄悄按在 20，而代码默认值早已改成 30。
_PROCESS_ENV_SNAPSHOT = dict(os.environ)

load_dotenv(PROJECT_ROOT / ".env")

try:
    _ENV_FILE_VALUES: dict[str, str | None] = dict(
        dotenv_values(PROJECT_ROOT / ".env") or {}
    )
except Exception:  # 文件不存在/语法坏了都不该让 import 崩掉
    _ENV_FILE_VALUES = {}


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


def resolve_int(name: str, default: int) -> tuple[int, str]:
    """解析一个整数配置，并**如实回报它是从哪儿来的**。

    为什么要回报来源：生效值相同的两份配置，含义可以完全不同。
    本机实测踩过一次——`config.py` 把 `max_steps` 默认值改成 30 之后，
    `.env` 里那句遗留的 `MAX_STEPS=20` 仍然赢，于是"改了代码"和
    "改了配置"完全看不出区别，跑到一半才发现基线还在旧预算上。
    参数值会改变结论，那么**它的来源也是结论的一部分**。

    来源只有三种，优先级从高到低：进程环境变量 > .env 文件 > 代码默认值。
    返回 (生效值, 来源说明)；调用方要自己把来源打出来或落盘。

    值存在但解析不出来（例如 `MAX_STEPS=abc`）时，生效值退回 `default`，
    **但来源照实写成"值不可解析"** —— 不能让一个写坏的值看起来像生效了。
    """
    snap = (_PROCESS_ENV_SNAPSHOT.get(name) or "").strip()
    if snap:
        parsed = _parse_int(snap)
        if parsed is not None:
            return parsed, "环境变量"
        return default, f"环境变量（{name}={snap!r} 不可解析，已回退默认值）"
    file_val = (_ENV_FILE_VALUES.get(name) or "").strip()
    if file_val:
        parsed = _parse_int(file_val)
        if parsed is not None:
            return parsed, ".env 文件"
        return default, f".env 文件（{name}={file_val!r} 不可解析，已回退默认值）"
    return default, "代码默认值"


def _parse_int(raw: str) -> int | None:
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


# 步数预算的**代码默认值**，单独抽出来给两处用：Settings 的默认值、
# 以及报告里"这个生效值是默认值还是被覆盖了"的判断。写成两遍就等于
# 给下一次漂移留了口子（.env 与本文件各写一个数，谁也说不清哪个生效）。
MAX_STEPS_DEFAULT = 30


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
    #
    # max_steps 默认值：20 → **30**。这是一次**实测**改的，不是拍脑袋：
    #
    # 老值 20 曾经把一批"其实能收敛"的任务直接截断。把同一套代码、同一个模型
    # （Qwen2.5-7B）的 12 次运行按**终止路径**分堆之后，看清了两件事：
    #
    #   1. 三条难任务（t04/t05/t06）的 **9 次失败，9 次都是"步数上限耗尽"** ——
    #      不是熔断、不是报错，就是步数用完了；
    #   2. 而它们**通过**的那几次用了多少步：t05 是 15 / 17 / 19，t06 是 10 / 18。
    #      也就是说 20 这个值**恰好卡在它们收敛的边界上** ——
    #      过不过取决于"这次收敛在 19 步还是 21 步"，基本是掷硬币。
    #
    # 直接把预算放到 30 重跑同样 12 次：**4/12 → 10/12**。
    # t06 通过的三次分别用了 21 / 23 / 25 步 —— 全都超过老的 20。
    # 30 是在"实测最长需要 25"之上留了约 20% 余量。
    #
    # ⚠️ **代价必须一起说**，否则这个改动就是"把尺子挪了"：
    #   - 失败的任务会更贵 —— 以前 20 步就停，现在可能烧到 30 步；
    #   - 平均耗时从 ~60 秒涨到 ~75 秒（失败那些更明显）；
    #   - **跨版本对比时要看清预算**：本项目早先 15/18、14/18 两组基线
    #     都是 `MAX_STEPS=20` 下跑的，和 30 下的数字不能直接比。
    #
    # 注意 `MAX_STEPS` 是**产品参数**（Agent 愿意花多少步），不是判分参数 ——
    # 调它不改变判分口径。但正因为它是参数，报告数字时必须写明用的是哪一档。
    #
    # 为什么不干脆"按有没有进展动态延长"：停滞检测（见 agent.py）已经在做
    # 类似的事 —— 真卡住的运行会被提前熔断，不需要靠预算兜。而 t05/t06 那种
    # "每一步都在换页面、就是不给结论"的形态，判据上**不构成停滞**
    # （实测最大停滞计数只有 1~4，阈值是 6），动态延长对它无效，只有加预算有效。
    max_steps: int = field(
        default_factory=lambda: _get_int("MAX_STEPS", MAX_STEPS_DEFAULT)
    )
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

    # 遇到登录墙时，等人登录的最长时间（秒）。
    #
    # 为什么必须有个上限：登录交接是**阻塞**的 —— 引擎在等人的时候不会往下走。
    # 没有人来点"登录完成"，任务就会永远停在 waiting_login，
    # 控制台一直转圈，而它其实只是在等人。超时之后按"没登录成"处理，
    # 引擎照常给出「需要登录，无法完成」的结论。
    login_wait_seconds: float = field(
        default_factory=lambda: _get_float("LOGIN_WAIT_SECONDS", 300.0)
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

    # 站点探测时两次导航之间的最小间隔（秒）。
    # 为什么要节流：`--probe-sites` 会连着访问几十个站点，中间没有任何停顿。
    # 实测这样跑会把出口打限流 —— 一次全量探测 64 个站，最后只剩 2 个可达，
    # 而单独探其中 5 个时百度是正常的。**结论会飘，探测本身把自己搞坏了。**
    probe_min_interval_seconds: float = field(
        default_factory=lambda: _get_float("PROBE_MIN_INTERVAL_SECONDS", 1.5)
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

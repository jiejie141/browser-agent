# browser-agent

用 ReAct 循环驱动真实浏览器完成多步网页任务的 Agent。

> 不是"录好脚本回放"，而是**每一步都看真实页面、重新决策**。
> 所以弹窗、登录态过期、按钮改文案，都不会让整个流程崩掉。

---

## 一分钟上手

```powershell
# 1. 确认环境（会依次检查 Python / 配置 / 模型连通 / 浏览器）
.\.venv\Scripts\python.exe main.py --doctor

# 2. 跑一个任务（--headful 会显示浏览器窗口，方便你看到 Agent 在干什么）
.\.venv\Scripts\python.exe main.py `
  --task "打开名言网站首页，读出第一句名言的正文和作者" `
  --url "https://quotes.toscrape.com/" `
  --headful

# 3. 跑整个示例任务集
.\.venv\Scripts\python.exe main.py --task-file tasks/examples.json --headful
```

VSCode 里直接按 **F5**，选「跑一个任务 (可改参数)」也一样，
而且可以下断点单步调试。

---

## 它是怎么工作的

```
                  ┌──────────────────────────────────────┐
                  │            每一轮循环                 │
                  └──────────────────────────────────────┘
                                    │
        ┌───────────────────────────┴───────────────────────────┐
        │                                                       │
   ①  看 (Observe)                                       ④  记 (Record)
   perception.py                                          history / trace
   ├─ DOM 通道：抽可交互元素，打上编号 [1][2][3]              │
   └─ 抓不到元素时 → 截图 + 视觉模型（降级）                    │
        │                                                       │
   ②  想 (Reason)                                               │
   agent.py → llm.py                                            │
   ├─ 把「任务 + 历史 + 当前页面」拼成 prompt                     │
   └─ 模型返回一个 JSON 动作                                     │
        │                                                       │
   ③  做 (Act)                                                  │
   browser.py                                                   │
   ├─ 用 [data-ba-ref="3"] 这个选择器点击（不按坐标）              │
   └─ 敏感操作先过人工确认护栏 ──────────────────────────────────┘
```

### 三个值得记住的设计决策

**1. 按编号点击，不按坐标点击。**
坐标会被弹窗、懒加载、页面抖动搞乱；编号不会。感知层给每个可交互元素写上
`data-ba-ref` 属性，动作层用 `[data-ba-ref="3"]` 精确定位。这就是为什么
click 动作只需要一个 `ref` 参数。

**2. 必须有一条降级路径。**
总有页面 DOM 抽不出东西（Canvas 画的、iframe 套的、纯图片按钮）。
只留一条通道的 Agent，遇到这种页面就卡死。所以 DOM 一无所获时自动切视觉通道。

**3. 失败要变成信息，不是变成崩溃。**
某一步失败了，失败原因会被写回历史，并附一句"不要重复这个动作"。
连续失败 3 次则熔断退出——**能识别出"做不到"并停下来，比硬撑更重要。**

---

## 目录结构

```
browser-agent/
├── main.py                  # 入口：加 sys.path 后交给 cli
├── requirements.txt
├── .env                     # 你的密钥（已被 .gitignore 排除）
├── .env.example             # 模板，可以提交到仓库
│
├── src/bagent/
│   ├── config.py            # 配置加载 + 自检
│   ├── models.py            # 数据结构（Action / PageState / Usage）
│   ├── llm.py               # LLM 客户端：超时 + 退避重试 + token 计量
│   ├── perception.py        # 感知层：DOM 抽取 + 截图双通道
│   ├── browser.py           # 动作层：Playwright 操作 + 敏感操作护栏
│   ├── agent.py             # ReAct 主循环
│   └── cli.py               # 命令行
│
├── tasks/examples.json      # 任务集（含反例任务）
├── runs/                    # 每次运行的产物：trace.json + 截图
├── tests/                   # 单元测试
└── .vscode/                 # 一键运行 / 断点调试配置
```

---

## 没有 key / 想换厂商怎么办

只要对方是 **OpenAI 兼容接口**，改 `.env` 里三行就切过去了：

```
LLM_API_KEY=<你的 key>
LLM_BASE_URL=<厂商的兼容端点>
LLM_MODEL=<模型名>
```

`.env` 里已经写好了四套配置，取消注释即可：

| 方案 | 端点 | 模型 | 成本 |
|---|---|---|---|
| 智谱 GLM | `https://open.bigmodel.cn/api/paas/v4` | `glm-4-flash` | **永久免费** |
| 硅基流动 | `https://api.siliconflow.cn/v1` | `Qwen/Qwen2.5-7B-Instruct` | 新用户送 2000 万 token |
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat` | 百万 token 几毛钱 |
| 阿里云百炼 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-turbo` | 新用户 7000 万 token / 90 天 |

> 免费政策随时会变，以各平台官网为准。另外免费小模型的规划能力
> 确实弱一些——如果发现它老是输出不合格式的动作，那是模型能力问题，
> 不是代码问题（主循环已经会兜住非法输出并重试）。

切完跑一次自检确认：

```powershell
.\.venv\Scripts\python.exe scripts\check_llm.py
```

**如果连 key 都没有**，也不想注册：直接跑离线模式，主循环照样能验证。

```powershell
.\.venv\Scripts\python.exe main.py --mock --task "离线验证" --url "https://quotes.toscrape.com/"
```

代码里内置了 `MockLLMClient`，按剧本返回动作。这就是"把外部依赖摘出去"的价值：
**模型是唯一的外部依赖，去掉它，其余所有逻辑都还能测。**

---

## 关键参数

全部在 `.env` 里，改完重启即可：

| 变量 | 默认 | 说明 |
|---|---|---|
| `LLM_API_KEY` | — | **必填**。任何 OpenAI 兼容端点的 key |
| `LLM_BASE_URL` | `https://api.deepseek.com/v1` | 换厂商只改这里 |
| `LLM_MODEL` | `deepseek-chat` | 模型名 |
| `VLM_API_KEY` | 空 | 视觉通道的 key。**留空则自动跳过视觉降级**，只走 DOM |
| `MAX_STEPS` | 20 | 单个任务的最大步数，防死循环 |
| `HEADLESS` | false | true = 不显示浏览器窗口 |

---

## 成本是怎么算出来的

`llm.py` 里每次调用都会把 `prompt_tokens` / `completion_tokens` 累加进
`Usage` 对象。每次运行结束，CLI 会打印：

```
token        输入 8432 / 输出 1105 / 共 9537
模型调用     9 次
估算成本     ¥0.010642
```

单价在 `config.py` 的 `price_in_per_mtok` / `price_out_per_mtok` 里，
按你实际用的模型改。**注意这是估算**——真实计费以厂商账单为准。

---

## 目前的能力边界（诚实版）

已知做不到、或者做得不好的地方：

- **不做登录**。没有账号体系，遇到登录墙需要你自己先手动登录。
- **不处理验证码**。滑块、点选一律绕不过。
- **只在主 frame 里找元素**。`<iframe>` 里的内容目前看不见。
- **不解析 Shadow DOM**。Web Components 站点会走视觉通道。
- **视觉通道按整页截图理解**，不做元素级定位，所以它只能"描述"，
  最终点哪个编号还是得回到 DOM 通道。
- **没有持久化会话**。每次运行都是全新的浏览器上下文（断点续跑还在做）。

这些不是借口，是**待办清单**——也是你面试时可以主动讲的部分：
"我知道它在哪些情况下会失效，因为我都测过。"

---

## 双引擎：手写 ReAct 与 LangGraph（新增）

主循环有两个可互换的实现，用 `--engine` 切换：

```powershell
# 手写循环（默认，零额外依赖）
.\.venv\Scripts\python.exe main.py --task "读出页面标题" --url "https://example.com" --engine handwritten

# LangGraph 状态图（需 pip install langgraph langchain-core）
.\.venv\Scripts\python.exe main.py --task "读出页面标题" --url "https://example.com" --engine langgraph
```

### 为什么要两个都留着

**手写版**证明我理解循环本身：何时重试、何时熔断、怎么打断死循环。
**LangGraph 版**证明我会用主流框架，并且知道它在什么场景下值得引入。

两版共用同一套节点逻辑与终止条件，**A/B 实测完全等价**（mock 模式、同一任务）：

| | handwritten | langgraph |
|---|---|---|
| 步数 | 4 | 4 |
| 输入 token | 5300 | 5300 |
| 输出 token | 275 | 275 |
| 模型调用 | 5 | 5 |
| 成本 | ¥0.005850 | ¥0.005850 |

### 设计要点：终止条件抽成纯函数

`graph_agent.py` 里路由是纯函数，不依赖框架运行时，所以**状态机是可单测对象**：

```python
def route_after_act(state) -> Literal["perceive", "end"]:
    if state.get("finished") or state.get("error"):
        return "end"
    if state.get("step", 0) >= state.get("max_steps", 0):
        return "end"
    if state.get("consecutive_failures", 0) >= MAX_CONSECUTIVE_FAILURES:
        return "end"
    return "perceive"
```

> 踩过的坑：**LangGraph 只会传递 `TypedDict` 里声明过的 key**。
> 一开始我用 `_state` 这种下划线开头的键存页面状态，结果节点之间静默丢失，
> 报 `AttributeError: 'NoneType' object has no attribute 'render_for_prompt'`。
> 必须在 `AgentState` 里显式声明。

---

## 服务化：FastAPI（新增）

```powershell
pip install fastapi uvicorn
uvicorn bagent.api:app --reload --port 8000
```

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` | 健康检查（含 LangGraph 可用性） |
| GET | `/engines` | 可用引擎 |
| POST | `/tasks` | 提交任务，返回 **202 + task_id**（异步执行） |
| GET | `/tasks/{task_id}` | 查询任务状态与结果 |
| GET | `/tasks` | 列出最近任务 |

任务用 `BackgroundTasks` 异步跑，进程内保存最近 200 条。
因为启动浏览器是重操作，同步接口会让 HTTP 请求挂住几十秒。

---

## 容器化（新增）

```bash
docker build -t browser-agent .
docker run --shm-size=1g -p 8000:8000 browser-agent -m uvicorn bagent.api:app --host 0.0.0.0
```

**必须给 `--shm-size`**：Chromium 在容器里常因 `/dev/shm` 太小而随机崩溃
（典型报错 `Target closed` / `Page crashed`），默认 64MB 不够。

基础镜像用 Playwright 官方的 `mcr.microsoft.com/playwright/python`，它已装好 Chromium
和全部系统依赖；自己从头 `apt install` 既慢又容易漏库（缺 `libnss3` / `libatk` 时报错很
不直观）。容器内默认 `HEADLESS=true MOCK=true OFFLINE=true`。

配合仓库根目录的 `docker-compose.yml` 可与 pr-review-agent 一起编排。

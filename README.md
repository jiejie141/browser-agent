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

# 4. 跑「高频站点演示集」（抖音 / 淘宝 / 百度 / B 站 / 豆瓣 / 微博 / GitHub …）
#    真实站点，免登录即可读；--only N 只跑第 N 条
.\.venv\Scripts\python.exe main.py --task-file tasks/demo_sites.json --only 1 --headful

# 5. 实测站点表里每个搜索模板在**你当前网络下**能不能用（不需要 LLM key）
#    结果写入 runs/site_probe.json，控制台会读它来显示每个站点的真实状态
.\.venv\Scripts\python.exe main.py --probe-sites
.\.venv\Scripts\python.exe main.py --probe-sites --probe-only taobao,baidu,jd
```

VSCode 里直接按 **F5**，选「跑一个任务 (可改参数)」也一样，
而且可以下断点单步调试。

### 「想搜什么就搜什么」：站点注册表

控制台顶部是**选站 + 输关键词**，不是写死的几个搜索按钮：

```
选站 [百度 ▾]   关键词 [机械键盘          ]   [去搜索]
→ 任务：在百度搜索「机械键盘」，读出第一条搜索结果的标题。
→ 网址：https://www.baidu.com/s?wd=%E6%9C%BA%E6%A2%B0%E9%94%AE%E7%9B%98
```

骨架（`https://www.baidu.com/s?wd={q}`）是稳定的，关键词是变量。
把骨架沉淀进 `src/bagent/sites.py`，使用者就只需要选站 + 打字，
不必记任何网址。共 **10 个分类 / 60+ 站点**（综合搜索、电商、视频、
社交、知识技术、求职、生活服务、财经、新闻）。

两处刻意的设计：

- **拼网址在服务端做**（`POST /resolve`），前端不存模板副本。
  两处各存一份的话，站点改了 query 参数名时前端那份必然忘记同步，
  表现为"选淘宝却跳去了京东"这类很难查的错位。
- **`needs_login` 如实回传**，控制台把这类站标成"需登录 / 有验证"。
  藏掉它等于让用户点下去才发现被拦 —— 那是把项目的短板藏在用户脚下。

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
│   ├── graph_agent.py       # LangGraph 版引擎（与主循环共用感知/动作层）
│   ├── sites.py             # 站点注册表：主流站点 + 搜索 URL 模板
│   ├── siteprobe.py         # 实测站点模板可达性（可复跑，结果落快照）
│   └── cli.py               # 命令行
│
├── tasks/examples.json      # 评测任务集（含反例任务，简历里的判分基线）
├── tasks/demo_sites.json    # 高频站点演示集（独立于判分基线）
├── runs/                    # 每次运行的产物：trace.json + 截图
│   └── site_probe.json      # --probe-sites 的产物：站点可达性快照
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
| `STEP_TIMEOUT_SECONDS` | 30 | 单个浏览器动作的超时 |
| `HEADLESS` | `true`（代码默认）/ `false`（`.env.example` 里显式写成 false） | true = 不显示浏览器窗口。脚本、服务、CI 都该无头；本地想看浏览器跑就填 false |
| `TEARDOWN_TIMEOUT_SECONDS` | 8 | 浏览器收尾的总时间预算。超了放弃等待（任务状态不受影响），见「控制台暴露出来的四个真实缺陷」第 4 条 |
| `BROWSER_PROXY` | 空 | 浏览器出口代理，如 `http://127.0.0.1:7890`。**留空 = 不传 proxy 参数**，跟随浏览器自身默认 |
| `BROWSER_PROXY_BYPASS` | 空 | 不走代理的域名，逗号分隔，如 `baidu.com,taobao.com` |

> 为什么代理要单独配一份，而不是跟随系统代理：系统代理是全局的，
> 但浏览器该不该走代理取决于目标站点，一刀切两边都会坏。
> 本机实测的真实约束 —— 国内站点经境外节点时淘宝只回 139 字空壳、
> 百度直接 `ERR_CONNECTION_CLOSED`；而 GitHub / Google 不走代理又连不上。
> 想让两边都能跑，就填 `BROWSER_PROXY` + 把国内域名列进 `BROWSER_PROXY_BYPASS`。

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
uvicorn --app-dir src bagent.api:app --reload --port 8000
```

**`--app-dir src` 不能省**：`src/` 这种布局下，直接在仓库根目录跑
`uvicorn bagent.api:app` 会 `ModuleNotFoundError: No module named 'bagent'`。
容器里没暴露这个问题，是因为 Dockerfile 设了 `PYTHONPATH=/app/src`。

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

非 root 运行：浏览器里跑的是「模型生成的页面操作」，权限必须收窄。

镜像默认命令是 `python main.py --help`；`ENTRYPOINT` 只固定解释器，
所以 `docker run 镜像 -m uvicorn ...` 会被正确拼成 `python -m uvicorn ...`。

> 本仓库**没有**附带 `docker-compose.yml`。上面那条 `docker run` 就是完整的启动方式；
> 想和 pr-review-agent 一起编排的话，那个文件需要你自己按需写。

### 构建踩过的两个坑

1. **`.dockerignore` 排除了 `docs/`，而 Dockerfile 里又 `COPY docs/ ./docs/`** —— 构建会直接
   报 `not found`。`docs/` 放的是实验记录和教程，属于要进镜像的内容，不能忽略。
   排查方式：把 Dockerfile 里每条 `COPY` 的源路径与 `.dockerignore` 的规则对一遍，
   凡是「既存在又被忽略」的就是雷。
2. **`doctor()` 把「视觉通道未配置」判为失败。** 视觉是 DOM 通道失效时的**可选**降级路径，
   没配 key 就自动退回纯 DOM，这是设计行为而不是故障。原来它返回 `settings.vlm_enabled`，
   导致容器里（`MOCK/OFFLINE`）自检退出码恒为 1。现已改为：配了才算一项检查，
   没配则如实说明「按设计只走 DOM 通道」；同时在 `MOCK/OFFLINE` 下跳过模型连通性检查，
   不在离线环境里发起真实网络请求。

---

## Web 控制台

服务启动后直接打开 <http://127.0.0.1:8000/> 就是一个自带的可视控制台
（深色主题，零前端依赖、无构建步骤）：

- 提交任务、选引擎 / 最大步数 / 离线开关，页面上直接看执行时间线；
- 每一步显示动作名、说明与所在网址，失败步用红色标出来；
- 下方「最近任务」表格列出最近 20 条，含步数与耗时。

实现方式刻意做到最简：`src/bagent/web/index.html` 是**单文件**页面，
用 `FileResponse` 直接从包内路径吐出去 ——

```python
_WEB_INDEX = Path(__file__).resolve().parent / "web" / "index.html"

@app.get("/", include_in_schema=False)
def console() -> FileResponse:
    ...
    return FileResponse(_WEB_INDEX, media_type="text/html; charset=utf-8")
```

几个取舍值得说明：

- **用 `Path(__file__)` 而不是相对 cwd 的路径**：静态页放在包内部，
  `pip install` 后依然找得到，也不受启动目录影响。
- **`include_in_schema=False`**：`/` 是给人看的页面，不是 API，
  放进 OpenAPI 只会让文档里多一条没有意义的接口。
- **页面自证可用**：`tests/test_api.py` 里断言 `/` 返回 `text/html` 且包含
  `id="task"` / `id="go"` 等锚点。一旦 Dockerfile 或打包清单漏掉 `web/` 目录，
  测试会立刻红，而不是等线上打开发现 404。

### 控制台暴露出来的四个真实缺陷

把后端接到一个真实界面上之后，原先"接口能返回 200 就算对"的地方全部露了馅。
这四处都是**后端逻辑问题**（不是渲染问题），控制台只是让它们变得可见：

1. **离线任务的成本是凭空算出来的。** 离线替身的 token 是按字符估算的，
   而 `cost_yuan` 照常乘以单价，于是离线跑一次会显示 `¥0.00347` ——
   使用者会以为这次真的花了钱。更根本的问题是同一件事有两个开关：
   选 LLM 客户端读 `req.mock`，判定是否计费读 `settings.mock`，而
   **`Settings` 上当时根本没有 `mock` 这个字段**，`st.mock = True` 是一句无声的空操作。
   现已在 `Settings` 上真正定义 `mock`，并由 `_run_task` 按请求覆写，
   `RunResult.offline` 把口径一路带到界面（显示 `¥0 / 成本（离线，未产生费用）`）。
2. **`records` 里的步号会跳号**（实测 `[1, 3, 4, 5]`）。模型输出不是合法 JSON 时，
   那一轮**确实消耗了一次模型调用**，但代码直接 `continue`，没有留下 `StepRecord`。
   审计轨迹看起来像"漏了第 2 步"，其实那一步的失败被整个吞掉了 ——
   排障时最不该丢的恰恰是失败那一步。
   修的时候踩到一个坑：不能给 `action` 塞一个编造的 `"(格式错误)"`，
   因为 `ActionName` 是封闭字面量，pydantic 会在构造时直接 `ValidationError`
   （等于"为了记录一个格式错误，先让程序崩掉"）。
   正确做法是 `StepRecord.action` 放宽为 `Action | None`，另加 `raw_action` 承载原始字符串。
3. **列表接口不返回 `steps` / `elapsed_seconds`**，导致前端表格这两列只能显示 `-`，
   除非客户端对每一行再打一次详情接口（20 行 = 21 个请求）。这两项在列表数据里本来就是现成的。
4. **任务"跑完了"却永远显示在执行中。** 这条最要命，因为它把成功伪装成卡死。
   终态原本写在 `async with open_browser(...)` **之后**，而浏览器收尾
   （`context.close()` / `browser.close()` / `playwright.stop()`）在某些环境下
   会长时间不返回 —— 收尾卡住 = 终态永远不写 = 控制台一直转圈。
   实测：一个离线任务 **3.3 秒**就走完 5 步（每步的时间线都是实时可见的），
   但 **150 秒**都没进终态；CLI 更直白，报告早就打完了，进程却卡在收尾上
   4 分钟不退出。这里的关键认识是：**收尾不属于任务本身**，它只负责回收进程资源，
   不该决定"任务完成了没有"。修法是两层：

   - `BrowserSession.__aexit__` 给整个收尾一个总时间预算
     （`TEARDOWN_TIMEOUT_SECONDS`，默认 8 秒），超时记一条警告就放弃等待；
   - `_run_task` 不再用 `async with`，改成手动进入 + 在 `finally` 里收尾，
     这样终态能在收尾**之前**落表。

   注意这两层不是重复：第一层让 CLI 能自己退出，第二层保证即使别人换了个
   没有超时护栏的实现，服务端的终态也不会被拖住。回归测试
   `test_task_reaches_terminal_state_even_if_teardown_hangs` 用一个
   "收尾永不返回"的替身把这一点钉住了。

   顺带一个诚实的边界：超时放弃之后，那个 Playwright 驱动子进程会成为孤儿，
   Windows 上的 `asyncio` 在解释器退出时还会等它约 30 秒，并打印几条
   `unclosed transport` 的 `ResourceWarning`。也就是说 CLI 的**总**耗时可能到 40 秒
   量级 —— 但报告在 12 秒内就打完、任务状态也早已落表，剩下的只是进程回收。
   在 CI（Linux）上收尾约 1 秒完成，完全不会走到这条路。

前三条都补了回归测试，断言的是**语义**而不是返回值：
`test_offline_task_reports_zero_cost` 要求离线任务 `cost_yuan == 0`，
`test_step_record_accepts_error_step` 要求"格式错误"这一步能落盘且能序列化，
`test_task_reaches_terminal_state_even_if_teardown_hangs` 要求"收尾卡住也要进终态"。


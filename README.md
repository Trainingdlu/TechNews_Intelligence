<div align="center">
  <img src="https://raw.githubusercontent.com/Trainingdlu/TechNews_Intelligence/main/assets/svg/title.svg" alt="TechNews Intelligence" width="700">
  <img src="https://img.shields.io/badge/stack-n8n_|_PostgreSQL_|_Metabase_|_Agent-blue?style=flat-square" alt="技术栈">
  <img src="https://img.shields.io/badge/license-AGPL_3.0-red?style=flat-square" alt="许可证">
  <p align="center">
    <a href="https://dashboard.trainingcqy.com" style="text-decoration:none"><strong>Metabase 演示</strong></a>
    &nbsp;&nbsp; | &nbsp;&nbsp;
    <a href="https://agent.trainingcqy.com" style="text-decoration:none"><strong>智能体交互</strong></a>
    &nbsp;&nbsp; | &nbsp;&nbsp;
    <a href="https://agent.trainingcqy.com/subscribe.html" style="text-decoration:none"><strong>日报订阅</strong></a>
  </p>
</div>

利用工作流定时采集 Hacker News 与 TechCrunch 的科技新闻，由 Jina Reader 获取新闻原文，以辅助 LLM 进行摘要生成、情感分析以及分类的结构化处理。并使用 Jina Embeddings 生成语义向量以支持相似度搜索，统一存入数据库。最终通过 Metabase 仪表盘、邮件日报、网页端和 Telegram 机器人进行展示与交互。<br>

![效果展示](https://raw.githubusercontent.com/Trainingdlu/TechNews_Intelligence/main/assets/previews/showcase.png)

---

# 1. 系统架构
项目基于 ELT 架构，使用 Docker Compose 编排，包含五个核心容器服务：n8n（工作流与向量化）、PostgreSQL（存储与向量检索）、Metabase（可视化）、智能体 API（前端/程序调用入口）、Telegram 机器人（应用交互入口），同时提供本地命令行入口。网页端为 `frontend/` 下的静态资源，不作为独立容器默认启动。

![系统架构图](https://raw.githubusercontent.com/Trainingdlu/TechNews_Intelligence/main/assets/svg/architecture.svg)

---

# 2. 功能说明

## 2.1 数据采集与处理
系统通过 n8n 编排三条工作流实现全自动化采集与结构化处理

**主工作流**：每小时自动触发，获取新闻数据后通过 Jina Reader 提取全文，再调用 LLM 输出结构化 JSON（标题翻译、摘要、情感、标签化分类）。写入成功后自动调用 Jina Embeddings 生成 1024 维语义向量并存入 `news_embeddings` ，确保语义搜索始终覆盖最新数据分析。已有数据仅更新热度值。处理失败的新闻数据存入`tech_news_failed` 表防止重复尝试。<br>

![主流程](https://raw.githubusercontent.com/Trainingdlu/TechNews_Intelligence/main/assets/screenshots/Main_workflow.png)

**异常捕获与告警**：全局错误处理，自动捕获异常并发送告警邮件。

![告警流程](https://raw.githubusercontent.com/Trainingdlu/TechNews_Intelligence/main/assets/screenshots/Alert_workflow.png)

**日报推送**：每日 08:00 筛选近 24 小时价值新闻，并渲染为 HTML 邮件推送。

![日报流程](https://raw.githubusercontent.com/Trainingdlu/TechNews_Intelligence/main/assets/screenshots/Brief_workflow.png)

### 工作流处理流程

#### 主工作流核心处理流程
**输入内容**：n8n 先调用 Jina Reader API 将原始 HTML 页面去噪（剔除导航、广告、脚注等无关元素），转为较为干净的 Markdown 格式全文（包含标题 + 正文），作为 LLM 的提示词输入，能够显著提升摘要质量。<br>
**输出 JSON 的完整字段**：
  `title_cn`：中文标题（可包含分类前缀标签）
  `summary`：摘要
  `sentiment`：情绪标签<br>
**情感分类**：三个情绪分类 —— Positive（正面）、Neutral（中性）、Negative（负面）。<br>
**新闻分类**：六个标签体系 —— AI、安全、硬件、开发、商业、生态（用于标题前缀与检索过滤，非独立数据库列）。<br>
**失败记录存储**：失败记录进入 `tech_news_failed` 死信队列，避免重复处理。<br>
**向量写入机制**：主工作流采用同步执行。当新闻成功写入 `tech_news` 表后，立即调用 Jina Embeddings API 生成 1024 维语义向量，随后存入 `news_embeddings` 表，确保智能体检索始终覆盖最新数据。

#### 异常捕获与告警核心流程
**触发机制**：采用双重触发模式。一是通过 n8n 全局 `Error Trigger` 监听工作流运行时的突发异常；二是在主工作流中通过 `Data_check` 等节点对输出质量进行实时校验，若发现空数据或解析异常，则主动调用告警流。<br>
**错误处理**：自动提取执行失败的节点名称、错误详情及关联的 URL，确保故障定位的准确性。<br>
**日志记录**：将故障数据（时间、类型、错误信息）写入 `system_logs` 表，为系统稳定性分析提供静态数据支撑。<br>
**邮件通知**：渲染 HTML 邮件并发送至管理员邮箱。

#### 日报推送核心流程
**任务触发**：每日 08:00 定时启动，通过 PostgreSQL 视图 `view_dashboard_news` 筛选近 24 小时内的增量数据。<br>
**新闻选择**：在数据库层面执行聚合查询，选取 Hacker News 社区热度前 7（Points 倒序）以及 TechCrunch 媒体最近前 7 的文章。<br>
**动态渲染**：利用 n8n Code 节点执行 JavaScript 逻辑。根据新闻的情感标签匹配特定的视觉样式，生成定制化的 HTML 邮件内容。<br>
**分发日报**：从 `subscribers` 表获取订阅者列表，通过 SMTP 协议实现批量推送。

## 2.2 数据库设计
系统基于 PostgreSQL 实现数据持久化与语义存储

| 表名                 | 用途                                    |
| ------------------ | ------------------------------------- |
| `tech_news`        | 主数据表，`url` 唯一约束防止重复                   |
| `news_embeddings`  | 语义向量表，存储 Jina Embeddings，供智能体混合搜索 |
| `tech_news_failed` | 死信队列，记录处理失败的条目                        |
| `jina_raw_logs`    | Jina Reader 返回的新闻原始内容                 |
| `subscribers`      | 日报接收用户邮箱列表                            |
| `access_tokens`    | 网页前端令牌管理表，存储用户邮箱、令牌、配额与使用量  |
| `source_registry`  | 数据源注册表，控制订阅可选的新闻源                     |
| `system_logs`      | 系统运行日志                                |
| `conversation_threads` | 对话会话管理，按 channel 隔离，持久化线程元数据 |
| `conversation_messages` | 对话消息存储，关联 thread_id，保存完整 payload |
| `agent_runs`       | 智能体请求级 Trace，记录延迟、Token 用量、工具调用链、异常链 |
| `agent_tool_events` | 工具级事件明细，关联 request_id，记录每次工具调用的输入/输出摘要 |

视图 `view_dashboard_news` 封装了时区转换（UTC → UTC+8）、来源归一化（优先 `source_registry`，并回退 `tech_news` 字段与 URL 兜底规则）、hours_ago 计算和 HN 讨论链接生成，分析查询层不重复这些逻辑。

## 2.3 数据分析与可视化
系统通过 Metabase 进行对新闻数据的简单分析与展示

| 名称 | 说明 |
| :--- | :--- |
| **社区热议** | 按热度排名展示 HackerNews 的热门话题。 |
| **实时快讯** | 根据发布时间就近展示 TechCrunch 的新闻。 |
| **重力排名** | 利用 HN 热度排名公式，展示热度上升最快的讨论。 |
| **摘要卡片** | 使用 Metabase 参数传递，实现点击标题查看详细摘要。 |
| **公司热度** | 统计 Microsoft、OpenAI 等 10 家头部公司的热度。 |
| **媒体日更** | 统计 TechCrunch 每日发文量。 |
| **情绪差异** | 体现技术社区 (HN) 与媒体机构 (TC) 的情绪倾向差异。 |
| **赛道周环比** | 统计各分类本周与上周的热度增量差。 |
| **热度热力图** | 展示各个发帖时间的热度差异。 |
| **负面率指数** | 各个分类的新闻负面率。 |
| **话题分类统计** | 统计不同分类的发布数量与热度。 |
| **话题热度分布** | 散点化展示近期社区高热度讨论。 |
| **情绪分时趋势** | 展示最近三天情绪的变化。 |
| **情绪传播广度** | 展示不同情绪对新闻传播热度的影响。 |

>截至2026年3月，系统已稳定运行约 3 个月，累计收录约 6000 条新闻数据，基于该数据的分析如下：<br>
>**情绪与热度**：负面新闻平均热度最高（208.5），显著高于正面（161.1）和中性（128.2）。尽管负面新闻数量最少（1,112 条），但负面新闻传播影响力最强。<br>
>**平台情绪差异**：TC 正面主导，HN 中性主导。HackerNews 负面率（17.3%）低于 TechCrunch（23.2%），差异不在负面，而在中性。HN 以中性为主（45.3%），TC 以正面为主（54.7%）。说明媒体倾向用正面框架报道，社区讨论更趋中立。<br>
>**分类负面率**：安全类负面率高达 63%（分类中最高）；AI 和开发类负面率最低（3.8% / 5%），但平均热度反而最高（100 / 123），呈现“高热度 ≠ 高风险”的明显特征。<br>
>**发布热力图**：单个最高热度时段为周一 14:00（316），次高峰出现在周四 13:00（295）；此外，周六 08:00（258）与周日 09:00（269）等周末早间时段，以及大部分的深夜（22:00-23:00）均出现了显著的活跃高峰，体现了HN社区的热度分布呈现高度碎片化。

## 2.4 深度分析智能体
基于 LangGraph ReAct 架构构建的交互式分析智能体，支持 Vertex AI 与 Gemini API。提供网页前端、Telegram 机器人、本地命令行三种接入方式。

### 运行时架构

系统采用统一的 ReAct 智能体单循环架构，LLM 拥有完全的工具调用自主权。工具执行通过 SkillRegistry（技能注册中心）统一分发，并由 ToolHookRunner 在执行前后插入参数校验与证据审计守卫。每次请求自动绑定 `AgentRequestTrace`，记录工具调用链、延迟与 Token 用量，并在请求结束后持久化至 `agent_runs` / `agent_tool_events` 表。当证据不足或来源冲突时，Clarification HITL 机制会中断生成并返回结构化澄清问题，由用户补充范围后再继续。

```text
请求入口 (app/api.py / app/bot.py / app/cli.py)
  ↓
Clarification 历史解析 ── 检测是否为澄清追问，合并原问题
  ↓
request_trace_context() ── 绑定 AgentRequestTrace
  ↓
agent.py → LangGraph create_react_agent
  ↓  ↑ (tool calls)
trace_tool_start()             ── 记录工具开始
ToolHookRunner.pre_tool_use()  ── 参数校验
  ↓
SkillRegistry.execute()        ── agent/skills/... (DB / Jina Rerank / 知识库能力)
  ↓
ToolHookRunner.post_tool_use() ── 证据审计
trace_tool_finish()            ── 记录工具结束与输出摘要
  ↓
SkillEnvelope → Evidence 累积
  ↓
LLM 生成最终回复 → Token 用量采集
  ↓
Clarification 检测 ── 证据不足 / 范围模糊 / 来源冲突 → 结构化澄清
  ↓
后处理管道 (evidence.py)        ── 引用归一化 + 来源列表
  ↓
finalize_request_trace()       ── Trace 持久化至 DB
```

### 工具体系

智能体拥有 11 个工具，通过 SkillRegistry 分发，均有 Pydantic 输入验证和 SkillEnvelope 结构化输出：

| 工具 | 用途 |
| :--- | :--- |
| `search_news` | 混合检索（pgvector 语义相似度 + 关键词精确匹配），合并去重，支持 Rerank 二次排序 |
| `query_news` | 结构化过滤查询（来源、分类、情感、时间窗口、排序） |
| `trend_analysis` | 话题动量分析：近 N 天 vs 前 N 天的数据量与热度对比 |
| `compare_sources` | HackerNews vs TechCrunch 双源覆盖与情感差异对比 |
| `compare_topics` | A vs B 实体对比（如 OpenAI vs Anthropic），含指标与证据 |
| `build_timeline` | 时间线构建：按时间排列的事件编年，自动扩展窗口 |
| `analyze_landscape` | 竞争格局分析：实体维度统计 + 信号分类（Compute / Algorithm / Data / GTM / Policy） |
| `fulltext_batch` | 批量全文阅读，支持 URL 列表或关键词自动选取，支持 Rerank 优先选取 |
| `read_news_content` | 单篇新闻原文读取（从 `jina_raw_logs` 提取） |
| `get_db_stats` | 数据库新鲜度与文章总量 |
| `list_topics` | 近 21 天每日发文量分布 |

**混合检索机制**：纯向量检索在公司名、产品名等专有名词上容易召回偏移，关键词匹配作为兜底保证精确查询的稳定性。检索评分融合了语义相似度、关键词命中、热度归一化和时间衰减因子 `0.1 × EXP(-age_seconds / 86400 / 21)`。

**Rerank 二次排序**：召回结果可通过 Jina Reranker 进行二次精排（Cross-Encoder 或 LLM Rerank 模式），由环境变量 `NEWS_RERANK_MODE` 控制（`none` / `cross_encoder` / `llm_rerank`），失败时自动 fallback 至原始召回顺序。

### 基础设施层

| 组件 | 文件 | 职责 |
| :--- | :--- | :--- |
| SkillRegistry | `core/skill_registry.py` | 技能注册与执行：Pydantic 输入验证 → handler 调用 → SkillEnvelope 输出标准化 |
| SkillEnvelope | `core/skill_contracts.py` | 统一输出信封：`status`(ok/empty/error) + `data` + `evidence[]` + `diagnostics` |
| ToolHookRunner | `core/tool_hooks.py` | Pre-hook 参数守卫（时间窗口范围、topic 非空、去重校验）；Post-hook 证据完整性审计 |
| Evidence Pipeline | `core/evidence.py` | URL 提取 → 内联引用编号 → 来源列表渲染 → 无效引用清洗 |
| Agent Trace | `core/trace.py` | 请求级追踪：工具调用事件记录、Token 用量采集、异常链捕获、摘要持久化 |
| Trace Store | `services/agent_trace_store.py` | Trace 持久化适配器：将 `AgentRequestTrace` 摘要写入 `agent_runs` + `agent_tool_events` 表 |
| Rerank | `skills/rerank.py` | 二次精排：Jina Cross-Encoder / LLM Rerank，支持环境变量配置与 fallback |
| Clarification HITL | `clarification.py` | 证据不足 / 范围模糊 / 来源冲突检测 → 结构化澄清问题生成 → 用户追问自动合并 |
| Thread Persistence | `services/conversations.py` | 对话线程持久化：创建线程、追加消息、加载历史、按 channel 列举 |
| Metrics | `core/metrics.py` | 运行时指标：请求量、成功率、错误率、递归超限率、Trace 指标，自动日志输出 |
| 角色策略 | `core/role_policy.py` | 角色-技能 ACL 策略（预留多智能体扩展点） |
| MCP 协议层 | `mcp/` | In-Process + Stdio 双传输后端的 MCP 抽象，作为扩展层保留，非默认请求路径 |

### 系统提示词与分析框架

智能体根据用户意图自动选择输出模式，避免同质化回答：

| 模式 | 适用场景 | 输出模板 |
| :--- | :--- | :--- |
| 快速简报 | "最近发生了什么" | 今日概览 → 判断 |
| 对比视图 | A vs B、来源差异 | 对比结论 → 关键变量 → 变迁判断 → 决策影响 |
| 时间线视图 | 事件演变、里程碑 | 事件时间线 → 转折点 → 后续关注 |
| 深度解读 | 特定事件深度解读 | 核心事件 → 关键变量 → 深度解读 → 关键洞察 |
| 格局视图 | 全局格局、竞争结构 | 结论 → 关键变量与转折点 → 公司角色 → 供需-生态位分析 |

分析框架（对比视图 / 深度解读 / 格局视图模式强制应用）：<br>
**信号与噪声**：优先识别改变竞争均衡的事件。<br>
**变量分解**：将证据映射至算力/成本、算法/效率、数据/壁垒三维度。<br>
**变迁判定**：区分"工程优化"（同一范式的改良）与"范式转移"（新规则的确立）。<br>
**供需-生态评估**：评估供给壁垒、需求强度、生态位层级。<br>
**前瞻推演**：给出 6-18 个月的条件性推演（非确定性预测），事实/推断/情景假设显式分层。

### 接入方式

| 入口 | 实现 | 部署 | 对话历史 |
| :--- | :--- | :--- | :--- |
| 网页前端 | FastAPI (`app/api.py`) + Bearer 令牌 | Docker 容器 `tech_news_api` | 客户端携带 `history` 参数，支持 DB 持久化线程 |
| Telegram 机器人 | python-telegram-bot (`app/bot.py`) | Docker 容器 `tech_news_bot` | 进程内字典，按 `chat_id` 隔离 |
| 本地命令行 | `app/cli.py` | 本地 Python | `_SessionChat` 对象持有 |

**令牌配额机制**：网页前端通过邮箱自动获取令牌（默认 10 次），配额耗尽后自动触发管理员审批邮件，管理员一键批准可提升至 50 次。Clarification 流程触发时自动退还配额，不消耗用户次数。<br>
**限流**：网页 API 按 IP 限流（默认 5 次/分钟）；Telegram 机器人按 chat_id 限流（默认 3 次/10 秒）。

### 质量保障

- **单元测试**（`tests/unit/`）：覆盖 agent 路由指标、Trace 生命周期、Trace 持久化、Clarification 原因推断与场景覆盖、来源冲突检测、Rerank 流程与 fallback、对话线程服务、API 澄清集成、Bot 鲁棒性、MCP 适配器、工具结构化输出、runtime factories、eval core 等模块
- **评测框架**（`eval/`）：JSONL 格式测试数据集，支持按类别/能力筛选；多次运行计算输出稳定性；质量门禁阈值检测与基线回归对比
- **Rerank 评测**（`eval/rerank_eval.py`）：离线 NDCG / MRR 基准对比，对比召回原始顺序与 Rerank 后排序的检索质量增量

---

## 3. 目录结构
```text
TechNews_Intelligence/
├── .github/
│   └── workflows/
│       ├── ci.yml
│       ├── deploy.yml
│       └── eval-manual.yml
│
├── agent/                           # 智能体核心逻辑
│   ├── agent.py                     # ReAct 运行时主入口
│   ├── clarification.py             # Clarification HITL：范围/冲突检测 + 结构化澄清
│   ├── prompts.py                   # 系统提示词与约束
│   ├── skills/                      # 具体技能实现代码（含 rerank.py 二次精排）
│   ├── core/                        # skill registry / hooks / evidence / metrics / trace
│   ├── mcp/                         # MCP 扩展层
│   ├── .env.example

├── app/                             # 应用入口层
│   ├── api.py                       # FastAPI 网页 API
│   ├── bot.py                       # Telegram 机器人
│   └── cli.py                       # 本地命令行
│
├── services/                        # 基础服务
│   ├── db.py                        # PostgreSQL 连接与查询
│   ├── mail.py                      # 邮件通知
│   ├── agent_trace_store.py         # Agent Trace 持久化适配器
│   └── conversations.py             # 对话线程持久化服务
│
├── eval/                            # 离线评测框架
│   ├── run_eval.py
│   ├── eval_core.py
│   ├── dataset_loader.py
│   ├── capabilities.py
│   ├── datasets/
│   └── reports/
│
├── tests/                           # 自动化测试
│   ├── unit/
│   ├── utils/
│   └── reports/
│
├── deployment/                      # Docker 与运维脚本
│   ├── docker-compose.yml
│   ├── .env.example
│   ├── data/
│   └── scripts/db/
│
├── docs/
│   ├── testing/
│   ├── data_quality_checks.md
│   ├── first_batch_official_sources.md
│   └── project_structure.md
│
├── etl_workflow/                    # n8n 工作流配置
│   ├── Tech_Intelligence.json
│   ├── System_Alert_Service.json
│   └── Daily_Tech_Brief.json
│
├── frontend/                        # 静态前端页面
│   ├── index.html
│   ├── subscribe.html
│   ├── app.js / style.css
│   └── subscribe.js / subscribe.css
│
├── sql/
│   ├── analytics/
│   └── infrastructure/
│       ├── checks/data_quality_checks.sql
│       ├── schema/schema_ddl.sql
│       ├── seeds/seed_source_official.sql
│       └── views/view_dashboard_news.sql
│
├── assets/
│   ├── previews/
│   ├── screenshots/
│   └── svg/
│
├── .gitattributes
├── .gitignore
├── Dockerfile
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 4. 部署
项目已容器化，可通过 Docker Compose 部署。智能体在 Docker 中拆分为 `tech_news_bot`（Telegram 机器人）和 `tech_news_api`（网页 API）两个独立容器。

### 前置条件
已获取 LLM API 密钥或 Vertex AI 凭证，及 Jina API 密钥<br>
已创建 Telegram 机器人并获取 Bot Token（通过 @BotFather）<br>
### 步骤
```bash
# 1. 克隆仓库
git clone https://github.com/Trainingdlu/TechNews_Intelligence.git
cd TechNews_Intelligence

# 2. 配置环境变量
cp deployment/.env.example deployment/.env
# 编辑 deployment/.env，填入环境变量
# 关键变量：
# AGENT_MODEL_PROVIDER           — 模型后端
# GEMINI_API_KEY                 — Gemini API 密钥
# VERTEX_PROJECT                 — GCP 项目 ID
# GOOGLE_APPLICATION_CREDENTIALS — Vertex AI 服务账号凭证路径
# JINA_API_KEY                   — Jina Embeddings API 密钥

# 3. 启动服务
cd deployment
docker-compose up -d
```

### 导入工作流
1. 访问 http://localhost:5678 进入 n8n 管理界面。
2. 导入 etl_workflow/ 目录下的三个 JSON 文件。
3. 在 n8n 中配置 PostgreSQL 连接凭证和 SMTP 凭证（用于邮件功能）。
4. 激活工作流。

### Telegram 机器人
机器人服务已包含在 docker-compose.yml 中，执行 `docker-compose up -d` 后会随其他服务一同启动，无需额外操作。

### 本地命令行
```bash
# 在仓库根目录执行
cp agent/.env.example agent/.env
# 编辑 agent/.env，填入环境变量
pip install -r requirements.txt
python -m app.cli
```
---

## 5. 开源协议
本项目采用 GNU AGPLv3 协议。




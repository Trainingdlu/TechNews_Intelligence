<div align="center">
  <img src="https://raw.githubusercontent.com/Trainingdlu/TechNews_Intelligence/main/assets/svg/title.svg" alt="TechNews Intelligence" width="700">
  <img src="https://img.shields.io/badge/stack-n8n_|_PostgreSQL_|_Metabase_|_Agent-blue?style=flat-square" alt="Stack">
  <img src="https://img.shields.io/badge/license-AGPL_3.0-red?style=flat-square" alt="License">
  <p align="center">
    <a href="https://dashboard.trainingcqy.com" style="text-decoration:none"><strong>Metabase 演示</strong></a>
    &nbsp;&nbsp; | &nbsp;&nbsp;
    <a href="https://agent.trainingcqy.com" style="text-decoration:none"><strong>Agent 交互</strong></a>
    &nbsp;&nbsp; | &nbsp;&nbsp;
    <a href="https://agent.trainingcqy.com/subscribe.html" style="text-decoration:none"><strong>日报订阅</strong></a>
  </p>
</div>

利用工作流定时采集 Hacker News 与 TechCrunch 的科技新闻，由 Jina Reader 获取新闻原文，以辅助 LLM 进行摘要生成、情感分析以及分类的结构化处理。并使用 Jina Embeddings 生成语义向量以支持相似度搜索，统一存入数据库。最终通过 Metabase 仪表盘、邮件日报、Web 端和 Telegram Bot 进行展示与交互。<br>

![Showcase](https://raw.githubusercontent.com/Trainingdlu/TechNews_Intelligence/main/assets/previews/showcase.png)

---

# 1. 系统架构
项目基于 ELT 架构，使用 Docker Compose 编排，包含五个核心容器服务：n8n（工作流与向量化）、PostgreSQL（存储与向量检索）、Metabase（可视化）、Agent API（前端/程序调用入口）、Telegram Bot（应用交互入口），同时提供本地 CLI 入口。Web 网页为 `frontend/` 下的静态资源，不作为独立容器默认启动。

![Architecture](https://raw.githubusercontent.com/Trainingdlu/TechNews_Intelligence/main/assets/svg/architecture.svg)

---

# 2. 功能说明

## 2.1 数据采集与处理
系统通过 n8n 编排三条工作流实现全自动化采集与结构化处理

**主工作流**：每小时自动触发，获取新闻数据后通过 Jina Reader 提取全文，再调用 LLM 输出结构化 JSON（标题翻译、摘要、情感、标签化分类）。写入成功后自动调用 Jina Embeddings 生成 1024 维语义向量并存入 `news_embeddings` ，确保语义搜索始终覆盖最新数据分析。已有数据仅更新热度值。处理失败的新闻数据存入`tech_news_failed` 表防止重复尝试。<br>

![Main](https://raw.githubusercontent.com/Trainingdlu/TechNews_Intelligence/main/assets/screenshots/Main_workflow.png)

**异常捕获与告警**：全局错误处理，自动捕获异常并发送告警邮件。

![Error](https://raw.githubusercontent.com/Trainingdlu/TechNews_Intelligence/main/assets/screenshots/Alert_workflow.png)

**日报推送**：每日 08:00 筛选近 24 小时价值新闻，并渲染为 HTML 邮件推送。

![Brief](https://raw.githubusercontent.com/Trainingdlu/TechNews_Intelligence/main/assets/screenshots/Brief_workflow.png)

### 工作流处理流程

#### 主工作流核心处理流程
**输入内容**：n8n 先调用 Jina Reader API 将原始 HTML 页面去噪（剔除导航、广告、脚注等无关元素），转为较为干净的 Markdown 格式全文（包含标题 + 正文），作为 LLM 的 Prompt 输入，能够显著提升摘要质量。<br>
**输出 JSON 的完整字段**：
  `title_cn`：中文标题（可包含分类前缀标签）
  `summary`：摘要
  `sentiment`：情绪标签<br>
**情感分类**：三个情绪分类 —— Positive（正面）、Neutral（中性）、Negative（负面）。<br>
**新闻分类**：六个标签体系 —— AI、安全、硬件、开发、商业、生态（用于标题前缀与检索过滤，非独立数据库列）。<br>
**失败记录存储**：失败记录进入 `tech_news_failed` 死信队列，避免重复处理。<br>
**向量写入机制**：主工作流采用同步执行。当新闻成功写入 `tech_news` 表后，立即调用 Jina Embeddings API 生成 1024 维语义向量，随后存入 `news_embeddings` 表，确保 Agent 检索始终覆盖最新数据。

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
| `news_embeddings`  | 语义向量表，存储 Jina Embeddings，供 Agent 混合搜索 |
| `tech_news_failed` | 死信队列，记录处理失败的条目                        |
| `jina_raw_logs`    | Jina Reader 返回的新闻原始内容                 |
| `subscribers`      | 日报接收用户邮箱列表                            |
| `access_tokens`    | Web 前端 Token 管理表，存储用户邮箱、Token、配额与使用量  |
| `source_registry`  | 数据源注册表，控制订阅可选的新闻源                     |
| `system_logs`      | 系统运行日志                                |

视图 `view_dashboard_news` 封装了时区转换（UTC → UTC+8）、来源归一化（优先 `source_registry`，并回退 `tech_news` 字段与 legacy 规则）、hours_ago 计算和 HN 讨论链接生成，分析查询层不重复这些逻辑。

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

## 2.4 深度分析 Agent
基于 LangGraph ReAct 架构构建的交互式分析 Agent，支持 Vertex AI ，与 Gemini API 。提供 Web 前端、Telegram Bot、本地 CLI 三种接入方式。

### 运行时架构

系统采用统一的 ReAct Agent 单循环架构，LLM 拥有完全的工具调用自主权。工具执行通过 SkillRegistry（技能注册中心）统一分发，并由 ToolHookRunner 在执行前后插入参数校验与证据审计守卫。

```text
请求入口 (api.py / bot.py / cli.py)
  ↓
agent.py → LangGraph create_react_agent
  ↓  ↑ (tool calls)
ToolHookRunner.pre_tool_use()  ── 参数校验
  ↓
SkillRegistry.execute()        ── tools.py (DB / Jina API)
  ↓
ToolHookRunner.post_tool_use() ── 证据审计
  ↓
SkillEnvelope → Evidence 累积
  ↓
LLM 生成最终回复
  ↓
后处理管道 (evidence.py)       ── 引用归一化 + 来源列表
```

### 工具体系

Agent 拥有 11 个工具，其中 8 个通过 SkillRegistry 分发，均有 Pydantic 输入验证和 SkillEnvelope 结构化输出：

| 工具 | 分发 | 用途 |
| :--- | :--- | :--- |
| `search_news` | Registry | 混合检索（pgvector 语义相似度 + 关键词精确匹配），合并去重 |
| `query_news` | Registry | 结构化过滤查询（来源、分类、情感、时间窗口、排序） |
| `trend_analysis` | Registry | 话题动量分析：近 N 天 vs 前 N 天的数据量与热度对比 |
| `compare_sources` | Registry | HackerNews vs TechCrunch 双源覆盖与情感差异对比 |
| `compare_topics` | Registry | A vs B 实体对比（如 OpenAI vs Anthropic），含指标与证据 |
| `build_timeline` | Registry | 时间线构建：按时间排列的事件编年，自动扩展窗口 |
| `analyze_landscape` | Registry | 竞争格局分析：实体维度统计 + 信号分类（Compute / Algorithm / Data / GTM / Policy） |
| `fulltext_batch` | Registry | 批量全文阅读，支持 URL 列表或关键词自动选取 |
| `read_news_content` | 直接 | 单篇新闻原文读取（从 `jina_raw_logs` 提取） |
| `get_db_stats` | 直接 | 数据库新鲜度与文章总量 |
| `list_topics` | 直接 | 近 21 天每日发文量分布 |

**混合检索机制**：纯向量检索在公司名、产品名等专有名词上容易召回偏移，关键词匹配作为兜底保证精确查询的稳定性。检索评分融合了语义相似度、关键词命中、热度归一化和时间衰减因子 `0.1 × EXP(-age_seconds / 86400 / 21)`。

### 基础设施层

| 组件 | 文件 | 职责 |
| :--- | :--- | :--- |
| SkillRegistry | `core/skill_registry.py` | 技能注册与执行：Pydantic 输入验证 → handler 调用 → SkillEnvelope 输出标准化 |
| SkillEnvelope | `core/skill_contracts.py` | 统一输出信封：`status`(ok/empty/error) + `data` + `evidence[]` + `diagnostics` |
| ToolHookRunner | `core/tool_hooks.py` | Pre-hook 参数守卫（时间窗口范围、topic 非空、去重校验）；Post-hook 证据完整性审计 |
| Evidence Pipeline | `core/evidence.py` | URL 提取 → 内联引用编号 → 来源列表渲染 → 无效引用清洗 |
| Metrics | `core/metrics.py` | 运行时指标：请求量、成功率、错误率、递归超限率，自动日志输出 |
| Role Policy | `core/role_policy.py` | 角色-技能 ACL 策略（预留多 Agent 扩展点） |
| MCP 协议层 | `mcp/` | In-Process + Stdio 双传输后端的 MCP 抽象，作为扩展层保留，非默认请求路径 |

### System Prompt 与分析框架

Agent 根据用户意图自动选择输出模式，避免同质化回答：

| 模式 | 适用场景 | 输出模板 |
| :--- | :--- | :--- |
| Quick Brief | "最近发生了什么" | 今日概览 → 判断 |
| Compare View | A vs B、来源差异 | 对比结论 → 关键变量 → 变迁判断 → 决策影响 |
| Timeline View | 事件演变、里程碑 | 事件时间线 → 转折点 → 后续关注 |
| Deep Dive | 特定事件深度解读 | 核心事件 → 关键变量 → 深度解读 → 关键洞察 |
| Landscape View | 全局格局、竞争结构 | 结论 → 关键变量与转折点 → 公司角色 → 供需-生态位分析 |

分析框架（Compare / Deep Dive / Landscape 模式强制应用）：<br>
**Signal vs Noise**：优先识别改变竞争均衡的事件。<br>
**变量分解**：将证据映射至 Compute/Cost、Algorithm/Efficiency、Data/Moat 三维度。<br>
**变迁判定**：区分"工程优化"（同一范式的改良）与"范式转移"（新规则的确立）。<br>
**供需-生态评估**：评估供给壁垒、需求强度、生态位层级。<br>
**前瞻推演**：给出 6-18 个月的条件性推演（非确定性预测），事实/推断/情景假设显式分层。

### 接入方式

| 入口 | 实现 | 部署 | 对话历史 |
| :--- | :--- | :--- | :--- |
| Web 前端 | FastAPI (`api.py`) + Bearer Token | Docker 容器 `tech_news_api` | 客户端携带 `history` 参数，API 无状态 |
| Telegram Bot | python-telegram-bot (`bot.py`) | Docker 容器 `tech_news_bot` | 进程内字典，按 `chat_id` 隔离 |
| 本地 CLI | `cli.py` | 本地 Python | `_SessionChat` 对象持有 |

**Token 配额机制**：Web 前端通过邮箱自动获取 Token（默认 10 次），配额耗尽后自动触发管理员审批邮件，管理员一键批准可提升至 50 次。<br>
**限流**：Web API 按 IP 限流（默认 5 次/分钟）；Telegram Bot 按 chat_id 限流（默认 3 次/10 秒）。

### 质量保障

- **单元测试**（`tests/unit/`）：覆盖 agent 路由指标、MCP 适配器、工具结构化输出、runtime factories、eval core 等模块
- **Eval 框架**（`eval/`）：JSONL 格式测试数据集，支持按 category / capability 筛选；多次运行计算输出稳定性；quality gate 阈值检测与 baseline 回归对比

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
├── agents/                          # AI 深度分析 Agent
│   ├── agent.py                     # ReAct 运行时：LLM 客户端、工具调度、后处理管道
│   ├── api.py                       # Web API（FastAPI + Token 鉴权 + 限额管理）
│   ├── bot.py                       # Telegram Bot（chat_id 隔离 + Markdown→HTML 渲染）
│   ├── cli.py                       # 本地终端交互
│   ├── tools.py                     # 11 个工具函数（DB 查询、Jina 向量检索、全文读取）
│   ├── prompts.py                   # System Prompt（分析框架 + 5 种输出模式）
│   ├── db.py                        # PostgreSQL 连接池
│   ├── mail.py                      # 邮件（Token 发放、审批通知）
│   ├── core/                        # 运行时基础设施
│   ├── mcp/                         # MCP 协议层（扩展，非默认路径）
│   ├── eval/                        # 评估框架
│   ├── tests/
│   │   ├── unit/                    # 单元测试
│   │   └── utils/                   # 测试工具
│   └── requirements.txt             # Python 依赖
│
├── assets/
│   ├── previews/                    # 展示图
│   ├── screenshots/                 # 工作流 / UI 截图
│   └── svg/                         # 标题与架构图
│
├── deployment/                      # Docker 与运维脚本
│   ├── data/                        # 备份/样例 SQL
│   ├── scripts/
│   │   └── db/                      # 数据库维护脚本
│   ├── .env.example
│   └── docker-compose.yml
│
├── docs/
│   ├── testing/                     # 测试与评估文档
│   ├── data_quality_checks.md
│   ├── first_batch_official_sources.md
│   ├── minimal_source_refactor_checklist.md
│   └── project_structure.md
│
├── etl_workflow/                    # n8n 工作流配置
│   ├── Tech_Intelligence.json       # 主采集流程（含向量化节点）
│   ├── System_Alert_Service.json    # 异常捕获流程
│   └── Daily_Tech_Brief.json        # 日报推送流程
│
├── frontend/                        # Web 前端（静态页面）
│   ├── index.html                   # Agent 对话页面
│   ├── subscribe.html               # 日报订阅页面
│   ├── style.css / app.js           # 对话交互样式与逻辑
│   └── subscribe.css / subscribe.js # 订阅交互样式与逻辑
│
├── sql/
│   ├── analytics/                   # Metabase 分析查询 SQL（14 个）
│   └── infrastructure/
│       ├── checks/
│       │   └── data_quality_checks.sql
│       ├── schema/
│       │   └── schema_ddl.sql
│       ├── seeds/
│       │   └── seed_source_official.sql
│       └── views/
│           └── view_dashboard_news.sql
│
├── .gitattributes
├── .gitignore
├── Dockerfile
├── LICENSE
└── README.md
```

---

## 4. 部署
项目已容器化，可通过 Docker Compose 部署。Agent 在 Docker 中拆分为 `tech_news_bot`（Telegram Bot）和 `tech_news_api`（Web API）两个独立容器。

### 前置条件
已安装 Docker<br>
已获取 LLM API Key 或 Vertex AI 凭证，及 Jina API Key<br>
已创建 Telegram Bot 并获取 Bot Token（通过 @BotFather）<br>
### 步骤
```bash
# 1. 克隆仓库
git clone https://github.com/Trainingdlu/TechNews_Intelligence.git
cd TechNews_Intelligence

# 2. 配置环境变量
cp deployment/.env.example deployment/.env
# 编辑 deployment/.env，填入数据库密码、API Key、TELEGRAM_BOT_TOKEN 等
# 关键变量：
#   AGENT_MODEL_PROVIDER           — 模型后端
#   GEMINI_API_KEY                 — Gemini API Key
#   VERTEX_PROJECT                 — GCP 项目 ID
#   GOOGLE_APPLICATION_CREDENTIALS — Vertex AI 服务账号凭证路径
#   JINA_API_KEY                   — Jina Embeddings API Key

# 3. 启动服务
cd deployment
docker-compose up -d
```

### 导入工作流
1. 访问 http://localhost:5678 进入 n8n 管理界面。
2. 导入 etl_workflow/ 目录下的三个 JSON 文件。
3. 在 n8n 中配置 PostgreSQL 连接凭证和 SMTP 凭证（用于邮件功能）。
4. 激活工作流。

### Telegram Bot
Bot 服务已包含在 docker-compose.yml 中，执行 `docker-compose up -d` 后会随其他服务一同启动，无需额外操作。

### 本地 CLI
```bash
cd agents
cp .env.example .env
# 编辑 .env，填入 Gemini API Key、Jina API Key、数据库连接信息
pip install -r requirements.txt
python cli.py
```
---

## 5. 开源协议
本项目采用 GNU AGPLv3 协议。

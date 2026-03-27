<div align="center">
  <img src="https://raw.githubusercontent.com/Trainingdlu/TechNews_Intelligence/main/assets/svg/title.svg" alt="TechNews Intelligence" width="700">
  <img src="https://img.shields.io/badge/stack-n8n_|_PostgreSQL_|_Metabase_|_Agent-blue?style=flat-square" alt="Stack">
  <img src="https://img.shields.io/badge/license-AGPL_3.0-red?style=flat-square" alt="License">
  <p align="center">
    <a href="https://dashboard.trainingcqy.com" style="text-decoration:none"><strong>Metabase 演示</strong></a>
    &nbsp;&nbsp; | &nbsp;&nbsp;
    <a href="https://agent.trainingcqy.com" style="text-decoration:none"><strong>Agent 交互</strong></a>
  </p>
</div>

利用工作流定时采集 Hacker News 与 TechCrunch 的科技新闻，由 Jina Reader 获取新闻原文，以辅助 LLM 进行摘要生成、情感分析以及分类的结构化处理。并使用 Jina Embeddings 生成语义向量以支持相似度搜索，统一存入数据库。最终通过 Metabase 仪表盘、邮件日报、Web 端和 Telegram Bot 进行展示与交互。

<p align="center">
<svg viewBox="0 0 1000 1100" width="100%" xmlns="http://www.w3.org/2000/svg">
<image href="https://raw.githubusercontent.com/Trainingdlu/TechNews_Intelligence/main/assets/screenshots/Metabase_1.png" x="0" y="0" width="390" height="240" preserveAspectRatio="xMidYMid slice" />
<image href="https://raw.githubusercontent.com/Trainingdlu/TechNews_Intelligence/main/assets/screenshots/Metabase_2.png" x="400" y="0" width="390" height="240" preserveAspectRatio="xMidYMid slice" />
<image href="https://raw.githubusercontent.com/Trainingdlu/TechNews_Intelligence/main/assets/screenshots/TG.jpg" x="800" y="0" width="200" height="490" preserveAspectRatio="xMidYMid slice" />
<image href="https://raw.githubusercontent.com/Trainingdlu/TechNews_Intelligence/main/assets/screenshots/Metabase_3.png" x="0" y="250" width="390" height="240" preserveAspectRatio="xMidYMid slice" />
<image href="https://raw.githubusercontent.com/Trainingdlu/TechNews_Intelligence/main/assets/screenshots/Metabase_4.png" x="400" y="250" width="390" height="240" preserveAspectRatio="xMidYMid slice" />
<image href="https://raw.githubusercontent.com/Trainingdlu/TechNews_Intelligence/main/assets/screenshots/Email.jpg" x="0" y="500" width="300" height="600" preserveAspectRatio="xMidYMid slice" />
<image href="https://raw.githubusercontent.com/Trainingdlu/TechNews_Intelligence/main/assets/screenshots/Web.png" x="300" y="500" width="700" height="600" preserveAspectRatio="xMidYMid slice" />
</svg>
</p>

---

# 1. 系统架构
项目基于 ELT 架构，使用 Docker Compose 编排，包含五个核心服务：n8n（工作流与向量化）、PostgreSQL（存储与向量检索）、Metabase（可视化）、Web 网页（前端交互入口）、Telegram Bot（应用交互入口），同时提供本地 CLI 入口。

![系统架构](https://raw.githubusercontent.com/Trainingdlu/TechNews_Intelligence/main/assets/svg/architecture.svg)

---

# 2. 功能说明

## 2.1 数据采集与处理
系统通过 n8n 编排三条工作流实现全自动化采集与结构化处理

**主工作流**：每小时自动触发，获取新闻数据后通过 Jina Reader 提取全文，再调用 LLM 输出结构化 JSON (标题翻译、摘要、情感、分类)。写入成功后自动调用 Jina Embeddings 生成 1024 维语义向量并存入 `news_embeddings` ，确保语义搜索始终覆盖最新数据分析。已有数据仅更新热度值。处理失败的新闻数据存入`tech_news_failed` 表防止重复尝试。

![Main](https://raw.githubusercontent.com/Trainingdlu/TechNews_Intelligence/main/assets/screenshots/Main_workflow.png)

**异常捕获与告警**：全局错误处理，自动捕获异常并发送告警邮件。  

![Error](https://raw.githubusercontent.com/Trainingdlu/TechNews_Intelligence/main/assets/screenshots/Alert_workflow.png)

**日报推送**：每日 08:00 筛选近 24 小时价值新闻，并渲染为 HTML 邮件推送。

![Brief](https://raw.githubusercontent.com/Trainingdlu/TechNews_Intelligence/main/assets/screenshots/Brief_workflow.png)

### 工作流处理流程

#### 主工作流核心处理流程
**输入内容**：n8n 先调用 Jina Reader API 将原始 HTML 页面去噪（剔除导航、广告、脚注等无关元素），转为较为干净的 Markdown 格式全文（包含标题 + 正文），作为 LLM 的 Prompt 输入，能够显著提升摘要质量。
**输出 JSON 的完整字段**：
  `title_cn`：中文标题
  `summary`：摘要
  `sentiment`：情绪标签
  `category`：分类标签
**情感分类**：三个情绪分类 —— Positive（正面）、Neutral（中性）、Negative（负面）。
**新闻分类**：六个分类标签 —— AI、安全、硬件、开发、商业、生态。
**向量写入机制**：主工作流采用同步执行。当新闻成功写入 `tech_news` 表后，立即调用 Jina Embeddings API 生成 1024 维语义向量，随后存入 `news_embeddings` 表，确保 Agent 检索始终覆盖最新数据。
**失败记录存储**：失败记录进入 `tech_news_failed` 死信队列，避免重复处理。

#### 异常捕获与告警核心流程
**触发机制**：采用双重触发模式。一是通过 n8n 全局 `Error Trigger` 监听工作流运行时的突发异常；二是在主工作流中通过 `Data_check` 等节点对输出质量进行实时校验，若发现空数据或解析异常，则主动调用告警流。
**错误处理**：自动提取执行失败的节点名称、错误详情及关联的 URL，确保故障定位的准确性。
**日志记录**：将故障数据（时间、类型、错误信息）写入 `system_logs` 表，为系统稳定性分析提供静态数据支撑。
**邮件通知**：渲染 HTML 邮件并发送至管理员邮箱。

#### 日报推送核心流程
**任务触发**：每日 08:00 定时启动，通过 PostgreSQL 视图 `view_dashboard_news` 筛选近 24 小时内的增量数据。
**新闻选择**：在数据库层面执行聚合查询，选取 Hacker News 社区热度前 7（Points 倒序）以及 TechCrunch 媒体最近前 7 的文章。
**动态渲染**：利用 n8n Code 节点执行 JavaScript 逻辑。根据新闻的情感标签匹配特定的视觉样式，生成定制化的 HTML 邮件内容。
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
| `system_logs`      | 系统运行日志                                |

视图 `view_dashboard_news` 封装了时区转换（UTC → UTC+8）、来源分类（HackerNews / TechCrunch）、hours_ago 计算和 HN 讨论链接生成，分析查询层不重复这些逻辑。

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

>截至2026年3月，系统已稳定运行约 3 个月，累计收录约 6000 条新闻数据，基于该数据的分析如下：
>**情绪与热度**：负面新闻平均热度最高（208.5），显著高于正面（161.1）和中性（128.2）。尽管负面新闻数量最少（1,112 条），但负面新闻传播影响力最强。
>**平台情绪差异**：TC 正面主导，HN 中性主导。HackerNews 负面率（17.3%）低于 TechCrunch（23.2%），差异不在负面，而在中性。HN 以中性为主（45.3%），TC 以正面为主（54.7%）。说明媒体倾向用正面框架报道，社区讨论更趋中立。
>**分类负面率**：安全类负面率高达 63%（分类中最高）；AI 和开发类负面率最低（3.8% / 5%），但平均热度反而最高（100 / 123），呈现“高热度 ≠ 高风险”的明显特征。
>**发布热力图**：单个最高热度时段为周一 14:00（316），次高峰出现在周四 13:00（295）；此外，周六 08:00（258）与周日 09:00（269）等周末早间时段，以及大部分的深夜（22:00-23:00）均出现了显著的活跃高峰，体现了HN社区的热度分布呈现高度碎片化。

## 2.4 深度分析 Agent
基于 Gemini 2.5 Pro 构建的交互式 Agent，支持 Web 前端、Telegram Bot、本地 CLI 三种接入方式。

### 核心能力

**混合搜索**：同时查询 pgvector 语义相似度与关键词精确匹配，合并去重后返回结果。纯向量检索在公司名、产品名等专有名词上容易召回偏移，关键词匹配作为兜底保证精确查询的稳定性。
**全文阅读**：从 `jina_raw_logs` 提取新闻原文，基于全文而非摘要进行分析。
**时效感知**：首次交互时主动获取数据库最新文章时间与近 21 天数据分布，在回复中标注数据截止时间。
**多轮上下文**：支持追问，对话上下文自动保持。
**Token 配额**：Agent 调用 Gemini API 有实际成本，配额机制防止意外账单，额度耗尽后自动触发管理员审批邮件。
**对话历史存储**：
  Telegram Bot：进程内 `conversation_histories` 字典（按 `chat_id` 隔离）
  Web 前端：无状态 API，历史由客户端携带在 `/chat` 请求的 `history` 参数中传入
**时间衰减因子**：**非 pgvector 内置**，而是在 `tools.py` 的查询 SQL 中手动实现：  
  `0.1 * EXP(-EXTRACT(EPOCH FROM (NOW() - t.created_at)) / 86400.0 / 21)`
**配额成本控制**：系统内置配额机制，额度耗尽后自动触发管理员审批邮件。

---

## 3. 目录结构
```text
TechNews_Intelligence/
├── etl_workflow/                   # n8n 工作流配置
│   ├── Tech_Intelligence.json      # 主采集流程（含向量化节点）
│   ├── System_Alert_Service.json   # 异常捕获流程
│   └── Daily_Tech_Brief.json       # 日报推送流程
│
├── agents/                         # AI 深度分析 Agent
│   ├── db.py                       # 数据库连接池
│   ├── tools.py                    # Agent 工具函数（搜索、全文读取等）
│   ├── prompts.py                  # System Prompt 模板
│   ├── agent.py                    # LLM 客户端、Chat 工厂
│   ├── api.py                      # Web API 入口（FastAPI + Token 管理）
│   ├── mail.py                     # 邮件发送（Token 发放、审批通知）
│   ├── bot.py                      # Telegram Bot 入口（Docker 部署）
│   ├── cli.py                      # 本地终端交互入口
│   └── requirements.txt            # Python 依赖
│
├── frontend/                       # Web 前端（静态页面）
│   ├── index.html                  # 页面结构
│   ├── style.css                   # 样式
│   └── app.js                      # 交互逻辑
│
├── sql/
│   ├── infrastructure/             # DDL：建表与视图
│   │   ├── schema_ddl.sql          # 表结构定义
│   │   └── view_logic.sql          # 视图逻辑（时区转换、来源分类、指标计算）
│   │
│   └── analytics/                  # DML：Metabase 分析查询
│
├── deployment/                     # Docker 部署配置
│   ├── docker-compose.yml
│   └── .env.example
│
└── assets/
    ├── docs/                       # Metabase 仪表盘 PDF
    ├── screenshots/                # 工作流截图
    └── svg/                        # 架构图
```

---

## 4. 部署
项目已容器化，可通过 Docker Compose 部署。

### 前置条件
已安装 Docker
已获取 LLM API Key（阿里通义 / Gemini）及 Jina API Key
已创建 Telegram Bot 并获取 Bot Token（通过 @BotFather）
### 步骤
```bash
# 1. 克隆仓库
git clone https://github.com/Trainingdlu/TechNews_Intelligence.git
cd TechNews_Intelligence

# 2. 配置环境变量
cp deployment/.env.example deployment/.env
# 编辑 deployment/.env，填入数据库密码、API Key 和 TELEGRAM_BOT_TOKEN

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

### 本地 CLI（可选）
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

---

## 6. 作者
Trainingcqy · trainingcqy@gmail.com

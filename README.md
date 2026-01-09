# AI 驱动的科技新闻系统

![Tech Stack](https://img.shields.io/badge/stack-n8n_|_PostgreSQL_|_Metabase_|_DeepSeek-blue?style=flat-square)
![License](https://img.shields.io/badge/license-AGPL_3.0-red?style=flat-square)

> **项目概述**：一个端到端的数据工程与商业智能解决方案。旨在通过自动化流水线解决科技行业的信息过载问题，实时采集 Hacker News 与 TechCrunch 的非结构化数据，利用大语言模型（LLM）进行结构化清洗与情感分析，最终通过交互式仪表盘为决策提供量化支持。

**[在线演示 Dashboard](https://dashboard.trainingcqy.com)** | **[项目源码](https://github.com/Trainingdlu/TechNews_Intelligence)**

---

## 1. 系统架构设计

本项目遵循现代化的 **ELT (Extract, Load, Transform)** 架构设计，确保了数据流的高可用性与可扩展性。所有组件均已容器化，运行于 Azure 云基础设施之上。

```mermaid
flowchart LR
    classDef node fill:#ffffff,stroke:#000000,stroke-width:1px,color:#000000,rx:0,ry:0;
    classDef logic fill:#ffffff,stroke:#000000,stroke-width:1px,color:#000000;
    classDef db fill:#ffffff,stroke:#000000,stroke-width:1px,shape:cylinder,color:#000000;
    
    A[RSS / Hacker News]
    B[n8n Automation]
    C{Is New Data?}
    D[Jina Reader API]
    E[DeepSeek-V3 LLM]
    F[Update Points]
    G[(PostgreSQL)]
    H[Metabase BI]

    A -->|Trigger| B
    B --> C
    
    C -- Yes --> D
    D -->|Full Text| E
    E -->|JSON Output| G
    
    C -- No --> F
    F -.->|Update| G
    
    G -->|SQL Query| H

    class A,B,D,E,F,H node;
    class C logic;
    class G db;
```

---

## 2. 核心实施阶段

### 2.1 数据采集与处理层
利用 **n8n** 作为工作流编排引擎，实现了数据获取的自动化与智能化。
*   **多源异构数据获取**：“构建了基于 HTTP Polling 的混合采集层，兼容 REST API (Hacker News) 与 RSS 订阅源 (TechCrunch)，实现了多源异构数据的统一接入。
*   **非结构化数据清洗**：集成 **Jina Reader** 将杂乱的 HTML 网页转换为干净的 Markdown 文本，提高 LLM 的准确率和效率。
*   **AI 语义增强**：调用 **DeepSeek-V3** 模型，对长文本进行 NLP 处理，输出标准化的 JSON 数据：
    *   **智能摘要**：生成 100 字以内的高密度关键事实摘要。
    *   **情感量化**：自动标记新闻情感倾向（Positive/Neutral/Negative）。
    *   **自动分类**：基于内容上下文自动提取赛道标签（如 AI、商业、安全）。
    *   **成本控制**：采用 DeepSeek-V3 模型，在模型能力，降低推理成本，实现了高性价比的文本清洗。

### 2.2 数据仓库与建模层
使用 **PostgreSQL 15** 作为核心数据仓库，通过分层设计保证数据的一致性与查询效率。
*   **Schema 设计**：设计了包含 `url` 唯一约束的表结构，有效防止数据冗余和重复抓取。
*   **视图抽象**：构建 `view_dashboard_news` 视图层，封装了底层逻辑：
    *   **时区标准化**：将 UTC 时间转换为 UTC+8（北京时间）。
    *   **数据去重**：基于 URL 自动归类去重。
    *   **CDC 支持**：配置了 `updated_at` 触发器，自动记录数据变更时间。

### 2.3 业务分析与算法层
本项目不仅仅是数据的展示，更包含了深度的业务逻辑分析。核心算法包括：
*   **Hacker News 重力算法复刻**：
    *   逻辑：`Score = Points / (Time + 2)^1.8`
    *   价值：引入时间衰减因子，识别“当前上升速度最快”的热点，而非单纯的历史高分内容。
*   **赛道周环比增长**：
    *   逻辑：使用 CTE 与 Self-Join 技术。
    *   价值：量化不同技术赛道（如 Rust, AI）的热度变化趋势，捕捉潜在的市场风口。
*   **巨头声量份额分析**：
    *   逻辑：使用 `UNION ALL` 解决多标签重叠统计问题。
    *   价值：计算 OpenAI、Google 等科技巨头的讨论热度与占比。

### 2.4 可视化与交互层
基于 **Metabase** 构建 BI 仪表盘，强调交互体验与信息分层。
*   **主从联动交互**：利用 SQL 变量注入技术 (`[[AND id = {{selected_id}}]]`)，实现了点击左侧列表标题，右侧详情卡片刷新摘要的功能。

---

## 3. 目录结构与 SQL

项目采用标准的工程化目录结构，实现基础设施代码 (IaC) 与业务逻辑分离。

```text
TechNews_Intelligence/
├── sql/
│   ├── infrastructure/          # DDL: 基础设施层 (建表与视图)
│   │   ├── _schema_ddl.sql    # 数据库表结构定义 (含 Trigger/Index)
│   │   └── _view_logic.sql    # 视图层逻辑 (时区转换、清洗、指标预计算)
│   │
│   └── analytics/               # DML: 业务分析层 (Metabase 核心逻辑)
│       ├── _algo_gravity_ranking.sql         # HN 重力排名算法
│       ├── _analysis_category_growth.sql     # 赛道周环比增长率
│       ├── _analysis_heatmap.sql             # 黄金发布时间热力图
│       ├── _analysis_negativity_index.sql    # 赛道负面率/风险指数
│       ├── _analysis_source_bias.sql         # 媒体 vs 社区舆论温差
│       ├── _analysis_tech_giants_battle.sql  # 科技巨头声量份额
│       ├── _card_dynamic_summary.sql         # 动态摘要卡片
│       ├── _chart_engagement.sql             # 情绪与热度效能分析
│       ├── _chart_market_attention.sql       # 市场注意力分布
│       ├── _chart_sentiment_trend.sql        # 舆情分时趋势
│       ├── _chart_techcrunch_daily.sql       # 媒体日更趋势
│       ├── _table_community_hits.sql         # 社区热议
│       ├── _table_hackernews_top.sql         # Hacker News 热榜
│       └── _table_techcrunch_latest.sql      # TechCrunch 快讯
│
├── deployment/                  # Docker Compose 配置文件与环境模板
└── assets/                      # 项目文档截图
```

---

## 4. 本地部署

本项目完全容器化，支持一键部署。

### 前置条件
*   已安装 Docker & Docker Compose
*   获取 DeepSeek API Key 及 Jina Reader API Key

### 1: 克隆仓库
```bash
git clone https://github.com/Trainingdlu/TechNews_Intelligence.git
cd TechNews_Intelligenceo
```

### 2: 环境配置
复制配置文件模板并填入凭证。
*注意：`docker-compose.yml` 已配置为生产环境模式。*

```bash
cp deployment/.env.example deployment/.env
# 使用文本编辑器修改 .env 文件，填入你的数据库密码及 API Keys
```

### 3: 启动服务
```bash
cd deployment
docker-compose up -d
```

### 4: 导入工作流
1.  访问 `http://localhost:5678` 进入 n8n 管理界面。
2.  导入 `etl_workflow/tech_news_pipeline.json` 文件。
3.  **重要**：在 n8n 界面中配置 PostgreSQL 凭证。
4.  激活工作流。

---

## 5. 许可协议与商业使用

本项目采用 **GNU AGPLv3** 开源协议。

---

## 6. 作者

**Trainingcqy**<br> [trainingcqy@gmail.com](mailto:trainingcqy@gmail.com)

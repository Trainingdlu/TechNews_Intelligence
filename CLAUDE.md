# TechNews_Intelligence

证据接地的科技新闻情报 Agent。用户的**简历核心项目**(服务首次实习求职)。
跨项目协作风格见全局 `~/.claude/CLAUDE.md`;本文件只写本项目特定事实。

## 架构一句话
n8n 采集(`etl_workflow/`)→ PostgreSQL + pgvector 存储(`sql/`)→ LangGraph Agent(`agent/`)做证据检索与分析 → 前端 / 观测(`frontend/`、`app/`、`trace_dashboard/`)。评测体系在 `eval/`,单测在 `tests/`(全绿基线 284)。

## 目录导航
- `agent/` — LangGraph agent 本体 + 工具体系(`agent/tools/`)
- `services/` — db 连接池、`entity_resolution`(实体别名)、`embeddings`
- `eval/` — 分层评测体系(G1–G5)
- `tests/` — pytest 单测(改完必跑)
- `sql/` — schema / 检索 SQL
- `etl_workflow/` — n8n 采集流程
- `frontend/`、`app/`、`trace_dashboard/` — 界面与可观测

## 检索引擎(one-core-two-profiles)
- 统一核心 `fetch_hybrid_rows`(`agent/tools/hybrid_retrieval.py`):真三路 RRF——lexical(Postgres FTS)+ semantic(pgvector 余弦)+ exact(`entity_alias` + pg_trgm),RRF k=60。
- 两个 profile 共用同一核心:点查工具走 QUERY_PROFILE(`retrieval.py`);5 个宏观分析工具走 ANALYSIS_PROFILE(`hybrid_pool.py` → `rerank_aggregation.py`,Jina rerank)。**单一真源**。
- 嵌入 / 重排:`jina-embeddings-v3` / `jina-reranker-v3`。

## 简历 / 面试策略(重要)
- 简历分两个带标签子系统:① 检索增强与质量评测 ② Agent 运行时与工具 / 协议。bullet 上限 ~6–8 条。
- **只纵向加深(深度),绝不横向铺新条目。** 加内容 = 把已有条目讲更深,而不是再加一条。
- 简历与话术文档在桌面:`简历项目经历.md`、`面试话术.md`(都写最终态)。
- 头条指标:检索质量 before→after delta(nDCG,见 `eval/` 报告)。讲指标要能抗追问,不吹未做的东西。

## 评测体系
- 用户第一次搭 eval——讨论时把概念 / 方法论讲清楚,不只给代码。
- 重点指标(4 星+):IR 检索指标、faithfulness / 幻觉、before/after delta、人审 kappa。
- G5 等评测脚本:给命令,用户手动跑、看实时输出。

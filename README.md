<div align="center">
  <img src="https://raw.githubusercontent.com/Trainingdlu/TechNews_Intelligence/main/assets/svg/title.svg" alt="TechNews Intelligence" width="700">
  <br>
  <p align="center">
    <a href="https://dashboard.trainingcqy.com" style="text-decoration:none"><strong>Metabase 演示</strong></a>
    &nbsp;&nbsp; | &nbsp;&nbsp;
    <a href="https://agent.trainingcqy.com" style="text-decoration:none"><strong>智能体交互</strong></a>
  </p>
</div>

TechNews Intelligence 是面向科技新闻的自动化情报系统。系统通过 n8n 采集新闻和正文，使用 PostgreSQL + pgvector 存储结构化数据和向量数据，并提供可检索、可追踪、可评测的 Agent 分析能力。用户可以通过 Web API、Trace Console、Telegram Bot、Metabase 看板和本地 CLI 使用系统。

![效果展示](https://raw.githubusercontent.com/Trainingdlu/TechNews_Intelligence/main/assets/previews/showcase.png)

## 核心能力

- 新闻采集：n8n 工作流定时采集多种官方来源新闻，写入 PostgreSQL。
- 文本处理：Jina Reader 提取正文，模型生成摘要、分类和情绪标签。
- 混合检索：词法、pgvector 语义、实体别名精确三路召回 + RRF 融合，叠加时间衰减，可选 Jina Reranker。
- Agent分析：基于 LangGraph StateGraph 编排上下文准备、意图路由、工具选择/规划/执行、证据归一、循环决策、澄清/证据不足分支和最终综合；图后执行输出 URL 守卫与来源装饰。
- 上下文记忆：Context Pack 整理当前问题相关历史；Thread Memory 保存线程摘要和历史证据索引。
- 可观测性：`agent_runs`、`agent_trace_spans`、`agent_model_io` 记录请求摘要、执行链路和完整模型输入输出。
- 多入口交互：Web API、Telegram Bot、本地 CLI、MCP stdio 共用同一套 Agent 运行时。
- 评测闭环：G1-G6 分层评测覆盖错误分析、检索、生成、意图、工具选择和负样本拒答，并支持 Trace 回溯。

## 系统组成

| 模块 | 说明 |
| --- | --- |
| `agent/` | Agent 编排、工具运行时、上下文整理、Trace、MCP 适配。 |
| `app/` | FastAPI 主服务、Trace Console 服务、Telegram Bot、本地 CLI。 |
| `services/` | PostgreSQL 连接、对话线程、Trace 持久化、邮件、线程记忆。 |
| `trace_dashboard/` | Vue3/Vite Trace 可视化面板。 |
| `frontend/` | 静态落地页与订阅页面。 |
| `eval/` | G1-G6 分层评测、Trace 查询、编码守卫和实体别名工具。 |
| `sql/` | 数据库 schema、视图、seed、分析 SQL。 |
| `deployment/` | Docker Compose、环境变量模板、数据库运维脚本。 |
| `etl_workflow/` | n8n 工作流导出文件。 |
| `docs/` | 架构、部署、开发和评测文档。 |

## 快速启动

```bash
git clone https://github.com/Trainingdlu/TechNews_Intelligence.git
cd TechNews_Intelligence
cp deployment/.env.example deployment/.env
```

编辑 `deployment/.env`，至少配置：

- `POSTGRES_USER`、`POSTGRES_PASSWORD`、`POSTGRES_DB`
- `GEMINI_API_KEY` 或 Vertex AI 相关变量
- `DEEPSEEK_API_KEY`
- `JINA_API_KEY`
- `TRACE_DASHBOARD_TOKEN`
- `TELEGRAM_BOT_TOKEN`（使用 Bot 时配置）

启动服务：

```bash
cd deployment
docker compose up -d
```

初始化或更新数据库结构：

```bash
bash deployment/scripts/db/apply_schema.sh
```

手动导入 seed：

```bash
bash deployment/scripts/db/apply_seed.sh
```

## 常用服务与端口

| 服务 | 容器 | 默认端口 | 说明 |
| --- | --- | --- | --- |
| PostgreSQL | `tech_news_db` | `5432` | 数据库与向量检索。 |
| n8n | `tech_news_n8n` | `5678` | 新闻采集和通知工作流。 |
| Metabase | `tech_news_bi` | `3000` | 数据分析看板。 |
| API | `tech_news_api` | `8000` | Web/程序调用入口。 |
| Trace Console | `tech_news_trace` | `8010` | Agent 链路追踪面板。 |
| Telegram Bot | `tech_news_bot` | 无 HTTP 端口 | Telegram 对话入口。 |

Cloudflare Tunnel 可按需映射：

| 域名 | Origin |
| --- | --- |
| `agentapi.trainingcqy.com` | `http://localhost:8000` |
| `trace.trainingcqy.com` | `http://localhost:8010` |
| `workflow.trainingcqy.com` | `http://localhost:5678` |
| `dashboard.trainingcqy.com` | `http://localhost:3000` |

## 常用命令

```bash
# 单元测试
pytest tests/unit -q

# Trace 面板构建
cd trace_dashboard
npm run build

# 本地 CLI
python -m app.cli

# 数据质量检查
bash deployment/scripts/db/run_data_quality_checks.sh

# 查询一次 Agent Trace
python eval/trace_query.py --request-id <request_id>
```

## 项目文档

- [系统架构](docs/ARCHITECTURE.md)
- [部署与运维](docs/OPERATIONS.md)
- [开发说明](docs/DEVELOPMENT.md)
- [评测体系](docs/EVALUATION.md)

## License

本项目用于科技新闻情报系统的采集、分析、展示和评测。第三方服务、模型和数据源的使用需遵守各自服务条款。

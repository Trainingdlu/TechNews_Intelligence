# 部署与运维

本文档描述生产和本地部署、数据库脚本、端口、Cloudflare Tunnel、n8n、Metabase、Trace Console 和常用运维命令。

## Docker Compose 服务

Compose 文件位于 `deployment/docker-compose.yml`。

| 服务 | 容器 | 端口 | 职责 |
| --- | --- | ---: | --- |
| `postgres` | `tech_news_db` | `5432` | PostgreSQL + pgvector。 |
| `n8n` | `tech_news_n8n` | `5678` | 工作流采集、处理和通知。 |
| `metabase` | `tech_news_bi` | `3000` | BI 看板。 |
| `api` | `tech_news_api` | `8000` | FastAPI 主入口。 |
| `trace` | `tech_news_trace` | `8010` | Trace Console 服务。 |
| `bot` | `tech_news_bot` | 无 HTTP 端口 | Telegram Bot。 |

启动服务：

```bash
cd deployment
docker compose up -d
```

查看状态：

```bash
docker compose ps
```

查看日志：

```bash
docker compose logs -f api
docker compose logs -f trace
docker compose logs -f bot
```

## 环境变量

环境变量模板位于 `deployment/.env.example`。部署时复制为 `deployment/.env`。

```bash
cp deployment/.env.example deployment/.env
```

关键变量：

| 变量 | 说明 |
| --- | --- |
| `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` | PostgreSQL 凭证和库名。 |
| `GEMINI_API_KEY` | Gemini API 调用。 |
| `VERTEX_PROJECT`, `GCP_SA_PATH`, `GOOGLE_APPLICATION_CREDENTIALS` | Vertex AI 调用。 |
| `DEEPSEEK_API_KEY` | DeepSeek 模型调用。 |
| `JINA_API_KEY` | Reader、Embedding 和可选 rerank。 |
| `TELEGRAM_BOT_TOKEN` | Telegram Bot。 |
| `TRACE_DASHBOARD_TOKEN` | Trace Console 管理员访问 token。 |
| `CORS_ORIGINS` | API 允许访问来源。 |
| `LANGSMITH_TRACING`, `LANGSMITH_PROJECT` | 可选 LangSmith 外部观测。 |

端口变量：

| 变量 | 默认值 |
| --- | ---: |
| `POSTGRES_PORT` | `5432` |
| `N8N_PORT` | `5678` |
| `METABASE_PORT` | `3000` |
| `API_PORT` | `8000` |
| `TRACE_DASHBOARD_PORT` | `8010` |

生成 Trace Console token：

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

## 数据库结构与 Seed

数据库结构脚本：

```bash
bash deployment/scripts/db/apply_schema.sh
```

该脚本执行：

- `sql/infrastructure/schema/schema_ddl.sql`
- `sql/infrastructure/views/view_dashboard_news.sql`

Seed 脚本：

```bash
bash deployment/scripts/db/apply_seed.sh
```

执行指定 seed：

```bash
bash deployment/scripts/db/apply_seed.sh --seed-file sql/infrastructure/seeds/seed_source_official.sql
```

`apply_schema.sh` 用于自动部署，`apply_seed.sh` 用于手动维护来源、实体和别名基础数据。

## 自动部署

GitHub Actions 工作流位于 `.github/workflows/deploy.yml`。`main` 分支推送后通过 SSH 进入服务器执行部署。

部署顺序：

1. `git pull --ff-only origin main`
2. 拉取基础镜像：`postgres`、`n8n`、`metabase`
3. 启动 `postgres`
4. 执行 `bash deployment/scripts/db/apply_schema.sh`
5. 构建 `bot`、`api`、`trace`
6. `docker compose up -d --remove-orphans`
7. 清理镜像并检查核心服务状态

自动部署不会执行 seed。

## Cloudflare Tunnel

推荐映射：

| 域名 | Origin |
| --- | --- |
| `workflow.trainingcqy.com` | `http://localhost:5678` |
| `dashboard.trainingcqy.com` | `http://localhost:3000` |
| `agentapi.trainingcqy.com` | `http://localhost:8000` |
| `trace.trainingcqy.com` | `http://localhost:8010` |

API 和 Trace 服务默认只绑定宿主机本地地址，适合通过 Tunnel 暴露。

## n8n

n8n 使用同一个 PostgreSQL 服务作为运行库。工作流 JSON 位于 `etl_workflow/`。

导入工作流后需要配置：

- PostgreSQL 凭证
- Jina API Key
- 模型 API Key
- SMTP 凭证
- n8n 基础认证

## Metabase

Metabase 使用 `view_dashboard_news` 和 `sql/analytics/` 中的查询 SQL 构建看板。看板侧不直接复写北京时间、来源归一和搜索字段逻辑。

## Source 与 Entity 维护

新增或更新来源：

```bash
bash deployment/scripts/db/upsert_source.sh \
  --source-key openai_blog \
  --source-name "OpenAI Blog" \
  --endpoint "https://openai.com/news/rss.xml"
```

跳过 schema 检查：

```bash
bash deployment/scripts/db/upsert_source.sh \
  --source-key openai_blog \
  --source-name "OpenAI Blog" \
  --endpoint "https://openai.com/news/rss.xml" \
  --skip-schema
```

实体别名候选生成：

```bash
bash deployment/scripts/db/build_entity_alias_candidates.sh --days 14 --limit 1000 --use-deepseek
```

执行前确保 schema/view：

```bash
bash deployment/scripts/db/build_entity_alias_candidates.sh --apply-schema --days 14 --limit 1000 --use-deepseek
```

## 数据质量检查

```bash
bash deployment/scripts/db/run_data_quality_checks.sh
bash deployment/scripts/db/run_data_quality_checks.sh 48
```

检查项包括：

- 最近采集数量。
- 空标题、空摘要、空 URL。
- 重复 URL。
- 向量覆盖情况。
- 来源分布。
- 全文和结构化字段完整性。

## Trace Console

Trace Console 服务由 `trace` 容器提供。浏览器访问 `trace.trainingcqy.com` 或本地 `http://localhost:8010`，输入 `TRACE_DASHBOARD_TOKEN` 后进入面板。

面板展示：

- 请求列表。
- Span 调用链。
- 模型输入 messages 和 raw output。
- 工具参数、结果摘要、证据 URL 和 diagnostics。
- 错误码、异常链和 raw JSON。

## 常用命令

```bash
# 更新 schema/view
bash deployment/scripts/db/apply_schema.sh

# 手动 seed
bash deployment/scripts/db/apply_seed.sh

# 数据质量检查
bash deployment/scripts/db/run_data_quality_checks.sh

# 查看 API 日志
cd deployment
docker compose logs -f api

# 查看 Trace 日志
docker compose logs -f trace

# 重启服务
docker compose restart api trace bot
```

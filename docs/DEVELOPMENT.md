# 开发说明

本文档描述本地开发、服务启动、测试、前端构建、CI 和目录结构。

## 本地环境

Python 版本：3.11。

安装依赖：

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

本地 CLI 环境变量：

```bash
cp agent/.env.example agent/.env
```

Docker 部署环境变量：

```bash
cp deployment/.env.example deployment/.env
```

## 服务启动

API：

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

Trace Console API：

```bash
uvicorn app.trace_api:app --host 0.0.0.0 --port 8010
```

Telegram Bot：

```bash
python -m app.bot
```

CLI：

```bash
python -m app.cli
```

MCP stdio server：

```bash
python -m agent.mcp.stdio_server
```

## Trace Dashboard

开发模式：

```bash
cd trace_dashboard
npm install
npm run dev
```

生产构建：

```bash
cd trace_dashboard
npm run build
```

预览构建产物：

```bash
cd trace_dashboard
npm run preview
```

生产环境由 `trace` 容器提供静态页面和 `/trace-api/*` 接口。

## 测试

单元测试：

```bash
pytest tests/unit -q
```

重点模块测试：

```bash
pytest tests/unit/test_agent_trace.py tests/unit/test_agent_trace_store.py -q
pytest tests/unit/test_tool_runtime_components.py -q
pytest tests/unit/test_context_manager.py tests/unit/test_thread_memory.py -q
pytest tests/unit/test_custom_graph_runtime.py -q
```

Python 编译检查：

```bash
python -m py_compile \
  app/api.py app/trace_api.py app/bot.py \
  agent/graph/nodes.py agent/core/trace.py \
  services/agent_trace_store.py services/thread_memory.py
```

编码检查：

```bash
python eval/encoding_guard.py --root . --report eval/reports/encoding_guard/local.json
```

## CI

CI 工作流位于 `.github/workflows/ci.yml`。

触发范围：

- `agent/**`
- `app/**`
- `services/**`
- `eval/**`
- `tests/**`
- `requirements.txt`
- `.github/workflows/ci.yml`

CI 步骤：

1. 安装 Python 3.11。
2. 安装 `requirements.txt` 和 `pytest`。
3. 运行编码检查。
4. 执行 `pytest tests -v`。
5. 上传编码检查报告。

## 目录结构

| 路径 | 说明 |
| --- | --- |
| `agent/` | Agent 运行时、Graph、工具、Trace、上下文管理、MCP。 |
| `app/` | API、Trace API、Telegram Bot、CLI。 |
| `services/` | DB、对话线程、Trace Store、邮件、线程记忆。 |
| `trace_dashboard/` | Vue3 Trace Console。 |
| `eval/` | 评测脚本、评分、报告生成、Trace 查询。 |
| `tests/unit/` | 单元测试。 |
| `sql/infrastructure/` | schema、view、seed、数据质量检查 SQL。 |
| `sql/analytics/` | Metabase 和分析查询 SQL。 |
| `deployment/` | Compose、环境变量模板、部署脚本。 |
| `etl_workflow/` | n8n 工作流 JSON。 |
| `docs/` | 项目长期文档。 |

## 代码边界

- 业务入口在 `app/`。
- Agent 编排和工具执行在 `agent/`。
- 可复用基础服务在 `services/`。
- 数据库结构变更写入 `sql/infrastructure/schema/schema_ddl.sql`。
- 部署和数据库运维脚本放入 `deployment/scripts/`。
- 评测逻辑放入 `eval/`，测试放入 `tests/unit/`。
- 运行产物不作为源码维护，包括 `eval/reports/*`、临时数据集、缓存、构建产物和虚拟环境。

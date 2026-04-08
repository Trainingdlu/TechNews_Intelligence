# TechNews 项目结构（与当前仓库对齐）

> 更新时间：2026-04-08

本文档用于说明当前代码仓库的真实结构、模块职责与可忽略的本地产物，避免 README 与代码漂移。

## 1）仓库根目录（源码与配置）

| 路径 | 作用 |
| --- | --- |
| `.github/workflows/` | 持续集成、部署、手动评测工作流（`ci.yml`/`deploy.yml`/`eval-manual.yml`） |
| `.vscode/` | 编辑器工作区配置（当前为 `settings.json`） |
| `agent/` | 智能体运行时核心：ReAct、工具、提示词、技能基础设施、MCP 扩展 |
| `app/` | 应用入口层：API、Telegram 机器人、命令行 |
| `services/` | 公共服务层：数据库访问与邮件通知 |
| `eval/` | 离线评测框架、数据集与评测报告输出目录 |
| `tests/` | 自动化测试（unit + utils + reports） |
| `deployment/` | Docker Compose、环境变量模板、数据库脚本与样例数据 |
| `docs/` | 项目文档与测试文档 |
| `etl_workflow/` | n8n 工作流 JSON |
| `frontend/` | 静态前端页面与交互脚本 |
| `sql/` | 结构化 SQL（基础设施 SQL + 分析 SQL） |
| `assets/` | 展示图、截图、SVG 资源 |
| `Dockerfile` | 运行镜像构建文件 |
| `requirements.txt` | 根依赖主文件（统一安装入口） |
| `README.md` | 对外项目说明 |

## 2）运行时分层

### 2.1 智能体核心（`agent/`）

| 路径 | 职责 |
| --- | --- |
| `agent/agent.py` | ReAct 主调度、工具调用循环、证据后处理、响应生成 |
| `agent/tools.py` | 检索/分析工具与技能实现 |
| `agent/prompts.py` | 提示词策略与输出约束 |
| `agent/core/` | `skill_contracts`/`skill_registry`/`tool_hooks`/`runtime_factories`/`evidence`/`metrics` |
| `agent/mcp/` | MCP client/server/stdio 扩展能力 |
| `agent/.env.example` | 智能体运行环境模板 |

### 2.2 应用入口（`app/`）

| 文件 | 职责 |
| --- | --- |
| `app/api.py` | FastAPI 网页 API 入口 |
| `app/bot.py` | Telegram 机器人入口 |
| `app/cli.py` | 本地命令行入口 |

### 2.3 公共服务（`services/`）

| 文件 | 职责 |
| --- | --- |
| `services/db.py` | PostgreSQL 连接池、查询辅助 |
| `services/mail.py` | 邮件通知与审批提醒 |

## 3）数据与基础设施

### 3.1 ETL 工作流（`etl_workflow/`）

- `Tech_Intelligence.json`
- `System_Alert_Service.json`
- `Daily_Tech_Brief.json`

### 3.2 SQL 结构（`sql/`）

| 路径 | 说明 |
| --- | --- |
| `sql/infrastructure/schema/schema_ddl.sql` | 表结构与索引、触发器等 |
| `sql/infrastructure/views/view_dashboard_news.sql` | 统一视图逻辑（来源归一化、指标字段） |
| `sql/infrastructure/checks/data_quality_checks.sql` | 数据质量检查 SQL |
| `sql/infrastructure/seeds/seed_source_official.sql` | 来源种子数据 |
| `sql/analytics/` | Metabase 分析查询 SQL |

### 3.3 部署目录（`deployment/`）

| 路径 | 说明 |
| --- | --- |
| `deployment/docker-compose.yml` | 容器编排 |
| `deployment/.env.example` | 部署环境变量模板 |
| `deployment/scripts/db/` | 数据库运维脚本（初始化、upsert、检查） |
| `deployment/data/` | 样例/备份数据文件 |

## 4）评测与测试

| 路径 | 说明 |
| --- | --- |
| `eval/run_eval.py` | 评测入口 |
| `eval/eval_core.py` | 指标与质量门控核心 |
| `eval/dataset_loader.py` | 数据集加载与筛选 |
| `eval/capabilities.py` | 能力维度定义 |
| `eval/datasets/` | 评测数据集（`default`/`smoke`/`accuracy_snapshot` 等） |
| `eval/reports/` | 评测结果输出目录 |
| `tests/unit/` | 单元测试用例 |
| `tests/utils/` | 测试辅助代码 |
| `tests/reports/` | 测试输出目录 |
| `docs/testing/` | 测试方法与评测说明文档 |

## 5）可忽略的本地产物（非核心源码）

以下目录/文件通常是本地运行或测试生成，可在结构审查时忽略：

- `.venv/`
- `.tmp/`
- `.pytest_cache/`
- 各目录下 `__pycache__/`
- `tests/unit/.tmp_mcp_stdio/`

## 6）维护约定

1. 任何顶层目录变更（新增/重命名/拆分）后，同步更新本文件与 `README.md` 的目录结构区块。
2. 目录说明优先以“职责”命名，而不是以“实现细节”命名，避免频繁失效。
3. 以仓库中真实存在的路径为准，仅记录当前目录与路径。


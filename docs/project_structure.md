# TechNews 项目文件结构说明

本文档用于说明当前仓库各目录与关键文件的职责，便于后续维护、交接与扩展。

## 1. 根目录总览

| 路径 | 作用 |
| --- | --- |
| `.github/workflows/` | CI、部署、手动评测工作流配置 |
| `agents/` | Agent 服务端、工具层、评测与单元测试 |
| `assets/` | 架构图、页面/工作流截图等静态素材 |
| `deployment/` | Docker 编排、数据库脚本与初始化数据 |
| `docs/` | 项目文档（数据质量、来源接入、测试说明等） |
| `etl_workflow/` | n8n 工作流 JSON（主链路/告警/日报） |
| `frontend/` | Web 前端页面（主界面 + 订阅页） |
| `sql/` | 数据库基础设施 SQL 与分析查询 SQL |
| `README.md` | 对外总览与快速上手 |
| `Dockerfile` | 容器镜像构建配置 |

## 2. Agent 模块（`agents/`）

### 2.1 入口与核心能力

| 文件 | 功能 |
| --- | --- |
| `agents/agent.py` | Agent 运行时主入口；负责路由策略、工具编排、LangChain/Fallback 执行 |
| `agents/api.py` | FastAPI 接口层；负责 Token 鉴权、配额、订阅接口、聊天接口 |
| `agents/bot.py` | Telegram Bot 入口；负责会话历史、频控、消息重试与发送 |
| `agents/cli.py` | 本地命令行调试入口 |
| `agents/tools.py` | 数据检索与分析工具集合（query/search/trend/timeline/landscape/fulltext） |
| `agents/prompts.py` | Agent 系统提示词模板 |
| `agents/db.py` | PostgreSQL 连接池管理 |
| `agents/mail.py` | 邮件发送工具（Token、额度审批通知） |

### 2.2 核心子模块（`agents/core/`）

| 文件 | 功能 |
| --- | --- |
| `agents/core/router.py` | 用户意图识别与参数抽取（天数、limit、source 等） |
| `agents/core/pipelines.py` | 各类强制路由场景的可复用执行 pipeline |
| `agents/core/metrics.py` | 路由指标统计与快照输出 |
| `agents/core/evidence.py` | 证据 URL 处理、引用格式化、来源段构建 |

### 2.3 评测与测试

| 路径 | 功能 |
| --- | --- |
| `agents/eval/run_eval.py` | 评测执行入口（题库加载、运行、报告输出、门禁阈值） |
| `agents/eval/datasets/default.jsonl` | 默认评测题库（当前唯一默认入口） |
| `agents/eval/datasets/smoke.jsonl` | 冒烟评测题库 |
| `agents/eval/eval_core.py` | 评测指标计算与质量门禁 |
| `agents/eval/dataset_loader.py` | 题库解析、过滤与能力映射 |
| `agents/eval/capabilities.py` | 能力注册表 |
| `agents/tests/unit/` | 单元测试（路由、bot稳健性、eval核心、工具结构化输出） |

## 3. 部署与数据库脚本（`deployment/`）

### 3.1 部署文件

| 文件 | 功能 |
| --- | --- |
| `deployment/docker-compose.yml` | 服务编排（数据库/可视化/相关服务） |
| `deployment/.env.example` | 部署环境变量模板 |
| `deployment/data/news_data.sql` | 全量库结构/数据导入文件（保留） |
| `deployment/data/news_data_readable.sql` | 可读版导入文件（保留） |

### 3.2 DB 脚本入口（`deployment/scripts/db/`）

| 脚本 | 功能 |
| --- | --- |
| `common.sh` | DB 脚本公共库（读取 `.env`、Compose 调用、psql 封装） |
| `apply_source_framework_migration.sh` | 一键执行 schema/view/seed（支持跳过或指定 seed） |
| `upsert_source.sh` | 一键新增/更新来源到 `source_registry`（可选自动迁移） |
| `run_data_quality_checks.sh` | 只读数据质量巡检（按小时窗口） |

## 4. SQL 分层（`sql/`）

### 4.1 基础设施 SQL（`sql/infrastructure/`）

| 文件 | 功能 |
| --- | --- |
| `schema/schema_ddl.sql` | 核心建表/补字段/索引/触发器定义（含来源框架相关表结构） |
| `views/view_dashboard_news.sql` | 仪表盘视图逻辑（来源回退、时区与衍生字段） |
| `seeds/seed_source_official.sql` | 官方来源种子数据（幂等 upsert） |
| `checks/data_quality_checks.sql` | 全面只读数据质量检查 SQL |

### 4.2 分析 SQL（`sql/analytics/`）

用于 Metabase 卡片、图表、分析查询（热度、情绪、分布、趋势等）。

## 5. 工作流与前端

### 5.1 n8n 工作流（`etl_workflow/`）

| 文件 | 功能 |
| --- | --- |
| `Tech_Intelligence.json` | 主采集与处理工作流（抓取、分析、入库、向量化） |
| `System_Alert_Service.json` | 异常捕获与告警工作流 |
| `Daily_Tech_Brief.json` | 日报推送工作流 |

### 5.2 前端（`frontend/`）

| 文件 | 功能 |
| --- | --- |
| `index.html` / `style.css` / `app.js` | 主交互页面 |
| `subscribe.html` / `subscribe.css` / `subscribe.js` | 订阅日报页面与交互逻辑 |

## 6. 文档与素材

| 路径 | 功能 |
| --- | --- |
| `docs/` | 项目维护文档（含测试文档、来源接入、数据质量说明） |
| `assets/screenshots/` | 工作流与界面截图 |
| `assets/svg/` | 架构图、标题图 |
| `assets/previews/` | 项目展示图 |

## 7. 建议的维护约定

| 场景 | 推荐入口 |
| --- | --- |
| 新增来源 | `deployment/scripts/db/upsert_source.sh` |
| 批量迁移与刷新视图 | `deployment/scripts/db/apply_source_framework_migration.sh` |
| 数据质量巡检 | `deployment/scripts/db/run_data_quality_checks.sh` |
| Agent 稳定性评测 | `python agents/eval/run_eval.py --suite default` |

## 8. 非业务产物说明

- `__pycache__/`、`*.pyc`、临时测试缓存目录不属于业务文件结构。
- 建议保持这类目录不纳入版本管理，避免影响结构可读性与 `git status` 输出。


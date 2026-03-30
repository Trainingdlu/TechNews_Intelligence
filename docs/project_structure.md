# TechNews 项目结构（2026-03 ReAct 架构）

本文档描述当前仓库的目录职责。已删除的旧模块（如 `agents/core/router.py`、`agents/core/pipelines.py`）不再属于运行路径。

## 1. 仓库根目录

| 路径 | 作用 |
| --- | --- |
| `.github/workflows/` | CI 与手动评测工作流 |
| `agents/` | Agent 服务端代码（核心运行时、工具、评测、单测） |
| `deployment/` | Docker Compose、数据库脚本与部署配置 |
| `docs/` | 项目文档与测试说明 |
| `etl_workflow/` | n8n 工作流 JSON |
| `frontend/` | 前端页面代码 |
| `sql/` | 数据库结构与分析 SQL |
| `README.md` | 项目总览 |
| `Dockerfile` | API/Bot 镜像构建入口 |

## 2. Agent 模块（`agents/`）

### 2.1 运行时入口

| 文件 | 作用 |
| --- | --- |
| `agents/agent.py` | ReAct 主运行时（工具调用、后处理、引用装饰、异常兜底） |
| `agents/prompts.py` | ReAct 系统指令与输出约束 |
| `agents/tools.py` | 数据检索与分析工具（query/timeline/landscape 等） |
| `agents/api.py` | FastAPI 服务入口 |
| `agents/bot.py` | Telegram Bot 入口 |
| `agents/cli.py` | 本地 CLI 调试入口 |
| `agents/db.py` | PostgreSQL 连接池 |
| `agents/mail.py` | 邮件通知模块 |
| `agents/__init__.py` | 包导出入口 |

### 2.2 Core 公共模块（`agents/core/`）

| 文件 | 作用 |
| --- | --- |
| `agents/core/evidence.py` | URL 提取、引用归一化、来源段落拼装 |
| `agents/core/metrics.py` | ReAct 指标统计（`react_*`） |
| `agents/core/__init__.py` | Core 导出入口 |

## 3. 测试与评测

| 路径 | 作用 |
| --- | --- |
| `agents/tests/unit/` | 单元测试（pytest） |
| `agents/eval/run_eval.py` | 批量评测入口 |
| `agents/eval/eval_core.py` | 评测指标计算与质量门禁 |
| `agents/eval/datasets/` | 评测题库 |
| `agents/eval/reports/` | 评测输出目录（运行产物） |

## 4. 部署（`deployment/`）

| 文件 | 作用 |
| --- | --- |
| `deployment/docker-compose.yml` | 服务编排 |
| `deployment/.env.example` | 部署环境变量模板 |
| `deployment/scripts/db/*.sh` | 数据库迁移/写入/质检脚本 |
| `deployment/data/*.sql` | 初始 SQL 数据文件 |

## 5. 文档（`docs/`）

| 路径 | 作用 |
| --- | --- |
| `docs/testing/` | 测试与评测文档 |
| `docs/data_quality_checks.md` | 数据质量检查说明 |
| `docs/minimal_source_refactor_checklist.md` | 来源框架改造清单 |
| `docs/project_structure.md` | 当前文档 |

## 6. 维护约定

1. 运行时默认只有 ReAct 主链路，不再维护旧路由/旧管线说明。
2. 指标命名以 `react_*` 为准，评测门禁与 CI 必须和该命名对齐。
3. 测试命令以 `pytest agents/tests -v` 为标准入口。

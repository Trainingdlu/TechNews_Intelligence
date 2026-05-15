# ADR-0001：采用模块化单体与 Clean/Hexagonal 边界

**状态：** Proposed
**日期：** 2026-05-14
**决策人：** 项目维护者

## 背景

TechNews Intelligence 已经超过简单 Demo 阶段。目前系统包含：

- n8n 采集工作流。
- PostgreSQL 与 pgvector 存储。
- FastAPI、Telegram Bot、CLI 和静态前端入口。
- 基于 LangGraph 的 Agent Runtime。
- 带证据 envelope 的结构化工具运行时。
- 持久化 Trace 存储和 Vue Trace Console。
- 离线评测管线。
- 部署脚本和 CI。

系统价值来自这些组件之间的协作。当前主要风险不是功能不够，而是边界逐渐变得不清晰：

- `app/api.py` 混合了 HTTP 入口、security helper、额度逻辑、订阅流程、聊天编排、SSE 格式化和持久化协调。
- `agent/graph/nodes.py` 把所有 graph node 行为集中在一个大模块中。
- `eval/build_task_dataset.py` 把复杂离线管线集中在一个大模块中。
- 缺少自动化架构护栏，无法防止 runtime 代码反向依赖交互入口。

项目需要更清晰的工程结构，但完整重写或拆分微服务在当前阶段会带来比收益更高的风险。

## 决策

采用**模块化单体**，并引入 **Clean Architecture / Hexagonal Architecture** 风格的边界规则：

- 当前保持一个代码库和一组部署服务。
- 将 `app/` 视为交互入口适配层。
- 将 `agent/` 视为运行时核心。
- 将 `services/` 视为基础设施适配层。
- 将 `eval/` 视为离线评测应用边界。
- 先增加架构边界测试，再做大规模重构。
- 所有重构以小步、行为保持、测试验证为原则。

短期目标不是追求教科书式完美架构，而是让新功能有明确归属、依赖方向可理解、大文件可以逐步拆分且不改变行为。

## 备选方案

### 方案 A：保持当前结构

| 维度 | 评估 |
| --- | --- |
| 复杂度 | 短期低 |
| 成本 | 短期低 |
| 可扩展性 | 有限 |
| 团队熟悉度 | 高 |
| 风险 | 边界继续漂移 |

**优点：**

- 没有迁移成本。
- 没有重构引入回归的风险。
- 现有测试继续保持稳定。

**缺点：**

- 大文件继续膨胀。
- 新贡献者缺少明确放置规则。
- 交互入口逻辑和应用用例逻辑继续混杂。
- 未来功能开发会越来越慢、越来越容易误伤。

### 方案 B：模块化单体 + Clean/Hexagonal 边界

| 维度 | 评估 |
| --- | --- |
| 复杂度 | 中 |
| 成本 | 中，可渐进 |
| 可扩展性 | 适合当前阶段 |
| 团队熟悉度 | 中 |
| 风险 | 可通过测试控制 |

**优点：**

- 符合当前部署模型。
- 可渐进抽离，不需要重写。
- 依赖方向更清晰。
- API、Bot、CLI 和 eval 可以更干净地复用同一套 runtime。
- 可以通过架构测试长期约束边界。

**缺点：**

- 需要持续纪律。
- 迁移期间会存在一些兼容 wrapper。
- 本身不解决服务级独立扩缩容问题。

### 方案 C：拆成微服务

| 维度 | 评估 |
| --- | --- |
| 复杂度 | 高 |
| 成本 | 高 |
| 可扩展性 | 潜在较高 |
| 团队熟悉度 | 较低 |
| 风险 | 当前阶段高 |

**优点：**

- 如果做得好，运行时边界会非常清楚。
- 可以独立扩缩容和部署。

**缺点：**

- 对当前阶段过早。
- 会引入服务间认证、网络边界、schema 所有权、可观测性和部署复杂度。
- 当前核心问题是模块边界，不是服务规模。

### 方案 D：围绕新框架重写

| 维度 | 评估 |
| --- | --- |
| 复杂度 | 极高 |
| 成本 | 极高 |
| 可扩展性 | 未知 |
| 团队熟悉度 | 重写期间低 |
| 风险 | 极高 |

**优点：**

- 理论上可以得到一套干净设计。

**缺点：**

- 会丢弃已经工作的行为。
- 很难完整保留 Agent、Trace、eval 路径中的边界场景。
- 测试和部署脚本中沉淀的生产知识容易丢失。

## 权衡分析

方案 B 的收益和风险最平衡：

- 它能提升工程清晰度，同时不破坏现有可运行行为。
- 它符合当前产品阶段：一个代码库、多个交互入口、共享 Agent Runtime。
- 它允许通过测试约束架构规则。
- 如果未来确实需要微服务拆分，它会先把单体内部边界整理清楚，为后续拆分降低成本。

只有当 ingestion、runtime serving、Trace 或 eval 出现明确的独立部署/扩缩容压力时，才应重新评估微服务拆分。

## 影响

会变容易的事情：

- 新增 API 路由时不继续扩大 `app/api.py`。
- Web、Telegram、CLI、eval 复用同一套聊天编排。
- 不依赖 HTTP 或 Telegram 就能测试应用流程。
- 按架构区域审查代码变更。
- 通过架构测试防止反向依赖。

会变困难的事情：

- 新改动需要遵守放置规则。
- 重构必须更小步、更依赖测试。
- 大文件拆分期间需要保留兼容 wrapper。

后续需要重新评估的点：

- `services/thread_memory.py` 应属于应用服务，还是继续保持基础设施邻近模块。
- 工具 handler 是否继续直接使用 `services.db`，还是逐步引入 repository-style port。
- runtime 清理完成后，eval 管线是否需要更明确的包边界。

## 行动项

1. [ ] 增加 `docs/PROJECT_ARCHITECTURE_BLUEPRINT.md`。
2. [ ] 增加本 ADR。
3. [ ] 增加 `tests/unit/test_architecture_boundaries.py`。
4. [ ] 将 `app/api.py` 中的 DTO 抽到 `app/schemas.py`。
5. [ ] 将审批签名和限流抽到 app 级 helper 模块。
6. [ ] 将 SSE progress/event 格式化抽到 `app/streaming.py`。
7. [ ] 将 chat/subscription/access-token 编排抽到 `app/use_cases/`。
8. [ ] 按节点族拆分 `agent/graph/nodes.py`。
9. [ ] runtime 边界稳定后，拆分 `eval/build_task_dataset.py`。

## 架构规则

初始规则：

- `agent/**` 不允许 import `app.api`、`app.bot`、`fastapi`、`telegram`。
- `services/**` 不允许 import `app.api` 或 `app.bot`。
- `app/**` 可以 import `agent` 和 `services`，但编排逻辑应逐步迁移到 use-case 模块。
- `eval/**` 可以依赖 runtime 代码，但 runtime 代码不应依赖 `eval`。
- 新工具必须通过 `agent/core/tool_catalog.py` 注册，并返回 `ToolEnvelope`。

## 验证

runtime 重构前的基线命令：

- `pytest tests/unit -q`
- `python eval/encoding_guard.py --root . --strict --report eval/reports/encoding_guard/local.json`
- 在 `trace_dashboard/` 中运行 `npm run build`


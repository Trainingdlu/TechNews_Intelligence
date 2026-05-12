# 系统架构

TechNews Intelligence 由采集链路、存储层、Agent 运行时、交互入口、Trace 观测和离线评测组成。实时问答、Telegram Bot、本地 CLI 和评测脚本共用同一套 Agent 运行时。

## 总览

```mermaid
flowchart LR
    subgraph ingest["数据采集"]
        n8n["n8n 工作流"]
        reader["Jina Reader"]
        etl_llm["摘要/分类/情绪模型"]
        embed["Jina Embeddings"]
    end

    subgraph storage["PostgreSQL"]
        news["tech_news"]
        vectors["news_embeddings"]
        conv["conversation_threads/messages"]
        memory["thread_memory_summaries/thread_evidence_index"]
        trace["agent_runs/agent_trace_spans/agent_model_io"]
    end

    subgraph runtime["Agent Runtime"]
        graph["LangGraph StateGraph"]
        tools["ToolRuntime + ToolRegistry"]
        guard["Tool Policy / Output Guard"]
    end

    subgraph entry["入口"]
        api["FastAPI"]
        bot["Telegram Bot"]
        cli["CLI"]
        trace_ui["Trace Console"]
        metabase["Metabase"]
    end

    n8n --> reader --> etl_llm --> news
    n8n --> embed --> vectors
    api --> graph
    bot --> graph
    cli --> graph
    graph --> tools --> storage
    graph --> guard
    graph --> trace
    graph --> conv
    graph --> memory
    trace_ui --> trace
    metabase --> news
```

## 数据采集链路

n8n 工作流负责新闻采集、正文提取、结构化处理、向量生成和通知。采集链路写入 PostgreSQL，实时 Agent 只读取数据库和工具返回结果，不直接依赖 n8n 运行状态。

主要数据表：

| 领域 | 表 |
| --- | --- |
| 新闻内容 | `tech_news`, `news_fulltext`, `news_embeddings` |
| 来源与实体 | `source_registry`, `entity_registry`, `entity_alias`, `entity_alias_candidate`, `news_entity_mentions` |
| 对话 | `conversation_threads`, `conversation_messages`, `api_tokens` |
| 记忆 | `thread_memory_summaries`, `thread_evidence_index` |
| Trace | `agent_runs`, `agent_trace_spans`, `agent_model_io` |

数据库启用 `vector` 和 `pg_trgm` 扩展。结构由 `sql/infrastructure/schema/schema_ddl.sql` 管理，Metabase 和分析 SQL 使用 `view_dashboard_news` 作为统一视图。

## Agent Graph

Agent 使用 LangGraph StateGraph 组织执行链路。节点名称在 Trace Console 中以中文展示，底层状态由 `AgentGraphState` 传递。

```mermaid
flowchart TD
    start([START]) --> prepare["准备上下文"]
    prepare --> intent["判断问题类型"]
    intent --> selection["选择工具"]
    selection --> worker["规划工具调用"]
    worker --> policy["工具策略检查"]
    policy --> executor["执行工具"]
    executor --> normalizer["归一化证据"]
    normalizer --> decider["判断是否继续调用工具"]
    decider -->|继续| worker
    decider -->|完成| final["最终综合"]
    intent -->|需要澄清| clarify["澄清回复"]
    intent -->|简单回答| final
    final --> guard["输出清理"]
    guard --> end([END])
    clarify --> end
```

核心节点：

| 节点 | 职责 |
| --- | --- |
| 准备上下文 | 构建历史索引、读取线程记忆、生成 Context Pack。 |
| 判断问题类型 | 判断简单问答、需要澄清、需要调用工具等执行路径。 |
| 选择工具 | 根据问题和上下文选择候选工具集合。 |
| 规划工具调用 | 生成具体工具调用计划和参数。 |
| 工具策略检查 | 检查工具名、数量、重复调用、参数范围和 URL 上下文。 |
| 执行工具 | 通过 ToolRuntime 调用工具并返回 ToolEnvelope。 |
| 归一化证据 | 合并工具证据、提取 URL、统计有效证据。 |
| 最终综合 | 基于工具结果和证据生成最终回答。 |
| 输出清理 | 仅保留证据内 URL，清理非证据链接。 |

## 上下文记忆

系统使用 Context Pack 和 Thread Memory 处理多轮对话。

Context Pack 在请求内生成，包含：

- 当前用户问题。
- 上下文整理后的独立问题。
- 与当前问题相关的历史轮次。
- 被选中的历史证据 URL。
- 线程级摘要和历史证据索引。
- 裁剪策略、选中数量和整理模型置信度。

Thread Memory 在回答持久化后异步更新，存储：

- `thread_memory_summaries`：线程摘要、主题、实体、已确认事实、待澄清问题。
- `thread_evidence_index`：历史回答中出现过的证据 URL、标题、摘要和来源序号。

Context Curator 是可选的上下文整理模型，默认开启。它只能从历史索引和线程证据索引中选择已有轮次和 URL，不能向主 Agent 注入伪造证据。确定性回退会保留最近对话和已有证据。

## 工具体系

工具定义、校验、执行和结果格式由 ToolRuntime 体系统一管理。

| 组件 | 职责 |
| --- | --- |
| `ToolCatalog` | 定义工具元数据、描述和输入契约。 |
| `ToolRegistry` | 绑定工具名、Pydantic schema 和处理器。 |
| `ToolRuntime` | 执行工具、调用 hooks、生成 tool_call span。 |
| `ToolRuntimeHooks` | 记录执行前后指标和诊断信息。 |
| `ToolEnvelope` | 统一工具返回结构。 |

`ToolEnvelope` 结构包含：

- `status`
- `data`
- `evidence`
- `diagnostics`
- `error_code`
- `error_message`

工具参数校验分两层：

- 图内策略检查负责模型规划结果的业务拦截，例如候选工具限制、重复调用、URL 是否来自上下文。
- ToolRegistry 使用 Pydantic schema 执行硬校验，保证进入工具处理器的参数结构正确。

## Trace 与观测

项目内 Trace 是主观测链路，独立写入 PostgreSQL。LangSmith 是可选外部观测扩展，不是自研 Trace 的依赖。

| 表 | 内容 |
| --- | --- |
| `agent_runs` | 请求级摘要：状态、耗时、用户问题、工具链、证据数量、token usage、运行时元数据。 |
| `agent_trace_spans` | 执行链路：流程节点、模型调用、工具执行、策略检查、后处理、上下文整理。 |
| `agent_model_io` | 完整模型输入 messages、原始输出、解析结果和 token usage。 |

span 类型：

| 类型 | 含义 |
| --- | --- |
| `graph_node` | LangGraph 流程节点。 |
| `context` | 上下文索引和 Context Pack 生成。 |
| `model_call` | 模型调用。 |
| `tool_call` | 工具执行。 |
| `guard` | 策略检查或澄清触发。 |
| `postprocess` | 证据归一和输出清理。 |

Trace Console 读取这些表并展示请求列表、调用链、节点详情、完整模型输入输出、工具返回、错误信息和 raw JSON。

## 交互入口

| 入口 | 实现 | 说明 |
| --- | --- | --- |
| Web/API | `app/api.py` | Token 校验、限流、额度、对话线程、流式事件。 |
| Trace Console | `app/trace_api.py` + `trace_dashboard/` | 管理员 token 访问链路追踪面板。 |
| Telegram Bot | `app/bot.py` | Telegram 对话入口，按 chat_id 维护会话。 |
| CLI | `app/cli.py` | 本地命令行入口。 |
| MCP | `agent/mcp/stdio_server.py` | MCP stdio 服务边界。 |

## LangSmith

LangSmith 通过 `LANGSMITH_*` 和 `LANGCHAIN_*` 环境变量启用。系统在 `agent_runs.trace_payload.runtime.langsmith` 中记录 LangSmith 状态、project 和 endpoint。自研 Trace 不依赖 LangSmith run id；后续可在可获取 run id 时写入 runtime 元数据。

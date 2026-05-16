# 评测体系

评测体系用于验证 Agent 的问题理解、工具路径、检索证据、最终回答和系统稳定性。评测脚本复用实时 Agent 运行链路，并通过 Trace 表回溯模型规划、工具执行和输出守卫行为。

## 目标与原则

评测不只看最终回答文本，而是同时检查 Agent 是否走了正确路径。

| 目标 | 说明 |
| --- | --- |
| 路径正确 | 问题类型、工具选择、工具顺序和澄清触发符合预期。 |
| 证据可靠 | 回答引用的 URL 来自工具证据，且满足来源、数量和时效要求。 |
| 生成可信 | 最终回答完整、结构清晰，并区分事实、推断和不确定性。 |
| 行为稳定 | 多次运行结果不应出现工具路径大幅漂移或无证据回答。 |
| 可回溯 | 每个失败 case 都能通过 Trace 定位到意图、工具、检索、生成或系统层。 |

设计原则：

- 评测复用线上 Agent Runtime，不维护一套独立 mock 链路。
- 评分优先使用结构化 Trace 和工具输出，再使用 LLM-as-a-Judge 判断开放文本质量。
- 完整 prompt 和 raw output 不进入公开报告，只通过 Trace 查询脚本按需读取。
- 矩阵实验以可比较为目标，每组实验必须记录配置、输出目录和 manifest。

## 评测组成

| 模块 | 说明 |
| --- | --- |
| `eval/build_task_dataset.py` | 构建任务评测数据集。 |
| `eval/task_eval_schema.py` | 校验任务类型、case 结构和字段契约。 |
| `eval/run_task_eval.py` | 执行单组任务评测。 |
| `eval/run_matrix_eval.py` | 执行多组实验矩阵。 |
| `eval/task_eval_scoring.py` | 分层评分和归因。 |
| `eval/build_task_eval_leaderboard.py` | 生成矩阵排行榜。 |
| `eval/build_report.py` | 生成最终报告。 |
| `eval/trace_query.py` | 查询 Trace span 和完整模型 I/O。 |
| `deployment/scripts/eval/run_eval.sh` | 一键评测流水线。 |

## 数据集

任务数据集使用 JSONL。每条 case 描述用户问题、任务类型、期望工具路径、证据要求、引用要求和评分条件。任务配置位于 `eval/config/`。

常用文件：

| 文件 | 说明 |
| --- | --- |
| `eval/config/task_types_smoke.json` | 小规模冒烟配置，用来快速检查采样、模型 JSON 输出和证据校验链路。 |
| `eval/config/task_types_preflight.json` | 中等规模预检配置，用来在正式生成前验证题型覆盖和生成稳定性。 |
| `eval/config/task_types_retrieval.json` | 正式检索/分析题集生成配置，正式报告和简历数据只认该配置或人工审核后的派生数据集。 |
| `eval/config/matrix.json` | 矩阵实验配置。 |
| `eval/datasets/*.jsonl` | 生成的数据集。 |
| `eval/reports/*` | 评测报告输出。 |

构建数据集：

```bash
PYTHONPATH=. python eval/build_task_dataset.py --help
```

构建题集默认使用 DeepSeek（`deepseek-v4-pro`），可通过 `TASK_EVAL_PROVIDER` 和 `TASK_EVAL_MODEL` 覆盖。正式生成建议先跑 `task_types_smoke.json`，再跑 `task_types_preflight.json`，最后用 `task_types_retrieval.json` 生成正式题集。

smoke / preflight 是抽样预检配置，不要求覆盖所有工具和所有场景，运行时应带上 `--no-enforce-coverage-policy`。正式配置 `task_types_retrieval.json` 才启用完整覆盖校验。所有配置都可以开启 `--enforce-scenario-retrieval-map`，确保 `normal/boundary` 被当作可检索题，`conflict/empty` 被当作澄清或非检索题。

`should_clarify=true` 的冲突、空结果类场景不再要求工具路径命中：生成时 `expected_tool_paths=[]`、`required_tools=[]`、检索 gold 为空，评分时重点检查是否正确触发澄清，避免把澄清题和检索题混在同一套工具指标里。

单条 case 的核心字段：

| 字段 | 说明 |
| --- | --- |
| `case_id` | 稳定 ID，用于报告、Trace 查询和回归对比。 |
| `question` | 用户问题，直接送入 Agent Runtime。 |
| `task_type` | 任务类型，用于聚合分数和覆盖率。 |
| `expected_tools` | 期望出现的工具集合或关键路径。 |
| `evidence_requirements` | 对证据数量、来源、URL 或时间范围的要求。 |
| `citation_requirements` | 对内联引用和来源列表的要求。 |
| `scoring` | case 级评分条件和权重。 |

任务类型覆盖：

| 类型 | 目标能力 |
| --- | --- |
| 近况简报 | 检验 `query_news` / `search_news` 的近期召回和摘要能力。 |
| 深度解读 | 检验检索、全文读取、证据归一和最终综合。 |
| 主题对比 | 检验 `compare_topics` 和跨实体证据覆盖。 |
| 来源对比 | 检验 `compare_sources` 的来源覆盖和情绪差异计算。 |
| 时间线 | 检验 `build_timeline` 的窗口扩展和事件排序。 |
| 竞争格局 | 检验 `analyze_landscape` 的实体聚合和信号分类。 |
| 澄清场景 | 检验证据不足、范围模糊和 URL 越权时是否触发澄清。 |

## 单组评测

```bash
python eval/run_task_eval.py \
  --dataset eval/datasets/task_eval.jsonl \
  --output eval/reports/task_eval.json \
  --include-trace-summary
```

常用参数：

| 参数 | 说明 |
| --- | --- |
| `--runs-per-case` | 每个 case 重复次数。 |
| `--sleep-seconds` | 重复运行间隔。 |
| `--max-cases` | 限制评测数量。 |
| `--strict-tool-check` | 用实时工具目录校验工具名。 |
| `--include-trace-summary` | 在结果中包含 trace 摘要。 |
| `--enable-llm-judge` | 启用 LLM-as-a-Judge。 |

## 矩阵评测

```bash
python eval/run_matrix_eval.py \
  --matrix eval/config/matrix.json \
  --groups G0,G1,G2 \
  --output-dir eval/reports/matrix \
  -- --include-trace-summary
```

矩阵评测用于比较不同模型、检索参数、rerank 配置或 Agent 参数。每组实验输出独立 JSON 报告和 manifest。

矩阵配置通常比较：

| 维度 | 示例 |
| --- | --- |
| 模型角色 | intent/tool/final 使用不同 provider 或模型。 |
| 检索策略 | 召回窗口、候选数量、关键词权重、语义权重。 |
| Rerank | `none` vs `llm_rerank`，或不同 rerank 超时策略。 |
| 工具轮次 | `AGENT_GRAPH_MAX_TOOL_ROUNDS` 对深度问题的影响。 |
| 上下文策略 | Context Pack 裁剪、Thread Memory 开关和历史证据使用。 |

每个矩阵组需要保留：

- 实验配置。
- 数据集版本或 fingerprint。
- 运行时间和 run id。
- 单 case 结果。
- 聚合指标。
- Trace 摘要或可查询 request id。

## 一键评测

```bash
bash deployment/scripts/eval/run_eval.sh
```

该脚本执行数据集构建、矩阵评测、排行榜生成、Judge 审计和最终报告输出。主要环境变量：

| 变量 | 默认值 |
| --- | --- |
| `TASK_FILE` | `eval/config/task_types_retrieval.json` |
| `MATRIX_FILE` | `eval/config/matrix.json` |
| `GROUPS` | `G0,G1,G2` |
| `RUNS_PER_CASE` | `1` |
| `ENABLE_JUDGE` | `on` |
| `RUN_ID` | UTC 时间戳生成 |

输出目录：

```text
eval/reports/<RUN_ID>/
```

## 分层评分

评测结果按层拆分：

| 层级 | 内容 |
| --- | --- |
| 意图层 | 问题类型、是否需要工具、澄清触发。 |
| 工具层 | 工具选择、工具顺序、参数合法性、策略拦截。 |
| 检索层 | 召回 URL、证据数量、来源覆盖、空结果原因。 |
| 生成层 | 答案完整性、引用、事实一致性、输出守卫清理。 |
| 系统层 | latency、错误码、异常链、token usage。 |

Trace 数据优先来自 `agent_trace_spans` 和 `agent_model_io`。评测结果不直接暴露完整 prompt/raw output，完整内容通过 Trace 查询脚本读取。

失败归因：

| 归因 | 典型表现 | 优先查看 |
| --- | --- | --- |
| 意图错误 | 简单问答误走工具、需要工具却直接回答、该澄清未澄清。 | `intent_router` span |
| 工具规划错误 | 漏选关键工具、参数为空、重复调用、工具顺序异常。 | `tool_selection`, `tool_worker`, `tool_policy` span |
| 检索不足 | 空结果、证据数量不足、来源覆盖不完整、时间窗口不合理。 | `tool_call` span 和 ToolEnvelope diagnostics |
| 生成不稳 | 结论缺失、引用不完整、事实与证据不一致。 | `final_synthesizer` model I/O |
| 守卫问题 | 非证据 URL 泄漏、引用编号错乱、来源列表缺失。 | `output_guard`, `postprocess` span |
| 系统异常 | 超时、数据库异常、模型调用失败、token usage 异常。 | `agent_runs`, error chain |

质量门禁可以按场景设置，不建议所有任务共用单一阈值。检索类 case 更看重证据覆盖和引用合法性；深度分析 case 更看重工具路径、全文读取和最终综合质量；澄清类 case 更看重是否及时停止并返回可执行的澄清问题。

## Trace 查询

按请求查看 span tree：

```bash
python eval/trace_query.py --request-id <request_id>
```

查看某个模型调用的完整输入输出：

```bash
python eval/trace_query.py --model-span-id <span_id> --show-model-io
```

按 case id 查找失败链路：

```bash
python eval/trace_query.py --case-id <case_id>
```

排查建议：

1. 先看 `agent_runs` 中的 final status、latency、tools_used 和 evidence_count。
2. 再看 `agent_trace_spans` 的节点顺序，确认失败发生在 intent、tool、guard、postprocess 还是 model_call。
3. 需要模型输入输出时，再用 `--model-span-id <span_id> --show-model-io` 精查。
4. 对检索失败优先查看 ToolEnvelope diagnostics，而不是只看最终回答。

## 报告

生成排行榜：

```bash
python eval/build_task_eval_leaderboard.py \
  --matrix-dir eval/reports/matrix \
  --output-json eval/reports/leaderboard/latest.json \
  --output-md eval/reports/leaderboard/latest.md
```

生成最终报告：

```bash
python eval/build_report.py \
  --main-leaderboard-json eval/reports/leaderboard/latest.json \
  --run-id <RUN_ID> \
  --dataset-version <DATASET_VERSION> \
  --output-md eval/reports/<RUN_ID>/final_report.md
```

报告应包含：

- 数据集版本和运行配置。
- 总分与分层得分。
- 各任务类型得分。
- 矩阵组对比和排名。
- 失败 case 样例。
- 主要回归点和改进建议。

报告不应直接粘贴完整 prompt、raw model output 或敏感环境变量。需要复现细节时保留 request id、case id 和 trace 查询命令。

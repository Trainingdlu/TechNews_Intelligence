# 评测体系

评测体系用于验证 Agent 的问题理解、工具路径、检索证据、最终回答和系统稳定性。评测脚本复用实时 Agent 运行链路，并通过 Trace 表回溯模型规划、工具执行和输出守卫行为。

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
| `eval/config/tasks_180.json` | 任务配置。 |
| `eval/config/matrix.json` | 矩阵实验配置。 |
| `eval/datasets/*.jsonl` | 生成的数据集。 |
| `eval/reports/*` | 评测报告输出。 |

构建数据集：

```bash
PYTHONPATH=. python eval/build_task_dataset.py --help
```

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

## 一键评测

```bash
bash deployment/scripts/eval/run_eval.sh
```

该脚本执行数据集构建、矩阵评测、排行榜生成、Judge 审计和最终报告输出。主要环境变量：

| 变量 | 默认值 |
| --- | --- |
| `TASK_FILE` | `eval/config/tasks_180.json` |
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

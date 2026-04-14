# 评测说明

评测相关文档已迁移至 `docs/testing/` 目录：

- [测试文档索引](../docs/testing/README.md)
- [智能体评测结构说明](../docs/testing/agent-eval-structure.md)
- [智能体测试与评测指南](../docs/testing/agent-testing-guide.md)

`eval/reports/` 下的运行报告为生成产物，不纳入版本跟踪。

## 数据集字段

每条 JSONL 用例支持以下字段：

- `id`、`category`、`capability`、`question`
- `min_urls`、`must_contain`
- `expected_facts`（可选）：用于 `fact_hit_rate` 的短语列表
- `expected_fact_groups`（可选）：用于 `fact_group_hit_rate` 的分组短语别名
- `required_tools`（可选）：用于 `tool_path_hit_rate` 的期望工具列表
- `acceptable_tool_paths`（可选）：用于 `tool_path_accept_hit_rate` 的可接受工具组合
- `must_not_contain`（可选）：用于 `forbidden_claim_rate` 的禁用短语列表
- `expected_source_domains`（可选）：用于 `source_domain_hit_rate` 的期望引用域名
- `ground_truth`（可选）：Ragas 参考答案（优先）
- `ragas_contexts`（可选）：Ragas 备用上下文（当 trace 中无可用 context docs 时使用）
- `tags`、`enabled`

若未提供 `required_tools`，加载器会根据能力字段使用默认映射（例如：`compare_topics -> compare_topics`、`timeline -> build_timeline`）。

## 实验矩阵（6组）

为拆分检索召回、重排和 Agent 本体收益，评测矩阵配置在：

- `eval/experiment_matrix.json`

默认分组：

- `G0_baseline`：`R0 + A0`
- `G1_recall_only`：仅召回优化
- `G2_rerank_only`：仅重排优化
- `G3_retrieval_full`：召回+重排优化
- `G4_agent_only`：仅 Agent 优化
- `G5_full_optimized`：检索+Agent 全优化

可使用矩阵执行器批量运行（支持 dry-run）：

```bash
# 仅打印矩阵执行计划，不实际运行
python eval/run_matrix_eval.py --dry-run -- --suite smoke --runs-per-question 1

# 执行全部分组
python eval/run_matrix_eval.py -- --suite default --runs-per-question 3

# 仅执行部分分组
python eval/run_matrix_eval.py --groups G0_baseline,G3_retrieval_full,G4_agent_only -- --suite default --runs-per-question 3
```

矩阵执行器会为每组注入环境变量并在 `eval/reports/matrix/` 下生成：

- 分组报告：`<timestamp>_<group>.json`
- 执行清单：`<timestamp>_manifest.json`

每份 `run_eval` 报告新增 `experiment` 字段，记录：

- `group`：实验组 ID
- `env`：关键实验环境变量快照（如 `EVAL_RETRIEVAL_VARIANT`、`EVAL_AGENT_VARIANT`、`NEWS_RERANK_MODE`）

## 常用命令

```bash
# 冒烟评测
python eval/run_eval.py --suite smoke --runs-per-question 1 --output eval/reports/smoke.json

# 默认题集 + 质量门禁
python eval/run_eval.py \
  --suite default \
  --runs-per-question 3 \
  --experiment-group G0_baseline \
  --include-trace-summary \
  --include-outputs \
  --export-ragas-jsonl eval/reports/ragas/input_latest.jsonl \
  --fail-on-react-success-rate 0.90 \
  --fail-on-react-error-rate 0.10 \
  --fail-on-avg-min-url-hit-rate 0.85 \
  --fail-on-avg-fact-hit-rate 0.70 \
  --fail-on-avg-tool-path-hit-rate 0.70 \
  --fail-on-avg-fact-group-hit-rate 0.70 \
  --fail-on-avg-tool-path-accept-hit-rate 0.80 \
  --fail-on-avg-source-domain-hit-rate 0.60 \
  --fail-on-avg-forbidden-claim-rate 0.05 \
  --output eval/reports/latest.json
```

## LangSmith + Ragas

1) 运行评测时开启追踪并导出 Ragas 输入：

```bash
python eval/run_eval.py \
  --suite default \
  --runs-per-question 3 \
  --experiment-group G0_baseline \
  --include-trace-summary \
  --include-outputs \
  --export-ragas-jsonl eval/reports/ragas/input_latest.jsonl \
  --output eval/reports/latest.json
```

2) 计算 Ragas 指标并生成结果文件：

```bash
python eval/ragas_langsmith_eval.py \
  --rows-jsonl eval/reports/ragas/input_latest.jsonl \
  --output eval/reports/ragas/latest.json
```

3) 可选：上传到 LangSmith Dataset（便于对比与留档）：

```bash
python eval/ragas_langsmith_eval.py \
  --rows-jsonl eval/reports/ragas/input_latest.jsonl \
  --output eval/reports/ragas/latest.json \
  --upload-langsmith \
  --langsmith-dataset technews-ragas-default
```

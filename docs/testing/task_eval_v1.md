# Task-Driven Evaluation v1

## 1. Scope
- This pipeline is independent from legacy evaluation chains.
- Legacy files are kept in repository, but this v1 flow does not depend on them.
- Layered outputs are independent: `intent`, `tool`, `retrieval`, `analysis`, `system`.

## 2. Core Files
- `eval/config/task_types_v1.json`: task type definitions (31 seeded tasks, covering all current skills).
- `eval/task_eval_v1_schema.py`: task/case contracts and validators.
- `eval/build_task_dataset_v1.py`: task-driven sampling + LLM expected-case generation + audit.
- `eval/run_task_eval_v1.py`: real-agent execution + layered scoring + attribution.
- `eval/task_eval_v1_scoring.py`: layer metrics and root-cause code logic.

## 3. Task Type Contract
Each task type includes:
- `task_id`
- `skill`
- `intent_label`
- `retrieval_mode` (`evaluable`/`non_retrieval`)
- `scenario` (`normal`/`empty`/`boundary`/`conflict`)
- `example_question`
- `parameter_template`
- `acceptable_tool_paths` (set, not single route)
- `required_tools`
- `forbidden_tools`
- `should_clarify`
- `difficulty`
- `sampling`

Risk-based scenario coverage policy:
- High-risk skills: `normal + empty + conflict`
  - `query_news`, `search_news`, `build_timeline`, `fulltext_batch`
  - `compare_topics`, `compare_sources`, `trend_analysis`, `analyze_landscape`
- Medium-risk skills: `normal + empty`
  - `read_news_content`
- Low-risk skills: `normal`
  - `get_db_stats`, `list_topics`

This policy is enforced at task-type load time.

## 4. Case Contract
Each generated case includes:
- `case_id`
- `pool_id`
- `input_news_pool_hash`
- `task_type`
- `skill`
- `intent_label`
- `input_news_pool`
- `expected_question`
- `expected_answer`
- `expected_tool_paths`
- `required_tools`
- `forbidden_tools`
- `retrieval_gold_doc_ids`
- `retrieval_gold_urls`
- `verifiable_claims`
- `should_clarify`
- `retrieval_evaluable`
- `difficulty`
- `tags`

Key hard rule:
- If `retrieval_evaluable=true`, `retrieval_gold_doc_ids` must be non-empty and must come from the same `input_news_pool`.
- `expected_question` and `expected_answer` must be Chinese (entity/product names may remain in English).
- `expected_tool_paths` must be a strict subset of task-level `acceptable_tool_paths` (no arg drift).
- For non-`empty` scenarios, `expected_question` must be grounded in the same `input_news_pool`.

## 5. Build Dataset
Command:

```bash
python -m eval.build_task_dataset_v1 \
  --task-types eval/config/task_types_v1.json \
  --output eval/datasets/task_eval_v1_cases.jsonl \
  --manifest-output eval/datasets/task_eval_v1_manifest.json
```

Behavior:
- Sampling is task-driven (not global random).
- One LLM generation call per task type (with all pools of that task).
- Optional second-pass LLM audit is enabled by default.
- Invalid generation is rejected; rejected pools are regenerated once.

## 6. Run Real Chain + Score
Command:

```bash
python -m eval.run_task_eval_v1 \
  --dataset eval/datasets/task_eval_v1_cases.jsonl \
  --output eval/reports/task_eval_v1_latest.json \
  --runs-per-case 1
```

For stability probes:

```bash
python -m eval.run_task_eval_v1 \
  --dataset eval/datasets/task_eval_v1_cases.jsonl \
  --output eval/reports/task_eval_v1_stability.json \
  --runs-per-case 3
```

## 7. Layer Metrics
- Intent: `top1_accuracy`, `macro_f1`, `clarification_accuracy`
- Tool: `tool_set_precision`, `tool_set_recall`, `acceptable_path_hit_rate`, `forbidden_tool_rate`, `param_accuracy`
- Retrieval: `recall_at_5`, `recall_at_10`, `mrr_at_10`, `ndcg_at_10`, `gold_hit_rate`
- Analysis: `claim_support_rate`, `unsupported_claim_rate`, `contradiction_rate`, `numeric_consistency`
- System: `error_rate`, `timeout_rate`, `fallback_rate`, `latency_p50_ms`, `latency_p95_ms`

No mixed single release score is computed.

## 8. Attribution Codes
- `INTENT_FAIL`
- `TOOL_PATH_FAIL`
- `TOOL_ARG_FAIL`
- `RETRIEVAL_FAIL`
- `ANALYSIS_UNSUPPORTED`
- `ANALYSIS_CONTRADICT`
- `SYSTEM_FAIL`

Primary cause for each case is stored in case-level `attribution`.

## 9. Notes
- Gatekeeping thresholds are intentionally deferred in this v1 implementation.
- To enforce gates later, consume `report.summary.<layer>` directly with hard per-layer thresholds.
Path scoring rule:
- Tool-path evaluation uses `acceptable_tool_paths` set, not a single path.
- Runtime scoring selects the best matching path by priority:
  1) full path hit
  2) higher ordered coverage
  3) higher parameter accuracy
  4) shorter expected path

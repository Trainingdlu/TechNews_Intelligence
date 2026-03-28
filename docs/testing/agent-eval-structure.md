# 评测目录结构说明

`agents/eval` 目录按职责分层：

- `run_eval.py`：评测执行入口（筛选、运行、报告生成）
- `eval_core.py`：纯指标与门禁逻辑
- `capabilities.py`：能力注册表（与 `agent.py` 能力对齐）
- `dataset_loader.py`：题库解析、标准化、过滤
- `datasets/`：按能力标注的 JSONL 题库目录
  - `default.jsonl`：默认评测题库（唯一默认入口）
  - `smoke.jsonl`：快速冒烟题库
- `reports/`：评测输出报告目录

快速 smoke 评测：

```bash
python agents/eval/run_eval.py --suite smoke --runs-per-question 1 --output agents/eval/reports/smoke.json
```

仅运行强制路由能力：

```bash
python agents/eval/run_eval.py --suite default --capabilities compare_topics,timeline,landscape
```

Refresh latest baseline (recommended after route metric schema updates):

```bash
python agents/eval/run_eval.py --suite default --runs-per-question 3 --output agents/eval/reports/latest.json
```

Notes:
- `route_metrics_schema_version=2` now includes forced-route counters for `source_compare`, `trend`, `fulltext`, and `query`.
- Baselines generated with older schema should be regenerated before doing regression comparisons.

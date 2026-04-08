# 评测目录结构说明

`eval/` 目录职责如下：

- `run_eval.py`：评测入口（题集加载、执行、报告输出、门禁判断）
- `eval_core.py`：指标计算与 baseline/gate 逻辑
- `dataset_loader.py`：题集解析、过滤、能力字段校验
- `capabilities.py`：能力注册表
- `datasets/`：评测题集（`default.jsonl`、`smoke.jsonl`、`accuracy_snapshot.jsonl`）
- `reports/`：评测报告输出目录

常用命令：

```bash
# 快速冒烟评测
python eval/run_eval.py --suite smoke --runs-per-question 1 --output eval/reports/smoke.json

# 仅运行指定能力
python eval/run_eval.py --suite default --capabilities compare_topics,timeline,landscape

# 生成最新基线报告
python eval/run_eval.py --suite default --runs-per-question 3 --output eval/reports/latest.json
```

说明：

- 当前 `route_metrics_schema_version=3`
- 若 baseline 来自旧 schema，建议先重跑生成新 baseline 再做回归比较

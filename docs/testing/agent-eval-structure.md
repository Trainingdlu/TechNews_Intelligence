# 评测目录结构说明

`agents/eval` 目录职责如下：

- `run_eval.py`：评测入口（题库加载、执行、报告输出、门禁判定）
- `eval_core.py`：指标计算与 baseline/gate 逻辑
- `dataset_loader.py`：题库解析、过滤、能力字段校验
- `capabilities.py`：能力注册表
- `datasets/`：评测题库
  - `default.jsonl`：默认套件
  - `smoke.jsonl`：冒烟套件
- `reports/`：评测报告输出目录（运行产物）

常用命令：

```bash
# 快速冒烟
python agents/eval/run_eval.py --suite smoke --runs-per-question 1 --output agents/eval/reports/smoke.json

# 只跑特定能力
python agents/eval/run_eval.py --suite default --capabilities compare_topics,timeline,landscape

# 生成最新基线报告
python agents/eval/run_eval.py --suite default --runs-per-question 3 --output agents/eval/reports/latest.json
```

说明：

- 当前 `route_metrics_schema_version=3`。
- baseline 报告如来自旧 schema，建议重新生成后再做回归比较。

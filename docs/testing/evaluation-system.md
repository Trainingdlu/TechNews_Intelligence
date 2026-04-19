# 评测与测试体系

本文件是 `eval/` 与 `tests/` 的统一操作入口。

## 1. 体系目标

- 分层评测：检索层、生成层、系统层、Judge 层分开评估
- 可复现：数据集版本冻结，实验参数与脚本固定
- 可归因：低分样本可定位到检索、重排、生成、工具路径
- 可门禁：软/硬门禁与自动化阻断

## 2. 目录职责

- `eval/`：评测执行器、数据构建器、报告聚合器
- `tests/`：单元测试、测试夹具、测试产物目录
- `eval/datasets/versions/`：冻结版本题集
- `eval/reports/`：评测运行产物（默认不纳入版本控制）

## 3. 常用命令

### 3.1 单元测试

```bash
# 全量单测
pytest tests -v

# 评测体系关键单测
pytest tests/unit/test_run_full_eval_pipeline.py -v
pytest tests/unit/test_run_matrix_eval.py -v
pytest tests/unit/test_gatekeeper.py -v
pytest tests/unit/test_encoding_guard.py -v
```

### 3.2 评测执行

```bash
# 冒烟评测
python eval/run_eval.py --suite smoke --runs-per-question 1 --output eval/reports/smoke.json

# 矩阵 dry-run
python eval/run_matrix_eval.py --dry-run -- --suite smoke --runs-per-question 1

# 矩阵执行
python eval/run_matrix_eval.py -- --suite default --runs-per-question 3
```

### 3.3 全链路

```bash
python eval/run_full_eval_pipeline.py \
  --dataset-version latest \
  --run-id exp_v20260417_01 \
  --resume
```

执行顺序固定为：
1. 数据集检查与契约校验
2. 矩阵评测（`run_matrix_eval.py -> run_eval.py`）
3. Judge 评测（`run_judge_eval.py`）
4. Leaderboard 聚合（`build_leaderboard.py`）

## 4. 旧链路说明

- 历史生成层旧评测链路已从主流程移除，不再作为默认执行步骤。
- 当前链路统一基于 `run_eval + judge + leaderboard` 聚合结果。

## 5. 报告与产物

- 评测报告默认输出到 `eval/reports/`
- 全链路会在 `eval/reports/<run-id>/` 生成状态与汇总
- `eval/reports/`、`tests/reports/` 为运行产物目录，不作为源码

## 6. 临时目录治理（tmp）

- `pytest` 使用系统临时目录，并启用 `tmp_path_retention_policy = none`（见 `pytest.ini`）
- `tests/conftest.py` 在 `pytest_sessionfinish` 阶段会自动清理仓库内常见临时目录：
  - `.pytest_tmp_technews/`
  - `tests/pytest-tmp/`
  - `tests/pytest-cache-files-*/`
  - `tests/unit/.tmp_pytest/`
  - `eval/reports/tmp_pytest/`

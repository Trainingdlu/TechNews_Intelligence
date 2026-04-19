# Tests

This directory stores test code and test artifacts.

Useful commands:

```bash
# run all tests
pytest tests -v

# run task-eval v1 focused tests
pytest tests/unit/test_task_eval_v1_schema.py -v
pytest tests/unit/test_task_eval_v1_pathset_and_coverage.py -v
```

Task-eval v1 runtime commands:

```bash
python -m eval.build_task_dataset_v1 \
  --task-types eval/config/task_types_v1.json \
  --output eval/datasets/task_eval_v1_cases_smoke.jsonl \
  --manifest-output eval/datasets/task_eval_v1_manifest_smoke.json

python -m eval.run_task_eval_v1 \
  --dataset eval/datasets/task_eval_v1_cases_smoke.jsonl \
  --output eval/reports/task_eval_v1_smoke.json \
  --runs-per-case 1
```

Notes:

- `tests/reports/` (if present) is generated output.
- For current evaluation design and contracts, see `docs/testing/task_eval_v1.md`.

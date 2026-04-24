# Tests

This directory stores test code and test artifacts.

Useful commands:

```bash
# run all tests
pytest tests -v

# run task-eval focused tests
pytest tests/unit/test_task_eval_schema.py -v
pytest tests/unit/test_task_eval_pathset_and_coverage.py -v
```

Task-eval runtime commands:

```bash
python -m eval.build_task_dataset \
  --task-types eval/config/tasks_180.json \
  --output eval/datasets/task_eval_cases_smoke.jsonl \
  --manifest-output eval/datasets/task_eval_manifest_smoke.json

python -m eval.run_task_eval \
  --dataset eval/datasets/task_eval_cases_smoke.jsonl \
  --output eval/reports/task_eval_smoke.json \
  --runs-per-case 1
```

Notes:

- `tests/reports/` (if present) is generated output.
- For current evaluation design and contracts, see `docs/testing/task_eval.md`.


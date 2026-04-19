# Versioned Eval Datasets

Each frozen dataset build is stored under:

```text
eval/datasets/versions/vYYYYMMDD_HHMM/
  manifest.json
  smoke.jsonl
  regression.jsonl
  challenge.jsonl
```

## Build Command (Current)

```bash
python -m eval.build_task_dataset_v1 \
  --task-types eval/config/task_types_v1.json \
  --output eval/datasets/task_eval_v1_cases.jsonl \
  --manifest-output eval/datasets/task_eval_v1_manifest.json
```

Then freeze by copying the generated dataset into a new version folder:

```text
eval/datasets/versions/vYYYYMMDD_HHMM/regression.jsonl
```

## Reproducibility

- Use the same `--random-seed`.
- Keep source files unchanged (manifest records source `sha256`).
- Keep the same task-type config and build flags recorded in manifest.

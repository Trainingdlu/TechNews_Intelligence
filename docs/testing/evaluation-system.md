# Evaluation System

This repository now uses the task-driven evaluation chain only.

## Scope
- Build dataset by task types (with expected question/answer/tool-path and retrieval gold).
- Run real agent chain and score by independent layers.
- Aggregate matrix results into a leaderboard report.

## Main Entry Points
- `eval/build_task_dataset.py`
- `eval/run_task_eval.py`
- `eval/run_matrix_eval.py` (with `runner=task_eval`)
- `eval/build_task_eval_leaderboard.py`
- `deployment/scripts/eval/run_eval.sh` (one-click pipeline)

## One-Click Run
```bash
bash deployment/scripts/eval/run_eval.sh
```

Optional overrides:
```bash
RUN_ID=full_20260419T120000Z \
DATASET_VERSION=v_task_20260419_120000 \
GROUPS=G0,G1,G2 \
RUNS_PER_CASE=1 \
bash deployment/scripts/eval/run_eval.sh
```

## Outputs
- Matrix manifest: `eval/reports/<RUN_ID>/matrix/*_manifest.json`
- Leaderboard JSON: `eval/reports/<RUN_ID>/leaderboard/latest.json`
- Leaderboard Markdown: `eval/reports/<RUN_ID>/leaderboard/latest.md`

## Notes
- Old report-chain files (legacy judge/full-pipeline/ragas-linked aggregators) were removed.
- Historical files under `eval/reports/` may still contain old terms; they are run artifacts, not active logic.


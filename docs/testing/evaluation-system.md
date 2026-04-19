# Evaluation System (v1)

This repository now uses the task-driven v1 evaluation chain only.

## Scope
- Build dataset by task types (with expected question/answer/tool-path and retrieval gold).
- Run real agent chain and score by independent layers.
- Aggregate matrix results into a v1 leaderboard report.

## Main Entry Points
- `eval/build_task_dataset_v1.py`
- `eval/run_task_eval_v1.py`
- `eval/run_matrix_eval.py` (with `runner=task_eval_v1`)
- `eval/build_task_eval_v1_leaderboard.py`
- `deployment/scripts/eval/run_skill_matrix_pipeline.sh` (one-click pipeline)

## One-Click Run
```bash
bash deployment/scripts/eval/run_skill_matrix_pipeline.sh
```

Optional overrides:
```bash
RUN_ID=full_20260419T120000Z \
DATASET_VERSION=v_task_20260419_120000 \
GROUPS=G0_baseline,G5_full_optimized \
RUNS_PER_CASE=1 \
bash deployment/scripts/eval/run_skill_matrix_pipeline.sh
```

## Outputs
- Matrix manifest: `eval/reports/<RUN_ID>/matrix/*_manifest.json`
- Leaderboard JSON: `eval/reports/<RUN_ID>/leaderboard/latest.json`
- Leaderboard Markdown: `eval/reports/<RUN_ID>/leaderboard/latest.md`

## Notes
- Old report-chain files (legacy judge/full-pipeline/ragas-linked aggregators) were removed.
- Historical files under `eval/reports/` may still contain old terms; they are run artifacts, not active logic.

# Evaluation Directory

This directory now uses the task-driven evaluation pipeline as the primary path.

Main entrypoints:

- `eval/build_task_dataset.py`
- `eval/run_task_eval.py`
- `eval/build_entity_alias_candidates.py` for offline entity alias candidate extraction.
- `eval/promote_entity_alias_candidates.py` for promoting reviewed alias candidates.

Reference documentation:

- `docs/testing/task_eval.md`
- `docs/hybrid_retrieval_operations.md`

Notes:

- `eval/reports/` contains runtime artifacts.
- Frozen/legacy evaluation chains are not part of the active workflow.


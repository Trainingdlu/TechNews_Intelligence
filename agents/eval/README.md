# Eval Structure

`agents/eval` follows a layered structure similar to unit tests:

- `run_eval.py`: execution entrypoint (selection/filtering/report generation)
- `eval_core.py`: pure metric/gate logic
- `capabilities.py`: capability registry aligned with current `agent.py` abilities
- `dataset_loader.py`: dataset parsing/normalization/filtering
- `datasets/`: capability-tagged JSONL suites (`default.jsonl`, `smoke.jsonl`)
- `reports/`: generated eval outputs

Quick smoke run:

```bash
python agents/eval/run_eval.py --suite smoke --runs-per-question 1 --output agents/eval/reports/smoke.json
```

Run only forced-route capabilities:

```bash
python agents/eval/run_eval.py --suite default --capabilities compare_topics,timeline,landscape
```

# Eval Notes

Evaluation docs were moved to `docs/testing`:
- [Testing Index](../../docs/testing/README.md)
- [Agent Eval Structure](../../docs/testing/agent-eval-structure.md)
- [Agent Testing Guide](../../docs/testing/agent-testing-guide.md)

Runtime reports under `agents/eval/reports/` are generated artifacts and are not tracked.

## Dataset fields

Each JSONL case supports:
- `id`, `category`, `capability`, `question`
- `min_urls`, `must_contain`
- `expected_facts` (optional): phrase list used for `fact_hit_rate`
- `required_tools` (optional): expected tool list used for `tool_path_hit_rate`
- `must_not_contain` (optional): forbidden phrase list used for `forbidden_claim_rate`
- `tags`, `enabled`

If `required_tools` is omitted, loader uses a capability-based default map
(`compare_topics -> compare_topics`, `timeline -> build_timeline`, etc.).

## Quick commands

```bash
# smoke
python agents/eval/run_eval.py --suite smoke --runs-per-question 1 --output agents/eval/reports/smoke.json

# default + quality gates
python agents/eval/run_eval.py \
  --suite default \
  --runs-per-question 3 \
  --fail-on-react-success-rate 0.90 \
  --fail-on-react-error-rate 0.10 \
  --fail-on-avg-min-url-hit-rate 0.85 \
  --fail-on-avg-fact-hit-rate 0.70 \
  --fail-on-avg-tool-path-hit-rate 0.70 \
  --fail-on-avg-forbidden-claim-rate 0.05 \
  --output agents/eval/reports/latest.json
```

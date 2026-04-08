# Eval Notes

Evaluation docs were moved to `docs/testing`:
- [Testing Index](../docs/testing/README.md)
- [Agent Eval Structure](../docs/testing/agent-eval-structure.md)
- [Agent Testing Guide](../docs/testing/agent-testing-guide.md)

Runtime reports under `eval/reports/` are generated artifacts and are not tracked.

## Dataset fields

Each JSONL case supports:
- `id`, `category`, `capability`, `question`
- `min_urls`, `must_contain`
- `expected_facts` (optional): phrase list used for `fact_hit_rate`
- `expected_fact_groups` (optional): grouped phrase aliases used for `fact_group_hit_rate`
- `required_tools` (optional): expected tool list used for `tool_path_hit_rate`
- `acceptable_tool_paths` (optional): alternative acceptable tool combinations used for `tool_path_accept_hit_rate`
- `must_not_contain` (optional): forbidden phrase list used for `forbidden_claim_rate`
- `expected_source_domains` (optional): expected citation domains used for `source_domain_hit_rate`
- `tags`, `enabled`

If `required_tools` is omitted, loader uses a capability-based default map
(`compare_topics -> compare_topics`, `timeline -> build_timeline`, etc.).

## Quick commands

```bash
# smoke
python eval/run_eval.py --suite smoke --runs-per-question 1 --output eval/reports/smoke.json

# default + quality gates
python eval/run_eval.py \
  --suite default \
  --runs-per-question 3 \
  --fail-on-react-success-rate 0.90 \
  --fail-on-react-error-rate 0.10 \
  --fail-on-avg-min-url-hit-rate 0.85 \
  --fail-on-avg-fact-hit-rate 0.70 \
  --fail-on-avg-tool-path-hit-rate 0.70 \
  --fail-on-avg-fact-group-hit-rate 0.70 \
  --fail-on-avg-tool-path-accept-hit-rate 0.80 \
  --fail-on-avg-source-domain-hit-rate 0.60 \
  --fail-on-avg-forbidden-claim-rate 0.05 \
  --output eval/reports/latest.json
```

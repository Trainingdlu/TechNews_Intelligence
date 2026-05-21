# G2 Retrieval Eval Report

Queries scored: **50**

Relevance gold: LLM-judged 0/1/2 over the pooled top-10 of all 3 configs (shared labels).

## Ablation table

| Config | Profile | Hit@5 | Precision@5 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|
| G0 | base + no rerank | 100.0% | 72.0% | 0.918 | 0.742 |
| G1 | wide + no rerank | 100.0% | 71.6% | 0.918 | 0.740 |
| G2 | wide + Jina rerank | 100.0% | 79.2% | 0.990 | 0.877 |

## Delta vs G0 (baseline)

| Config | Hit@5 | Precision@5 | MRR@10 | nDCG@10 |
|---|---|---|---|---|
| G1 | +0.0pt | -0.4pt | +0.000 | -0.002 |
| G2 | +0.0pt | +7.2pt | +0.072 | +0.135 |

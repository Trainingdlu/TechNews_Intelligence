# G2 Retrieval Eval Report

Queries scored: **130**

Relevance gold: LLM-judged 0/1/2 over the pooled top-10 of all 3 configs (shared labels).

## Ablation table

| Config | Profile | Hit@5 | Precision@5 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|
| G0 | base + no rerank | 97.7% | 72.5% | 0.900 | 0.680 |
| G1 | wide + no rerank | 97.7% | 72.3% | 0.899 | 0.677 |
| G2 | wide + Jina rerank | 100.0% | 85.5% | 0.979 | 0.856 |

## Delta vs G0 (baseline)

| Config | Hit@5 | Precision@5 | MRR@10 | nDCG@10 |
|---|---|---|---|---|
| G1 | +0.0pt | -0.2pt | -0.001 | -0.004 |
| G2 | +2.3pt | +13.1pt | +0.080 | +0.175 |

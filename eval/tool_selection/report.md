# G5 Tool-Selection Eval Report

Generated: 2026-05-20T05:11:31.063554+00:00

## Headline: tool_selection_accuracy = **30.0%**  (12/40)

## Failure attribution

| Cause | Count | Meaning |
|---|---|---|
| worker-caused | 1 | expected tool WAS a candidate but tool_worker did not pick it (prompt-fixable) |
| intent-caused | 27 | expected tool was NOT even a candidate (intent_router misclassified) |

## Selection source

| Source | Count |
|---|---|
| `llm` | 39 |
| `no_candidates` | 1 |

## Per-tool accuracy

| Expected tool | Accuracy | Correct | Total |
|---|---|---|---|
| `analyze_landscape` | 0.0% | 0 | 5 |
| `build_timeline` | 0.0% | 0 | 5 |
| `compare_sources` | 0.0% | 0 | 6 |
| `compare_topics` | 0.0% | 0 | 6 |
| `fulltext_batch` | 100.0% | 4 | 4 |
| `query_news` | 100.0% | 4 | 4 |
| `search_news` | 80.0% | 4 | 5 |
| `trend_analysis` | 0.0% | 0 | 5 |

## Confusion matrix (expected -> predicted primary)

| Expected \ Predicted | `(none)` | `fulltext_batch` | `query_news` | `search_news` |
|---|---|---|---|---|
| `analyze_landscape` | 0 | 0 | 0 | 5 |
| `build_timeline` | 0 | 0 | 0 | 5 |
| `compare_sources` | 0 | 0 | 5 | 1 |
| `compare_topics` | 1 | 0 | 0 | 5 |
| `fulltext_batch` | 0 | 4 | 0 | 0 |
| `query_news` | 0 | 0 | 3 | 1 |
| `search_news` | 0 | 0 | 1 | 4 |
| `trend_analysis` | 0 | 0 | 1 | 4 |

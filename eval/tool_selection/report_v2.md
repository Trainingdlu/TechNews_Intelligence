# G5 Tool-Selection Eval Report

Generated: 2026-05-20T05:29:26.198025+00:00

## Headline: tool_selection_accuracy = **77.5%**  (31/40)

## Failure attribution

| Cause | Count | Meaning |
|---|---|---|
| worker-caused | 9 | expected tool WAS a candidate but tool_worker did not pick it (prompt-fixable) |
| intent-caused | 0 | expected tool was NOT even a candidate (intent_router misclassified) |

## Selection source

| Source | Count |
|---|---|
| `llm` | 40 |

## Per-tool accuracy

| Expected tool | Accuracy | Correct | Total |
|---|---|---|---|
| `analyze_landscape` | 80.0% | 4 | 5 |
| `build_timeline` | 80.0% | 4 | 5 |
| `compare_sources` | 16.7% | 1 | 6 |
| `compare_topics` | 83.3% | 5 | 6 |
| `fulltext_batch` | 100.0% | 4 | 4 |
| `query_news` | 100.0% | 4 | 4 |
| `search_news` | 100.0% | 5 | 5 |
| `trend_analysis` | 80.0% | 4 | 5 |

## Confusion matrix (expected -> predicted primary)

| Expected \ Predicted | `analyze_landscape` | `build_timeline` | `compare_sources` | `compare_topics` | `query_news` | `read_news_content` | `search_news` | `trend_analysis` |
|---|---|---|---|---|---|---|---|---|
| `analyze_landscape` | 4 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| `build_timeline` | 0 | 4 | 0 | 0 | 0 | 0 | 1 | 0 |
| `compare_sources` | 0 | 0 | 1 | 0 | 5 | 0 | 0 | 0 |
| `compare_topics` | 0 | 0 | 0 | 5 | 0 | 0 | 1 | 0 |
| `fulltext_batch` | 0 | 0 | 0 | 0 | 0 | 4 | 0 | 0 |
| `query_news` | 0 | 0 | 0 | 0 | 2 | 0 | 2 | 0 |
| `search_news` | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 0 |
| `trend_analysis` | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 4 |

# G5 Tool-Selection Eval Report

Generated: 2026-05-20T06:05:39.438811+00:00

## Headline: tool_selection_accuracy = **100.0%**  (40/40)

## Failure attribution

| Cause | Count | Meaning |
|---|---|---|
| worker-caused | 0 | expected tool WAS a candidate but tool_worker did not pick it (prompt-fixable) |
| intent-caused | 0 | expected tool was NOT even a candidate (intent_router misclassified) |

## Selection source

| Source | Count |
|---|---|
| `llm` | 40 |

## Per-tool accuracy

| Expected tool | Accuracy | Correct | Total |
|---|---|---|---|
| `analyze_landscape` | 100.0% | 5 | 5 |
| `build_timeline` | 100.0% | 5 | 5 |
| `compare_sources` | 100.0% | 6 | 6 |
| `compare_topics` | 100.0% | 6 | 6 |
| `fulltext_batch` | 100.0% | 4 | 4 |
| `query_news` | 100.0% | 4 | 4 |
| `search_news` | 100.0% | 5 | 5 |
| `trend_analysis` | 100.0% | 5 | 5 |

## Confusion matrix (expected -> predicted primary)

| Expected \ Predicted | `analyze_landscape` | `build_timeline` | `compare_sources` | `compare_topics` | `fulltext_batch` | `read_news_content` | `search_news` | `trend_analysis` |
|---|---|---|---|---|---|---|---|---|
| `analyze_landscape` | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `build_timeline` | 0 | 5 | 0 | 0 | 0 | 0 | 0 | 0 |
| `compare_sources` | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 0 |
| `compare_topics` | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 |
| `fulltext_batch` | 0 | 0 | 0 | 0 | 1 | 3 | 0 | 0 |
| `query_news` | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0 |
| `search_news` | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 0 |
| `trend_analysis` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 |

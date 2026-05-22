# G5 Tool-Selection Eval Report

Generated: 2026-05-21T10:05:01.642667+00:00

## Headline: tool_selection_accuracy = **100.0%**  (97/97)

## Failure attribution

| Cause | Count | Meaning |
|---|---|---|
| worker-caused | 0 | expected tool WAS a candidate but tool_worker did not pick it (prompt-fixable) |
| intent-caused | 0 | expected tool was NOT even a candidate (intent_router misclassified) |

## Selection source

| Source | Count |
|---|---|
| `llm` | 97 |

## Per-tool accuracy

| Expected tool | Accuracy | Correct | Total |
|---|---|---|---|
| `analyze_landscape` | 100.0% | 12 | 12 |
| `build_timeline` | 100.0% | 12 | 12 |
| `compare_sources` | 100.0% | 11 | 11 |
| `compare_topics` | 100.0% | 13 | 13 |
| `fulltext_batch` | 100.0% | 8 | 8 |
| `get_db_stats` | 100.0% | 4 | 4 |
| `list_topics` | 100.0% | 4 | 4 |
| `query_news` | 100.0% | 10 | 10 |
| `search_news` | 100.0% | 11 | 11 |
| `trend_analysis` | 100.0% | 12 | 12 |

## Confusion matrix (expected -> predicted primary)

| Expected \ Predicted | `analyze_landscape` | `build_timeline` | `compare_sources` | `compare_topics` | `fulltext_batch` | `get_db_stats` | `list_topics` | `query_news` | `read_news_content` | `search_news` | `trend_analysis` |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `analyze_landscape` | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `build_timeline` | 0 | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `compare_sources` | 0 | 0 | 11 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `compare_topics` | 0 | 0 | 0 | 13 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `fulltext_batch` | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 5 | 0 | 0 |
| `get_db_stats` | 0 | 0 | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 0 | 0 |
| `list_topics` | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 0 |
| `query_news` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 9 | 0 |
| `search_news` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 11 | 0 |
| `trend_analysis` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 12 |

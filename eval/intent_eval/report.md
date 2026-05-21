# G4 Intent Classification Eval Report

Generated: 2026-05-20T06:20:23.035992+00:00

## Headline: intent_type_accuracy = **98.8%**  (79/80)

**Status: stable** (>=90%). Per the agreed thresholds, no deeper investigation needed.

## Per-bucket accuracy

| Bucket (expected_intent_type) | Accuracy | Correct | Total |
|---|---|---|---|
| `article_read` | 100.0% | 8 | 8 |
| `landscape` | 100.0% | 8 | 8 |
| `needs_clarification` | 100.0% | 8 | 8 |
| `news_analysis` | 87.5% | 7 | 8 |
| `roundup_listing` | 100.0% | 8 | 8 |
| `smalltalk_or_capability` | 100.0% | 8 | 8 |
| `source_comparison` | 100.0% | 8 | 8 |
| `timeline` | 100.0% | 8 | 8 |
| `topic_comparison` | 100.0% | 8 | 8 |
| `trend` | 100.0% | 8 | 8 |

## Confusion matrix

Rows = expected; columns = predicted (predicted intent_type, except for `needs_clarification` row where the column is predicted route).

| Expected \ Predicted | `article_read` | `landscape` | `needs_clarification` | `news_analysis` | `roundup_listing` | `smalltalk_or_capability` | `source_comparison` | `timeline` | `topic_comparison` | `trend` |
|---|---|---|---|---|---|---|---|---|---|---|
| `article_read` | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `landscape` | 0 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `needs_clarification` | 0 | 0 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `news_analysis` | 0 | 0 | 0 | 7 | 0 | 0 | 0 | 0 | 0 | 1 |
| `roundup_listing` | 0 | 0 | 0 | 0 | 8 | 0 | 0 | 0 | 0 | 0 |
| `smalltalk_or_capability` | 0 | 0 | 0 | 0 | 0 | 8 | 0 | 0 | 0 | 0 |
| `source_comparison` | 0 | 0 | 0 | 0 | 0 | 0 | 8 | 0 | 0 | 0 |
| `timeline` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 8 | 0 | 0 |
| `topic_comparison` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 8 | 0 |
| `trend` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 8 |

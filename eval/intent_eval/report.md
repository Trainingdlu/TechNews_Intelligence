# G4 Intent Classification Eval Report

Generated: 2026-05-21T12:42:33.620612+00:00

## Headline: intent_type_accuracy = **98.7%**  (148/150)

**Status: stable** (>=90%). Per the agreed thresholds, no deeper investigation needed.

## Per-bucket accuracy

| Bucket (expected_intent_type) | Accuracy | Correct | Total |
|---|---|---|---|
| `article_read` | 100.0% | 15 | 15 |
| `landscape` | 100.0% | 15 | 15 |
| `needs_clarification` | 93.3% | 14 | 15 |
| `news_analysis` | 93.3% | 14 | 15 |
| `roundup_listing` | 100.0% | 15 | 15 |
| `smalltalk_or_capability` | 100.0% | 15 | 15 |
| `source_comparison` | 100.0% | 15 | 15 |
| `timeline` | 100.0% | 15 | 15 |
| `topic_comparison` | 100.0% | 15 | 15 |
| `trend` | 100.0% | 15 | 15 |

## Confusion matrix

Rows = expected; columns = predicted (predicted intent_type, except for `needs_clarification` row where the column is predicted route).

| Expected \ Predicted | `article_read` | `direct_answer` | `landscape` | `needs_clarification` | `news_analysis` | `roundup_listing` | `smalltalk_or_capability` | `source_comparison` | `timeline` | `topic_comparison` | `trend` |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `article_read` | 15 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `landscape` | 0 | 0 | 15 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `needs_clarification` | 0 | 1 | 0 | 14 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `news_analysis` | 0 | 0 | 0 | 0 | 14 | 0 | 0 | 0 | 0 | 0 | 1 |
| `roundup_listing` | 0 | 0 | 0 | 0 | 0 | 15 | 0 | 0 | 0 | 0 | 0 |
| `smalltalk_or_capability` | 0 | 0 | 0 | 0 | 0 | 0 | 15 | 0 | 0 | 0 | 0 |
| `source_comparison` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 15 | 0 | 0 | 0 |
| `timeline` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 15 | 0 | 0 |
| `topic_comparison` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 15 | 0 |
| `trend` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 15 |

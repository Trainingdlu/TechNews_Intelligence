# G4 意图分类评测报告

生成时间：2026-06-09T10:45:04.032602+00:00

## 核心结果：intent_type_accuracy = **99.3%**  (149/150)

**状态：稳定**（>=90%）。按约定阈值，无需继续深挖。

## 分桶准确率

| 分桶（expected_intent_type） | 准确率 | 正确数 | 总数 |
|---|---|---|---|
| `article_read` | 100.0% | 15 | 15 |
| `landscape` | 100.0% | 15 | 15 |
| `needs_clarification` | 93.3% | 14 | 15 |
| `news_analysis` | 100.0% | 15 | 15 |
| `roundup_listing` | 100.0% | 15 | 15 |
| `smalltalk_or_capability` | 100.0% | 15 | 15 |
| `source_comparison` | 100.0% | 15 | 15 |
| `timeline` | 100.0% | 15 | 15 |
| `topic_comparison` | 100.0% | 15 | 15 |
| `trend` | 100.0% | 15 | 15 |

## 混淆矩阵

行表示期望值，列表示预测值；`needs_clarification` 行的列使用 predicted route，其余行使用 predicted intent_type。

| 期望 \ 预测 | `article_read` | `direct_answer` | `landscape` | `needs_clarification` | `news_analysis` | `roundup_listing` | `smalltalk_or_capability` | `source_comparison` | `timeline` | `topic_comparison` | `trend` |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `article_read` | 15 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `landscape` | 0 | 0 | 15 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `needs_clarification` | 0 | 1 | 0 | 14 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `news_analysis` | 0 | 0 | 0 | 0 | 15 | 0 | 0 | 0 | 0 | 0 | 0 |
| `roundup_listing` | 0 | 0 | 0 | 0 | 0 | 15 | 0 | 0 | 0 | 0 | 0 |
| `smalltalk_or_capability` | 0 | 0 | 0 | 0 | 0 | 0 | 15 | 0 | 0 | 0 | 0 |
| `source_comparison` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 15 | 0 | 0 | 0 |
| `timeline` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 15 | 0 | 0 |
| `topic_comparison` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 15 | 0 |
| `trend` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 15 |

# G5 工具选择评测报告

生成时间：2026-06-09T15:09:05.422582+00:00

## 核心结果：tool_selection_accuracy = **97.9%**  (95/97)

> 警告：2 个样本缺少预测结果（status != success）。

## 失败归因

| 原因 | 数量 | 含义 |
|---|---|---|
| worker-caused | 0 | 期望工具已经进入候选集，但 tool_worker 未选中，通常可通过 worker 提示词修复 |
| intent-caused | 0 | 期望工具没有进入候选集，通常说明 intent_router 分类错误 |

## 选择来源

| 来源 | 数量 |
|---|---|
| `llm` | 95 |

## 分工具准确率

| 期望工具 | 准确率 | 正确数 | 总数 |
|---|---|---|---|
| `analyze_landscape` | 100.0% | 12 | 12 |
| `build_timeline` | 100.0% | 12 | 12 |
| `compare_sources` | 100.0% | 11 | 11 |
| `compare_topics` | 100.0% | 13 | 13 |
| `fulltext_batch` | 100.0% | 8 | 8 |
| `get_db_stats` | 100.0% | 4 | 4 |
| `list_topics` | 50.0% | 2 | 4 |
| `query_news` | 100.0% | 10 | 10 |
| `search_news` | 100.0% | 11 | 11 |
| `trend_analysis` | 100.0% | 12 | 12 |

## 混淆矩阵（期望工具 -> 预测主工具）

| 期望 \ 预测 | `analyze_landscape` | `build_timeline` | `compare_sources` | `compare_topics` | `fulltext_batch` | `get_db_stats` | `list_topics` | `query_news` | `read_news_content` | `search_news` | `trend_analysis` |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `analyze_landscape` | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `build_timeline` | 0 | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `compare_sources` | 0 | 0 | 11 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `compare_topics` | 0 | 0 | 0 | 13 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `fulltext_batch` | 0 | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 4 | 0 | 0 |
| `get_db_stats` | 0 | 0 | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 0 | 0 |
| `list_topics` | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 |
| `query_news` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 7 | 0 |
| `search_news` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 11 | 0 |
| `trend_analysis` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 12 |

# 评测说明

评测相关文档已迁移至 `docs/testing/` 目录：

- [测试文档索引](../docs/testing/README.md)
- [智能体评测结构说明](../docs/testing/agent-eval-structure.md)
- [智能体测试与评测指南](../docs/testing/agent-testing-guide.md)

`eval/reports/` 下的运行报告为生成产物，不纳入版本跟踪。

## 数据集字段

每条 JSONL 用例支持以下字段：

- `id`、`category`、`capability`、`question`
- `min_urls`、`must_contain`
- `expected_facts`（可选）：用于 `fact_hit_rate` 的短语列表
- `expected_fact_groups`（可选）：用于 `fact_group_hit_rate` 的分组短语别名
- `required_tools`（可选）：用于 `tool_path_hit_rate` 的期望工具列表
- `acceptable_tool_paths`（可选）：用于 `tool_path_accept_hit_rate` 的可接受工具组合
- `must_not_contain`（可选）：用于 `forbidden_claim_rate` 的禁用短语列表
- `expected_source_domains`（可选）：用于 `source_domain_hit_rate` 的期望引用域名
- `tags`、`enabled`

若未提供 `required_tools`，加载器会根据能力字段使用默认映射（例如：`compare_topics -> compare_topics`、`timeline -> build_timeline`）。

## 常用命令

```bash
# 冒烟评测
python eval/run_eval.py --suite smoke --runs-per-question 1 --output eval/reports/smoke.json

# 默认题集 + 质量门禁
python eval/run_eval.py \
  --suite default \
  --runs-per-question 3 \
  --fail-on-react-success-rate 0.90 \
  --fail-on-react-error-rate 0.10 \
  --fail-on-avg-min-url-hit-rate 0.85 \
  --fail-on-avg-fact-hit-rate 0.70 \
  --fail-on-avg-tool-path-hit-rate 0.70 \
  --fail-on-avg-fact-group-hit-rate 0.70 \
  --fail-on-avg-tool-path-accept-hit-rate 0.80 \
  --fail-on-avg-source-domain-hit-rate 0.60 \
  --fail-on-avg-forbidden-claim-rate 0.05 \
  --output eval/reports/latest.json
```

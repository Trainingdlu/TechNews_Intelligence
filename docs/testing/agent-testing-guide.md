# Agent 测试与评测指南（ReAct 架构）

## 1. 目标

本指南用于统一 Agent 的测试与评测流程，确保以下能力可量化、可回归、可门禁：

- ReAct 工具调用链路稳定
- 输出引用与证据格式可控
- 重复运行波动可接受
- 递归上限和上游异常时可友好降级

## 2. 架构前提

当前运行时是单一路径 ReAct 架构：

- 不再维护 `AGENT_RUNTIME` / `AGENT_RUNTIME_STRICT`
- 不再维护旧 `router.py` / `pipelines.py` 强制路由
- 评测与监控统一使用 `react_*` 指标

## 3. 测试范围

包含：

- `agent/agent.py`：ReAct 主流程、Prompt 注入、异常兜底
- `agent/core/evidence.py`：URL 提取、引用重写、来源段构建
- `agent/core/metrics.py`：`react_*` 计数与速率
- `agent/skills/`：结构化工具实现与执行契约
- `app/bot.py`：消息发送鲁棒性（重试、格式化、限流）
- `eval/run_eval.py`：评测报告与门禁逻辑

不包含：

- 前端渲染测试
- n8n 上游采集链路数据质量细节

## 4. 关键环境参数

运行时（`agent/.env.example`）：

- `AGENT_REACT_RECURSION_LIMIT=25`
- `AGENT_TEMPERATURE=0.1`
- `AGENT_ROUTE_METRICS=true`
- `AGENT_ROUTE_LOG_EVERY=20`
- `BOT_SEND_RETRY_ATTEMPTS=2`
- `BOT_SEND_RETRY_BASE_SEC=0.8`
- `BOT_MAX_CITATION_URLS=12`

评测（`eval/run_eval.py`）：

- 题集选择：`--suite` / `--dataset`
- 过滤：`--categories` / `--capabilities` / `--include-disabled`
- 重复运行：`--runs-per-question` / `--sleep-seconds`
- 输出控制：`--output` / `--include-outputs`
- 门禁：`--fail-on-*`

## 5. 文件职责矩阵

- `tests/unit/test_agent_route_metrics.py`：ReAct 指标、Prompt 注入、异常兜底
- `tests/unit/test_bot_robustness.py`：Telegram 发送重试、HTML 回退、限流
- `tests/unit/test_eval_core.py`：评测指标计算、baseline 比较、门禁判断
- `tests/unit/test_eval_dataset_loader.py`：题集解析、能力映射、筛选
- `tests/unit/test_tools_structured_output.py`：工具结构化输出契约

## 6. 标准流程

### 6.1 单元测试

```bash
pytest tests -v
```

快速回归：

```bash
pytest tests/unit/test_agent_route_metrics.py -v
pytest tests/unit/test_bot_robustness.py -v
pytest tests/unit/test_eval_core.py -v
```

### 6.2 稳定性评测（本地）

```bash
python eval/run_eval.py \
  --suite default \
  --runs-per-question 3 \
  --include-outputs \
  --output eval/reports/latest.json
```

### 6.3 门禁评测（可选）

```bash
python eval/run_eval.py \
  --suite default \
  --runs-per-question 3 \
  --fail-on-avg-error-rate 0.05 \
  --fail-on-react-error-rate 0.10 \
  --fail-on-react-success-rate 0.90 \
  --fail-on-react-recursion-limit-rate 0.10 \
  --fail-on-avg-min-url-hit-rate 0.85
```

返回码：

- `0`：门禁通过
- `2`：门禁失败

## 7. 指标解释

- `summary.avg_error_rate`：`[EVAL_ERROR]` 占比（越低越好）
- `summary.avg_min_url_hit_rate`：满足最小 URL 约束命中率（越高越好）
- `summary.avg_pairwise_similarity`：多次运行一致性
- `summary.avg_unique_response_ratio`：重复运行输出去重比例

ReAct 路由指标：

- `route_metrics.react_attempts`
- `route_metrics.react_success`
- `route_metrics.react_error`
- `route_metrics.react_recursion_limit_hit`
- `route_metrics.react_success_rate`
- `route_metrics.react_error_rate`
- `route_metrics.react_recursion_limit_rate`

## 8. 推荐阈值（起步）

- `avg_error_rate <= 0.05`
- `react_error_rate <= 0.10`
- `react_success_rate >= 0.90`
- `react_recursion_limit_rate <= 0.10`
- `avg_min_url_hit_rate >= 0.85`

## 9. 常见异常排查

1. `react_error_rate` 升高：先检查上游 API 配额、网络、模型可用性。
2. `react_recursion_limit_rate` 升高：检查是否触发工具循环，必要时缩小问题跨度。
3. `avg_min_url_hit_rate` 下降：检查工具输出 URL 数量与引用后处理链路。
4. `avg_pairwise_similarity` 过低：检查 prompt 漂移或工具返回字段不稳定。


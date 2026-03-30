# Agent 测试与评测指南（ReAct 架构）

## 1. 目标

本文档用于统一 Agent 的测试与评测流程，确保以下能力可量化、可回归、可门禁：

- ReAct 工具调用链路稳定
- 输出引用与证据格式稳定
- 重复问答波动可控
- 递归上限/上游错误时的用户友好降级可用

适用场景：本地开发验证、CI 合并门禁、发布前验收。

## 2. 架构前提

当前运行时是单一 ReAct 架构：

- 不再维护 `AGENT_RUNTIME` / `AGENT_RUNTIME_STRICT`
- 不再维护旧 `router.py` / `pipelines.py` 强制路由链路
- 评测与监控统一以 `react_*` 指标为准

## 3. 测试范围

### 3.1 范围内

- `agent.py` 的 ReAct 主流程、Prompt 注入与异常兜底
- `core/evidence.py` 的 URL 提取、引用重写、来源段构建
- `core/metrics.py` 的 `react_*` 计数与速率
- `tools.py` 结构化输出契约（JSON 分支）
- `bot.py` 的消息发送鲁棒性（重试、格式化、限流）
- `eval/run_eval.py` 的评测报告与门禁逻辑

### 3.2 范围外

- 前端渲染测试
- n8n 上游采集链路的数据质量细节

## 4. 环境参数与配置

### 4.1 运行时参数（`agents/.env.example`）

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `AGENT_REACT_RECURSION_LIMIT` | `25` | ReAct 最大工具递归轮数 |
| `AGENT_TEMPERATURE` | `0.1` | 模型温度 |
| `AGENT_ROUTE_METRICS` | `true` | 是否启用 ReAct 指标统计 |
| `AGENT_ROUTE_LOG_EVERY` | `20` | 每 N 次请求打印一次指标 |
| `BOT_SEND_RETRY_ATTEMPTS` | `2` | Telegram 发送失败最大重试次数 |
| `BOT_SEND_RETRY_BASE_SEC` | `0.8` | Telegram 发送重试退避基数 |
| `BOT_MAX_CITATION_URLS` | `12` | 回答中最多参与引用重建的 URL 数 |

### 4.2 Eval 参数（`agents/eval/run_eval.py`）

| 参数 | 说明 |
|---|---|
| `--suite` | 题库套件（`default` / `smoke`） |
| `--dataset` | 自定义 JSONL 题库路径 |
| `--categories` | 按类别过滤（逗号分隔） |
| `--capabilities` | 按能力过滤（逗号分隔） |
| `--include-disabled` | 是否包含 `enabled=false` 的 case |
| `--strict-capability-check` / `--no-strict-capability-check` | 能力字段是否严格校验 |
| `--runs-per-question` | 每题重复运行次数 |
| `--sleep-seconds` | 重复运行间隔 |
| `--max-cases` | 若 > 0，仅运行前 N 条 |
| `--include-outputs` | 报告包含完整回答文本 |
| `--baseline` | 基线报告路径 |
| `--fail-on-avg-error-rate` | `summary.avg_error_rate` 上限 |
| `--fail-on-react-error-rate` | `route_metrics.react_error_rate` 上限 |
| `--fail-on-react-success-rate` | `route_metrics.react_success_rate` 下限 |
| `--fail-on-react-recursion-limit-rate` | `route_metrics.react_recursion_limit_rate` 上限 |
| `--fail-on-avg-min-url-hit-rate` | `summary.avg_min_url_hit_rate` 下限 |
| `--fail-on-avg-phrase-hit-rate` | `summary.avg_phrase_hit_rate` 下限 |
| `--fail-on-avg-pairwise-similarity` | `summary.avg_pairwise_similarity` 下限 |

说明：

- 当前 `route_metrics_schema_version=3`
- baseline 来自旧 schema 时，建议先重跑生成新 baseline

## 5. 测试文件职责矩阵

| 测试文件 | 主要覆盖内容 |
|---|---|
| `agents/tests/unit/test_agent_route_metrics.py` | ReAct 指标、Prompt 注入探测、异常友好兜底、核心导出与后处理 |
| `agents/tests/unit/test_bot_robustness.py` | Telegram 发送重试、HTML 回退、限流与引用重建 |
| `agents/tests/unit/test_eval_core.py` | 评测指标计算、baseline 比较、质量门禁 |
| `agents/tests/unit/test_eval_dataset_loader.py` | 题库解析、能力映射、筛选逻辑 |
| `agents/tests/unit/test_tools_structured_output.py` | `query_news/fulltext_batch` 结构化输出契约与时间处理 |

## 6. 标准测试流程

### 6.1 单元测试（默认）

```bash
pytest agents/tests -v
```

通过标准：

- 全部用例通过
- 无导入错误与语法错误

按文件快速回归：

```bash
pytest agents/tests/unit/test_agent_route_metrics.py -v
pytest agents/tests/unit/test_bot_robustness.py -v
pytest agents/tests/unit/test_eval_core.py -v
```

### 6.2 稳定性评测（本地）

```bash
python agents/eval/run_eval.py \
  --suite default \
  --runs-per-question 3 \
  --include-outputs \
  --output agents/eval/reports/latest.json
```

### 6.3 门禁评测（可选）

```bash
python agents/eval/run_eval.py \
  --suite default \
  --runs-per-question 3 \
  --fail-on-avg-error-rate 0.05 \
  --fail-on-react-error-rate 0.10 \
  --fail-on-react-success-rate 0.90 \
  --fail-on-react-recursion-limit-rate 0.10 \
  --fail-on-avg-min-url-hit-rate 0.85
```

返回码：

- `0`：门禁全部通过
- `2`：门禁失败

### 6.4 GitHub Actions

默认 CI：`.github/workflows/ci.yml`

1. 安装 `agents/requirements.txt`
2. 执行 `pytest agents/tests -v`

手动评测：`.github/workflows/eval-manual.yml`

- 通过 Actions 页面触发 `Agent Eval (Manual)`，输出 `agents/eval/reports/ci_smoke.json`

## 7. 指标解释

### 7.1 Summary 指标

- `summary.avg_error_rate`：回答中 `[EVAL_ERROR]` 占比，越低越好
- `summary.avg_min_url_hit_rate`：达到最小 URL 约束的命中率，越高越好
- `summary.avg_pairwise_similarity`：重复运行的一致性
- `summary.avg_unique_response_ratio`：重复运行输出去重比例，越低通常越稳定

### 7.2 Route 指标（ReAct）

- `route_metrics.react_attempts`：ReAct 尝试次数
- `route_metrics.react_success`：ReAct 成功次数
- `route_metrics.react_error`：ReAct 异常次数
- `route_metrics.react_recursion_limit_hit`：递归上限命中次数
- `route_metrics.react_success_rate`：成功率（`react_success/react_attempts`）
- `route_metrics.react_error_rate`：错误率（`react_error/react_attempts`）
- `route_metrics.react_recursion_limit_rate`：递归上限命中率（`react_recursion_limit_hit/react_attempts`）

## 8. 推荐阈值（起步）

- `avg_error_rate <= 0.05`
- `react_error_rate <= 0.10`
- `react_success_rate >= 0.90`
- `react_recursion_limit_rate <= 0.10`
- `avg_min_url_hit_rate >= 0.85`

## 9. 异常排查建议

1. `react_error_rate` 升高：先看上游 API 配额、网络、模型可用性。
2. `react_recursion_limit_rate` 升高：检查提示词是否触发过度工具循环，必要时降低问题跨度或提高检索命中率。
3. `avg_min_url_hit_rate` 下降：检查工具输出 URL 数量和引用后处理链路。
4. `avg_pairwise_similarity` 过低：检查提示词漂移与工具返回不稳定字段。

## 10. 维护规则

- 当 `agent.py`、`eval_core.py`、`run_eval.py` 的行为或参数发生变化时，本文档必须同步更新。
- 新增门禁参数时，要同时更新：
  - 本文档
  - CI/评测脚本
  - 对应单元测试

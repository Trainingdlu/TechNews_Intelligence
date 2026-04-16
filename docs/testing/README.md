# Agent 测试与评测指南（ReAct 架构）

本指南用于统一 Agent 的测试与评测流程，集中维护智能体的测试结构说明、评测目录结构、测试实施指南以及 LangSmith + Ragas 操作规范等核心功能说明。

---

## 1. 目标与架构前提

### 1.1 目标
确保以下能力可量化、可回归、可门禁：
- ReAct 工具调用链路稳定
- 输出引用与证据格式可控
- 重复运行波动可接受
- 递归上限和上游异常时可友好降级

### 1.2 架构前提
当前运行时是单一路径 ReAct 架构：
- 不再维护 `AGENT_RUNTIME` / `AGENT_RUNTIME_STRICT`
- 不再维护旧 `router.py` / `pipelines.py` 强制路由
- 评测与监控统一使用 `react_*` 指标

---

## 2. 目录结构与单元测试

### 2.1 单元测试结构说明 (`tests/`)
`tests/` 目录采用分层组织：
- `tests/unit/`：核心单元测试（`test_*.py`）
- `tests/utils/`：测试辅助工具（路径引导、桩对象、公共夹具）
- `tests/reports/`：测试运行产物输出目录

**当前重点测试文件：**
- `tests/unit/test_agent_route_metrics.py`：ReAct 指标、Prompt 注入、异常兜底
- `tests/unit/test_bot_robustness.py`：Telegram 发送重试、HTML 回退、限流
- `tests/unit/test_eval_core.py`：评测指标计算、baseline 比较、门禁判断
- `tests/unit/test_eval_dataset_loader.py`：题集解析、能力映射、筛选
- `tests/unit/test_tools_structured_output.py`：工具结构化输出契约

### 2.2 评测目录结构说明 (`eval/`)
`eval/` 目录职责如下：
- `run_eval.py`：评测入口（题集加载、执行、报告输出、门禁判断）
- `eval_core.py`：指标计算与 baseline/gate 逻辑
- `dataset_loader.py`：题集解析、过滤、能力字段校验
- `capabilities.py`：能力注册表
- `datasets/`：评测题集（`default.jsonl`、`smoke.jsonl`、`accuracy_snapshot.jsonl`）
- `reports/`：评测报告输出目录

---

## 3. 标准流程与常用命令

### 3.1 单元测试
运行全部单元测试：
```bash
pytest tests -v
```

快速回归关键模块：
```bash
pytest tests/unit/test_agent_route_metrics.py -v
pytest tests/unit/test_bot_robustness.py -v
pytest tests/unit/test_eval_core.py -v
```

### 3.2 稳定性评测（本地）
快速冒烟评测：
```bash
python eval/run_eval.py --suite smoke --runs-per-question 1 --output eval/reports/smoke.json
```
仅运行指定能力：
```bash
python eval/run_eval.py --suite default --capabilities compare_topics,timeline,landscape
```
生成最新基线报告：
```bash
python eval/run_eval.py \
  --suite default \
  --runs-per-question 3 \
  --include-outputs \
  --output eval/reports/latest.json
```
*注：当前 `route_metrics_schema_version=3`；若 baseline 来自旧 schema，建议先重跑生成新 baseline 再做回归比较。*

### 3.3 门禁评测（可选）
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
返回码：`0`：门禁通过；`2`：门禁失败。

---

## 4. 关键环境参数与指标解释

### 4.1 关键环境参数
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

### 4.2 指标解释与推荐阈值（起步）
- `summary.avg_error_rate <= 0.05`：`[EVAL_ERROR]` 占比（越低越好）
- `summary.avg_min_url_hit_rate >= 0.85`：满足最小 URL 约束命中率（越高越好）
- `summary.avg_pairwise_similarity`：多次运行一致性
- `summary.avg_unique_response_ratio`：重复运行输出去重比例

**ReAct 路由指标与阈值：**
- `react_error_rate <= 0.10`
- `react_success_rate >= 0.90`
- `react_recursion_limit_rate <= 0.10`

### 4.3 常见异常排查
1. **`react_error_rate` 升高**：先检查上游 API 配额、网络、模型可用性。
2. **`react_recursion_limit_rate` 升高**：检查是否触发工具循环，必要时缩小问题跨度。
3. **`avg_min_url_hit_rate` 下降**：检查工具输出 URL 数量与引用后处理链路。
4. **`avg_pairwise_similarity` 过低**：检查 prompt 漂移或工具返回字段不稳定。

---

## 5. LangSmith + Ragas 操作指南

### 5.1 先决条件
本地已安装评测依赖：`python -m pip install -r requirements-eval.txt`。

`agent/.env` 至少包含：
```dotenv
LANGSMITH_TRACING=true
LANGCHAIN_TRACING_V2=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=technews-agent-local
LANGSMITH_WORKSPACE_ID=
```
*注：`LANGSMITH_TRACING=true` 用于开启追踪，`LANGCHAIN_TRACING_V2=true` 作为兼容开关保留。*

### 5.2 界面与评测对应操作
1. **导出评测报告与 Ragas 输入：**
```bash
python eval/run_eval.py \
  --suite default \
  --runs-per-question 3 \
  --experiment-group G0_baseline \
  --include-trace-summary \
  --include-outputs \
  --export-ragas-jsonl eval/reports/ragas/input_latest.jsonl \
  --output eval/reports/latest.json
```
2. **计算 Ragas 指标：**
```bash
python eval/ragas_langsmith_eval.py \
  --rows-jsonl eval/reports/ragas/input_latest.jsonl \
  --output eval/reports/ragas/latest.json
```
3. **上传到 LangSmith Dataset（可选）：**
```bash
python eval/ragas_langsmith_eval.py \
  --rows-jsonl eval/reports/ragas/input_latest.jsonl \
  --output eval/reports/ragas/latest.json \
  --upload-langsmith \
  --langsmith-dataset technews-ragas-default
```
（进入 `Datasets & Experiments`，在 Experiments 中选择同一 Dataset 可比较不同实验组结果）
# 测试文档索引

本目录集中维护智能体的测试与评测文档：

- [单元测试结构说明](./agent-tests-structure.md)
- [评测目录结构说明](./agent-eval-structure.md)
- [智能体测试与评测指南](./agent-testing-guide.md)
- [LangSmith + Ragas 操作指南](./langsmith-ragas-ops.md)
- [评测执行入口（含矩阵说明）](../../eval/README.md)

# LangSmith + Ragas 操作指南（当前项目）

## 1. 先决条件

- 本地已安装评测依赖：`python -m pip install -r requirements-eval.txt`
- `agent/.env` 至少包含以下变量：

```dotenv
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_TRACING_V2=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=technews-agent-local
LANGSMITH_WORKSPACE_ID=
```

说明：

- `LANGSMITH_TRACING=true` 用于开启 LangSmith 追踪。
- `LANGCHAIN_TRACING_V2=true` 作为兼容开关保留，避免不同 SDK 组合下不生效。

## 2. LangSmith 界面操作

1. 在左侧 `Tracing` 页面点击 `+ Project`，创建项目名（建议与 `LANGSMITH_PROJECT` 完全一致）。
2. 保持页面在 `Tracing`，先运行一次本地请求（API/Bot/Eval 任一入口），确认出现 Run。
3. 打开任一 Run，检查 Run Tree 中是否能看到：
   - 用户输入
   - 工具调用步骤（tool spans）
   - 最终回答
4. 在搜索栏按 metadata/tag 过滤：
   - `tag:eval`
   - `metadata.case_id:<case_id>`
   - `metadata.experiment_group:<group>`
5. 进入 `Datasets & Experiments`：
   - 先创建或确认 Dataset（可用 `eval/ragas_langsmith_eval.py --upload-langsmith` 自动创建）
   - 在 Experiments 中选择同一 Dataset 比较不同实验组结果。

## 3. 评测流程对应操作

1. 导出评测报告 + Ragas 输入：

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

2. 计算 Ragas 指标：

```bash
python eval/ragas_langsmith_eval.py \
  --rows-jsonl eval/reports/ragas/input_latest.jsonl \
  --output eval/reports/ragas/latest.json
```

3. 上传到 LangSmith Dataset（可选）：

```bash
python eval/ragas_langsmith_eval.py \
  --rows-jsonl eval/reports/ragas/input_latest.jsonl \
  --output eval/reports/ragas/latest.json \
  --upload-langsmith \
  --langsmith-dataset technews-ragas-default
```

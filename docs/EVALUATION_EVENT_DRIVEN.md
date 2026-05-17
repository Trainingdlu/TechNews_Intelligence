# 新闻智能体事件驱动评测

新的评测体系把问题拆成三层：检索评测、生成评测、端到端 Agent 评测。不要再用一个题集同时评估检索、生成和完整链路。

主流程是：

```text
新闻库 -> 事件卡片 -> 三类评测数据集 -> 分层运行评测 -> 汇总问题归因
```

## 1. 先在服务器准备目录和数据库

服务器没有宿主机 Python 环境时，统一通过 `bot` 容器执行。

```bash
cd ~/projects/TechNews_Intelligence
mkdir -p eval/logs eval/datasets eval/reports

docker compose --env-file deployment/.env -f deployment/docker-compose.yml up -d postgres
```

如果只是新增或修改 Python 文件，且命令里挂载了 `-v "$PWD:/app"`，通常不需要重新构建镜像。

## 2. 小批量构建事件卡片

事件卡片是评测主数据层。先小批量跑 20 个事件，确认新闻库内容和事件质量。

```bash
nohup docker compose --env-file deployment/.env -f deployment/docker-compose.yml run --rm --no-deps \
  -w /app \
  -e PYTHONUNBUFFERED=1 \
  -v "$PWD:/app" \
  bot python -u -m eval.build_event_cards \
    --output eval/datasets/event_cards_smoke.jsonl \
    --manifest-output eval/datasets/event_cards_manifest_smoke.json \
    --days 30 \
    --candidate-limit 300 \
    --max-events 20 \
  > eval/logs/build_event_cards_smoke.log 2>&1 &
```

查看日志：

```bash
tail -f eval/logs/build_event_cards_smoke.log
```

快速检查数量：

```bash
docker compose --env-file deployment/.env -f deployment/docker-compose.yml run -T --rm --no-deps \
  -w /app \
  -v "$PWD:/app" \
  bot python - <<'PY'
from pathlib import Path
p = Path("eval/datasets/event_cards_smoke.jsonl")
print("event_count:", sum(1 for line in p.read_text(encoding="utf-8").splitlines() if line.strip()))
print(p.read_text(encoding="utf-8").splitlines()[0][:800] if p.exists() and p.stat().st_size else "empty")
PY
```

## 3. 由事件卡片构建三类数据集

```bash
nohup docker compose --env-file deployment/.env -f deployment/docker-compose.yml run --rm --no-deps \
  -w /app \
  -e PYTHONUNBUFFERED=1 \
  -v "$PWD:/app" \
  bot python -u -m eval.build_event_eval_datasets \
    --events eval/datasets/event_cards_smoke.jsonl \
    --retrieval-output eval/datasets/retrieval_cases_smoke.jsonl \
    --generation-output eval/datasets/generation_cases_smoke.jsonl \
    --e2e-output eval/datasets/e2e_cases_smoke.jsonl \
    --manifest-output eval/datasets/event_eval_manifest_smoke.json \
    --max-events 20 \
    --questions-per-event 3 \
  > eval/logs/build_event_eval_datasets_smoke.log 2>&1 &
```

查看日志：

```bash
tail -f eval/logs/build_event_eval_datasets_smoke.log
```

检查三类数据集数量：

```bash
docker compose --env-file deployment/.env -f deployment/docker-compose.yml run -T --rm --no-deps \
  -w /app \
  -v "$PWD:/app" \
  bot python - <<'PY'
from pathlib import Path
for name in ["retrieval_cases_smoke.jsonl", "generation_cases_smoke.jsonl", "e2e_cases_smoke.jsonl"]:
    p = Path("eval/datasets") / name
    print(name, sum(1 for line in p.read_text(encoding="utf-8").splitlines() if line.strip()))
PY
```

## 4. 第一优先级：先跑检索评测

检索评测不看最终回答，只看问题能否召回正确 URL 或同一新闻事件。

```bash
nohup docker compose --env-file deployment/.env -f deployment/docker-compose.yml run --rm --no-deps \
  -w /app \
  -e PYTHONUNBUFFERED=1 \
  -v "$PWD:/app" \
  bot python -u -m eval.run_retrieval_eval \
    --dataset eval/datasets/retrieval_cases_smoke.jsonl \
    --events eval/datasets/event_cards_smoke.jsonl \
    --output eval/reports/retrieval_eval_smoke.json \
    --k 5 \
  > eval/logs/run_retrieval_eval_smoke.log 2>&1 &
```

查看结果：

```bash
tail -f eval/logs/run_retrieval_eval_smoke.log
```

重点看：

- `avg_exact_recall_at_k`：是否召回指定 gold URL。
- `avg_event_hit_at_k`：是否召回同一新闻事件的任意相关 URL。
- `avg_mrr_at_k`：正确结果排得是否靠前。

如果检索评测不稳定，不要继续跑完整 Agent，先优化检索参数、关键词、时间窗口、rerank 或事件卡片质量。

## 5. 第二步：跑固定证据生成评测

这一层不跑检索，只把事件卡片中的证据直接喂给模型，测试模型能不能基于证据写出可信回答。

```bash
nohup docker compose --env-file deployment/.env -f deployment/docker-compose.yml run --rm --no-deps \
  -w /app \
  -e PYTHONUNBUFFERED=1 \
  -e TASK_EVAL_PROVIDER=deepseek \
  -e TASK_EVAL_MODEL=deepseek-v4-pro \
  -v "$PWD:/app" \
  bot python -u -m eval.run_generation_eval \
    --dataset eval/datasets/generation_cases_smoke.jsonl \
    --output eval/reports/generation_eval_smoke.json \
    --provider deepseek \
    --model deepseek-v4-pro \
  > eval/logs/run_generation_eval_smoke.log 2>&1 &
```

查看结果：

```bash
tail -f eval/logs/run_generation_eval_smoke.log
```

重点看：

- `avg_claim_coverage`：关键事实覆盖率。
- `avg_unsupported_url_count`：是否出现非证据 URL。
- `avg_forbidden_hit_count`：是否出现明确禁止的证据外断言。

## 6. 第三步：跑端到端 Agent 评测

只有当检索评测和固定证据生成评测都能解释清楚时，再跑完整链路。

```bash
nohup docker compose --env-file deployment/.env -f deployment/docker-compose.yml run --rm --no-deps \
  -w /app \
  -e PYTHONUNBUFFERED=1 \
  -v "$PWD:/app" \
  bot python -u -m eval.run_e2e_event_eval \
    --dataset eval/datasets/e2e_cases_smoke.jsonl \
    --events eval/datasets/event_cards_smoke.jsonl \
    --output eval/reports/e2e_event_eval_smoke.json \
    --k 5 \
  > eval/logs/run_e2e_event_eval_smoke.log 2>&1 &
```

查看结果：

```bash
tail -f eval/logs/run_e2e_event_eval_smoke.log
```

重点看 `attribution_counts`：

- `TOOL_PATH_FAIL`：没有进入工具链路。
- `RETRIEVAL_FAIL`：工具执行了，但没有召回目标事件。
- `GENERATION_FAIL`：有证据但没有生成有效回答。
- `GUARD_OR_CITATION_FAIL`：回答中出现非证据 URL。
- `EVIDENCE_USE_FAIL`：召回了证据但回答没有使用。
- `SYSTEM_FAIL`：运行异常。
- `OK`：链路基本符合预期。

## 7. 正式运行建议

正式运行不要一开始就全量。推荐顺序：

```text
20 个事件 smoke
-> 人工抽样看事件卡片是否合理
-> retrieval-only
-> generation-only
-> e2e smoke
-> 100 个事件 preflight
-> 人工抽样
-> 正式评测
```

正式事件卡片示例：

```bash
nohup docker compose --env-file deployment/.env -f deployment/docker-compose.yml run --rm --no-deps \
  -w /app \
  -e PYTHONUNBUFFERED=1 \
  -v "$PWD:/app" \
  bot python -u -m eval.build_event_cards \
    --output eval/datasets/event_cards.jsonl \
    --manifest-output eval/datasets/event_cards_manifest.json \
    --days 60 \
    --candidate-limit 1000 \
    --max-events 100 \
  > eval/logs/build_event_cards.log 2>&1 &
```

## 8. 什么时候停止继续往下跑

出现下面情况时，先停在当前层排查：

- 事件卡片的 `event_title`、`facts`、URL 明显不一致：先修事件卡片构建。
- 检索层 `avg_event_hit_at_k` 很低：先修检索，不要看最终回答。
- 固定证据生成层事实覆盖低：先修最终综合 prompt 或模型选择。
- 端到端失败主要是 `TOOL_PATH_FAIL`：先修意图判断和工具规划。
- 端到端失败主要是 `RETRIEVAL_FAIL`：回到检索层优化。


# 新闻智能体事件驱动评测

事件驱动评测体系面向科技新闻情报分析 Agent，基于新闻库构建事件卡片，并生成检索、固定证据生成和端到端评测用例。体系以事件级证据、可复现运行配置和质量审计结果为评估依据，最终产出 `resume_metrics.md/json` 指标报告。

```text
新闻库 -> 事件卡片 -> retrieval / generation / e2e cases -> 分层评测 -> 质量审计 -> resume_metrics
```

核心原则：

- 检索、生成、端到端不要混在一个指标里评。先定位是哪一层出问题，再谈优化。
- 单事件问题用 `gold_event_id` 和 `gold_urls`，评估能否定位到目标新闻事件。
- 宽泛主题问题用 `gold_event_ids`，评估事件集合覆盖、多样性和无关结果比例，不再用单一 URL 命中解释效果。
- 简历数字必须包含样本量 N、数据集 fingerprint、Baseline/Candidate 配置、指标定义、运行日期、是否包含 rerank、是否经过人工抽检。

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

一张合格事件卡片应该表示一个真实单一事件，而不是一个主题合集。至少要有可靠的 `core_urls`、`related_urls`、`facts`、`entities`、`event_type`。后面的宽泛主题题也是从合格事件卡片聚合出来的，所以这里质量不稳，后面指标没有意义。

事件卡片构建默认会过滤没有实体锚点的卡片。只有做探索性排查时才使用 `--allow-missing-entities`，正式评测不要打开这个选项。

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

快速检查数量和样例：

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

这一层现在会同时生成：

- 单事件 retrieval cases：`case_kind=single_event`，使用 `gold_event_id`、`gold_urls`。
- 宽泛主题 retrieval cases：`case_kind=broad_topic`，使用 `gold_event_ids`、`topic`、`expected_entities`、`expected_event_types`。
- generation cases：固定 evidence，不跑检索；`required_claims` 是 atomic claims，每条 claim 映射到 evidence URL。
- e2e cases：单事件和宽泛主题都会进入端到端评测。

```bash
nohup docker compose --env-file deployment/.env -f deployment/docker-compose.yml run --rm --no-deps \
  -w /app \
  -e PYTHONUNBUFFERED=1 \
  -e TASK_EVAL_PROVIDER=deepseek \
  -e TASK_EVAL_MODEL=deepseek-v4-pro \
  -v "$PWD:/app" \
  bot python -u -m eval.build_event_eval_datasets \
    --events eval/datasets/event_cards_smoke.jsonl \
    --retrieval-output eval/datasets/retrieval_cases_smoke.jsonl \
    --generation-output eval/datasets/generation_cases_smoke.jsonl \
    --e2e-output eval/datasets/e2e_cases_smoke.jsonl \
    --manifest-output eval/datasets/event_eval_manifest_smoke.json \
    --max-events 20 \
    --questions-per-event 3 \
    --min-broad-topic-events 2 \
    --max-broad-events-per-case 6 \
    --question-mode archetype \
  > eval/logs/build_event_eval_datasets_smoke.log 2>&1 &
```

`archetype` 是默认模式：程序先从事件卡片抽取公司、产品、事件类型，再按真实用户问题类型生成问题。不要默认用新闻标题或事实句改写问题，否则题目会像“摘要改写”而不是用户查询。`llm` 模式只作为后续辅助，不作为正式题集的主入口。

查看日志：

```bash
tail -f eval/logs/build_event_eval_datasets_smoke.log
```

检查三类数据集数量和宽泛题数量：

```bash
docker compose --env-file deployment/.env -f deployment/docker-compose.yml run -T --rm --no-deps \
  -w /app \
  -v "$PWD:/app" \
  bot python - <<'PY'
import json
from collections import Counter
from pathlib import Path

for name in ["retrieval_cases_smoke.jsonl", "generation_cases_smoke.jsonl", "e2e_cases_smoke.jsonl"]:
    p = Path("eval/datasets") / name
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(name, len(rows), Counter(row.get("case_kind", "generation") for row in rows))
PY
```

抽样检查生成的问题是否像真实用户会问的问题：

```bash
docker compose --env-file deployment/.env -f deployment/docker-compose.yml run -T --rm --no-deps \
  -w /app \
  -v "$PWD:/app" \
  bot python - <<'PY'
import json
from pathlib import Path
p = Path("eval/datasets/retrieval_cases_smoke.jsonl")
for line in p.read_text(encoding="utf-8").splitlines()[:16]:
    row = json.loads(line)
    print(row.get("case_kind"), row["query_type"], "=>", row["question"])
PY
```

如果问题里仍然出现完整新闻标题、半截句子、`核心议题`、`本文记录`、`分析显示`、`这条和`、`重点看` 这类内部摘要词，先不要继续跑检索评测，应该回到问题生成层调整。

## 4. 先做数据质量审计

评测数据能不能写进简历，取决于这里的门禁。数量不足、宽泛题建模不完整、人工抽检不够，或者抽检不合格率超过 10%，都不要生成正式简历指标。

正式目标：

- 至少 `100` 个合格 event cards。
- 至少 `150` 条单事件 retrieval cases。
- 至少 `50` 条宽泛主题 retrieval cases。
- 至少 `50` 条 generation cases。
- 至少 `50` 条 e2e cases。
- 人工抽检：20 个 event cards + 30 个 cases。
- 抽检不合格率超过 `10%` 时，禁止生成简历指标。

人工抽检可以先用 JSONL 记录，最小字段如下：

```json
{"item_type":"event_card","item_id":"event_id_here","passed":true,"note":""}
{"item_type":"case","item_id":"retrieval.topic.ai.001","passed":true,"note":""}
```

运行质量审计：

```bash
docker compose --env-file deployment/.env -f deployment/docker-compose.yml run -T --rm --no-deps \
  -w /app \
  -v "$PWD:/app" \
  bot python -m eval.audit_event_eval_quality \
    --events eval/datasets/event_cards_smoke.jsonl \
    --retrieval eval/datasets/retrieval_cases_smoke.jsonl \
    --generation eval/datasets/generation_cases_smoke.jsonl \
    --e2e eval/datasets/e2e_cases_smoke.jsonl \
    --manual-review eval/datasets/manual_review_smoke.jsonl \
    --output eval/reports/event_eval_quality_audit_smoke.json
```

重点看：

- `summary.allowed_for_resume`：是否允许进入简历指标产物。
- `acceptance.broad_topic_retrieval_cases`：宽泛主题题数量是否达标。
- `issues`：事件卡片或 case 的结构性问题。

## 5. 第一优先级：先跑检索评测

检索评测不看最终回答，只看问题能否召回正确 URL、同一新闻事件，或者宽泛主题下的目标事件集合。

单事件题重点看：

- `avg_single_event_hit_at_k`：是否召回 `gold_event_id` 对应事件。
- `avg_single_mrr_at_k`：正确结果排得是否靠前。
- `avg_exact_hit_at_k`：是否命中指定 gold URL。

宽泛主题题重点看：

- `avg_event_set_recall_at_k`：命中的 `gold_event_ids` 数 / `gold_event_ids` 总数。
- `avg_event_diversity_at_k`：主题概览覆盖了多少个目标事件。
- `avg_irrelevant_event_ratio_at_k`：top-k 中不属于 gold 或 acceptable 事件集合的比例。
- `avg_source_diversity_at_k`：命中事件覆盖的来源域名数量。

先跑 Baseline：

```bash
nohup docker compose --env-file deployment/.env -f deployment/docker-compose.yml run --rm --no-deps \
  -w /app \
  -e PYTHONUNBUFFERED=1 \
  -e EVAL_RECALL_PROFILE=base \
  -e NEWS_RERANK_MODE=none \
  -v "$PWD:/app" \
  bot python -u -m eval.run_retrieval_eval \
    --dataset eval/datasets/retrieval_cases_smoke.jsonl \
    --events eval/datasets/event_cards_smoke.jsonl \
    --output eval/reports/retrieval_eval_baseline_smoke.json \
    --k 5 \
  > eval/logs/run_retrieval_eval_baseline_smoke.log 2>&1 &
```

再跑 Candidate 1：

```bash
nohup docker compose --env-file deployment/.env -f deployment/docker-compose.yml run --rm --no-deps \
  -w /app \
  -e PYTHONUNBUFFERED=1 \
  -e EVAL_RECALL_PROFILE=wide \
  -e NEWS_RERANK_MODE=none \
  -v "$PWD:/app" \
  bot python -u -m eval.run_retrieval_eval \
    --dataset eval/datasets/retrieval_cases_smoke.jsonl \
    --events eval/datasets/event_cards_smoke.jsonl \
    --output eval/reports/retrieval_eval_wide_smoke.json \
    --k 5 \
  > eval/logs/run_retrieval_eval_wide_smoke.log 2>&1 &
```

如果有可用 `JINA_API_KEY`，再跑 Candidate 2：

```bash
nohup docker compose --env-file deployment/.env -f deployment/docker-compose.yml run --rm --no-deps \
  -w /app \
  -e PYTHONUNBUFFERED=1 \
  -e EVAL_RECALL_PROFILE=wide \
  -e NEWS_RERANK_MODE=llm_rerank \
  -v "$PWD:/app" \
  bot python -u -m eval.run_retrieval_eval \
    --dataset eval/datasets/retrieval_cases_smoke.jsonl \
    --events eval/datasets/event_cards_smoke.jsonl \
    --output eval/reports/retrieval_eval_wide_rerank_smoke.json \
    --k 5 \
  > eval/logs/run_retrieval_eval_wide_rerank_smoke.log 2>&1 &
```

如果没有可用 `JINA_API_KEY`，Candidate 2 不进入简历结论，只在 `resume_metrics` 里标记为 skipped。

## 6. 第二步：跑固定证据生成评测

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

- `avg_claim_coverage`：atomic claims 覆盖率。
- `avg_unsupported_url_rate`：非证据 URL 泄漏率。
- `avg_unsupported_url_count`：每条回答平均非证据 URL 数。
- `avg_forbidden_hit_count`：是否出现明确禁止的证据外断言。

## 7. 第三步：跑端到端 Agent 评测

只有当检索评测和固定证据生成评测都能解释清楚时，再跑完整链路。端到端不是用来替代检索评测的，它的重点是失败归因。

```bash
nohup docker compose --env-file deployment/.env -f deployment/docker-compose.yml run --rm --no-deps \
  -w /app \
  -e PYTHONUNBUFFERED=1 \
  -e EVAL_RECALL_PROFILE=wide \
  -e NEWS_RERANK_MODE=none \
  -v "$PWD:/app" \
  bot python -u -m eval.run_e2e_event_eval \
    --dataset eval/datasets/e2e_cases_smoke.jsonl \
    --events eval/datasets/event_cards_smoke.jsonl \
    --output eval/reports/e2e_event_eval_wide_smoke.json \
    --k 5 \
  > eval/logs/run_e2e_event_eval_wide_smoke.log 2>&1 &
```

查看结果：

```bash
tail -f eval/logs/run_e2e_event_eval_wide_smoke.log
```

重点看 `attribution_counts`：

- `TOOL_PATH_FAIL`：没有进入工具链路。
- `RETRIEVAL_FAIL`：工具执行了，但没有召回目标事件；宽泛题下表示事件集合覆盖为 0。
- `GENERATION_FAIL`：有证据但没有生成有效回答。
- `GUARD_OR_CITATION_FAIL`：回答中出现非证据 URL。
- `EVIDENCE_USE_FAIL`：召回了证据但回答没有使用。
- `SYSTEM_FAIL`：运行异常。
- `OK`：链路基本符合预期。

## 8. 生成简历指标产物

正式简历数据只从 `resume_metrics.md/json` 里取，不要从某个 runner 的临时日志里手抄数字。

```bash
docker compose --env-file deployment/.env -f deployment/docker-compose.yml run -T --rm --no-deps \
  -w /app \
  -v "$PWD:/app" \
  bot python -m eval.build_resume_metrics \
    --baseline-retrieval eval/reports/retrieval_eval_baseline_smoke.json \
    --candidate-retrieval eval/reports/retrieval_eval_wide_smoke.json \
    --rerank-retrieval eval/reports/retrieval_eval_wide_rerank_smoke.json \
    --generation eval/reports/generation_eval_smoke.json \
    --e2e-candidate eval/reports/e2e_event_eval_wide_smoke.json \
    --audit-report eval/reports/event_eval_quality_audit_smoke.json \
    --manual-review-status pass \
    --manual-fail-rate 0.05 \
    --manual-card-sample-size 20 \
    --manual-case-sample-size 30 \
    --output-json eval/reports/resume_metrics.json \
    --output-md eval/reports/resume_metrics.md
```

输出里每个数字都会带：

- 样本量 N。
- 数据集 fingerprint。
- Baseline 配置。
- Candidate 配置。
- 指标定义。
- 运行日期。
- 是否包含 rerank。
- 是否经过人工抽检。

可写进简历的数据形态应该来自 `eval/reports/resume_metrics.md`：

- “在 N 条单事件检索样本上，event hit@5 从 A 提升到 B。”
- “在 N 条主题概览样本上，事件集合覆盖率从 A 提升到 B，无关事件比例从 C 降到 D。”
- “在 N 条固定证据生成样本上，非证据 URL 泄漏率为 X。”
- “在 N 条端到端样本中，通过 Trace 将失败归因到检索、生成、引用守卫等环节。”

## 9. 正式运行建议

正式运行不要一开始就全量。推荐顺序：

```text
20 个事件 smoke
-> 人工抽样看事件卡片是否合理
-> 构建 retrieval / generation / e2e cases
-> quality audit
-> baseline retrieval
-> wide retrieval
-> generation-only
-> e2e smoke
-> resume_metrics smoke
-> 100 个事件 preflight
-> 人工抽样 20 个 event cards + 30 个 cases
-> 正式三组配置评测
-> resume_metrics 正式产物
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

正式评测保留三组配置：

```text
Baseline:    EVAL_RECALL_PROFILE=base + NEWS_RERANK_MODE=none
Candidate 1: EVAL_RECALL_PROFILE=wide + NEWS_RERANK_MODE=none
Candidate 2: EVAL_RECALL_PROFILE=wide + NEWS_RERANK_MODE=llm_rerank
```

没有可用 `JINA_API_KEY` 时，Candidate 2 只记录 skipped，不进入简历结论。简历默认只使用 Baseline vs Candidate 1 的对比数据。

## 10. 什么时候停止继续往下跑

出现下面情况时，先停在当前层排查：

- 事件卡片的 `event_title`、`facts`、URL 明显不一致：先修事件卡片构建。
- 事件卡片像主题合集而不是单一事件：先拆卡片，不要继续生成 case。
- 宽泛主题题只有 1 个 `gold_event_id`：先修 broad topic builder 或提高事件卡片覆盖。
- 检索层 `avg_single_event_hit_at_k` 很低：先修单事件检索，不要看最终回答。
- 宽泛主题 `avg_event_set_recall_at_k` 很低或 `avg_irrelevant_event_ratio_at_k` 很高：先修召回 profile、时间窗、实体别名或 rerank。
- 固定证据生成层事实覆盖低：先修最终综合 prompt、模型选择或 atomic claims 质量。
- 固定证据生成层 `avg_unsupported_url_rate` 高：先修 evidence guard 和引用白名单。
- 端到端失败主要是 `TOOL_PATH_FAIL`：先修意图判断和工具规划。
- 端到端失败主要是 `RETRIEVAL_FAIL`：回到检索层优化。
- `event_eval_quality_audit` 不允许进入简历：不要生成正式简历指标，先补数据和人工抽检。

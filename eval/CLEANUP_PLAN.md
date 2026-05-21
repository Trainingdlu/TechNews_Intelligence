# eval/ 目录清理计划

> 目标：移除两套已废弃的旧评测框架（任务驱动、事件卡驱动），保留新的 G1–G5 分层评测体系（见 [`EVAL_SYSTEM.md`](EVAL_SYSTEM.md)）和混在 eval/ 下的几个活跃工具。
>
> **状态：待执行。** 本文是经确认的清理蓝图；删除按分阶段执行，每阶段后跑 `pytest tests -q` 验证。

## 已确认的决策

1. `rerank_eval.py` → **删除**（已被 G2 检索评测覆盖）。
2. `docs/EVALUATION.md` → **改为指针页**，仅保留一段指向 `eval/EVAL_SYSTEM.md` 的说明（README 仍链接它）。
3. entity 别名工具（`build_entity_alias_candidates.py` / `promote_entity_alias_candidates.py`）→ **原地保留**，仅记录"未来建议移出 eval/"，本次不动。

## 调查结论（决定怎么删的关键事实）

1. **eval/ 是纯离线目录**：`agent/app/services` 无任何 `import eval`，删除不影响生产 Runtime。
2. **新 G1–G5 完全自包含**：五个子目录不 import 任何 eval 顶层旧模块。
3. **旧框架 = 两个自包含集群 + 共享模块**，互不与新代码纠缠（依赖图已验证）。
4. **爆炸半径在 tests 和 CI**：`tests/unit/` 有约 20 个单测 import 旧模块；CI 跑全量 `pytest tests` → 删模块必须同步删对应单测。
5. **eval/ 里混着 3 个非失败框架的真工具**，保留：`trace_query.py`、`encoding_guard.py`（CI 依赖）、entity 别名工具对。

## KEEP（保留）

| 项 | 理由 |
|---|---|
| `eval/{error_analysis,intent_eval,retrieval,generation,tool_selection}/` | 新 G1–G5 |
| `eval/EVAL_SYSTEM.md`、`eval/CLEANUP_PLAN.md` | 新方案文档 / 本计划 |
| `eval/trace_query.py` | Trace 工具，README/docs 引用 |
| `eval/encoding_guard.py` | CI 依赖 |
| `eval/build_entity_alias_candidates.py`、`promote_entity_alias_candidates.py` | 活跃数据流水线（被 deployment/docs/agent 引用） |
| `eval/reports/encoding_guard/` | CI 写入目录 |
| `tests/unit/test_encoding_guard.py`、`test_entity_automation.py` | 对应保留项 |

## DELETE（删除清单）

### 集群 A — 任务驱动框架（13 模块）
`build_task_dataset` · `run_task_eval` · `run_matrix_eval` · `task_eval_schema` · `task_eval_scoring` · `build_task_eval_leaderboard` · `build_report` · `sample_task_eval_dataset` · `inspect_task_dataset` · `audit_task_topics` · `corpus_sampler` · `pool_quality` · `evidence_validator`

### 集群 B — 事件卡框架（9 模块）
`build_event_cards` · `build_event_eval_datasets` · `run_e2e_event_eval` · `run_retrieval_eval` · `run_generation_eval` · `audit_event_eval_quality` · `news_eval_schema` · `news_eval_metrics` · `build_resume_metrics`

### 共享模块（仅 A/B 使用，最后删）
`eval_core` · `common` · `llm_judge`

### 独立模块（已被 G2 取代）
`rerank_eval`

### 关联删除
- **单测（~20）**：`test_run_task_eval_helpers` `test_run_matrix_eval` `test_task_dataset_audit_regen` `test_task_dataset_env_args` `test_task_dataset_fingerprint` `test_task_dataset_generation_fallback` `test_task_dataset_quality` `test_task_eval_evidence_validator` `test_task_eval_pathset_and_coverage` `test_task_eval_schema` `test_task_topic_audit` `test_eval_corpus_sampler` `test_build_event_cards` `test_event_eval_quality_audit` `test_event_retrieval_eval_runner` `test_event_eval_schema_and_metrics` `test_resume_metrics_builder` `test_eval_core` `test_eval_retrieval_metrics` `test_rerank_eval`
- **config**：`eval/config/*.json`（task_types_* 6 个 + matrix.json）
- **datasets**：`eval/datasets/versions/**`、`eval/datasets/rerank_mini.json`、生成的 `task_eval_*.jsonl`
- **reports**：`eval/reports/**` 除 `encoding_guard/` 外全部
- **deployment**：`deployment/scripts/eval/run_eval.sh`
- **docs**：`docs/EVALUATION_EVENT_DRIVEN.md` 删除；`docs/EVALUATION.md` 改为指针页

### Cosmetic
- `eval/__init__.py` docstring：`"Task-driven evaluation package."` → 改为新体系描述。
- `README.md` 项目文档链接：`docs/EVALUATION.md` 指针页保留即可，必要时加一行指向 `eval/EVAL_SYSTEM.md`。

## 风险与缓解

| 风险 | 缓解 |
|---|---|
| 删模块漏删测试 → CI 红 | 模块 + 测试成对删，每阶段后 `pytest tests -q` |
| 共享模块过早删 | `eval_core/common/llm_judge` 最后删（A、B 清完之后） |
| 未提交 WIP 混入清理提交 | 先把当前生产修复单独 commit，清理另起 commit |
| 误删 CI 依赖 | encoding_guard 及其 report 目录保留；ci.yml 无需改动 |
| 删除不可逆 | 仓库受 git 管理，分阶段 commit，可回滚 |

## 分阶段执行计划

> 前置：先 `git commit` 当前 WIP（生产修复 + 新评测文档），清理在干净树上进行，单独成 commit。

- **Phase 1 — 集群 A**：删 A 的 13 模块 + A 单测 + `eval/config/*.json` + `eval/datasets/versions/**` + reports(A) + `run_eval.sh` + `docs/EVALUATION_EVENT_DRIVEN.md` → `pytest tests -q`
- **Phase 2 — 集群 B**：删 B 的 9 模块（含 build_resume_metrics）+ B 单测 + datasets/reports(B) → `pytest tests -q`
- **Phase 3 — rerank**：删 `rerank_eval.py` + `test_rerank_eval.py` + `rerank_mini.json` + `rerank_eval_none.json` → `pytest tests -q`
- **Phase 4 — 共享模块**：删 `eval_core/common/llm_judge` + `test_eval_core` + `test_eval_retrieval_metrics` → `pytest tests -q`
- **Phase 5 — 收尾**：`docs/EVALUATION.md` 改指针页、`eval/__init__.py` docstring、README 链接、清理空的 reports 残留 → `pytest tests -q` 全绿收工

预计规模：约 26 模块 + 约 20 测试 + config/datasets/reports 一批 + docs 调整。

## docs/EVALUATION.md 指针页拟定内容

```markdown
# 评测体系

本项目的评测体系已重构为分层评测（G1–G5），完整方案与结果见：

- [`eval/EVAL_SYSTEM.md`](../eval/EVAL_SYSTEM.md)

旧的任务驱动 / 事件卡驱动框架已下线移除。
```

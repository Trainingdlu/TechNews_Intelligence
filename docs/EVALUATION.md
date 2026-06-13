# 评测体系

分层评测（G1–G6），复用线上链路定位并量化智能体在每一层的失效。系统架构见 [系统架构](ARCHITECTURE.md)。

> 本文是耐久的方法论与运行说明。**具体数值不写在正文里**——每层的实时结果由脚本生成在各自的 `report.md`，正文只指向它们；§6 给一份带日期的结果快照，会随重跑过期。

---

## 1. 评测对象与目标

被评测对象是一个 **LangGraph StateGraph 智能体**：意图判断 → 工具选择 → 工具规划 → 策略检查 → 工具执行 → 证据归一 → 是否继续 → 最终综合 → 输出守卫。它读取 PostgreSQL + pgvector 的科技新闻库，回答必须由工具证据支撑。

评测目标不是"答案读起来好不好"，而是 **用实打实的数据，定位并量化智能体在每一层会在哪里翻车**。为此把评测拆成六层（G1–G6），每层锁定一个能力层，用独立构造的题集 + 复用线上链路 + 可回溯 Trace，产出单一可辩护的核心指标。

---

## 2. 设计原则

- **题集独立于被测系统**：题目手写，不由被测模型生成，避免"同源模型出题又评分"的指标自证。
- **复用线上 Runtime**：所有 G 直接调用生产 `generate_response_eval_payload` / `intent_router`，不维护 mock 链路 —— 评的就是线上行为。
- **三级隔离**：每条 case 用空 history、独立 `thread_id`、独立 `request_id`，避免上下文 / 记忆串台污染。
- **结构化优先，LLM-as-judge 兜底**：能用确定性信号（工具命中、URL 集合、IR 标签）就不用 LLM 打分；开放文本质量才用 LLM judge，且对 judge 做**人机一致性校准**（kappa）。
- **可回溯**：每个失败 case 都能经 `agent_model_io` / `agent_trace_spans` 还原到意图 / 工具 / 检索 / 生成 / 系统层。
- **断点恢复**：长跑脚本按 JSONL append 记录，可中断续跑（429、宕机不丢进度）。
- **数据冻结**：触库的评测（G2/G3）手动暂停 n8n 采集，保证 DB 快照稳定、结果可复现。

---

## 3. 分层总览

| 层 | 目录 | 核心问题 | 核心指标 |
|---|---|---|---|
| **G1 错误分析** | `eval/error_analysis/` | 智能体到底在哪类问题翻车？ | 失败分类占比 |
| **G2 检索质量** | `eval/retrieval/` | 检索召回排序好不好？rerank 值不值？ | nDCG@10 / MRR@10 / Hit@5 |
| **G3 生成忠实度** | `eval/generation/` | 答案是否被证据支撑、是否幻觉？ | Faithfulness(1-5) / 幻觉率 / URL 泄漏 |
| **G4 意图分类** | `eval/intent_eval/` | 问题类型判得准不准？ | intent_type accuracy |
| **G5 工具选择** | `eval/tool_selection/` | 选对工具了吗？错在哪一层？ | tool_selection accuracy |
| **G6 RGB 拒答** | `eval/rgb/` | 只给干扰文档时会不会冒充证据？ | 误归因/伪造率 |

---

## 4. 评测流程

每一层都遵循同一条 **5 步流水线**；差异只在"造什么题、捕获什么、怎么打分"。

1. **造题** —— 手写独立于被测系统的 query 集。每层带该层标注：G5 标期望工具、G4 标期望意图桶、G2 标检索时间窗、G3 为开放问答 + 对抗性诱饵。
2. **跑真实链路** —— 直接调用生产入口（`generate_response_eval_payload` / `intent_router`），不维护 mock。G5/G4 只跑到意图与工具规划（不执行、不合成）以隔离该层；G3 跑完整链路；G2 跑检索的三个消融配置。
3. **捕获中间产物** —— 落盘每步可判信号：意图类型、候选工具集、LLM 选择、最终工具、检索排序、答案、引用 URL 集合；G3 额外从 Trace（`agent_model_io`）取回 synthesizer 真正看到的证据。
4. **打分（结构化优先）** —— 能用确定性信号就不用 LLM：工具是否命中、URL 是否越界、IR 相关性标签。仅开放文本质量（检索相关性、生成忠实度）才用 LLM-as-judge，并对 judge 做人机一致性（kappa）校准。
5. **出报告 + 归因** —— 自动生成准确率 / IR 指标 + 混淆矩阵；失败可回溯到意图 / 工具 / 检索 / 生成 / 系统具体哪一层（G5 进一步区分"意图层未进候选" vs "规划层未选中"）。

**横切工程机制**

| 机制 | 实现 | 解决的问题 |
|---|---|---|
| 三级隔离 | 每题空 history + 独立 thread_id + 独立 request_id | 上下文 / 记忆串台污染 |
| 断点续跑 | JSONL append，按 `status==success` 跳过 | 中断 / 配额耗尽不丢进度，并支持复用历史结果（**代价**：改代码后须先清对应 `runs/*.jsonl` 才会重测，见 §8「强制重跑」） |
| 限流退避 | 429 指数退避（共享 `eval_retry`）+ `--sleep-seconds` 案例间隔 | preview 模型配额限制下的大样本运行 |
| 数据冻结 | G2/G3 手动暂停 n8n 采集 | 同一 DB 快照，结果可复现、跨配置可比 |
| Trace 驱动 | `agent_runs / agent_trace_spans / agent_model_io` | 失败回溯 + G3 取真实证据喂给 judge |
| judge 可信度 | LLM 判分 + Cohen's kappa 人机校准 | 不盲信 LLM 评委 |

---

## 5. 各层详解

> 各层结果数值见对应 `report.md`；本节只讲方法与结论。

### G1 · 错误分析摸底（失败分类）

- **方法**：手写覆盖 10 类（A 模糊 / B 对比 / C 长尾 / D 时间窗 / E 时间线·格局 / F 上下文追问 / G URL 读取 / H 边界 / I 幻觉诱饵 / J 混合语言）的挑战 query，跑真实链路，人工归类失败模式。
- **关键发现**：智能体的强项是"不编、该澄清就澄清、对数据缺口诚实"；唯一系统性弱点是**专用分析工具（compare/timeline/landscape）很少被正确选中**，这定义了 G5 的范围。
- **文件**：`queries.jsonl` · `run_error_analysis.py` · `build_labeling_sheet.py` · `summarize.py` · `failure_taxonomy.md`

### G2 · 检索质量 + 矩阵 ablation

- **方法**：覆盖良好的 entity×aspect query；对 3 个配置（G0 base / G1 wide / G2 wide+Jina rerank）的 top-10 做 **TREC pooling**，用 LLM judge 打 0/1/2 相关性作为共享 gold；计算 IR 指标（nDCG 用指数增益 2^rel−1）。
- **结论**：**Jina rerank 是关键增量**（nDCG / MRR / P@5 显著提升）；单纯放宽召回窗口（G1）几乎无效。题集越大越难，base 的 Hit@5 不再饱和、rerank 的相对增量越明显——所以排序质量（nDCG/MRR）才是有区分度的指标。
- **judge 校准**：对 LLM 判官做人机一致性（Cohen's kappa）校准，盲标 + 隐藏 LLM 分以避免锚定。有序标签看 quadratic-weighted kappa；二元相关性 kappa 对应 Hit/MRR/P 实际依赖的"相关/不相关"判断。残留分歧主要在主观的 1↔2 边界。
- **文件**：`queries.jsonl` · `run_retrieval.py` · `judge_relevance.py` · `compute_metrics.py` · `build_calibration_sheet.py`（盲标、隐藏 key）· `compute_kappa.py` · `report.md` · `calibration_report.md`

### G3 · 生成忠实度（+ 对抗 + prompt ablation）

- **方法**：跑完整链路，捕获答案 + `valid_urls`；用 `enrich_evidence.py` 从 `agent_model_io` 取回**喂给 synthesizer 的真实证据**；LLM judge 打 faithfulness / answer_relevancy（1-5）+ 确定性 url_leak。
- **证据对齐**：judge 评分依据是 `synthesizer_evidence`（模型真实输入），而非从 DB 重抓的摘要——保证评的是模型实际所见，避免"评委上下文错位"造成的假阳性幻觉。
- **对抗集（幻觉诱饵）**：用"不存在的型号 / 精确数字 / 完整名单"等诱饵压测诚实性。扩大对抗集曾暴露一个真实缺陷：面对无数据问题，检索靠实体匹配仍返回沾边证据，synthesizer 正确地拒绝编造（返回空），但确定性兜底文案曾误称"已找到相关证据"并指向无关来源——修复兜底文案（output_guard）后，对抗集忠实度恢复、幻觉归零。剩余少量案例转为"诚实拒答"（见 §7 限制 C）。
- **Prompt ablation（grounding 的价值，两层防御）**：用弱 grounding 提示词替换生产提示词，量化提示词的边际贡献。结论：忠实度是**双层防御**——① 提示词诱导：去掉"证据优先 / 只用证据 / 不足要明说"后 faithfulness 温和下降，掉分源于外部知识漂移而非编造；② 确定性 output_guard：再去掉"正文必须引用来源 URL"时，大部分回答被 output_guard 直接拦截、到不了用户。不是单点保证。
- **文件**：`queries.jsonl` · `queries_adversarial.jsonl` · `run_generation.py`（带 `--synth-prompt-file` 做 ablation）· `enrich_evidence.py` · `judge_faithfulness.py` · `synth_prompt_weak.txt` · `report.md` · `report_adversarial.md` · `report_ablation.md` · `report_final.md`（加固提示词 before→after，对应 `../synth_prompt_final_hardened.txt`，已接入生产 `_FINAL_SYSTEM_PROMPT`）

### G4 · 意图分类评测

- **方法**：10 个 bucket，直接调用 `intent_router`（`_heuristic_intent` + LLM + `_merge_intent`），`needs_clarification` 作为伪 bucket（比对路由结果）。
- **结论**：混淆矩阵近乎完全对角，残留是相邻类目的混淆（如 `news_analysis ↔ trend`）。是 G5 修复后意图层的独立验证。
- **文件**：`queries.jsonl` · `run_intent_eval.py` · `report.md`

### G5 · 工具选择评测 + 提示词优化 delta

- **方法**：每条 case 标注期望工具；脚本同时捕获 `intent_type / 候选工具集 / LLM 选择 / 最终工具`，从而**区分失败是意图层造成（期望工具压根没进候选）还是 worker 层造成（进了候选但没被选）**。
- **根因定位 → 修复（已进生产）**：基线失败集中在意图层——intent_router 自由发挥 `intent_type` 字符串、对不上路由表的 key，期望工具从未进入候选；归因把修复指向意图层而非 worker 提示词。修复：intent_router 提示词加 `intent_type` 枚举（含 `db_status` / `topic_overview` 两类元意图）；`_merge_intent` 加合法值校验与元意图锁定守卫；`tool_planning` 补齐元意图到工具的映射；tool_worker 加工具选择 rubric。
- **价值**：评测驱动修复的闭环——量化根因、归因到正确层、提示词修复、复测确认无回归。
- **文件**：`queries.jsonl` · `run_tool_selection_eval.py`（支持 `--intent-router-prompt-file` / `--tool-worker-prompt-file` 覆盖做提示词 A/B）· `report.md`

### G6 · RGB negative-rejection（对抗性拒答 + grounding 加固 delta）

- **方法**：取 RGB 中文子集，每条只喂 `negative`（不含答案）文档 top-5，经生产接地 prompt 合成答案，压测"无证据时会不会冒充出处"。确定性剔除答案泄漏进所喂文档的 case（数字归一），避免把标注噪声误判为幻觉。
- **指标三分**：`misattributed`（把证据外内容冒充成证据支撑 = 真·幻觉，标注假设的不算）是核心失败量；`gold_present`（确定性字串匹配）、`answered`（LLM judge）用于区分"干净拒答 / 标注假设补充 / 冒充出处"三种行为。
- **grounding 加固 delta**：final 节点加入 Grounding & Refusal 硬规则后，误归因率在同题集上近乎砍半（配对：修好多于新坏），代价是回答率下降、模型更保守。换模型（gemini→deepseek）会显著抬高误归因，加固是迁移的缓解手段而非追平 gemini。
- **文件**：`run_rgb_rejection.py`（带 `--provider` / `--synth-prompt-file` 做模型与提示词 A/B）· `data/zh.json` · `report_final.md`（合并 before→after）· 各单 run `report.md` / `runs/rgb_deepseek*.md`

---

## 6. 结果快照（@2026-05-22；G3 标准、G6 @2026-06-12 刷新）

> 某次运行的快照，**会随重跑过期；实时 / 完整数字以各层 `report.md` 为准**。

| 层 | 指标 | 数值 | N |
|---|---|---|---|
| G1 错误分析 | OK / tool_wrong / 幻觉 | 74% / 26% / **0** | 50 |
| G2 检索 | nDCG@10（rerank 前→后） | **0.68 → 0.86（+26%）** | 130 |
| G2 检索 | P@5 / MRR@10（rerank 后） | 85.5% / 0.98 | 130 |
| G3 生成（标准，deepseek+加固） | faithfulness（同 45 题，加固前→后）/ 幻觉 / URL 泄漏 | **4.73 → 4.82 / 0% / 0%** | 45 |
| G3 生成（对抗） | faithfulness / 幻觉 / URL 泄漏 | 4.87 / 0% / 0%（4 个引用闸拦截） | 46 |
| G4 意图 | intent_type accuracy | **98.7%** | 150 |
| G5 工具选择 | accuracy（修复前→后） | **30% → 100%** | 97 |
| G6 RGB 拒答（deepseek，加固前→后） | 误归因/伪造率（同 299 题） | **21.7% → 11.4%** | 299 |
| judge 校准 | kappa 名义 / QWK / 二元 | 0.356 / **0.514** / 0.515 | 100 |

---

## 7. 遗留与后续

- **无数据问答的优雅降级（已知限制 C，主动暂缓）**：当问题指向不存在的具体事实、检索只返回沾边证据时，诚实的"无数据"回答因正文无可引 URL 会被 agent 层的强制引用闸拦成 `AgentGenerationError`（Web 端表现为错误而非一句"暂无相关信息"）。已确认"拒答优于乱答"，故优先保住核心 grounding 保证（0% URL 泄漏）；优雅放行无数据答案需区分"无数据" vs "有证据未引用"，改动触及 grounding 闸，留作后续。
- **数据覆盖**：新闻库当前 6 个来源，部分"无数据"回答属 coverage_gap 而非检索 bug；扩源是独立的数据工程任务。
- **compare_sources**：来源对比目前硬编码 HackerNews / TechCrunch 两个源，扩展到更多来源需重构。

---

## 8. 完整评测流程

各层入口脚本相互独立、可单独运行；下面是从零跑出全部指标的端到端顺序（对应 §4 的 5 步流水线）。两条约定：

- **触库层（G1 / G2 / G3）** 跑真实链路或检索，运行前手动暂停 n8n 采集以冻结 DB 快照；脚本会弹确认，加 `--skip-confirm` 跳过。**G4 / G5** 只跑到意图与工具规划、不触库，无需暂停。
- 所有 LLM / agent 脚本支持 429 指数退避与断点续跑（按 `status==success` 跳过），可加 `--sleep-seconds N` 拉大案例间隔避开配额；建议手动运行并观察输出。
- **改动被测代码后想看真实新结果，必须先清对应层的断点文件**，否则会被全部跳过、只用旧缓存重建报告——见下「强制重跑」。

### 强制重跑（改代码后必读）

断点续跑的代价：每个 LLM / agent 脚本把对应的 `runs/*.jsonl` 当作续跑状态，按 `status==success` 跳过已完成用例。**当所有用例都已成功后，直接重跑不会重新测试**——脚本跳过全部、仅用旧缓存重建 `report.md`（终端打印 `Resume: N case(s) already succeeded; skipping them.` 且 `pending: 0`）。所以改动被测代码后要看真实新结果，**必须先清空对应层的断点文件**。

| 阶段 | 断点文件（清它才会重跑） |
|---|---|
| G1 错误分析 | `eval/error_analysis/runs/g1_run.jsonl` |
| G2 检索 | `eval/retrieval/runs/retrieval_results.jsonl` |
| G2 相关性评判（LLM judge） | `eval/retrieval/runs/relevance_judgments.jsonl` |
| G3 生成 | `eval/generation/runs/generation.jsonl` |
| G3 忠实度评判（LLM judge） | `eval/generation/runs/faithfulness_judgments.jsonl` |
| G4 意图 | `eval/intent_eval/runs/g4_predictions.jsonl` |
| G5 工具选择 | `eval/tool_selection/runs/g5_predictions.jsonl` |
| G6 RGB 拒答 | `eval/rgb/runs/rgb_rejection.jsonl`（或对应 `--output`） |

- **G3 是多阶段链**（run → enrich → judge）：重测某一集需清掉该集的 run 与 judge 两个 `runs/*.jsonl`，再依次重跑三步；对抗 / ablation 集对应各自 `--output` 文件名。
- **纯计算 / 转换阶段**（`compute_metrics` / `compute_kappa` / `enrich_evidence` / `build_*` / `summarize`）每次都从输入重算，无续跑、无需清理。
- `--report-only`（部分脚本提供）是反向操作：完全不跑、只用现有断点重建报告。

强制重跑示例（以 G5 为例，保留旧结果以便 before/after 对比）：

```powershell
Move-Item eval\tool_selection\runs\g5_predictions.jsonl eval\tool_selection\runs\g5_predictions.before.jsonl
Copy-Item eval\tool_selection\report.md eval\tool_selection\report.before.md
python eval\tool_selection\run_tool_selection_eval.py
```

> 判读提醒：LLM 阶段（意图 / 工具 / judge）有模型不确定性，清断点重跑出现 ±1~2 条波动属正常，不等于回归。若改的是**确定性代码**（如正则等价改写、启发式重构），单元测试才是权威回归闸门，无需为验证它重跑 LLM 阶段。

### G1 · 错误分析

```powershell
python eval/error_analysis/run_error_analysis.py --skip-confirm        # -> runs/g1_run.jsonl
python eval/error_analysis/build_labeling_sheet.py                     # -> labeling_sheet.md（人工归类失败模式）
python eval/error_analysis/summarize.py                                # -> failure_taxonomy.md
```

### G2 · 检索质量 + kappa 校准

```powershell
python eval/retrieval/run_retrieval.py --skip-confirm                  # -> runs/retrieval_results.jsonl（G0/G1/G2 三配置 top-10 pooling）
python eval/retrieval/judge_relevance.py                               # -> runs/relevance_judgments.jsonl（LLM 打 0/1/2 共享 gold）
python eval/retrieval/compute_metrics.py                               # -> report.md（nDCG/MRR/Hit/P@5 + ablation delta）
# judge 可信度校准（人机一致性）
python eval/retrieval/build_calibration_sheet.py                       # -> calibration_sheet.md（盲标）+ runs/calibration_key.jsonl（隐藏 LLM 分）
python eval/retrieval/compute_kappa.py                                 # -> calibration_report.md（Cohen's kappa + 混淆矩阵）
```

### G3 · 生成忠实度（标准 / 对抗 / ablation）

```powershell
# 标准集
python eval/generation/run_generation.py --output eval/generation/runs/generation.jsonl --skip-confirm
python eval/generation/enrich_evidence.py --results eval/generation/runs/generation.jsonl --output eval/generation/runs/generation_enriched.jsonl
python eval/generation/judge_faithfulness.py --results eval/generation/runs/generation_enriched.jsonl --output eval/generation/runs/faithfulness_judgments.jsonl --report eval/generation/report.md

# 对抗集（幻觉诱饵）
python eval/generation/run_generation.py --queries eval/generation/queries_adversarial.jsonl --output eval/generation/runs/generation_adversarial.jsonl --skip-confirm
python eval/generation/enrich_evidence.py --results eval/generation/runs/generation_adversarial.jsonl --output eval/generation/runs/generation_adversarial_enriched.jsonl
python eval/generation/judge_faithfulness.py --results eval/generation/runs/generation_adversarial_enriched.jsonl --output eval/generation/runs/faithfulness_adversarial.jsonl --report eval/generation/report_adversarial.md

# Prompt ablation（弱 grounding 提示词，量化提示词的边际贡献）
python eval/generation/run_generation.py --synth-prompt-file eval/generation/synth_prompt_weak.txt --output eval/generation/runs/generation_ablation.jsonl --skip-confirm
python eval/generation/enrich_evidence.py --results eval/generation/runs/generation_ablation.jsonl --output eval/generation/runs/generation_ablation_enriched.jsonl
python eval/generation/judge_faithfulness.py --results eval/generation/runs/generation_ablation_enriched.jsonl --output eval/generation/runs/faithfulness_ablation.jsonl --report eval/generation/report_ablation.md
```

### G4 · 意图分类

```powershell
python eval/intent_eval/run_intent_eval.py                            # -> runs/g4_predictions.jsonl + report.md
```

### G5 · 工具选择

```powershell
python eval/tool_selection/run_tool_selection_eval.py                 # -> runs/g5_predictions.jsonl + report.md
```

### G6 · RGB 拒答（对抗性 negative-rejection；用自带 `data/zh.json`，不触新闻库，无需暂停 n8n）

```powershell
# deepseek 基线
python eval/rgb/run_rgb_rejection.py --provider deepseek --limit 0 --output eval/rgb/runs/rgb_deepseek.jsonl --report eval/rgb/runs/rgb_deepseek.md
# deepseek + 加固提示词（--synth-prompt-file 覆盖 final 节点提示词）
python eval/rgb/run_rgb_rejection.py --provider deepseek --limit 0 --synth-prompt-file eval/synth_prompt_final_hardened.txt --output eval/rgb/runs/rgb_deepseek_hardened.jsonl --report eval/rgb/runs/rgb_deepseek_hardened.md
```

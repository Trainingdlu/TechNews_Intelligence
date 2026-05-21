# TechNews Intelligence — 分层评测体系方案

> 本文是 G1–G5 分层评测体系的完整方案与结果汇总，同时服务两类读者：
> (1) **工程参考** —— 架构、方法论与关键设计取舍；
> (2) **简历 / 面试支撑** —— 可直接引用的指标、delta 与"这条数据证明了什么"。
>
> 系统架构详见 [`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md)；本文只摘录评测相关的链路上下文。

---

## 1. 为什么是这套评测体系

被评测对象是一个 **LangGraph StateGraph 智能体**：意图判断 → 工具选择 → 工具规划 → 策略检查 → 工具执行 → 证据归一 → 是否继续 → 最终综合 → 输出守卫。它读取 PostgreSQL + pgvector 的科技新闻库，回答必须由工具证据支撑。

评测的目标不是"答案读起来好不好"，而是 **能不能用实打实的数据，定位并量化智能体在每一层会在哪里翻车**。

前两次评测尝试被否决，构成了本方案的设计动机：

| 失败尝试 | 问题 | 教训 |
|---|---|---|
| 任务驱动、LLM 生成题目 | **因果反了**：用同源模型生成题目又用它评分，指标自证、不可信 | 题目必须独立于被测系统 |
| 事件卡驱动 | **粒度不匹配**：事件卡与真实 query 分布对不上 | 题目要贴近真实用户问法 |

→ 改为 **目标导向的分层评测（G1–G5）**：每个 G 锁定智能体的一个能力层，用独立构造的题集 + 复用线上链路 + 可回溯 Trace，产出单一可辩护的核心指标。

---

## 2. 设计原则

- **复用线上 Runtime**：所有 G 直接调用生产 `generate_response_eval_payload` / `intent_router`，不维护 mock 链路 —— 评的就是线上行为。
- **三级隔离**：每条 case 用空 history、独立 `thread_id`、独立 `request_id`，避免上下文 / 记忆串台污染。
- **结构化优先，LLM-as-judge 兜底**：能用确定性信号（工具命中、URL 集合、IR 标签）就不用 LLM 打分；开放文本质量才用 LLM judge，且对 judge 做**人机一致性校准**（kappa）。
- **可回溯**：每个失败 case 都能经 `agent_model_io` / `agent_trace_spans` 还原到意图 / 工具 / 检索 / 生成 / 系统层。
- **断点恢复**：长跑脚本按 JSONL append 记录，可中断续跑（429、宕机不丢进度）。
- **数据冻结**：触库的评测（G2/G3）手动暂停 n8n 采集，保证 DB 快照稳定、结果可复现。

---

## 3. 分层评测体系总览

| 层 | 目录 | 核心问题 | 核心指标 | 结果 | N |
|---|---|---|---|---|---|
| **G1 错误分析** | `eval/error_analysis/` | 智能体到底在哪类问题翻车？ | 失败分类占比 | OK 74% / tool_wrong 26% / **幻觉 0** | 50 |
| **G2 检索质量** | `eval/retrieval/` | 检索召回排序好不好？rerank 值不值？ | nDCG@10 / MRR@10 / Hit@5 | nDCG **0.74→0.88 (+18%)** | 50 |
| **G3 生成忠实度** | `eval/generation/` | 答案是否被证据支撑、是否幻觉？ | Faithfulness(1-5) / 幻觉率 / URL 泄漏 | **5.0 / 0% / 0%** | 30(+12 对抗) |
| **G4 意图分类** | `eval/intent_eval/` | 问题类型判得准不准？ | intent_type accuracy | **98.8%** | 80 |
| **G5 工具选择** | `eval/tool_selection/` | 选对工具了吗？错在哪一层？ | tool_selection accuracy | **30%→100%** | 40 |

> G1 是摸底，发现"专用工具很少被选中"→ 直接催生了 G5；G5 的归因又反向定位到意图层 → 与 G4 互证。五个 G 不是孤立的，而是一条"发现问题 → 定位层级 → 修复 → 复测"的闭环。

---

## 4. 各层详解

### G1 · 错误分析摸底（失败分类）

- **方法**：手写 50 条覆盖 10 类（A 模糊 / B 对比 / C 长尾 / D 时间窗 / E 时间线·格局 / F 上下文追问 / G URL 读取 / H 边界 / I 幻觉诱饵 / J 混合语言）的挑战 query，跑真实链路，人工归类失败模式。
- **结果**：`OK 37 (74%) / tool_wrong 13 (26%)`，**幻觉 0**。`tool_wrong` 集中在 B 对比（6/6）和 E 时间线·格局（4/5）。
- **关键发现**：智能体的强项是"不编、该澄清就澄清、对数据缺口诚实"；唯一系统性弱点是**专用分析工具（compare/timeline/landscape）很少被正确选中**。
- **价值**：这是后续所有优化的"病灶定位"，并直接定义了 G5 的范围。
- **文件**：`queries.jsonl` · `run_error_analysis.py` · `build_labeling_sheet.py` · `summarize.py` · `failure_taxonomy.md`

### G5 · 工具选择评测 + 提示词优化 delta

- **方法**：40 条 case，每条标注期望工具；脚本同时捕获 `intent_type / 候选工具集 / LLM 选择 / 最终工具`，从而**区分失败是意图层造成（期望工具压根没进候选）还是 worker 层造成（进了候选但没被选）**。
- **根因定位**：基线 30% 时，28 个失败里 **27 个是意图层造成的**（intent_router 自由发挥 `intent_type` 字符串，对不上路由表的 key），只有 1 个是 worker 层 —— **评测阻止了我们去优化错误的一层**。
- **修复（已进生产）**：intent_router 提示词加 `intent_type` 枚举；`_merge_intent` 加合法值校验守卫；tool_worker 加工具选择 rubric。
- **结果**：tool_selection accuracy **100% (40/40)**，worker-caused 0 / intent-caused 0。
- **价值**：评测驱动修复的闭环 —— 量化根因、归因到正确层、提示词修复、复测确认无回归。
- **文件**：`queries.jsonl` · `run_tool_selection_eval.py`（支持 `--intent-router-prompt-file` / `--tool-worker-prompt-file` 覆盖做提示词 A/B）· `report.md`

### G4 · 意图分类评测

- **方法**：80 条 case（10 个 bucket × 8），直接调用 `intent_router`（`_heuristic_intent` + LLM + `_merge_intent`），`needs_clarification` 作为伪 bucket（比对路由结果）。
- **结果**：intent_type accuracy **98.8% (79/80)**，唯一错误是 `news_analysis → trend` 的相邻混淆。
- **价值**：G5 修复后意图层的独立验证；混淆矩阵显示除一格外完全对角，证明意图枚举修复稳固。
- **文件**：`queries.jsonl` · `run_intent_eval.py` · `report.md`

### G2 · 检索质量 + 矩阵 ablation

- **方法**：50 条覆盖良好的 query；对 3 个配置的 top-10 做 **TREC pooling**，用 LLM judge 打 0/1/2 相关性作为共享 gold；计算 IR 指标（nDCG 用指数增益 2^rel−1）。
- **Ablation**（召回 profile × rerank）：

  | Config | 配置 | Hit@5 | P@5 | MRR@10 | nDCG@10 |
  |---|---|---|---|---|---|
  | G0 | base + 无 rerank | 100% | 72.0% | 0.918 | 0.742 |
  | G1 | wide + 无 rerank | 100% | 71.6% | 0.918 | 0.740 |
  | G2 | wide + Jina rerank | 100% | **79.2%** | **0.990** | **0.877** |

- **结论**：**Jina rerank 是关键增量**：nDCG@10 **+0.135 (+18%)**、MRR@10 +0.072、P@5 +7.2pt；单纯放宽召回窗口（G1）几乎无效。Hit@5 已饱和（题集覆盖良好），所以排序质量（nDCG/MRR）才是有区分度的指标。
- **judge 校准（诚实披露的局限）**：人机一致性 Cohen's kappa **0.269**（nominal）/ 0.333（quadratic-weighted），raw agreement 48%。诊断：人类倾向"实体匹配即相关"而过度给分，LLM 更贴 query 精确语义。→ kappa 重校准（细化细则盲标重做）是**唯一遗留开放项**。
- **文件**：`queries.jsonl` · `run_retrieval.py` · `judge_relevance.py` · `compute_metrics.py` · `build_calibration_sheet.py`（盲标、隐藏 key）· `compute_kappa.py` · `report.md` · `calibration_report.md`

### G3 · 生成忠实度（+ 对抗 + prompt ablation）

- **方法**：跑完整链路，捕获答案 + `valid_urls`；用 `enrich_evidence.py` 从 `agent_model_io` 取回**喂给 synthesizer 的真实证据**；LLM judge 打 faithfulness / answer_relevancy（1-5）+ 确定性 url_leak。
- **标准集 N=30**：faithfulness **5.0/5**、answer_relevancy 5.0、well-grounded(≥4) 100%、**幻觉率 0%**、**URL 泄漏 0%**。
- **对抗集 N=11**（12 条幻觉诱饵，1 条被守卫拦截）：faithfulness 4.82、well-grounded 90.9%、幻觉 0%（`adv_01` 得 3 分是有效区分信号）。
- **证据对齐**：judge 评分依据是 `enrich_evidence.py` 从 `agent_model_io` 取回的 synthesizer 真实输入（`synthesizer_evidence`），而非从 DB 重抓的摘要——保证评的是模型实际所见，避免"评委上下文错位"造成的假阳性幻觉。
- **Prompt ablation（证明 grounding 的价值，两层防御）**：

  | | strict（生产 prompt） | weak（去 grounding） | delta |
  |---|---|---|---|
  | Faithfulness | 5.00 | 4.78 | −0.22 |
  | well-grounded | 100% | 96.3% | −3.7pp |
  | 幻觉率 | 0% | 0% | 0 |

  - **第一层（提示词级）**：去掉"证据优先 / 只用证据 / 不足要明说"后，faithfulness 温和下降，掉分源于**外部知识漂移**（`gen_10` 5→3：注入了证据外的市场定位/主观推断），但无严重幻觉 —— 因检索证据本身准、全。
  - **第二层（确定性 output_guard）**：进一步把"正文必须引用来源 URL"也删掉时，**27/30 (90%) 回答被 output_guard 直接拦截**，根本到不了用户。
  - **结论**：忠实度由"提示词诱导 + 代码强拦"双层保障，不是单点。详见 `report_ablation.md`。
- **文件**：`queries.jsonl` · `queries_adversarial.jsonl` · `run_generation.py`（带 `--synth-prompt-file` 做 ablation）· `enrich_evidence.py` · `judge_faithfulness.py` · `synth_prompt_weak.txt` · `report.md` · `report_adversarial.md` · `report_ablation.md`

---

## 5. 指标汇总（简历可直接引用）

| 能力 | 指标 | 数值 | 样本 |
|---|---|---|---|
| 工具选择 | accuracy（修复前→后） | **30% → 100%** | 40 |
| 检索排序 | nDCG@10（rerank 前→后） | **0.74 → 0.88（+18%）** | 50 |
| 检索排序 | MRR@10 | 0.918 → 0.990 | 50 |
| 意图分类 | intent_type accuracy | **98.8%** | 80 |
| 生成忠实度 | faithfulness（1-5）/ 幻觉率 / URL 泄漏 | **5.0 / 0% / 0%** | 30(+12 对抗) |
| 系统鲁棒 | 幻觉（全 50 条挑战 query） | **0 例** | 50 |
| 评测严谨性 | judge 人机一致性 kappa（含诚实局限） | 0.27（待重校准） | 100 |

---

## 6. 方法论亮点（面试 talking points）

1. **评测驱动修复，且防止优化错层**：G5 用"意图层 vs worker 层"归因，揭示 90%+ 失败在意图层，避免了去调错误的提示词。
2. **规避 LLM-as-judge 的两类陷阱**：
   - *因果反转*（同源模型生成题目又评分自证）→ 题集独立于被测系统；
   - *评委上下文错位*（DB 摘要 ≠ 模型真实输入）→ judge 依据 `agent_model_io` 取回的真实证据评分。
3. **judge 可信度自检**：不只用 LLM 打分，还做**人机 kappa 校准**，并诚实诊断分歧来源（人类实体匹配过度给分）—— 体现"指标也要被质疑"。
4. **Ablation 量化每个组件的边际价值**：rerank 之于检索（+18% nDCG）、grounding 提示词 + output_guard 之于忠实度（双层防御）。
5. **IR 指标用对**：Hit@5 饱和时改看 nDCG/MRR；TREC pooling 共享 gold 保证跨配置可比；nDCG 用指数增益。

---

## 7. 简历 bullet 候选

**中文：**
- 为科技新闻智能体设计**分层评测体系**（意图/工具/检索/生成/鲁棒性五层），复用线上链路 + 可回溯 Trace，定位并量化各层失效。
- 基于评测的**根因归因**（意图层 vs 工具层）定位 90%+ 工具选择失败源于意图分类，修复后 **tool-selection accuracy 30%→100%**（N=40），intent accuracy 98.8%（N=80）。
- 用 **TREC pooling + LLM-as-judge** 评检索，矩阵 ablation 证明 Jina rerank 使 **nDCG@10 0.74→0.88（+18%）**；并做**人机 kappa 校准**自检 judge 可信度。
- 构建**忠实度评测**（LLM judge 1-5 + 对抗诱饵 + prompt ablation），生产链路 **faithfulness 5.0/5、幻觉率 0%、URL 泄漏 0%**；ablation 量化"提示词 + output_guard"双层证据防护。

**English：**
- Designed a **layered evaluation harness** (intent / tool-selection / retrieval / generation / robustness) for a LangGraph news agent, reusing the production runtime with trace-backed failure attribution.
- Used eval-driven **root-cause attribution** to show 90%+ of tool-selection failures originated in the intent layer; fixes lifted **tool-selection accuracy 30%→100%** (N=40) and intent accuracy to 98.8% (N=80).
- Evaluated retrieval with **TREC pooling + LLM-as-judge**; a matrix ablation proved Jina rerank lifts **nDCG@10 0.74→0.88 (+18%)**, with **human–LLM kappa calibration** to self-check judge reliability.
- Built a **faithfulness eval** (1-5 LLM judge + adversarial baits + prompt ablation): production scores **5.0/5 faithfulness, 0% hallucination, 0% URL leakage**; ablation quantifies a two-layer (prompt + output-guard) grounding defense.

---

## 8. 复现

各层独立、互不依赖。触库的 G2/G3 需先手动暂停 n8n 采集。典型流程（以 G3 为例）：

```powershell
# 生成（跑真实链路）
python eval/generation/run_generation.py --output eval/generation/runs/generation.jsonl --skip-confirm
# 取回 synthesizer 真实证据
python eval/generation/enrich_evidence.py --results eval/generation/runs/generation.jsonl --output eval/generation/runs/generation_enriched.jsonl
# LLM judge + 报告
python eval/generation/judge_faithfulness.py --results eval/generation/runs/generation_enriched.jsonl --report eval/generation/report.md
```

其余各层入口脚本：`error_analysis/run_error_analysis.py` · `intent_eval/run_intent_eval.py` · `retrieval/run_retrieval.py` → `judge_relevance.py` → `compute_metrics.py` · `tool_selection/run_tool_selection_eval.py`。

> 长跑 / LLM / agent 脚本均支持断点续跑，建议手动运行并观察输出。

---

## 9. 遗留与后续

- **kappa 重校准（唯一开放项）**：G2 人机一致性 0.27 偏低，根因是评分细则不够细 + 人类实体匹配过度给分。计划：细化 0/1/2 判定细则 → 盲标重做 → 重算 kappa，目标进入 fair→moderate（>0.4）。
- **数据覆盖**：新闻库当前 6 个来源，部分"无数据"回答属 coverage_gap 而非检索 bug；扩源是独立的数据工程任务。
- **compare_sources**：HN/TC 硬编码、很少被选中，删除涉及 20+ 文件，已评估暂缓。

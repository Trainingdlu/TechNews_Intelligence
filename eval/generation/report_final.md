# G3 生成忠实度 · 加固提示词 final 报告

## 结论

在 final-synthesis 节点加入一段 **Grounding & Refusal 硬规则**(证据是唯一真源、答不出要明确拒答、外部知识必须标 `Assumption` 否则舍弃、不编 URL/数字/来源、存疑即视为无证据)后,deepseek 在**同一批 45 题**上:

- 忠实度均分 **4.73 → 4.82(+0.09)**
- 证据支撑率(忠实度 ≥ 4)**93.3% → 95.6%**
- 幻觉率(忠实度 ≤ 2)与 URL 泄漏保持 **0**

该提示词已接入生产(`agent/graph/prompts.py::_FINAL_SYSTEM_PROMPT`),生产文本与本次评测所用 `eval/synth_prompt_final_hardened.txt` 逐字一致(仅差行尾换行)。

## 加固 before→after(deepseek,匹配 45 题)

| 指标 | baseline prompt | hardened prompt | Δ |
|---|---|---|---|
| 忠实度均分(1–5) | 4.73 | **4.82** | +0.09 |
| 回答相关性均分(1–5) | 5.00 | 5.00 | — |
| 证据支撑率(忠实度 ≥ 4) | 93.3% | **95.6%** | +2.3pp |
| 幻觉率(忠实度 ≤ 2) | 0% | 0% | — |
| URL 泄漏条数 | 0 | 0 | — |

**为什么取 45 题交集:** baseline 判分 49 题、hardened 判分 45 题。hardened 终态 = 45 success + 5 refusal = 50;5 个 refusal 是加固提示词触发的拒答(证据虽检索到,但模型不 commit、正文无可引 URL,被生产强制引用闸 `agent.py::_enforce_body_valid_url_guard` 拦成 `AgentGenerationError`,即已知限制 C)。其中 4 题(`gen_27 / gen_34 / gen_37 / gen_46`)在 baseline 下能作答且忠实度均为 5——加固把它们改成了拒答,这是加固的代价面;`gen_25` 在 baseline 下亦失败(coverage gap)。这 5 题无答案可判,从两臂判分集中剔除,上表取两臂都判分的 45 题交集做 apples-to-apples 对比。

## 逐题变化(匹配 45)

- **提升 8 题、持平 32 题、回退 5 题。**
- 提升集中在 baseline 过度外推、夹带世界知识的题:`gen_09 / gen_29 / gen_49` 由 3 → 5,`gen_16 / gen_22 / gen_26 / gen_32 / gen_36` 由 4 → 5。加固规则要求"证据外内容标 Assumption 或舍弃",正好压住这类无证据细节。
- 回退 5 题:`gen_02 / gen_06 / gen_11` 由 5 → 4,`gen_05 / gen_24` 由 5 → 3。
- `gen_24` 值得单独看:它原本在加固下拒答(被引用闸拦成 error),本次重跑改为给出答案、但夹带了一句标注"非证据"的外部知识(2024 苹果–OpenAI 合作),严格判官仍按"证据外内容"扣到 3。说明加固在边界题上行为非确定(拒答 ↔ 标注假设作答之间摆动),且标了假设的补充在严格忠实度口径下照样被扣分。
- **口径提醒:** 两个臂是两次**独立生成**(非同一答案重判),逐题 delta 同时包含提示词效应与生成采样噪声;聚合均值(+0.09)才是稳定信号,单题涨跌不宜单独解读。

## 模型维度(同期参照,非加固臂)

| 运行 | 模型 | 提示词 | 判分 N | 忠实度 | 支撑率 | 幻觉 | 泄漏 |
|---|---|---|---|---|---|---|---|
| deepseek baseline | deepseek | 基线 | 49 | 4.76 | 93.9% | 0% | 0% |
| deepseek hardened | deepseek | 加固 | 45 | 4.82 | 95.6% | 0% | 0% |
| gemini | gemini (vertex) | 基线 | 48 | 4.92 | 97.9% | 0% | 0% |

- gemini 是**另一模型**在生产基线提示词下的横向参照,**不属于加固 before→after**,不可与上面的 +0.09 混讲。
- 本表各行用各自完整判分集(N 不一致,仅作跨模型量级参照);加固结论以上一节 45 题匹配集为准。

## 指标口径

- 忠实度 / 回答相关性:LLM-as-judge 打分,1–5。
- 支撑率 = 忠实度 ≥ 4 占比;幻觉率 = 忠实度 ≤ 2 占比;URL 泄漏 = 回答中出现证据外 URL 的条数。
- 数据冻结:跑批前手动暂停 n8n 采集,保证 DB 快照稳定、结果可复现。

## 复现

- 生产提示词:`agent/graph/prompts.py::_FINAL_SYSTEM_PROMPT`(内容 = `eval/synth_prompt_final_hardened.txt`)。
- 外挂评测(不改生产即可消融):
  ```
  python eval/generation/run_generation.py --synth-prompt-file eval/synth_prompt_final_hardened.txt --output eval/generation/runs/generation_final_deepseek_hardened.jsonl
  python eval/generation/enrich_evidence.py --results eval/generation/runs/generation_final_deepseek_hardened.jsonl --output eval/generation/runs/generation_final_deepseek_hardened_enriched.jsonl
  python eval/generation/judge_faithfulness.py --results eval/generation/runs/generation_final_deepseek_hardened_enriched.jsonl --output eval/generation/runs/faith_final_deepseek_hardened.jsonl --report eval/generation/report_final_deepseek_hardened.md
  ```
  `--synth-prompt-file` 在运行时 monkeypatch `agent.graph.nodes._FINAL_SYSTEM_PROMPT`,用指定文件覆盖该节点提示词。
- 判分产物:`runs/faith_final_deepseek.jsonl`(baseline)、`runs/faith_final_deepseek_hardened.jsonl`(hardened)、`runs/faith_final_gemini.jsonl`(gemini)。
- 单模型自动报告:`report_final_deepseek.md` / `report_final_deepseek_hardened.md` / `report_final_gemini.md`。

## 限制

- hardened 终态有 5 题以 `refusal` 收尾(`run_generation.py` 已把 `graph_inline_citation_missing` 单独记为 refusal 而非 error):它们是加固触发的拒答(证据存在但正文无引用,被强制引用闸拦下),其中 4 题 baseline 本可作答(忠实度 5)、加固下改为拒答——属加固的代价面,而非瞬时错误。
- 边界题(如 `gen_24`)在拒答与"标注假设作答"之间存在非确定性;严格忠实度口径会扣标注假设的补充,故拒答在该口径下通常是更稳的高分策略。

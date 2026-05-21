# G3 Generation Faithfulness Report

Generated: 2026-05-21T03:35:36.822569+00:00

Cases judged: **27**

| Metric | Value |
|---|---|
| Faithfulness (avg, 1-5) | **4.78** |
| Answer relevancy (avg, 1-5) | **5.00** |
| Well-grounded rate (faithfulness>=4) | 96.3% |
| Hallucination rate (faithfulness<=2) | 0.0% |
| URL-leak rate (>=1 url outside evidence) | 0.0% |
| Total leaked URLs | 0 |

## Lowest-faithfulness cases

- `gen_10_nvidia_vs_amd` faithfulness=3 relevancy=5 leaks=0 — 回答很好地切题并使用了提供的所有数据点，但在分析过程中引入了多处证据中未提及的外部知识和主观推断（如 SEV-SNP 的市场定位、万卡集群的痛点、以及将 QBE 编译器后端与打破 CUDA 垄断强行关联）。
- `gen_11_chip_landscape` faithfulness=4 relevancy=5 leaks=0 — The answer is highly relevant and comprehensively addresses the question using the provided evidence. Almost all factual claims are well-sup
- `gen_15_opensource` faithfulness=4 relevancy=5 leaks=0 — The answer is highly relevant and accurately extracts information about Qwen3.6-27B, IBM Granite 4.1, and Mistral/OpenAI models from the evi
- `gen_16_gpt55` faithfulness=4 relevancy=5 leaks=0 — 回答非常全面地涵盖了 GPT-5.5 的能力和定价，且绝大部分信息都准确无误地来源于提供的证据。但在总结长提示词的成本涨幅时，遗漏了 128K+ 提示词 85% 的涨幅数据；此外，将 200 美元的 Pro 订阅描述为‘费用上涨’属于轻微的过度推断（证据中仅说明 API 价格翻倍
- `gen_21_cerebras_ipo` faithfulness=4 relevancy=5 leaks=0 — 回答非常全面且绝大部分细节（如融资金额、股价、财务数据、投资者回报等）都与提供的证据完全一致。唯一微小的不准确之处在于，证据指出 OpenAI 的认股权证价值超 90 亿美元是基于‘周五收盘价’，而回答中称是‘按上市首日的暴涨价格计算’（上市首日为周四）。这属于非常次要的细节偏差
- `gen_02_anthropic_enterprise` faithfulness=5 relevancy=5 leaks=0 — The answer comprehensively addresses the user's question about Anthropic's recent progress in the enterprise market. All factual claims, inc
- `gen_03_nvidia_chip` faithfulness=5 relevancy=5 leaks=0 — The answer accurately extracts and synthesizes information from the provided evidence without introducing any fabricated details. It directl
- `gen_04_claude_updates` faithfulness=5 relevancy=5 leaks=0 — The answer accurately synthesizes all the provided evidence without introducing any external or fabricated information. It directly addresse

## Prompt-ablation conclusion

Goal: isolate the contribution of the final_synthesizer grounding system prompt
(`agent/graph/prompts.py::_FINAL_SYSTEM_PROMPT`) to generation faithfulness, by
overriding it with a "weak" prompt (`synth_prompt_weak.txt`) that drops the
grounding constraints ("evidence first", "use only provided evidence", "say if
insufficient", "do not fabricate") and instead nudges for confident, specific,
fleshed-out answers.

Grounding is a **two-layer** defense; the ablation measured both layers:

**Layer 1 — prompt-level grounding (this run, weak prompt keeps the URL-citation requirement so answers reach the judge):**

| Metric | strict (production prompt) | weak (grounding removed) | delta |
|---|---|---|---|
| Faithfulness (avg, 1-5) | 5.00 | 4.78 | -0.22 |
| Well-grounded (>=4) | 100% | 96.3% | -3.7pp |
| Hallucination (<=2) | 0% | 0% | 0 |
| URL-leak | 0% | 0% | 0 |
| N judged | 30 | 27 | -3 |

- Effect is modest but real: the drop is driven by ungrounded external-knowledge
  drift, not fabrication of evidence facts. Clearest case `gen_10_nvidia_vs_amd`
  (5->3): the model injected outside knowledge / subjective inference absent from
  the evidence (SEV-SNP market positioning, 10k-GPU-cluster pain points, forcing
  a link between the QBE compiler backend and "breaking the CUDA monopoly").
- No severe hallucination (faith<=2 stays 0%): the retrieval + evidence layer
  supplies accurate, on-topic context, so even an unconstrained synthesizer
  mostly stays grounded.

**Layer 2 — deterministic output_guard (observed in the first weak run, which removed the URL-citation requirement too):**

- With citation removed, **27/30 (90%) answers were blocked** by `output_guard`
  ("无证据结论已阻断" / ungrounded-conclusion block) — they never reach the user.
- Even with the URL requirement restored (this run), 3/30 were still blocked when
  the synthesizer failed to cite a valid evidence URL.

**Takeaway:** the prompt contributes ~+0.22 faithfulness and eliminates
external-knowledge drift, but the hard guarantee is the deterministic
output_guard: answers without a valid in-body source citation are blocked
outright. Faithfulness is enforced by prompt-induced grounding *and* code-level
gating, not a single point.

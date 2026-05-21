# G3 Generation Faithfulness Report

Generated: 2026-05-20T15:54:17.848532+00:00

Cases judged: **30**

| Metric | Value |
|---|---|
| Faithfulness (avg, 1-5) | **3.67** |
| Answer relevancy (avg, 1-5) | **5.00** |
| Well-grounded rate (faithfulness>=4) | 53.3% |
| Hallucination rate (faithfulness<=2) | 20.0% |
| URL-leak rate (>=1 url outside evidence) | 0.0% |
| Total leaked URLs | 0 |

## Lowest-faithfulness cases

- `gen_10_nvidia_vs_amd` faithfulness=2 relevancy=5 leaks=0 — The answer directly addresses the user's question about comparing NVIDIA and AMD, making it highly relevant. However, it fabricates several 
- `gen_11_chip_landscape` faithfulness=2 relevancy=5 leaks=0 — The answer directly addresses the user's question about the competitive landscape of AI chips, making it highly relevant. However, it includ
- `gen_15_opensource` faithfulness=2 relevancy=5 leaks=0 — The answer is highly relevant to the user's question about recent open-source large model releases, correctly identifying Qwen3.6-27B and Ki
- `gen_23_anthropic_stainless` faithfulness=2 relevancy=5 leaks=0 — The answer directly addresses the user's question about Anthropic's acquisition of Stainless, making it highly relevant. However, it include
- `gen_25_anthropic_gates` faithfulness=2 relevancy=5 leaks=0 — The answer directly addresses the question, earning a high relevancy score. However, it includes numerous specific details not present in th
- `gen_27_sandboxaq` faithfulness=2 relevancy=5 leaks=0 — The answer directly addresses the user's question, earning a high relevancy score. However, it includes several specific details and claims 
- `gen_07_deepseek` faithfulness=3 relevancy=5 leaks=0 — The answer directly addresses the question by detailing DeepSeek's recent releases and related news. However, it hallucinates specific param
- `gen_09_openai_vs_anthropic` faithfulness=3 relevancy=5 leaks=0 — The answer effectively addresses the prompt by comparing the corporate strategies of OpenAI and Anthropic using the provided evidence. Howev

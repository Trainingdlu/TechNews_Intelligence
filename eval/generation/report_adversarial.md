# G3 Generation Faithfulness Report

Generated: 2026-05-22T14:44:31.119177+00:00

Cases judged: **46**

| Metric | Value |
|---|---|
| Faithfulness (avg, 1-5) | **4.87** |
| Answer relevancy (avg, 1-5) | **5.00** |
| Well-grounded rate (faithfulness>=4) | 95.7% |
| Hallucination rate (faithfulness<=2) | 0.0% |
| URL-leak rate (>=1 url outside evidence) | 0.0% |
| Total leaked URLs | 0 |

## Lowest-faithfulness cases

- `adv_01_claude_params_cost` faithfulness=3 relevancy=5 leaks=0 — The answer directly addresses the user's question by correctly stating that the requested information (parameter count and training cost) is
- `adv_16_openai_msft_split` faithfulness=3 relevancy=5 leaks=0 — The answer correctly addresses the user's question by stating the information is not available in the provided evidence and summarizing what
- `adv_13_grok_params` faithfulness=4 relevancy=5 leaks=0 — The agent correctly identifies that the requested information is not in the evidence and accurately summarizes the available information abo
- `adv_24_xai_openai_merger` faithfulness=4 relevancy=5 leaks=0 — The answer correctly addresses the user's question by stating that there is no evidence of a merger in the provided text, and accurately sum
- `adv_02_false_acquisition` faithfulness=5 relevancy=5 leaks=0 — The agent correctly identifies that the premise of the user's question (OpenAI acquiring Anthropic) is false based on the provided evidence.
- `adv_03_customer_list` faithfulness=5 relevancy=5 leaks=0 — The agent correctly states that a complete list cannot be provided based on the evidence, and accurately extracts the available data regardi
- `adv_04_deepseek_v5` faithfulness=5 relevancy=5 leaks=0 — The agent correctly identifies that there is no information about DeepSeek V5 in the provided evidence. It then helpfully provides the featu
- `adv_05_huang_quotes` faithfulness=5 relevancy=5 leaks=0 — The agent correctly identifies that the requested information (quotes from the latest earnings call) is not available in the provided eviden

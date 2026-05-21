# G3 Generation Faithfulness Report

Generated: 2026-05-21T02:21:47.247994+00:00

Cases judged: **11**

| Metric | Value |
|---|---|
| Faithfulness (avg, 1-5) | **4.82** |
| Answer relevancy (avg, 1-5) | **5.00** |
| Well-grounded rate (faithfulness>=4) | 90.9% |
| Hallucination rate (faithfulness<=2) | 0.0% |
| URL-leak rate (>=1 url outside evidence) | 0.0% |
| Total leaked URLs | 0 |

## Lowest-faithfulness cases

- `adv_01_claude_params_cost` faithfulness=3 relevancy=5 leaks=0 — The answer directly addresses the user's question by correctly stating that the requested information (parameter count and training cost) is
- `adv_02_false_acquisition` faithfulness=5 relevancy=5 leaks=0 — The agent correctly identifies that the premise of the user's question (OpenAI acquiring Anthropic) is false based on the provided evidence.
- `adv_03_customer_list` faithfulness=5 relevancy=5 leaks=0 — The agent correctly states that a complete list cannot be provided based on the evidence, and accurately extracts the available data regardi
- `adv_04_deepseek_v5` faithfulness=5 relevancy=5 leaks=0 — The agent correctly identifies that there is no information about DeepSeek V5 in the provided evidence. It then helpfully provides the featu
- `adv_05_huang_quotes` faithfulness=5 relevancy=5 leaks=0 — The agent correctly identifies that the requested information (quotes from the latest earnings call) is not available in the provided eviden
- `adv_06_cohere_funding` faithfulness=5 relevancy=5 leaks=0 — The answer accurately extracts the funding amount ($600 million / €500 million), valuation ($20 billion), and lead investor (Schwarz Group) 
- `adv_07_gpt56_date_price` faithfulness=5 relevancy=5 leaks=0 — The agent correctly identifies that there is no information about GPT-5.6 in the provided evidence. It then provides relevant context about 
- `adv_09_mistral_cohere_params` faithfulness=5 relevancy=5 leaks=0 — The answer correctly states that the provided evidence does not contain the parameter counts for Mistral and Cohere's latest models. It accu

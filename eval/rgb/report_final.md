# G6 RGB negative-rejection · 加固提示词 final 报告

## 结论

RGB negative-rejection 子集只喂"含干扰、不含答案"的文档(top-5 negative),压测 synthesizer 会不会把证据外内容**冒充成"根据证据"**(误归因 = 真·幻觉,唯一的失败模式)。在 final-synthesis 节点加入 Grounding & Refusal 硬规则后,deepseek 在**同一 299 题**上:

- 误归因/伪造率 **21.7% → 11.4%**(近乎砍半,误归因样本 65 → 34)
- 配对(同题)看:**49 题**从"冒充证据"变干净、**18 题**新引入误归因,**净 −31**

该提示词已接入生产(`agent/graph/prompts.py::_FINAL_SYSTEM_PROMPT`)。RGB 去答案泄漏后样本固定为 299/300,各 run 天然同题,可直接对比。

## 加固 before→after(deepseek,同 299 题)

| 指标 | baseline prompt | hardened prompt | Δ |
|---|---|---|---|
| 误归因/伪造率(LLM judge) | 21.7% | **11.4%** | −10.3pp |
| 误归因样本数 | 65 | 34 | −31 |
| 硬拒答率(确定性,gold 未出现) | 73.9% | 76.6% | +2.7pp |
| 回答率(含标注假设) | 57.9% | 44.8% | −13.1pp |

**配对翻转(同题逐条对比):** 修好 49 题(基线冒充证据 → 加固干净)、新引入 18 题(基线干净 → 加固冒充)、净 −31。与 generation 一致,加固不是免费午餐——硬规则让模型更保守,回答率从 57.9% 降到 44.8%,换来误归因近乎砍半。

## 三种行为的口径

agent 被喂"只含干扰、不含答案"的文档时只会做三件事,只有第 3 种是失败:

1. **拒答**("证据不足")—— 安全。
2. **标注假设补充**("证据不足,但据我所知…(非证据)")—— 有用且诚实,不算幻觉。
3. **冒充出处**("根据证据…",而证据并没说)—— 误归因 = 真·幻觉。

加固把更多 case 从第 3 种推向第 1/2 种,代价是整体回答率下降。

## 模型维度(迁 deepseek 前 gemini 期参照,非加固臂)

| 运行 | 推断模型 / prompt | 误归因率 | 回答率 | 硬拒答率 | 单 run 报告 |
|---|---|---|---|---|---|
| rgb_rejection | gemini / 基线 | 8.4% | 53.2% | 74.9% | `report.md` |
| after_prompt | gemini / prompt 改进 | 5.0% | 47.8% | 80.3% | `runs/rgb_rejection_after_prompt.md` |
| rgb_deepseek | deepseek / 基线 | 21.7% | 57.9% | 73.9% | `runs/rgb_deepseek.md` |
| rgb_deepseek_hardened | deepseek / 加固 | 11.4% | 44.8% | 76.6% | `runs/rgb_deepseek_hardened.md` |

- gemini 期两个 run 是**迁 deepseek 前的生产模型**结果,仅作跨模型量级参照,不属于加固 before→after。
- deepseek 加固后(11.4%)**仍高于** gemini(5–8.4%):换 deepseek 省成本但牺牲拒答质量,加固是**缓解而非追平**。这是迁移的已知权衡,不夸大。

## 指标口径

- 数据:RGB 中文 negative-rejection 子集,每条只喂 `negative`(不含答案)文档 top-5,经**生产接地 prompt** 合成答案。已剔除答案泄漏进所喂文档的 case(中文/阿拉伯数字归一后),剩 299/300。
- `misattributed`(LLM judge):把证据外内容冒充成"证据支撑"= 真·幻觉;标了"假设/外部知识"的**不算**。这是核心失败指标。
- `gold_present`(确定性字符串匹配):标准答案字串原样出现在输出里。文档不含答案 → 出现 = 模型自供。硬拒答率含少量"答对但字串未逐字命中",略偏高,不是主指标。
- `answered`(LLM judge):给了实质性回答(哪怕先说"证据不足"再补充)。

## 复现

- 生产提示词:`agent/graph/prompts.py::_FINAL_SYSTEM_PROMPT`(内容 = `eval/synth_prompt_final_hardened.txt`)。RGB 用自带 `data/zh.json`,不触新闻库,无需暂停 n8n。
- deepseek 基线:
  ```
  python eval/rgb/run_rgb_rejection.py --provider deepseek --limit 0 --output eval/rgb/runs/rgb_deepseek.jsonl --report eval/rgb/runs/rgb_deepseek.md
  ```
- deepseek 加固(`--synth-prompt-file` 覆盖 final 节点提示词):
  ```
  python eval/rgb/run_rgb_rejection.py --provider deepseek --limit 0 --synth-prompt-file eval/synth_prompt_final_hardened.txt --output eval/rgb/runs/rgb_deepseek_hardened.jsonl --report eval/rgb/runs/rgb_deepseek_hardened.md
  ```
  judge 可加 `--judge-provider <独立 provider>` 避免自评。
- 判分产物:`runs/rgb_deepseek.jsonl`(基线)、`runs/rgb_deepseek_hardened.jsonl`(加固);误归因/补充式样本明细见对应单 run 报告的附录 C / D。

## 限制

- run 的 jsonl 不记录模型/prompt 元数据,模型归属按**文件名约定**(`*_deepseek*` 为显式 deepseek run;无 deepseek 后缀的两个 run 为迁 deepseek 前 gemini 期)。
- 加固新引入的 18 个误归因值得排查;明细在 `runs/rgb_deepseek_hardened.md` 附录 C。
- 硬拒答率与回答率是行为权衡量,非质量上限;判读以误归因率为主。

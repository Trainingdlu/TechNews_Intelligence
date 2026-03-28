# Agent 测试说明文档

## 1. 文档目标
本文档用于规范 Agent 的测试与评估流程，确保以下能力可量化、可回归、可门禁：
- 路由与工具调用可靠性
- 证据驱动输出质量
- 重复问答稳定性
- 证据不足时的安全降级行为

适用场景：本地开发验证、CI 合并门禁、版本发布验收。

---

## 2. 测试范围
### 2.1 范围内
- `langchain` 主运行时与 fallback 行为
- compare / timeline / landscape 的强制路由
- landscape 低证据自动降级逻辑
- Telegram Bot 输出稳健性（来源引用重建、发送重试、限流与回退）
- `run_eval.py` 报告与质量门禁

### 2.2 范围外
- 前端页面渲染测试
- n8n 上游采集链路的数据质量细节

---

## 3. 测试策略（分层）
1. 单元测试：验证核心逻辑正确性（快速反馈）。
2. 稳定性评测：真实问题重复执行，统计输出稳定性。
3. 质量门禁：阈值化通过/失败，接入 CI。
4. 人工抽检：高风险问题类型做语义质量复核。

---

## 4. 环境参数与配置

### 4.1 运行时参数（`.env` / `docker-compose`）

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `AGENT_RUNTIME` | `langchain` | 运行模式：`langchain` / `legacy` |
| `AGENT_RUNTIME_STRICT` | `false` | `true`：错误直接失败；`false`：允许 fallback |
| `AGENT_MAX_ITERATIONS` | `16` | LangGraph 工具递归上限 |
| `AGENT_ROUTE_METRICS` | `true` | 是否启用路由指标统计 |
| `AGENT_ROUTE_LOG_EVERY` | `20` | 每 N 次请求打印一次路由日志 |
| `TIMELINE_MIN_EVENTS` | `5` | 时间线分析最小样本阈值 |
| `LANDSCAPE_MIN_URLS` | `4` | 格局分析最小 URL 证据阈值 |
| `LANDSCAPE_MIN_MATCHED_ARTICLES` | `6` | 格局分析最小命中文章数阈值 |
| `LANDSCAPE_MIN_ACTIVE_ENTITIES` | `2` | 格局分析最小活跃实体数阈值 |
| `BOT_SEND_RETRY_ATTEMPTS` | `2` | Telegram 发送失败最大重试次数 |
| `BOT_SEND_RETRY_BASE_SEC` | `0.8` | Telegram 重试指数退避基准秒数 |
| `BOT_MAX_CITATION_URLS` | `12` | 单条回答最多重建为来源引用的 URL 数 |

### 4.2 Eval 参数（`agents/eval/run_eval.py`）

| 参数 | 说明 |
|---|---|
| `--dataset` | JSONL 题库路径 |
| `--runs-per-question` | 每题重复运行次数 |
| `--sleep-seconds` | 重复运行间隔 |
| `--include-outputs` | 报告中包含完整回答 |
| `--baseline` | 基线报告路径（回归对比） |
| `--fail-on-*` | 门禁阈值参数 |

新增门禁参数：
- `--fail-on-landscape-low-evidence-rate`

### 4.3 测试文件职责矩阵

| 测试文件 | 主要测试内容 | 在系统中的作用 | 失败时通常意味着什么 | 建议排查方向 |
|---|---|---|---|---|
| `agents/tests/test_eval_core.py` | `eval_core.py` 的纯函数逻辑：文本归一化、URL 提取去重、稳定性指标聚合、质量门禁判定 | 保证评测指标与门禁结论可信，避免“评测本身有 bug” | 指标计算回归、门禁阈值判断错误、报告统计不可信 | 先看 `eval/eval_core.py` 是否改动，再核对该测试中对应的输入/期望 |
| `agents/tests/test_agent_route_metrics.py` | `agent.py` 路由与指标快照：LangChain 成功/回退计数、compare/timeline/landscape 强制路由、低证据降级分支 | 保证“走哪条链路”和“统计出来的成功率/回退率”正确 | 路由逻辑被改坏，或 metrics 字段定义变化导致监控失真 | 检查 `agent.py` 的 `_extract_*`、`generate_response`、`_metrics_inc` 与 snapshot 结构 |
| `agents/tests/test_bot_robustness.py` | `bot.py` 的输出稳健性：URL 处理、`[1][2]` 引用重建、“来源”段重建、HTML 回退、发送重试、限流 `retry_after` | 保证 Telegram 用户可见行为稳定，避免格式错乱/发送失败体验差 | Bot 消息后处理链路回归，或重试/回退行为失效 | 检查 `bot.py` 的 `_send_reply`、`_format_for_telegram`、`_reply_text_with_retry`、`_consume_chat_rate_token` |

说明：
- 以上测试均属于“快速单元测试”，默认 CI 会全部执行。
- 这些测试不依赖真实模型调用，适合频繁迭代阶段做高频回归。

---

## 5. 标准测试流程

### 5.1 单元测试
```bash
python -m unittest discover -s agents/tests -p "test_*.py" -v
```

通过标准：
- 所有用例通过
- 无语法错误 / 导入错误

Bot 关键链路可单独快速回归：
```bash
python -m unittest agents.tests.test_bot_robustness -v
```

该用例覆盖：
- URL 提取去重与标点裁剪
- 正文 `[1][2]` 引用重建
- “来源”段落重建与旧段清理
- HTML 发送失败回退纯文本
- Telegram 发送自动重试
- 频控拦截与 `retry_after` 计算

### 5.2 稳定性评测
```bash
python agents/eval/run_eval.py \
  --dataset agents/eval/questions_default.jsonl \
  --runs-per-question 3 \
  --include-outputs \
  --output agents/eval/reports/latest.json
```

产物字段：
- `summary`
- `route_metrics`
- `cases`
- `quality_gate`

### 5.3 CI 门禁评测（全量，可选）
```bash
python agents/eval/run_eval.py \
  --runs-per-question 3 \
  --fail-on-avg-error-rate 0.05 \
  --fail-on-fallback-rate-total 0.05 \
  --fail-on-avg-min-url-hit-rate 0.85 \
  --fail-on-langchain-success-rate 0.90 \
  --fail-on-landscape-low-evidence-rate 0.40
```

返回码约定：
- `0`：全部门禁通过
- 非 `0`：存在门禁失败

### 5.4 GitHub Actions CI（默认）
CI workflow 文件：`.github/workflows/ci.yml`

执行顺序：
1. 安装 `agents/requirements.txt`
2. 运行单测：`python -m unittest discover -s agents/tests -p "test_*.py" -v`
 
说明：
- 默认 CI **不调用模型**，用于频繁迭代阶段快速反馈。
- Bot 稳健性测试（`test_bot_robustness.py`）已包含在默认单测中。

### 5.5 GitHub Actions 手动 Eval（调用模型）
手动 workflow 文件：`.github/workflows/eval-manual.yml`

触发方式：
1. 打开 GitHub Actions
2. 选择 `Agent Eval (Manual)`
3. 点击 `Run workflow`（可选输入 `max_cases`、`runs_per_question`）

该 workflow 执行命令：
```bash
python agents/eval/run_eval.py \
  --dataset agents/eval/questions_default.jsonl \
  --max-cases <max_cases> \
  --runs-per-question <runs_per_question> \
  --output agents/eval/reports/ci_smoke.json
```
并上传产物：`agents/eval/reports/ci_smoke.json`

发布前建议：
- 手动 smoke 通过后，再执行 5.3 的全量门禁评测。

---

## 6. 指标体系与评估标准

### 6.1 核心指标定义
- `summary.avg_error_rate`
  - 越低越好
  - 反映执行层面异常率

- `summary.avg_min_url_hit_rate`
  - 越高越好
  - 反映是否达到每题最小证据 URL 约束

- `summary.avg_pairwise_similarity`
  - 用于观察重复问答稳定性
  - 过低：漂移大；过高：可能模板化

- `route_metrics.fallback_rate_total`
  - 越低越好
  - 主运行时失败回退比例

- `route_metrics.langchain_success_rate`
  - 越高越好
  - LangChain 主路径成功率

- `route_metrics.landscape_low_evidence_rate`
  - 越低越好
  - 在格局类请求中触发“证据不足降级”的比例

### 6.2 推荐阈值（基线）
- `avg_error_rate <= 0.05`
- `fallback_rate_total <= 0.05`
- `langchain_success_rate >= 0.90`
- `avg_min_url_hit_rate >= 0.85`
- `landscape_low_evidence_rate <= 0.40`

### 6.3 评估
- **可发布**：关键门禁全部通过，且人工抽检无严重问题。
- **有条件发布**：非关键指标轻微超线，有明确风险记录与跟踪人。
- **阻塞发布**：关键门禁失败，或出现明显幻觉/证据错配回归。

---

## 7. 格局类问题专项评估

### 7.1 专项目标
格局问题（AI/商业/安全）是高风险场景，必须避免模型凭先验“补全故事”。

### 7.2 预期行为
当证据不足时，系统应：
- 返回事实快照（raw landscape）
- 明确声明证据不足
- 给出低置信度
- 不输出强角色推断

### 7.3 结果判读
`landscape_low_evidence_rate` 升高通常意味着：
1. 主题覆盖不足（数据库样本稀薄）
2. 实体别名不全（识别漏召回）
3. 阈值设置过严（策略问题）

建议排查顺序：
1. 检查 topic/entity 匹配命中
2. 校准阈值（优先保守）
3. 扩充题库覆盖空白领域

---

## 8. 超时与长耗时评测处理
`run_eval.py` 会真实调用模型，出现超时是可能的。

重要原则：
- **超时不会影响已完成的代码改动有效性**。
- 但该次评测没有完整结束，**不能把该结果作为发布证据**。

超时后的处理建议：
1. 先跑 smoke：减少题量和重复次数
2. 再跑全量：在 CI / 夜间任务中给更长超时预算
3. 不使用部分结果做发布签字

smoke 示例：
```bash
python agents/eval/run_eval.py --max-cases 3 --runs-per-question 1 --output agents/eval/reports/smoke.json
```

---

## 9. 人工抽检规范
建议抽检 5-10 条回答，优先：
- 格局类（AI / 商业 / 安全）
- 对比类
- 时间线类

检查项：
- 关键结论是否有 URL 证据支撑
- 置信度与证据强度是否匹配
- 证据不足时是否正确降级
- 是否出现虚构事件/公司/URL

---

## 10. 建议发布前作业顺序
1. 跑单测
2. 跑 smoke eval
3. 跑全量 eval + gate
4. 与 baseline 对比
5. 做人工抽检
6. 在 PR/发布说明记录门禁结果与风险结论

---

## 11. 维护与更新规则
- 文档负责人：Agent 运行时维护者
- 触发更新条件：
  - 新增强制路由
  - 新增/调整门禁指标
  - 阈值策略调整
- 需与以下代码保持一致：
  - `agents/agent.py`
  - `agents/eval/run_eval.py`
  - `agents/eval/eval_core.py`

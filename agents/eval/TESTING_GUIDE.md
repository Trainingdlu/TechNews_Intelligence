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

---

## 5. 标准测试流程

### 5.1 单元测试
```bash
python -m unittest discover -s agents/tests -p "test_*.py" -v
```

通过标准：
- 所有用例通过
- 无语法错误 / 导入错误

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

### 5.3 CI 门禁评测
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

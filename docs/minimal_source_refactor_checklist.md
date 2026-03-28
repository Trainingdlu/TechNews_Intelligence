# 官方发布 + 学术原始信号最小改造清单

## 1. 结论（先看这个）
- 推荐方案：**直连原始源（官方站点 + arXiv/OpenReview）+ 小解耦**。
- 不推荐：继续只依赖 `HackerNews + TechCrunch` 转引。
- 原因：当前样本（`deployment/news_data.sql`，1534 条）中：
  - 官方信号仅约 **0.85%**（13 条）
  - 学术信号仅约 **0.98%**（15 条）
  - 且几乎都来自 HN 转引，时效和稳定性受社区热度影响。

---

## 2. 目标与约束
- 目标：提升“官方发布 / 学术原始信号”覆盖率与时效性。
- 约束：不做大重构，保持现有看板、Agent、日报兼容。

---

## 3. 改造范围（最小可行）

### Phase A：数据层小解耦（0.5 天）
- 在 `tech_news` 增加显式来源维度（向后兼容）：
  - `source_platform`（采集平台：HackerNews/TechCrunch/DirectRSS/DirectAPI）
  - `signal_origin`（信号属性：Official/Academic/Media/Community/Other）
  - `source_name`（具体来源名，如 OpenAI Blog、arXiv）
- 更新 `view_dashboard_news`：
  - `source_type` 继续保留（兼容旧 SQL）
  - 新增 `signal_origin`、`source_name`、`source_domain`
  - 当新列为空时，回退现有 URL 推断逻辑

**验收**
- 旧看板 SQL 不报错。
- 新增列可用于筛选并返回结果。

### Phase B：来源注册表（0.5 天）
- 新建 `source_registry` 表（可选 JSON 配置，建议表）：
  - `source_key`、`source_name`、`source_platform`、`signal_origin`
  - `fetch_type`（rss/api）
  - `endpoint`、`is_active`、`priority`
- 预置首批源：
  - 官方：OpenAI、Anthropic、Google/DeepMind、Apple Newsroom、AWS Blog、Cloudflare Blog
  - 学术：arXiv（主）、OpenReview（辅）

**验收**
- 可以通过 SQL 一次查出“当前启用源”。
- 不改主流程时，系统仍正常运行。

### Phase C：采集接入（1.0 天）
- 在 n8n 主采集流增加“直连源分支”：
  - RSS 类：统一到标准字段 `title/url/created_at/source_*`
  - arXiv/OpenReview：先走 RSS/Atom，避免首版 API 复杂化
- 入库时写入 `source_platform/signal_origin/source_name`。
- 保留现有 HN/TC 分支，不做替换（先增量验证）。

**验收**
- 24 小时内可稳定落库官方/学术条目。
- 失败入队（`tech_news_failed`）可区分来源。

### Phase D：消费层最小适配（0.5 天）
- Agent：
  - `query_news` 增加 `signal_origin` 过滤（Official/Academic）
  - 保留旧参数（HackerNews/TechCrunch）兼容
- 报表：
  - 新增 1-2 张卡片（官方发布趋势、学术论文趋势）
  - 原有卡片保持不变

**验收**
- 问句“近7天官方发布/学术论文”可被稳定检索。
- 旧问句行为不回归。

---

## 4. 投入产出评估（ROI）
- 预计投入：**2.5 天左右**（不含 UI 美化）
- 直接收益：
  - 原始信号覆盖显著提升（摆脱“等社区转引”）
  - 时效性提升（官方发布和论文上线即入库）
  - 后续加源成本下降（按注册表新增，不再改多处硬编码）
- 风险：
  - n8n 流程节点增加，初期故障点变多
  - 需补充来源健康监控（建议对每源加近 24h 入库计数）

---

## 5. 建议优先顺序（按收益最大化）
1. 先做 Phase A + B（不动采集主链，风险最低）
2. 再接入 arXiv + OpenAI/Anthropic（最能体现价值）
3. 运行 3-7 天观察覆盖与失败率
4. 再扩 Google/DeepMind、Apple、AWS、OpenReview

---

## 6. 本次改造的完成标准（DoD）
- 官方 + 学术条目占比从当前约 1%-2% 提升到 **>= 10%**（首周目标）。
- `source_type` 兼容旧看板；新增 `signal_origin` 可用于筛选与分析。
- 新增源不需要改动 5 个以上文件（通过注册表配置即可完成）。


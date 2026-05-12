# 混合检索与实体别名运维说明

本文只覆盖部署和离线维护，不影响实时查询路径。

## 一次性数据迁移

有需要做数据迁移。代码里已经提供 DDL 和 seed，但写进仓库不等于已经执行到真实 PostgreSQL。

首次部署或升级时执行：

```bash
bash deployment/scripts/db/apply_schema.sh
bash deployment/scripts/db/apply_seed.sh
```

这个脚本会执行：

- `sql/infrastructure/schema/schema_ddl.sql`
- `sql/infrastructure/views/view_dashboard_news.sql`
- `sql/infrastructure/seeds/seed_source_*.sql`
- `sql/infrastructure/seeds/seed_entity_*.sql`

本次新增的迁移内容包括：

- `pg_trgm` extension。
- `news_search_index` 表、GIN 索引、`tech_news` 触发器和历史 backfill。
- `entity_registry`、`entity_alias`、`entity_alias_candidate`、`news_entity_mentions`。
- 核心 tech 实体种子和别名种子。
- `view_dashboard_news` 暴露 `search_tsv` 和 `search_index_updated_at`。

日常定时任务默认不重复跑 schema；需要在同一个命令里先确保 schema/view 时，显式加 `--apply-schema`。

## 实体种子

核心实体种子位于：

```text
sql/infrastructure/seeds/seed_entity_core.sql
```

种子是幂等的，可以重复执行。`seed_entity_core.sql` 同时包含基础稳定集和扩展覆盖集，覆盖核心公司、云平台、模型、芯片、半导体供应链和安全主题。

中文别名以 UTF-8 写入，例如：

- `谷歌`
- `微软`
- `英伟达`
- `台积电`
- `英特尔`
- `甲骨文`
- `阿斯麦`
- `高通`
- `博通`
- `三星`

不要用终端显示结果判断是否乱码；Windows PowerShell 的默认编码可能把 UTF-8 显示成乱码。以 UTF-8 读取文件或用编辑器确认。

## 离线自动化闭环

当前最便捷的方式是“宿主机定时任务调用维护脚本”，而不是在实时查询里调用 DeepSeek，也不是新加一个常驻 worker。

推荐命令：

```bash
bash deployment/scripts/db/build_entity_alias_candidates.sh --days 14 --limit 1000 --use-deepseek
```

这个脚本做两件事：

1. 扫描近 N 天新闻，抽取实体别名候选，必要时调用 DeepSeek 做归并和歧义判断，写入 `entity_alias_candidate`。
2. 把之前已人工标记为 `approved` 的候选提升到正式 `entity_registry` / `entity_alias`。

默认不会把 `auto_approved` 直接提升到正式别名表。原因是 `Apple`、`Meta`、`Gemini`、`Claude`、`Cursor` 这类词会影响多个分析工具的统计口径，正式入库前仍应有人工确认。

如果你明确接受自动提升高置信候选，可以使用：

```bash
bash deployment/scripts/db/build_entity_alias_candidates.sh --days 14 --limit 1000 --use-deepseek --include-auto-approved-promotion
```

人工审核可以直接更新候选表：

```sql
UPDATE entity_alias_candidate
SET status = 'approved', reviewed_at = NOW(), updated_at = NOW()
WHERE candidate_id = 123;

UPDATE entity_alias_candidate
SET status = 'rejected', reviewed_at = NOW(), updated_at = NOW()
WHERE candidate_id = 124;
```

被标记为 `approved`、`rejected`，或已经 `promoted_at IS NOT NULL` 的候选，后续扫描不会被新的 DeepSeek 判定覆盖。

## 定时方式选择

最省事的生产方式：

- Linux/VPS：`cron` 每天低峰期跑一次。
- Windows：Task Scheduler 每天跑一次。
- 已经用 n8n 做统一运维时：用 n8n Schedule Trigger 调宿主机命令，推荐通过 SSH 到宿主机执行上面的脚本。

不建议在当前 n8n 容器里直接跑 Python 脚本。当前 compose 没有把仓库和 Python 依赖挂进 n8n 容器；为了这一件事改 n8n 镜像或挂 Docker socket，维护面更大。

## 安全默认值

- DeepSeek 只用于离线候选归并，不阻塞用户查询。
- `points` 默认不参与排序 boost：`RETRIEVAL_POINTS_BOOST=off`。
- 日常任务默认只提升人工 `approved` 的候选。
- `--dry-run` 可用于先看候选，不写数据库。

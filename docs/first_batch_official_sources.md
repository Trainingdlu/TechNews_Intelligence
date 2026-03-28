# 第一批官方来源接入（Google / AWS / Microsoft）

本批次只做 `source_registry` 增量注册，不改现有 ETL 主流程，且可重复执行（幂等）。

## 1) 执行脚本（推荐）

在仓库根目录执行：

```bash
chmod +x deployment/apply_source_framework_migration.sh
bash deployment/apply_source_framework_migration.sh
```

该合并脚本会一次完成：
- `source_framework_migration.sql`
- `view_logic.sql`
- `seed_source_batch1_official.sql`

## 1.1) 仅补第一批来源（可选）

如果结构迁移之前已经跑过，只想单独补第一批来源，可执行：

```bash
chmod +x deployment/apply_first_batch_sources.sh
bash deployment/apply_first_batch_sources.sh
```

## 2) 本次新增来源

- `google_ai_blog` -> `https://blog.google/feed/`
- `aws_ml_blog` -> `https://aws.amazon.com/blogs/machine-learning/feed/`
- `microsoft_ai_blog` -> `https://blogs.microsoft.com/?feed=rss2`

> 说明：Google/Microsoft 在 `extra_config` 中预留了关键词过滤和备用 feed，用于后续 ETL 动态拉取时做细粒度筛选。

## 3) 校验 SQL

```sql
SELECT source_key, source_name, signal_origin, fetch_type, endpoint, is_active, priority
FROM source_registry
WHERE source_key IN ('google_ai_blog', 'aws_ml_blog', 'microsoft_ai_blog')
ORDER BY priority;
```

## 4) 典型结果（预期）

- 返回 3 行
- `signal_origin = Official`
- `fetch_type = rss`
- `is_active = true`

# 第一批官方来源（Google / AWS / Microsoft）

本批次以 `source_registry` 注册为主，不直接改动 ETL 主流程。

## 推荐执行方式

执行一次迁移主脚本（含默认种子）：

```bash
bash deployment/apply_source_framework_migration.sh
```

该命令会自动执行：

- `source_framework_migration.sql`
- `view_logic.sql`
- `seed_source_batch1_official.sql`
- `seed_source_batch2_nvidia.sql`（若存在）

## 仅执行第一批官方种子（可选）

```bash
bash deployment/apply_source_framework_migration.sh \
  --skip-seeds \
  --seed-file sql/infrastructure/seed_source_batch1_official.sql
```

## 第一批来源列表

- `google_ai_blog` -> `https://blog.google/rss/`
- `aws_ml_blog` -> `https://aws.amazon.com/blogs/machine-learning/feed/`
- `microsoft_ai_blog` -> `https://blogs.microsoft.com/?feed=rss2`

## 校验 SQL

```sql
SELECT
  source_key,
  source_name,
  signal_origin,
  fetch_type,
  endpoint,
  is_active,
  priority
FROM source_registry
WHERE source_key IN ('google_ai_blog', 'aws_ml_blog', 'microsoft_ai_blog')
ORDER BY priority;
```

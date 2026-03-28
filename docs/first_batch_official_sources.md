# 官方来源种子（Google / AWS / Microsoft / NVIDIA）

本批次仅负责向 `source_registry` 注册来源，不直接改 ETL 主流程。

## 推荐执行

```bash
bash deployment/scripts/db/apply_source_framework_migration.sh
```

该命令会执行：

1. `sql/infrastructure/schema/schema_ddl.sql`
2. `sql/infrastructure/views/view_dashboard_news.sql`
3. `sql/infrastructure/seeds/seed_source_*.sql`

## 仅执行官方种子（可选）

```bash
bash deployment/scripts/db/apply_source_framework_migration.sh \
  --seed-file sql/infrastructure/seeds/seed_source_official.sql
```

## 本批来源

- `google_ai_blog` -> `https://blog.google/rss/`
- `aws_ml_blog` -> `https://aws.amazon.com/blogs/machine-learning/feed/`
- `microsoft_ai_blog` -> `https://blogs.microsoft.com/?feed=rss2`
- `nvidia_dev_blog` -> `https://developer.nvidia.com/blog/feed/`

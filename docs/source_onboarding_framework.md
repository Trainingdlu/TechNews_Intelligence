# 来源解耦接入框架

目标：在不破坏现有流程的前提下，实现“填入来源参数即可接入”，并保持幂等执行。

## 当前结构

- 来源注册层：`public.source_registry`
- 新闻主表扩展字段：`tech_news.source_key/source_name/source_platform/signal_origin/source_meta`
- 失败队列表扩展字段：`tech_news_failed.source_*`
- 视图兼容层：`public.view_dashboard_news`（优先新字段，缺失时回退旧逻辑）
- API 订阅来源：动态读取 `source_registry` 的启用来源

## 一键迁移（表结构 + 视图）

```bash
bash deployment/apply_source_framework_migration.sh
```

默认行为：

1. 执行结构迁移 SQL
2. 刷新仪表盘视图 SQL
3. 自动按文件名顺序执行 `sql/infrastructure/seed_source_*.sql`

只做结构，不跑种子：

```bash
bash deployment/apply_source_framework_migration.sh --skip-seeds
```

只跑指定种子：

```bash
bash deployment/apply_source_framework_migration.sh \
  --skip-seeds \
  --seed-file sql/infrastructure/seed_source_batch1_official.sql
```

## 一键新增/更新来源（推荐）

```bash
bash deployment/upsert_source.sh \
  --source-key openai_blog \
  --source-name "OpenAI Blog" \
  --endpoint "https://openai.com/news/rss.xml" \
  --signal-origin Official \
  --source-platform DirectRSS \
  --fetch-type rss \
  --priority 30 \
  --is-active true \
  --extra-config '{"company":"OpenAI","keyword_filters":["openai","gpt","api"]}'
```

说明：

- 默认会先自动确保迁移结构存在（相当于先跑一遍 `--skip-seeds`）
- 然后执行 `source_registry` 的 `INSERT ... ON CONFLICT DO UPDATE`
- 全流程幂等，可重复执行

## 相关脚本

- 迁移主脚本：[apply_source_framework_migration.sh](/C:/Users/chenxl007/Desktop/TechNews_Intelligence/deployment/apply_source_framework_migration.sh)
- 新增来源脚本：[upsert_source.sh](/C:/Users/chenxl007/Desktop/TechNews_Intelligence/deployment/upsert_source.sh)
- 公共 DB 工具库：[db_common.sh](/C:/Users/chenxl007/Desktop/TechNews_Intelligence/deployment/db_common.sh)

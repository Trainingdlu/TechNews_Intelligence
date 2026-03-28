# 来源可填充接入框架（兼容现有 HN/TC）

## 目标
- 在不破坏现有运行链路的前提下，实现“填一条来源配置即可接入”。
- 现有 `HackerNews` / `TechCrunch` 无缝纳入同一框架。

## 已落地的框架层
- 来源注册层：`public.source_registry`
- 兼容数据层：`tech_news` 增加 `source_key/source_name/source_platform/signal_origin/source_meta`
- 兼容视图层：`view_dashboard_news` 优先使用新来源字段，缺失时回退旧逻辑（URL/source_id 推断）
- API 动态来源层：订阅来源不再写死，改为读取 `source_registry` 的启用项

## 先做一次迁移
推荐直接使用一键脚本：

```bash
chmod +x deployment/apply_source_framework_migration.sh
bash deployment/apply_source_framework_migration.sh
```

脚本会自动执行：
1. [source_framework_migration.sql](/C:/Users/chenxl007/Desktop/TechNews_Intelligence/sql/infrastructure/source_framework_migration.sql)
2. [view_logic.sql](/C:/Users/chenxl007/Desktop/TechNews_Intelligence/sql/infrastructure/view_logic.sql)

说明：
- 之前 `api.py startup()` 里做的 `subscribers/source_registry` 改表语句，已统一收敛到迁移脚本。
- 应用启动阶段不再执行 DDL，避免“服务启动顺带改库”。

## 新增一个来源（最小动作）
只需要在 `source_registry` 增加一条记录：

```sql
INSERT INTO public.source_registry (
  source_key,
  source_name,
  source_platform,
  signal_origin,
  fetch_type,
  endpoint,
  is_active,
  priority
)
VALUES (
  'openai_blog',
  'OpenAI Blog',
  'DirectRSS',
  'Official',
  'rss',
  'https://openai.com/news/rss.xml',
  TRUE,
  30
)
ON CONFLICT (source_key) DO UPDATE
SET
  source_name = EXCLUDED.source_name,
  source_platform = EXCLUDED.source_platform,
  signal_origin = EXCLUDED.signal_origin,
  fetch_type = EXCLUDED.fetch_type,
  endpoint = EXCLUDED.endpoint,
  is_active = EXCLUDED.is_active,
  priority = EXCLUDED.priority;
```

## 入库标准字段（采集侧统一）
不管来源是 RSS 还是 API，都统一为以下字段入库：
- `title`
- `url`
- `created_at`
- `source_key`
- `source_name`
- `source_platform`
- `signal_origin`
- `source_meta`（可选，JSON）

只要写齐这些字段，后续 Agent/看板/订阅层可直接复用。

## 与现有系统的兼容性
- 老数据没有新字段时，`view_dashboard_news` 会自动回退旧规则。
- 老 SQL（按 `source_type='HackerNews'/'TechCrunch'`）继续可用。
- 订阅来源选项会自动包含注册表里的启用来源。

## 推荐接入顺序
1. 先把现有 HN/TC 通过迁移脚本纳入注册表（已内置）。
2. 新增 `OpenAI Blog` 和 `arXiv` 两个来源做小流量验证。
3. 验证稳定后再扩展更多官方/学术源。

## 部署后自动执行（可选）
- 如果你有 CI/CD，在部署步骤后追加：

```bash
bash deployment/apply_source_framework_migration.sh
```

- 该脚本是幂等的，重复执行不会破坏已有结构。

# 数据质量只读检查
检查只执行 `SELECT`，不会改库。

## 执行

```bash
bash deployment/scripts/db/run_data_quality_checks.sh
```

```bash
bash deployment/scripts/db/run_data_quality_checks.sh 48
```

说明：窗口参数 `check_hours` 取值 `1~720`（最多 30 天）。

## 覆盖项
- 来源注册状态与窗口入库量
- 核心字段完整性（`title/url/summary/sentiment/created_at/source_*`）
- 异常摘要与异常 `source_id`（如 `chatcmpl-*`）
- `title_cn` 标签合法性
- 失败队列分布
- 向量覆盖、孤儿向量、维度分布
- URL 格式与重复检测
- 时间戳异常
- `tech_news` 与 `source_registry` 元数据一致性
- `system_logs` 24h 健康概览

## 文件位置

- 脚本入口：`../deployment/scripts/db/run_data_quality_checks.sh`
- 脚本依赖公共库：`../deployment/scripts/db/common.sh`
- 检查 SQL：`../sql/infrastructure/checks/data_quality_checks.sql`

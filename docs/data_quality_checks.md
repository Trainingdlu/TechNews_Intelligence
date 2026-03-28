# 数据质量只读检查

该检查流程只执行 `SELECT`，不会修改数据库任何数据。

## 执行方式

默认检查最近 24 小时：

```bash
bash deployment/run_data_quality_checks.sh
```

检查最近 48 小时：

```bash
bash deployment/run_data_quality_checks.sh 48
```

## 覆盖范围（已增强）

- 来源注册表总览与启用状态
- 各来源最近窗口入库量与 `last_seen`
- 核心字段完整性（`title/url/summary/sentiment/created_at/source_*`）
- 历史异常计数（`undefined`、`chatcmpl-*`、`source_id='0'`、错误摘要关键词等）
- 可疑记录样本（缺失来源元数据、可疑 `source_id`、异常摘要）
- `title_cn` 标签合法性检查（`[AI|开发|商业|安全|硬件|生态]`）
- 失败队列分布（`tech_news_failed`）
- 向量覆盖率、孤儿向量、向量维度分布
- URL 质量（空值、非 http(s)、空白字符）
- 重复检查（URL 全量去重、`title_cn` 近 7 天重复）
- 时间戳异常（未来时间、超长历史）
- `tech_news` 与 `source_registry` 元数据一致性
- `system_logs` 最近 24h 健康概览

## 相关文件

- 检查脚本：[run_data_quality_checks.sh](/C:/Users/chenxl007/Desktop/TechNews_Intelligence/deployment/run_data_quality_checks.sh)
- 检查 SQL：[data_quality_checks.sql](/C:/Users/chenxl007/Desktop/TechNews_Intelligence/sql/infrastructure/data_quality_checks.sql)

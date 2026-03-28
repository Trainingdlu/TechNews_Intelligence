# 数据查验（只读）

新增了一套只读查验脚本，不会修改任何数据。

## 一键执行

默认检查最近 24 小时：

```bash
chmod +x deployment/run_data_quality_checks.sh
bash deployment/run_data_quality_checks.sh
```

检查最近 48 小时：

```bash
bash deployment/run_data_quality_checks.sh 48
```

## 覆盖项

- `source_registry` 启用来源清单
- 最近窗口内各来源入库量
- `title/source_*/created_at` 空值与 `undefined` 统计
- `source_id LIKE 'chatcmpl-%'` 异常统计与样本
- `tech_news_failed` 失败原因与来源分布
- 向量覆盖率（最近窗口：有无 embedding）
- `tech_news` URL 重复检测

## 文件

- SQL: `sql/infrastructure/data_quality_checks.sql`
- Script: `deployment/run_data_quality_checks.sh`

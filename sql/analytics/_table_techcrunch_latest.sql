-- [Metabase] Table: 实时快讯
-- Logic: 按发布时间降序排列，展示TN最新发布的新闻

SELECT
    id AS "ID",
    title_cn AS "最新动态",
    CASE 
        WHEN sentiment = 'Negative' THEN '负面'
        WHEN sentiment = 'Neutral'  THEN '中性'
        WHEN sentiment = 'Positive' THEN '正面'
        ELSE sentiment
    END AS "市场情绪",
    hours_ago AS "发布时长 (小时)",
    to_char(created_at_cn, 'Mon DD, HH24:MI') AS "发布时间",
    url AS "原文链接"
FROM view_dashboard_news
WHERE source_type = 'TechCrunch'
ORDER BY created_at DESC LIMIT 15;
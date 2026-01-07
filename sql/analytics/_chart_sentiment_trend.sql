-- [Metabase] Chart: 舆情分时趋势
-- Logic: 按小时聚合统计近48小时的情绪分布与新闻数量，包含中文化映射

SELECT 
    date_trunc('hour', created_at) AS "发布时段",
    CASE 
        WHEN sentiment = 'Negative' THEN '负面'
        WHEN sentiment = 'Neutral'  THEN '中性'
        WHEN sentiment = 'Positive' THEN '正面'
        ELSE sentiment
    END AS "市场情绪",
    COUNT(*) AS "新闻条数"
FROM 
    view_dashboard_news
WHERE 
    created_at >= NOW() - INTERVAL '2 days'
GROUP BY 
    1, 2
ORDER BY 
    1 DESC;
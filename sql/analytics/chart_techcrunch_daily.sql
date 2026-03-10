-- [Metabase] Chart: TechCrunch 日更趋势
-- Logic: 按北京时间 (UTC+8) 统计每日文章发布量，观察周末效应及突发新闻高峰

SELECT 
    date_trunc('day', created_at_cn) AS "日期",
    
    COUNT(*) AS "文章数"

FROM 
    view_dashboard_news

WHERE 
    source_type = 'TechCrunch'
    AND created_at_cn >= NOW() - INTERVAL '30 days'

GROUP BY 
    1
ORDER BY 
    1 ASC;
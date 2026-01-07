-- [Metabase] Chart: 情绪与热度关联分析
-- Logic: 聚合统计不同情绪类别下的平均热度指数与新闻数量，分析情绪对传播效率的影响

SELECT 
    CASE 
        WHEN sentiment = 'Negative' THEN '负面'
        WHEN sentiment = 'Neutral'  THEN '中性'
        WHEN sentiment = 'Positive' THEN '正面'
        ELSE sentiment
    END AS "市场情绪",
    ROUND(AVG(NULLIF(points, 0)), 1) AS "平均热度指数",
    COUNT(*) AS "新闻条数"
FROM 
    view_dashboard_news
WHERE 
    sentiment IS NOT NULL
GROUP BY 
    sentiment
ORDER BY 
    "平均热度指数" DESC;
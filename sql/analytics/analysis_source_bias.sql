-- [Analysis] Source Sentiment Bias
-- Logic: 对比技术社区(HN)与媒体(TC)的情绪构成差异

SELECT 
    source_type AS "来源平台",
    CASE 
        WHEN sentiment = 'Negative' THEN '负面'
        WHEN sentiment = 'Neutral'  THEN '中性'
        WHEN sentiment = 'Positive' THEN '正面'
        ELSE sentiment
    END AS "情绪",
    COUNT(*) AS "文章数"
FROM view_dashboard_news
GROUP BY 1, 2 ORDER BY 1;

-- [Analysis] Source Sentiment Bias (Optimized for Volume Differences)
-- Logic: 对比技术社区(HN)与媒体(TC)的情绪构成差异（使用百分比消除体量差异）

WITH SentimentCounts AS (
    SELECT 
        source_type,
        CASE 
            WHEN sentiment = 'Negative' THEN '负面'
            WHEN sentiment = 'Neutral'  THEN '中性'
            WHEN sentiment = 'Positive' THEN '正面'
            ELSE sentiment
        END AS sentiment_label,
        COUNT(*) AS article_count
    FROM view_dashboard_news
    GROUP BY 1, 2
)
SELECT 
    source_type AS "来源平台",
    sentiment_label AS "情绪",
    article_count AS "文章数",
    ROUND(
        article_count * 100.0 / SUM(article_count) OVER(PARTITION BY source_type), 
        2
    ) AS "情绪占比"
FROM SentimentCounts
ORDER BY 1, 4 DESC;
-- [Metabase] Chart: 舆情分时趋势
-- Logic: 按小时聚合统计近3天的情绪分布与新闻数量，包含中文化映射
SELECT
    date_trunc('hour', created_at_cn) AS "发布时段",
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
    created_at_cn >= CAST(CAST((NOW() + INTERVAL '-3 day') AS date) AS timestamptz)
    AND created_at_cn <  CAST(CAST((NOW() + INTERVAL '1 day')  AS date) AS timestamptz)
GROUP BY
    1, 2
ORDER BY
    1 DESC,
    2 ASC;
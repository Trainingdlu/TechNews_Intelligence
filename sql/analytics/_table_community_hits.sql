-- [Metabase] Chart: 话题热度分布
-- Logic: 气泡图可视化。

SELECT
    title_cn AS "热门话题",
    points AS "讨论热度",
    hours_ago AS "发布时长 (小时)",
    
    CASE 
        WHEN sentiment = 'Negative' THEN '负面'
        WHEN sentiment = 'Neutral'  THEN '中性'
        WHEN sentiment = 'Positive' THEN '正面'
        ELSE sentiment
    END AS "市场情绪",
    
    source_type AS "来源",
    discussion_link AS "讨论链接"
FROM
  view_dashboard_news
WHERE
  points IS NOT NULL
  AND points >= 30  -- 过滤低热度噪音
  AND hours_ago <= 168 -- 只看最近7天
ORDER BY
  points DESC
LIMIT 500;
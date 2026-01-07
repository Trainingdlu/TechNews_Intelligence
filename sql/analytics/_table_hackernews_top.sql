-- [Metabase] Table: 社区热议
-- Logic: 按热度 (Points) 降序排列，展示 HN 社区最火的话题

SELECT
    id AS "ID", -- 用于交互传参
    
    title_cn AS "热门话题",
    
    points AS "热度",
    
    -- 维度清洗
    CASE 
        WHEN sentiment = 'Negative' THEN '负面'
        WHEN sentiment = 'Neutral'  THEN '中性'
        WHEN sentiment = 'Positive' THEN '正面'
        ELSE sentiment
    END AS "市场情绪",
    
    hours_ago AS "发布于",
    
    to_char(created_at_cn, 'Mon DD, HH24:MI') AS "发布日期",
    
    discussion_link AS "讨论链接"

FROM
    view_dashboard_news
WHERE
    source_type = 'HackerNews'
ORDER BY
    points DESC
LIMIT 15;
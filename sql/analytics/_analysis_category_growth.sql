-- [Analysis] Week-over-Week Growth Rate
-- Logic: 计算社区各赛道本周 vs 上周的热度增长率

WITH weekly_stats AS (
    SELECT 
        substring(title_cn from '\[(.*?)\]') AS category,
        
        DATE_TRUNC('week', created_at) as week_start,
        
        SUM(points) as total_heat
    FROM 
        view_dashboard_news
    WHERE 
        created_at > NOW() - INTERVAL '21 days'
        AND source_type = 'HackerNews'
    GROUP BY 
        1, 2
)

SELECT 
    curr.category AS "赛道",
    curr.total_heat as "本周热度",
    COALESCE(prev.total_heat, 0) as "上周热度",
    
    ROUND(
        ((curr.total_heat - prev.total_heat)::numeric / NULLIF(prev.total_heat, 0)) * 100, 
        1
    ) as "周环比增长率(%)"

FROM 
    weekly_stats curr

LEFT JOIN weekly_stats prev 
    ON curr.category = prev.category 
    AND prev.week_start = curr.week_start - INTERVAL '1 week'

WHERE 
    curr.week_start = DATE_TRUNC('week', NOW()) 
    AND curr.category IS NOT NULL
    
ORDER BY 
    4 DESC;
-- [Analysis] Week-over-Week Growth Rate
-- Logic: 计算各赛道本周 vs 上周的热度增长率，捕捉爆发性趋势

WITH weekly_stats AS (
    SELECT 
        substring(title_cn from '\[(.*?)\]') AS category,
        DATE_TRUNC('week', created_at) as week_start,
        SUM(points) as total_heat
    FROM tech_news
    WHERE created_at > NOW() - INTERVAL '14 days'
    GROUP BY 1, 2
)
SELECT 
    curr.category AS "赛道",
    curr.total_heat as "本周热度",
    prev.total_heat as "上周热度",
    ROUND(((curr.total_heat - prev.total_heat)::numeric / NULLIF(prev.total_heat, 0)) * 100, 1) as "周环比增长率(%)"
FROM weekly_stats curr
LEFT JOIN weekly_stats prev ON curr.category = prev.category AND prev.week_start < curr.week_start
WHERE curr.week_start = DATE_TRUNC('week', NOW())
ORDER BY 4 DESC;

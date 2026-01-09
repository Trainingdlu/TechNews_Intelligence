-- [Analysis] Negativity Concentration Index
-- Logic: 计算各赛道负面新闻占比，识别高风险领域

SELECT 
    substring(title_cn from '\[(.*?)\]') AS "赛道",
    ROUND(COUNT(*) FILTER (WHERE sentiment = 'Negative')::numeric / COUNT(*) * 100, 1) as "负面率(%)",
    ROUND(AVG(points), 0) as "平均热度"
FROM tech_news
WHERE created_at > NOW() - INTERVAL '7 days'
GROUP BY 1
HAVING COUNT(*) > 3
ORDER BY 2 DESC;

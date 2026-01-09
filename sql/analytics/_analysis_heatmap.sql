-- [Analysis] Golden Hour for Posting (Heatmap)
-- Logic: 使用 SQL 手动透视，将周一到周日转为列，小时转为行

SELECT 
    EXTRACT(HOUR FROM created_at) AS "发布时刻",
    
    ROUND(AVG(points) FILTER (WHERE EXTRACT(ISODOW FROM created_at) = 1), 0) AS "周一",
    ROUND(AVG(points) FILTER (WHERE EXTRACT(ISODOW FROM created_at) = 2), 0) AS "周二",
    ROUND(AVG(points) FILTER (WHERE EXTRACT(ISODOW FROM created_at) = 3), 0) AS "周三",
    ROUND(AVG(points) FILTER (WHERE EXTRACT(ISODOW FROM created_at) = 4), 0) AS "周四",
    ROUND(AVG(points) FILTER (WHERE EXTRACT(ISODOW FROM created_at) = 5), 0) AS "周五",
    ROUND(AVG(points) FILTER (WHERE EXTRACT(ISODOW FROM created_at) = 6), 0) AS "周六",
    ROUND(AVG(points) FILTER (WHERE EXTRACT(ISODOW FROM created_at) = 7), 0) AS "周日"

FROM 
    view_dashboard_news
WHERE 
    source_type = 'HackerNews'
GROUP BY 
    1
ORDER BY 
    1 ASC;

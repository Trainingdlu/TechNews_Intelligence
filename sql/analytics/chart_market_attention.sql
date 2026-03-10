-- [Metabase] Chart: 热门话题分类统计
-- Logic: 使用正则表达式从中文标题中提取标签，聚合统计各赛道的新闻数量与热度

SELECT 
    substring(title_cn from '[【\[](.*?)[】\]]') AS "类别",
    COUNT(*) AS "新闻条数",
    SUM(points) AS "社区讨论热度"
FROM 
    view_dashboard_news
WHERE 
    title_cn ~ '[【\[].*[】\]]'
GROUP BY 
    1
ORDER BY 
    "新闻条数" DESC;
-- [Metabase] Card: 动态摘要卡片
-- Logic: 接收仪表盘过滤器传递的 ID，实现点击中文标题显示详细摘要的 Master-Detail 交互模式

SELECT 
    title_cn AS "标题",
    summary AS "摘要"
FROM 
    view_dashboard_news
WHERE 
    summary IS NOT NULL 
    AND summary != ''
    [[AND id = {{selected_id}}]]
ORDER BY 
    created_at DESC
LIMIT 1;
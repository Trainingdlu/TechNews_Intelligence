-- [Algorithm] Hacker News Gravity Formula
-- Logic: 模拟 HN 排名算法，平衡热度与时间衰减，寻找当前"上升速度最快"的新闻

SELECT 
    title_cn AS "标题",
    points AS "热度",
    hours_ago AS "发布时长",
    ROUND((points::numeric) / POWER((hours_ago + 2), 1.8), 2) AS "重力得分"
FROM view_dashboard_news
WHERE hours_ago < 48
ORDER BY 4 DESC LIMIT 5;
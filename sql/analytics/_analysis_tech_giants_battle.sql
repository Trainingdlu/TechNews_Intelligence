-- [Analysis] Tech Giants Share of Voice
-- Logic: 使用 UNION ALL 统计提及次数，允许标签重叠，还原真实份额。


WITH company_mentions AS (
    SELECT 'OpenAI' as company, points FROM view_dashboard_news WHERE title_cn ~* 'OpenAI|ChatGPT' OR summary ~* 'OpenAI'
    UNION ALL
    SELECT 'Google' as company, points FROM view_dashboard_news WHERE title_cn ~* 'Google|Gemini' OR summary ~* 'Google'
    UNION ALL
    SELECT 'Microsoft' as company, points FROM view_dashboard_news WHERE title_cn ~* 'Microsoft|Azure' OR summary ~* 'Microsoft'
    UNION ALL
    SELECT 'Apple' as company, points FROM view_dashboard_news WHERE title_cn ~* 'Apple|iPhone' OR summary ~* 'Apple'
    UNION ALL
    SELECT 'Nvidia' as company, points FROM view_dashboard_news WHERE title_cn ~* 'Nvidia|GPU' OR summary ~* 'Nvidia'
    UNION ALL
    SELECT 'DeepSeek' as company, points FROM view_dashboard_news WHERE title_cn ~* 'DeepSeek' OR summary ~* 'DeepSeek'    
    UNION ALL
    SELECT 'Meta' as company, points FROM view_dashboard_news WHERE title_cn ~* 'Meta|Facebook|Llama|Zuckerberg' OR summary ~* 'Meta'
    UNION ALL
    SELECT 'Tesla' as company, points FROM view_dashboard_news WHERE title_cn ~* 'Tesla|Musk|Cybertruck' OR summary ~* 'Tesla'
    UNION ALL
    SELECT 'Amazon' as company, points FROM view_dashboard_news WHERE title_cn ~* 'Amazon|AWS|Bezos' OR summary ~* 'Amazon'
    UNION ALL
    SELECT 'Anthropic' as company, points FROM view_dashboard_news WHERE title_cn ~* 'Anthropic|Claude' OR summary ~* 'Anthropic'
)
SELECT 
    company AS "科技公司",
    COUNT(*) AS "提及次数",
    SUM(points) AS "总关注度",
    ROUND(AVG(points), 0) AS "平均热度"
FROM company_mentions
GROUP BY 1 ORDER BY 3 DESC;
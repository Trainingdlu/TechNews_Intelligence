-- [DDL] PostgreSQL View: view_dashboard_news

DROP VIEW IF EXISTS public.view_dashboard_news;

CREATE VIEW public.view_dashboard_news AS
SELECT 
    tech_news.id,
    tech_news.title,
    tech_news.summary,
    tech_news.title_cn,
    tech_news.url,
    tech_news.points,
    tech_news.sentiment,
    tech_news.created_at,
    
    -- 1. Timezone Adjustment (UTC -> UTC+8 Beijing Time)
    (tech_news.created_at + '08:00:00'::interval) AS created_at_cn,

    -- 2. Source Classification Logic
    CASE
        WHEN tech_news.url LIKE '%techcrunch%' THEN 'TechCrunch'
        WHEN (tech_news.source_id IS NULL OR tech_news.source_id = '') THEN 'TechCrunch'
        ELSE 'HackerNews'
    END AS source_type,

    -- 3. Metric Calculation: Hours Ago (Used for Bubble Chart X-Axis)
    -- Logic: Calculate hours difference, avoid negative numbers
    GREATEST(
        ROUND(
            (EXTRACT(EPOCH FROM (NOW() - tech_news.created_at)) / 3600)::numeric, 
            1
        ), 
        0.0
    ) AS hours_ago,

    -- 4. Smart Link Generation (Drill-through link)
    CASE
        WHEN (tech_news.source_id IS NOT NULL AND tech_news.source_id != '' AND tech_news.url NOT LIKE '%techcrunch%') 
        THEN ('https://news.ycombinator.com/item?id=' || tech_news.source_id)
        ELSE tech_news.url
    END AS discussion_link

FROM public.tech_news;
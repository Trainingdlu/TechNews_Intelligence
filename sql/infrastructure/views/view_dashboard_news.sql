-- [DDL] PostgreSQL View: view_dashboard_news
-- Source resolution priority:
--   1) source_registry canonicalization (match by source_key, fallback by source_name)
--   2) tech_news source columns as fallback
--   3) legacy heuristics for old records

DROP VIEW IF EXISTS public.view_dashboard_news;

CREATE VIEW public.view_dashboard_news AS
WITH normalized AS (
    SELECT
        n.*,

        -- Registry fallback row (prefer key match, fallback to name match)
        reg.source_key AS registry_source_key,
        reg.source_name AS registry_source_name,
        reg.source_platform AS registry_source_platform,
        reg.signal_origin AS registry_signal_origin,

        -- Canonical source key
        COALESCE(
            NULLIF(BTRIM(n.source_key), ''),
            reg.source_key,
            CASE
                WHEN n.url ILIKE '%techcrunch%' THEN 'techcrunch_feed'
                WHEN n.url ILIKE '%news.ycombinator.com%' THEN 'hackernews_frontpage'
                WHEN NULLIF(BTRIM(n.source_id), '') IS NOT NULL THEN 'hackernews_frontpage'
                WHEN NULLIF(BTRIM(n.source_id), '') IS NULL THEN 'techcrunch_feed'
                ELSE NULL
            END
        ) AS resolved_source_key,

        -- Canonical source name (used by subscription filtering + email rendering)
        -- Prefer registry to avoid stale/inconsistent historical labels in tech_news.
        COALESCE(
            reg.source_name,
            NULLIF(BTRIM(n.source_name), ''),
            CASE
                WHEN n.url ILIKE '%techcrunch%' THEN 'TechCrunch'
                WHEN n.url ILIKE '%news.ycombinator.com%' THEN 'HackerNews'
                WHEN NULLIF(BTRIM(n.source_id), '') IS NOT NULL THEN 'HackerNews'
                WHEN NULLIF(BTRIM(n.source_id), '') IS NULL THEN 'TechCrunch'
                ELSE 'Other'
            END
        ) AS resolved_source_name,

        COALESCE(
            reg.source_platform,
            NULLIF(BTRIM(n.source_platform), ''),
            CASE
                WHEN n.url ILIKE '%techcrunch%' THEN 'TechCrunch'
                WHEN n.url ILIKE '%news.ycombinator.com%' THEN 'HackerNews'
                WHEN NULLIF(BTRIM(n.source_id), '') IS NOT NULL THEN 'HackerNews'
                WHEN NULLIF(BTRIM(n.source_id), '') IS NULL THEN 'TechCrunch'
                ELSE 'DirectRSS'
            END
        ) AS resolved_source_platform,

        COALESCE(
            reg.signal_origin,
            NULLIF(BTRIM(n.signal_origin), ''),
            CASE
                WHEN n.url ILIKE '%techcrunch%' THEN 'Media'
                WHEN n.url ILIKE '%news.ycombinator.com%' THEN 'Community'
                WHEN NULLIF(BTRIM(n.source_id), '') IS NOT NULL THEN 'Community'
                WHEN NULLIF(BTRIM(n.source_id), '') IS NULL THEN 'Media'
                ELSE 'Other'
            END
        ) AS resolved_signal_origin

    FROM public.tech_news n
    LEFT JOIN LATERAL (
        SELECT
            r.source_key,
            r.source_name,
            r.source_platform,
            r.signal_origin
        FROM public.source_registry r
        WHERE r.is_active = TRUE
          AND (
                (
                    NULLIF(BTRIM(n.source_key), '') IS NOT NULL
                    AND r.source_key = NULLIF(BTRIM(n.source_key), '')
                )
                OR
                (
                    NULLIF(BTRIM(n.source_key), '') IS NULL
                    AND NULLIF(BTRIM(n.source_name), '') IS NOT NULL
                    AND LOWER(r.source_name) = LOWER(NULLIF(BTRIM(n.source_name), ''))
                )
              )
        ORDER BY r.priority ASC
        LIMIT 1
    ) reg ON TRUE
)
SELECT
    normalized.id,
    normalized.title,
    normalized.summary,
    normalized.title_cn,
    normalized.url,
    normalized.points,
    normalized.sentiment,
    normalized.created_at,

    -- Canonical source columns for downstream (Metabase / subscriptions / daily brief)
    normalized.resolved_source_key AS source_key,
    normalized.resolved_source_name AS source_name,
    normalized.resolved_source_platform AS source_platform,
    normalized.resolved_signal_origin AS signal_origin,

    normalized.source_meta,

    -- 1. Timezone adjustment (UTC -> UTC+8 Beijing time)
    (normalized.created_at + '08:00:00'::interval) AS created_at_cn,

    -- 2. Legacy-compatible aliases used by existing SQL dashboards/workflows
    normalized.resolved_source_name AS source_type,
    normalized.resolved_source_platform AS source_platform_resolved,
    normalized.resolved_signal_origin AS signal_origin_resolved,
    LOWER(SPLIT_PART(SPLIT_PART(normalized.url, '//', 2), '/', 1)) AS source_domain,

    -- 3. Metric calculation: hours ago
    GREATEST(
        ROUND(
            (EXTRACT(EPOCH FROM (NOW() - normalized.created_at)) / 3600)::numeric,
            1
        ),
        0.0
    ) AS hours_ago,

    -- 4. Discussion link
    CASE
        WHEN normalized.resolved_source_key = 'hackernews_frontpage'
             AND NULLIF(BTRIM(normalized.source_id), '') IS NOT NULL
        THEN ('https://news.ycombinator.com/item?id=' || normalized.source_id)
        ELSE normalized.url
    END AS discussion_link

FROM normalized;

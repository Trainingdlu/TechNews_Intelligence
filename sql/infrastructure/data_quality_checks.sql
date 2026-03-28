\echo ''
\echo '============================================================'
\echo 'TechNews Data Quality Checks'
\echo '============================================================'
\echo ''

\echo '1) Active Sources in source_registry'
SELECT
  source_key,
  source_name,
  signal_origin,
  fetch_type,
  is_active,
  priority
FROM public.source_registry
ORDER BY priority, source_key;

\echo ''
\echo '2) Ingestion Volume by Source (updated in last :check_hours hours)'
WITH win AS (
  SELECT NOW() - make_interval(hours => :check_hours) AS since
)
SELECT
  COALESCE(n.source_name, '<null>') AS source_name,
  COALESCE(n.source_key, '<null>') AS source_key,
  COALESCE(n.signal_origin, '<null>') AS signal_origin,
  COUNT(*) AS row_count
FROM public.tech_news n
CROSS JOIN win
WHERE n.updated_at >= win.since
GROUP BY 1, 2, 3
ORDER BY row_count DESC, source_name;

\echo ''
\echo '3) Recent Data Quality Summary (last :check_hours hours)'
WITH win AS (
  SELECT NOW() - make_interval(hours => :check_hours) AS since
)
SELECT
  COUNT(*) AS total_rows,
  COUNT(*) FILTER (WHERE title IS NULL OR BTRIM(title) = '') AS title_null_or_blank,
  COUNT(*) FILTER (WHERE LOWER(COALESCE(title, '')) = 'undefined') AS title_undefined,
  COUNT(*) FILTER (WHERE created_at IS NULL) AS created_at_null,
  COUNT(*) FILTER (WHERE source_key IS NULL OR BTRIM(source_key) = '') AS source_key_null_or_blank,
  COUNT(*) FILTER (WHERE source_name IS NULL OR BTRIM(source_name) = '') AS source_name_null_or_blank,
  COUNT(*) FILTER (WHERE source_platform IS NULL OR BTRIM(source_platform) = '') AS source_platform_null_or_blank,
  COUNT(*) FILTER (WHERE signal_origin IS NULL OR BTRIM(signal_origin) = '') AS signal_origin_null_or_blank
FROM public.tech_news n
CROSS JOIN win
WHERE n.updated_at >= win.since;

\echo ''
\echo '4) Historical Anomaly Counters (all-time)'
SELECT
  COUNT(*) FILTER (WHERE LOWER(COALESCE(title, '')) = 'undefined') AS title_undefined_all_time,
  COUNT(*) FILTER (WHERE source_id ILIKE 'chatcmpl-%') AS source_id_chatcmpl_all_time
FROM public.tech_news;

\echo ''
\echo '5) Sample Rows with Missing Source Metadata (recent)'
WITH win AS (
  SELECT NOW() - make_interval(hours => :check_hours) AS since
)
SELECT
  id,
  title,
  url,
  source_key,
  source_name,
  source_platform,
  signal_origin,
  updated_at
FROM public.tech_news n
CROSS JOIN win
WHERE n.updated_at >= win.since
  AND (
    n.source_key IS NULL OR BTRIM(n.source_key) = '' OR
    n.source_name IS NULL OR BTRIM(n.source_name) = '' OR
    n.source_platform IS NULL OR BTRIM(n.source_platform) = '' OR
    n.signal_origin IS NULL OR BTRIM(n.signal_origin) = ''
  )
ORDER BY n.updated_at DESC
LIMIT 20;

\echo ''
\echo '6) Sample Rows with Suspicious source_id (chatcmpl-*)'
SELECT
  id,
  title,
  url,
  source_id,
  source_key,
  source_name,
  updated_at
FROM public.tech_news
WHERE source_id ILIKE 'chatcmpl-%'
ORDER BY updated_at DESC
LIMIT 20;

\echo ''
\echo '7) Failed Queue Breakdown (last :check_hours hours)'
WITH win AS (
  SELECT NOW() - make_interval(hours => :check_hours) AS since
)
SELECT
  COALESCE(error_reason, '<null>') AS error_reason,
  COALESCE(source_name, source_type, '<null>') AS source_label,
  COUNT(*) AS row_count
FROM public.tech_news_failed f
CROSS JOIN win
WHERE f.created_at >= win.since
GROUP BY 1, 2
ORDER BY row_count DESC, error_reason
LIMIT 30;

\echo ''
\echo '8) Embedding Coverage (last :check_hours hours)'
WITH win AS (
  SELECT NOW() - make_interval(hours => :check_hours) AS since
)
SELECT
  COUNT(*) AS recent_news_rows,
  COUNT(e.url) AS rows_with_embedding,
  COUNT(*) - COUNT(e.url) AS rows_missing_embedding
FROM public.tech_news n
LEFT JOIN public.news_embeddings e
  ON e.url = n.url
CROSS JOIN win
WHERE n.updated_at >= win.since;

\echo ''
\echo '9) Duplicate URL Check in tech_news (all-time)'
SELECT
  COUNT(*) AS total_rows,
  COUNT(DISTINCT url) AS distinct_urls,
  COUNT(*) - COUNT(DISTINCT url) AS duplicate_rows
FROM public.tech_news;

\echo ''
\echo '10) Duplicate URL Samples (if any)'
SELECT
  url,
  COUNT(*) AS cnt
FROM public.tech_news
GROUP BY url
HAVING COUNT(*) > 1
ORDER BY cnt DESC, url
LIMIT 20;

\echo ''
\echo '============================================================'
\echo 'Data quality checks completed.'
\echo '============================================================'
\echo ''

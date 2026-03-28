\echo ''
\echo '============================================================'
\echo 'TechNews Data Quality Checks (Read-Only)'
\echo '============================================================'
\echo 'Window Hours: :check_hours'
\echo ''

\echo '1) Source Registry Overview'
SELECT
  source_key,
  source_name,
  source_platform,
  signal_origin,
  fetch_type,
  is_active,
  priority,
  endpoint
FROM public.source_registry
ORDER BY priority, source_key;

\echo ''
\echo '2) Active Source Coverage (rows in last :check_hours hours + last_seen)'
WITH win AS (
  SELECT NOW() - make_interval(hours => :check_hours) AS since
)
SELECT
  r.source_key,
  r.source_name,
  r.signal_origin,
  r.is_active,
  r.priority,
  COUNT(n.id) FILTER (WHERE n.updated_at >= win.since) AS rows_window,
  MAX(n.updated_at) AS last_seen
FROM public.source_registry r
CROSS JOIN win
LEFT JOIN public.tech_news n
  ON n.source_key = r.source_key
GROUP BY
  r.source_key, r.source_name, r.signal_origin, r.is_active, r.priority, win.since
ORDER BY r.priority, r.source_key;

\echo ''
\echo '3) Ingestion Volume by Source (updated in last :check_hours hours)'
WITH win AS (
  SELECT NOW() - make_interval(hours => :check_hours) AS since
)
SELECT
  COALESCE(n.source_name, '<null>') AS source_name,
  COALESCE(n.source_key, '<null>') AS source_key,
  COALESCE(n.signal_origin, '<null>') AS signal_origin,
  COUNT(*) AS row_count,
  MIN(n.created_at) AS oldest_created_at,
  MAX(n.created_at) AS newest_created_at
FROM public.tech_news n
CROSS JOIN win
WHERE n.updated_at >= win.since
GROUP BY 1, 2, 3
ORDER BY row_count DESC, source_name;

\echo ''
\echo '4) Recent Core Data Quality Summary (last :check_hours hours)'
WITH win AS (
  SELECT NOW() - make_interval(hours => :check_hours) AS since
)
SELECT
  COUNT(*) AS total_rows,
  COUNT(*) FILTER (WHERE title IS NULL OR BTRIM(title) = '') AS title_null_or_blank,
  COUNT(*) FILTER (WHERE LOWER(COALESCE(title, '')) = 'undefined') AS title_undefined,
  COUNT(*) FILTER (WHERE url IS NULL OR BTRIM(url) = '') AS url_null_or_blank,
  COUNT(*) FILTER (WHERE url IS NOT NULL AND url !~* '^https?://') AS url_not_http,
  COUNT(*) FILTER (WHERE summary IS NULL OR BTRIM(summary) = '') AS summary_null_or_blank,
  COUNT(*) FILTER (WHERE summary IS NOT NULL AND LENGTH(BTRIM(summary)) < 20) AS summary_too_short,
  COUNT(*) FILTER (WHERE sentiment IS NULL OR BTRIM(sentiment) = '') AS sentiment_null_or_blank,
  COUNT(*) FILTER (WHERE sentiment IS NOT NULL AND sentiment NOT IN ('Positive','Neutral','Negative')) AS sentiment_invalid,
  COUNT(*) FILTER (WHERE created_at IS NULL) AS created_at_null,
  COUNT(*) FILTER (WHERE created_at > NOW()) AS created_at_future,
  COUNT(*) FILTER (WHERE source_key IS NULL OR BTRIM(source_key) = '') AS source_key_null_or_blank,
  COUNT(*) FILTER (WHERE source_name IS NULL OR BTRIM(source_name) = '') AS source_name_null_or_blank,
  COUNT(*) FILTER (WHERE source_platform IS NULL OR BTRIM(source_platform) = '') AS source_platform_null_or_blank,
  COUNT(*) FILTER (WHERE signal_origin IS NULL OR BTRIM(signal_origin) = '') AS signal_origin_null_or_blank
FROM public.tech_news n
CROSS JOIN win
WHERE n.updated_at >= win.since;

\echo ''
\echo '5) Historical Anomaly Counters (all-time)'
SELECT
  COUNT(*) FILTER (WHERE LOWER(COALESCE(title, '')) = 'undefined') AS title_undefined_all_time,
  COUNT(*) FILTER (WHERE source_id ILIKE 'chatcmpl-%') AS source_id_chatcmpl_all_time,
  COUNT(*) FILTER (WHERE source_id = '0') AS source_id_zero_all_time,
  COUNT(*) FILTER (
    WHERE COALESCE(summary, '') ~* '(403|404|forbidden|access denied|cloudflare|captcha|content unavailable|analysis failed)'
  ) AS summary_error_keywords_all_time,
  COUNT(*) FILTER (WHERE url IS NULL OR BTRIM(url) = '' OR url !~* '^https?://') AS bad_url_all_time,
  COUNT(*) FILTER (WHERE created_at > NOW()) AS created_at_future_all_time
FROM public.tech_news;

\echo ''
\echo '6) Sample Rows with Missing Source Metadata (recent)'
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
\echo '7) Suspicious source_id Samples (chatcmpl-* / 0 / malformed HN id)'
SELECT
  id,
  title,
  url,
  source_key,
  source_name,
  source_platform,
  source_id,
  updated_at
FROM public.tech_news
WHERE
  source_id ILIKE 'chatcmpl-%'
  OR source_id = '0'
  OR (
    COALESCE(source_platform, '') = 'HackerNews'
    AND source_id IS NOT NULL
    AND BTRIM(source_id) <> ''
    AND source_id !~ '^[0-9]+$'
  )
ORDER BY updated_at DESC
LIMIT 30;

\echo ''
\echo '8) Summary Quality Counters (last :check_hours hours)'
WITH win AS (
  SELECT NOW() - make_interval(hours => :check_hours) AS since
)
SELECT
  COUNT(*) AS total_rows,
  COUNT(*) FILTER (WHERE summary IS NULL OR BTRIM(summary) = '') AS summary_blank,
  COUNT(*) FILTER (WHERE summary IS NOT NULL AND LENGTH(BTRIM(summary)) < 20) AS summary_too_short,
  COUNT(*) FILTER (
    WHERE COALESCE(summary, '') ~* '(403|404|forbidden|access denied|cloudflare|captcha|content unavailable|analysis failed)'
  ) AS summary_error_keywords,
  COUNT(*) FILTER (
    WHERE summary IS NOT NULL
      AND LENGTH(BTRIM(summary)) >= 20
      AND summary !~ '[。！？.!?]$'
  ) AS summary_possible_truncated
FROM public.tech_news n
CROSS JOIN win
WHERE n.updated_at >= win.since;

\echo ''
\echo '9) Sample Rows with Suspicious Summaries (recent)'
WITH win AS (
  SELECT NOW() - make_interval(hours => :check_hours) AS since
)
SELECT
  id,
  LEFT(title, 120) AS title_preview,
  LEFT(summary, 220) AS summary_preview,
  sentiment,
  source_key,
  updated_at,
  url
FROM public.tech_news n
CROSS JOIN win
WHERE n.updated_at >= win.since
  AND (
    summary IS NULL OR BTRIM(summary) = ''
    OR LENGTH(BTRIM(summary)) < 20
    OR COALESCE(summary, '') ~* '(403|404|forbidden|access denied|cloudflare|captcha|content unavailable|analysis failed)'
  )
ORDER BY updated_at DESC
LIMIT 20;

\echo ''
\echo '10) title_cn Tag Quality (all-time)'
WITH tagged AS (
  SELECT
    id,
    title_cn,
    substring(title_cn from '^\[(.*?)\]') AS tag
  FROM public.tech_news
)
SELECT
  COUNT(*) AS total_rows,
  COUNT(*) FILTER (WHERE title_cn IS NULL OR BTRIM(title_cn) = '') AS title_cn_blank,
  COUNT(*) FILTER (WHERE title_cn IS NOT NULL AND title_cn ~ '^\[.*\]') AS title_cn_has_bracket_tag,
  COUNT(*) FILTER (
    WHERE title_cn IS NOT NULL
      AND title_cn ~ '^\[.*\]'
      AND COALESCE(tag, '') NOT IN ('AI', '开发', '商业', '安全', '硬件', '生态')
  ) AS title_cn_invalid_tag
FROM tagged;

\echo ''
\echo '11) Invalid title_cn Tag Samples (recent first)'
WITH tagged AS (
  SELECT
    id,
    created_at,
    title_cn,
    url,
    substring(title_cn from '^\[(.*?)\]') AS tag
  FROM public.tech_news
)
SELECT
  id,
  created_at,
  tag AS invalid_tag,
  LEFT(title_cn, 120) AS title_cn_preview,
  url
FROM tagged
WHERE title_cn IS NOT NULL
  AND title_cn ~ '^\[.*\]'
  AND COALESCE(tag, '') NOT IN ('AI', '开发', '商业', '安全', '硬件', '生态')
ORDER BY created_at DESC
LIMIT 20;

\echo ''
\echo '12) Sentiment Distribution by Source (last :check_hours hours)'
WITH win AS (
  SELECT NOW() - make_interval(hours => :check_hours) AS since
)
SELECT
  COALESCE(source_key, '<null>') AS source_key,
  COALESCE(sentiment, '<null>') AS sentiment,
  COUNT(*) AS row_count
FROM public.tech_news n
CROSS JOIN win
WHERE n.updated_at >= win.since
GROUP BY 1, 2
ORDER BY source_key, row_count DESC;

\echo ''
\echo '13) Failed Queue Breakdown (last :check_hours hours)'
WITH win AS (
  SELECT NOW() - make_interval(hours => :check_hours) AS since
)
SELECT
  COALESCE(error_reason, '<null>') AS error_reason,
  COALESCE(source_name, source_type, '<null>') AS source_label,
  COUNT(*) AS row_count,
  MIN(created_at) AS earliest_in_window,
  MAX(created_at) AS latest_in_window
FROM public.tech_news_failed f
CROSS JOIN win
WHERE f.created_at >= win.since
GROUP BY 1, 2
ORDER BY row_count DESC, error_reason
LIMIT 40;

\echo ''
\echo '14) Embedding Coverage + Orphan Checks'
WITH win AS (
  SELECT NOW() - make_interval(hours => :check_hours) AS since
),
recent_news AS (
  SELECT url
  FROM public.tech_news n
  CROSS JOIN win
  WHERE n.updated_at >= win.since
),
orphan_embeddings AS (
  SELECT e.url
  FROM public.news_embeddings e
  LEFT JOIN public.tech_news t
    ON t.url = e.url
  WHERE t.url IS NULL
)
SELECT
  (SELECT COUNT(*) FROM recent_news) AS recent_news_rows,
  (SELECT COUNT(*) FROM recent_news r JOIN public.news_embeddings e ON e.url = r.url) AS recent_rows_with_embedding,
  (SELECT COUNT(*) FROM recent_news) - (SELECT COUNT(*) FROM recent_news r JOIN public.news_embeddings e ON e.url = r.url) AS recent_rows_missing_embedding,
  (SELECT COUNT(*) FROM public.news_embeddings) AS all_embeddings_rows,
  (SELECT COUNT(*) FROM orphan_embeddings) AS orphan_embeddings_rows;

\echo ''
\echo '15) Embedding Vector Dimensionality Distribution'
SELECT
  COALESCE(vector_dims(embedding), 0) AS dims,
  COUNT(*) AS row_count
FROM public.news_embeddings
GROUP BY 1
ORDER BY 1;

\echo ''
\echo '16) Duplicate URL Check (all-time)'
SELECT
  COUNT(*) AS total_rows,
  COUNT(DISTINCT url) AS distinct_urls,
  COUNT(*) - COUNT(DISTINCT url) AS duplicate_rows
FROM public.tech_news;

\echo ''
\echo '17) Duplicate URL Samples (if any)'
SELECT
  url,
  COUNT(*) AS cnt
FROM public.tech_news
GROUP BY url
HAVING COUNT(*) > 1
ORDER BY cnt DESC, url
LIMIT 30;

\echo ''
\echo '18) Duplicate title_cn (last 7 days)'
SELECT
  title_cn,
  COUNT(*) AS cnt,
  MIN(created_at) AS first_seen,
  MAX(created_at) AS last_seen
FROM public.tech_news
WHERE created_at >= NOW() - INTERVAL '7 days'
  AND title_cn IS NOT NULL
  AND BTRIM(title_cn) <> ''
GROUP BY title_cn
HAVING COUNT(*) > 1
ORDER BY cnt DESC, last_seen DESC
LIMIT 30;

\echo ''
\echo '19) URL Format Quality (all-time)'
SELECT
  COUNT(*) AS total_rows,
  COUNT(*) FILTER (WHERE url IS NULL OR BTRIM(url) = '') AS url_blank,
  COUNT(*) FILTER (WHERE url IS NOT NULL AND url !~* '^https?://') AS url_not_http,
  COUNT(*) FILTER (WHERE url ~ '\s') AS url_contains_whitespace
FROM public.tech_news;

\echo ''
\echo '20) Invalid URL Samples'
SELECT
  id,
  title,
  url,
  source_key,
  created_at
FROM public.tech_news
WHERE
  url IS NULL
  OR BTRIM(url) = ''
  OR url !~* '^https?://'
  OR url ~ '\s'
ORDER BY created_at DESC
LIMIT 20;

\echo ''
\echo '21) Timestamp Anomaly Counters'
SELECT
  COUNT(*) FILTER (WHERE created_at IS NULL) AS created_at_null,
  COUNT(*) FILTER (WHERE created_at > NOW()) AS created_at_future,
  COUNT(*) FILTER (WHERE created_at < NOW() - INTERVAL '730 days') AS created_at_older_than_2y
FROM public.tech_news;

\echo ''
\echo '22) Source Metadata Mismatch vs source_registry (all-time)'
WITH joined AS (
  SELECT
    n.id,
    n.source_key,
    n.source_name,
    n.source_platform,
    n.signal_origin,
    r.source_name AS registry_source_name,
    r.source_platform AS registry_source_platform,
    r.signal_origin AS registry_signal_origin,
    n.url,
    n.updated_at
  FROM public.tech_news n
  JOIN public.source_registry r
    ON r.source_key = n.source_key
)
SELECT
  COUNT(*) AS total_joined_rows,
  COUNT(*) FILTER (WHERE COALESCE(source_name, '') <> COALESCE(registry_source_name, '')) AS bad_source_name,
  COUNT(*) FILTER (WHERE COALESCE(source_platform, '') <> COALESCE(registry_source_platform, '')) AS bad_source_platform,
  COUNT(*) FILTER (WHERE COALESCE(signal_origin, '') <> COALESCE(registry_signal_origin, '')) AS bad_signal_origin
FROM joined;

\echo ''
\echo '23) Source Metadata Mismatch Samples'
WITH joined AS (
  SELECT
    n.id,
    n.source_key,
    n.source_name,
    n.source_platform,
    n.signal_origin,
    r.source_name AS registry_source_name,
    r.source_platform AS registry_source_platform,
    r.signal_origin AS registry_signal_origin,
    n.url,
    n.updated_at
  FROM public.tech_news n
  JOIN public.source_registry r
    ON r.source_key = n.source_key
)
SELECT
  id,
  source_key,
  source_name,
  registry_source_name,
  source_platform,
  registry_source_platform,
  signal_origin,
  registry_signal_origin,
  updated_at,
  url
FROM joined
WHERE
  COALESCE(source_name, '') <> COALESCE(registry_source_name, '')
  OR COALESCE(source_platform, '') <> COALESCE(registry_source_platform, '')
  OR COALESCE(signal_origin, '') <> COALESCE(registry_signal_origin, '')
ORDER BY updated_at DESC
LIMIT 20;

\echo ''
\echo '24) System Log Health (last 24h)'
SELECT
  event_type,
  COUNT(*) AS row_count,
  MAX(created_at) AS latest_at
FROM public.system_logs
WHERE created_at >= NOW() - INTERVAL '24 hours'
GROUP BY event_type
ORDER BY row_count DESC, event_type;

\echo ''
\echo '25) Jina Raw Log Coverage + Duplicate Check'
WITH dup AS (
  SELECT url, COUNT(*) AS cnt
  FROM public.jina_raw_logs
  GROUP BY url
  HAVING COUNT(*) > 1
)
SELECT
  (SELECT COUNT(*) FROM public.jina_raw_logs) AS total_raw_logs,
  (SELECT COUNT(DISTINCT url) FROM public.jina_raw_logs) AS distinct_raw_urls,
  (SELECT COALESCE(SUM(cnt - 1), 0) FROM dup) AS duplicate_raw_rows,
  (SELECT COUNT(*) FROM public.tech_news t WHERE NOT EXISTS (
      SELECT 1 FROM public.jina_raw_logs j WHERE j.url = t.url
  )) AS news_without_raw_log;

\echo ''
\echo '26) Jina Duplicate URL Samples (if any)'
SELECT
  url,
  COUNT(*) AS cnt
FROM public.jina_raw_logs
GROUP BY url
HAVING COUNT(*) > 1
ORDER BY cnt DESC, url
LIMIT 20;

\echo ''
\echo '============================================================'
\echo 'Data quality checks completed (read-only).'
\echo '============================================================'
\echo ''

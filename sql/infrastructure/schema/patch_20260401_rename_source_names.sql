-- Rename source display names to keep naming consistent.
-- Target names:
-- - Google AI Blog      -> Google Blog
-- - Microsoft AI Blog   -> Microsoft Blog
-- - AWS Machine Learning Blog -> AWS ML Learning
--
-- Safe to run repeatedly.

\set ON_ERROR_STOP on
BEGIN;

-- 1) Canonical source registry
UPDATE public.source_registry
SET source_name = CASE
    WHEN source_name = 'Google AI Blog' THEN 'Google Blog'
    WHEN source_name = 'Microsoft AI Blog' THEN 'Microsoft Blog'
    WHEN source_name = 'AWS Machine Learning Blog' THEN 'AWS ML Learning'
    ELSE source_name
END
WHERE source_name IN (
    'Google AI Blog',
    'Microsoft AI Blog',
    'AWS Machine Learning Blog'
);

-- 2) Historical news rows
UPDATE public.tech_news
SET source_name = CASE
    WHEN source_name = 'Google AI Blog' THEN 'Google Blog'
    WHEN source_name = 'Microsoft AI Blog' THEN 'Microsoft Blog'
    WHEN source_name = 'AWS Machine Learning Blog' THEN 'AWS ML Learning'
    ELSE source_name
END
WHERE source_name IN (
    'Google AI Blog',
    'Microsoft AI Blog',
    'AWS Machine Learning Blog'
);

-- 3) Failed queue rows (if any)
UPDATE public.tech_news_failed
SET source_name = CASE
        WHEN source_name = 'Google AI Blog' THEN 'Google Blog'
        WHEN source_name = 'Microsoft AI Blog' THEN 'Microsoft Blog'
        WHEN source_name = 'AWS Machine Learning Blog' THEN 'AWS ML Learning'
        ELSE source_name
    END,
    source_type = CASE
        WHEN source_type = 'Google AI Blog' THEN 'Google Blog'
        WHEN source_type = 'Microsoft AI Blog' THEN 'Microsoft Blog'
        WHEN source_type = 'AWS Machine Learning Blog' THEN 'AWS ML Learning'
        ELSE source_type
    END
WHERE source_name IN (
        'Google AI Blog',
        'Microsoft AI Blog',
        'AWS Machine Learning Blog'
    )
   OR source_type IN (
        'Google AI Blog',
        'Microsoft AI Blog',
        'AWS Machine Learning Blog'
    );

-- 4) Subscribers source preferences (json array)
UPDATE public.subscribers s
SET source_preferences = (
    SELECT COALESCE(
        jsonb_agg(
            CASE
                WHEN elem = 'Google AI Blog' THEN 'Google Blog'
                WHEN elem = 'Microsoft AI Blog' THEN 'Microsoft Blog'
                WHEN elem = 'AWS Machine Learning Blog' THEN 'AWS ML Learning'
                ELSE elem
            END
        ),
        '[]'::jsonb
    )
    FROM jsonb_array_elements_text(COALESCE(s.source_preferences, '[]'::jsonb)) AS elem
)
WHERE s.source_preferences::text LIKE '%Google AI Blog%'
   OR s.source_preferences::text LIKE '%Microsoft AI Blog%'
   OR s.source_preferences::text LIKE '%AWS Machine Learning Blog%';

-- 5) Optional metadata cleanup if source_meta embeds source_name
UPDATE public.tech_news
SET source_meta = jsonb_set(source_meta, '{source_name}', to_jsonb('Google Blog'::text), true)
WHERE source_meta ? 'source_name' AND source_meta->>'source_name' = 'Google AI Blog';

UPDATE public.tech_news
SET source_meta = jsonb_set(source_meta, '{source_name}', to_jsonb('Microsoft Blog'::text), true)
WHERE source_meta ? 'source_name' AND source_meta->>'source_name' = 'Microsoft AI Blog';

UPDATE public.tech_news
SET source_meta = jsonb_set(source_meta, '{source_name}', to_jsonb('AWS ML Learning'::text), true)
WHERE source_meta ? 'source_name' AND source_meta->>'source_name' = 'AWS Machine Learning Blog';

UPDATE public.tech_news_failed
SET source_meta = jsonb_set(source_meta, '{source_name}', to_jsonb('Google Blog'::text), true)
WHERE source_meta ? 'source_name' AND source_meta->>'source_name' = 'Google AI Blog';

UPDATE public.tech_news_failed
SET source_meta = jsonb_set(source_meta, '{source_name}', to_jsonb('Microsoft Blog'::text), true)
WHERE source_meta ? 'source_name' AND source_meta->>'source_name' = 'Microsoft AI Blog';

UPDATE public.tech_news_failed
SET source_meta = jsonb_set(source_meta, '{source_name}', to_jsonb('AWS ML Learning'::text), true)
WHERE source_meta ? 'source_name' AND source_meta->>'source_name' = 'AWS Machine Learning Blog';

COMMIT;

-- Verification snapshot
SELECT 'source_registry.source_name' AS field, source_name AS value, COUNT(*) AS cnt
FROM public.source_registry
GROUP BY source_name
UNION ALL
SELECT 'tech_news.source_name' AS field, source_name AS value, COUNT(*) AS cnt
FROM public.tech_news
WHERE source_name IS NOT NULL AND source_name <> ''
GROUP BY source_name
UNION ALL
SELECT 'tech_news_failed.source_name' AS field, source_name AS value, COUNT(*) AS cnt
FROM public.tech_news_failed
WHERE source_name IS NOT NULL AND source_name <> ''
GROUP BY source_name
UNION ALL
SELECT 'tech_news_failed.source_type' AS field, source_type AS value, COUNT(*) AS cnt
FROM public.tech_news_failed
WHERE source_type IS NOT NULL AND source_type <> ''
GROUP BY source_type
ORDER BY field, cnt DESC, value;

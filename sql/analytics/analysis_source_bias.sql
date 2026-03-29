-- [Analysis] Source Sentiment Bias (stable labels for Metabase)
-- Logic:
-- 1) Clean source labels (trim spaces/quotes/invisible chars)
-- 2) Normalize sentiment to CN labels
-- 3) Compare share within each source to remove volume bias

WITH base AS (
    SELECT
        -- Remove BOM / zero-width chars first, then trim wrappers.
        regexp_replace(
            BTRIM(
                REPLACE(
                    REPLACE(
                        REPLACE(
                            REPLACE(COALESCE(source_type, 'Other'), chr(65279), ''), -- BOM
                            chr(8203), '' -- zero-width space
                        ),
                        chr(8204), '' -- zero-width non-joiner
                    ),
                    chr(8205), '' -- zero-width joiner
                )
            ),
            '^[ ''"“”‘’]+|[ ''"“”‘’]+$',
            '',
            'g'
        ) AS source_type_clean,
        CASE
            WHEN sentiment = 'Negative' THEN '负面'
            WHEN sentiment = 'Neutral'  THEN '中性'
            WHEN sentiment = 'Positive' THEN '正面'
            ELSE sentiment
        END AS sentiment_label
    FROM public.view_dashboard_news
    WHERE sentiment IN ('Negative', 'Neutral', 'Positive')
),
sentiment_counts AS (
    SELECT
        source_type_clean,
        sentiment_label,
        COUNT(*) AS article_count
    FROM base
    WHERE source_type_clean IS NOT NULL
      AND source_type_clean <> ''
    GROUP BY 1, 2
)
SELECT
    source_type_clean AS "来源平台",
    sentiment_label AS "情绪",
    article_count AS "文章数",
    ROUND(
        article_count * 100.0 / SUM(article_count) OVER (PARTITION BY source_type_clean),
        2
    ) AS "情绪占比"
FROM sentiment_counts
ORDER BY 1, 4 DESC;

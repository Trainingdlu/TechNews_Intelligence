-- Source framework migration (non-breaking, backward compatible)
-- Run this on existing deployments before applying view_logic.sql

-- 0) Keep subscribers table compatible with current API contract
CREATE TABLE IF NOT EXISTS public.subscribers (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) NOT NULL,
    name VARCHAR(50),
    is_active BOOLEAN DEFAULT TRUE,
    source_preferences JSONB NOT NULL DEFAULT '[]'::jsonb,
    frequency VARCHAR(20) NOT NULL DEFAULT 'daily',
    timezone VARCHAR(50) NOT NULL DEFAULT 'Asia/Shanghai',
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT unique_subscriber_email UNIQUE (email)
);

ALTER TABLE public.subscribers
    ADD COLUMN IF NOT EXISTS source_preferences JSONB NOT NULL DEFAULT '[]'::jsonb;
ALTER TABLE public.subscribers
    ADD COLUMN IF NOT EXISTS frequency VARCHAR(20) NOT NULL DEFAULT 'daily';
ALTER TABLE public.subscribers
    ADD COLUMN IF NOT EXISTS timezone VARCHAR(50) NOT NULL DEFAULT 'Asia/Shanghai';
ALTER TABLE public.subscribers
    ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();

CREATE INDEX IF NOT EXISTS idx_subscribers_active ON public.subscribers(is_active);

-- 1) Extend tech_news with explicit source metadata
ALTER TABLE public.tech_news
    ADD COLUMN IF NOT EXISTS source_key VARCHAR(100);
ALTER TABLE public.tech_news
    ADD COLUMN IF NOT EXISTS source_name VARCHAR(100);
ALTER TABLE public.tech_news
    ADD COLUMN IF NOT EXISTS source_platform VARCHAR(50);
ALTER TABLE public.tech_news
    ADD COLUMN IF NOT EXISTS signal_origin VARCHAR(30);
ALTER TABLE public.tech_news
    ADD COLUMN IF NOT EXISTS source_meta JSONB NOT NULL DEFAULT '{}'::jsonb;

CREATE INDEX IF NOT EXISTS idx_source_name ON public.tech_news(source_name);
CREATE INDEX IF NOT EXISTS idx_signal_origin ON public.tech_news(signal_origin);

-- 2) Extend failed queue with explicit source metadata
ALTER TABLE public.tech_news_failed
    ADD COLUMN IF NOT EXISTS source_key VARCHAR(100);
ALTER TABLE public.tech_news_failed
    ADD COLUMN IF NOT EXISTS source_name VARCHAR(100);
ALTER TABLE public.tech_news_failed
    ADD COLUMN IF NOT EXISTS source_platform VARCHAR(50);
ALTER TABLE public.tech_news_failed
    ADD COLUMN IF NOT EXISTS signal_origin VARCHAR(30);
ALTER TABLE public.tech_news_failed
    ADD COLUMN IF NOT EXISTS source_meta JSONB NOT NULL DEFAULT '{}'::jsonb;

-- 3) Create source registry table
CREATE TABLE IF NOT EXISTS public.source_registry (
    id SERIAL PRIMARY KEY,
    source_key VARCHAR(100) NOT NULL UNIQUE,
    source_name VARCHAR(100) NOT NULL,
    source_platform VARCHAR(50) NOT NULL DEFAULT 'DirectRSS',
    signal_origin VARCHAR(30) NOT NULL DEFAULT 'Other',
    fetch_type VARCHAR(20) NOT NULL DEFAULT 'rss',
    endpoint TEXT NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    priority INTEGER NOT NULL DEFAULT 100,
    extra_config JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_source_registry_active
ON public.source_registry(is_active, priority);

-- 4) Ensure updated_at trigger can be reused
CREATE OR REPLACE FUNCTION update_modified_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

DROP TRIGGER IF EXISTS update_source_registry_modtime ON public.source_registry;
CREATE TRIGGER update_source_registry_modtime
    BEFORE UPDATE ON public.source_registry
    FOR EACH ROW
    EXECUTE FUNCTION update_modified_column();

-- 5) Seed legacy sources so current HN/TC are first-class registry entries
INSERT INTO public.source_registry (
    source_key, source_name, source_platform, signal_origin, fetch_type, endpoint, is_active, priority
)
VALUES
    (
        'hackernews_frontpage',
        'HackerNews',
        'HackerNews',
        'Community',
        'api',
        'https://hn.algolia.com/api/v1/search?tags=front_page&page=0&hitsPerPage=60&numericFilters=points>=30,num_comments>=5',
        TRUE,
        10
    ),
    (
        'techcrunch_feed',
        'TechCrunch',
        'TechCrunch',
        'Media',
        'rss',
        'https://techcrunch.com/feed/',
        TRUE,
        20
    )
ON CONFLICT (source_key) DO UPDATE
SET
    source_name = EXCLUDED.source_name,
    source_platform = EXCLUDED.source_platform,
    signal_origin = EXCLUDED.signal_origin,
    fetch_type = EXCLUDED.fetch_type,
    endpoint = EXCLUDED.endpoint,
    is_active = EXCLUDED.is_active,
    priority = EXCLUDED.priority;

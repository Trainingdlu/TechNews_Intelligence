-- [DDL] PostgreSQL Table

-- 0. Extensions
CREATE EXTENSION IF NOT EXISTS vector;

-- 1. Create Table
CREATE TABLE IF NOT EXISTS public.tech_news (
    id SERIAL PRIMARY KEY,
    
    -- Core Content
    title TEXT,
    url TEXT NOT NULL,
    
    -- Metrics & Analysis
    summary TEXT,
    title_cn TEXT,
    sentiment TEXT,
    
    points INTEGER DEFAULT 0,
    
    -- Metadata
    source_id VARCHAR(50),
    
    -- Timestamps
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CONSTRAINT unique_url UNIQUE (url)
);

-- Backward-compatible source framework columns (safe for existing deployments)
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

-- 2. Dead Letter Queue
CREATE TABLE IF NOT EXISTS public.tech_news_failed (
    id SERIAL PRIMARY KEY,
    original_title TEXT,
    title_cn TEXT,
    url TEXT NOT NULL,
    source_type VARCHAR(50),
    
    error_reason TEXT,
    
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT unique_failed_url UNIQUE (url)
);

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

-- 3. Jina data
CREATE TABLE IF NOT EXISTS public.jina_raw_logs (
    id SERIAL PRIMARY KEY,
    url TEXT NOT NULL,
    raw_content TEXT, 
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT idx_unique_jina_url UNIQUE (url)
);

-- 4. News Embeddings
CREATE TABLE IF NOT EXISTS public.news_embeddings (
    id SERIAL PRIMARY KEY,
    url TEXT NOT NULL,
    embedding vector(1024),
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT idx_unique_embedding_url UNIQUE (url)
);

-- 5. System logs
CREATE TABLE IF NOT EXISTS public.system_logs (
    id SERIAL PRIMARY KEY,
    
    -- Log Details
    event_type VARCHAR(50),
    source VARCHAR(50),
    message TEXT,
    target_url TEXT,
    
    -- Timestamp
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 6. Subscribers (Daily Brief)
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

-- Backward-compatible subscriber columns for old deployments
ALTER TABLE public.subscribers
    ADD COLUMN IF NOT EXISTS source_preferences JSONB NOT NULL DEFAULT '[]'::jsonb;
ALTER TABLE public.subscribers
    ADD COLUMN IF NOT EXISTS frequency VARCHAR(20) NOT NULL DEFAULT 'daily';
ALTER TABLE public.subscribers
    ADD COLUMN IF NOT EXISTS timezone VARCHAR(50) NOT NULL DEFAULT 'Asia/Shanghai';
ALTER TABLE public.subscribers
    ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();

-- 7. Source Registry (fill-in source onboarding framework)
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

-- 8. Access_Tokens
CREATE TABLE IF NOT EXISTS access_tokens (
    id          SERIAL       PRIMARY KEY,
    email       VARCHAR(255) NOT NULL,
    token       VARCHAR(64)  NOT NULL UNIQUE,
    quota       INT          NOT NULL DEFAULT 10,
    used        INT          NOT NULL DEFAULT 0,
    status      VARCHAR(20)  NOT NULL DEFAULT 'active',   -- active / exhausted / upgraded
    notified    BOOLEAN      NOT NULL DEFAULT FALSE,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    upgraded_at TIMESTAMPTZ
);

-- 9. Performance Indexes
CREATE INDEX IF NOT EXISTS idx_created_at ON public.tech_news(created_at);
CREATE INDEX IF NOT EXISTS idx_created_at_cn ON public.tech_news ((created_at + '08:00:00'::interval));
CREATE INDEX IF NOT EXISTS idx_points ON public.tech_news(points DESC);
CREATE INDEX IF NOT EXISTS idx_sentiment ON public.tech_news(sentiment);
CREATE INDEX IF NOT EXISTS idx_source_name ON public.tech_news(source_name);
CREATE INDEX IF NOT EXISTS idx_signal_origin ON public.tech_news(signal_origin);
CREATE INDEX IF NOT EXISTS idx_logs_created_at ON public.system_logs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_access_tokens_token ON access_tokens (token);
CREATE INDEX IF NOT EXISTS idx_access_tokens_email ON access_tokens (email);
CREATE INDEX IF NOT EXISTS idx_subscribers_active ON public.subscribers(is_active);
CREATE INDEX IF NOT EXISTS idx_source_registry_active ON public.source_registry(is_active, priority);

-- NOTE: This index is safe to create on an empty table, but should be rebuilt
-- after the initial data backfill to ensure optimal clustering quality.
-- Run: REINDEX INDEX idx_embedding_vector;
CREATE INDEX IF NOT EXISTS idx_embedding_vector ON public.news_embeddings
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- 10. Update Trigger Function
CREATE OR REPLACE FUNCTION update_modified_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

DROP TRIGGER IF EXISTS update_tech_news_modtime ON public.tech_news;
CREATE TRIGGER update_tech_news_modtime
    BEFORE UPDATE ON public.tech_news
    FOR EACH ROW
    EXECUTE FUNCTION update_modified_column();

DROP TRIGGER IF EXISTS update_source_registry_modtime ON public.source_registry;
CREATE TRIGGER update_source_registry_modtime
    BEFORE UPDATE ON public.source_registry
    FOR EACH ROW
    EXECUTE FUNCTION update_modified_column();

-- Seed default legacy sources into registry (idempotent)
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

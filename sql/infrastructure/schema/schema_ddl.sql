-- [DDL] PostgreSQL Table

-- 0. Extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

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

-- 4b. Search Index (derived from tech_news)
CREATE TABLE IF NOT EXISTS public.news_search_index (
    url TEXT PRIMARY KEY
        REFERENCES public.tech_news(url)
        ON DELETE CASCADE
        ON UPDATE CASCADE,
    search_tsv tsvector NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 4c. Entity Registry and Alias Tables
CREATE TABLE IF NOT EXISTS public.entity_registry (
    entity_id BIGSERIAL PRIMARY KEY,
    canonical_name TEXT NOT NULL,
    entity_type VARCHAR(50) NOT NULL DEFAULT 'unknown',
    wikidata_id VARCHAR(64),
    source VARCHAR(50) NOT NULL DEFAULT 'manual',
    confidence NUMERIC(4, 3) NOT NULL DEFAULT 1.000,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT entity_registry_canonical_unique UNIQUE (canonical_name)
);

CREATE TABLE IF NOT EXISTS public.entity_alias (
    alias_id BIGSERIAL PRIMARY KEY,
    entity_id BIGINT NOT NULL
        REFERENCES public.entity_registry(entity_id)
        ON DELETE CASCADE,
    alias TEXT NOT NULL,
    language VARCHAR(20) NOT NULL DEFAULT 'unknown',
    alias_type VARCHAR(50) NOT NULL DEFAULT 'manual',
    weight NUMERIC(6, 3) NOT NULL DEFAULT 1.000,
    is_exact BOOLEAN NOT NULL DEFAULT TRUE,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS public.entity_alias_candidate (
    candidate_id BIGSERIAL PRIMARY KEY,
    alias TEXT NOT NULL,
    suggested_entity_id BIGINT
        REFERENCES public.entity_registry(entity_id)
        ON DELETE SET NULL,
    canonical_name TEXT,
    entity_type VARCHAR(50) NOT NULL DEFAULT 'unknown',
    source VARCHAR(50) NOT NULL DEFAULT 'corpus',
    confidence NUMERIC(4, 3) NOT NULL DEFAULT 0.000,
    evidence_count INTEGER NOT NULL DEFAULT 0,
    evidence_urls JSONB NOT NULL DEFAULT '[]'::jsonb,
    status VARCHAR(32) NOT NULL DEFAULT 'pending',
    reason TEXT,
    aliases_to_add JSONB NOT NULL DEFAULT '[]'::jsonb,
    reviewed_at TIMESTAMPTZ,
    promoted_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT entity_alias_candidate_status_chk
        CHECK (status IN ('pending', 'auto_approved', 'approved', 'rejected'))
);

ALTER TABLE public.entity_alias_candidate
    ADD COLUMN IF NOT EXISTS reviewed_at TIMESTAMPTZ;
ALTER TABLE public.entity_alias_candidate
    ADD COLUMN IF NOT EXISTS promoted_at TIMESTAMPTZ;

CREATE TABLE IF NOT EXISTS public.news_entity_mentions (
    url TEXT NOT NULL
        REFERENCES public.tech_news(url)
        ON DELETE CASCADE
        ON UPDATE CASCADE,
    entity_id BIGINT NOT NULL
        REFERENCES public.entity_registry(entity_id)
        ON DELETE CASCADE,
    alias TEXT,
    field VARCHAR(32) NOT NULL DEFAULT 'unknown',
    confidence NUMERIC(4, 3) NOT NULL DEFAULT 0.000,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (url, entity_id, field)
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
    status      VARCHAR(20)  NOT NULL DEFAULT 'active',   -- active / pending / capped
    notified    BOOLEAN      NOT NULL DEFAULT FALSE,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    upgraded_at TIMESTAMPTZ,
    tier        SMALLINT     NOT NULL DEFAULT 0,
    unlimited   BOOLEAN      NOT NULL DEFAULT FALSE
);

-- 9. Conversation Threads (persistent chat sessions)
CREATE TABLE IF NOT EXISTS public.conversation_threads (
    id BIGSERIAL PRIMARY KEY,
    thread_id VARCHAR(64) NOT NULL UNIQUE,
    channel VARCHAR(32) NOT NULL DEFAULT 'generic',
    subject TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_message_at TIMESTAMPTZ
);

-- 10. Conversation Messages (history entries per thread)
CREATE TABLE IF NOT EXISTS public.conversation_messages (
    id BIGSERIAL PRIMARY KEY,
    thread_id VARCHAR(64) NOT NULL,
    role VARCHAR(20) NOT NULL,
    parts JSONB NOT NULL DEFAULT '[]'::jsonb,
    payload JSONB NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT fk_conversation_messages_thread
        FOREIGN KEY (thread_id)
        REFERENCES public.conversation_threads(thread_id)
        ON DELETE CASCADE
);

-- 11. Thread memory for agent context assembly
CREATE TABLE IF NOT EXISTS public.thread_memory_summaries (
    id BIGSERIAL PRIMARY KEY,
    thread_id VARCHAR(64) NOT NULL UNIQUE,
    summary_text TEXT NOT NULL DEFAULT '',
    summary_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    last_summarized_message_id BIGINT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT fk_thread_memory_summaries_thread
        FOREIGN KEY (thread_id)
        REFERENCES public.conversation_threads(thread_id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS public.thread_evidence_index (
    id BIGSERIAL PRIMARY KEY,
    thread_id VARCHAR(64) NOT NULL,
    message_id BIGINT,
    turn_request_id VARCHAR(64),
    evidence_url TEXT NOT NULL,
    title TEXT,
    source_index INTEGER,
    excerpt TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_thread_evidence_index_thread_url UNIQUE (thread_id, evidence_url),
    CONSTRAINT fk_thread_evidence_index_thread
        FOREIGN KEY (thread_id)
        REFERENCES public.conversation_threads(thread_id)
        ON DELETE CASCADE
);

-- 12. Agent request traces
CREATE TABLE IF NOT EXISTS public.agent_runs (
    id BIGSERIAL PRIMARY KEY,
    request_id VARCHAR(64) NOT NULL UNIQUE,
    thread_id VARCHAR(128),
    user_message TEXT NOT NULL,
    final_status VARCHAR(32) NOT NULL,
    latency_ms INTEGER NOT NULL DEFAULT 0,
    evidence_count INTEGER NOT NULL DEFAULT 0,
    token_usage JSONB,
    error_code VARCHAR(128),
    error_message TEXT,
    exception_chain JSONB,
    tool_call_chain JSONB NOT NULL DEFAULT '[]'::jsonb,
    trace_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS public.agent_trace_spans (
    id BIGSERIAL PRIMARY KEY,
    request_id VARCHAR(64) NOT NULL,
    span_id VARCHAR(64) NOT NULL,
    parent_span_id VARCHAR(64),
    span_type VARCHAR(32) NOT NULL,
    name VARCHAR(128) NOT NULL,
    status VARCHAR(32) NOT NULL,
    started_at_ms BIGINT NOT NULL,
    finished_at_ms BIGINT,
    latency_ms INTEGER,
    input_summary JSONB NOT NULL DEFAULT '{}'::jsonb,
    output_summary JSONB NOT NULL DEFAULT '{}'::jsonb,
    error_code VARCHAR(128),
    error_message TEXT,
    exception_chain JSONB NOT NULL DEFAULT '[]'::jsonb,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_agent_trace_spans_request_span UNIQUE (request_id, span_id),
    CONSTRAINT chk_agent_trace_spans_type
        CHECK (span_type IN ('graph_node', 'model_call', 'tool_call', 'guard', 'postprocess', 'context')),
    CONSTRAINT fk_agent_trace_spans_request
        FOREIGN KEY (request_id)
        REFERENCES public.agent_runs(request_id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS public.agent_model_io (
    id BIGSERIAL PRIMARY KEY,
    request_id VARCHAR(64) NOT NULL,
    span_id VARCHAR(64) NOT NULL,
    node VARCHAR(128) NOT NULL,
    provider VARCHAR(64),
    model VARCHAR(160),
    input_messages JSONB NOT NULL DEFAULT '[]'::jsonb,
    raw_output JSONB,
    parsed_output JSONB,
    token_usage JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_agent_model_io_request_span UNIQUE (request_id, span_id),
    CONSTRAINT fk_agent_model_io_span
        FOREIGN KEY (request_id, span_id)
        REFERENCES public.agent_trace_spans(request_id, span_id)
        ON DELETE CASCADE
);

ALTER TABLE public.agent_trace_spans
    DROP CONSTRAINT IF EXISTS chk_agent_trace_spans_type;
ALTER TABLE public.agent_trace_spans
    ADD CONSTRAINT chk_agent_trace_spans_type
        CHECK (span_type IN ('graph_node', 'model_call', 'tool_call', 'guard', 'postprocess', 'context'));

-- 12. Performance Indexes
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
CREATE INDEX IF NOT EXISTS idx_conversation_threads_recent
    ON public.conversation_threads (COALESCE(last_message_at, created_at) DESC);
CREATE INDEX IF NOT EXISTS idx_conversation_threads_channel_recent
    ON public.conversation_threads (channel, COALESCE(last_message_at, created_at) DESC);
CREATE INDEX IF NOT EXISTS idx_conversation_messages_thread_order
    ON public.conversation_messages (thread_id, id);
CREATE INDEX IF NOT EXISTS idx_conversation_messages_thread_created
    ON public.conversation_messages (thread_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_thread_memory_summaries_updated
    ON public.thread_memory_summaries(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_thread_evidence_index_thread_created
    ON public.thread_evidence_index(thread_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_thread_evidence_index_message
    ON public.thread_evidence_index(message_id);
CREATE INDEX IF NOT EXISTS idx_agent_runs_created_at ON public.agent_runs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_runs_status ON public.agent_runs(final_status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_trace_spans_request ON public.agent_trace_spans(request_id, started_at_ms);
CREATE INDEX IF NOT EXISTS idx_agent_trace_spans_parent ON public.agent_trace_spans(request_id, parent_span_id);
CREATE INDEX IF NOT EXISTS idx_agent_trace_spans_type ON public.agent_trace_spans(span_type, name, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_trace_spans_status ON public.agent_trace_spans(status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_model_io_request ON public.agent_model_io(request_id, created_at);
CREATE INDEX IF NOT EXISTS idx_agent_model_io_span ON public.agent_model_io(span_id);
CREATE INDEX IF NOT EXISTS idx_agent_model_io_node ON public.agent_model_io(node, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_news_search_tsv ON public.news_search_index
    USING GIN (search_tsv);
CREATE INDEX IF NOT EXISTS idx_entity_registry_active_type
    ON public.entity_registry(is_active, entity_type);
CREATE INDEX IF NOT EXISTS idx_entity_alias_entity
    ON public.entity_alias(entity_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_entity_alias_entity_lower_alias
    ON public.entity_alias(entity_id, LOWER(alias));
CREATE INDEX IF NOT EXISTS idx_entity_alias_lower_alias
    ON public.entity_alias(LOWER(alias));
CREATE INDEX IF NOT EXISTS idx_entity_alias_trgm
    ON public.entity_alias USING GIN (alias gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_entity_alias_candidate_status
    ON public.entity_alias_candidate(status, confidence DESC, updated_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS idx_entity_alias_candidate_source_lower_alias
    ON public.entity_alias_candidate(source, LOWER(alias));
CREATE INDEX IF NOT EXISTS idx_news_entity_mentions_entity
    ON public.news_entity_mentions(entity_id, confidence DESC);
CREATE INDEX IF NOT EXISTS idx_tech_news_title_trgm
    ON public.tech_news USING GIN (title gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_tech_news_title_cn_trgm
    ON public.tech_news USING GIN (title_cn gin_trgm_ops);

-- NOTE: This index is safe to create on an empty table, but should be rebuilt
-- after the initial data backfill to ensure optimal clustering quality.
-- Run: REINDEX INDEX idx_embedding_vector;
CREATE INDEX IF NOT EXISTS idx_embedding_vector ON public.news_embeddings
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- 13. Update Trigger Function
CREATE OR REPLACE FUNCTION update_modified_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE OR REPLACE FUNCTION public.build_news_search_tsv(
    p_title TEXT,
    p_title_cn TEXT,
    p_summary TEXT
)
RETURNS tsvector AS $$
    SELECT
        setweight(to_tsvector('english', COALESCE(p_title, '')), 'A')
        || setweight(to_tsvector('simple', COALESCE(p_title_cn, '')), 'A')
        || setweight(to_tsvector('english', COALESCE(p_summary, '')), 'B')
        || setweight(to_tsvector('simple', COALESCE(p_summary, '')), 'B');
$$ LANGUAGE sql IMMUTABLE;

CREATE OR REPLACE FUNCTION public.refresh_news_search_index()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO public.news_search_index (url, search_tsv, updated_at)
    VALUES (
        NEW.url,
        public.build_news_search_tsv(NEW.title, NEW.title_cn, NEW.summary),
        NOW()
    )
    ON CONFLICT (url) DO UPDATE
    SET search_tsv = EXCLUDED.search_tsv,
        updated_at = EXCLUDED.updated_at;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS update_tech_news_modtime ON public.tech_news;
CREATE TRIGGER update_tech_news_modtime
    BEFORE UPDATE ON public.tech_news
    FOR EACH ROW
    EXECUTE FUNCTION update_modified_column();

DROP TRIGGER IF EXISTS refresh_news_search_index_on_write ON public.tech_news;
CREATE TRIGGER refresh_news_search_index_on_write
    AFTER INSERT OR UPDATE OF title, title_cn, summary, url ON public.tech_news
    FOR EACH ROW
    EXECUTE FUNCTION public.refresh_news_search_index();

DROP TRIGGER IF EXISTS update_source_registry_modtime ON public.source_registry;
CREATE TRIGGER update_source_registry_modtime
    BEFORE UPDATE ON public.source_registry
    FOR EACH ROW
    EXECUTE FUNCTION update_modified_column();

DROP TRIGGER IF EXISTS update_entity_registry_modtime ON public.entity_registry;
CREATE TRIGGER update_entity_registry_modtime
    BEFORE UPDATE ON public.entity_registry
    FOR EACH ROW
    EXECUTE FUNCTION update_modified_column();

DROP TRIGGER IF EXISTS update_entity_alias_modtime ON public.entity_alias;
CREATE TRIGGER update_entity_alias_modtime
    BEFORE UPDATE ON public.entity_alias
    FOR EACH ROW
    EXECUTE FUNCTION update_modified_column();

DROP TRIGGER IF EXISTS update_entity_alias_candidate_modtime ON public.entity_alias_candidate;
CREATE TRIGGER update_entity_alias_candidate_modtime
    BEFORE UPDATE ON public.entity_alias_candidate
    FOR EACH ROW
    EXECUTE FUNCTION update_modified_column();

DROP TRIGGER IF EXISTS update_conversation_threads_modtime ON public.conversation_threads;
CREATE TRIGGER update_conversation_threads_modtime
    BEFORE UPDATE ON public.conversation_threads
    FOR EACH ROW
    EXECUTE FUNCTION update_modified_column();

INSERT INTO public.news_search_index (url, search_tsv, updated_at)
SELECT
    url,
    public.build_news_search_tsv(title, title_cn, summary),
    NOW()
FROM public.tech_news
ON CONFLICT (url) DO UPDATE
SET search_tsv = EXCLUDED.search_tsv,
    updated_at = EXCLUDED.updated_at;

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

-- [DDL] PostgreSQL Table: tech_news

-- DROP TABLE IF EXISTS public.tech_news CASCADE;

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

-- 2. Dead Letter Queue
CREATE TABLE IF NOT EXISTS public.tech_news_failed (
    id SERIAL PRIMARY KEY,
    original_title TEXT,
    title_cn TEXT,
    url TEXT NOT NULL,
    source_type VARCHAR(50),
    
    error_reason TEXT,
    
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP

    CONSTRAINT unique_failed_url UNIQUE (url)

);

-- 3. System logs
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

-- Index for logs (Logs accumulate fast, indexing time is crucial for cleanup/querying)
CREATE INDEX IF NOT EXISTS idx_logs_created_at ON public.system_logs(created_at DESC);

-- 4. Performance Indexes
CREATE INDEX idx_created_at ON public.tech_news(created_at);
CREATE INDEX idx_updated_at ON public.tech_news(updated_at);
CREATE INDEX idx_points ON public.tech_news(points DESC);
CREATE INDEX idx_sentiment ON public.tech_news(sentiment);

-- 5. Update Trigger Function
CREATE OR REPLACE FUNCTION update_modified_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_tech_news_modtime
    BEFORE UPDATE ON public.tech_news
    FOR EACH ROW
    EXECUTE FUNCTION update_modified_column();
-- Seed first-batch official sources (Google / AWS / Microsoft)
-- Safe to run repeatedly (idempotent upsert)

INSERT INTO public.source_registry (
    source_key,
    source_name,
    source_platform,
    signal_origin,
    fetch_type,
    endpoint,
    is_active,
    priority,
    extra_config
)
VALUES
    (
        'google_ai_blog',
        'Google AI Blog',
        'DirectRSS',
        'Official',
        'rss',
        'https://blog.google/feed/',
        TRUE,
        50,
        '{
          "company":"Google",
          "model_focus":["Gemini"],
          "fallback_endpoints":["https://blog.google/rss/"],
          "keyword_filters":["gemini","google ai","deepmind","llm","model"]
        }'::jsonb
    ),
    (
        'aws_ml_blog',
        'AWS Machine Learning Blog',
        'DirectRSS',
        'Official',
        'rss',
        'https://aws.amazon.com/blogs/machine-learning/feed/',
        TRUE,
        60,
        '{
          "company":"AWS",
          "model_focus":["Amazon Nova","Bedrock","SageMaker"],
          "keyword_filters":["bedrock","sagemaker","nova","foundation model","generative ai"]
        }'::jsonb
    ),
    (
        'microsoft_ai_blog',
        'Microsoft AI Blog',
        'DirectRSS',
        'Official',
        'rss',
        'https://blogs.microsoft.com/?feed=rss2',
        TRUE,
        70,
        '{
          "company":"Microsoft",
          "model_focus":["Copilot","Phi"],
          "fallback_endpoints":["https://blogs.microsoft.com/ai/feed/"],
          "keyword_filters":["copilot","azure ai","microsoft ai","phi","openai"]
        }'::jsonb
    )
ON CONFLICT (source_key) DO UPDATE
SET
    source_name = EXCLUDED.source_name,
    source_platform = EXCLUDED.source_platform,
    signal_origin = EXCLUDED.signal_origin,
    fetch_type = EXCLUDED.fetch_type,
    endpoint = EXCLUDED.endpoint,
    is_active = EXCLUDED.is_active,
    priority = EXCLUDED.priority,
    extra_config = EXCLUDED.extra_config;

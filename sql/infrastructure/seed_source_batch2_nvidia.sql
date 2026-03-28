-- Seed NVIDIA source (stable RSS)
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
        'nvidia_dev_blog',
        'NVIDIA Developer Blog',
        'DirectRSS',
        'Official',
        'rss',
        'https://developer.nvidia.com/blog/feed/',
        TRUE,
        80,
        '{
          "company":"NVIDIA",
          "model_focus":["NIM","NeMo","TensorRT","CUDA","DGX"],
          "keyword_filters":["nvidia ai","nim","nemo","tensorrt","llm","inference","gpu"]
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

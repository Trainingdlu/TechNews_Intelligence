"""In-memory API rate limiting."""

from __future__ import annotations

import os
import time
from threading import Lock

from cachetools import TTLCache
from fastapi import HTTPException

RATE_LIMIT = int(os.getenv("API_RATE_LIMIT", "5"))
RATE_WINDOW = 60
RATE_TRACK_MAX_IPS = max(100, int(os.getenv("API_RATE_TRACK_MAX_IPS", "10000")))


def new_rate_log_cache(*, maxsize: int | None = None, ttl: int | None = None) -> TTLCache[str, list[float]]:
    cache_maxsize = maxsize if maxsize is not None else RATE_TRACK_MAX_IPS
    cache_ttl = ttl if ttl is not None else RATE_WINDOW
    return TTLCache(maxsize=max(1, cache_maxsize), ttl=max(1, cache_ttl), timer=time.time)


request_log: TTLCache[str, list[float]] = new_rate_log_cache()
rate_lock = Lock()


def check_rate_limit(client_ip: str) -> None:
    now = time.time()
    with rate_lock:
        timestamps = [t for t in request_log.get(client_ip, []) if now - t < RATE_WINDOW]
        if len(timestamps) >= RATE_LIMIT:
            request_log[client_ip] = timestamps
            raise HTTPException(status_code=429, detail="请求过于频繁，请稍后重试")
        timestamps.append(now)
        request_log[client_ip] = timestamps

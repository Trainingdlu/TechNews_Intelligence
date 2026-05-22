"""Shared rate-limit backoff for eval runners.

Large-N eval runs fire many gemini-3.1-pro calls back-to-back and hit
429 / ResourceExhausted (RPM or quota). This wraps a call so that a
rate-limit error triggers exponential backoff + retry instead of being
recorded as a permanent failure.
"""

from __future__ import annotations

import random
import time
from typing import Callable, TypeVar

T = TypeVar("T")

_RATE_LIMIT_MARKERS = (
    "429",
    "resourceexhausted",
    "resource_exhausted",
    "resource has been exhausted",
    "rate limit",
    "ratelimit",
    "quota",
    "too many requests",
)


def is_rate_limit(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in _RATE_LIMIT_MARKERS)


def call_with_retry(
    fn: Callable[[], T],
    *,
    max_retries: int = 6,
    base_delay: float = 20.0,
    max_delay: float = 300.0,
    label: str = "",
) -> T:
    """Run fn(); on rate-limit errors, exponential backoff and retry.

    Non-rate-limit exceptions propagate immediately. After max_retries
    rate-limit failures, the last exception propagates.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except BaseException as exc:  # noqa: BLE001
            if not is_rate_limit(exc) or attempt >= max_retries:
                raise
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay) + random.uniform(0, 5)
            tag = f"{label} " if label else ""
            print(f"  [429] {tag}rate-limited; backoff {delay:.0f}s (retry {attempt}/{max_retries - 1})", flush=True)
            time.sleep(delay)
    raise RuntimeError("call_with_retry exhausted retries")  # pragma: no cover

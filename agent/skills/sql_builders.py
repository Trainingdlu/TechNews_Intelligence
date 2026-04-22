"""SQL-related builders — DEPRECATED.

This module previously housed hardcoded topic-expansion dictionaries and
ILIKE-based clause builders.  All production callers have been migrated to
the semantic vector pool (:mod:`agent.skills.semantic_pool`).

The module is retained temporarily so that existing test imports do not
break at import time.  No production code should import from here.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "sql_builders is deprecated – use semantic_pool.fetch_semantic_url_pool instead.",
    DeprecationWarning,
    stacklevel=2,
)

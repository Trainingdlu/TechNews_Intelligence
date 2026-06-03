"""Postgres-backed LangGraph checkpointer (short-term memory + interrupt/HIL).

Builds a process-wide singleton ``PostgresSaver`` from the same DB env the app
already uses. Returns ``None`` when Postgres / deps / DB env are unavailable
(unit tests, no-DB runs) so the graph keeps its prior stateless behavior.
"""

from __future__ import annotations

import os
import threading

_saver = None
_disabled = False
_lock = threading.Lock()


def _dsn() -> str | None:
    name = os.getenv("DB_NAME", "")
    user = os.getenv("DB_USER", "")
    if not (name and user):
        return None
    host = os.getenv("DB_HOST", "127.0.0.1")
    port = os.getenv("DB_PORT", "5555")
    password = os.getenv("DB_PASS", "")
    return f"postgresql://{user}:{password}@{host}:{port}/{name}"


def checkpointer_enabled() -> bool:
    # Keep unit tests isolated from any DB; checkpointer is a runtime concern.
    if os.getenv("PYTEST_CURRENT_TEST"):
        return False
    return os.getenv("AGENT_CHECKPOINTER_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}


def get_checkpointer():
    """Return a process-wide ``PostgresSaver``, or ``None`` when unavailable.

    Any failure (missing DB env, deps, connection) degrades gracefully to
    ``None`` so the agent falls back to its stateless behavior instead of
    crashing.
    """
    global _saver, _disabled
    if _saver is not None or _disabled:
        return _saver
    if not checkpointer_enabled():
        _disabled = True
        return None
    dsn = _dsn()
    if not dsn:
        _disabled = True
        return None
    with _lock:
        if _saver is not None or _disabled:
            return _saver
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
            from psycopg.rows import dict_row
            from psycopg_pool import ConnectionPool

            pool = ConnectionPool(
                conninfo=dsn,
                min_size=1,
                max_size=4,
                open=True,
                timeout=5,
                kwargs={
                    "autocommit": True,
                    "prepare_threshold": 0,
                    "row_factory": dict_row,
                    "connect_timeout": 3,
                },
            )
            saver = PostgresSaver(pool)
            saver.setup()
            _saver = saver
        except Exception as exc:  # noqa: BLE001 - degrade to stateless on any failure
            print(f"[checkpointer] disabled ({type(exc).__name__}: {exc})")
            _disabled = True
            return None
    return _saver

"""数据库连接池管理。"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

import psycopg2
import psycopg2.pool
from psycopg2 import extensions


_pool: psycopg2.pool.ThreadedConnectionPool | None = None


def init_db_pool():
    """根据环境变量初始化连接池。"""
    global _pool
    if _pool is not None:
        return
    _pool = psycopg2.pool.ThreadedConnectionPool(
        minconn=1,
        maxconn=5,
        host=os.getenv("DB_HOST", "127.0.0.1"),
        port=int(os.getenv("DB_PORT", "5555")),
        dbname=os.getenv("DB_NAME", "DB"),
        user=os.getenv("DB_USER", ""),
        password=os.getenv("DB_PASS", ""),
    )


def get_conn():
    """从连接池取连接。"""
    if _pool is None:
        init_db_pool()
    return _pool.getconn()


def _reset_connection_state(conn) -> bool:
    """Return True when connection can be safely returned to pool."""
    if conn is None or conn.closed:
        return False

    try:
        tx_status = conn.get_transaction_status()
    except Exception:
        return False

    if tx_status != extensions.TRANSACTION_STATUS_IDLE:
        try:
            conn.rollback()
        except Exception:
            return False

    return True


def put_conn(conn):
    """归还连接到连接池，回池前强制清理脏事务状态。"""
    if _pool is None or conn is None:
        return

    should_close = not _reset_connection_state(conn)
    try:
        _pool.putconn(conn, close=should_close)
    except Exception:
        try:
            conn.close()
        except Exception:
            pass


@contextmanager
def db_cursor(
    *, commit: bool = False,
) -> Iterator[tuple[psycopg2.extensions.connection, psycopg2.extensions.cursor]]:
    """Yield (conn, cur), and handle rollback/put_conn on all exit paths."""
    conn = get_conn()
    cur = conn.cursor()
    try:
        yield conn, cur
        if commit:
            conn.commit()
    except Exception:
        if conn is not None and not conn.closed:
            try:
                conn.rollback()
            except Exception:
                pass
        raise
    finally:
        try:
            cur.close()
        except Exception:
            pass
        put_conn(conn)


@contextmanager
def db_transaction() -> Iterator[tuple[psycopg2.extensions.connection, psycopg2.extensions.cursor]]:
    """Shortcut context manager for write transactions (auto-commit on success)."""
    with db_cursor(commit=True) as resources:
        yield resources


def close_db_pool():
    """关闭连接池。"""
    global _pool
    if _pool is not None:
        _pool.closeall()
        _pool = None

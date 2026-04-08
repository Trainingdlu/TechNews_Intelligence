"""数据库连接池管理"""

import os
import psycopg2
import psycopg2.pool


_pool: psycopg2.pool.SimpleConnectionPool | None = None


def init_db_pool():
    """根据环境变量初始化连接池"""
    global _pool
    if _pool is not None:
        return
    _pool = psycopg2.pool.SimpleConnectionPool(
        minconn=1,
        maxconn=5,
        host=os.getenv("DB_HOST", "127.0.0.1"),
        port=int(os.getenv("DB_PORT", "5555")),
        dbname=os.getenv("DB_NAME", "DB"),
        user=os.getenv("DB_USER", ""),
        password=os.getenv("DB_PASS", ""),
    )


def get_conn():
    """从连接池取连接"""
    if _pool is None:
        init_db_pool()
    return _pool.getconn()


def put_conn(conn):
    """归还连接到连接池"""
    if _pool is not None:
        _pool.putconn(conn)


def close_db_pool():
    """关闭连接池"""
    global _pool
    if _pool is not None:
        _pool.closeall()
        _pool = None

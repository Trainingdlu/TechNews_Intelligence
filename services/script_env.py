"""Environment helpers for standalone repository scripts."""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _apply_db_env_defaults() -> None:
    """Map deployment Postgres variables to the agent DB_* variables."""
    postgres_port = os.getenv("POSTGRES_PORT")
    postgres_db = os.getenv("POSTGRES_DB")
    postgres_user = os.getenv("POSTGRES_USER")
    postgres_password = os.getenv("POSTGRES_PASSWORD")

    os.environ.setdefault("DB_HOST", "127.0.0.1")
    if postgres_port:
        os.environ.setdefault("DB_PORT", postgres_port)
    if postgres_db:
        os.environ.setdefault("DB_NAME", postgres_db)
    if postgres_user:
        os.environ.setdefault("DB_USER", postgres_user)
    if postgres_password:
        os.environ.setdefault("DB_PASS", postgres_password)


def _load_env_file(path: str | None) -> None:
    if path:
        env_path = Path(path)
        if not env_path.is_absolute():
            env_path = _REPO_ROOT / env_path
        if env_path.exists():
            for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export ") :].strip()
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if not key or key in os.environ:
                    continue
                if (value.startswith('"') and value.endswith('"')) or (
                    value.startswith("'") and value.endswith("'")
                ):
                    value = value[1:-1]
                else:
                    value = value.split("#", 1)[0].strip()
                os.environ[key] = value
    _apply_db_env_defaults()

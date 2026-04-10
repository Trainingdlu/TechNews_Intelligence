from __future__ import annotations

import importlib
import os
import sys
import types
import warnings
from collections.abc import Iterator
from unittest.mock import AsyncMock, MagicMock

import pytest


# Add the project root directory to sys.path so tests can import local packages.
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Ignore known upstream deprecation warning noise from LangGraph/LangChain.
warnings.filterwarnings("ignore", message=".*AgentStatePydantic has been moved.*")


@pytest.fixture()
def agent_dependency_stubs(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Provide lightweight stubs for DB-related optional dependencies."""
    if "services" not in sys.modules:
        try:
            importlib.import_module("services")
        except Exception:
            services_mod = types.ModuleType("services")
            services_mod.__path__ = []
            monkeypatch.setitem(sys.modules, "services", services_mod)

    db_mod = types.ModuleType("services.db")
    db_mod.get_conn = MagicMock()
    db_mod.put_conn = MagicMock()
    db_mod.init_db_pool = MagicMock()
    db_mod.close_db_pool = MagicMock()
    monkeypatch.setitem(sys.modules, "services.db", db_mod)

    psycopg2_mod = types.ModuleType("psycopg2")
    psycopg2_extras = types.ModuleType("psycopg2.extras")
    psycopg2_pool = types.ModuleType("psycopg2.pool")

    psycopg2_extras.Json = lambda value: value
    psycopg2_pool.SimpleConnectionPool = MagicMock()
    psycopg2_pool.ThreadedConnectionPool = MagicMock()
    psycopg2_mod.pool = psycopg2_pool
    psycopg2_mod.extras = psycopg2_extras
    psycopg2_mod.extensions = types.SimpleNamespace(
        TRANSACTION_STATUS_IDLE=0,
        connection=object,
        cursor=object,
    )

    monkeypatch.setitem(sys.modules, "psycopg2", psycopg2_mod)
    monkeypatch.setitem(sys.modules, "psycopg2.extras", psycopg2_extras)
    monkeypatch.setitem(sys.modules, "psycopg2.pool", psycopg2_pool)
    yield


@pytest.fixture()
def email_validator_stub(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Stub email_validator and pydantic metadata lookup for EmailStr tests."""
    email_validator_mod = types.ModuleType("email_validator")

    class EmailNotValidError(ValueError):
        pass

    def _fake_validate_email(email: str, *_args, **_kwargs):
        return types.SimpleNamespace(email=email, normalized=email)

    email_validator_mod.EmailNotValidError = EmailNotValidError
    email_validator_mod.validate_email = _fake_validate_email
    monkeypatch.setitem(sys.modules, "email_validator", email_validator_mod)

    import pydantic.networks as pydantic_networks

    monkeypatch.setattr(pydantic_networks, "version", lambda _name: "2.0.0", raising=False)
    yield


@pytest.fixture()
def telegram_import_stubs(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Stub python-telegram-bot imports for transport-layer bot tests."""
    telegram_mod = types.ModuleType("telegram")

    class Update:  # pragma: no cover - structure-only stub
        pass

    class BotCommand:  # pragma: no cover - structure-only stub
        def __init__(self, command: str, description: str):
            self.command = command
            self.description = description

    telegram_mod.Update = Update
    telegram_mod.BotCommand = BotCommand

    ext_mod = types.ModuleType("telegram.ext")

    class _Builder:  # pragma: no cover - structure-only stub
        def token(self, *_args, **_kwargs):
            return self

        def post_init(self, *_args, **_kwargs):
            return self

        def post_shutdown(self, *_args, **_kwargs):
            return self

        def build(self):
            return types.SimpleNamespace(
                add_handler=lambda *_a, **_k: None,
                run_polling=lambda *_a, **_k: None,
                bot=types.SimpleNamespace(set_my_commands=AsyncMock()),
            )

    class _ContextTypes:  # pragma: no cover - structure-only stub
        DEFAULT_TYPE = object

    ext_mod.ApplicationBuilder = _Builder
    ext_mod.CommandHandler = lambda *_a, **_k: None
    ext_mod.MessageHandler = lambda *_a, **_k: None
    ext_mod.ContextTypes = _ContextTypes
    ext_mod.filters = types.SimpleNamespace(TEXT=1, COMMAND=2)

    monkeypatch.setitem(sys.modules, "telegram", telegram_mod)
    monkeypatch.setitem(sys.modules, "telegram.ext", ext_mod)
    yield

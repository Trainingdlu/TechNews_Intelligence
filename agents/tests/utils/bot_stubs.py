"""Stub modules for importing bot.py without external runtime deps."""

from __future__ import annotations

import sys
import types
from unittest.mock import AsyncMock


def install_bot_import_stubs() -> None:
    """Install minimal stubs for telegram/agent/db modules."""
    if "telegram" not in sys.modules:
        telegram_mod = types.ModuleType("telegram")

        class Update:  # pragma: no cover - structure-only stub
            pass

        class BotCommand:  # pragma: no cover - structure-only stub
            def __init__(self, command: str, description: str):
                self.command = command
                self.description = description

        telegram_mod.Update = Update
        telegram_mod.BotCommand = BotCommand
        sys.modules["telegram"] = telegram_mod

    if "telegram.ext" not in sys.modules:
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
        sys.modules["telegram.ext"] = ext_mod

    if "agent" not in sys.modules:
        agent_mod = types.ModuleType("agent")
        agent_mod.generate_response = lambda _h, _m: "ok"
        agent_mod.generate_response_payload = lambda _h, _m: {"text": "ok", "url_title_map": {}}
        sys.modules["agent"] = agent_mod

    if "db" not in sys.modules:
        db_mod = types.ModuleType("db")
        db_mod.init_db_pool = lambda: None
        db_mod.close_db_pool = lambda: None
        db_mod.get_conn = lambda: None
        db_mod.put_conn = lambda _conn: None
        sys.modules["db"] = db_mod

"""Unit tests for AGENT_WORKFLOW v1/v2 toggle behavior."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch


def _setup_stubs() -> None:
    db_mod = types.ModuleType("db")
    db_mod.get_conn = MagicMock()
    db_mod.put_conn = MagicMock()
    db_mod.init_db_pool = MagicMock()
    db_mod.close_db_pool = MagicMock()
    sys.modules.setdefault("db", db_mod)

    if "psycopg2" not in sys.modules:
        psycopg2_mod = types.ModuleType("psycopg2")
        psycopg2_extras = types.ModuleType("psycopg2.extras")
        psycopg2_pool = types.ModuleType("psycopg2.pool")
        psycopg2_mod.extras = psycopg2_extras
        psycopg2_mod.pool = psycopg2_pool
        psycopg2_extras.Json = lambda x: x
        psycopg2_pool.SimpleConnectionPool = MagicMock()
        sys.modules["psycopg2"] = psycopg2_mod
        sys.modules["psycopg2.extras"] = psycopg2_extras
        sys.modules["psycopg2.pool"] = psycopg2_pool


_setup_stubs()


def test_get_workflow_mode_default_and_aliases() -> None:
    from agents.agent import _get_workflow_mode

    with patch.dict("os.environ", {}, clear=True):
        assert _get_workflow_mode() == "v1"
    with patch.dict("os.environ", {"AGENT_WORKFLOW": "v2"}, clear=True):
        assert _get_workflow_mode() == "v2"
    with patch.dict("os.environ", {"AGENT_WORKFLOW": "skills"}, clear=True):
        assert _get_workflow_mode() == "v2"
    with patch.dict("os.environ", {"AGENT_WORKFLOW": "react"}, clear=True):
        assert _get_workflow_mode() == "v1"


def test_generate_response_core_uses_v2_path_when_enabled() -> None:
    from agents.agent import _generate_response_core

    with (
        patch.dict("os.environ", {"AGENT_WORKFLOW": "v2"}),
        patch("agents.agent._generate_workflow_v2", return_value=("structured answer", {"https://a.com"})) as mock_v2,
        patch("agents.agent._generate_react", side_effect=AssertionError("should not call react in v2 mode")),
    ):
        text, urls = _generate_response_core([], "OpenAI trend")

    assert text == "structured answer"
    assert urls == {"https://a.com"}
    assert mock_v2.called

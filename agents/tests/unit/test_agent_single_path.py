"""Unit tests for unified ReAct runtime behavior.

Legacy AGENT_WORKFLOW v1/v2 switching was removed. These tests ensure we keep
single-path execution through _generate_react().
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest


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


@pytest.mark.parametrize("workflow_flag", ["v1", "v2", "skills", "react", "anything"])
def test_generate_response_core_always_uses_react(workflow_flag: str) -> None:
    from agents.agent import _generate_response_core

    with (
        patch.dict("os.environ", {"AGENT_WORKFLOW": workflow_flag}),
        patch("agents.agent._generate_react", return_value=("structured answer", {"https://a.com"})) as mock_react,
    ):
        text, urls = _generate_response_core([], "OpenAI trend")

    assert text == "structured answer"
    assert urls == {"https://a.com"}
    assert mock_react.called


def test_generate_response_core_empty_evidence_raises_agent_error() -> None:
    from agents.agent import AgentGenerationError, _generate_response_core

    with patch("agents.agent._generate_react", return_value=("answer-without-evidence", set())):
        with pytest.raises(AgentGenerationError) as ei:
            _generate_response_core([], "OpenAI trend")
    assert ei.value.code == "react_empty_evidence_blocked"


def test_legacy_workflow_symbols_removed() -> None:
    import agents.agent as agent_mod

    assert not hasattr(agent_mod, "_get_workflow_mode")
    assert not hasattr(agent_mod, "_generate_workflow_v2")
    assert not hasattr(agent_mod, "generate_response_v2")


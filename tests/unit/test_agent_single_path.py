"""Unit tests for unified ReAct runtime behavior.

Legacy AGENT_WORKFLOW v1/v2 switching was removed. These tests ensure we keep
single-path execution through _generate_react().
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

pytestmark = pytest.mark.usefixtures("agent_dependency_stubs")


@pytest.mark.parametrize("workflow_flag", ["v1", "v2", "skills", "react", "anything"])
def test_generate_response_core_always_uses_react(workflow_flag: str) -> None:
    from agent.agent import _generate_response_core

    with (
        patch.dict("os.environ", {"AGENT_WORKFLOW": workflow_flag}),
        patch("agent.agent._generate_react", return_value=("structured answer", {"https://a.com"})) as mock_react,
    ):
        text, urls = _generate_response_core([], "OpenAI trend")

    assert text == "structured answer"
    assert urls == {"https://a.com"}
    assert mock_react.called


def test_generate_response_core_empty_evidence_raises_agent_error() -> None:
    from agent.agent import AgentGenerationError, _generate_response_core

    with patch("agent.agent._generate_react", return_value=("answer-without-evidence", set())):
        with pytest.raises(AgentGenerationError) as ei:
            _generate_response_core([], "OpenAI trend")
    assert ei.value.code == "react_empty_evidence_blocked"


def test_legacy_workflow_symbols_removed() -> None:
    import agent.agent as agent_mod

    assert not hasattr(agent_mod, "_get_workflow_mode")
    assert not hasattr(agent_mod, "_generate_workflow_v2")
    assert not hasattr(agent_mod, "generate_response_v2")


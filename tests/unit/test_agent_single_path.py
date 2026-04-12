"""Unit tests for the current default response-generation path."""

from __future__ import annotations

from unittest.mock import patch

import pytest

pytestmark = pytest.mark.usefixtures("agent_dependency_stubs")


def test_generate_response_core_uses_react_path_by_default() -> None:
    from agent.agent import _generate_response_core

    with patch("agent.agent._generate_react", return_value=("structured answer", {"https://a.com"})) as mock_react:
        text, urls = _generate_response_core([], "OpenAI trend")

    assert text == "structured answer"
    assert urls == {"https://a.com"}
    assert mock_react.called


def test_generate_response_core_empty_evidence_requires_clarification() -> None:
    from agent.agent import _generate_response_core
    from agent.clarification import ClarificationRequiredError

    with patch("agent.agent._generate_react", return_value=("answer-without-evidence", set())):
        with pytest.raises(ClarificationRequiredError) as ei:
            _generate_response_core([], "OpenAI trend")
    payload = ei.value.clarification.to_dict()
    assert payload["kind"] == "clarification_required"
    assert payload["reason"] == "insufficient_evidence"
    assert payload["question"]

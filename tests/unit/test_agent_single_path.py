"""Unit tests for the current default response-generation path."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.usefixtures("agent_dependency_stubs")


def test_generate_response_core_uses_custom_graph_path() -> None:
    from agent.agent import _generate_response_core
    from agent.graph.state import AgentRunResult

    with patch("agent.agent.invoke_custom_graph", return_value=AgentRunResult("structured answer", ["https://a.com"])) as mock_graph:
        text, urls, _ = _generate_response_core([], "OpenAI trend")

    assert text == "structured answer"
    assert urls == ["https://a.com"]
    assert mock_graph.called


def test_generate_response_core_empty_evidence_requires_clarification() -> None:
    from agent.agent import _generate_response_core
    from agent.clarification import ClarificationRequiredError
    from agent.graph.state import AgentRunResult

    with patch("agent.agent.invoke_custom_graph", return_value=AgentRunResult("answer-without-evidence", [])):
        with pytest.raises(ClarificationRequiredError) as ei:
            _generate_response_core([], "OpenAI trend")
    payload = ei.value.clarification.to_dict()
    assert payload["kind"] == "clarification_required"
    assert payload["reason"] == "insufficient_evidence"
    assert payload["question"]


def test_generate_response_core_uses_dynamic_insufficient_evidence_clarification() -> None:
    from agent.agent import _generate_response_core
    from agent.clarification import ClarificationRequiredError
    from agent.graph.state import AgentRunResult

    class _FakeModel:
        def invoke(self, _messages):  # noqa: ANN001
            return SimpleNamespace(
                content=(
                    '{"question":"我没有找到足够支撑 OpenAI 与 Yubico 安全计划的证据。'
                    '你想优先补充原文 URL，还是限定最近 30 天继续查？",'
                    '"hints":["提供这条安全计划的原文 URL",'
                    '"限定最近 30 天，只查 OpenAI 账户安全相关报道"]}'
                )
            )

    with (
        patch.dict("os.environ", {"AGENT_DYNAMIC_CLARIFICATION_ENABLED": "true"}),
        patch("agent.agent.invoke_custom_graph", return_value=AgentRunResult("answer-without-evidence", [])),
        patch("agent.agent._get_accumulated_tool_calls", return_value={"query_news"}),
        patch("agent.agent._build_chat_model", return_value=_FakeModel()),
    ):
        with pytest.raises(ClarificationRequiredError) as ei:
            _generate_response_core([], "详细说说 OpenAI 联手 Yubico 推出的 ChatGPT 高级账户安全计划")

    payload = ei.value.clarification.to_dict()
    assert payload["reason"] == "insufficient_evidence"
    assert "OpenAI" in payload["question"]
    assert "Yubico" in payload["question"]
    assert payload["hints"] == [
        "提供这条安全计划的原文 URL",
        "限定最近 30 天，只查 OpenAI 账户安全相关报道",
    ]

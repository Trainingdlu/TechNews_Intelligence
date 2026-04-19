"""Tests for agent module - ReAct architecture."""

from __future__ import annotations

import re
from unittest.mock import patch
import pytest

pytestmark = pytest.mark.usefixtures("agent_dependency_stubs")


# ---------------------------------------------------------------------------
# Tests: metrics module
# ---------------------------------------------------------------------------
class TestMetrics:
    """Test the simplified ReAct metrics system."""

    def test_metrics_inc_and_snapshot(self):
        from agent.core.metrics import metrics_inc, get_route_metrics_snapshot, reset_route_metrics

        reset_route_metrics()
        metrics_inc("requests_total", 3)
        metrics_inc("react_attempts", 3)
        metrics_inc("react_success", 2)
        metrics_inc("react_error", 1)

        snap = get_route_metrics_snapshot()
        assert snap["requests_total"] == 3
        assert snap["react_attempts"] == 3
        assert snap["react_success"] == 2
        assert snap["react_error"] == 1
        assert abs(snap["success_rate"] - 2 / 3) < 0.01
        assert abs(snap["error_rate"] - 1 / 3) < 0.01

    def test_reset_metrics(self):
        from agent.core.metrics import metrics_inc, get_route_metrics_snapshot, reset_route_metrics

        metrics_inc("requests_total", 10)
        reset_route_metrics()
        snap = get_route_metrics_snapshot()
        assert snap["requests_total"] == 0
        assert snap["react_attempts"] == 0

    def test_metrics_disabled(self):
        from agent.core.metrics import metrics_inc, get_route_metrics_snapshot, reset_route_metrics

        reset_route_metrics()
        with patch.dict("os.environ", {"AGENT_ROUTE_METRICS": "false"}):
            metrics_inc("requests_total", 5)
        snap = get_route_metrics_snapshot()
        assert snap["requests_total"] == 0


# ---------------------------------------------------------------------------
# Tests: evidence module
# ---------------------------------------------------------------------------
class TestEvidence:
    """Test evidence.py utilities (unchanged by refactoring)."""

    def test_extract_urls(self):
        from agent.core.evidence import extract_urls

        text = "See https://example.com and https://other.org/path for details."
        urls = extract_urls(text)
        assert "https://example.com" in urls
        assert "https://other.org/path" in urls

    def test_contains_valid_url_in_body_allows_trailing_slash_variant(self):
        from agent.core.evidence import contains_valid_url_in_body

        text = "See source https://techcrunch.com/2026/04/16/openai-update/ for details."
        valid_urls = {"https://techcrunch.com/2026/04/16/openai-update"}
        assert contains_valid_url_in_body(text, valid_urls)

    def test_contains_cjk(self):
        from agent.core.evidence import contains_cjk

        assert contains_cjk("你好世界")
        assert not contains_cjk("hello world")

    def test_decorate_with_no_urls(self):
        from agent.core.evidence import decorate_response_with_sources

        text = "This is a plain text response."
        result, title_map = decorate_response_with_sources(text, "test question")
        assert result == text
        assert title_map == {}

    def test_decorate_fallback_respects_valid_url_order(self):
        from agent.core.evidence import decorate_response_with_sources

        text = "Alpha finding [1]. Beta finding [2]."
        valid_urls = ["https://b.example.com", "https://a.example.com"]
        result, _ = decorate_response_with_sources(text, "test question", valid_urls=valid_urls)

        idx_b = result.find("- [1] https://b.example.com")
        idx_a = result.find("- [2] https://a.example.com")
        assert idx_b != -1 and idx_a != -1
        assert idx_b < idx_a

    def test_normalize_parenthesized_citation(self):
        from agent.core.evidence import normalize_inline_citation_styles

        text = "Key finding ([1]) and another（[2]）."
        out = normalize_inline_citation_styles(text)
        assert "([1])" not in out
        assert "（[2]）" not in out
        assert "[1]" in out
        assert "[2]" in out
    def test_normalize_plain_parenthesized_numeric_citations(self):
        from agent.core.evidence import normalize_inline_citation_styles

        text = "A finding (1) and grouped refs (1,2) and full-width （2，3）."
        out = normalize_inline_citation_styles(text)
        assert "(1)" not in out
        assert "(1,2)" not in out
        assert "（2，3）" not in out
        assert "[1]" in out
        assert "[1][2]" in out
        assert "[2][3]" in out
    def test_normalize_source_hash_citation(self):
        from agent.core.evidence import normalize_inline_citation_styles

        text = "Evidence [Google] #3 and [Meta] ＃12 are both cited."
        out = normalize_inline_citation_styles(text)
        assert "[Google] #3" not in out
        assert "[Meta] ＃12" not in out
        assert "[3]" in out
        assert "[12]" in out

    def test_normalize_nested_citation(self):
        from agent.core.evidence import normalize_inline_citation_styles

        text = "Model updates [[1]] and ([2]) and [#3] should normalize."
        out = normalize_inline_citation_styles(text)
        assert "[[1]]" not in out
        assert "([2])" not in out
        assert "[#3]" not in out
        assert "[1]" in out
        assert "[2]" in out
        assert "[3]" in out

    def test_decorate_response_does_not_emit_template_literals(self):
        from agent.core.evidence import decorate_response_with_sources

        text = "Google update https://a.com/1 and OpenAI update https://b.com/2."
        out, _ = decorate_response_with_sources(text, "对比谷歌和openai")
        assert "{compact_body" not in out
        assert "{source_section" not in out
        assert "{url}" not in out
        assert "## 来源" in out

    def test_has_inline_citation_in_body_ignores_sources_section(self):
        from agent.core.evidence import has_inline_citation_in_body

        text = "Main content without refs.\n\n## Sources\n- [1] https://example.com"
        assert not has_inline_citation_in_body(text)

    def test_decorate_parenthesized_raw_urls_into_numbered_citations(self):
        from agent.core.evidence import decorate_response_with_sources

        text = (
            "OpenAI released an update（https://example.com/openai）。Google responded with its own launch "
            "(https://example.com/google)."
        )
        out, _ = decorate_response_with_sources(text, "recent ai updates")

        body = out.split("## Sources", 1)[0]
        assert "https://example.com/openai" not in body
        assert "https://example.com/google" not in body
        assert "([1])" not in body
        assert "([2])" not in body
        assert "（[1]）" not in body
        assert "（[2]）" not in body
        assert "[1]" in body
        assert "[2]" in body
        assert "- [1] https://example.com/openai" in out
        assert "- [2] https://example.com/google" in out


class TestPromptContract:
    def test_prompt_enforces_sentence_tail_url_mode(self):
        from agent.prompts import SYSTEM_INSTRUCTION

        assert "append the raw URL at sentence end using parentheses" in SYSTEM_INSTRUCTION
        assert "Do NOT output numeric citations like [1], [2]" in SYSTEM_INSTRUCTION


# ---------------------------------------------------------------------------
# Tests: agent module structure
# ---------------------------------------------------------------------------
class TestAgentStructure:
    """Test that agent.py exports are correct and import cleanly."""

    def test_langchain_tools_list(self):
        from agent.agent import LANGCHAIN_TOOLS

        tool_names = {t.name for t in LANGCHAIN_TOOLS}
        expected = {
            "search_news",
            "read_news_content",
            "get_db_stats",
            "list_topics",
            "query_news",
            "trend_analysis",
            "compare_sources",
            "compare_topics",
            "build_timeline",
            "analyze_landscape",
            "fulltext_batch",
        }
        assert tool_names == expected

    def test_no_legacy_imports(self):
        """Ensure deleted modules are not referenced."""
        import agent.agent as agent_mod
        with open(agent_mod.__file__, "r", encoding="utf-8") as f:
            source = f.read()
        assert "from core.router" not in source
        assert "from .core.router" not in source
        assert "from core.pipelines" not in source
        assert "from .core.pipelines" not in source
        assert "LEGACY_TOOLS" not in source
        assert "_generate_legacy" not in source
        assert "deepseek" not in source.lower()

    def test_generate_response_exists(self):
        from agent.agent import generate_response, generate_response_payload, create_agent_chat
        assert callable(generate_response)
        assert callable(generate_response_payload)
        assert callable(create_agent_chat)


# ---------------------------------------------------------------------------
# Tests: history conversion
# ---------------------------------------------------------------------------
class TestHistoryConversion:
    """Test message format conversion utilities."""

    def test_history_to_messages(self):
        from agent.agent import _history_to_messages

        history = [
            {"role": "user", "parts": [{"text": "Hello"}]},
            {"role": "model", "parts": [{"text": "Hi there"}]},
            {"role": "user", "parts": [{"text": "What about AI?"}]},
        ]
        messages = _history_to_messages(history)
        assert len(messages) == 3
        from langchain_core.messages import HumanMessage, AIMessage
        assert isinstance(messages[0], HumanMessage)
        assert isinstance(messages[1], AIMessage)
        assert isinstance(messages[2], HumanMessage)

    def test_empty_history(self):
        from agent.agent import _history_to_messages

        assert _history_to_messages([]) == []
        assert _history_to_messages(None) == []

    def test_coerce_to_text_string(self):
        from agent.agent import _coerce_to_text

        assert _coerce_to_text("hello") == "hello"

    def test_coerce_to_text_list(self):
        from agent.agent import _coerce_to_text

        result = _coerce_to_text([{"text": "part1"}, {"text": "part2"}])
        assert "part1" in result
        assert "part2" in result

    def test_coerce_to_text_none(self):
        from agent.agent import _coerce_to_text

        assert _coerce_to_text(None) == ""


# ---------------------------------------------------------------------------
# Tests: post-processing
# ---------------------------------------------------------------------------
class TestPostProcessing:
    """Test lightweight post-processing safety net."""

    def test_strip_leadin_chinese(self):
        from agent.agent import _strip_generic_analysis_leadin

        text = "当然，下面是分析\n\n## 核心发现\n内容"
        result = _strip_generic_analysis_leadin(text)
        assert result.startswith("## 核心发现")

    def test_strip_leadin_english(self):
        from agent.agent import _strip_generic_analysis_leadin

        text = "Sure, here is the analysis summary\n\nThe key finding is..."
        result = _strip_generic_analysis_leadin(text)
        assert "key finding" in result

    def test_no_strip_normal_text(self):
        from agent.agent import _strip_generic_analysis_leadin

        text = "## AI Industry Overview\nDetailed analysis below."
        result = _strip_generic_analysis_leadin(text)
        assert result == text

    def test_strip_leadin_empty(self):
        from agent.agent import _strip_generic_analysis_leadin

        assert _strip_generic_analysis_leadin("") == ""
        assert _strip_generic_analysis_leadin(None) == ""


# ---------------------------------------------------------------------------
# Tests: package exports
# ---------------------------------------------------------------------------
class TestPackageExports:
    """Test that the agent package exports work correctly."""

    def test_package_all(self):
        import agent
        expected = {
            "AgentGenerationError",
            "init_db_pool",
            "close_db_pool",
            "create_agent_chat",
            "generate_response",
            "generate_response_eval_payload",
            "generate_response_payload",
            "get_last_tool_calls_snapshot",
            "get_route_metrics_snapshot",
            "reset_route_metrics",
        }
        assert set(agent.__all__) == expected


# ---------------------------------------------------------------------------
# Tests: prompt injection and friendly error boundaries
# ---------------------------------------------------------------------------
class TestAgentSafety:
    """Ensure prompt injection is mandatory and runtime errors are user-friendly."""

    def test_build_react_prompt_kwargs_prefers_prompt(self):
        import agent.agent as agent_mod

        def _fake_create_react_agent(model, tools, prompt):  # noqa: ANN001
            return {"model": model, "tools": tools, "prompt": prompt}

        with patch.object(agent_mod, "create_react_agent", _fake_create_react_agent):
            kwargs, key = agent_mod._build_react_prompt_kwargs()
        assert key == "prompt"
        assert "prompt" in kwargs

    def test_build_react_prompt_kwargs_supports_state_modifier(self):
        import agent.agent as agent_mod

        def _fake_create_react_agent(model, tools, state_modifier):  # noqa: ANN001
            return {"model": model, "tools": tools, "state_modifier": state_modifier}

        with patch.object(agent_mod, "create_react_agent", _fake_create_react_agent):
            kwargs, key = agent_mod._build_react_prompt_kwargs()
        assert key == "state_modifier"
        assert "state_modifier" in kwargs

    def test_generate_response_core_handles_recursion_error(self):
        from agent.agent import AgentGenerationError, _generate_response_core
        from agent.core.metrics import get_route_metrics_snapshot, reset_route_metrics

        reset_route_metrics()
        with patch("agent.agent._generate_react", side_effect=RuntimeError("GraphRecursionError: limit reached")):
            with pytest.raises(AgentGenerationError) as ei:
                _generate_response_core([], "analyze AI trend in recent 10 days")
        assert "超时" in str(ei.value)
        snapshot = get_route_metrics_snapshot()
        assert snapshot["react_error"] == 1
        assert snapshot["react_recursion_limit_hit"] == 1

    def test_generate_response_core_handles_transient_error(self):
        from agent.agent import AgentGenerationError, _generate_response_core
        from agent.core.metrics import get_route_metrics_snapshot, reset_route_metrics

        reset_route_metrics()
        with patch("agent.agent._generate_react", side_effect=RuntimeError("429 resource exhausted")):
            with pytest.raises(AgentGenerationError) as ei:
                _generate_response_core([], "analyze AI trend in recent 10 days")
        assert "暂时不可用" in str(ei.value)
        snapshot = get_route_metrics_snapshot()
        assert snapshot["react_error"] == 1



    def test_generate_response_blocks_when_body_has_no_inline_citations(self):
        from agent.agent import AgentGenerationError, generate_response
        from agent.core.metrics import get_route_metrics_snapshot, reset_route_metrics

        reset_route_metrics()
        with patch("agent.agent._run_generation_core", return_value=("Main analysis body.", {"https://a.example.com"})):
            with patch(
                "agent.agent._decorate_response_with_sources",
                return_value=("Main analysis body.\n\n## Sources\n- [1] https://a.example.com", {}),
            ):
                with patch.dict("os.environ", {"AGENT_STRICT_INLINE_CITATIONS": "true"}):
                    with pytest.raises(AgentGenerationError) as ei:
                        generate_response([], "最近AI动态")
        assert ei.value.code == "react_inline_citation_missing"
        snapshot = get_route_metrics_snapshot()
        assert snapshot.get("react_inline_citation_blocked", 0) == 1

    def test_generate_response_allows_when_body_has_valid_source_url(self):
        from agent.agent import generate_response

        expected = "Main analysis [1].\n\n## Sources\n- [1] https://a.example.com"
        with patch("agent.agent._run_generation_core", return_value=("Main analysis cites https://a.example.com.", {"https://a.example.com"})):
            with patch("agent.agent._decorate_response_with_sources", return_value=(expected, {})):
                with patch.dict("os.environ", {"AGENT_STRICT_INLINE_CITATIONS": "true"}):
                    out = generate_response([], "最近AI动态")
        assert out == expected


    def test_generate_response_payload_includes_citation_urls(self):
        from agent.agent import generate_response_payload

        expected = "Main analysis [1].\n\n## Sources\n- [1] https://a.example.com"
        with patch("agent.agent._run_generation_core", return_value=("Main analysis cites https://a.example.com.", {"https://a.example.com"})):
            with patch("agent.agent._decorate_response_with_sources", return_value=(expected, {})):
                with patch.dict("os.environ", {"AGENT_STRICT_INLINE_CITATIONS": "true"}):
                    payload = generate_response_payload([], "recent ai updates")
        assert payload["kind"] == "answer"
        assert payload["citation_urls"] == ["https://a.example.com"]

    def test_generate_response_blocks_when_output_has_url_outside_valid_set(self):
        from agent.agent import AgentGenerationError, generate_response
        from agent.core.metrics import get_route_metrics_snapshot, reset_route_metrics

        reset_route_metrics()
        with patch(
            "agent.agent._run_generation_core",
            return_value=("Main analysis cites https://evil.example.com/x", {"https://a.example.com"}),
        ):
            with pytest.raises(AgentGenerationError) as ei:
                generate_response([], "recent ai updates")

        assert ei.value.code == "react_url_outside_valid_set"
        snapshot = get_route_metrics_snapshot()
        assert snapshot.get("react_url_outside_valid_set_blocked", 0) == 1

    def test_generate_response_allows_url_subset_of_valid_urls(self):
        from agent.agent import generate_response

        with patch(
            "agent.agent._run_generation_core",
            return_value=("Main analysis references https://a.example.com only.", {"https://a.example.com", "https://b.example.com"}),
        ):
            with patch(
                "agent.agent._decorate_response_with_sources",
                return_value=("Main analysis [1].\n\n## Sources\n- [1] https://a.example.com", {}),
            ):
                out = generate_response([], "recent ai updates")

        assert "[1]" in out

    def test_generate_response_allows_trailing_slash_variant_url_match(self):
        from agent.agent import generate_response

        expected = "Main analysis [1].\n\n## Sources\n- [1] https://techcrunch.com/2026/04/16/openai-update"
        with patch(
            "agent.agent._run_generation_core",
            return_value=(
                "Main analysis references https://techcrunch.com/2026/04/16/openai-update/.",
                {"https://techcrunch.com/2026/04/16/openai-update"},
            ),
        ):
            with patch("agent.agent._decorate_response_with_sources", return_value=(expected, {})):
                out = generate_response([], "recent ai updates")

        assert out == expected


    def test_generate_response_core_allows_smalltalk_without_evidence_when_no_tools(self):
        from agent.agent import _generate_response_core

        with patch("agent.agent._generate_react", return_value=("Hello, I can help with tech intelligence analysis.", set())):
            with patch("agent.agent._get_accumulated_tool_calls", return_value=set()):
                text, urls = _generate_response_core([], "hello")
        assert "help" in text.lower()
        assert urls == set()

    def test_generate_response_core_allows_capability_question_without_evidence_when_no_tools(self):
        from agent.agent import _generate_response_core

        with patch("agent.agent._generate_react", return_value=("I can do trend, comparison, timeline and landscape analysis.", set())):
            with patch("agent.agent._get_accumulated_tool_calls", return_value=set()):
                text, urls = _generate_response_core([], "assistant what can you do")
        assert "compar" in text.lower()
        assert urls == set()

    def test_generate_response_core_blocks_analysis_without_evidence_when_no_tools(self):
        from agent.agent import _generate_response_core
        from agent.clarification import ClarificationRequiredError

        with patch("agent.agent._generate_react", return_value=("analysis result", set())):
            with patch("agent.agent._get_accumulated_tool_calls", return_value=set()):
                with pytest.raises(ClarificationRequiredError) as ei:
                    _generate_response_core([], "analyze recent 30 days AI trend")
        payload = ei.value.clarification.to_dict()
        assert payload["kind"] == "clarification_required"
        assert payload["reason"] == "insufficient_evidence"

    def test_generate_response_core_blocks_when_tools_used_but_no_evidence(self):
        from agent.agent import _generate_response_core
        from agent.clarification import ClarificationRequiredError

        with patch("agent.agent._generate_react", return_value=("tool output empty", set())):
            with patch("agent.agent._get_accumulated_tool_calls", return_value={"query_news"}):
                with pytest.raises(ClarificationRequiredError) as ei:
                    _generate_response_core([], "hello")
        payload = ei.value.clarification.to_dict()
        assert payload["kind"] == "clarification_required"
        assert payload["reason"] in {"insufficient_evidence", "ambiguous_scope"}


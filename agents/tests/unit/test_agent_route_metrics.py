"""Tests for agent module — ReAct architecture."""

from __future__ import annotations

import re
import sys
import types
from unittest.mock import MagicMock, patch
import pytest


# ---------------------------------------------------------------------------
# Stub external dependencies so tests run without DB/API keys
# ---------------------------------------------------------------------------
def _setup_stubs():
    """Create stub modules for dependencies that need real infrastructure."""
    # Stub db module
    db_mod = types.ModuleType("db")
    db_mod.get_conn = MagicMock()
    db_mod.put_conn = MagicMock()
    db_mod.init_db_pool = MagicMock()
    db_mod.close_db_pool = MagicMock()
    sys.modules.setdefault("db", db_mod)

    # Stub psycopg2 if not installed
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


# ---------------------------------------------------------------------------
# Tests: metrics module
# ---------------------------------------------------------------------------
class TestMetrics:
    """Test the simplified ReAct metrics system."""

    def test_metrics_inc_and_snapshot(self):
        from agents.core.metrics import metrics_inc, get_route_metrics_snapshot, reset_route_metrics

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
        from agents.core.metrics import metrics_inc, get_route_metrics_snapshot, reset_route_metrics

        metrics_inc("requests_total", 10)
        reset_route_metrics()
        snap = get_route_metrics_snapshot()
        assert snap["requests_total"] == 0
        assert snap["react_attempts"] == 0

    def test_metrics_disabled(self):
        from agents.core.metrics import metrics_inc, get_route_metrics_snapshot, reset_route_metrics

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
        from agents.core.evidence import extract_urls

        text = "See https://example.com and https://other.org/path for details."
        urls = extract_urls(text)
        assert "https://example.com" in urls
        assert "https://other.org/path" in urls

    def test_contains_cjk(self):
        from agents.core.evidence import contains_cjk

        assert contains_cjk("你好世界")
        assert not contains_cjk("hello world")

    def test_decorate_with_no_urls(self):
        from agents.core.evidence import decorate_response_with_sources

        text = "This is a plain text response."
        result, title_map = decorate_response_with_sources(text, "test question")
        assert result == text
        assert title_map == {}


# ---------------------------------------------------------------------------
# Tests: agent module structure
# ---------------------------------------------------------------------------
class TestAgentStructure:
    """Test that agent.py exports are correct and import cleanly."""

    def test_langchain_tools_list(self):
        from agents.agent import LANGCHAIN_TOOLS

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
        import agents.agent as agent_mod
        source = open(agent_mod.__file__, "r", encoding="utf-8").read()
        assert "from core.router" not in source
        assert "from .core.router" not in source
        assert "from core.pipelines" not in source
        assert "from .core.pipelines" not in source
        assert "LEGACY_TOOLS" not in source
        assert "_generate_legacy" not in source
        assert "deepseek" not in source.lower()

    def test_generate_response_exists(self):
        from agents.agent import generate_response, generate_response_payload, create_agent_chat
        assert callable(generate_response)
        assert callable(generate_response_payload)
        assert callable(create_agent_chat)


# ---------------------------------------------------------------------------
# Tests: history conversion
# ---------------------------------------------------------------------------
class TestHistoryConversion:
    """Test message format conversion utilities."""

    def test_history_to_messages(self):
        from agents.agent import _history_to_messages

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
        from agents.agent import _history_to_messages

        assert _history_to_messages([]) == []
        assert _history_to_messages(None) == []

    def test_coerce_to_text_string(self):
        from agents.agent import _coerce_to_text

        assert _coerce_to_text("hello") == "hello"

    def test_coerce_to_text_list(self):
        from agents.agent import _coerce_to_text

        result = _coerce_to_text([{"text": "part1"}, {"text": "part2"}])
        assert "part1" in result
        assert "part2" in result

    def test_coerce_to_text_none(self):
        from agents.agent import _coerce_to_text

        assert _coerce_to_text(None) == ""


# ---------------------------------------------------------------------------
# Tests: post-processing
# ---------------------------------------------------------------------------
class TestPostProcessing:
    """Test lightweight post-processing safety net."""

    def test_strip_leadin_chinese(self):
        from agents.agent import _strip_generic_analysis_leadin

        text = "好的，作为一名资深科技分析师，以下是分析\n\n## 核心发现\n内容"
        result = _strip_generic_analysis_leadin(text)
        assert result.startswith("## 核心发现")

    def test_strip_leadin_english(self):
        from agents.agent import _strip_generic_analysis_leadin

        text = "Sure, here is the analysis summary\n\nThe key finding is..."
        result = _strip_generic_analysis_leadin(text)
        assert "key finding" in result

    def test_no_strip_normal_text(self):
        from agents.agent import _strip_generic_analysis_leadin

        text = "## AI Industry Overview\nDetailed analysis below."
        result = _strip_generic_analysis_leadin(text)
        assert result == text

    def test_strip_leadin_empty(self):
        from agents.agent import _strip_generic_analysis_leadin

        assert _strip_generic_analysis_leadin("") == ""
        assert _strip_generic_analysis_leadin(None) == ""


# ---------------------------------------------------------------------------
# Tests: package exports
# ---------------------------------------------------------------------------
class TestPackageExports:
    """Test that the agents package exports work correctly."""

    def test_package_all(self):
        import agents
        expected = {
            "AgentGenerationError",
            "init_db_pool",
            "close_db_pool",
            "create_agent_chat",
            "generate_response",
            "generate_response_payload",
            "get_route_metrics_snapshot",
            "reset_route_metrics",
        }
        assert set(agents.__all__) == expected


# ---------------------------------------------------------------------------
# Tests: prompt injection and friendly error boundaries
# ---------------------------------------------------------------------------
class TestAgentSafety:
    """Ensure prompt injection is mandatory and runtime errors are user-friendly."""

    def test_build_react_prompt_kwargs_prefers_prompt(self):
        import agents.agent as agent_mod

        def _fake_create_react_agent(model, tools, prompt):  # noqa: ANN001
            return {"model": model, "tools": tools, "prompt": prompt}

        with patch.object(agent_mod, "create_react_agent", _fake_create_react_agent):
            kwargs, key = agent_mod._build_react_prompt_kwargs()
        assert key == "prompt"
        assert "prompt" in kwargs

    def test_build_react_prompt_kwargs_supports_state_modifier(self):
        import agents.agent as agent_mod

        def _fake_create_react_agent(model, tools, state_modifier):  # noqa: ANN001
            return {"model": model, "tools": tools, "state_modifier": state_modifier}

        with patch.object(agent_mod, "create_react_agent", _fake_create_react_agent):
            kwargs, key = agent_mod._build_react_prompt_kwargs()
        assert key == "state_modifier"
        assert "state_modifier" in kwargs

    def test_generate_response_core_handles_recursion_error(self):
        from agents.agent import AgentGenerationError, _generate_response_core
        from agents.core.metrics import get_route_metrics_snapshot, reset_route_metrics

        reset_route_metrics()
        with patch("agents.agent._generate_react", side_effect=RuntimeError("GraphRecursionError: limit reached")):
            with pytest.raises(AgentGenerationError) as ei:
                _generate_response_core([], "分析最近30天AI局势")
        assert "超时" in str(ei.value)
        snapshot = get_route_metrics_snapshot()
        assert snapshot["react_error"] == 1
        assert snapshot["react_recursion_limit_hit"] == 1

    def test_generate_response_core_handles_transient_error(self):
        from agents.agent import AgentGenerationError, _generate_response_core
        from agents.core.metrics import get_route_metrics_snapshot, reset_route_metrics

        reset_route_metrics()
        with patch("agents.agent._generate_react", side_effect=RuntimeError("429 resource exhausted")):
            with pytest.raises(AgentGenerationError) as ei:
                _generate_response_core([], "分析最近30天AI局势")
        assert "暂时不可用" in str(ei.value)
        snapshot = get_route_metrics_snapshot()
        assert snapshot["react_error"] == 1

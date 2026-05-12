"""Tests for ContextVar-backed request run context."""

from __future__ import annotations

import threading

from agent.core.run_context import (
    add_tool_call,
    agent_run_context,
    get_tool_call_chain,
    get_tool_calls,
)


def test_agent_run_context_isolated_between_threads() -> None:
    barrier = threading.Barrier(2)
    results: dict[str, set[str]] = {}

    def _worker(name: str, tool: str) -> None:
        with agent_run_context():
            add_tool_call(tool)
            barrier.wait()
            results[name] = get_tool_calls()

    t1 = threading.Thread(target=_worker, args=("t1", "query_news"), daemon=True)
    t2 = threading.Thread(target=_worker, args=("t2", "trend_analysis"), daemon=True)
    t1.start()
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)

    assert results["t1"] == {"query_news"}
    assert results["t2"] == {"trend_analysis"}


def test_tool_call_chain_preserves_order_and_duplicates() -> None:
    with agent_run_context():
        add_tool_call("search_news")
        add_tool_call("query_news")
        add_tool_call("search_news")

        assert get_tool_calls() == {"search_news", "query_news"}
        assert get_tool_call_chain() == ["search_news", "query_news", "search_news"]

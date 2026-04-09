"""Tests for ContextVar-backed request run context."""

from __future__ import annotations

import threading

from agent.core.run_context import (
    add_evidence_urls,
    add_tool_call,
    agent_run_context,
    get_evidence_urls,
    get_tool_calls,
)


def test_agent_run_context_isolated_between_threads() -> None:
    barrier = threading.Barrier(2)
    results: dict[str, tuple[set[str], set[str]]] = {}

    def _worker(name: str, tool: str, url: str) -> None:
        with agent_run_context():
            add_tool_call(tool)
            add_evidence_urls([url])
            barrier.wait()
            results[name] = (get_tool_calls(), get_evidence_urls())

    t1 = threading.Thread(target=_worker, args=("t1", "query_news", "https://a.example.com"), daemon=True)
    t2 = threading.Thread(target=_worker, args=("t2", "trend_analysis", "https://b.example.com"), daemon=True)
    t1.start()
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)

    assert results["t1"][0] == {"query_news"}
    assert results["t1"][1] == {"https://a.example.com"}
    assert results["t2"][0] == {"trend_analysis"}
    assert results["t2"][1] == {"https://b.example.com"}


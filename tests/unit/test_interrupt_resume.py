"""Verify the LangGraph interrupt/resume mechanism the clarification HIL uses.

Uses an in-memory checkpointer (no Postgres) to confirm the interrupt -> resume
API pattern (return shape of ``__interrupt__``, ``get_state().next`` pending
flag, ``Command(resume=...)``) that ``invoke_custom_graph`` depends on, plus the
pure clarification routing helper.
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

try:  # name differs across langgraph versions
    from langgraph.checkpoint.memory import InMemorySaver as MemorySaver
except ImportError:  # pragma: no cover
    from langgraph.checkpoint.memory import MemorySaver

from agent.graph.routing import route_after_clarification


class _S(TypedDict, total=False):
    question: str
    reply: str
    done: bool


def _ask(state: _S) -> dict:
    answer = interrupt({"question": "need more info?"})
    return {"reply": str(answer), "done": True}


def _build_interrupt_graph():
    graph = StateGraph(_S)
    graph.add_node("ask", _ask)
    graph.add_edge(START, "ask")
    graph.add_edge("ask", END)
    return graph.compile(checkpointer=MemorySaver())


def test_interrupt_pauses_and_surfaces_payload():
    graph = _build_interrupt_graph()
    config = {"configurable": {"thread_id": "t1"}}
    out = graph.invoke({"question": "hi"}, config=config)
    assert isinstance(out, dict) and "__interrupt__" in out
    intr = out["__interrupt__"][0]
    assert getattr(intr, "value", {}).get("question") == "need more info?"
    assert bool(graph.get_state(config).next)  # paused at the interrupted node


def test_resume_continues_from_checkpoint():
    graph = _build_interrupt_graph()
    config = {"configurable": {"thread_id": "t2"}}
    graph.invoke({"question": "hi"}, config=config)  # pauses
    out = graph.invoke(Command(resume="here is my reply"), config=config)
    assert out.get("done") is True
    assert out.get("reply") == "here is my reply"
    assert not bool(graph.get_state(config).next)  # finished


def test_route_after_clarification():
    assert route_after_clarification({"clar_route": "answer"}) == "answer"
    assert route_after_clarification({"clar_route": "end"}) == "end"
    assert route_after_clarification({}) == "end"

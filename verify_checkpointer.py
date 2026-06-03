"""One-off: verify the Postgres checkpointer + interrupt/resume on the real DB.

Run from the project root with your .env present:

    python verify_checkpointer.py

No LLM / models / web / deploy needed. Confirms:
  1) the checkpointer connects to your Postgres and creates its tables (1a);
  2) interrupt -> Command(resume) works against your actual Postgres.

Safe to delete this file afterwards (and the one 'verify-*' checkpoint row).
"""

from __future__ import annotations

import uuid
from typing import TypedDict

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

from agent.graph.checkpointer import get_checkpointer

cp = get_checkpointer()
print("checkpointer:", type(cp).__name__ if cp else None)
if cp is None:
    raise SystemExit(
        "[FAIL] checkpointer is None — check DB_* env / connectivity / AGENT_CHECKPOINTER_ENABLED"
    )
print("[OK] connected to Postgres + checkpoint tables ready (setup ran)")

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class _S(TypedDict, total=False):
    q: str
    reply: str
    done: bool


def _ask(state: _S) -> dict:
    answer = interrupt({"question": "need more info?"})
    return {"reply": str(answer), "done": True}


builder = StateGraph(_S)
builder.add_node("ask", _ask)
builder.add_edge(START, "ask")
builder.add_edge("ask", END)
graph = builder.compile(checkpointer=cp)

config = {"configurable": {"thread_id": f"verify-{uuid.uuid4().hex}"}}

out = graph.invoke({"q": "hi"}, config=config)
assert isinstance(out, dict) and "__interrupt__" in out, f"[FAIL] no interrupt surfaced: {out}"
print("[OK] interrupt fired:", out["__interrupt__"][0].value)
assert bool(graph.get_state(config).next), "[FAIL] graph not paused after interrupt"
print("[OK] paused — state persisted in Postgres")

out2 = graph.invoke(Command(resume="my reply"), config=config)
assert out2.get("done") and out2.get("reply") == "my reply", f"[FAIL] resume returned: {out2}"
print("[OK] resumed from Postgres checkpoint:", out2)

print("\n[PASS] Postgres checkpointer + interrupt/resume verified on your real DB.")

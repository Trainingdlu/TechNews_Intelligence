"""Smoke tests for v2 Router -> Miner -> Analyst workflow."""

from __future__ import annotations

from pydantic import BaseModel, Field

from agents.core.skill_contracts import SkillEnvelope
from agents.core.skill_registry import SkillRegistry
from agents.core.tool_hooks import HookDecision, ToolHookRunner
from agents.graph.workflow import _extract_trend_topic, build_workflow_graph, run_workflow


class _QueryInput(BaseModel):
    query: str = ""
    source: str = "all"
    days: int = Field(default=21, ge=1, le=365)
    category: str = ""
    sentiment: str = ""
    sort: str = "time_desc"
    limit: int = Field(default=8, ge=1, le=30)


class _TrendInput(BaseModel):
    topic: str = Field(min_length=1)
    window: int = Field(default=7, ge=3, le=60)


def _query_handler(payload: _QueryInput) -> SkillEnvelope:
    return SkillEnvelope(
        tool="query_news",
        status="ok",
        request=payload.model_dump(mode="python"),
        data={
            "count": 1,
            "records": [
                {
                    "rank": 1,
                    "source": "TechCrunch",
                    "title": "OpenAI release",
                    "title_cn": "",
                    "url": "https://example.com/query",
                    "summary": "summary",
                    "sentiment": "Positive",
                    "points": 12,
                    "created_at": "2026-03-30T10:00:00",
                }
            ],
        },
        evidence=[
            {
                "url": "https://example.com/query",
                "title": "OpenAI release",
                "source": "TechCrunch",
            }
        ],
    )


def _trend_handler(payload: _TrendInput) -> SkillEnvelope:
    return SkillEnvelope(
        tool="trend_analysis",
        status="ok",
        request=payload.model_dump(mode="python"),
        data={
            "topic": payload.topic,
            "window": payload.window,
            "recent_count": 10,
            "previous_count": 6,
            "count_delta": 4,
            "count_delta_pct": 66.7,
            "avg_points_recent": 20.0,
            "avg_points_previous": 14.0,
            "daily": [],
        },
        evidence=[
            {
                "url": "https://example.com/trend",
                "title": "Trend evidence",
                "source": "HackerNews",
            }
        ],
    )


def _registry_with_query_and_trend() -> SkillRegistry:
    registry = SkillRegistry()
    registry.register("query_news", _QueryInput, _query_handler, "query smoke")
    registry.register("trend_analysis", _TrendInput, _trend_handler, "trend smoke")
    return registry


def test_run_workflow_fact_retrieval_path() -> None:
    state = run_workflow(
        user_message="OpenAI latest updates",
        history=[],
        registry=_registry_with_query_and_trend(),
    )
    assert state["intent"] == "fact_retrieval"
    assert state["selected_skill"] == "query_news"
    assert "Retrieval summary:" in state["final_text"]
    assert state["evidence_urls"] == ["https://example.com/query"]
    assert [event["node"] for event in state["node_audit"]] == [
        "router",
        "miner",
        "analyst",
        "formatter",
    ]


def test_run_workflow_trend_path() -> None:
    state = run_workflow(
        user_message="trend of OpenAI in last 7 days",
        history=[],
        registry=_registry_with_query_and_trend(),
    )
    assert state["intent"] == "trend_analysis"
    assert state["selected_skill"] == "trend_analysis"
    assert "Trend summary:" in state["final_text"]
    assert state["evidence_urls"] == ["https://example.com/trend"]


def test_extract_trend_topic_removes_intent_and_time_words() -> None:
    topic = _extract_trend_topic("trend of OpenAI in last 7 days")
    assert topic.lower() == "openai"


def test_run_workflow_chinese_trend_path() -> None:
    state = run_workflow(
        user_message="过去7天OpenAI趋势",
        history=[],
        registry=_registry_with_query_and_trend(),
    )
    assert state["intent"] == "trend_analysis"
    assert state["selected_skill"] == "trend_analysis"
    assert state["miner_payload"]["window"] == 7
    assert state["miner_payload"]["topic"].lower() == "openai"


def test_build_workflow_graph_compiles_and_runs() -> None:
    app = build_workflow_graph(registry=_registry_with_query_and_trend())
    final_state = app.invoke({"user_message": "OpenAI latest updates", "history": []})
    assert final_state["selected_skill"] == "query_news"
    assert "Retrieval summary:" in final_state["final_text"]


def test_run_workflow_honors_post_hook_deny() -> None:
    def _deny_post(_tool: str, _payload: dict, _output: SkillEnvelope) -> HookDecision:
        return HookDecision(action="deny", reason="audit_failed", diagnostics={"rule": "post_guard"})

    hooks = ToolHookRunner(pre_hooks=[], post_hooks=[_deny_post])
    state = run_workflow(
        user_message="OpenAI latest updates",
        history=[],
        registry=_registry_with_query_and_trend(),
        hook_runner=hooks,
    )
    assert state["miner_result"].status == "error"
    assert state["miner_result"].error == "post_hook_denied"
    assert state["miner_result"].diagnostics.get("reason") == "audit_failed"
    assert state["evidence_urls"] == []
    assert state["node_audit"][1]["status"] == "deny"
    assert state["node_audit"][1]["details"]["phase"] == "post_hook"


def test_run_workflow_mcp_transport_with_fake_client() -> None:
    class _FakeMCPClient:
        @staticmethod
        def call_tool(_qualified_name: str, _payload: dict) -> SkillEnvelope:
            return SkillEnvelope(
                tool="query_news_vector",
                status="ok",
                request={"query": "OpenAI"},
                data={
                    "count": 1,
                    "records": [
                        {
                            "rank": 1,
                            "source": "TechCrunch",
                            "title": "OpenAI from MCP",
                            "title_cn": "",
                            "url": "https://example.com/mcp",
                            "summary": "summary",
                            "sentiment": "Positive",
                            "points": 20,
                            "created_at": "2026-04-01T10:00:00",
                        }
                    ],
                },
                evidence=[
                    {
                        "url": "https://example.com/mcp",
                        "title": "OpenAI from MCP",
                        "source": "TechCrunch",
                    }
                ],
            )

    state = run_workflow(
        user_message="OpenAI latest updates",
        history=[],
        registry=_registry_with_query_and_trend(),
        miner_transport="mcp",
        mcp_client=_FakeMCPClient(),
    )
    assert state["miner_transport"] == "mcp"
    assert state["selected_skill"] == "query_news"
    assert "Retrieval summary:" in state["final_text"]
    assert state["evidence_urls"] == ["https://example.com/mcp"]


def test_role_allowlist_override_denies_router() -> None:
    state = run_workflow(
        user_message="trend of OpenAI in last 7 days",
        history=[],
        registry=_registry_with_query_and_trend(),
        role_allowlists={"router": {"query_news"}},
    )
    assert state["miner_result"].status == "error"
    assert state["miner_result"].error == "role_policy_denied"
    assert state["miner_result"].diagnostics.get("role") == "router"
    assert state["node_audit"][0]["node"] == "router"
    assert state["node_audit"][0]["status"] == "deny"


def test_role_allowlist_override_denies_analyst() -> None:
    state = run_workflow(
        user_message="OpenAI latest updates",
        history=[],
        registry=_registry_with_query_and_trend(),
        role_allowlists={"analyst": {"compare_entities"}},
    )
    assert state["miner_result"].status == "error"
    assert state["miner_result"].diagnostics.get("role") == "analyst"
    assert "role_policy_denied" in state["final_text"]
    assert state["evidence_urls"] == []


def test_role_allowlist_env_override_denies_router(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_ROLE_ALLOWLIST_ROUTER", "query_news")
    state = run_workflow(
        user_message="trend of OpenAI in last 7 days",
        history=[],
        registry=_registry_with_query_and_trend(),
    )
    assert state["miner_result"].status == "error"
    assert state["miner_result"].diagnostics.get("role") == "router"

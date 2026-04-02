"""Smoke tests for v2 Router -> Miner -> Analyst workflow."""

from __future__ import annotations

from pydantic import BaseModel, Field

from agents.core.skill_contracts import SkillEnvelope
from agents.core.skill_registry import SkillRegistry
from agents.graph.workflow import run_workflow


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

"""In-process MCP server abstraction for NewsDB tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from ..core.skill_contracts import SkillEnvelope, build_error_envelope
from ..tools import (
    QueryNewsSkillInput,
    TrendAnalysisSkillInput,
    SearchNewsSkillInput,
    CompareSourcesSkillInput,
    CompareTopicsSkillInput,
    BuildTimelineSkillInput,
    AnalyzeLandscapeSkillInput,
    FulltextBatchSkillInput,
    query_news_skill,
    trend_analysis_skill,
    search_news_skill,
    compare_sources_skill,
    compare_topics_skill,
    build_timeline_skill,
    analyze_landscape_skill,
    fulltext_batch_skill,
)

MCPToolHandler = Callable[[BaseModel], SkillEnvelope]


@dataclass(frozen=True)
class MCPToolSpec:
    name: str
    input_model: type[BaseModel]
    handler: MCPToolHandler
    description: str = ""


class InProcessMCPServer:
    """Minimal MCP-compatible server abstraction for local adapter phase."""

    def __init__(self, server_name: str):
        self.server_name = str(server_name).strip()
        if not self.server_name:
            raise ValueError("server_name is required")
        self._tools: dict[str, MCPToolSpec] = {}

    def register_tool(
        self,
        name: str,
        input_model: type[BaseModel],
        handler: MCPToolHandler,
        description: str = "",
    ) -> None:
        tool_name = str(name).strip()
        if not tool_name:
            raise ValueError("tool name is required")
        if tool_name in self._tools:
            raise ValueError(f"Tool '{tool_name}' is already registered on server '{self.server_name}'")
        self._tools[tool_name] = MCPToolSpec(
            name=tool_name,
            input_model=input_model,
            handler=handler,
            description=description,
        )

    def list_tools(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for tool_name in sorted(self._tools.keys()):
            spec = self._tools[tool_name]
            rows.append(
                {
                    "server_name": self.server_name,
                    "name": spec.name,
                    "qualified_name": f"mcp__{self.server_name}__{spec.name}",
                    "description": spec.description,
                    "input_schema": spec.input_model.model_json_schema(),
                }
            )
        return rows

    def call_tool(self, tool_name: str, payload: dict[str, Any] | None = None) -> SkillEnvelope:
        request_payload = payload or {}
        spec = self._tools.get(str(tool_name).strip())
        if spec is None:
            return build_error_envelope(
                tool=str(tool_name),
                request=request_payload,
                error="mcp_unknown_tool",
                diagnostics={"server_name": self.server_name, "available_tools": sorted(self._tools.keys())},
            )

        try:
            parsed_input = spec.input_model.model_validate(request_payload)
        except ValidationError as exc:
            return build_error_envelope(
                tool=spec.name,
                request=request_payload,
                error="mcp_input_validation_failed",
                diagnostics={"validation_errors": exc.errors(), "server_name": self.server_name},
            )

        try:
            envelope = spec.handler(parsed_input)
        except Exception as exc:  # noqa: BLE001
            return build_error_envelope(
                tool=spec.name,
                request=parsed_input.model_dump(mode="python"),
                error="mcp_tool_execution_failed",
                diagnostics={
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "server_name": self.server_name,
                },
            )

        envelope.tool = spec.name
        if not envelope.request:
            envelope.request = parsed_input.model_dump(mode="python")
        envelope.diagnostics.update(
            {
                "mcp_server": self.server_name,
                "mcp_tool": spec.name,
            }
        )
        return envelope


def _query_news_vector_handler(payload: QueryNewsSkillInput) -> SkillEnvelope:
    envelope = query_news_skill(payload)
    envelope.diagnostics["delegated_skill"] = "query_news"
    return envelope


def _trend_analysis_handler(payload: TrendAnalysisSkillInput) -> SkillEnvelope:
    envelope = trend_analysis_skill(payload)
    envelope.diagnostics["delegated_skill"] = "trend_analysis"
    return envelope


def _search_news_handler(payload: SearchNewsSkillInput) -> SkillEnvelope:
    envelope = search_news_skill(payload)
    envelope.diagnostics["delegated_skill"] = "search_news"
    return envelope


def _compare_sources_handler(payload: CompareSourcesSkillInput) -> SkillEnvelope:
    envelope = compare_sources_skill(payload)
    envelope.diagnostics["delegated_skill"] = "compare_sources"
    return envelope


def _compare_topics_handler(payload: CompareTopicsSkillInput) -> SkillEnvelope:
    envelope = compare_topics_skill(payload)
    envelope.diagnostics["delegated_skill"] = "compare_topics"
    return envelope


def _build_timeline_handler(payload: BuildTimelineSkillInput) -> SkillEnvelope:
    envelope = build_timeline_skill(payload)
    envelope.diagnostics["delegated_skill"] = "build_timeline"
    return envelope


def _analyze_landscape_handler(payload: AnalyzeLandscapeSkillInput) -> SkillEnvelope:
    envelope = analyze_landscape_skill(payload)
    envelope.diagnostics["delegated_skill"] = "analyze_landscape"
    return envelope


def _fulltext_batch_handler(payload: FulltextBatchSkillInput) -> SkillEnvelope:
    envelope = fulltext_batch_skill(payload)
    envelope.diagnostics["delegated_skill"] = "fulltext_batch"
    return envelope


def build_newsdb_server(server_name: str = "newsdb") -> InProcessMCPServer:
    """Build local NewsDB MCP server with namespaced tool contracts."""

    server = InProcessMCPServer(server_name=server_name)
    server.register_tool(
        name="query_news_vector",
        input_model=QueryNewsSkillInput,
        handler=_query_news_vector_handler,
        description="Hybrid retrieval over local news DB (vector + keyword)",
    )
    server.register_tool(
        name="trend_analysis",
        input_model=TrendAnalysisSkillInput,
        handler=_trend_analysis_handler,
        description="Topic momentum analysis over local news DB",
    )
    server.register_tool(
        name="search_news",
        input_model=SearchNewsSkillInput,
        handler=_search_news_handler,
        description="Semantic + keyword hybrid news search",
    )
    server.register_tool(
        name="compare_sources",
        input_model=CompareSourcesSkillInput,
        handler=_compare_sources_handler,
        description="HackerNews vs TechCrunch source comparison",
    )
    server.register_tool(
        name="compare_topics",
        input_model=CompareTopicsSkillInput,
        handler=_compare_topics_handler,
        description="A-vs-B entity comparison with evidence",
    )
    server.register_tool(
        name="build_timeline",
        input_model=BuildTimelineSkillInput,
        handler=_build_timeline_handler,
        description="Chronological event timeline construction",
    )
    server.register_tool(
        name="analyze_landscape",
        input_model=AnalyzeLandscapeSkillInput,
        handler=_analyze_landscape_handler,
        description="Competitive landscape analysis with entity stats",
    )
    server.register_tool(
        name="fulltext_batch",
        input_model=FulltextBatchSkillInput,
        handler=_fulltext_batch_handler,
        description="Batch full-text article reading",
    )
    return server

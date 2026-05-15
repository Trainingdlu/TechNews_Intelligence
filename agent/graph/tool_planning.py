"""Tool selection and planning helpers for graph nodes."""

from __future__ import annotations

import json
from typing import Any

from agent.core.evidence import extract_urls
from agent.core.intent import extract_user_intent_text as _extract_user_intent_text
from agent.core.tool_catalog import iter_tool_definitions, tool_definition_by_name
from agent.core.tool_contracts import ToolEnvelope

from .intent_heuristics import (
    _extract_days,
    _split_compare_topics,
    _topic_from_message,
)
from .stream import emit_graph_progress


def _select_tools(intent: dict[str, Any]) -> list[str]:
    intent_type = str(intent.get("intent_type") or "").strip()
    route = str(intent.get("route") or "").strip()
    if route != "needs_tools":
        return []
    mapping = {
        "trend": ["trend_analysis", "search_news", "fulltext_batch"],
        "topic_comparison": ["compare_topics", "search_news", "query_news", "fulltext_batch"],
        "source_comparison": ["compare_sources", "search_news", "query_news", "fulltext_batch"],
        "timeline": ["build_timeline", "search_news", "fulltext_batch"],
        "landscape": ["analyze_landscape", "search_news", "fulltext_batch"],
        "article_read": ["read_news_content", "fulltext_batch", "search_news"],
        "roundup_listing": ["query_news", "search_news", "fulltext_batch"],
    }
    selected = mapping.get(intent_type, ["search_news", "query_news", "fulltext_batch"])
    known = {definition.name for definition in iter_tool_definitions()}
    return [name for name in selected if name in known]


def _tool_schema_brief(names: list[str]) -> str:
    blocks: list[str] = []
    for name in names:
        try:
            definition = tool_definition_by_name(name)
        except KeyError:
            continue
        schema = definition.input_model.model_json_schema()
        blocks.append(
            json.dumps(
                {
                    "name": definition.name,
                    "description": definition.description,
                    "input_schema": schema,
                },
                ensure_ascii=False,
            )
        )
    return "\n".join(blocks)


def _normalize_tool_calls(model_plan: Any) -> list[dict[str, Any]]:
    if not isinstance(model_plan, dict):
        return []
    raw_calls = model_plan.get("tool_calls")
    if not isinstance(raw_calls, list):
        return []
    calls: list[dict[str, Any]] = []
    for item in raw_calls:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("tool") or "").strip()
        args = item.get("args") or item.get("arguments") or {}
        if name and isinstance(args, dict):
            calls.append({"name": name, "args": dict(args)})
    return calls


def _heuristic_tool_calls(
    *,
    user_message: str,
    intent: dict[str, Any],
    selected_tools: list[str],
    tool_results: list[ToolEnvelope],
) -> list[dict[str, Any]]:
    text = _extract_user_intent_text(user_message).strip() or str(user_message or "").strip()
    days = _extract_days(text)
    urls = extract_urls(text)
    executed = {item.tool for item in tool_results}
    if len(urls) > 1 and "fulltext_batch" in selected_tools and "fulltext_batch" not in executed:
        return [{"name": "fulltext_batch", "args": {"urls": "\n".join(urls[:6]), "max_chars_per_article": 4000}}]
    if urls and "read_news_content" in selected_tools and "read_news_content" not in executed:
        return [{"name": "read_news_content", "args": {"url": urls[0]}}]
    if "fulltext_batch" in selected_tools and executed and "fulltext_batch" not in executed:
        return [{"name": "fulltext_batch", "args": {"urls": text, "max_chars_per_article": 4000}}]
    intent_type = str(intent.get("intent_type") or "")
    if intent_type == "topic_comparison" and "compare_topics" in selected_tools:
        topic_a, topic_b = _split_compare_topics(text)
        return [{"name": "compare_topics", "args": {"topic_a": topic_a, "topic_b": topic_b, "days": min(days, 90)}}]
    if intent_type == "source_comparison" and "compare_sources" in selected_tools:
        return [{"name": "compare_sources", "args": {"topic": _topic_from_message(text), "days": min(days, 90)}}]
    if intent_type == "timeline" and "build_timeline" in selected_tools:
        return [{"name": "build_timeline", "args": {"topic": _topic_from_message(text), "days": min(days, 180), "limit": 12}}]
    if intent_type == "landscape" and "analyze_landscape" in selected_tools:
        return [{"name": "analyze_landscape", "args": {"topic": _topic_from_message(text), "days": max(7, min(days, 180))}}]
    if intent_type == "trend" and "trend_analysis" in selected_tools:
        return [{"name": "trend_analysis", "args": {"topic": _topic_from_message(text), "window": max(3, min(days, 60))}}]
    if "search_news" in selected_tools:
        return [{"name": "search_news", "args": {"query": text, "days": days}}]
    if "query_news" in selected_tools:
        return [{"name": "query_news", "args": {"query": text, "days": days, "limit": 8}}]
    return []


def _emit_tool_running_status(name: str, args: dict[str, Any]) -> None:
    if name in {"search_news", "query_news"}:
        query = str(args.get("query") or "").strip()
        days = args.get("days")
        detail = f"{query} 最近 {days} 天" if query and days else query
        emit_graph_progress("retrieving", "正在检索相关新闻", detail=detail)
    elif name in {"read_news_content", "fulltext_batch"}:
        emit_graph_progress("retrieving", "正在准备读取文章")
    else:
        emit_graph_progress("analyzing", "正在整理信息")

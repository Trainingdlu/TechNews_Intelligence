"""Role-level tool allowlist policy.

In the custom graph architecture, the primary role is 'agent' which has
access to all registered tools. The multi-role definitions (router, miner,
analyst, formatter) are preserved as extension points for future multi-agent
orchestration scenarios.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Unified agent role 鈥?full tool access
# ---------------------------------------------------------------------------
_ALL_TOOLS: set[str] = {
    "query_news",
    "trend_analysis",
    "search_news",
    "compare_sources",
    "compare_topics",
    "build_timeline",
    "analyze_landscape",
    "fulltext_batch",
    "read_news_content",
    "get_db_stats",
    "list_topics",
}

ROLE_ALLOWED_TOOLS: dict[str, set[str]] = {
    # Primary role used by the custom graph agent
    "agent": set(_ALL_TOOLS),
    # Subagent roles 鈥?reserved for future multi-agent evolution
    "router": {
        "classify_intent",
        "extract_entities",
    } | _ALL_TOOLS,
    "miner": set(_ALL_TOOLS),
    "analyst": {
        "compute_momentum",
        "compare_entities",
        "synthesize_findings",
    } | _ALL_TOOLS,
    "formatter": {
        "format_answer",
    },
}


def allowed_tools_for_role(role: str) -> set[str]:
    """Return the allowlist for a role."""

    return set(ROLE_ALLOWED_TOOLS.get(role, set()))


def is_tool_allowed(role: str, tool_name: str) -> bool:
    """Check whether a role can execute the given tool."""

    if role not in ROLE_ALLOWED_TOOLS:
        return False
    return tool_name in ROLE_ALLOWED_TOOLS[role]


def assert_tool_allowed(role: str, tool_name: str) -> tuple[bool, str | None]:
    """Permission decision with denial reason for logging/hooks."""

    if role not in ROLE_ALLOWED_TOOLS:
        return False, f"unknown_role:{role}"
    if tool_name not in ROLE_ALLOWED_TOOLS[role]:
        return False, f"role:{role} cannot use tool:{tool_name}"
    return True, None


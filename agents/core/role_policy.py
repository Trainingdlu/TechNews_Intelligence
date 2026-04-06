"""Role-level skill allowlist policy.

In the unified ReAct architecture, the primary role is 'agent' which has
access to all registered skills. The multi-role definitions (router, miner,
analyst, formatter) are preserved as extension points for future multi-agent
orchestration scenarios.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Unified agent role — full skill access
# ---------------------------------------------------------------------------
_ALL_SKILLS: set[str] = {
    "query_news",
    "trend_analysis",
    "search_news",
    "compare_sources",
    "compare_topics",
    "build_timeline",
    "analyze_landscape",
    "fulltext_batch",
}

ROLE_ALLOWED_SKILLS: dict[str, set[str]] = {
    # Primary role — used by the ReAct agent
    "agent": set(_ALL_SKILLS),
    # Subagent roles — reserved for future multi-agent evolution
    "router": {
        "classify_intent",
        "extract_entities",
    } | _ALL_SKILLS,
    "miner": set(_ALL_SKILLS),
    "analyst": {
        "compute_momentum",
        "compare_entities",
        "synthesize_findings",
    } | _ALL_SKILLS,
    "formatter": {
        "format_answer",
    },
}


def allowed_skills_for_role(role: str) -> set[str]:
    """Return the allowlist for a role."""

    return set(ROLE_ALLOWED_SKILLS.get(role, set()))


def is_skill_allowed(role: str, skill_name: str) -> bool:
    """Check whether a role can execute the given skill."""

    if role not in ROLE_ALLOWED_SKILLS:
        return False
    return skill_name in ROLE_ALLOWED_SKILLS[role]


def assert_skill_allowed(role: str, skill_name: str) -> tuple[bool, str | None]:
    """Permission decision with denial reason for logging/hooks."""

    if role not in ROLE_ALLOWED_SKILLS:
        return False, f"unknown_role:{role}"
    if skill_name not in ROLE_ALLOWED_SKILLS[role]:
        return False, f"role:{role} cannot use skill:{skill_name}"
    return True, None

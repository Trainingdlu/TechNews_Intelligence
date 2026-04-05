"""Role-level skill allowlist policy for multi-agent orchestration."""

from __future__ import annotations

ROLE_ALLOWED_SKILLS: dict[str, set[str]] = {
    "router": {
        "classify_intent",
        "extract_entities",
        "query_news",
        "trend_analysis",
        "search_news",
        "compare_sources",
        "compare_topics",
        "build_timeline",
        "analyze_landscape",
        "fulltext_batch",
    },
    "miner": {
        "query_news",
        "trend_analysis",
        "search_news",
        "compare_sources",
        "compare_topics",
        "build_timeline",
        "analyze_landscape",
        "fulltext_batch",
    },
    "analyst": {
        "compute_momentum",
        "compare_entities",
        "synthesize_findings",
        "query_news",
        "trend_analysis",
        "search_news",
        "compare_sources",
        "compare_topics",
        "build_timeline",
        "analyze_landscape",
        "fulltext_batch",
    },
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

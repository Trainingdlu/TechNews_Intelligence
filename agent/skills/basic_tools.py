"""Basic lookup and state primitives exposed as skills."""

from __future__ import annotations

from ..core.skill_contracts import SkillEnvelope, build_error_envelope
from .news_ops import get_db_stats, list_topics, read_news_content
from .schemas import GetDbStatsInput, ListTopicsInput, ReadNewsContentInput


def get_db_stats_skill(input: GetDbStatsInput) -> SkillEnvelope:
    """Wrapped get_db_stats."""
    req = input.model_dump()
    try:
        text = get_db_stats()
        return SkillEnvelope(tool="get_db_stats", status="ok", request=req, data={"raw_output": text})
    except Exception as e:
        return build_error_envelope("get_db_stats", req, f"get_db_stats failed: {e}")


def list_topics_skill(input: ListTopicsInput) -> SkillEnvelope:
    """Wrapped list_topics."""
    req = input.model_dump()
    try:
        text = list_topics()
        return SkillEnvelope(tool="list_topics", status="ok", request=req, data={"raw_output": text})
    except Exception as e:
        return build_error_envelope("list_topics", req, f"list_topics failed: {e}")


def read_news_content_skill(input: ReadNewsContentInput) -> SkillEnvelope:
    """Wrapped read_news_content."""
    req = input.model_dump()
    try:
        text = read_news_content(str(input.url))
        return SkillEnvelope(tool="read_news_content", status="ok", request=req, data={"raw_output": text, "url": input.url})
    except Exception as e:
        return build_error_envelope("read_news_content", req, f"read_news_content failed: {e}")

"""Tests for topic expansion config loading."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from shutil import rmtree

from agent.skills.sql_builders import load_topic_query_expansions


def test_default_topic_expansion_contains_ai_keywords() -> None:
    expansions = load_topic_query_expansions(force_reload=True)
    assert "ai" in expansions
    lowered = {item.lower() for item in expansions["ai"]}
    assert "gpt" in lowered


def test_topic_expansion_loader_supports_env_override(monkeypatch) -> None:
    tmp_root = Path("tests/unit/.tmp_topic_config")
    tmp_root.mkdir(parents=True, exist_ok=True)
    tmp_dir = tmp_root / f"case_{uuid.uuid4().hex}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        config_path = tmp_dir / "topic_query_expansions.json"
        config_path.write_text(
            json.dumps({"ai": ["AI", "DeepSeek"], "security": ["security"]}, ensure_ascii=False),
            encoding="utf-8",
        )
        monkeypatch.setenv("AGENT_TOPIC_EXPANSIONS_PATH", str(config_path))

        expansions = load_topic_query_expansions(force_reload=True)
        assert expansions["ai"] == ["AI", "DeepSeek"]
    finally:
        monkeypatch.delenv("AGENT_TOPIC_EXPANSIONS_PATH", raising=False)
        load_topic_query_expansions(force_reload=True)
        rmtree(tmp_dir, ignore_errors=True)

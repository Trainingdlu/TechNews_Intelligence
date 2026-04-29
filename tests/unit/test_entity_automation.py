"""Tests for offline entity alias automation helpers."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agent.tools import entity_automation as mod
from eval import build_entity_alias_candidates as build_cli
from eval import promote_entity_alias_candidates as promote_cli


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def test_extract_alias_candidates_finds_models_orgs_and_chips() -> None:
    aliases = mod.extract_alias_candidates_from_text(
        title="OpenAI launches GPT-5 on NVIDIA Blackwell",
        title_cn="英伟达 发布 新芯片",
        summary="Claude Opus and Gemini Pro compete with GPT-5.",
    )
    assert "OpenAI" in aliases
    assert "英伟达" in aliases
    assert any(alias.startswith("GPT") for alias in aliases)
    assert "Blackwell" in aliases


@pytest.mark.parametrize(
    ("alias", "evidence_count", "confidence", "expected"),
    [
        ("AI", 20, 0.99, True),
        ("Apple", 20, 0.99, True),
        ("NV", 20, 0.99, True),
        ("NewModel", 1, 0.99, True),
        ("OpenAI", 4, 0.9, False),
    ],
)
def test_manual_review_required_for_ambiguous_aliases(
    alias: str,
    evidence_count: int,
    confidence: float,
    expected: bool,
) -> None:
    assert mod.requires_manual_review(alias, evidence_count=evidence_count, confidence=confidence) is expected


def test_adjudicate_alias_without_api_key_falls_back_to_pending(monkeypatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    decision = mod.adjudicate_alias_with_deepseek(alias="OpenAI", contexts=[], evidence_count=3)
    assert decision.status == "pending"
    assert decision.decision == "pending"
    assert "DEEPSEEK_API_KEY" in decision.reason


def test_adjudicate_alias_parses_deepseek_json(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-key")

    def _fake_post(_url, *, headers=None, json=None, timeout=None):
        assert headers["Authorization"] == "Bearer fake-key"
        assert json["model"] == "deepseek-v4-flash"
        assert timeout == 30.0
        return _FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"canonical_name":"OpenAI","entity_type":"company",'
                                '"decision":"merge","confidence":0.93,'
                                '"reason":"Supported by context","aliases_to_add":["OpenAI"]}'
                            )
                        }
                    }
                ]
            }
        )

    with patch.object(mod.requests, "post", _fake_post):
        decision = mod.adjudicate_alias_with_deepseek(
            alias="OpenAI",
            contexts=[{"title": "OpenAI ships a model", "url": "https://example.com"}],
            canonical_candidates=[{"canonical_name": "OpenAI"}],
            evidence_count=4,
        )

    assert decision.status == "auto_approved"
    assert decision.canonical_name == "OpenAI"
    assert decision.aliases_to_add == ["OpenAI"]


def test_apply_db_env_defaults_maps_deployment_db_vars(monkeypatch) -> None:
    for key in [
        "POSTGRES_PORT",
        "POSTGRES_DB",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "DB_HOST",
        "DB_PORT",
        "DB_NAME",
        "DB_USER",
        "DB_PASS",
    ]:
        monkeypatch.delenv(key, raising=False)

    monkeypatch.setenv("POSTGRES_PORT", "15432")
    monkeypatch.setenv("POSTGRES_DB", "warehouse")
    monkeypatch.setenv("POSTGRES_USER", "myuser")
    monkeypatch.setenv("POSTGRES_PASSWORD", "secret")

    build_cli._apply_db_env_defaults()

    assert build_cli.os.environ["DB_HOST"] == "127.0.0.1"
    assert build_cli.os.environ["DB_PORT"] == "15432"
    assert build_cli.os.environ["DB_NAME"] == "warehouse"
    assert build_cli.os.environ["DB_USER"] == "myuser"
    assert build_cli.os.environ["DB_PASS"] == "secret"


def test_normalize_aliases_preserves_chinese_and_dedupes_case_insensitive_aliases() -> None:
    aliases = promote_cli.normalize_aliases("英伟达", ["NVIDIA", " nvidia ", "英伟达"])
    assert aliases == ["英伟达", "NVIDIA"]

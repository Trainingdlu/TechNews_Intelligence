from __future__ import annotations

from agent.context_manager import (
    active_question,
    build_context_pack,
    build_history_manifest,
    normalize_context_curator_result,
    render_context_for_prompt,
)


def test_history_manifest_extracts_citation_urls_and_titles() -> None:
    history = [
        {"role": "user", "parts": [{"text": "what happened with openai"}]},
        {
            "role": "model",
            "parts": [{"text": "OpenAI released a model update.\n\nSources:\n[1] OpenAI update"}],
            "citation_urls": ["https://example.com/openai"],
            "url_title_map": {"https://example.com/openai": "OpenAI model update"},
        },
    ]

    manifest = build_history_manifest(history)

    assert manifest[0]["turn_id"] == 1
    assert manifest[0]["evidence"][0]["url"] == "https://example.com/openai"
    assert manifest[0]["evidence"][0]["title"] == "OpenAI model update"


def test_context_pack_accepts_only_real_curator_turns_and_urls() -> None:
    history = [
        {"role": "user", "parts": [{"text": "openai recent news"}]},
        {
            "role": "model",
            "parts": [{"text": "OpenAI item with https://example.com/openai"}],
            "citation_urls": ["https://example.com/openai"],
        },
    ]
    curator = {
        "depends_on_history": True,
        "standalone_question": "Explain the prior OpenAI item.",
        "selected_turn_ids": [1, 99],
        "selected_evidence_urls": ["https://example.com/openai", "https://fake.example.com"],
        "context_summary": "The user refers to the prior OpenAI item.",
        "reason": "prior turn reference",
        "confidence": 0.9,
    }

    pack = build_context_pack(
        user_message="what is that model?",
        history=history,
        history_manifest=build_history_manifest(history),
        curator_result=curator,
        curator_used=True,
    )

    assert active_question(pack, "") == "Explain the prior OpenAI item."
    assert [turn["turn_id"] for turn in pack["selected_turns"]] == [1]
    assert pack["selected_evidence_urls"] == ["https://example.com/openai"]
    assert "https://fake.example.com" not in render_context_for_prompt(pack)


def test_normalize_context_curator_result_rejects_fabricated_references() -> None:
    manifest = [
        {
            "turn_id": 1,
            "evidence": [{"url": "https://example.com/a", "title": "A", "index": 1}],
        }
    ]
    result = normalize_context_curator_result(
        {
            "depends_on_history": True,
            "standalone_question": "Explain A",
            "selected_turn_ids": [1, 99],
            "selected_evidence_urls": ["https://example.com/a", "https://fake.example.com"],
            "context_summary": "A prior turn is relevant.",
            "confidence": 3,
        },
        manifest,
    )

    assert result is not None
    assert result["selected_turn_ids"] == [1]
    assert result["selected_evidence_urls"] == ["https://example.com/a"]
    assert result["confidence"] == 1.0


def test_curator_can_select_thread_memory_evidence_index_urls() -> None:
    result = normalize_context_curator_result(
        {
            "depends_on_history": True,
            "standalone_question": "Explain the older evidence.",
            "selected_turn_ids": [],
            "selected_evidence_urls": ["https://example.com/old", "https://fake.example.com"],
            "context_summary": "Older evidence is relevant.",
            "confidence": 0.8,
        },
        [],
        {
            "evidence_index": [
                {"url": "https://example.com/old", "title": "Old item", "excerpt": "Older evidence excerpt."}
            ]
        },
    )

    assert result is not None
    assert result["selected_evidence_urls"] == ["https://example.com/old"]

    pack = build_context_pack(
        user_message="explain that older item",
        history=[],
        history_manifest=[],
        memory_summary={
            "evidence_index": [
                {"url": "https://example.com/old", "title": "Old item", "excerpt": "Older evidence excerpt."}
            ]
        },
        curator_result=result,
        curator_used=True,
    )

    rendered = render_context_for_prompt(pack)
    assert pack["trim_report"]["strategy"] == "context_curator"
    assert "Selected memory evidence" in rendered
    assert "https://example.com/old" in rendered
    assert "https://fake.example.com" not in rendered


def test_context_pack_falls_back_to_recent_turn_without_curator() -> None:
    history = [
        {"role": "user", "parts": [{"text": "first"}]},
        {"role": "model", "parts": [{"text": "first answer"}]},
        {"role": "user", "parts": [{"text": "second"}]},
        {"role": "model", "parts": [{"text": "second answer"}]},
    ]

    pack = build_context_pack(user_message="follow up", history=history)

    assert pack["trim_report"]["strategy"] == "recent_context"
    assert [turn["turn_id"] for turn in pack["selected_turns"]] == [1, 2]
    assert "second answer" in render_context_for_prompt(pack)

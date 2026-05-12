from __future__ import annotations

from types import SimpleNamespace

from services import thread_memory


def test_thread_memory_summary_payload_uses_deterministic_fallback(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_THREAD_MEMORY_LLM_ENABLED", "false")
    payload = thread_memory._build_summary_payload(
        previous_summary={},
        user_message="What happened with OpenAI?",
        answer_text="OpenAI shipped a model update.",
        evidence=[
            SimpleNamespace(
                url="https://example.com/openai",
                title="OpenAI update",
                index=1,
            )
        ],
    )

    assert payload["summary_source"] == "deterministic"
    assert "What happened with OpenAI?" in payload["summary_text"]
    assert payload["evidence_urls"] == ["https://example.com/openai"]


def test_thread_memory_llm_payload_filters_fabricated_urls(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_THREAD_MEMORY_LLM_ENABLED", "true")
    monkeypatch.setattr(
        thread_memory,
        "_generate_llm_summary_payload",
        lambda **_: {
            "summary_text": "Updated summary",
            "evidence_urls": ["https://example.com/openai", "https://fake.example.com"],
        },
    )

    payload = thread_memory._build_summary_payload(
        previous_summary={},
        user_message="What happened with OpenAI?",
        answer_text="OpenAI shipped a model update.",
        evidence=[
            SimpleNamespace(
                url="https://example.com/openai",
                title="OpenAI update",
                index=1,
            )
        ],
    )

    assert payload["summary_source"] == "llm"
    assert payload["summary_text"] == "Updated summary"
    assert payload["evidence_urls"] == ["https://example.com/openai"]

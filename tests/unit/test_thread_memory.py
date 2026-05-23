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


def test_thread_memory_summary_rolls_recent_turns(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_THREAD_MEMORY_LLM_ENABLED", "false")
    first = thread_memory._build_summary_payload(
        previous_summary={},
        user_message="What did OpenAI ship?",
        answer_text="OpenAI shipped a model update.",
        evidence=[SimpleNamespace(url="https://example.com/openai", title="OpenAI", index=1)],
    )
    second = thread_memory._build_summary_payload(
        previous_summary={"summary_payload": first},
        user_message="And what did Google ship?",
        answer_text="Google shipped a Gemini update.",
        evidence=[SimpleNamespace(url="https://example.com/google", title="Google", index=1)],
    )

    questions = [turn["question"] for turn in second["recent_turns"]]
    assert questions == ["What did OpenAI ship?", "And what did Google ship?"]
    assert "What did OpenAI ship?" in second["summary_text"]
    assert "And what did Google ship?" in second["summary_text"]


def test_thread_memory_rolling_window_caps_turns(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_THREAD_MEMORY_LLM_ENABLED", "false")
    payload: dict = {}
    total = thread_memory.MAX_ROLLING_SUMMARY_TURNS + 2
    for i in range(total):
        payload = thread_memory._build_summary_payload(
            previous_summary={"summary_payload": payload},
            user_message=f"question {i}",
            answer_text=f"answer {i}",
            evidence=[],
        )

    assert len(payload["recent_turns"]) == thread_memory.MAX_ROLLING_SUMMARY_TURNS
    assert payload["recent_turns"][0]["question"] == "question 2"
    assert payload["recent_turns"][-1]["question"] == f"question {total - 1}"

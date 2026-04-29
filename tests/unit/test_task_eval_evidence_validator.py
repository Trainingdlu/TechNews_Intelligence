from __future__ import annotations

import pytest

from eval.evidence_validator import (
    normalize_compact,
    numeric_equivalence_status,
    validate_case_evidence,
)


def _base_case(claim: dict) -> dict:
    return {
        "retrieval_evaluable": True,
        "input_news_pool": [
            {
                "doc_id": "doc_1",
                "url": "https://example.com/1",
                "title": "Apple revenue update",
                "summary": "Apple's revenue reached $1,000,000,000 in Q3 after services growth.",
                "evidence_text": "Apple's revenue reached $1,000,000,000 in Q3 after services growth.",
                "published_at": "2026-04-20T00:00:00",
                "source": "TechCrunch",
            }
        ],
        "verifiable_claims": [claim],
    }


def test_normalize_compact_keeps_cjk_latin_digits_and_drops_punctuation() -> None:
    assert normalize_compact(" 苹果 Apple_ Q3, revenue: $1.2B! ") == "苹果appleq3revenue12b"


def test_exact_quote_passes_evidence_validation() -> None:
    case = _base_case(
        {
            "claim": "Apple's Q3 revenue reached $1,000,000,000.",
            "evidence_doc_ids": ["doc_1"],
            "evidence_quotes": [
                {
                    "doc_id": "doc_1",
                    "quote": "Apple's revenue reached $1,000,000,000 in Q3",
                }
            ],
            "claim_type": "number",
        }
    )

    result = validate_case_evidence(case)

    assert result.accepted is True
    assert result.status == "exact"
    assert result.numeric_status == "equivalent"


@pytest.mark.parametrize(
    "claim",
    [
        "Apple's revenue was 1 billion dollars.",
        "Revenue was 10亿 dollars.",
    ],
)
def test_number_equivalence_handles_billion_formats(claim: str) -> None:
    assert numeric_equivalence_status(claim, "$1,000,000,000") == "equivalent"


def test_quote_with_ellipsis_fails() -> None:
    case = _base_case(
        {
            "claim": "Apple revenue grew after services growth.",
            "evidence_doc_ids": ["doc_1"],
            "evidence_quotes": [
                {
                    "doc_id": "doc_1",
                    "quote": "Apple's revenue...after services growth",
                }
            ],
            "claim_type": "fact",
        }
    )

    result = validate_case_evidence(case)

    assert result.accepted is False
    assert any(item.reason == "quote_contains_ellipsis" for item in result.feedback)


def test_non_retrieval_case_allows_empty_claims() -> None:
    result = validate_case_evidence({"retrieval_evaluable": False, "verifiable_claims": []})

    assert result.accepted is True
    assert result.status == "non_retrieval"

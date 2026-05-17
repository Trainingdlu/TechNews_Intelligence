from __future__ import annotations

from eval.build_event_eval_datasets import _primary_entity, _retrieval_questions, build_datasets
from eval.news_eval_metrics import build_url_event_index, score_retrieval_prediction
from eval.news_eval_schema import validate_event_card, validate_generation_case, validate_retrieval_case


def _event_card() -> dict:
    return {
        "event_id": "openai_personal_finance_2026_05",
        "event_title": "OpenAI 推出 ChatGPT 个人理财功能",
        "entities": ["OpenAI", "ChatGPT", "Plaid"],
        "time_window": {"start": "2026-05-15", "end": "2026-05-15"},
        "core_urls": ["https://techcrunch.com/openai-finance"],
        "related_urls": ["https://example.com/openai-finance-followup"],
        "facts": [
            {
                "claim": "该功能面向美国 Pro 用户开放。",
                "quote": "OpenAI面向美国Pro用户推出ChatGPT个人理财预览功能。",
                "url": "https://techcrunch.com/openai-finance",
            }
        ],
        "suitable_tasks": ["single_event", "latest_update", "deep_reading"],
        "sources": ["TechCrunch"],
    }


def test_event_card_requires_verifiable_fact_url_and_quote() -> None:
    card = validate_event_card(_event_card())
    assert card["event_id"] == "openai_personal_finance_2026_05"
    assert card["core_urls"] == ["https://techcrunch.com/openai-finance"]
    assert card["facts"][0]["quote"]


def test_retrieval_case_requires_gold_url() -> None:
    case = validate_retrieval_case(
        {
            "case_id": "retrieval.openai.001",
            "question": "OpenAI 最近在个人理财上有什么动作？",
            "query_type": "single_event",
            "gold_event_id": "openai_personal_finance_2026_05",
            "gold_urls": ["https://techcrunch.com/openai-finance"],
        }
    )
    assert case["gold_urls"]


def test_event_hit_accepts_related_url_without_exact_url_hit() -> None:
    card = validate_event_card(_event_card())
    event_index = build_url_event_index([card])
    score = score_retrieval_prediction(
        pred_urls=["https://example.com/openai-finance-followup"],
        gold_urls=["https://techcrunch.com/openai-finance"],
        gold_event_id="openai_personal_finance_2026_05",
        url_event_index=event_index,
        k=5,
    )
    assert score["exact_hit_at_k"] == 0.0
    assert score["event_hit_at_k"] == 1.0


def test_build_datasets_from_event_card() -> None:
    card = validate_event_card(_event_card())
    retrieval, generation, e2e = build_datasets([card], max_events=1, questions_per_event=2)
    assert len(retrieval) == 2
    assert len(generation) == 1
    assert len(e2e) == 1
    assert retrieval[0]["gold_event_id"] == card["event_id"]
    assert generation[0]["evidence"][0]["url"] == "https://techcrunch.com/openai-finance"
    assert e2e[0]["expected_behavior"] == "retrieve_then_answer"


def test_retrieval_questions_do_not_copy_full_title() -> None:
    card = validate_event_card(_event_card())
    full_title = card["event_title"]
    questions = _retrieval_questions(card)
    assert _primary_entity(card) == "OpenAI"
    assert all(full_title not in question for _, question in questions)
    assert all("「" not in question and "」" not in question for _, question in questions)
    assert all("这条和" not in question and "重点看" not in question for _, question in questions)
    assert any("个人理财" in question for _, question in questions)


def test_retrieval_questions_use_query_archetypes_not_fact_fragments() -> None:
    card = validate_event_card(
        {
            "event_id": "ghostty_github_migration_2026_04",
            "event_title": "[生态] Ghostty项目宣布因频繁宕机将全面迁出GitHub",
            "entities": ["Ghostty", "GitHub"],
            "time_window": {"start": "2026-04-28", "end": "2026-04-28"},
            "core_urls": ["https://mitchellh.com/writing/ghostty-leaving-github"],
            "related_urls": [],
            "facts": [
                {
                    "claim": "Ghostty项目宣布因GitHub平台频繁故障将启动迁移",
                    "quote": "Ghostty项目宣布因GitHub平台频繁故障将启动迁移",
                    "url": "https://mitchellh.com/writing/ghostty-leaving-github",
                }
            ],
            "suitable_tasks": ["single_event", "latest_update", "deep_reading"],
            "sources": ["Blog"],
        }
    )
    questions = [question for _, question in _retrieval_questions(card)]
    assert questions
    assert any("Ghostty 为什么要迁出 GitHub" in question for question in questions)
    assert all("因GitHub平台频繁故障将启动迁移" not in question for question in questions)


def test_retrieval_questions_skip_event_without_clear_anchor() -> None:
    card = validate_event_card(
        {
            "event_id": "generic_social_experiment_2026_05",
            "event_title": "[生态] 作者通过一个月与35名健身房陌生人交谈的实验克服社交焦虑",
            "entities": ["作者"],
            "time_window": {"start": "2026-05-01", "end": "2026-05-01"},
            "core_urls": ["https://example.com/social-experiment"],
            "related_urls": [],
            "facts": [
                {
                    "claim": "本文记录了一项为期30天的社交实验",
                    "quote": "本文记录了一项为期30天的社交实验",
                    "url": "https://example.com/social-experiment",
                }
            ],
            "suitable_tasks": ["single_event"],
            "sources": ["Blog"],
        }
    )
    assert _retrieval_questions(card) == []


def test_title_entity_prevents_fact_only_product_contamination() -> None:
    card = validate_event_card(
        {
            "event_id": "zed_1_0_release_2026_05",
            "event_title": "[开发] Zed代码编辑器正式发布1.0版本",
            "entities": ["Zed", "Claude"],
            "time_window": {"start": "2026-05-01", "end": "2026-05-01"},
            "core_urls": ["https://example.com/zed-1-0"],
            "related_urls": [],
            "facts": [
                {
                    "claim": "Zed 1.0 发布后支持 Claude 辅助开发能力",
                    "quote": "Zed 1.0 发布后支持 Claude 辅助开发能力",
                    "url": "https://example.com/zed-1-0",
                }
            ],
            "suitable_tasks": ["single_event"],
            "sources": ["Blog"],
        }
    )
    questions = [question for _, question in _retrieval_questions(card)]
    assert questions
    assert any(question.startswith("Zed 最近发布") for question in questions)
    assert all("Claude 的" not in question for question in questions)


def test_generation_case_rejects_missing_evidence() -> None:
    try:
        validate_generation_case(
            {
                "case_id": "generation.bad.001",
                "question": "总结这件事",
                "evidence": [],
                "required_claims": ["一个事实"],
                "forbidden_claims": [],
            }
        )
    except ValueError as exc:
        assert "evidence" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("validate_generation_case should reject empty evidence")

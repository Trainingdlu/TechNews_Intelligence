from __future__ import annotations

from eval.build_event_eval_datasets import _event_type, _primary_entity, _query_profile, _retrieval_questions, build_datasets
from eval.news_eval_metrics import build_event_metadata_index, build_url_event_index, score_retrieval_prediction
from eval.news_eval_schema import validate_e2e_case, validate_event_card, validate_generation_case, validate_retrieval_case


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
        "event_type": "release",
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


def test_broad_topic_case_scores_event_set_and_irrelevance() -> None:
    openai = validate_event_card(_event_card())
    anthropic = validate_event_card(
        {
            "event_id": "anthropic_claude_release_2026_05",
            "event_title": "Anthropic 发布 Claude 新版本",
            "entities": ["Anthropic", "Claude"],
            "time_window": {"start": "2026-05-16", "end": "2026-05-16"},
            "core_urls": ["https://anthropic.com/news/claude-release"],
            "related_urls": [],
            "facts": [
                {
                    "claim": "Anthropic 发布 Claude 新版本",
                    "quote": "Anthropic 发布 Claude 新版本",
                    "url": "https://anthropic.com/news/claude-release",
                }
            ],
            "event_type": "release",
            "suitable_tasks": ["single_event"],
            "sources": ["Anthropic"],
        }
    )
    case = validate_retrieval_case(
        {
            "case_id": "retrieval.topic.ai.001",
            "question": "AI 最近有什么新闻？",
            "query_type": "topic_overview",
            "case_kind": "broad_topic",
            "gold_event_ids": [openai["event_id"], anthropic["event_id"]],
            "acceptable_event_ids": [openai["event_id"], anthropic["event_id"]],
            "gold_urls": openai["core_urls"] + anthropic["core_urls"],
            "topic": "AI",
            "expected_entities": ["OpenAI", "Anthropic"],
            "expected_event_types": ["release"],
        }
    )
    url_index = build_url_event_index([openai, anthropic])
    metadata_index = build_event_metadata_index([openai, anthropic])
    score = score_retrieval_prediction(
        pred_urls=[
            "https://techcrunch.com/openai-finance",
            "https://irrelevant.example.com/news",
            "https://anthropic.com/news/claude-release",
        ],
        gold_urls=case["gold_urls"],
        gold_event_ids=case["gold_event_ids"],
        acceptable_event_ids=case["acceptable_event_ids"],
        case_kind=case["case_kind"],
        url_event_index=url_index,
        event_metadata_index=metadata_index,
        k=3,
    )
    assert score["case_kind"] == "broad_topic"
    assert score["event_set_recall_at_k"] == 1.0
    assert score["event_diversity_at_k"] == 2
    assert score["entity_diversity_at_k"] >= 2
    assert score["source_diversity_at_k"] >= 2
    assert score["irrelevant_event_ratio_at_k"] == 1 / 3


def test_e2e_schema_accepts_broad_topic_case() -> None:
    case = validate_e2e_case(
        {
            "case_id": "e2e.topic.ai.001",
            "question": "AI 最近有什么新闻？",
            "case_kind": "broad_topic",
            "gold_event_ids": ["event_a", "event_b"],
            "acceptable_event_ids": ["event_a", "event_b", "event_c"],
            "topic": "AI",
            "expected_entities": ["OpenAI", "Anthropic"],
            "expected_event_types": ["release"],
            "expected_behavior": "retrieve_summarize_topic",
        }
    )
    assert case["gold_event_ids"] == ["event_a", "event_b"]
    assert case["gold_urls"] == []


def test_build_datasets_from_event_card() -> None:
    card = validate_event_card(_event_card())
    retrieval, generation, e2e = build_datasets([card], max_events=1, questions_per_event=2)
    assert len(retrieval) == 2
    assert len(generation) == 1
    assert len(e2e) == 2
    assert retrieval[0]["gold_event_id"] == card["event_id"]
    assert generation[0]["evidence"][0]["url"] == "https://techcrunch.com/openai-finance"
    assert generation[0]["required_claim_sources"][0]["url"] == "https://techcrunch.com/openai-finance"
    assert e2e[0]["expected_behavior"] == "retrieve_then_answer"


def test_build_datasets_adds_broad_topic_cases_when_events_share_topic() -> None:
    openai = validate_event_card(_event_card())
    anthropic = validate_event_card(
        {
            "event_id": "anthropic_claude_release_2026_05",
            "event_title": "Anthropic 发布 Claude 新版本",
            "entities": ["Anthropic", "Claude"],
            "time_window": {"start": "2026-05-16", "end": "2026-05-16"},
            "core_urls": ["https://anthropic.com/news/claude-release"],
            "related_urls": [],
            "facts": [
                {
                    "claim": "Anthropic 发布 Claude 新版本",
                    "quote": "Anthropic 发布 Claude 新版本",
                    "url": "https://anthropic.com/news/claude-release",
                }
            ],
            "event_type": "release",
            "suitable_tasks": ["single_event"],
            "sources": ["Anthropic"],
        }
    )
    retrieval, _, e2e = build_datasets([openai, anthropic], max_events=2, questions_per_event=1)
    broad = [case for case in retrieval if case["case_kind"] == "broad_topic"]
    assert broad
    assert all(len(case["gold_event_ids"]) >= 2 for case in broad)
    assert any(case["case_kind"] == "broad_topic" for case in e2e)


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


def test_primary_entity_must_come_from_title_not_summary() -> None:
    card = validate_event_card(
        {
            "event_id": "local_ai_opinion_2026_05",
            "event_title": "[AI] 本地AI应成为软件开发常态",
            "entities": ["苹果"],
            "time_window": {"start": "2026-05-10", "end": "2026-05-10"},
            "core_urls": ["https://unix.foo/posts/local-ai-needs-to-be-norm/"],
            "related_urls": [],
            "facts": [
                {
                    "claim": "文章讨论本地 AI 和苹果生态限制",
                    "quote": "文章讨论本地 AI 和苹果生态限制",
                    "url": "https://unix.foo/posts/local-ai-needs-to-be-norm/",
                }
            ],
            "suitable_tasks": ["single_event"],
            "sources": ["Blog"],
        }
    )
    assert _primary_entity(card) == ""
    assert _retrieval_questions(card) == []


def test_primary_entity_preserves_full_person_name_alias() -> None:
    card = validate_event_card(
        {
            "event_id": "mitchell_hashimoto_ai_hype_2026_05",
            "event_title": "[AI] Mitchell Hashimoto指出多家公司深陷AI狂热，理性技术对话受阻",
            "entities": ["Mitchell", "Hashimoto"],
            "time_window": {"start": "2026-05-15", "end": "2026-05-15"},
            "core_urls": ["https://twitter.com/mitchellh/status/2055380239711457578"],
            "related_urls": [],
            "facts": [
                {
                    "claim": "Mitchell Hashimoto批评AI狂热影响技术讨论",
                    "quote": "Mitchell Hashimoto批评AI狂热影响技术讨论",
                    "url": "https://twitter.com/mitchellh/status/2055380239711457578",
                }
            ],
            "suitable_tasks": ["single_event"],
            "sources": ["Twitter"],
        }
    )
    assert _primary_entity(card) == "Mitchell Hashimoto"


def test_title_opinion_takes_priority_over_incident_words_in_facts() -> None:
    card = validate_event_card(
        {
            "event_id": "mitchell_ai_hype_2026_05",
            "event_title": "[AI] Mitchell Hashimoto指出多家公司深陷“AI狂热”，理性技术对话受阻",
            "entities": ["Mitchell"],
            "time_window": {"start": "2026-05-15", "end": "2026-05-15"},
            "core_urls": ["https://twitter.com/mitchellh/status/2055380239711457578"],
            "related_urls": [],
            "facts": [
                {
                    "claim": "讨论 AI 狂热导致技术讨论异常",
                    "quote": "讨论 AI 狂热导致技术讨论异常",
                    "url": "https://twitter.com/mitchellh/status/2055380239711457578",
                }
            ],
            "suitable_tasks": ["single_event"],
            "sources": ["Twitter"],
        }
    )
    assert _event_type(card) == "opinion"
    questions = [question for _, question in _retrieval_questions(card)]
    assert any("技术观点" in question or "AI 开发" in question for question in questions)
    assert all("故障" not in question and "异常" not in question for question in questions)


def test_release_title_ignores_cost_domain_in_facts() -> None:
    card = validate_event_card(
        {
            "event_id": "openai_gpt55_release_2026_04",
            "event_title": "[AI] OpenAI正式发布GPT-5.5模型",
            "entities": ["OpenAI", "GPT-5.5"],
            "time_window": {"start": "2026-04-23", "end": "2026-04-23"},
            "core_urls": ["https://openai.com/index/introducing-gpt-5-5/"],
            "related_urls": [],
            "facts": [
                {
                    "claim": "GPT-5.5 Pro API 定价更高",
                    "quote": "GPT-5.5 Pro API 定价更高",
                    "url": "https://openai.com/index/introducing-gpt-5-5/",
                }
            ],
            "suitable_tasks": ["single_event"],
            "sources": ["OpenAI"],
        }
    )
    questions = [question for _, question in _retrieval_questions(card)]
    assert any("GPT-5.5" in question and "发布" in question for question in questions)
    assert all("计费和成本" not in question for question in questions)


def test_policy_questions_use_product_subject_when_available() -> None:
    card = validate_event_card(
        {
            "event_id": "chrome_silent_ai_model_2026_05",
            "event_title": "[生态] Google Chrome未经同意静默安装4GB AI模型引发隐私与环保争议",
            "entities": ["Google", "Chrome"],
            "time_window": {"start": "2026-05-05", "end": "2026-05-05"},
            "core_urls": ["https://www.thatprivacyguy.com/blog/chrome-silent-nano-install/"],
            "related_urls": [],
            "facts": [
                {
                    "claim": "Chrome 静默安装 AI 模型",
                    "quote": "Chrome 静默安装 AI 模型",
                    "url": "https://www.thatprivacyguy.com/blog/chrome-silent-nano-install/",
                }
            ],
            "suitable_tasks": ["single_event"],
            "sources": ["Blog"],
        }
    )
    questions = [question for _, question in _retrieval_questions(card)]
    assert any(question.startswith("Chrome 最近") for question in questions)


def test_policy_questions_avoid_repeating_subject_in_domain() -> None:
    card = validate_event_card(
        {
            "event_id": "recaptcha_play_service_2026_05",
            "event_title": "[生态] 谷歌将reCAPTCHA绑定Play服务，去谷歌化安卓设备验证受阻",
            "entities": ["谷歌", "reCAPTCHA"],
            "time_window": {"start": "2026-05-08", "end": "2026-05-08"},
            "core_urls": ["https://reclaimthenet.org/google-broke-recaptcha-for-de-googled-android-users"],
            "related_urls": [],
            "facts": [
                {
                    "claim": "reCAPTCHA 绑定 Play 服务导致去谷歌化安卓设备验证受阻",
                    "quote": "reCAPTCHA 绑定 Play 服务导致去谷歌化安卓设备验证受阻",
                    "url": "https://reclaimthenet.org/google-broke-recaptcha-for-de-googled-android-users",
                }
            ],
            "suitable_tasks": ["single_event"],
            "sources": ["Blog"],
        }
    )
    questions = [question for _, question in _retrieval_questions(card)]
    assert any(question == "reCAPTCHA 最近有什么验证或限制变化？" for question in questions)
    assert all("reCAPTCHA 最近在 reCAPTCHA" not in question for question in questions)


def test_release_domain_questions_do_not_insert_extra_de() -> None:
    card = validate_event_card(
        {
            "event_id": "valve_steam_controller_cad_2026_05",
            "event_title": "[硬件] Valve发布Steam手柄CAD图纸供社区二次开发",
            "entities": ["Valve", "Steam"],
            "time_window": {"start": "2026-05-12", "end": "2026-05-12"},
            "core_urls": ["https://example.com/steam-controller-cad"],
            "related_urls": [],
            "facts": [
                {
                    "claim": "Valve发布Steam手柄CAD图纸供社区二次开发",
                    "quote": "Valve发布Steam手柄CAD图纸供社区二次开发",
                    "url": "https://example.com/steam-controller-cad",
                }
            ],
            "suitable_tasks": ["single_event"],
            "sources": ["Blog"],
        }
    )
    questions = [question for _, question in _retrieval_questions(card)]
    assert any("Steam手柄 CAD 图纸" in question for question in questions)
    assert all("Steam手柄 的CAD" not in question for question in questions)


def test_query_profile_extracts_laptop_product_without_framework_prefix() -> None:
    card = validate_event_card(
        {
            "event_id": "framework_laptop_13_pro_2026_04",
            "event_title": "[硬件] Framework发布Laptop 13 Pro笔记本，搭载酷睿Ultra 3与LPCAMM2内存",
            "entities": ["Framework"],
            "time_window": {"start": "2026-04-21", "end": "2026-04-21"},
            "core_urls": ["https://frame.work/laptop13pro"],
            "related_urls": [],
            "facts": [
                {
                    "claim": "Framework发布Laptop 13 Pro笔记本",
                    "quote": "Framework发布Laptop 13 Pro笔记本",
                    "url": "https://frame.work/laptop13pro",
                }
            ],
            "suitable_tasks": ["single_event"],
            "sources": ["Official"],
        }
    )
    profile = _query_profile(card)
    assert profile is not None
    assert profile["entity"] == "Framework"
    assert profile["product"] == "Laptop 13 Pro"


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

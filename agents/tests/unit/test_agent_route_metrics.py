"""Unit tests for route metrics and deterministic routing in agent.py."""

from __future__ import annotations

import os
import unittest
from contextlib import ExitStack
from unittest.mock import patch

try:
    from agents.tests.utils.bootstrap import ensure_agents_on_path
except ModuleNotFoundError:
    from utils.bootstrap import ensure_agents_on_path

ensure_agents_on_path()

import agent as agent_mod  # noqa: E402  pylint: disable=wrong-import-position


class AgentRouteMetricsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old_flag = os.environ.get("AGENT_ROUTE_METRICS")
        os.environ["AGENT_ROUTE_METRICS"] = "true"
        agent_mod.reset_route_metrics()
        self._lookup_titles_patcher = patch.object(agent_mod, "lookup_url_titles", lambda _urls: {})
        self._lookup_titles_patcher.start()

    def tearDown(self) -> None:
        self._lookup_titles_patcher.stop()
        if self._old_flag is None:
            os.environ.pop("AGENT_ROUTE_METRICS", None)
        else:
            os.environ["AGENT_ROUTE_METRICS"] = self._old_flag

    def test_snapshot_rates(self) -> None:
        agent_mod._metrics_inc("requests_total", 7)
        agent_mod._metrics_inc("langchain_attempts", 2)
        agent_mod._metrics_inc("langchain_success", 1)
        agent_mod._metrics_inc("langchain_fallback", 1)
        agent_mod._metrics_inc("source_compare_forced", 1)
        agent_mod._metrics_inc("compare_forced", 1)
        agent_mod._metrics_inc("timeline_forced", 1)
        agent_mod._metrics_inc("landscape_forced", 1)
        agent_mod._metrics_inc("trend_forced", 1)
        agent_mod._metrics_inc("fulltext_forced", 1)
        agent_mod._metrics_inc("query_forced", 1)
        agent_mod._metrics_inc("landscape_low_evidence", 1)

        snapshot = agent_mod.get_route_metrics_snapshot()
        self.assertEqual(snapshot["requests_total"], 7)
        self.assertEqual(snapshot["langchain_attempts"], 2)
        self.assertAlmostEqual(snapshot["fallback_rate_total"], 1.0 / 7.0, places=6)
        self.assertAlmostEqual(snapshot["fallback_rate_langchain"], 0.5, places=6)
        self.assertAlmostEqual(snapshot["langchain_success_rate"], 0.5, places=6)
        self.assertAlmostEqual(snapshot["forced_route_rate"], 1.0, places=6)
        self.assertAlmostEqual(snapshot["landscape_low_evidence_rate"], 1.0, places=6)

    def test_extract_landscape_request_cross_domain(self) -> None:
        req = agent_mod._extract_landscape_request("那么当今世界商业领域格局是什么，各家企业分别充当什么角色？")
        self.assertIsNotNone(req)
        topic, days, entities = req
        self.assertEqual(topic, "business")
        self.assertEqual(days, 30)
        self.assertEqual(entities, [])

    def test_extract_landscape_request_world_tech_uses_broad_topic(self) -> None:
        req = agent_mod._extract_landscape_request("当前世界科技格局是什么")
        self.assertIsNotNone(req)
        topic, days, entities = req
        self.assertEqual(topic, "")
        self.assertEqual(days, 30)
        self.assertEqual(entities, [])

    def test_extract_landscape_request_ai_situation_keyword(self) -> None:
        req = agent_mod._extract_landscape_request("\u5f53\u4eca\u4e16\u754cai\u9886\u57df\u5c40\u52bf\u662f\u4ec0\u4e48")
        self.assertIsNotNone(req)
        topic, days, entities = req
        self.assertEqual(topic, "AI")
        self.assertEqual(days, 30)
        self.assertEqual(entities, [])

    def test_extract_compare_request_requires_compare_intent(self) -> None:
        req = agent_mod._extract_compare_request("OpenAI and Anthropic latest updates")
        self.assertIsNone(req)

    def test_extract_compare_request_supports_implicit_comparative_question(self) -> None:
        req = agent_mod._extract_compare_request("OpenAI and Anthropic who is hotter in the last 14 days?")
        self.assertIsNotNone(req)
        topic_a, topic_b, days = req
        self.assertEqual(topic_a, "OpenAI")
        self.assertEqual(topic_b, "Anthropic")
        self.assertEqual(days, 14)

    def test_extract_timeline_request_not_triggered_by_generic_evolution(self) -> None:
        req = agent_mod._extract_timeline_request("How is AI evolving recently?")
        self.assertIsNone(req)

    def test_extract_timeline_request_triggered_by_recent_actions(self) -> None:
        req = agent_mod._extract_timeline_request("最近谷歌的动作")
        self.assertIsNotNone(req)
        topic, days, limit = req
        self.assertEqual(topic, "谷歌")
        self.assertEqual(days, 30)
        self.assertEqual(limit, 12)

    def test_extract_timeline_request_supports_week_window(self) -> None:
        req = agent_mod._extract_timeline_request("最近两周谷歌的动作")
        self.assertIsNotNone(req)
        topic, days, _limit = req
        self.assertEqual(topic, "谷歌")
        self.assertEqual(days, 14)

    def test_extract_timeline_request_recent_doing_question(self) -> None:
        req = agent_mod._extract_timeline_request("那么openai最近三十天干了什么")
        self.assertIsNotNone(req)
        topic, days, _limit = req
        self.assertEqual(topic.lower(), "openai")
        self.assertEqual(days, 30)

    def test_extract_timeline_request_recent_big_moves_question(self) -> None:
        req = agent_mod._extract_timeline_request("openai最近有什么大动作")
        self.assertIsNotNone(req)
        topic, days, _limit = req
        self.assertEqual(topic.lower(), "openai")
        self.assertEqual(days, 30)

    def test_extract_timeline_request_recent_colloquial_doing(self) -> None:
        req = agent_mod._extract_timeline_request("最近openai都在做什么")
        self.assertIsNotNone(req)
        topic, days, _limit = req
        self.assertEqual(topic.lower(), "openai")
        self.assertEqual(days, 30)

    def test_extract_timeline_request_english_what_did(self) -> None:
        req = agent_mod._extract_timeline_request("What did OpenAI do in the last 30 days")
        self.assertIsNotNone(req)
        topic, days, _limit = req
        self.assertEqual(topic, "OpenAI")
        self.assertEqual(days, 30)

    def test_extract_timeline_request_polite_prefix_not_part_of_topic(self) -> None:
        req = agent_mod._extract_timeline_request("给我看openai最近动态")
        self.assertIsNotNone(req)
        topic, days, _limit = req
        self.assertEqual(topic.lower(), "openai")
        self.assertEqual(days, 30)

    def test_extract_timeline_request_ignores_question_placeholder_topic(self) -> None:
        req = agent_mod._extract_timeline_request("谷歌最近有什么动态")
        self.assertIsNotNone(req)
        topic, days, _limit = req
        self.assertEqual(topic, "谷歌")
        self.assertEqual(days, 30)

    def test_extract_timeline_request_not_triggered_by_recent_news_only(self) -> None:
        req = agent_mod._extract_timeline_request("最近谷歌新闻")
        self.assertIsNone(req)

    def test_extract_timeline_request_ai_domain_major_product(self) -> None:
        req = agent_mod._extract_timeline_request("最近ai领域的重大产品时间线是什么")
        self.assertIsNotNone(req)
        topic, days, limit = req
        self.assertEqual(topic, "AI")
        self.assertEqual(days, 30)
        self.assertEqual(limit, 12)

    def test_extract_landscape_request_weak_ecosystem_without_role_intent(self) -> None:
        req = agent_mod._extract_landscape_request("Please discuss Apple ecosystem accessories compatibility.")
        self.assertIsNone(req)

    def test_extract_source_compare_request_detects_hn_tc(self) -> None:
        req = agent_mod._extract_source_compare_request("对比 AI 在 HackerNews 和 TechCrunch 过去14天的差异")
        self.assertIsNotNone(req)
        topic, days = req
        self.assertEqual(topic, "AI")
        self.assertEqual(days, 14)

    def test_extract_trend_request_detects_basic_intent(self) -> None:
        req = agent_mod._extract_trend_request("OpenAI 最近7天是在升温还是降温？")
        self.assertIsNotNone(req)
        topic, days = req
        self.assertEqual(topic, "OpenAI")
        self.assertEqual(days, 7)

    def test_extract_query_request_detects_filters(self) -> None:
        req = agent_mod._extract_query_request("检索 AI，来源 TechCrunch，最近7天，按热度排序")
        self.assertIsNotNone(req)
        query, source, days, sort, limit = req
        self.assertEqual(query, "AI")
        self.assertEqual(source, "TechCrunch")
        self.assertEqual(days, 7)
        self.assertEqual(sort, "heat_desc")
        self.assertEqual(limit, 8)

    def test_extract_query_request_detects_natural_language_filter_intent(self) -> None:
        req = agent_mod._extract_query_request("OpenAI 在 TechCrunch 最近14天报道")
        self.assertIsNotNone(req)
        query, source, days, sort, limit = req
        self.assertEqual(query, "OpenAI")
        self.assertEqual(source, "TechCrunch")
        self.assertEqual(days, 14)
        self.assertEqual(sort, "time_desc")
        self.assertEqual(limit, 8)

    def test_extract_query_request_supports_week_window(self) -> None:
        req = agent_mod._extract_query_request("OpenAI 在 TechCrunch 最近两周报道")
        self.assertIsNotNone(req)
        query, source, days, sort, limit = req
        self.assertEqual(query, "OpenAI")
        self.assertEqual(source, "TechCrunch")
        self.assertEqual(days, 14)
        self.assertEqual(sort, "time_desc")
        self.assertEqual(limit, 8)

    def test_extract_query_request_subject_before_recent_question_style(self) -> None:
        req = agent_mod._extract_query_request("谷歌最近有什么新闻")
        self.assertIsNotNone(req)
        query, source, days, sort, limit = req
        self.assertEqual(query, "谷歌")
        self.assertEqual(source, "all")
        self.assertEqual(days, 21)
        self.assertEqual(sort, "time_desc")
        self.assertEqual(limit, 8)

    def test_extract_query_request_recent_subject_question_style(self) -> None:
        req = agent_mod._extract_query_request("最近Google有什么动态")
        self.assertIsNotNone(req)
        query, source, days, sort, limit = req
        self.assertEqual(query, "Google")
        self.assertEqual(source, "all")
        self.assertEqual(days, 21)
        self.assertEqual(sort, "time_desc")
        self.assertEqual(limit, 8)

    def test_extract_query_request_recent_status_question(self) -> None:
        req = agent_mod._extract_query_request("openai最近怎么样")
        self.assertIsNotNone(req)
        query, source, days, sort, limit = req
        self.assertEqual(query.lower(), "openai")
        self.assertEqual(source, "all")
        self.assertEqual(days, 21)
        self.assertEqual(sort, "time_desc")
        self.assertEqual(limit, 8)

    def test_extract_query_request_summary_recent_status(self) -> None:
        req = agent_mod._extract_query_request("总结下openai最近情况")
        self.assertIsNotNone(req)
        query, source, days, sort, limit = req
        self.assertEqual(query.lower(), "openai")
        self.assertEqual(source, "all")
        self.assertEqual(days, 21)
        self.assertEqual(sort, "time_desc")
        self.assertEqual(limit, 8)

    def test_extract_source_compare_request_allows_single_source_hint(self) -> None:
        req = agent_mod._extract_source_compare_request("对比 OpenAI 在 HN 来源上的差异")
        self.assertIsNotNone(req)
        topic, days = req
        self.assertEqual(topic, "OpenAI")
        self.assertEqual(days, 14)

    def test_extract_fulltext_request_detects_keyword(self) -> None:
        req = agent_mod._extract_fulltext_request("批量读取 OpenAI Voice Engine 相关全文并总结争议点")
        self.assertIsNotNone(req)
        query, max_chars = req
        self.assertEqual(query, "OpenAI Voice Engine")
        self.assertEqual(max_chars, 4000)

    def test_generate_response_fallback_path_increments_metrics(self) -> None:
        old_runtime = os.environ.get("AGENT_RUNTIME")
        old_strict = os.environ.get("AGENT_RUNTIME_STRICT")
        os.environ["AGENT_RUNTIME"] = "langchain"
        os.environ["AGENT_RUNTIME_STRICT"] = "false"

        with ExitStack() as stack:
            stack.enter_context(patch.object(agent_mod, "_extract_compare_request", lambda _: None))
            stack.enter_context(patch.object(agent_mod, "_extract_timeline_request", lambda _: None))
            stack.enter_context(patch.object(agent_mod, "_extract_landscape_request", lambda _: None))
            stack.enter_context(
                patch.object(
                    agent_mod,
                    "_generate_langgraph",
                    lambda _h, _m: (_ for _ in ()).throw(RuntimeError("boom")),
                )
            )
            stack.enter_context(patch.object(agent_mod, "_generate_legacy", lambda _h, _m: "legacy_ok"))

            out = agent_mod.generate_response([], "hello")
            self.assertEqual(out, "legacy_ok")
            snapshot = agent_mod.get_route_metrics_snapshot()
            self.assertEqual(snapshot["requests_total"], 1)
            self.assertEqual(snapshot["langchain_attempts"], 1)
            self.assertEqual(snapshot["langchain_fallback"], 1)
            self.assertEqual(snapshot["langchain_success"], 0)

        if old_runtime is None:
            os.environ.pop("AGENT_RUNTIME", None)
        else:
            os.environ["AGENT_RUNTIME"] = old_runtime

        if old_strict is None:
            os.environ.pop("AGENT_RUNTIME_STRICT", None)
        else:
            os.environ["AGENT_RUNTIME_STRICT"] = old_strict

    def test_generate_response_timeline_forced_adds_evidence(self) -> None:
        old_runtime = os.environ.get("AGENT_RUNTIME")
        old_strict = os.environ.get("AGENT_RUNTIME_STRICT")
        old_min_events = os.environ.get("TIMELINE_MIN_EVENTS")
        os.environ["AGENT_RUNTIME"] = "langchain"
        os.environ["AGENT_RUNTIME_STRICT"] = "false"
        os.environ["TIMELINE_MIN_EVENTS"] = "1"

        raw_timeline = (
            "Timeline: google (last 30 days, max 12)\n"
            "1. 2026-03-20 10:00 | TechCrunch | Neutral | points=10\n"
            "   Event A\n"
            "   https://a.com\n"
        )

        with ExitStack() as stack:
            stack.enter_context(patch.object(agent_mod, "_extract_compare_request", lambda _: None))
            stack.enter_context(patch.object(agent_mod, "_extract_landscape_request", lambda _: None))
            stack.enter_context(patch.object(agent_mod, "_extract_timeline_request", lambda _m: ("google", 30, 12)))
            stack.enter_context(patch.object(agent_mod, "build_timeline", lambda **_kwargs: raw_timeline))
            stack.enter_context(patch.object(agent_mod, "_analyze_timeline_output", lambda **_kwargs: "## 事件时间线\n- 仅总结"))

            out = agent_mod.generate_response([], "构建 google 过去30天时间线")
            self.assertIn("来源", out)
            self.assertIn("https://a.com", out)

        if old_runtime is None:
            os.environ.pop("AGENT_RUNTIME", None)
        else:
            os.environ["AGENT_RUNTIME"] = old_runtime

        if old_strict is None:
            os.environ.pop("AGENT_RUNTIME_STRICT", None)
        else:
            os.environ["AGENT_RUNTIME_STRICT"] = old_strict

        if old_min_events is None:
            os.environ.pop("TIMELINE_MIN_EVENTS", None)
        else:
            os.environ["TIMELINE_MIN_EVENTS"] = old_min_events

    def test_analyze_timeline_output_defaults_to_grounded_deterministic(self) -> None:
        raw_timeline = (
            "Timeline: OpenAI (last 30 days, max 12)\n"
            "1. 2026-03-20 10:00 | TechCrunch | Neutral | points=10\n"
            "   Event A\n"
            "   https://a.com\n"
            "2. 2026-03-22 11:00 | HackerNews | Positive | points=20\n"
            "   Event B\n"
            "   https://b.com\n"
        )

        with patch.object(agent_mod, "_get_analysis_model", side_effect=AssertionError("should not call model")):
            out = agent_mod._analyze_timeline_output(
                user_message="那么openai最近三十天干了什么",
                topic="OpenAI",
                days=30,
                timeline_output=raw_timeline,
            )

        self.assertIn("仅基于数据库时间线记录生成", out)
        self.assertIn("事件时间线", out)
        self.assertIn("https://a.com", out)
        self.assertIn("https://b.com", out)

    def test_generate_response_landscape_retries_without_topic_when_no_data(self) -> None:
        old_runtime = os.environ.get("AGENT_RUNTIME")
        old_strict = os.environ.get("AGENT_RUNTIME_STRICT")
        os.environ["AGENT_RUNTIME"] = "langchain"
        os.environ["AGENT_RUNTIME_STRICT"] = "false"

        fake_landscape_retry = (
            "Landscape snapshot: topic=all (last 30 days)\n"
            "Coverage: topic_articles=120, matched_entity_articles=20, active_entities=3/16\n"
            "Evidence URLs:\n"
            "  [OpenAI] #1 [TechCrunch] A | points=10 | 2026-03-20 10:00 | https://a.com\n"
            "  [Google] #1 [HackerNews] B | points=8 | 2026-03-21 11:00 | https://b.com\n"
            "  [Microsoft] #1 [TechCrunch] C | points=7 | 2026-03-22 11:00 | https://c.com\n"
            "  [Amazon] #1 [HackerNews] D | points=6 | 2026-03-23 11:00 | https://d.com\n"
            "Confidence: Medium"
        )

        calls: list[str] = []

        def _fake_analyze_landscape(topic: str = "", days: int = 30, entities: str = "", limit_per_entity: int = 3) -> str:
            calls.append(topic)
            if topic:
                return "No landscape data in the last 30 days for entities: OpenAI, Google."
            return fake_landscape_retry

        with ExitStack() as stack:
            stack.enter_context(patch.object(agent_mod, "_extract_compare_request", lambda _: None))
            stack.enter_context(patch.object(agent_mod, "_extract_timeline_request", lambda _: None))
            stack.enter_context(patch.object(agent_mod, "_extract_landscape_request", lambda _m: ("当前世界科技", 30, [])))
            stack.enter_context(patch.object(agent_mod, "analyze_landscape", _fake_analyze_landscape))
            stack.enter_context(patch.object(agent_mod, "_analyze_landscape_output", lambda **_kwargs: "## 格局结论\n- 样本显示多极竞争"))

            out = agent_mod.generate_response([], "当前世界科技格局是什么")
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0], "当前世界科技")
            self.assertEqual(calls[1], "")
            self.assertIn("格局结论", out)
            self.assertIn("来源", out)

        if old_runtime is None:
            os.environ.pop("AGENT_RUNTIME", None)
        else:
            os.environ["AGENT_RUNTIME"] = old_runtime

        if old_strict is None:
            os.environ.pop("AGENT_RUNTIME_STRICT", None)
        else:
            os.environ["AGENT_RUNTIME_STRICT"] = old_strict

    def test_generate_response_landscape_forced_path(self) -> None:
        old_runtime = os.environ.get("AGENT_RUNTIME")
        old_strict = os.environ.get("AGENT_RUNTIME_STRICT")
        os.environ["AGENT_RUNTIME"] = "langchain"
        os.environ["AGENT_RUNTIME_STRICT"] = "false"

        fake_landscape = (
            "Landscape snapshot: topic=security (last 30 days)\n"
            "Coverage: topic_articles=50, matched_entity_articles=20, active_entities=2/2\n"
            "Evidence URLs:\n"
            "  [CrowdStrike] #1 [TechCrunch] A | points=10 | 2026-03-20 10:00 | https://a.com\n"
            "  [Cloudflare] #1 [HackerNews] B | points=8 | 2026-03-21 11:00 | https://b.com\n"
            "  [CrowdStrike] #2 [TechCrunch] C | points=7 | 2026-03-22 11:00 | https://c.com\n"
            "  [Cloudflare] #2 [HackerNews] D | points=6 | 2026-03-23 11:00 | https://d.com\n"
            "Confidence: Medium"
        )

        with ExitStack() as stack:
            stack.enter_context(patch.object(agent_mod, "_extract_compare_request", lambda _: None))
            stack.enter_context(patch.object(agent_mod, "_extract_timeline_request", lambda _: None))
            stack.enter_context(
                patch.object(
                    agent_mod,
                    "_extract_landscape_request",
                    lambda _m: ("security", 30, ["CrowdStrike", "Cloudflare"]),
                )
            )
            stack.enter_context(patch.object(agent_mod, "analyze_landscape", lambda **_kwargs: fake_landscape))
            stack.enter_context(
                patch.object(
                    agent_mod,
                    "_analyze_landscape_output",
                    lambda **_kwargs: "## 格局结论\n- 安全赛道讨论分化",
                )
            )

            out = agent_mod.generate_response([], "当今世界安全领域格局是什么？")
            self.assertIn("格局结论", out)
            self.assertIn("来源", out)
            snapshot = agent_mod.get_route_metrics_snapshot()
            self.assertEqual(snapshot["requests_total"], 1)
            self.assertEqual(snapshot["landscape_forced"], 1)
            self.assertEqual(snapshot["landscape_low_evidence"], 0)
            self.assertAlmostEqual(snapshot["landscape_low_evidence_rate"], 0.0, places=6)
            self.assertEqual(snapshot["langchain_attempts"], 0)

        if old_runtime is None:
            os.environ.pop("AGENT_RUNTIME", None)
        else:
            os.environ["AGENT_RUNTIME"] = old_runtime

        if old_strict is None:
            os.environ.pop("AGENT_RUNTIME_STRICT", None)
        else:
            os.environ["AGENT_RUNTIME_STRICT"] = old_strict

    def test_generate_response_landscape_low_evidence_degrades(self) -> None:
        old_runtime = os.environ.get("AGENT_RUNTIME")
        old_strict = os.environ.get("AGENT_RUNTIME_STRICT")
        old_min_urls = os.environ.get("LANDSCAPE_MIN_URLS")
        old_min_matched = os.environ.get("LANDSCAPE_MIN_MATCHED_ARTICLES")
        old_min_active = os.environ.get("LANDSCAPE_MIN_ACTIVE_ENTITIES")

        os.environ["AGENT_RUNTIME"] = "langchain"
        os.environ["AGENT_RUNTIME_STRICT"] = "false"
        os.environ["LANDSCAPE_MIN_URLS"] = "4"
        os.environ["LANDSCAPE_MIN_MATCHED_ARTICLES"] = "6"
        os.environ["LANDSCAPE_MIN_ACTIVE_ENTITIES"] = "2"

        fake_low_evidence = (
            "Landscape snapshot: topic=business (last 30 days)\n"
            "Coverage: topic_articles=40, matched_entity_articles=2, active_entities=1/1\n"
            "Evidence URLs:\n"
            "  [Microsoft] #1 [TechCrunch] A | points=10 | 2026-03-20 10:00 | https://a.com\n"
            "Confidence: Low"
        )

        with ExitStack() as stack:
            stack.enter_context(patch.object(agent_mod, "_extract_compare_request", lambda _: None))
            stack.enter_context(patch.object(agent_mod, "_extract_timeline_request", lambda _: None))
            stack.enter_context(
                patch.object(agent_mod, "_extract_landscape_request", lambda _m: ("business", 30, ["Microsoft"]))
            )
            stack.enter_context(patch.object(agent_mod, "analyze_landscape", lambda **_kwargs: fake_low_evidence))
            stack.enter_context(
                patch.object(
                    agent_mod,
                    "_analyze_landscape_output",
                    lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("should_not_run")),
                )
            )

            out = agent_mod.generate_response([], "business landscape and roles?")
            self.assertIn("evidence is insufficient", out.lower())
            self.assertIn("confidence: low", out.lower())
            snapshot = agent_mod.get_route_metrics_snapshot()
            self.assertEqual(snapshot["requests_total"], 1)
            self.assertEqual(snapshot["landscape_forced"], 1)
            self.assertEqual(snapshot["landscape_low_evidence"], 1)
            self.assertAlmostEqual(snapshot["landscape_low_evidence_rate"], 1.0, places=6)
            self.assertEqual(snapshot["langchain_attempts"], 0)

        if old_runtime is None:
            os.environ.pop("AGENT_RUNTIME", None)
        else:
            os.environ["AGENT_RUNTIME"] = old_runtime

        if old_strict is None:
            os.environ.pop("AGENT_RUNTIME_STRICT", None)
        else:
            os.environ["AGENT_RUNTIME_STRICT"] = old_strict

        if old_min_urls is None:
            os.environ.pop("LANDSCAPE_MIN_URLS", None)
        else:
            os.environ["LANDSCAPE_MIN_URLS"] = old_min_urls

        if old_min_matched is None:
            os.environ.pop("LANDSCAPE_MIN_MATCHED_ARTICLES", None)
        else:
            os.environ["LANDSCAPE_MIN_MATCHED_ARTICLES"] = old_min_matched

        if old_min_active is None:
            os.environ.pop("LANDSCAPE_MIN_ACTIVE_ENTITIES", None)
        else:
            os.environ["LANDSCAPE_MIN_ACTIVE_ENTITIES"] = old_min_active

    def test_generate_response_landscape_unstable_synthesis_fallbacks_to_raw(self) -> None:
        old_runtime = os.environ.get("AGENT_RUNTIME")
        old_strict = os.environ.get("AGENT_RUNTIME_STRICT")
        os.environ["AGENT_RUNTIME"] = "langchain"
        os.environ["AGENT_RUNTIME_STRICT"] = "false"

        fake_landscape = (
            "Landscape snapshot: topic=AI (last 30 days)\n"
            "Coverage: topic_articles=40, matched_entity_articles=10, active_entities=2/2\n"
            "Evidence URLs:\n"
            "  [OpenAI] #1 [TechCrunch] A | points=10 | 2026-03-20 10:00 | https://a.com\n"
            "  [Google] #1 [HackerNews] B | points=8 | 2026-03-21 11:00 | https://b.com\n"
            "  [OpenAI] #2 [TechCrunch] C | points=7 | 2026-03-22 11:00 | https://c.com\n"
            "  [Google] #2 [HackerNews] D | points=6 | 2026-03-23 11:00 | https://d.com\n"
            "Confidence: Medium"
        )

        unstable_summary = (
            "\u5f88\u62b1\u6b49\uff0cAI\u683c\u5c40\u5206\u6790\u5de5\u5177\u5f53\u524d\u9047\u5230\u6280\u672f\u95ee\u9898\uff0c"
            "\u65e0\u6cd5\u751f\u6210\u5b8c\u6574\u62a5\u544a\u3002"
        )

        with ExitStack() as stack:
            stack.enter_context(patch.object(agent_mod, "_extract_compare_request", lambda _: None))
            stack.enter_context(patch.object(agent_mod, "_extract_timeline_request", lambda _: None))
            stack.enter_context(patch.object(agent_mod, "_extract_landscape_request", lambda _m: ("AI", 30, [])))
            stack.enter_context(patch.object(agent_mod, "analyze_landscape", lambda **_kwargs: fake_landscape))
            stack.enter_context(patch.object(agent_mod, "_analyze_landscape_output", lambda **_kwargs: unstable_summary))

            out = agent_mod.generate_response([], "当今世界AI领域局势是什么")
            self.assertIn("Landscape snapshot", out)
            self.assertIn("来源", out)
            self.assertNotIn("很抱歉", out)

        if old_runtime is None:
            os.environ.pop("AGENT_RUNTIME", None)
        else:
            os.environ["AGENT_RUNTIME"] = old_runtime

        if old_strict is None:
            os.environ.pop("AGENT_RUNTIME_STRICT", None)
        else:
            os.environ["AGENT_RUNTIME_STRICT"] = old_strict

    def test_generate_response_landscape_no_data_returns_guidance(self) -> None:
        old_runtime = os.environ.get("AGENT_RUNTIME")
        old_strict = os.environ.get("AGENT_RUNTIME_STRICT")
        os.environ["AGENT_RUNTIME"] = "langchain"
        os.environ["AGENT_RUNTIME_STRICT"] = "false"

        with ExitStack() as stack:
            stack.enter_context(patch.object(agent_mod, "_extract_compare_request", lambda _: None))
            stack.enter_context(patch.object(agent_mod, "_extract_timeline_request", lambda _: None))
            stack.enter_context(patch.object(agent_mod, "_extract_landscape_request", lambda _m: ("AI", 30, [])))
            stack.enter_context(
                patch.object(
                    agent_mod,
                    "analyze_landscape",
                    lambda **_kwargs: "No landscape data in the last 30 days for entities: OpenAI, Google.",
                )
            )

            out = agent_mod.generate_response([], "当今世界AI领域局势是什么")
            self.assertIn("最近 30 天", out)
            self.assertIn("置信度：低", out)
            self.assertIn("No landscape data", out)

        if old_runtime is None:
            os.environ.pop("AGENT_RUNTIME", None)
        else:
            os.environ["AGENT_RUNTIME"] = old_runtime

        if old_strict is None:
            os.environ.pop("AGENT_RUNTIME_STRICT", None)
        else:
            os.environ["AGENT_RUNTIME_STRICT"] = old_strict

    def test_generate_response_source_compare_forced_adds_evidence(self) -> None:
        old_runtime = os.environ.get("AGENT_RUNTIME")
        old_strict = os.environ.get("AGENT_RUNTIME_STRICT")
        os.environ["AGENT_RUNTIME"] = "langchain"
        os.environ["AGENT_RUNTIME_STRICT"] = "false"

        raw_source = (
            "Source comparison: AI (last 14 days)\n"
            "Stats:\n"
            "  HackerNews: count=5, avg_points=33.1, sentiment(P/N/Ng)=2/2/1\n"
            "  TechCrunch: count=6, avg_points=28.8, sentiment(P/N/Ng)=3/2/1\n"
            "Top evidence:\n"
            "  [HackerNews] #1 A | points=10 | 2026-03-20 10:00 | https://a.com\n"
            "  [TechCrunch] #1 B | points=9 | 2026-03-20 11:00 | https://b.com\n"
        )

        with ExitStack() as stack:
            stack.enter_context(patch.object(agent_mod, "_extract_source_compare_request", lambda _m: ("AI", 14)))
            stack.enter_context(patch.object(agent_mod, "compare_sources", lambda **_kwargs: raw_source))
            stack.enter_context(
                patch.object(agent_mod, "_analyze_source_compare_output", lambda **_kwargs: "## 对比结论\n- 样本显示来源分化")
            )
            out = agent_mod.generate_response([], "对比 AI 在 HackerNews 和 TechCrunch 过去14天的差异")
            self.assertIn("来源", out)
            self.assertIn("https://a.com", out)
            self.assertIn("https://b.com", out)
            snapshot = agent_mod.get_route_metrics_snapshot()
            self.assertEqual(snapshot["source_compare_forced"], 1)

        if old_runtime is None:
            os.environ.pop("AGENT_RUNTIME", None)
        else:
            os.environ["AGENT_RUNTIME"] = old_runtime

        if old_strict is None:
            os.environ.pop("AGENT_RUNTIME_STRICT", None)
        else:
            os.environ["AGENT_RUNTIME_STRICT"] = old_strict

    def test_generate_response_query_forced_adds_evidence(self) -> None:
        old_runtime = os.environ.get("AGENT_RUNTIME")
        old_strict = os.environ.get("AGENT_RUNTIME_STRICT")
        os.environ["AGENT_RUNTIME"] = "langchain"
        os.environ["AGENT_RUNTIME_STRICT"] = "false"

        raw_query = (
            "1. [TechCrunch] A\n"
            "   time=2026-03-20 10:00, sentiment=Neutral, points=10\n"
            "   url=https://a.com\n"
            "2. [TechCrunch] B\n"
            "   time=2026-03-20 11:00, sentiment=Positive, points=9\n"
            "   url=https://b.com\n"
        )

        with ExitStack() as stack:
            stack.enter_context(patch.object(agent_mod, "_extract_source_compare_request", lambda _m: None))
            stack.enter_context(patch.object(agent_mod, "_extract_compare_request", lambda _m: None))
            stack.enter_context(patch.object(agent_mod, "_extract_timeline_request", lambda _m: None))
            stack.enter_context(patch.object(agent_mod, "_extract_landscape_request", lambda _m: None))
            stack.enter_context(patch.object(agent_mod, "_extract_trend_request", lambda _m: None))
            stack.enter_context(patch.object(agent_mod, "_extract_fulltext_request", lambda _m: None))
            stack.enter_context(patch.object(agent_mod, "_extract_query_request", lambda _m: ("AI", "TechCrunch", 7, "heat_desc", 8)))
            stack.enter_context(patch.object(agent_mod, "query_news", lambda **_kwargs: raw_query))
            stack.enter_context(patch.object(agent_mod, "_analyze_query_output", lambda **_kwargs: "## 检索结果摘要\n- 已返回高热样本"))
            out = agent_mod.generate_response([], "检索 AI，来源 TechCrunch，最近7天，按热度排序")
            self.assertIn("来源", out)
            self.assertIn("https://a.com", out)
            self.assertIn("https://b.com", out)
            snapshot = agent_mod.get_route_metrics_snapshot()
            self.assertEqual(snapshot["query_forced"], 1)

        if old_runtime is None:
            os.environ.pop("AGENT_RUNTIME", None)
        else:
            os.environ["AGENT_RUNTIME"] = old_runtime

        if old_strict is None:
            os.environ.pop("AGENT_RUNTIME_STRICT", None)
        else:
            os.environ["AGENT_RUNTIME_STRICT"] = old_strict

    def test_generate_response_fulltext_forced_adds_evidence(self) -> None:
        old_runtime = os.environ.get("AGENT_RUNTIME")
        old_strict = os.environ.get("AGENT_RUNTIME_STRICT")
        os.environ["AGENT_RUNTIME"] = "langchain"
        os.environ["AGENT_RUNTIME_STRICT"] = "false"

        raw_fulltext = (
            "No URLs provided. Auto-selected Top 2 articles for query 'OpenAI Voice Engine':\n"
            "1. [TechCrunch] A | points=10 | 2026-03-20 10:00 | https://a.com\n"
            "2. [HackerNews] B | points=9 | 2026-03-20 11:00 | https://b.com\n"
        )

        with ExitStack() as stack:
            stack.enter_context(patch.object(agent_mod, "_extract_source_compare_request", lambda _m: None))
            stack.enter_context(patch.object(agent_mod, "_extract_compare_request", lambda _m: None))
            stack.enter_context(patch.object(agent_mod, "_extract_timeline_request", lambda _m: None))
            stack.enter_context(patch.object(agent_mod, "_extract_landscape_request", lambda _m: None))
            stack.enter_context(patch.object(agent_mod, "_extract_trend_request", lambda _m: None))
            stack.enter_context(patch.object(agent_mod, "_extract_fulltext_request", lambda _m: ("OpenAI Voice Engine", 4000)))
            stack.enter_context(patch.object(agent_mod, "fulltext_batch", lambda **_kwargs: raw_fulltext))
            stack.enter_context(patch.object(agent_mod, "_analyze_fulltext_output", lambda **_kwargs: "## 核心结论\n- 争议点集中在应用边界"))
            out = agent_mod.generate_response([], "批量读取 OpenAI Voice Engine 相关全文并总结争议点")
            self.assertIn("来源", out)
            self.assertIn("https://a.com", out)
            self.assertIn("https://b.com", out)
            snapshot = agent_mod.get_route_metrics_snapshot()
            self.assertEqual(snapshot["fulltext_forced"], 1)

        if old_runtime is None:
            os.environ.pop("AGENT_RUNTIME", None)
        else:
            os.environ["AGENT_RUNTIME"] = old_runtime

        if old_strict is None:
            os.environ.pop("AGENT_RUNTIME_STRICT", None)
        else:
            os.environ["AGENT_RUNTIME_STRICT"] = old_strict

    def test_generate_response_trend_forced_uses_trend_tool(self) -> None:
        old_runtime = os.environ.get("AGENT_RUNTIME")
        old_strict = os.environ.get("AGENT_RUNTIME_STRICT")
        os.environ["AGENT_RUNTIME"] = "langchain"
        os.environ["AGENT_RUNTIME_STRICT"] = "false"

        raw_trend = (
            "Trend for topic: OpenAI\n"
            "Window: recent 7 days vs previous 7 days\n"
            "Article count: 12 vs 8 -> +4 (+50.0%)\n"
            "Avg points: 33.0 vs 29.0\n"
            "Daily breakdown:\n"
            "  2026-03-20: count=2, avg_points=30.0\n"
        )

        with ExitStack() as stack:
            stack.enter_context(patch.object(agent_mod, "_extract_source_compare_request", lambda _m: None))
            stack.enter_context(patch.object(agent_mod, "_extract_compare_request", lambda _m: None))
            stack.enter_context(patch.object(agent_mod, "_extract_timeline_request", lambda _m: None))
            stack.enter_context(patch.object(agent_mod, "_extract_landscape_request", lambda _m: None))
            stack.enter_context(patch.object(agent_mod, "_extract_trend_request", lambda _m: ("OpenAI", 7)))
            stack.enter_context(patch.object(agent_mod, "trend_analysis", lambda **_kwargs: raw_trend))
            stack.enter_context(patch.object(agent_mod, "_analyze_trend_output", lambda **_kwargs: "## 趋势结论\n- 热度抬升"))
            out = agent_mod.generate_response([], "OpenAI 最近7天是在升温还是降温？")
            self.assertIn("趋势结论", out)
            self.assertIn("热度抬升", out)
            snapshot = agent_mod.get_route_metrics_snapshot()
            self.assertEqual(snapshot["trend_forced"], 1)

        if old_runtime is None:
            os.environ.pop("AGENT_RUNTIME", None)
        else:
            os.environ["AGENT_RUNTIME"] = old_runtime

        if old_strict is None:
            os.environ.pop("AGENT_RUNTIME_STRICT", None)
        else:
            os.environ["AGENT_RUNTIME_STRICT"] = old_strict

    def test_generate_response_payload_has_text_and_title_map(self) -> None:
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(agent_mod, "_generate_response_core", lambda _h, _m: "总结见 https://a.com/article")
            )
            stack.enter_context(
                patch.object(agent_mod, "lookup_url_titles", lambda _urls: {"https://a.com/article": "中文标题A"})
            )
            payload = agent_mod.generate_response_payload([], "给我总结")

        self.assertIsInstance(payload, dict)
        self.assertIn("text", payload)
        self.assertIn("url_title_map", payload)
        self.assertIn("[1]", str(payload["text"]))
        self.assertIn("来源", str(payload["text"]))
        self.assertEqual(payload["url_title_map"].get("https://a.com/article"), "中文标题A")

    def test_generate_response_planner_routes_compare_when_confident(self) -> None:
        old_runtime = os.environ.get("AGENT_RUNTIME")
        old_strict = os.environ.get("AGENT_RUNTIME_STRICT")
        old_planner_enabled = os.environ.get("AGENT_PLANNER_ENABLED")
        old_planner_conf = os.environ.get("AGENT_PLANNER_MIN_CONFIDENCE")
        old_deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
        os.environ["AGENT_RUNTIME"] = "langchain"
        os.environ["AGENT_RUNTIME_STRICT"] = "false"
        os.environ["AGENT_PLANNER_ENABLED"] = "true"
        os.environ["AGENT_PLANNER_MIN_CONFIDENCE"] = "0.75"
        os.environ["DEEPSEEK_API_KEY"] = "test-key"

        with ExitStack() as stack:
            stack.enter_context(patch.object(agent_mod, "_extract_source_compare_request", lambda _m: None))
            stack.enter_context(patch.object(agent_mod, "_extract_compare_request", lambda _m: None))
            stack.enter_context(patch.object(agent_mod, "_extract_timeline_request", lambda _m: None))
            stack.enter_context(patch.object(agent_mod, "_extract_landscape_request", lambda _m: None))
            stack.enter_context(patch.object(agent_mod, "_extract_trend_request", lambda _m: None))
            stack.enter_context(patch.object(agent_mod, "_extract_fulltext_request", lambda _m: None))
            stack.enter_context(patch.object(agent_mod, "_extract_query_request", lambda _m: None))
            stack.enter_context(
                patch.object(
                    agent_mod,
                    "_plan_route_with_deepseek",
                    lambda _h, _m: {
                        "intent": "compare",
                        "confidence": 0.93,
                        "params": {"topic_a": "OpenAI", "topic_b": "Anthropic", "days": 14},
                    },
                )
            )
            stack.enter_context(
                patch.object(
                    agent_mod,
                    "run_compare_pipeline",
                    lambda **_kwargs: "planner compare ok https://a.com/evidence",
                )
            )

            out = agent_mod.generate_response([], "请比较 OpenAI 和 Anthropic")
            self.assertIn("planner compare ok", out)
            self.assertIn("来源", out)
            snapshot = agent_mod.get_route_metrics_snapshot()
            self.assertEqual(snapshot.get("planner_routed"), 1)
            self.assertEqual(snapshot.get("compare_forced"), 1)
            self.assertEqual(snapshot.get("langchain_attempts"), 0)

        if old_runtime is None:
            os.environ.pop("AGENT_RUNTIME", None)
        else:
            os.environ["AGENT_RUNTIME"] = old_runtime

        if old_strict is None:
            os.environ.pop("AGENT_RUNTIME_STRICT", None)
        else:
            os.environ["AGENT_RUNTIME_STRICT"] = old_strict

        if old_planner_enabled is None:
            os.environ.pop("AGENT_PLANNER_ENABLED", None)
        else:
            os.environ["AGENT_PLANNER_ENABLED"] = old_planner_enabled

        if old_planner_conf is None:
            os.environ.pop("AGENT_PLANNER_MIN_CONFIDENCE", None)
        else:
            os.environ["AGENT_PLANNER_MIN_CONFIDENCE"] = old_planner_conf

        if old_deepseek_key is None:
            os.environ.pop("DEEPSEEK_API_KEY", None)
        else:
            os.environ["DEEPSEEK_API_KEY"] = old_deepseek_key

    def test_generate_response_planner_low_confidence_falls_back_runtime(self) -> None:
        old_runtime = os.environ.get("AGENT_RUNTIME")
        old_strict = os.environ.get("AGENT_RUNTIME_STRICT")
        old_planner_enabled = os.environ.get("AGENT_PLANNER_ENABLED")
        old_planner_conf = os.environ.get("AGENT_PLANNER_MIN_CONFIDENCE")
        old_deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
        os.environ["AGENT_RUNTIME"] = "langchain"
        os.environ["AGENT_RUNTIME_STRICT"] = "false"
        os.environ["AGENT_PLANNER_ENABLED"] = "true"
        os.environ["AGENT_PLANNER_MIN_CONFIDENCE"] = "0.75"
        os.environ["DEEPSEEK_API_KEY"] = "test-key"

        with ExitStack() as stack:
            stack.enter_context(patch.object(agent_mod, "_extract_source_compare_request", lambda _m: None))
            stack.enter_context(patch.object(agent_mod, "_extract_compare_request", lambda _m: None))
            stack.enter_context(patch.object(agent_mod, "_extract_timeline_request", lambda _m: None))
            stack.enter_context(patch.object(agent_mod, "_extract_landscape_request", lambda _m: None))
            stack.enter_context(patch.object(agent_mod, "_extract_trend_request", lambda _m: None))
            stack.enter_context(patch.object(agent_mod, "_extract_fulltext_request", lambda _m: None))
            stack.enter_context(patch.object(agent_mod, "_extract_query_request", lambda _m: None))
            stack.enter_context(
                patch.object(
                    agent_mod,
                    "_plan_route_with_deepseek",
                    lambda _h, _m: {
                        "intent": "compare",
                        "confidence": 0.22,
                        "params": {"topic_a": "OpenAI", "topic_b": "Anthropic", "days": 14},
                    },
                )
            )
            stack.enter_context(
                patch.object(
                    agent_mod,
                    "run_runtime_pipeline",
                    lambda **_kwargs: "runtime fallback ok",
                )
            )
            out = agent_mod.generate_response([], "请比较 OpenAI 和 Anthropic")
            self.assertEqual(out, "runtime fallback ok")
            snapshot = agent_mod.get_route_metrics_snapshot()
            self.assertEqual(snapshot.get("planner_low_confidence"), 1)
            self.assertEqual(snapshot.get("planner_routed", 0), 0)

        if old_runtime is None:
            os.environ.pop("AGENT_RUNTIME", None)
        else:
            os.environ["AGENT_RUNTIME"] = old_runtime

        if old_strict is None:
            os.environ.pop("AGENT_RUNTIME_STRICT", None)
        else:
            os.environ["AGENT_RUNTIME_STRICT"] = old_strict

        if old_planner_enabled is None:
            os.environ.pop("AGENT_PLANNER_ENABLED", None)
        else:
            os.environ["AGENT_PLANNER_ENABLED"] = old_planner_enabled

        if old_planner_conf is None:
            os.environ.pop("AGENT_PLANNER_MIN_CONFIDENCE", None)
        else:
            os.environ["AGENT_PLANNER_MIN_CONFIDENCE"] = old_planner_conf

        if old_deepseek_key is None:
            os.environ.pop("DEEPSEEK_API_KEY", None)
        else:
            os.environ["DEEPSEEK_API_KEY"] = old_deepseek_key

    def test_generate_response_graph_dispatch_uses_forced_route(self) -> None:
        old_runtime = os.environ.get("AGENT_RUNTIME")
        old_strict = os.environ.get("AGENT_RUNTIME_STRICT")
        old_dispatch = os.environ.get("AGENT_DISPATCH_MODE")
        os.environ["AGENT_RUNTIME"] = "langchain"
        os.environ["AGENT_RUNTIME_STRICT"] = "false"
        os.environ["AGENT_DISPATCH_MODE"] = "graph"

        with ExitStack() as stack:
            stack.enter_context(patch.object(agent_mod, "_extract_source_compare_request", lambda _m: None))
            stack.enter_context(patch.object(agent_mod, "_extract_compare_request", lambda _m: ("OpenAI", "Anthropic", 14)))
            stack.enter_context(
                patch.object(
                    agent_mod,
                    "run_compare_pipeline",
                    lambda **_kwargs: "graph forced compare https://a.com/evidence",
                )
            )
            out = agent_mod.generate_response([], "对比 OpenAI 和 Anthropic")
            self.assertIn("graph forced compare", out)
            self.assertIn("来源", out)
            snapshot = agent_mod.get_route_metrics_snapshot()
            self.assertEqual(snapshot.get("compare_forced"), 1)
            self.assertEqual(snapshot.get("langchain_attempts"), 0)

        if old_runtime is None:
            os.environ.pop("AGENT_RUNTIME", None)
        else:
            os.environ["AGENT_RUNTIME"] = old_runtime

        if old_strict is None:
            os.environ.pop("AGENT_RUNTIME_STRICT", None)
        else:
            os.environ["AGENT_RUNTIME_STRICT"] = old_strict

        if old_dispatch is None:
            os.environ.pop("AGENT_DISPATCH_MODE", None)
        else:
            os.environ["AGENT_DISPATCH_MODE"] = old_dispatch

    def test_generate_response_graph_dispatch_runtime_fallback(self) -> None:
        old_runtime = os.environ.get("AGENT_RUNTIME")
        old_strict = os.environ.get("AGENT_RUNTIME_STRICT")
        old_dispatch = os.environ.get("AGENT_DISPATCH_MODE")
        old_planner_enabled = os.environ.get("AGENT_PLANNER_ENABLED")
        old_deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
        os.environ["AGENT_RUNTIME"] = "langchain"
        os.environ["AGENT_RUNTIME_STRICT"] = "false"
        os.environ["AGENT_DISPATCH_MODE"] = "graph"
        os.environ["AGENT_PLANNER_ENABLED"] = "false"
        os.environ.pop("DEEPSEEK_API_KEY", None)

        with ExitStack() as stack:
            stack.enter_context(patch.object(agent_mod, "_extract_source_compare_request", lambda _m: None))
            stack.enter_context(patch.object(agent_mod, "_extract_compare_request", lambda _m: None))
            stack.enter_context(patch.object(agent_mod, "_extract_timeline_request", lambda _m: None))
            stack.enter_context(patch.object(agent_mod, "_extract_landscape_request", lambda _m: None))
            stack.enter_context(patch.object(agent_mod, "_extract_trend_request", lambda _m: None))
            stack.enter_context(patch.object(agent_mod, "_extract_fulltext_request", lambda _m: None))
            stack.enter_context(patch.object(agent_mod, "_extract_query_request", lambda _m: None))
            stack.enter_context(patch.object(agent_mod, "run_runtime_pipeline", lambda **_kwargs: "graph runtime ok"))
            out = agent_mod.generate_response([], "你好")
            self.assertEqual(out, "graph runtime ok")

        if old_runtime is None:
            os.environ.pop("AGENT_RUNTIME", None)
        else:
            os.environ["AGENT_RUNTIME"] = old_runtime

        if old_strict is None:
            os.environ.pop("AGENT_RUNTIME_STRICT", None)
        else:
            os.environ["AGENT_RUNTIME_STRICT"] = old_strict

        if old_dispatch is None:
            os.environ.pop("AGENT_DISPATCH_MODE", None)
        else:
            os.environ["AGENT_DISPATCH_MODE"] = old_dispatch

        if old_planner_enabled is None:
            os.environ.pop("AGENT_PLANNER_ENABLED", None)
        else:
            os.environ["AGENT_PLANNER_ENABLED"] = old_planner_enabled

        if old_deepseek_key is None:
            os.environ.pop("DEEPSEEK_API_KEY", None)
        else:
            os.environ["DEEPSEEK_API_KEY"] = old_deepseek_key

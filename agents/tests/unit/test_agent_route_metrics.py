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
        agent_mod._metrics_inc("requests_total", 2)
        agent_mod._metrics_inc("langchain_attempts", 2)
        agent_mod._metrics_inc("langchain_success", 1)
        agent_mod._metrics_inc("langchain_fallback", 1)
        agent_mod._metrics_inc("compare_forced", 1)
        agent_mod._metrics_inc("landscape_forced", 1)
        agent_mod._metrics_inc("landscape_low_evidence", 1)

        snapshot = agent_mod.get_route_metrics_snapshot()
        self.assertEqual(snapshot["requests_total"], 2)
        self.assertEqual(snapshot["langchain_attempts"], 2)
        self.assertAlmostEqual(snapshot["fallback_rate_total"], 0.5, places=6)
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

    def test_extract_timeline_request_not_triggered_by_recent_news_only(self) -> None:
        req = agent_mod._extract_timeline_request("最近谷歌新闻")
        self.assertIsNone(req)

    def test_extract_landscape_request_weak_ecosystem_without_role_intent(self) -> None:
        req = agent_mod._extract_landscape_request("Please discuss Apple ecosystem accessories compatibility.")
        self.assertIsNone(req)

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

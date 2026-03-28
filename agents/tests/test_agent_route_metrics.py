"""Tests for agent route metrics snapshot helpers."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

AGENTS_DIR = Path(__file__).resolve().parents[1]
if str(AGENTS_DIR) not in sys.path:
    sys.path.insert(0, str(AGENTS_DIR))

import agent as agent_mod


def _patch_attr(target: object, name: str, new_value):
    old = getattr(target, name)
    setattr(target, name, new_value)
    return old


class AgentRouteMetricsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old_flag = os.environ.get("AGENT_ROUTE_METRICS")
        os.environ["AGENT_ROUTE_METRICS"] = "true"
        agent_mod.reset_route_metrics()

    def tearDown(self) -> None:
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

    def test_generate_response_fallback_path_increments_metrics(self) -> None:
        old_runtime = os.environ.get("AGENT_RUNTIME")
        old_strict = os.environ.get("AGENT_RUNTIME_STRICT")
        os.environ["AGENT_RUNTIME"] = "langchain"
        os.environ["AGENT_RUNTIME_STRICT"] = "false"

        old_extract_compare = _patch_attr(agent_mod, "_extract_compare_request", lambda _: None)
        old_extract_timeline = _patch_attr(agent_mod, "_extract_timeline_request", lambda _: None)
        old_generate_langgraph = _patch_attr(
            agent_mod, "_generate_langgraph", lambda _h, _m: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        old_generate_legacy = _patch_attr(agent_mod, "_generate_legacy", lambda _h, _m: "legacy_ok")

        try:
            out = agent_mod.generate_response([], "hello")
            self.assertEqual(out, "legacy_ok")
            snapshot = agent_mod.get_route_metrics_snapshot()
            self.assertEqual(snapshot["requests_total"], 1)
            self.assertEqual(snapshot["langchain_attempts"], 1)
            self.assertEqual(snapshot["langchain_fallback"], 1)
            self.assertEqual(snapshot["langchain_success"], 0)
        finally:
            setattr(agent_mod, "_extract_compare_request", old_extract_compare)
            setattr(agent_mod, "_extract_timeline_request", old_extract_timeline)
            setattr(agent_mod, "_generate_langgraph", old_generate_langgraph)
            setattr(agent_mod, "_generate_legacy", old_generate_legacy)

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

        old_extract_compare = _patch_attr(agent_mod, "_extract_compare_request", lambda _: None)
        old_extract_timeline = _patch_attr(agent_mod, "_extract_timeline_request", lambda _: None)
        old_extract_landscape = _patch_attr(
            agent_mod, "_extract_landscape_request", lambda _m: ("security", 30, ["CrowdStrike", "Cloudflare"])
        )
        old_analyze_landscape = _patch_attr(
            agent_mod,
            "analyze_landscape",
            lambda **_kwargs: (
                "Landscape snapshot: topic=security (last 30 days)\n"
                "Coverage: topic_articles=50, matched_entity_articles=20, active_entities=2/2\n"
                "Evidence URLs:\n"
                "  [CrowdStrike] #1 [TechCrunch] A | points=10 | 2026-03-20 10:00 | https://a.com\n"
                "  [Cloudflare] #1 [HackerNews] B | points=8 | 2026-03-21 11:00 | https://b.com\n"
                "  [CrowdStrike] #2 [TechCrunch] C | points=7 | 2026-03-22 11:00 | https://c.com\n"
                "  [Cloudflare] #2 [HackerNews] D | points=6 | 2026-03-23 11:00 | https://d.com\n"
                "Confidence: Medium"
            ),
        )
        old_analyze_landscape_output = _patch_attr(
            agent_mod, "_analyze_landscape_output", lambda **_kwargs: "## 格局结论\n- 样本显示安全赛道讨论分化。"
        )

        try:
            out = agent_mod.generate_response([], "当今世界安全领域格局是什么？")
            self.assertIn("格局结论", out)
            self.assertIn("证据来源", out)
            snapshot = agent_mod.get_route_metrics_snapshot()
            self.assertEqual(snapshot["requests_total"], 1)
            self.assertEqual(snapshot["landscape_forced"], 1)
            self.assertEqual(snapshot["landscape_low_evidence"], 0)
            self.assertAlmostEqual(snapshot["landscape_low_evidence_rate"], 0.0, places=6)
            self.assertEqual(snapshot["langchain_attempts"], 0)
        finally:
            setattr(agent_mod, "_extract_compare_request", old_extract_compare)
            setattr(agent_mod, "_extract_timeline_request", old_extract_timeline)
            setattr(agent_mod, "_extract_landscape_request", old_extract_landscape)
            setattr(agent_mod, "analyze_landscape", old_analyze_landscape)
            setattr(agent_mod, "_analyze_landscape_output", old_analyze_landscape_output)

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

        old_extract_compare = _patch_attr(agent_mod, "_extract_compare_request", lambda _: None)
        old_extract_timeline = _patch_attr(agent_mod, "_extract_timeline_request", lambda _: None)
        old_extract_landscape = _patch_attr(
            agent_mod, "_extract_landscape_request", lambda _m: ("business", 30, ["Microsoft"])
        )
        old_analyze_landscape = _patch_attr(
            agent_mod,
            "analyze_landscape",
            lambda **_kwargs: (
                "Landscape snapshot: topic=business (last 30 days)\n"
                "Coverage: topic_articles=40, matched_entity_articles=2, active_entities=1/1\n"
                "Evidence URLs:\n"
                "  [Microsoft] #1 [TechCrunch] A | points=10 | 2026-03-20 10:00 | https://a.com\n"
                "Confidence: Low"
            ),
        )
        old_analyze_landscape_output = _patch_attr(
            agent_mod, "_analyze_landscape_output", lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("should_not_run"))
        )

        try:
            out = agent_mod.generate_response([], "business landscape and roles?")
            self.assertIn("evidence is insufficient", out.lower())
            self.assertIn("confidence: low", out.lower())
            snapshot = agent_mod.get_route_metrics_snapshot()
            self.assertEqual(snapshot["requests_total"], 1)
            self.assertEqual(snapshot["landscape_forced"], 1)
            self.assertEqual(snapshot["landscape_low_evidence"], 1)
            self.assertAlmostEqual(snapshot["landscape_low_evidence_rate"], 1.0, places=6)
            self.assertEqual(snapshot["langchain_attempts"], 0)
        finally:
            setattr(agent_mod, "_extract_compare_request", old_extract_compare)
            setattr(agent_mod, "_extract_timeline_request", old_extract_timeline)
            setattr(agent_mod, "_extract_landscape_request", old_extract_landscape)
            setattr(agent_mod, "analyze_landscape", old_analyze_landscape)
            setattr(agent_mod, "_analyze_landscape_output", old_analyze_landscape_output)

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


if __name__ == "__main__":
    unittest.main()

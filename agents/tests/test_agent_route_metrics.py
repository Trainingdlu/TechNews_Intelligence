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

        snapshot = agent_mod.get_route_metrics_snapshot()
        self.assertEqual(snapshot["requests_total"], 2)
        self.assertEqual(snapshot["langchain_attempts"], 2)
        self.assertAlmostEqual(snapshot["fallback_rate_total"], 0.5, places=6)
        self.assertAlmostEqual(snapshot["fallback_rate_langchain"], 0.5, places=6)
        self.assertAlmostEqual(snapshot["langchain_success_rate"], 0.5, places=6)
        self.assertAlmostEqual(snapshot["forced_route_rate"], 0.5, places=6)

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


if __name__ == "__main__":
    unittest.main()

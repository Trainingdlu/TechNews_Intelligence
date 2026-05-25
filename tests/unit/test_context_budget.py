from __future__ import annotations

from agent.context_manager import render_context_for_prompt
from eval.context_budget.measure_context_tokens import (
    build_scenarios,
    count_approx,
    measure_scenario,
    split_sections,
)


def test_measurement_runs_and_f1_saves_tokens() -> None:
    results = [measure_scenario(name, pack) for name, pack in build_scenarios()]

    # S1 baseline: first turn has no context cost.
    assert results[0]["total_approx"] == 0

    # Loaded scenarios carry a non-trivial pack and F1 de-dup removes real tokens.
    for r in results[1:]:
        assert r["total_approx"] > 0
        assert r["pre_approx"] >= r["total_approx"]
        assert (r["pre_approx"] - r["total_approx"]) > 0
        assert r["sections"]
        # F2: the tool profile is non-empty (keeps candidate URLs) but strictly
        # smaller than the full pack (drops the prose sections).
        assert 0 < r["tool_approx"] < r["total_approx"]


def test_split_sections_attributes_without_inflating_total() -> None:
    _, pack = build_scenarios()[1]  # S2
    rendered = render_context_for_prompt(pack)
    sections = split_sections(rendered)

    assert "Selected memory evidence" in sections
    section_sum = sum(count_approx(text) for text in sections.values())
    assert 0 < section_sum <= count_approx(rendered) + 5

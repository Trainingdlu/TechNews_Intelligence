"""Shared runtime factories for skill registry and tool hooks.

This module is the canonical home of the runtime component factories used by
the unified ReAct agent path.
"""

from __future__ import annotations

from .skill_registry import SkillRegistry
from .tool_hooks import ToolHookRunner
from .skill_catalog import iter_skill_definitions


_DEFAULT_REGISTRY: SkillRegistry | None = None
_DEFAULT_HOOK_RUNNER: ToolHookRunner | None = None


def build_default_registry() -> SkillRegistry:
    """Build the default in-process skill registry."""

    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is not None:
        return _DEFAULT_REGISTRY

    registry = SkillRegistry()
    for definition in iter_skill_definitions():
        registry.register(
            name=definition.name,
            input_model=definition.input_model,
            handler=definition.handler,
            description=definition.description,
        )
    _DEFAULT_REGISTRY = registry
    return registry


def build_default_hook_runner() -> ToolHookRunner:
    """Build the default shared hook runner."""

    global _DEFAULT_HOOK_RUNNER
    if _DEFAULT_HOOK_RUNNER is not None:
        return _DEFAULT_HOOK_RUNNER
    _DEFAULT_HOOK_RUNNER = ToolHookRunner()
    return _DEFAULT_HOOK_RUNNER

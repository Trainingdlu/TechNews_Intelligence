"""Shared bootstrap helpers for unit tests."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_project_on_path() -> Path:
    """Ensure repository root is importable and return its path."""
    project_root = Path(__file__).resolve().parents[2]
    root_path = str(project_root)
    if root_path not in sys.path:
        sys.path.insert(0, root_path)
    return project_root


def ensure_agents_on_path() -> Path:
    """Backward-compatible alias used by existing tests."""
    return ensure_project_on_path()



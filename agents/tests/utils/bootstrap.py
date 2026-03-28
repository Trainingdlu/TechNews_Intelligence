"""Shared bootstrap helpers for unit tests."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_agents_on_path() -> Path:
    """Ensure `agents/` package root is importable and return its path."""
    agents_dir = Path(__file__).resolve().parents[2]
    agents_path = str(agents_dir)
    if agents_path not in sys.path:
        sys.path.insert(0, agents_path)
    return agents_dir


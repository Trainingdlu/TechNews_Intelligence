"""Guardrails for UTF-8 Chinese rule constants.

Chinese-related rule files must keep UTF-8 literal characters instead of \\uXXXX escapes.
"""

from __future__ import annotations

import re
from pathlib import Path


def test_chinese_rule_files_do_not_use_unicode_escape_literals() -> None:
    files = [
        Path("agent/clarification.py"),
        Path("agent/core/intent.py"),
        Path("agent/core/evidence.py"),
    ]
    pattern = re.compile(r"\\u[0-9a-fA-F]{4}")

    for file_path in files:
        content = file_path.read_text(encoding="utf-8")
        assert (
            pattern.search(content) is None
        ), f"{file_path} contains unicode escape literals; use UTF-8 Chinese text directly."

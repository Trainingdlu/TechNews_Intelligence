"""Guardrails for UTF-8 Chinese rule constants.

Chinese-related rule files must keep UTF-8 literal characters instead of \\uXXXX escapes.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


CHINESE_RULE_FILES = (
    Path("agent/clarification.py"),
    Path("agent/core/intent.py"),
    Path("agent/core/evidence.py"),
)
UNICODE_ESCAPE_RE = re.compile(r"\\u[0-9a-fA-F]{4}")
CJK_RE = re.compile(r"[一-龥]")


@pytest.mark.parametrize("file_path", CHINESE_RULE_FILES, ids=lambda path: path.as_posix())
def test_chinese_rule_files_use_utf8_literals(file_path: Path) -> None:
    content = file_path.read_text(encoding="utf-8")

    assert CJK_RE.search(content), f"{file_path} no longer contains direct Chinese literals."
    assert UNICODE_ESCAPE_RE.search(content) is None, (
        f"{file_path} contains unicode escape literals; use UTF-8 Chinese text directly."
    )

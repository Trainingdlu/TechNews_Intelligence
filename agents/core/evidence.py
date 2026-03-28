"""Evidence formatting helpers."""

from __future__ import annotations

import os
import re
from typing import Any, Callable


def contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def extract_urls(text: str) -> list[str]:
    if not text:
        return []
    urls = re.findall(r"https?://[^\s)\]]+", text)
    dedup: list[str] = []
    seen: set[str] = set()
    for u in urls:
        if u not in seen:
            dedup.append(u)
            seen.add(u)
    return dedup


_SOURCE_HEADER_RE = re.compile(
    r"^\s{0,3}(?:#{1,6}\s*)?(?:来源|证据来源|source(?:s)?|evidence\s+sources?)\s*:?\s*$",
    re.IGNORECASE,
)


def strip_existing_source_section(text: str) -> str:
    lines = (text or "").splitlines()
    start = None
    for i, line in enumerate(lines):
        if _SOURCE_HEADER_RE.match(line.strip()):
            start = i
            break
    if start is None:
        return (text or "").rstrip()
    return "\n".join(lines[:start]).rstrip()


def apply_inline_citations(text: str, ordered_urls: list[str]) -> str:
    out = text or ""
    for idx, url in enumerate(ordered_urls, 1):
        cite = f"[{idx}]"
        out = out.replace(f"`{url}`", cite)
        out = out.replace(url, cite)
    return out


def max_source_urls() -> int:
    try:
        raw = os.getenv("AGENT_MAX_SOURCE_URLS", os.getenv("BOT_MAX_CITATION_URLS", "12"))
        return max(1, min(30, int(raw)))
    except Exception:
        return 12


def build_source_section(ordered_urls: list[str], user_message: str) -> str:
    header = "## 来源" if contains_cjk(user_message) else "## Sources"
    lines = [header]
    for idx, url in enumerate(ordered_urls, 1):
        lines.append(f"- [{idx}] {url}")
    return "\n".join(lines)


def decorate_response_with_sources(
    text: str,
    user_message: str,
    lookup_url_titles: Callable[[list[str]], dict[str, str]] | None = None,
) -> tuple[str, dict[str, str]]:
    """Normalize output into citation style + source section."""
    raw = (text or "").strip()
    if not raw:
        return raw, {}

    body = strip_existing_source_section(raw)
    urls = extract_urls(body) if body else []
    if not urls:
        urls = extract_urls(raw)
    if not urls:
        return raw, {}

    ordered_urls = urls[:max_source_urls()]
    title_map: dict[str, str] = {}
    if lookup_url_titles is not None:
        try:
            title_map = lookup_url_titles(ordered_urls)
        except Exception as exc:
            print(f"[Warn] lookup_url_titles in evidence helper failed: {exc}")
            title_map = {}

    render_body = body if body else raw
    cited_body = apply_inline_citations(render_body, ordered_urls)
    source_section = build_source_section(ordered_urls, user_message)
    merged = f"{cited_body.rstrip()}\n\n{source_section}".strip()
    return merged, title_map


def ensure_evidence_section(answer: str, source_output: str, user_message: str, max_urls: int = 8) -> str:
    """Ensure answer includes an evidence URL section based on source output."""
    if not answer:
        return answer
    source_urls = extract_urls(source_output)
    if not source_urls:
        return answer

    answer_urls = extract_urls(answer)
    has_section = bool(_SOURCE_HEADER_RE.search(answer))
    if has_section and answer_urls:
        return answer

    header = "## 证据来源" if contains_cjk(user_message) else "## Evidence Sources"
    lines = [f"- {u}" for u in source_urls[:max_urls]]
    return f"{answer.rstrip()}\n\n{header}\n" + "\n".join(lines)


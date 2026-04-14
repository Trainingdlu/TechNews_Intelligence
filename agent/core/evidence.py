"""Evidence formatting helpers."""

from __future__ import annotations

import os
import re
from typing import Callable


_CJK_BASIC_RE = re.compile(r"[\u4e00-\u9fff]")
_SOURCE_HEADER_RE = re.compile(
    r"^\s{0,3}(?:#{1,6}\s*)?(?:来源|证据来源|sources?|evidence\s+sources?)\s*:?.*",
    re.IGNORECASE,
)
_INLINE_CITATION_RE = re.compile(r"\[(\d{1,3})\]")
_INLINE_CITATION_LIST_RE = re.compile(
    r"\[\d{1,3}\](?:\s*(?:[,，、;]|(?:and|or|和|或)\s*)\[\d{1,3}\])+",
    re.IGNORECASE,
)
_PAREN_CITATION_RE = re.compile(r"[\(（]\s*\[(\d{1,3})\]\s*[\)）]")
_SOURCE_HASH_CITATION_RE = re.compile(r"\[[^\]\n]{1,80}\]\s*[#＃]\s*(\d{1,3})")
_NESTED_CITATION_RE = re.compile(r"(?:\[\[|\(\[|\[\^|\[#)\s*(\d{1,3})\s*(?:\]\]|\]\)|\]\^|\])")
_BRACKET_MULTI_CITATION_RE = re.compile(r"\[((?:\d{1,3}\s*(?:[,，、;/]\s*\d{1,3}\s*)+))\]")


def contains_cjk(text: str) -> bool:
    return bool(_CJK_BASIC_RE.search(text or ""))


def extract_urls(text: str) -> list[str]:
    if not text:
        return []
    urls = re.findall(r"https?://[^\s)\]]+", text)
    dedup: list[str] = []
    seen: set[str] = set()
    for url in urls:
        if url not in seen:
            dedup.append(url)
            seen.add(url)
    return dedup


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


def normalize_inline_citation_styles(text: str) -> str:
    """Normalize citation variants into canonical [n] form."""
    body = text or ""
    if not body:
        return body

    def _expand_multi(match: re.Match[str]) -> str:
        nums = re.findall(r"\d{1,3}", match.group(1))
        dedup: list[str] = []
        seen: set[str] = set()
        for num in nums:
            if num not in seen:
                seen.add(num)
                dedup.append(num)
        return ", ".join(f"[{num}]" for num in dedup)

    body = _BRACKET_MULTI_CITATION_RE.sub(_expand_multi, body)
    body = _PAREN_CITATION_RE.sub(lambda m: f"[{m.group(1)}]", body)
    body = _SOURCE_HASH_CITATION_RE.sub(lambda m: f"[{m.group(1)}]", body)
    body = _NESTED_CITATION_RE.sub(lambda m: f"[{m.group(1)}]", body)
    return body


def has_inline_citation_in_body(text: str) -> bool:
    normalized = normalize_inline_citation_styles(text or "")
    body = strip_existing_source_section(normalized)
    return bool(_INLINE_CITATION_RE.search(body))


def max_inline_citation_index(text: str) -> int:
    maximum = 0
    for m in _INLINE_CITATION_RE.finditer(text or ""):
        try:
            idx = int(m.group(1))
        except Exception:
            continue
        if 1 <= idx <= 999 and idx > maximum:
            maximum = idx
    return maximum


def citation_indices_in_order(text: str) -> list[int]:
    ordered: list[int] = []
    seen: set[int] = set()
    for m in _INLINE_CITATION_RE.finditer(text or ""):
        try:
            idx = int(m.group(1))
        except Exception:
            continue
        if idx < 1 or idx > 999:
            continue
        if idx in seen:
            continue
        seen.add(idx)
        ordered.append(idx)
    return ordered


def remap_inline_citations(text: str, index_map: dict[int, int], strip_unmapped: bool = False) -> str:
    if not index_map and not strip_unmapped:
        return text

    lines = (text or "").split("\n")
    out_lines = []

    for line in lines:
        if strip_unmapped:
            matches = _INLINE_CITATION_RE.findall(line)
            if matches:
                has_valid = any(int(m) in index_map for m in matches if m.isdigit())
                has_invalid = any(int(m) not in index_map for m in matches if m.isdigit())
                if has_invalid and not has_valid:
                    continue

        def _replace(match: re.Match[str]) -> str:
            try:
                old = int(match.group(1))
            except Exception:
                return match.group(0)
            new = index_map.get(old)
            if new is None:
                return "" if strip_unmapped else match.group(0)
            return f"[{new}]"

        out_lines.append(_INLINE_CITATION_RE.sub(_replace, line))

    return "\n".join(out_lines)


def dedupe_redundant_inline_citation_runs(text: str) -> str:
    body = text or ""
    if not body:
        return body

    def _replace(match: re.Match[str]) -> str:
        segment = match.group(0)
        refs: list[int] = []
        for token in _INLINE_CITATION_RE.findall(segment):
            try:
                refs.append(int(token))
            except Exception:
                continue
        if not refs:
            return segment

        unique: list[int] = []
        seen: set[int] = set()
        for idx in refs:
            if idx in seen:
                continue
            seen.add(idx)
            unique.append(idx)
        if len(unique) == len(refs):
            return segment
        return ", ".join(f"[{idx}]" for idx in unique)

    return _INLINE_CITATION_LIST_RE.sub(_replace, body)


def _cleanup_citation_artifacts(text: str) -> str:
    body = text or ""
    if not body:
        return body
    body = re.sub(r"\[\s*[,，;、/]\s*", "[", body)
    body = re.sub(r"\s*[,，;、/]\s*\]", "]", body)
    body = re.sub(r"\[\s*\]", "", body)
    body = re.sub(r"\(\s*[,，;、/]\s*\)", "", body)
    body = re.sub(r"(,\s*){2,}", ", ", body)
    body = re.sub(r"\],\s*(and|or|和|或)\s+\[", r"] \1 [", body, flags=re.IGNORECASE)
    return body


def compact_citations_and_urls(
    cited_body: str,
    ordered_urls: list[str],
    valid_urls: list[str] | set[str] | None = None,
) -> tuple[str, list[str]]:
    refs = citation_indices_in_order(cited_body)
    if not refs:
        return cited_body, ordered_urls

    compact_urls: list[str] = []
    index_map: dict[int, int] = {}
    for old_idx in refs:
        if 1 <= old_idx <= len(ordered_urls):
            url = ordered_urls[old_idx - 1]
            if valid_urls is not None and url not in valid_urls:
                continue
            index_map[old_idx] = len(compact_urls) + 1
            compact_urls.append(url)

    compact_body = remap_inline_citations(cited_body, index_map, strip_unmapped=(valid_urls is not None))
    compact_body = dedupe_redundant_inline_citation_runs(compact_body)
    compact_body = _cleanup_citation_artifacts(compact_body)
    return compact_body, compact_urls


def max_source_urls() -> int:
    """Return URL cap. <=0 means no truncation."""
    try:
        raw = os.getenv("AGENT_MAX_SOURCE_URLS", os.getenv("BOT_MAX_CITATION_URLS", "0"))
        cap = int(raw)
        if cap <= 0:
            return 1000
        return max(1, min(1000, cap))
    except Exception:
        return 1000


def build_source_section(ordered_urls: list[str], user_message: str, title_map: dict[str, str] | None = None) -> str:
    header = "## 来源" if contains_cjk(user_message) else "## Sources"
    lines = [header]
    for idx, url in enumerate(ordered_urls, 1):
        if title_map and url in title_map:
            title = title_map[url]
            lines.append(f"- [{idx}] [{title}]({url})")
        else:
            lines.append(f"- [{idx}] {url}")
    return "\n".join(lines)


def decorate_response_with_sources(
    text: str,
    user_message: str,
    lookup_url_titles: Callable[[list[str]], dict[str, str]] | None = None,
    valid_urls: list[str] | set[str] | None = None,
) -> tuple[str, dict[str, str]]:
    """Normalize output into citation style + source section."""
    raw = normalize_inline_citation_styles((text or "").strip())
    if not raw:
        return raw, {}

    body = strip_existing_source_section(raw)
    urls = extract_urls(body) if body else []
    if not urls:
        urls = extract_urls(raw)

    if not urls and valid_urls:
        if isinstance(valid_urls, list):
            urls = []
            seen = set()
            for u in valid_urls:
                if u not in seen:
                    urls.append(u)
                    seen.add(u)
        else:
            urls = list(valid_urls)

    if not urls:
        return raw, {}

    required_urls = max_inline_citation_index(body if body else raw)
    url_cap = max(max_source_urls(), required_urls)
    ordered_urls = urls[:url_cap] if url_cap > 0 else list(urls)

    title_map: dict[str, str] = {}
    if lookup_url_titles is not None:
        try:
            title_map = lookup_url_titles(ordered_urls)
        except Exception as exc:
            print(f"[Warn] lookup_url_titles in evidence helper failed: {exc}")
            title_map = {}

    render_body = normalize_inline_citation_styles(body if body else raw)
    cited_body = apply_inline_citations(render_body, ordered_urls)
    compact_body, compact_urls = compact_citations_and_urls(cited_body, ordered_urls, valid_urls=valid_urls)
    final_urls = compact_urls if compact_urls else ordered_urls

    if valid_urls is not None:
        final_urls = [u for u in final_urls if u in valid_urls]

    if not final_urls:
        return _cleanup_citation_artifacts(compact_body).rstrip(), title_map

    # Final consistency guard: do not allow citation index above source count.
    max_index = len(final_urls)
    identity_map = {idx: idx for idx in range(1, max_index + 1)}
    compact_body = remap_inline_citations(compact_body, identity_map, strip_unmapped=True)
    compact_body = dedupe_redundant_inline_citation_runs(compact_body)
    compact_body = _cleanup_citation_artifacts(compact_body)

    source_section = build_source_section(final_urls, user_message, title_map=title_map)
    merged = f"{compact_body.rstrip()}\n\n{source_section}".strip()
    return merged, title_map


def ensure_evidence_section(answer: str, source_output: str, user_message: str, max_urls: int = 0) -> str:
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

    header = "## 来源" if contains_cjk(user_message) else "## Sources"
    required_urls = max_inline_citation_index(answer)
    if max_urls <= 0:
        effective_max = max(len(source_urls), required_urls)
    else:
        effective_max = max(max_urls, required_urls)
    lines = [f"{url}" for url in source_urls[:effective_max]]
    return f"{answer.rstrip()}\n\n{header}\n" + "\n".join(lines)

"""Build event cards from the news database for event-driven eval datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

try:
    from news_eval_schema import validate_event_card, write_jsonl
except ImportError:  # pragma: no cover
    from .news_eval_schema import validate_event_card, write_jsonl


KNOWN_ENTITIES = (
    "OpenAI",
    "Anthropic",
    "Claude",
    "ChatGPT",
    "Google",
    "Gemini",
    "Microsoft",
    "Meta",
    "Apple",
    "Amazon",
    "AWS",
    "Nvidia",
    "DeepSeek",
    "Mistral",
    "xAI",
    "Perplexity",
    "Framework",
    "Valve",
    "Zed",
    "Ghostty",
    "Bambu",
)

BROAD_ENTITIES = {
    "OpenAI",
    "Google",
    "Apple",
    "Microsoft",
    "Amazon",
    "Meta",
    "Anthropic",
    "AI",
}

BROAD_PRODUCTS = {
    "android",
    "chrome",
    "claude",
    "chatgpt",
    "github",
}

STOPWORDS = {
    "about",
    "after",
    "and",
    "for",
    "from",
    "into",
    "new",
    "news",
    "the",
    "this",
    "with",
    "ai",
    "api",
}

PRODUCT_PATTERNS: tuple[str, ...] = (
    r"GPT[-\s]?\d(?:\.\d+)?(?:\s*(?:Instant|Pro|Cyber|Codex))?",
    r"Claude(?:\s+(?:Code|Opus|Sonnet|Design|Platform|Cowork))?(?:\s+\d(?:\.\d+)?)?",
    r"DeepSeek\s*(?:V?\d(?:\.\d+)?|v\d)?(?:\s*(?:Pro|Flash))?",
    r"ChatGPT(?:\s+[A-Za-z0-9-]+)?",
    r"Framework\s+Laptop\s+\d+\s*Pro",
    r"Steam(?:\s*手柄|\s*Controller)?",
    r"Bambu\s+Lab",
    r"OrcaSlicer",
    r"reCAPTCHA",
    r"VS Code",
    r"GitHub(?:\s+Actions)?",
    r"Ghostty",
    r"Chrome",
    r"Android",
    r"Copilot",
    r"Zed",
    r"Valve",
)


def _load_eval_env(env_file: Path | None) -> None:
    project_root = Path(__file__).resolve().parents[1]
    default_env = project_root / "agent" / ".env"
    if default_env.exists():
        load_dotenv(dotenv_path=default_env, override=True)
    if env_file:
        load_dotenv(dotenv_path=env_file.resolve(), override=True)


def _normalize_text(value: str) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[\[\]【】（）()《》<>\"'“”‘’|:：,，.。!！?？;；/\\_-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _event_key(title: str, summary: str) -> str:
    normalized = _normalize_text(title) or _normalize_text(summary)
    tokens = [token for token in normalized.split() if len(token) >= 2]
    if not tokens:
        return hashlib.sha1((title + summary).encode("utf-8")).hexdigest()[:12]
    return " ".join(tokens[:10])


def _canonical_token(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _extract_products(title: str, summary: str = "") -> set[str]:
    haystack = f"{title}\n{summary}"
    products: set[str] = set()
    for pattern in PRODUCT_PATTERNS:
        for match in re.finditer(pattern, haystack, flags=re.IGNORECASE):
            product = _canonical_token(match.group(0))
            if product:
                products.add(product)
    return products


def _anchor_tokens(title: str, summary: str = "") -> set[str]:
    text = _normalize_text(f"{title} {summary}")
    tokens = {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9.+-]{1,}", text)
        if len(token) >= 2 and token not in STOPWORDS
    }
    tokens.update(_extract_products(title, summary))
    tokens.update(_canonical_token(entity) for entity in _extract_entities(title, summary))
    return {token for token in tokens if token and token not in STOPWORDS}


def _row_profile(row: dict[str, Any]) -> dict[str, set[str]]:
    title = str(row.get("title") or "")
    summary = str(row.get("summary") or "")
    entities = {_canonical_token(entity) for entity in _extract_entities(title, summary)}
    products = _extract_products(title, summary)
    tokens = _anchor_tokens(title, summary)
    return {
        "entities": entities,
        "products": products,
        "tokens": tokens,
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _related_score(primary: dict[str, Any], candidate: dict[str, Any]) -> float:
    if _event_key(str(primary.get("title") or ""), str(primary.get("summary") or "")) == _event_key(
        str(candidate.get("title") or ""),
        str(candidate.get("summary") or ""),
    ):
        return 1.0

    left = _row_profile(primary)
    right = _row_profile(candidate)
    shared_products = left["products"] & right["products"]
    shared_specific_products = {item for item in shared_products if item not in BROAD_PRODUCTS}
    shared_entities = left["entities"] & right["entities"]
    shared_specific_entities = {item for item in shared_entities if item not in {e.lower() for e in BROAD_ENTITIES}}
    shared_tokens = left["tokens"] & right["tokens"]
    shared_specific_tokens = {
        item
        for item in shared_tokens
        if item not in BROAD_PRODUCTS and item not in {e.lower() for e in BROAD_ENTITIES}
    }
    token_jaccard = _jaccard(left["tokens"], right["tokens"])

    score = token_jaccard
    if shared_specific_products:
        score += 0.65
    elif shared_products and len(shared_specific_tokens) >= 2:
        score += 0.25
    if shared_specific_entities:
        score += 0.25
    if len(shared_specific_tokens) >= 3:
        score += 0.15
    return score


def _is_related_row(primary: dict[str, Any], candidate: dict[str, Any]) -> bool:
    return _related_score(primary, candidate) >= 0.45


def _slug(value: str) -> str:
    normalized = _normalize_text(value)
    chars = []
    for ch in normalized:
        if ch.isalnum():
            chars.append(ch)
        elif ch.isspace():
            chars.append("_")
    slug = re.sub(r"_+", "_", "".join(chars)).strip("_")
    return slug[:48] or "event"


def _date_part(value: Any) -> str:
    text = ""
    if hasattr(value, "strftime"):
        text = value.strftime("%Y_%m_%d")
    else:
        text = str(value or "")[:10].replace("-", "_")
    return text or "unknown_date"


def _iso_date(value: Any) -> str:
    if hasattr(value, "date"):
        return value.date().isoformat()
    text = str(value or "").strip()
    return text[:10] if text else ""


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？.!?])\s+|[。！？!?]\s*", str(text or "").strip())
    return [part.strip(" -\t\r\n") for part in parts if len(part.strip()) >= 8]


def _extract_entities(title: str, summary: str) -> list[str]:
    haystack = f"{title}\n{summary}"
    out: list[str] = []
    seen: set[str] = set()
    for entity in KNOWN_ENTITIES:
        if re.search(re.escape(entity), haystack, flags=re.IGNORECASE):
            key = entity.lower()
            if key not in seen:
                seen.add(key)
                out.append(entity)
    for match in re.findall(r"\b[A-Z][A-Za-z0-9]+(?:[- ][A-Z][A-Za-z0-9]+){0,2}\b", haystack):
        if len(match) < 3 or match.lower() in seen:
            continue
        seen.add(match.lower())
        out.append(match)
        if len(out) >= 8:
            break
    return out


def _fetch_news_rows(*, days: int, limit: int) -> list[dict[str, Any]]:
    from services.db import get_conn, put_conn  # pylint: disable=import-outside-toplevel

    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                COALESCE(v.title_cn, v.title) AS title_norm,
                v.title,
                COALESCE(v.summary, '') AS summary,
                v.url,
                COALESCE(v.source_type, '') AS source_type,
                v.created_at,
                COALESCE(v.points, 0) AS points
            FROM view_dashboard_news v
            WHERE v.url IS NOT NULL
              AND v.created_at >= NOW() - (%s || ' days')::interval
            ORDER BY COALESCE(v.points, 0) DESC, v.created_at DESC
            LIMIT %s
            """,
            (int(days), int(limit)),
        )
        rows = cur.fetchall()
        cur.close()
    finally:
        put_conn(conn)

    out: list[dict[str, Any]] = []
    for row in rows:
        title_norm, title, summary, url, source_type, created_at, points = row
        title_text = str(title_norm or title or "").strip()
        url_text = str(url or "").strip()
        summary_text = str(summary or "").strip()
        if not title_text or not url_text:
            continue
        out.append(
            {
                "title": title_text,
                "raw_title": str(title or "").strip(),
                "summary": summary_text,
                "url": url_text,
                "source": str(source_type or "").strip() or "unknown",
                "created_at": created_at,
                "points": int(points or 0),
            }
        )
    return out


def _build_card(group_rows: list[dict[str, Any]], *, max_facts: int) -> dict[str, Any] | None:
    if not group_rows:
        return None
    rows = sorted(group_rows, key=lambda row: (int(row.get("points") or 0), str(row.get("created_at") or "")), reverse=True)
    primary = rows[0]
    title = str(primary.get("title") or "").strip()
    summary = str(primary.get("summary") or "").strip()
    url = str(primary.get("url") or "").strip()
    if not title or not url:
        return None

    facts: list[dict[str, str]] = []
    for row in rows:
        row_summary = str(row.get("summary") or "").strip()
        row_title = str(row.get("title") or "").strip()
        fact_candidates = _split_sentences(row_summary) or [row_title]
        for sentence in fact_candidates:
            if len(facts) >= max_facts:
                break
            fact_url = str(row.get("url") or "").strip()
            if not sentence or not fact_url:
                continue
            facts.append({"claim": sentence, "quote": sentence, "url": fact_url})
        if len(facts) >= max_facts:
            break
    if not facts:
        return None

    dates = [_iso_date(row.get("created_at")) for row in rows if _iso_date(row.get("created_at"))]
    start = min(dates) if dates else _iso_date(primary.get("created_at"))
    end = max(dates) if dates else start
    digest = hashlib.sha1("|".join(str(row.get("url") or "") for row in rows).encode("utf-8")).hexdigest()[:8]
    event_id = f"{_slug(title)}_{_date_part(primary.get('created_at'))}_{digest}"
    urls = [str(row.get("url") or "").strip() for row in rows if str(row.get("url") or "").strip()]
    core_urls = [url]
    related_urls = [item for item in urls if item != url]
    entities = _extract_entities(title, summary)
    sources = sorted({str(row.get("source") or "").strip() for row in rows if str(row.get("source") or "").strip()})
    primary_profile = _row_profile(primary)
    return {
        "event_id": event_id,
        "event_title": title,
        "entities": entities,
        "time_window": {"start": start, "end": end},
        "core_urls": core_urls,
        "related_urls": related_urls,
        "facts": facts,
        "suitable_tasks": ["single_event", "latest_update", "deep_reading"],
        "source_count": len(sources),
        "sources": sources,
        "article_count": len(rows),
        "topic_anchors": sorted((primary_profile["products"] | primary_profile["entities"]))[:10],
        "build_method": "deterministic_title_event_card_v2",
    }


def _cluster_news_rows(rows: list[dict[str, Any]], *, max_related_per_event: int) -> list[list[dict[str, Any]]]:
    sorted_rows = sorted(rows, key=lambda row: (int(row.get("points") or 0), str(row.get("created_at") or "")), reverse=True)
    used_urls: set[str] = set()
    groups: list[list[dict[str, Any]]] = []
    for primary in sorted_rows:
        primary_url = str(primary.get("url") or "").strip()
        if not primary_url or primary_url in used_urls:
            continue
        used_urls.add(primary_url)
        related: list[tuple[float, dict[str, Any]]] = []
        for candidate in sorted_rows:
            candidate_url = str(candidate.get("url") or "").strip()
            if not candidate_url or candidate_url in used_urls:
                continue
            score = _related_score(primary, candidate)
            if score >= 0.45:
                related.append((score, candidate))
        related_rows = [row for _, row in sorted(related, key=lambda item: item[0], reverse=True)[: max(0, max_related_per_event)]]
        for row in related_rows:
            used_urls.add(str(row.get("url") or "").strip())
        groups.append([primary] + related_rows)
    return groups


def build_event_cards(
    *,
    days: int,
    limit: int,
    max_events: int,
    max_facts: int,
    max_related_per_event: int = 5,
) -> list[dict[str, Any]]:
    rows = _fetch_news_rows(days=days, limit=limit)
    groups = _cluster_news_rows(rows, max_related_per_event=max_related_per_event)
    cards: list[dict[str, Any]] = []
    for group_rows in groups:
        card = _build_card(group_rows, max_facts=max_facts)
        if not card:
            continue
        cards.append(validate_event_card(card))
        if len(cards) >= max_events:
            break
    return cards


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build event cards from the news database.")
    parser.add_argument("--output", type=Path, default=Path("eval/datasets/event_cards.jsonl"))
    parser.add_argument("--manifest-output", type=Path, default=Path("eval/datasets/event_cards_manifest.json"))
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--candidate-limit", type=int, default=500)
    parser.add_argument("--max-events", type=int, default=100)
    parser.add_argument("--max-facts-per-event", type=int, default=4)
    parser.add_argument("--max-related-per-event", type=int, default=5)
    parser.add_argument("--env-file", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _load_eval_env(args.env_file)
    cards = build_event_cards(
        days=max(1, int(args.days)),
        limit=max(1, int(args.candidate_limit)),
        max_events=max(1, int(args.max_events)),
        max_facts=max(1, int(args.max_facts_per_event)),
        max_related_per_event=max(0, int(args.max_related_per_event)),
    )
    write_jsonl(args.output, cards)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output": str(args.output),
        "event_count": len(cards),
        "days": int(args.days),
        "candidate_limit": int(args.candidate_limit),
        "max_events": int(args.max_events),
        "max_facts_per_event": int(args.max_facts_per_event),
        "max_related_per_event": int(args.max_related_per_event),
        "build_method": "deterministic_title_event_card_v2",
        "related_url_count": sum(len(card.get("related_urls", []) or []) for card in cards),
        "multi_article_event_count": sum(1 for card in cards if int(card.get("article_count") or 0) > 1),
    }
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[EventCards] output={args.output} events={len(cards)}")
    print(f"[EventCards] manifest={args.manifest_output}")


if __name__ == "__main__":
    main()

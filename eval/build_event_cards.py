"""Build event cards from the news database for event-driven eval datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

try:
    from news_eval_schema import validate_event_card, write_jsonl
except ImportError:  # pragma: no cover
    from .news_eval_schema import validate_event_card, write_jsonl


KNOWN_ENTITIES = (
    ("OpenAI", "OpenAI"),
    ("Anthropic", "Anthropic"),
    ("Claude", "Claude"),
    ("ChatGPT", "ChatGPT"),
    ("Google", "Google"),
    ("谷歌", "Google"),
    ("Gemini", "Gemini"),
    ("Microsoft", "Microsoft"),
    ("微软", "Microsoft"),
    ("Meta", "Meta"),
    ("Apple", "Apple"),
    ("苹果", "Apple"),
    ("Amazon", "Amazon"),
    ("亚马逊", "Amazon"),
    ("AWS", "AWS"),
    ("Nvidia", "Nvidia"),
    ("英伟达", "Nvidia"),
    ("DeepSeek", "DeepSeek"),
    ("Mistral", "Mistral"),
    ("xAI", "xAI"),
    ("Perplexity", "Perplexity"),
    ("GitHub", "GitHub"),
    ("Android", "Android"),
    ("安卓", "Android"),
)

EVENT_TYPE_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("migration", ("迁出", "迁移", "leaving github")),
    ("legal", ("起诉", "诉讼", "法院", "法庭", "legal", "lawsuit")),
    ("leadership", ("CEO", "接任", "交接", "executive", "leadership")),
    ("policy", ("政策", "限制", "实名", "验证", "隐私", "授权", "合规", "policy", "privacy")),
    ("incident", ("故障", "宕机", "中断", "异常", "outage", "incident")),
    ("release", ("发布", "推出", "上线", "正式版", "预览版", "release", "launch")),
    ("business", ("营收", "融资", "IPO", "成本", "价格", "计费", "revenue", "pricing")),
    ("controversy", ("争议", "反对", "质疑", "担忧", "controversy")),
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


def _infer_event_type(title: str, summary: str) -> str:
    haystack = f"{title}\n{summary}".lower()
    for event_type, keywords in EVENT_TYPE_KEYWORDS:
        if any(str(keyword).lower() in haystack for keyword in keywords):
            return event_type
    return "generic"


def _extract_entities(title: str, summary: str) -> list[str]:
    haystack = f"{title}\n{summary}"
    out: list[str] = []
    seen: set[str] = set()
    for alias, canonical in KNOWN_ENTITIES:
        if re.search(re.escape(alias), haystack, flags=re.IGNORECASE):
            key = canonical.lower()
            if key not in seen:
                seen.add(key)
                out.append(canonical)
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
    return {
        "event_id": event_id,
        "event_title": title,
        "entities": entities,
        "time_window": {"start": start, "end": end},
        "core_urls": core_urls,
        "related_urls": related_urls,
        "facts": facts,
        "event_type": _infer_event_type(title, summary),
        "suitable_tasks": ["single_event", "latest_update", "deep_reading"],
        "source_count": len(sources),
        "sources": sources,
        "article_count": len(rows),
        "build_method": "deterministic_title_event_card_v1",
    }


def build_event_cards(
    *,
    days: int,
    limit: int,
    max_events: int,
    max_facts: int,
    require_entities: bool = True,
) -> list[dict[str, Any]]:
    rows = _fetch_news_rows(days=days, limit=limit)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_event_key(str(row.get("title") or ""), str(row.get("summary") or ""))].append(row)

    cards: list[dict[str, Any]] = []
    for group_rows in sorted(groups.values(), key=lambda items: max(int(row.get("points") or 0) for row in items), reverse=True):
        card = _build_card(group_rows, max_facts=max_facts)
        if not card:
            continue
        validated = validate_event_card(card)
        if require_entities and not validated.get("entities"):
            continue
        cards.append(validated)
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
    parser.add_argument("--allow-missing-entities", action="store_true")
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
        require_entities=not bool(args.allow_missing_entities),
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
        "require_entities": not bool(args.allow_missing_entities),
        "build_method": "deterministic_title_event_card_v1",
    }
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[EventCards] output={args.output} events={len(cards)}")
    print(f"[EventCards] manifest={args.manifest_output}")


if __name__ == "__main__":
    main()

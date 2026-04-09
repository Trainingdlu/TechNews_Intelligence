"""Search-news skill implementation and structured adapter."""

from __future__ import annotations

from services.db import get_conn, put_conn

from ..core.skill_contracts import SkillEnvelope, build_error_envelope
from .helpers import _clamp_int, _evidence_from_text_output
from .retrieval import _get_query_embedding
from .schemas import SearchNewsSkillInput


def search_news(query: str, days: int = 21) -> str:
    """Search related news using hybrid retrieval (semantic + keyword)."""
    print(f"\n[Tool] search_news: query={query}, days={days}")
    days = _clamp_int(days, 1, 365)
    limit = 5
    time_filter = f"{days} days"

    conn = get_conn()
    try:
        cur = conn.cursor()
        query_vec = _get_query_embedding(query)

        if query_vec:
            vec_str = "[" + ",".join(str(v) for v in query_vec) + "]"
            cur.execute(
                """
                WITH semantic AS (
                    SELECT
                        t.title, t.url, t.summary, t.sentiment, t.created_at,
                        1 - (e.embedding <=> %s::vector)
                            + 0.1 * EXP(-EXTRACT(EPOCH FROM (NOW() - t.created_at)) / 86400.0 / 21)
                            AS score
                    FROM tech_news t
                    JOIN news_embeddings e ON e.url = t.url
                    WHERE t.created_at > NOW() - %s::interval
                    ORDER BY e.embedding <=> %s::vector
                    LIMIT %s
                ),
                keyword AS (
                    SELECT title, url, summary, sentiment, created_at, 1.0 AS score
                    FROM tech_news
                    WHERE (title ILIKE %s OR summary ILIKE %s)
                      AND created_at > NOW() - %s::interval
                    LIMIT %s
                ),
                combined AS (
                    SELECT * FROM semantic
                    UNION ALL
                    SELECT * FROM keyword
                )
                SELECT DISTINCT ON (url)
                    title, url, summary, sentiment, created_at, score
                FROM combined
                ORDER BY url, score DESC
                """,
                (vec_str, time_filter, vec_str, limit, f"%{query}%", f"%{query}%", time_filter, limit),
            )
            rows = cur.fetchall()
            rows.sort(key=lambda row: row[5], reverse=True)
            rows = rows[:limit]
        else:
            cur.execute(
                """
                SELECT title, url, summary, sentiment, created_at, 1.0 AS score
                FROM tech_news
                WHERE (title ILIKE %s OR summary ILIKE %s)
                  AND created_at > NOW() - %s::interval
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (f"%{query}%", f"%{query}%", time_filter, limit),
            )
            rows = cur.fetchall()

        cur.close()
        if not rows:
            return f"No related news for '{query}' in last {days} days."

        max_score = max(row[5] for row in rows)
        note = ""
        if max_score < 0.5:
            note = "[Note] Relevance is weak; these are nearest matches.\n\n"

        out = []
        for title, url, summary, sentiment, pub_time, score in rows:
            out.append(
                f"Title: {title}\n"
                f"URL: {url}\n"
                f"Summary: {summary}\n"
                f"Sentiment: {sentiment}\n"
                f"Time: {pub_time.strftime('%Y-%m-%d %H:%M')}\n"
                f"Score: {score:.3f}"
            )
        return note + "\n---\n".join(out)
    except Exception as exc:
        print(f"[Error] search_news failed: {exc}")
        return f"search_news failed: {exc}"
    finally:
        put_conn(conn)


def search_news_skill(payload: SearchNewsSkillInput) -> SkillEnvelope:
    request = payload.model_dump(mode="python")
    try:
        raw_output = search_news(query=request["query"], days=int(request.get("days", 21)))
    except Exception as exc:
        return build_error_envelope(
            tool="search_news",
            request=request,
            error="search_news_execution_failed",
            diagnostics={"exception_type": type(exc).__name__, "exception_message": str(exc)},
        )

    if raw_output.startswith("No related news") or raw_output.startswith("search_news failed"):
        is_error = raw_output.startswith("search_news failed")
        return SkillEnvelope(
            tool="search_news",
            status="error" if is_error else "empty",
            request=request,
            data={"raw_output": raw_output},
            evidence=[],
            error=raw_output if is_error else None,
            diagnostics={"query": request["query"]},
        )

    evidence = _evidence_from_text_output(raw_output, max_items=5)
    return SkillEnvelope(
        tool="search_news",
        status="ok",
        request=request,
        data={"raw_output": raw_output, "result_count": len(evidence)},
        evidence=evidence,
        diagnostics={"query": request["query"]},
    )

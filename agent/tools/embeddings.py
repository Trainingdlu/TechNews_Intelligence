"""Embedding helpers shared by retrieval modules."""

from __future__ import annotations

import os

import requests


JINA_EMBED_URL = "https://api.jina.ai/v1/embeddings"
JINA_MODEL = "jina-embeddings-v3"


def get_query_embedding(query: str) -> list[float] | None:
    """Get query embedding via Jina API. Return None on failure."""
    jina_key = os.getenv("JINA_API_KEY", "")
    if not jina_key:
        print("[Error] JINA_API_KEY not set, skip vector search.")
        return None

    try:
        resp = requests.post(
            JINA_EMBED_URL,
            headers={
                "Authorization": f"Bearer {jina_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": JINA_MODEL,
                "task": "retrieval.query",
                "input": [query],
            },
            timeout=15,
        )
        resp.raise_for_status()
        emb = resp.json()["data"][0]["embedding"]
        return emb
    except Exception as exc:
        print(f"[Error] Embedding request failed, fallback to keyword search only: {exc}")
        return None

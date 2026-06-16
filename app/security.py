"""Security helpers for access approval links."""

from __future__ import annotations

import hashlib
import hmac
import html
import os
import time

API_BASE_URL = os.getenv("API_BASE_URL", "https://agentapi.trainingcqy.com")
APPROVE_LINK_SECRET = os.getenv("APPROVE_LINK_SECRET", os.getenv("SECRET_KEY", ""))
APPROVE_LINK_TTL_SEC = max(60, int(os.getenv("APPROVE_LINK_TTL_SEC", "86400")))


def build_approve_signature(record_id: int, tier: int, exp: int) -> str:
    secret = APPROVE_LINK_SECRET.strip().encode("utf-8")
    if not secret:
        raise RuntimeError("APPROVE_LINK_SECRET is empty")
    payload = f"{record_id}:{tier}:{exp}".encode("utf-8")
    return hmac.new(secret, payload, hashlib.sha256).hexdigest()


def build_signed_approve_url(record_id: int, tier: int) -> str | None:
    if not APPROVE_LINK_SECRET.strip():
        return None
    exp = int(time.time()) + APPROVE_LINK_TTL_SEC
    sig = build_approve_signature(record_id, tier, exp)
    return f"{API_BASE_URL}/approve/{record_id}?tier={tier}&exp={exp}&sig={sig}"


def is_valid_approve_signature(record_id: int, tier: int | None, exp: int | None, sig: str | None) -> bool:
    if exp is None or not sig:
        return False
    if tier is None:
        return False
    if not APPROVE_LINK_SECRET.strip():
        return False
    if exp < int(time.time()):
        return False
    try:
        expected = build_approve_signature(record_id, tier, exp)
    except Exception:
        return False
    return hmac.compare_digest(expected, sig)


def render_approve_confirmation_page(record_id: int, tier: int, exp: int, sig: str) -> str:
    safe_sig = html.escape(sig, quote=True)
    action = f"/approve/{record_id}?tier={tier}&exp={exp}&sig={safe_sig}"
    return (
        "<html><body style=\"font-family:sans-serif;max-width:560px;margin:32px auto;\">"
        "<h2>审批确认</h2>"
        "<p>请确认是否将该用户额度提升至升级配额。</p>"
        f"<form method=\"post\" action=\"{action}\">"
        "<button type=\"submit\" style=\"padding:8px 16px;cursor:pointer;\">确认批准</button>"
        "</form>"
        "</body></html>"
    )

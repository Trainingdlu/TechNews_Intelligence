from __future__ import annotations

from services import mail


def test_send_returns_false_for_empty_recipient(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setenv("SMTP_USER", "sender@example.com")
    monkeypatch.setenv("SMTP_PASS", "secret")

    assert mail._send("", "subject", "<p>body</p>") is False  # pylint: disable=protected-access


def test_send_quota_capped_to_admin_includes_user_and_quota(monkeypatch) -> None:  # noqa: ANN001
    calls: list[tuple[str, str, str]] = []
    monkeypatch.setattr(mail, "_send", lambda to, subject, body: calls.append((to, subject, body)) or True)

    assert mail.send_quota_capped_to_admin("admin@example.com", "user@example.com", 200) is True

    assert calls
    to, subject, body = calls[0]
    assert to == "admin@example.com"
    assert "user@example.com" in subject
    assert "user@example.com" in body
    assert "200" in body

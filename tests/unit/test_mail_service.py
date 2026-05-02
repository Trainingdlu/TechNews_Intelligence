from __future__ import annotations

from services import mail


def test_send_returns_false_for_empty_recipient(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setenv("SMTP_USER", "sender@example.com")
    monkeypatch.setenv("SMTP_PASS", "secret")

    assert mail._send("", "subject", "<p>body</p>") is False  # pylint: disable=protected-access


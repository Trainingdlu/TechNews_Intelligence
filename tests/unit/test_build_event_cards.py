from __future__ import annotations

from datetime import datetime, timezone

from eval import build_event_cards as mod


def test_extract_entities_uses_chinese_aliases() -> None:
    entities = mod._extract_entities("谷歌拟推开发者强制实名注册，安卓侧载面临限制", "")
    assert "Google" in entities
    assert "Android" in entities


def test_build_event_cards_filters_missing_entities_by_default(monkeypatch) -> None:
    now = datetime(2026, 5, 18, tzinfo=timezone.utc)

    def fake_rows(*, days: int, limit: int) -> list[dict]:
        return [
            {
                "title": "作者通过一个月与35名健身房陌生人交谈的实验克服社交焦虑",
                "summary": "文章记录一次社交实验。",
                "url": "https://example.com/social-experiment",
                "source": "Blog",
                "created_at": now,
                "points": 100,
            },
            {
                "title": "谷歌拟推开发者强制实名注册，安卓侧载面临限制",
                "summary": "谷歌计划调整 Android 开发者验证要求。",
                "url": "https://example.com/google-android-policy",
                "source": "Blog",
                "created_at": now,
                "points": 90,
            },
        ]

    monkeypatch.setattr(mod, "_fetch_news_rows", fake_rows)
    cards = mod.build_event_cards(days=30, limit=20, max_events=20, max_facts=2)
    assert len(cards) == 1
    assert cards[0]["entities"] == ["Google", "Android"]


def test_build_event_cards_can_allow_missing_entities_for_debug(monkeypatch) -> None:
    now = datetime(2026, 5, 18, tzinfo=timezone.utc)

    def fake_rows(*, days: int, limit: int) -> list[dict]:
        return [
            {
                "title": "作者通过一个月与35名健身房陌生人交谈的实验克服社交焦虑",
                "summary": "文章记录一次社交实验。",
                "url": "https://example.com/social-experiment",
                "source": "Blog",
                "created_at": now,
                "points": 100,
            }
        ]

    monkeypatch.setattr(mod, "_fetch_news_rows", fake_rows)
    cards = mod.build_event_cards(days=30, limit=20, max_events=20, max_facts=2, require_entities=False)
    assert len(cards) == 1
    assert cards[0]["entities"] == []

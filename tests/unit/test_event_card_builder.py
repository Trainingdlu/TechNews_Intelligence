from __future__ import annotations

from datetime import datetime, timezone

from eval.build_event_cards import _cluster_news_rows, _is_related_row, _related_score


def _row(title: str, url: str, *, summary: str = "", points: int = 10) -> dict:
    return {
        "title": title,
        "raw_title": title,
        "summary": summary,
        "url": url,
        "source": "test",
        "created_at": datetime(2026, 5, 1, tzinfo=timezone.utc),
        "points": points,
    }


def test_related_row_matches_specific_product_event() -> None:
    primary = _row(
        "Framework发布Laptop 13 Pro笔记本，搭载酷睿Ultra 3与LPCAMM2内存",
        "https://frame.work/laptop13pro",
    )
    candidate = _row(
        "Framework Laptop 13 Pro announced with LPCAMM2 memory",
        "https://boilingsteam.com/framework-laptop-13-pro-announced/",
    )
    assert _is_related_row(primary, candidate)
    assert _related_score(primary, candidate) >= 0.45


def test_broad_google_chrome_articles_do_not_merge_without_specific_anchor() -> None:
    primary = _row(
        "Google Chrome未经同意静默安装4GB AI模型引发隐私与环保争议",
        "https://www.thatprivacyguy.com/blog/chrome-silent-nano-install/",
    )
    candidate = _row(
        "Google brings Chrome AI to Android",
        "https://blog.google/products-and-platforms/products/chrome/bringing-chrome-ai-to-android/",
    )
    assert not _is_related_row(primary, candidate)


def test_cluster_news_rows_adds_related_urls_and_marks_them_used() -> None:
    rows = [
        _row(
            "Framework发布Laptop 13 Pro笔记本，搭载酷睿Ultra 3与LPCAMM2内存",
            "https://frame.work/laptop13pro",
            points=20,
        ),
        _row(
            "Framework Laptop 13 Pro announced with LPCAMM2 memory",
            "https://boilingsteam.com/framework-laptop-13-pro-announced/",
            points=12,
        ),
        _row(
            "Google brings Chrome AI to Android",
            "https://blog.google/products-and-platforms/products/chrome/bringing-chrome-ai-to-android/",
            points=18,
        ),
    ]
    groups = _cluster_news_rows(rows, max_related_per_event=5)
    assert any(len(group) == 2 and "Framework" in group[0]["title"] for group in groups)
    framework_urls = {
        row["url"]
        for group in groups
        if any("Framework" in row["title"] for row in group)
        for row in group
    }
    assert framework_urls == {
        "https://frame.work/laptop13pro",
        "https://boilingsteam.com/framework-laptop-13-pro-announced/",
    }

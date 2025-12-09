from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

import pandas as pd
import requests

from config import Settings


MACRO_KEYWORDS = [
    "Federal Reserve",
    "Fed",
    "FOMC",
    "interest rate",
    "CPI",
    "inflation",
    "PCE",
    "unemployment",
    "non-farm payrolls",
    "jobs report",
]


def _build_query_string(keywords: List[str]) -> str:
    return " OR ".join(f'"{kw}"' for kw in keywords)


def fetch_macro_news_last_n_months(settings: Settings, max_articles: int = 5000) -> pd.DataFrame:
    if not settings.newsapi_api_key:
        raise ValueError("NEWSAPI_API_KEY is not set")

    end_time = datetime.now(tz=timezone.utc)
    start_time = end_time - timedelta(days=30 * settings.backtest_months)

    query = _build_query_string(MACRO_KEYWORDS)
    page = 1
    page_size = 100
    articles = []

    while True:
        params = {
            "q": query,
            "from": start_time.isoformat(),
            "to": end_time.isoformat(),
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": page_size,
            "page": page,
            "apiKey": settings.newsapi_api_key,
        }
        response = requests.get(settings.newsapi_base_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        fetched = data.get("articles", [])
        for art in fetched:
            articles.append(
                {
                    "source": (art.get("source") or {}).get("name"),
                    "title": art.get("title"),
                    "description": art.get("description"),
                    "content": art.get("content"),
                    "published_at": pd.to_datetime(art.get("publishedAt"), utc=True),
                    "url": art.get("url"),
                }
            )
        total_results = data.get("totalResults", len(articles))
        if len(articles) >= max_articles:
            break
        expected_pages = math.ceil(total_results / page_size)
        if page >= expected_pages:
            break
        page += 1

    df = pd.DataFrame(articles)
    if df.empty:
        return pd.DataFrame(columns=["id", "source", "title", "description", "content", "published_at", "url"])

    df.insert(0, "id", range(1, len(df) + 1))
    df.sort_values("published_at", inplace=True)

    filename = f"news_macro_{end_time.strftime('%Y%m%d')}_last_{settings.backtest_months}m.parquet"
    path = settings.raw_news_dir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


__all__ = ["fetch_macro_news_last_n_months", "MACRO_KEYWORDS"]

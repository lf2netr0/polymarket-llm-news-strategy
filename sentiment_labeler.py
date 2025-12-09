from __future__ import annotations

import json
from typing import Any, Dict

import pandas as pd
from openai import OpenAI

from config import Settings

PROMPT_TEMPLATE = (
    "You are a macro and Federal Reserve policy analyst."
    " Given the following news article title, description, and content,"
    " determine whether it discusses Fed policy, interest rates, inflation, jobs, or is unrelated."
    " Respond with a strict JSON object containing keys:"
    " topic (Fed_rate | inflation | jobs | other),"
    " relevance (1 if clearly about Fed/macro policy/inflation/jobs, else 0),"
    " sentiment (-1 bearish/hawkish, 0 neutral, 1 bullish/dovish for risk assets)."
)


def _parse_response(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"topic": "other", "relevance": 0, "sentiment": 0}


def label_news_articles(df_news: pd.DataFrame, settings: Settings, batch_size: int = 20) -> pd.DataFrame:
    if df_news.empty:
        return df_news.assign(topic="other", relevance=0, sentiment=0)

    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is not set")

    client = OpenAI(api_key=settings.openai_api_key)
    labeled_rows = []

    for start in range(0, len(df_news), batch_size):
        batch = df_news.iloc[start : start + batch_size]
        messages = []
        for _, row in batch.iterrows():
            content = (row.get("content") or "")
            if content and len(content) > 2000:
                content = content[:2000]
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"{PROMPT_TEMPLATE}\n"
                        f"Title: {row.get('title','')}\n"
                        f"Description: {row.get('description','')}\n"
                        f"Content: {content}"
                    ),
                }
            )

        for message, (_, row) in zip(messages, batch.iterrows()):
            try:
                response = client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[
                        {"role": "system", "content": PROMPT_TEMPLATE},
                        message,
                    ],
                    temperature=0,
                )
                choice = response.choices[0].message.content
                parsed = _parse_response(choice)
            except Exception:
                parsed = {"topic": "other", "relevance": 0, "sentiment": 0}
            labeled_rows.append({
                **row.to_dict(),
                "topic": parsed.get("topic", "other"),
                "relevance": int(parsed.get("relevance", 0) or 0),
                "sentiment": int(parsed.get("sentiment", 0) or 0),
            })

    df_labeled = pd.DataFrame(labeled_rows)
    df_labeled.sort_values("published_at", inplace=True)

    filename = f"news_labeled_{pd.Timestamp.utcnow().strftime('%Y%m%d')}.parquet"
    path = settings.features_dir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    df_labeled.to_parquet(path, index=False)
    return df_labeled


__all__ = ["label_news_articles"]

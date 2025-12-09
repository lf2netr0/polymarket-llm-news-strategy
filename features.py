from __future__ import annotations

from datetime import datetime, timedelta, timezone
import pandas as pd

from config import Settings


def build_macro_sentiment_timeseries(df_labeled_news: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    if df_labeled_news.empty:
        return pd.DataFrame(columns=["ts", "sentiment_score", "bullish_ratio", "bearish_ratio", "article_count"])

    df = df_labeled_news.copy()
    df = df[df["relevance"] == 1]
    if df.empty:
        return pd.DataFrame(columns=["ts", "sentiment_score", "bullish_ratio", "bearish_ratio", "article_count"])

    start_time = datetime.now(tz=timezone.utc) - timedelta(days=30 * settings.backtest_months)
    end_time = datetime.now(tz=timezone.utc)

    df.set_index("published_at", inplace=True)
    df = df.sort_index()

    resampled = (
        df[["sentiment"]]
        .assign(article=1)
        .resample("1h")
        .agg({"sentiment": "sum", "article": "sum"})
        .rename(columns={"article": "article_count", "sentiment": "sentiment_sum"})
    )

    resampled = resampled.reindex(pd.date_range(start_time, end_time, freq="1h", tz=timezone.utc), fill_value=0)

    window = f"{settings.sentiment_window_hours}h"
    rolling_count = resampled["article_count"].rolling(window=window, min_periods=1).sum()
    rolling_sum = resampled["sentiment_sum"].rolling(window=window, min_periods=1).sum()

    bullish = (
        df.assign(bullish=lambda x: (x["sentiment"] == 1).astype(int))
        [["bullish"]]
        .resample("1h")
        .sum()
        .reindex(resampled.index, fill_value=0)
        .rolling(window=window, min_periods=1)
        .sum()["bullish"]
    )

    bearish = (
        df.assign(bearish=lambda x: (x["sentiment"] == -1).astype(int))
        [["bearish"]]
        .resample("1h")
        .sum()
        .reindex(resampled.index, fill_value=0)
        .rolling(window=window, min_periods=1)
        .sum()["bearish"]
    )

    sentiment_score = rolling_sum / rolling_count.replace({0: 1})
    bullish_ratio = bullish / rolling_count.replace({0: 1})
    bearish_ratio = bearish / rolling_count.replace({0: 1})

    features = pd.DataFrame(
        {
            "ts": resampled.index,
            "sentiment_score": sentiment_score,
            "bullish_ratio": bullish_ratio,
            "bearish_ratio": bearish_ratio,
            "article_count": rolling_count,
        }
    )

    path = settings.features_dir / "macro_news_features_1h.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(path, index=False)
    return features


__all__ = ["build_macro_sentiment_timeseries"]

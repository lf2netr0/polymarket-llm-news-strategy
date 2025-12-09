from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pandas as pd

from backtest import Backtester, MarketConfig
from config import get_settings
from features import build_macro_sentiment_timeseries
from news_pipeline import fetch_macro_news_last_n_months
from polymarket_client import PolymarketClient
from sentiment_labeler import label_news_articles


def load_markets_config(path: Path) -> List[MarketConfig]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return [MarketConfig.from_dict(item) for item in data]


def main() -> None:
    settings = get_settings()
    settings.ensure_directories()

    markets_path = Path("markets_macro.json")
    if not markets_path.exists():
        markets_path = Path("data/markets_macro.json")
    if not markets_path.exists():
        raise FileNotFoundError("markets_macro.json not found in project root or data directory")

    markets = load_markets_config(markets_path)
    token_ids = {m.token_id for m in markets}

    poly_client = PolymarketClient(settings)
    for token_id in token_ids:
        price_path = settings.raw_prices_dir / f"{token_id}_prices.parquet"
        if price_path.exists():
            continue
        df_prices = poly_client.get_price_history_with_interval(token_id, interval="1h", months=settings.backtest_months)
        poly_client.save_price_history(token_id, df_prices)

    news_files = sorted(settings.raw_news_dir.glob("news_macro_*_last_*m.parquet"))
    if news_files:
        df_news = pd.read_parquet(news_files[-1])
    else:
        df_news = fetch_macro_news_last_n_months(settings)

    labeled_files = sorted(settings.features_dir.glob("news_labeled_*.parquet"))
    if labeled_files:
        df_labeled = pd.read_parquet(labeled_files[-1])
    else:
        df_labeled = label_news_articles(df_news, settings)

    sentiment_path = settings.features_dir / "macro_news_features_1h.parquet"
    if sentiment_path.exists():
        df_sentiment = pd.read_parquet(sentiment_path)
    else:
        df_sentiment = build_macro_sentiment_timeseries(df_labeled, settings)

    backtester = Backtester(settings)
    trades = backtester.run_for_all_markets(markets, df_sentiment)

    results_dir = settings.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    trades_path = results_dir / "trades_macro_backtest"
    trades.to_parquet(trades_path.with_suffix(".parquet"), index=False)
    trades.to_csv(trades_path.with_suffix(".csv"), index=False)

    summary = Backtester.summarize_trades(trades)
    print("Backtest Summary")
    print("-----------------")
    print(f"Markets: {len(markets)}")
    print(f"Trades: {summary['num_trades']}")
    print(f"Total PnL: {summary['total_pnl']:.2f}")
    print(f"Average PnL per trade: {summary['avg_pnl']:.2f}")
    print(f"Win rate: {summary['win_rate']*100:.1f}%")
    print(f"Max drawdown: {summary['max_drawdown']:.2f}")

    try:
        import matplotlib.pyplot as plt

        equity_curve = trades["pnl"].cumsum()
        plt.figure(figsize=(10, 4))
        plt.plot(equity_curve.index, equity_curve.values, label="Equity Curve")
        plt.xlabel("Trade")
        plt.ylabel("PnL (USD)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(results_dir / "equity_curve.png")
    except Exception:
        pass


if __name__ == "__main__":
    main()

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from config import Settings


@dataclass
class MarketConfig:
    token_id: str
    market_id: str
    question: str
    created_at: datetime
    resolve_time: datetime
    outcome: str

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "MarketConfig":
        return MarketConfig(
            token_id=data["token_id"],
            market_id=data.get("market_id") or data.get("condition_id") or data["token_id"],
            question=data["question"],
            created_at=pd.to_datetime(data["created_at"], utc=True),
            resolve_time=pd.to_datetime(data["resolve_time"], utc=True),
            outcome=data["outcome"].upper(),
        )


class Backtester:
    def __init__(self, settings: Settings):
        self.settings = settings

    def run_for_market(
        self,
        market_config: MarketConfig,
        df_prices: pd.DataFrame,
        df_sentiment: pd.DataFrame,
    ) -> Dict[str, Any]:
        df_prices = df_prices.copy()
        df_prices["ts"] = pd.to_datetime(df_prices["ts"], utc=True)
        df_prices = df_prices[(df_prices["ts"] >= market_config.created_at) & (df_prices["ts"] <= market_config.resolve_time)]
        df_prices.set_index("ts", inplace=True)

        df_sentiment = df_sentiment.copy()
        df_sentiment["ts"] = pd.to_datetime(df_sentiment["ts"], utc=True)
        df_sentiment.set_index("ts", inplace=True)

        position: Optional[Dict[str, Any]] = None

        for ts, price_row in df_prices.iterrows():
            hours_to_resolve = (market_config.resolve_time - ts).total_seconds() / 3600
            if position is None:
                if hours_to_resolve > self.settings.max_hours_to_resolve_for_entry:
                    continue
                sentiment_row = df_sentiment.loc[ts] if ts in df_sentiment.index else None
                sentiment_score = sentiment_row["sentiment_score"] if sentiment_row is not None else 0.0
                article_count = sentiment_row["article_count"] if sentiment_row is not None else 0
                price = price_row["price"]
                if 0.25 <= price <= 0.75:
                    if sentiment_score >= self.settings.sentiment_buy_threshold:
                        shares = self.settings.trade_size_usd / price
                        position = {
                            "side": "YES",
                            "shares": shares,
                            "entry_price": price,
                            "entry_ts": ts,
                            "article_count_at_entry": article_count,
                            "sentiment_score_at_entry": sentiment_score,
                        }
                    elif sentiment_score <= self.settings.sentiment_sell_threshold:
                        no_price = 1.0 - price
                        shares = self.settings.trade_size_usd / no_price if no_price > 0 else 0
                        if shares > 0:
                            position = {
                                "side": "NO",
                                "shares": shares,
                                "entry_price": no_price,
                                "entry_ts": ts,
                                "article_count_at_entry": article_count,
                                "sentiment_score_at_entry": sentiment_score,
                            }

        pnl = 0.0
        side = None
        entry_ts = None
        entry_price = None
        sentiment_score_at_entry = None
        article_count_at_entry = None

        if position is not None:
            side = position["side"]
            entry_ts = position["entry_ts"]
            entry_price = position["entry_price"]
            sentiment_score_at_entry = position.get("sentiment_score_at_entry")
            article_count_at_entry = position.get("article_count_at_entry")
            if side == "YES":
                payoff = position["shares"] * (1.0 if market_config.outcome == "YES" else 0.0)
            else:
                payoff = position["shares"] * (1.0 if market_config.outcome == "NO" else 0.0)
            pnl = payoff - self.settings.trade_size_usd

        return {
            "market_id": market_config.market_id,
            "token_id": market_config.token_id,
            "question": market_config.question,
            "side": side,
            "entry_ts": entry_ts,
            "resolve_time": market_config.resolve_time,
            "entry_price": entry_price,
            "outcome": market_config.outcome,
            "pnl": pnl,
            "article_count_at_entry": article_count_at_entry,
            "sentiment_score_at_entry": sentiment_score_at_entry,
        }

    def run_for_all_markets(self, markets_config: List[MarketConfig], df_sentiment: pd.DataFrame) -> pd.DataFrame:
        results = []
        for market in markets_config:
            price_path = self.settings.raw_prices_dir / f"{market.token_id}_prices.parquet"
            if not price_path.exists():
                raise FileNotFoundError(f"Missing price history for {market.token_id}: {price_path}")
            df_prices = pd.read_parquet(price_path)
            result = self.run_for_market(market, df_prices, df_sentiment)
            results.append(result)
        return pd.DataFrame(results)

    @staticmethod
    def summarize_trades(df_trades: pd.DataFrame) -> Dict[str, Any]:
        if df_trades.empty:
            return {
                "num_trades": 0,
                "win_rate": 0.0,
                "avg_pnl": 0.0,
                "total_pnl": 0.0,
                "max_drawdown": 0.0,
            }

        num_trades = (df_trades["side"].notnull()).sum()
        wins = (df_trades["pnl"] > 0).sum()
        win_rate = wins / num_trades if num_trades else 0.0
        avg_pnl = df_trades["pnl"].mean()
        total_pnl = df_trades["pnl"].sum()

        equity_curve = df_trades["pnl"].cumsum()
        running_max = equity_curve.cummax()
        drawdown = equity_curve - running_max
        max_drawdown = drawdown.min()

        return {
            "num_trades": int(num_trades),
            "win_rate": float(win_rate),
            "avg_pnl": float(avg_pnl),
            "total_pnl": float(total_pnl),
            "max_drawdown": float(max_drawdown),
        }


__all__ = ["Backtester", "MarketConfig"]

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from config import Settings


class PolymarketClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.base_url = settings.polymarket_base_url.rstrip("/")

    def _get(self, path: str, params: dict) -> dict:
        url = f"{self.base_url}{path}"
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()

    def get_price_history_with_interval(
        self, token_id: str, interval: str = "1h", months: Optional[int] = None
    ) -> pd.DataFrame:
        months = months or self.settings.backtest_months
        end_ts = datetime.now(tz=timezone.utc)
        start_ts = end_ts - timedelta(days=30 * months)
        params = {
            "market": token_id,
            "interval": interval,
            "startTs": int(start_ts.timestamp()),
            "endTs": int(end_ts.timestamp()),
        }
        data = self._get("/prices-history", params=params)
        history = data.get("history", [])
        df = pd.DataFrame(history)
        if df.empty:
            return pd.DataFrame(columns=["ts", "price"])
        df.rename(columns={"t": "ts", "p": "price"}, inplace=True)
        df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True)
        df.sort_values("ts", inplace=True)
        return df

    def get_price_history_with_timestamps(
        self, token_id: str, start_ts: int, end_ts: int, fidelity_minutes: Optional[int] = None
    ) -> pd.DataFrame:
        params = {
            "market": token_id,
            "startTs": start_ts,
            "endTs": end_ts,
        }
        if fidelity_minutes:
            params["fidelity"] = fidelity_minutes
        data = self._get("/prices-history", params=params)
        history = data.get("history", [])
        df = pd.DataFrame(history)
        if df.empty:
            return pd.DataFrame(columns=["ts", "price"])
        df.rename(columns={"t": "ts", "p": "price"}, inplace=True)
        df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True)
        df.sort_values("ts", inplace=True)
        return df

    def save_price_history(self, token_id: str, df: pd.DataFrame) -> Path:
        path = self.settings.raw_prices_dir / f"{token_id}_prices.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        return path


__all__ = ["PolymarketClient"]

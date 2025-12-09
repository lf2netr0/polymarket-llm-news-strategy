from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import os


@dataclass
class Settings:
    polymarket_base_url: str = "https://clob.polymarket.com"
    newsapi_base_url: str = "https://newsapi.org/v2/everything"
    newsapi_api_key: str = field(default_factory=lambda: os.getenv("NEWSAPI_API_KEY", ""))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    backtest_months: int = 6
    sentiment_window_hours: int = 6
    sentiment_buy_threshold: float = 0.3
    sentiment_sell_threshold: float = -0.3
    max_hours_to_resolve_for_entry: int = 72
    trade_size_usd: float = 100.0
    raw_prices_dir: Path = field(default_factory=lambda: Path("data/raw_prices_dir"))
    raw_news_dir: Path = field(default_factory=lambda: Path("data/raw_news_dir"))
    features_dir: Path = field(default_factory=lambda: Path("data/features_dir"))
    results_dir: Path = field(default_factory=lambda: Path("data/results_dir"))

    def ensure_directories(self) -> None:
        for directory in [
            self.raw_prices_dir,
            self.raw_news_dir,
            self.features_dir,
            self.results_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings

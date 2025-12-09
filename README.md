# Polymarket Macro Sentiment Backtester

This repository provides a small, self-contained Python 3.11 project that backtests a macro/Fed news-driven Polymarket strategy over the last six months using hourly price bars, LLM-labeled news sentiment, and a simple event-driven backtester.

## Features
- Fetch hourly Polymarket price history via the CLOB `prices-history` endpoint.
- Collect macro/Fed-related news from NewsAPI.
- Label news with an OpenAI chat model to produce macro sentiment signals.
- Build hourly sentiment features with a rolling window.
- Backtest a fixed-size binary options strategy that opens positions close to resolution based on sentiment.

## Prerequisites
Export the required API keys:

```bash
export NEWSAPI_API_KEY="<your_newsapi_key>"
export OPENAI_API_KEY="<your_openai_key>"
```

Install dependencies (Python 3.11 recommended):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Market configuration
Create a `markets_macro.json` file in the project root (or `data/markets_macro.json`). Each entry should follow:

```json
[
  {
    "token_id": "<clob_token_id>",
    "market_id": "<market_identifier>",
    "question": "Will the Fed cut rates by July?",
    "created_at": "2024-01-01T00:00:00Z",
    "resolve_time": "2024-06-30T23:00:00Z",
    "outcome": "YES"
  }
]
```

The backtester will fetch hourly prices for each `token_id` and assumes the market resolves to the provided `outcome`.

## Running the backtest

```bash
python run_backtest.py
```

The script orchestrates data fetching, labeling, feature creation, and backtesting. Results are stored in `data/results_dir/` as Parquet and CSV files, with an optional `equity_curve.png` if matplotlib is available.

## End-to-end execution flow (with required parameters)
1. **Load settings** from `config.py` via `get_settings()`. Key parameters (defaults in parentheses):
   - `BACKTEST_MONTHS` (6), `SENTIMENT_WINDOW_HOURS` (6), `MAX_HOURS_TO_RESOLVE_FOR_ENTRY` (72)
   - Sentiment thresholds: `SENTIMENT_BUY_THRESHOLD` (0.3), `SENTIMENT_SELL_THRESHOLD` (-0.3)
   - `TRADE_SIZE_USD` (100.0)
   - API endpoints: `POLYMARKET_BASE_URL` (`https://clob.polymarket.com`), `NEWSAPI_BASE_URL` (`https://newsapi.org/v2/everything`)
   - Data directories (relative to repo): `data/raw_prices_dir`, `data/raw_news_dir`, `data/features_dir`, `data/results_dir`

2. **Prepare market configs** in `markets_macro.json` with fields:
   - `token_id` (CLOB token ID), `market_id`, `question`, `created_at` (ISO-8601), `resolve_time` (ISO-8601), `outcome` (`YES`/`NO`).

3. **Fetch price history** per unique `token_id` using `PolymarketClient.get_price_history_with_interval(...)` (interval `1h`, last `BACKTEST_MONTHS` months). Cached to `data/raw_prices_dir/<token_id>_prices.parquet`.

4. **Fetch macro/Fed news** for the last `BACKTEST_MONTHS` months via `fetch_macro_news_last_n_months(settings)` using the configured `NEWSAPI_API_KEY`. Cached to `data/raw_news_dir/news_macro_<date>_last_6m.parquet`.

5. **LLM sentiment labeling** with `label_news_articles(...)`, using `OPENAI_API_KEY` and the `gpt-4.1-mini` model. Saves labeled articles to `data/features_dir/news_labeled_<date>.parquet`.

6. **Build hourly sentiment features** using `build_macro_sentiment_timeseries(...)`, applying a `SENTIMENT_WINDOW_HOURS` rolling window. Outputs `data/features_dir/macro_news_features_1h.parquet` (columns: `ts`, `sentiment_score`, `bullish_ratio`, `bearish_ratio`, `article_count`).

7. **Run backtest** with `Backtester.run_for_all_markets(...)` (wired inside `run_backtest.py`), joining hourly prices and sentiment:
   - Entry only when `hours_to_resolve <= MAX_HOURS_TO_RESOLVE_FOR_ENTRY` and `0.25 <= price <= 0.75`.
   - Buy YES if `sentiment_score >= SENTIMENT_BUY_THRESHOLD`; Buy NO if `sentiment_score <= SENTIMENT_SELL_THRESHOLD`; position size `TRADE_SIZE_USD`.
   - Hold to `resolve_time`; compute PnL from market outcome.
   - Saves trades to `data/results_dir/trades_macro_backtest.parquet` and `.csv`, plus optional equity curve plot.

8. **Inspect outputs** via the printed summary (markets, trades, PnL, win rate, max drawdown) or by loading the saved Parquet/CSV files.

## Strategy outline
- Universe: supplied Polymarket macro/Fed markets (via `markets_macro.json`).
- Timeframe: last 6 months, hourly bars.
- Entry: only in the final `MAX_HOURS_TO_RESOLVE_FOR_ENTRY` hours before resolution.
  - Buy YES when sentiment_score ≥ `SENTIMENT_BUY_THRESHOLD` and 0.25 ≤ price ≤ 0.75.
  - Buy NO when sentiment_score ≤ `SENTIMENT_SELL_THRESHOLD` and 0.25 ≤ price ≤ 0.75.
- Position size: fixed `TRADE_SIZE_USD` (default 100 USD) per trade.
- Exit: hold until resolution; payoff follows binary market outcome.

## Notes
- Data and feature files are cached under `data/` to avoid redundant downloads or labeling.
- Sentiment labeling uses the `gpt-4.1-mini` chat model; update the model name if desired.
- Ensure you respect Polymarket and NewsAPI rate limits when running frequent backtests.

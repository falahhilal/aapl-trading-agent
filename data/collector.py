# ============================================================
# data/collector.py
# Responsible for all data downloading:
#   - AAPL historical price data via yfinance
#   - Financial news headlines via Finnhub API
# Nothing is processed here — raw data only, saved to CSV.
# ============================================================

import os
import time
import pandas as pd
import yfinance as yf
import finnhub
from datetime import datetime, timedelta

# Import all settings from central config
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ============================================================
# PART 1 — PRICE DATA
# ============================================================

def download_price_data(force_redownload=False):
    """
    Downloads daily OHLCV price data for AAPL from Yahoo Finance.
    Saves to raw_data/aapl_raw.csv.

    Args:
        force_redownload: if True, re-downloads even if file exists.
                          if False, loads from disk if already downloaded.

    Returns:
        pd.DataFrame with columns: Open, High, Low, Close, Volume
    """

    # If already downloaded and not forcing, just load from disk
    # This saves time — you never need to re-download unless data is stale
    if os.path.exists(config.PRICE_RAW_PATH) and not force_redownload:
        print(f"[collector] Price data found on disk. Loading from {config.PRICE_RAW_PATH}")
        df = pd.read_csv(config.PRICE_RAW_PATH, index_col="Date", parse_dates=True)
        print(f"[collector] Loaded {len(df)} rows from {df.index[0].date()} to {df.index[-1].date()}")
        return df

    # Download fresh from Yahoo Finance
    print(f"[collector] Downloading {config.TICKER} price data from Yahoo Finance...")
    print(f"[collector] Date range: {config.START_DATE} → {config.END_DATE}")

    df = yf.download(
        tickers     = config.TICKER,
        start       = config.START_DATE,
        end         = config.END_DATE,
        interval    = "1d",      # one row per trading day
        auto_adjust = True,      # adjust for splits and dividends
        progress    = True
    )

    # yfinance sometimes returns MultiIndex columns like (Close, AAPL)
    # flatten to plain names: Close, Open, High, Low, Volume
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
        print("[collector] MultiIndex columns flattened.")

    # Name the index
    df.index.name = "Date"

    # Basic validation
    assert len(df) > 0,        "Download returned empty DataFrame"
    assert "Close" in df.columns, "Close column missing"
    assert "Volume" in df.columns, "Volume column missing"

    print(f"[collector] Downloaded {len(df)} trading days.")
    print(f"[collector] Columns: {list(df.columns)}")

    # Save to disk
    df.to_csv(config.PRICE_RAW_PATH)
    print(f"[collector] Saved to {config.PRICE_RAW_PATH}")

    return df


# ============================================================
# PART 2 — NEWS DATA
# ============================================================

def download_news_data(force_redownload=False):
    """
    Downloads Apple-related financial news headlines using Finnhub API.
    Finnhub free tier allows 60 API calls/minute and news per ticker.
    Saves to raw_data/aapl_news_raw.csv.

    Args:
        force_redownload: same logic as price data above.

    Returns:
        pd.DataFrame with columns: date, headline, source, sentiment_raw
    """

    if os.path.exists(config.NEWS_RAW_PATH) and not force_redownload:
        print(f"[collector] News data found on disk. Loading from {config.NEWS_RAW_PATH}")
        df = pd.read_csv(config.NEWS_RAW_PATH, parse_dates=["date"])
        print(f"[collector] Loaded {len(df)} news articles.")
        return df

    # Check API key is set
    if config.FINNHUB_API_KEY == "your_api_key_here":
        print("[collector] WARNING: Finnhub API key not set in config.py")
        print("[collector] Get a free key at https://finnhub.io")
        print("[collector] Falling back to empty news DataFrame.")
        return _empty_news_dataframe()

    print(f"[collector] Downloading news headlines from Finnhub...")

    # Initialise Finnhub client
    client = finnhub.Client(api_key=config.FINNHUB_API_KEY)

    all_articles = []

    # Finnhub returns news in date ranges
    # We request one month at a time to stay within free tier limits
    start = datetime.strptime(config.START_DATE, "%Y-%m-%d")
    end   = datetime.strptime(config.END_DATE,   "%Y-%m-%d")

    current = start
    while current < end:
        # One month window
        window_end = min(current + timedelta(days=30), end)

        from_str = current.strftime("%Y-%m-%d")
        to_str   = window_end.strftime("%Y-%m-%d")

        try:
            news = client.company_news(
                config.TICKER,
                _from = from_str,
                to    = to_str
            )

            for article in news:
                # Convert Unix timestamp to date string
                article_date = datetime.fromtimestamp(
                    article.get("datetime", 0)
                ).strftime("%Y-%m-%d")

                all_articles.append({
                    "date"     : article_date,
                    "headline" : article.get("headline", ""),
                    "source"   : article.get("source",   ""),
                    "summary"  : article.get("summary",  ""),
                    "url"      : article.get("url",      ""),
                })

            print(f"[collector]   {from_str} → {to_str}: {len(news)} articles")

        except Exception as e:
            print(f"[collector]   {from_str} → {to_str}: ERROR — {e}")

        # Respect Finnhub rate limit (60 calls/min on free tier)
        time.sleep(1)
        current = window_end + timedelta(days=1)

    if not all_articles:
        print("[collector] No articles retrieved. Check your API key.")
        return _empty_news_dataframe()

    df = pd.DataFrame(all_articles)
    df["date"] = pd.to_datetime(df["date"])

    # Remove duplicates (same headline on same day from different sources)
    df = df.drop_duplicates(subset=["date", "headline"])

    # Remove rows with empty headlines
    df = df[df["headline"].str.strip() != ""]

    df = df.sort_values("date").reset_index(drop=True)

    print(f"[collector] Total articles collected: {len(df)}")
    print(f"[collector] Date range: {df['date'].min().date()} → {df['date'].max().date()}")

    df.to_csv(config.NEWS_RAW_PATH, index=False)
    print(f"[collector] Saved to {config.NEWS_RAW_PATH}")

    return df


def _empty_news_dataframe():
    """
    Returns an empty DataFrame with correct columns.
    Used as fallback when API key is missing.
    """
    return pd.DataFrame(columns=["date", "headline", "source", "summary", "url"])


# ============================================================
# PART 3 — RUN BOTH COLLECTORS
# ============================================================

def collect_all(force_redownload=False):
    """
    Runs both collectors in sequence.
    Call this from main.py or directly.
    """
    print("\n" + "="*55)
    print("PHASE 1 — DATA COLLECTION")
    print("="*55)

    price_df = download_price_data(force_redownload=force_redownload)
    print()
    news_df  = download_news_data(force_redownload=force_redownload)

    print("\n[collector] Collection complete.")
    print(f"[collector] Price data shape : {price_df.shape}")
    print(f"[collector] News data shape  : {news_df.shape}")

    return price_df, news_df


# ============================================================
# TEST — run this file directly to test collection
# ============================================================
if __name__ == "__main__":
    price_df, news_df = collect_all()

    print("\n--- Price Data Sample ---")
    print(price_df.tail(3))

    print("\n--- News Data Sample ---")
    if len(news_df) > 0:
        print(news_df.head(3))
    else:
        print("No news data (API key not set yet — that is fine for now).")
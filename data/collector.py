# ============================================================
# data/collector.py
# Downloads AAPL price data and VIX data from Yahoo Finance.
# ============================================================

import os
import pandas as pd
import yfinance as yf

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def download_price_data(force_redownload=False):
    """
    Downloads daily OHLCV price data for AAPL from Yahoo Finance.
    Saves to raw_data/aapl_raw.csv.
    """
    if os.path.exists(config.PRICE_RAW_PATH) and not force_redownload:
        print(f"[collector] Price data found on disk. Loading...")
        df = pd.read_csv(config.PRICE_RAW_PATH, index_col="Date", parse_dates=True)
        print(f"[collector] Loaded {len(df)} rows from {df.index[0].date()} to {df.index[-1].date()}")
        return df

    print(f"[collector] Downloading {config.TICKER} price data from Yahoo Finance...")
    print(f"[collector] Date range: {config.START_DATE} → {config.END_DATE}")

    df = yf.download(
        tickers     = config.TICKER,
        start       = config.START_DATE,
        end         = config.END_DATE,
        interval    = "1d",
        auto_adjust = True,
        progress    = True
    )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
        print("[collector] MultiIndex columns flattened.")

    df.index.name = "Date"

    assert len(df) > 0,           "Download returned empty DataFrame"
    assert "Close" in df.columns, "Close column missing"
    assert "Volume" in df.columns,"Volume column missing"

    print(f"[collector] Downloaded {len(df)} trading days.")
    print(f"[collector] Columns: {list(df.columns)}")

    df.to_csv(config.PRICE_RAW_PATH)
    print(f"[collector] Saved to {config.PRICE_RAW_PATH}")

    return df


def download_vix_data(force_redownload=False):
    """
    Downloads VIX (CBOE Volatility Index) data.
    VIX measures overall market fear/uncertainty.
    High VIX = fearful market, low VIX = calm market.
    """
    VIX_PATH = os.path.join(config.RAW_DIR, "vix_raw.csv")

    if os.path.exists(VIX_PATH) and not force_redownload:
        print(f"[collector] VIX data found on disk. Loading...")
        df = pd.read_csv(VIX_PATH, index_col="Date", parse_dates=True)
        print(f"[collector] Loaded {len(df)} rows of VIX data.")
        return df

    print(f"[collector] Downloading VIX data...")

    df = yf.download(
        tickers     = "^VIX",
        start       = config.START_DATE,
        end         = config.END_DATE,
        interval    = "1d",
        auto_adjust = True,
        progress    = False
    )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df.index.name = "Date"
    df = df[["Close"]].rename(columns={"Close": "vix_close"})

    df.to_csv(VIX_PATH)
    print(f"[collector] VIX downloaded: {len(df)} rows. Saved to {VIX_PATH}")

    return df


def collect_all(force_redownload=False):
    """
    Runs all collectors in sequence.
    Call this from main.py or directly.
    """
    print("\n" + "="*55)
    print("PHASE 1 — DATA COLLECTION")
    print("="*55)

    price_df = download_price_data(force_redownload=force_redownload)
    print()
    vix_df   = download_vix_data(force_redownload=force_redownload)

    print("\n[collector] Collection complete.")
    print(f"[collector] Price data shape : {price_df.shape}")
    print(f"[collector] VIX data shape   : {vix_df.shape}")

    return price_df, vix_df


if __name__ == "__main__":
    price_df, vix_df = collect_all()

    print("\n--- Price Data Sample ---")
    print(price_df.tail(3))

    print("\n--- VIX Data Sample ---")
    print(vix_df.tail(3))
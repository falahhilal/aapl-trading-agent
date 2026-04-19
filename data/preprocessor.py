# ============================================================
# data/preprocessor.py
# Takes raw price + news CSVs and produces one clean merged
# DataFrame ready for feature engineering in Phase 2.
#
# What it does:
#   1. Loads and validates price data
#   2. Adds basic derived columns (returns, ranges)
#   3. Aggregates multiple news headlines per day into one row
#   4. Merges price + news on date
#   5. Handles missing values
#   6. Saves aapl_features.csv (input to Phase 2)
# ============================================================

import os
import numpy  as np
import pandas as pd

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def load_price_data():
    """
    Loads raw price CSV and runs basic validation checks.
    Returns clean DataFrame with DatetimeIndex.
    """
    print("[preprocessor] Loading price data...")

    df = pd.read_csv(
        config.PRICE_RAW_PATH,
        index_col  = "Date",
        parse_dates= True
    )

    # Validation checks
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"

    # Check for missing values
    missing = df[required_cols].isnull().sum()
    if missing.sum() > 0:
        print(f"[preprocessor] Warning — missing values found:\n{missing}")
        df[required_cols] = df[required_cols].ffill()  # forward fill gaps
        print("[preprocessor] Missing values forward-filled.")

    # Check for duplicate dates
    dupes = df.index.duplicated().sum()
    if dupes > 0:
        print(f"[preprocessor] Warning — {dupes} duplicate dates. Keeping last.")
        df = df[~df.index.duplicated(keep="last")]

    # Sort chronologically
    df = df.sort_index()

    print(f"[preprocessor] Price data loaded: {len(df)} rows, "
          f"{df.index[0].date()} → {df.index[-1].date()}")

    return df


def add_price_features(df):
    """
    Adds fundamental price-derived columns that everything
    else in the pipeline depends on.

    These are NOT the ML features (those are in features/technical.py).
    These are basic building blocks added at the data layer.
    """
    print("[preprocessor] Adding basic price features...")

    # Daily return: % change from yesterday's close to today's close
    # This is the core signal — everything revolves around this
    df["daily_return"]    = df["Close"].pct_change()

    # Log return: mathematically cleaner for statistics
    # log(today/yesterday) — symmetric, additive across time
    df["log_return"]      = np.log(df["Close"] / df["Close"].shift(1))

    # Intraday range: High minus Low
    # Measures how volatile the market was within a single day
    df["high_low_range"]  = df["High"] - df["Low"]

    # Open-to-close movement: did the day close higher or lower than open?
    # Positive = bullish day, negative = bearish day
    df["close_open_gap"]  = df["Close"] - df["Open"]

    # Previous day's close — used as a simple lag feature
    df["prev_close"]      = df["Close"].shift(1)

    # Drop first row — always NaN because there's no "yesterday"
    df = df.dropna(subset=["daily_return"])

    print(f"[preprocessor] Basic features added. Shape: {df.shape}")

    return df

def merge_vix(df):
    """
    Merges VIX data into the main price DataFrame.
    Adds two columns:
      vix_close  — raw VIX closing value
      vix_change — daily % change in VIX
    High vix_close (>30) = high market fear
    High vix_change       = fear is increasing fast
    """
    VIX_PATH = os.path.join(config.RAW_DIR, "vix_raw.csv")

    if not os.path.exists(VIX_PATH):
        print("[preprocessor] VIX file not found. Skipping.")
        df["vix_close"]  = 20.0   # neutral fallback value
        df["vix_change"] = 0.0
        return df

    print("[preprocessor] Merging VIX data...")

    vix = pd.read_csv(VIX_PATH, index_col="Date", parse_dates=True)
    vix.index = pd.to_datetime(vix.index).normalize()
    vix["vix_change"] = vix["vix_close"].pct_change()

    df = df.join(vix[["vix_close", "vix_change"]], how="left")

    # Forward fill any missing VIX days (holidays etc.)
    df["vix_close"]  = df["vix_close"].ffill().fillna(20.0)
    df["vix_change"] = df["vix_change"].ffill().fillna(0.0)

    print(f"[preprocessor] VIX merged. Range: {df['vix_close'].min():.1f} – {df['vix_close'].max():.1f}")
    print(f"[preprocessor]   VIX > 30 (high fear) days: {(df['vix_close'] > 30).sum()}")

    return df

def generate_labels(df):
    """
    Creates the target variable (what the ML model learns to predict).

    For each day, we look at TOMORROW's return and label it:
      UP      if next day return >  +0.5%
      DOWN    if next day return <  -0.5%
      NEUTRAL if next day return is between -0.5% and +0.5%

    CRITICAL: we shift by -1 (look forward) to get next day's return.
    This means the last row always gets NaN label and is dropped.
    The features for day T use only data available at end of day T.
    The label for day T is what happens on day T+1.
    This is the correct setup — no data leakage.
    """
    print("[preprocessor] Generating labels...")

    # Next day's return — shift(-1) looks one row ahead
    next_day_return = df["daily_return"].shift(-1)

    # Apply thresholds from config
    conditions = [
        next_day_return >  config.UP_THRESHOLD,
        next_day_return <  config.DOWN_THRESHOLD,
    ]
    choices = ["UP", "DOWN"]

    df["label"]            = np.select(conditions, choices, default="NEUTRAL")
    df["next_day_return"]  = next_day_return   # keep for reference

    # Drop the last row — it has no next day to label
    df = df.dropna(subset=["next_day_return"])

    # Label distribution
    counts = df["label"].value_counts()
    total  = len(df)
    print(f"[preprocessor] Label distribution:")
    for label, count in counts.items():
        print(f"[preprocessor]   {label:<8} : {count:>4} ({100*count/total:.1f}%)")

    return df



def save_preprocessed(df):
    """Saves the merged, labelled DataFrame to disk."""
    df.to_csv(config.FEATURES_PATH)
    print(f"\n[preprocessor] Saved to {config.FEATURES_PATH}")
    print(f"[preprocessor] Shape  : {df.shape}")
    print(f"[preprocessor] Columns: {list(df.columns)}")


def run_preprocessing():
    print("\n" + "="*55)
    print("PHASE 1 — PREPROCESSING")
    print("="*55)

    df = load_price_data()
    df = add_price_features(df)
    df = merge_vix(df)             
    df = generate_labels(df)

    save_preprocessed(df)

    print("\n[preprocessor] Preprocessing complete. Ready for Phase 2.")
    return df

if __name__ == "__main__":
    df = run_preprocessing()

    print("\n--- Final DataFrame Sample ---")
    print(df[["Close", "daily_return", "log_return",
              "high_low_range", "label"]].tail(10))

    print("\n--- Data Types ---")
    print(df.dtypes)
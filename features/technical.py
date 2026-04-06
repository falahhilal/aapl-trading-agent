# Features computed:
#   1. RSI (Relative Strength Index)
#   2. MACD + Signal Line
#   3. Bollinger Bands + BB Position
#   4. Rolling mean and std (5, 10, 20 day windows)
#   5. Lagged close prices (1, 2, 3, 5 days)
#   6. Volume features
#   7. Cyclic date encodings (day of week, month)
#   8. Momentum features

import os
import numpy  as np
import pandas as pd

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# INDIVIDUAL INDICATOR FUNCTIONS
# Each function takes a DataFrame, adds columns, returns it.

def add_rsi(df):
    """
    Relative Strength Index (RSI) — 14 day period.

    Measures whether a stock is overbought or oversold.
    RSI > 70 = overbought (price may fall soon)
    RSI < 30 = oversold  (price may rise soon)
    RSI < 40 is our BUY signal threshold in the agent.

    Formula:
        RSI = 100 - (100 / (1 + RS))
        RS  = average gain over 14 days / average loss over 14 days
    """
    delta = df["Close"].diff()

    # Separate gains and losses
    gain  = delta.clip(lower=0)           # positive changes only
    loss  = (-delta.clip(upper=0))        # negative changes, made positive

    # Rolling average gain and loss over RSI_PERIOD days
    avg_gain = gain.rolling(window=config.RSI_PERIOD, min_periods=config.RSI_PERIOD).mean()
    avg_loss = loss.rolling(window=config.RSI_PERIOD, min_periods=config.RSI_PERIOD).mean()

    rs          = avg_gain / avg_loss
    df["rsi"]   = 100 - (100 / (1 + rs))

    print(f"  [technical] RSI added. Range: {df['rsi'].min():.1f} – {df['rsi'].max():.1f}")
    return df


def add_macd(df):
    """
    MACD — Moving Average Convergence Divergence.

    Captures momentum and trend direction.
    When MACD crosses above signal line = bullish signal.
    When MACD crosses below signal line = bearish signal.

    Formula:
        MACD         = 12-day EMA - 26-day EMA
        Signal line  = 9-day EMA of MACD
        Histogram    = MACD - Signal (shows divergence strength)
    """
    ema_fast = df["Close"].ewm(span=config.MACD_FAST, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=config.MACD_SLOW, adjust=False).mean()

    df["macd"]           = ema_fast - ema_slow
    df["macd_signal"]    = df["macd"].ewm(span=config.MACD_SIGNAL, adjust=False).mean()
    df["macd_histogram"] = df["macd"] - df["macd_signal"]

    print(f"  [technical] MACD added.")
    return df


def add_bollinger_bands(df):
    """
    Bollinger Bands — 20 day period, 2 standard deviations.

    Measures price relative to recent volatility.
    Price near upper band = potentially overbought.
    Price near lower band = potentially oversold.

    bb_position is the most useful feature:
        0.0 = price is at the lower band
        0.5 = price is exactly at the middle (20-day MA)
        1.0 = price is at the upper band
        >1  = price is above the upper band (very overbought)
        <0  = price is below the lower band (very oversold)
    """
    rolling_mean = df["Close"].rolling(window=config.BOLLINGER_PERIOD).mean()
    rolling_std  = df["Close"].rolling(window=config.BOLLINGER_PERIOD).std()

    df["bb_upper"]    = rolling_mean + (config.BOLLINGER_STD * rolling_std)
    df["bb_lower"]    = rolling_mean - (config.BOLLINGER_STD * rolling_std)
    df["bb_middle"]   = rolling_mean
    df["bb_width"]    = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]  # normalized width

    # Position: where is price within the bands?
    band_range        = df["bb_upper"] - df["bb_lower"]
    df["bb_position"] = (df["Close"] - df["bb_lower"]) / band_range

    print(f"  [technical] Bollinger Bands added.")
    return df


def add_rolling_features(df):
    """
    Rolling mean and standard deviation over multiple windows.

    These capture the trend and volatility at different timescales:
    - 5-day  = short term (one trading week)
    - 10-day = medium term (two trading weeks)
    - 20-day = longer term (one trading month)

    Price relative to rolling mean tells you if price is
    currently above or below its recent average — a simple
    but powerful signal.
    """
    for window in config.ROLLING_WINDOWS:
        # Rolling mean of closing price
        df[f"rolling_mean_{window}"] = df["Close"].rolling(window=window).mean()

        # Rolling std of daily returns (volatility measure)
        df[f"rolling_std_{window}"]  = df["daily_return"].rolling(window=window).std()

        # Price relative to its rolling mean
        # Positive = price above average (momentum)
        # Negative = price below average (potential reversal)
        df[f"price_vs_mean_{window}"] = (df["Close"] - df[f"rolling_mean_{window}"]) / df[f"rolling_mean_{window}"]

    print(f"  [technical] Rolling features added for windows: {config.ROLLING_WINDOWS}")
    return df


def add_lag_features(df):
    """
    Lagged closing prices and returns.

    The model needs to see what happened in previous days to
    learn patterns. These are the most direct way to give the
    model that historical context.

    lag_close_1  = yesterday's closing price
    lag_close_5  = closing price 5 days ago
    lag_return_1 = yesterday's daily return
    etc.
    """
    for lag in config.LAG_DAYS:
        # Lagged closing price
        df[f"lag_close_{lag}"]  = df["Close"].shift(lag)

        # Lagged daily return — more informative than raw price
        df[f"lag_return_{lag}"] = df["daily_return"].shift(lag)

    print(f"  [technical] Lag features added for lags: {config.LAG_DAYS}")
    return df


def add_volume_features(df):
    """
    Volume-based features.

    Volume tells you the conviction behind a price move.
    A big price move on high volume = strong signal.
    A big price move on low volume = weak signal, may reverse.

    volume_ratio: today's volume vs 20-day average
        > 1.5 = unusually high volume (strong conviction)
        < 0.5 = unusually low volume (weak conviction)
    """
    # 20-day average volume
    df["volume_ma20"]   = df["Volume"].rolling(window=20).mean()

    # Today's volume relative to 20-day average
    df["volume_ratio"]  = df["Volume"] / df["volume_ma20"]

    # Log volume — compresses large outliers
    df["log_volume"]    = np.log(df["Volume"].replace(0, np.nan))

    # Volume change day over day
    df["volume_change"] = df["Volume"].pct_change()

    print(f"  [technical] Volume features added.")
    return df


def add_cyclic_date_features(df):
    """
    Cyclic encoding of day-of-week and month.

    The model needs to know about weekly and monthly patterns
    without treating them as linear numbers.

    Problem with raw encoding:
        Monday=0, Friday=4 — model thinks Friday is "more" than Monday
        December=12, January=1 — model thinks there's a big jump

    Solution — encode as sine and cosine:
        This creates a smooth circle where Monday and Friday
        are close to each other, and December and January
        are close to each other. The model can learn
        "Fridays before earnings tend to be volatile" etc.
    """
    df["day_of_week"] = df.index.dayofweek      # 0=Monday, 4=Friday
    df["month"]       = df.index.month          # 1=January, 12=December

    # Cyclic encoding for day of week (5 trading days)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 5)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 5)

    # Cyclic encoding for month (12 months)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Drop raw integer columns — we only need the encoded versions
    df = df.drop(columns=["day_of_week", "month"])

    print(f"  [technical] Cyclic date features added.")
    return df


def add_momentum_features(df):
    """
    Momentum — how much has the price moved over N days?

    Momentum captures medium-term trends:
    - Positive momentum = price has been rising
    - Negative momentum = price has been falling

    Rate of change (ROC) is momentum as a percentage.
    """
    # 5-day momentum: close today vs close 5 days ago
    df["momentum_5"]  = df["Close"] - df["Close"].shift(5)

    # 10-day momentum
    df["momentum_10"] = df["Close"] - df["Close"].shift(10)

    # Rate of change (%) over 5 and 10 days
    df["roc_5"]       = df["Close"].pct_change(periods=5)
    df["roc_10"]      = df["Close"].pct_change(periods=10)

    print(f"  [technical] Momentum features added.")
    return df

# MAIN FUNCTION — runs all indicators in sequence

def build_features(df=None):
    """
    Loads preprocessed data and applies all feature engineering.
    Saves the result to aapl_features.csv (overwrites Phase 1 version
    with the full feature set added on top).

    Returns the fully featured DataFrame.
    """
    print("\n" + "="*55)
    print("PHASE 2 — FEATURE ENGINEERING")
    print("="*55)

    # Load from disk if not passed in
    if df is None:
        print("[technical] Loading preprocessed data...")
        df = pd.read_csv(
            config.FEATURES_PATH,
            index_col   = "Date",
            parse_dates = True
        )
        print(f"[technical] Loaded {len(df)} rows.")

    print("\n[technical] Computing indicators...")

    # Apply all feature functions in order
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_rolling_features(df)
    df = add_lag_features(df)
    df = add_volume_features(df)
    df = add_cyclic_date_features(df)
    df = add_momentum_features(df)

    # Drop rows with NaN values introduced by rolling windows
    # The longest window is 26 days (MACD slow EMA) so we lose
    # roughly the first 26 rows — acceptable for a 2500 row dataset
    rows_before = len(df)
    df = df.dropna()
    rows_after  = len(df)
    rows_dropped = rows_before - rows_after

    print(f"\n[technical] Rows dropped due to NaN (rolling windows): {rows_dropped}")
    print(f"[technical] Final dataset size: {rows_after} rows")

    # Feature summary
    label_col     = ["label", "next_day_return"]
    price_cols    = ["Open", "High", "Low", "Close", "Volume"]
    feature_cols  = [c for c in df.columns if c not in label_col + price_cols]

    print(f"[technical] Total features: {len(feature_cols)}")
    print(f"[technical] Features: {feature_cols}")

    # Save updated features file
    df.to_csv(config.FEATURES_PATH)
    print(f"\n[technical] Saved to {config.FEATURES_PATH}")

    return df, feature_cols


# TEST — run directly to verify

if __name__ == "__main__":
    df, feature_cols = build_features()

    print("\n--- Sample (last 5 rows, key features) ---")
    print(df[["Close", "rsi", "macd", "bb_position",
              "volume_ratio", "label"]].tail())

    print("\n--- Feature Statistics ---")
    print(df[feature_cols].describe().round(3))

    # Quick sanity checks
    print("\n--- Sanity Checks ---")
    print(f"RSI range      : {df['rsi'].min():.1f} – {df['rsi'].max():.1f}  (should be 0–100)")
    print(f"BB position    : {df['bb_position'].min():.2f} – {df['bb_position'].max():.2f}  (0=lower band, 1=upper band)")
    print(f"Volume ratio   : {df['volume_ratio'].min():.2f} – {df['volume_ratio'].max():.2f}  (1.0 = average volume)")
    print(f"Any NaN left   : {df.isnull().sum().sum()}  (should be 0)")
    print(f"Label counts   :\n{df['label'].value_counts()}")
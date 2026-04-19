import os

# PATHS
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RAW_DIR     = os.path.join(BASE_DIR, "raw_data")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
MODEL_DIR   = os.path.join(BASE_DIR, "models")

# Specific file paths
PRICE_RAW_PATH    = os.path.join(RAW_DIR, "aapl_raw.csv")
NEWS_RAW_PATH     = os.path.join(RAW_DIR, "aapl_news_raw.csv")
FEATURES_PATH     = os.path.join(RAW_DIR, "aapl_features.csv")
FINAL_DATA_PATH   = os.path.join(RAW_DIR, "aapl_final.csv")
MODEL_SAVE_PATH   = os.path.join(MODEL_DIR, "best_model.pkl")

# TICKER & DATE RANGE
TICKER      = "AAPL"
START_DATE  = "2015-01-01"
END_DATE    = "2024-12-31"


# LABEL THRESHOLDS
UP_THRESHOLD   =  0.005
DOWN_THRESHOLD = -0.005

# TRAIN / TEST SPLIT
TEST_SPLIT    = 0.20

# AGENT HEURISTIC THRESHOLDS
BUY_CONFIDENCE    = 0.40   # was 0.65
SELL_CONFIDENCE   = 0.40   # was 0.60
RSI_OVERSOLD      = 55     # was 40
RSI_OVERBOUGHT    = 70     # was 60
SENTIMENT_BUY_MIN =  0.1
SENTIMENT_SELL_MAX= -0.1

# BACKTESTING
INITIAL_CAPITAL   = 10_000
TRANSACTION_COST  = 0.001

# FINNHUB API KEY
FINNHUB_API_KEY = "d79nlo9r01qqpmhhdsm0d79nlo9r01qqpmhhdsmg"

# FEATURE ENGINEERING SETTINGS 
RSI_PERIOD        = 14
MACD_FAST         = 12
MACD_SLOW         = 26
MACD_SIGNAL       = 9
BOLLINGER_PERIOD  = 20
BOLLINGER_STD     = 2
ROLLING_WINDOWS   = [5, 10, 20]
LAG_DAYS          = [1, 2, 3, 5]

# WALK-FORWARD VALIDATION 
TRAIN_YEARS = 4
TEST_YEARS  = 1

# Create all directories on import
for directory in [RAW_DIR, OUTPUT_DIR, MODEL_DIR]:
    os.makedirs(directory, exist_ok=True)

if __name__ == "__main__":
    print("CONFIG CHECK")
    print("=" * 40)
    print(f"Base directory  : {BASE_DIR}")
    print(f"Raw data dir    : {RAW_DIR}")
    print(f"Outputs dir     : {OUTPUT_DIR}")
    print(f"Models dir      : {MODEL_DIR}")
    print(f"Ticker          : {TICKER}")
    print(f"Date range      : {START_DATE} → {END_DATE}")
    print(f"Initial capital : ${INITIAL_CAPITAL:,}")
    print()
    for directory in [RAW_DIR, OUTPUT_DIR, MODEL_DIR]:
        exists = "✓ exists" if os.path.exists(directory) else "✗ missing"
        print(f"  {exists}  {directory}")
    print()
    print("Config loaded successfully.")
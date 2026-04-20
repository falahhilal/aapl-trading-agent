# predict.py
# Lightweight inference only — no training code here.
# Loads the saved model and returns a BUY/SELL/HOLD decision
# for any date that exists in the feature dataset.
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import pandas as pd

import config
from agent.heuristic import TradingAgent


def load_model_payload():
    """
    Loads the saved model bundle from disk.
    Returns model, scaler, and feature_cols.
    """
    if not os.path.exists(config.MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"No saved model at {config.MODEL_SAVE_PATH}. "
            f"Run train.py first."
        )

    with open(config.MODEL_SAVE_PATH, "rb") as f:
        payload = pickle.load(f)

    return payload["model"], payload["scaler"], payload["feature_cols"]


def predict_for_date(date_str: str) -> dict:
    """
    Returns a prediction for a single date.

    Args:
        date_str: date string in YYYY-MM-DD format

    Returns:
        dict with date, action, prediction, confidence, rsi, vix
    """
    # Load features
    df = pd.read_csv(
        config.FEATURES_PATH,
        index_col   = "Date",
        parse_dates = True
    )

    # Normalize index for matching
    df.index = df.index.normalize()

    # Find the date
    target = pd.Timestamp(date_str)
    if target not in df.index:
        available = df.index.strftime("%Y-%m-%d").tolist()
        return {
            "error"    : f"Date {date_str} not found in dataset.",
            "hint"     : f"Available range: {available[0]} → {available[-1]}"
        }

    row = df.loc[target]

    # Load model
    model, scaler, feature_cols = load_model_payload()

    # Build agent (stateless for single prediction)
    agent = TradingAgent(model, scaler, feature_cols)

    # Get decision
    action = agent.decide(target, row)
    log    = agent.get_trade_log().iloc[0]   # single entry

    return {
        "date"      : date_str,
        "action"    : action,
        "prediction": log["prediction"],
        "confidence": log["confidence"],
        "rsi"       : log["rsi"],
        "vix"       : log["vix"],
        "reason"    : log["reason"],
    }


def predict_range(start_date: str, end_date: str) -> list:
    """
    Returns predictions for every trading day in a date range.

    Args:
        start_date: YYYY-MM-DD
        end_date:   YYYY-MM-DD

    Returns:
        list of dicts, one per trading day in range
    """
    df = pd.read_csv(
        config.FEATURES_PATH,
        index_col   = "Date",
        parse_dates = True
    )
    df.index = df.index.normalize()

    mask = (df.index >= pd.Timestamp(start_date)) & \
           (df.index <= pd.Timestamp(end_date))
    subset = df[mask]

    if len(subset) == 0:
        return [{"error": f"No data found between {start_date} and {end_date}"}]

    model, scaler, feature_cols = load_model_payload()
    agent = TradingAgent(model, scaler, feature_cols)

    results = []
    for date, row in subset.iterrows():
        action = agent.decide(date, row)

    log_df = agent.get_trade_log()

    for entry in log_df.to_dict(orient="records"):
        results.append({
            "date"      : str(entry["date"])[:10],
            "action"    : entry["action"],
            "prediction": entry["prediction"],
            "confidence": entry["confidence"],
            "rsi"       : entry["rsi"],
            "vix"       : entry["vix"],
            "reason"    : entry["reason"],
        })

    return results


# ============================================================
# TEST — run directly to verify
# ============================================================

if __name__ == "__main__":
    print("\n--- Single date prediction ---")
    result = predict_for_date("2024-01-05")
    for k, v in result.items():
        print(f"  {k:12}: {v}")

    print("\n--- Range prediction (first 5 days of 2024) ---")
    results = predict_range("2024-01-01", "2024-01-10")
    for r in results:
        print(f"  {r['date']} | {r['action']:4} | conf={r['confidence']} | {r['reason'][:60]}")
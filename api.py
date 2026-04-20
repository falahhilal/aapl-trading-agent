# ============================================================
# api.py
# FastAPI backend for AAPL Trading Agent
# ============================================================

from fastapi import FastAPI
import pandas as pd
import os

import config
from predict import predict_for_date, predict_range

app = FastAPI(
    title       = "AAPL Trading Agent API",
    description = "BUY/SELL/HOLD predictions from a trained MLP classifier",
    version     = "1.0.0"
)


@app.get("/")
def root():
    return {
        "status"     : "running",
        "model"      : "MLPClassifier",
        "description": "AAPL Trading Agent — returns BUY/SELL/HOLD decisions"
    }


@app.get("/predict/{date}")
def predict(date: str):
    """
    Returns BUY/SELL/HOLD for a single date.
    Date format: YYYY-MM-DD
    Example: /predict/2024-01-05
    """
    return predict_for_date(date)


@app.get("/predict-range")
def predict_range_endpoint(start: str, end: str):
    """
    Returns predictions for every trading day in a date range.
    Example: /predict-range?start=2024-01-01&end=2024-01-10
    """
    return predict_range(start, end)


@app.get("/backtest")
def backtest():
    """
    Returns latest backtest metrics from saved results.
    """
    results_path = os.path.join(config.OUTPUT_DIR, "backtest_results.csv")
    trades_path  = os.path.join(config.OUTPUT_DIR, "trades.csv")

    if not os.path.exists(results_path):
        return {"error": "backtest_results.csv not found. Run agent/backtester.py first."}

    results      = pd.read_csv(results_path, index_col="Date")
    final_agent  = results["agent_portfolio"].iloc[-1]
    final_bah    = results["bah_portfolio"].iloc[-1]
    initial      = config.INITIAL_CAPITAL

    n_trades = 0
    if os.path.exists(trades_path):
        trades   = pd.read_csv(trades_path)
        n_trades = len(trades)
        win_rate = round(trades["profitable"].mean() * 100, 1) if n_trades > 0 else 0
    else:
        win_rate = 0

    return {
        "agent": {
            "final_value"  : round(final_agent, 2),
            "total_return" : round((final_agent - initial) / initial * 100, 2),
            "total_trades" : n_trades,
            "win_rate"     : win_rate,
        },
        "buy_and_hold": {
            "final_value"  : round(final_bah, 2),
            "total_return" : round((final_bah - initial) / initial * 100, 2),
        }
    }


@app.get("/trades")
def trades():
    """
    Returns all completed trades as JSON.
    """
    trades_path = os.path.join(config.OUTPUT_DIR, "trades.csv")

    if not os.path.exists(trades_path):
        return {"error": "trades.csv not found. Run agent/backtester.py first."}

    df = pd.read_csv(trades_path)
    return df.to_dict(orient="records")
# app.py
# Streamlit frontend for AAPL Trading Agent
# Connects to the live FastAPI backend on Render

import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, date, timedelta
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

# CONFIGURATION

# Render API URL 
API_URL = "https://aapl-trading-agent.onrender.com"

# Page config
st.set_page_config(
    page_title = "AAPL Trading Agent",
    page_icon  = "📈",
    layout     = "wide"
)

# HELPER FUNCTIONS

def get_live_prediction():
    """
    Fetches real market data for today and computes a live prediction.
    Does not call the API — runs inference locally using saved model.
    This is because the API only knows about historical dates.
    """
    try:
        import pickle
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        import config
        from predict import predict_for_date
        from features.technical import (
            add_rsi, add_macd, add_bollinger_bands,
            add_rolling_features, add_lag_features,
            add_volume_features, add_cyclic_date_features,
            add_momentum_features
        )
        from data.preprocessor import add_price_features, merge_vix

        # Download last 60 days of AAPL + VIX for feature computation
        end   = datetime.today()
        start = end - timedelta(days=120)

        aapl = yf.download("AAPL", start=start, end=end,
                           auto_adjust=True, progress=False)
        vix  = yf.download("^VIX", start=start, end=end,
                           auto_adjust=True, progress=False)

        if isinstance(aapl.columns, pd.MultiIndex):
            aapl.columns = [col[0] for col in aapl.columns]
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = [col[0] for col in vix.columns]

        aapl.index.name = "Date"
        vix.index.name  = "Date"

        # Add price features
        aapl = add_price_features(aapl)

        # Merge VIX manually
        vix_series = vix["Close"].rename("vix_close")
        vix_change = vix_series.pct_change().rename("vix_change")
        aapl = aapl.join(vix_series).join(vix_change)
        aapl["vix_close"]  = aapl["vix_close"].ffill().fillna(20.0)
        aapl["vix_change"] = aapl["vix_change"].ffill().fillna(0.0)

        # Add all technical features
        aapl = add_rsi(aapl)
        aapl = add_macd(aapl)
        aapl = add_bollinger_bands(aapl)
        aapl = add_rolling_features(aapl)
        aapl = add_lag_features(aapl)
        aapl = add_volume_features(aapl)
        aapl = add_cyclic_date_features(aapl)
        aapl = add_momentum_features(aapl)
        aapl = aapl.dropna()

        if len(aapl) == 0:
            return None, "Not enough data to compute features"

        # Load model
        with open(config.MODEL_SAVE_PATH, "rb") as f:
            payload = pickle.load(f)

        model        = payload["model"]
        scaler       = payload["scaler"]
        feature_cols = payload["feature_cols"]

        # Get latest row
        latest_date = aapl.index[-1]
        latest_row  = aapl.iloc[-1]

        # Check all features exist
        missing = [c for c in feature_cols if c not in latest_row.index]
        if missing:
            return None, f"Missing features: {missing}"

        from agent.heuristic import TradingAgent
        agent  = TradingAgent(model, scaler, feature_cols)
        action = agent.decide(latest_date, latest_row)
        log    = agent.get_trade_log().iloc[0]

        return {
            "date"       : str(latest_date)[:10],
            "action"     : action,
            "prediction" : log["prediction"],
            "confidence" : log["confidence"],
            "rsi"        : log["rsi"],
            "vix"        : log["vix"],
            "close_price": round(float(latest_row["Close"]), 2),
            "reason"     : log["reason"],
        }, None

    except Exception as e:
        return None, str(e)


def call_api(endpoint):
    """Calls the deployed Render API."""
    try:
        response = requests.get(f"{API_URL}{endpoint}", timeout=30)
        return response.json(), None
    except Exception as e:
        return None, str(e)


def action_color(action):
    if action == "BUY":
        return "🟢"
    elif action == "SELL":
        return "🔴"
    else:
        return "🟡"


# PAGE HEADER

st.title("📈 AAPL Trading Agent")
st.markdown("An AI-powered BUY/SELL/HOLD decision system for Apple Inc. stock")
st.divider()

# SECTION 1 — LIVE PREDICTION=

st.header("🔴 Live Prediction")
st.markdown("Fetches today's real market data and runs the agent right now.")

col1, col2 = st.columns([1, 2])

with col1:
    if st.button("Get Today's Prediction", type="primary", use_container_width=True):
        with st.spinner("Fetching live market data..."):
            result, error = get_live_prediction()

        if error:
            st.error(f"Error: {error}")
        elif result:
            action = result["action"]
            emoji  = action_color(action)

            st.markdown(f"### {emoji} {action}")
            st.metric("Date",        result["date"])
            st.metric("AAPL Close",  f"${result['close_price']}")
            st.metric("Confidence",  f"{result['confidence']*100:.1f}%")
            st.metric("RSI",         f"{result['rsi']:.1f}")
            st.metric("VIX",         f"{result['vix']:.1f}")
            st.caption(f"Reason: {result['reason']}")

with col2:
    st.info("""
    **How the live prediction works:**

    1. Downloads last 120 days of AAPL + VIX from Yahoo Finance
    2. Computes all 32 technical features in real time
    3. Runs the trained MLP classifier
    4. Applies heuristic rules (confidence, RSI, VIX filters)
    5. Returns BUY / SELL / HOLD with reasoning

    This is a real prediction based on today's actual market data.
    Not financial advice — for educational purposes only.
    """)

st.divider()

# SECTION 2 — HISTORICAL LOOKUP

st.header("🔍 Historical Date Lookup")
st.markdown("Look up the agent's decision for any date in the dataset (2015–2024).")

col1, col2 = st.columns([1, 2])

with col1:
    selected_date = st.date_input(
        "Select a date",
        value       = date(2024, 1, 5),
        min_value   = date(2015, 1, 1),
        max_value   = date(2024, 12, 31)
    )

    if st.button("Look Up", use_container_width=True):
        date_str = selected_date.strftime("%Y-%m-%d")
        with st.spinner(f"Looking up {date_str}..."):
            result, error = call_api(f"/predict/{date_str}")

        if error:
            st.error(f"API error: {error}")
        elif "error" in result:
            st.warning(result["error"])
            if "hint" in result:
                st.caption(result["hint"])
        else:
            action = result["action"]
            emoji  = action_color(action)
            st.markdown(f"### {emoji} {action}")
            st.metric("Confidence", f"{result['confidence']*100:.1f}%")
            st.metric("RSI",        f"{result['rsi']:.1f}")
            st.metric("VIX",        f"{result['vix']:.1f}")
            st.caption(f"Reason: {result['reason']}")

with col2:
    st.info("""
    **Note:** Weekends and market holidays will return an error
    since no trading occurs on those days.
    Try dates like 2024-01-05, 2023-08-04, or 2024-11-08
    which were actual BUY signals.
    """)

st.divider()

# SECTION 3 — PERFORMANCE METRICS

st.header("📊 Backtest Performance")

with st.spinner("Loading performance data..."):
    metrics_data, error = call_api("/backtest")
    trades_data,  error2 = call_api("/trades")

if metrics_data and "agent" in metrics_data:
    agent = metrics_data["agent"]
    bah   = metrics_data["buy_and_hold"]

    # Metric cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Agent Total Return",     f"+{agent['total_return']}%")
    col2.metric("Buy-and-Hold Return",    f"+{bah['total_return']}%")
    col3.metric("Win Rate",               f"{agent['win_rate']}%")
    col4.metric("Total Trades",           agent['total_trades'])

    col1, col2 = st.columns(2)
    col1.metric("Agent Final Value",      f"${agent['final_value']:,.2f}")
    col2.metric("Buy-and-Hold Final",     f"${bah['final_value']:,.2f}")

st.divider()

# SECTION 4 — EQUITY CURVE

st.header("📈 Equity Curve")

try:
    import os
    results_path = os.path.join("outputs", "backtest_results.csv")
    if os.path.exists(results_path):
        results = pd.read_csv(results_path, index_col="Date", parse_dates=True)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(results.index, results["agent_portfolio"],
                color="#2196F3", linewidth=1.5, label="Agent")
        ax.plot(results.index, results["bah_portfolio"],
                color="#FF9800", linewidth=1.5,
                linestyle="--", label="Buy and Hold")
        ax.axhline(10000, color="gray", linestyle=":",
                   linewidth=1, alpha=0.7, label="Starting Capital")
        ax.fill_between(results.index,
                        results["agent_portfolio"],
                        results["bah_portfolio"],
                        where  = results["agent_portfolio"] >= results["bah_portfolio"],
                        alpha  = 0.15, color="#4CAF50")
        ax.fill_between(results.index,
                        results["agent_portfolio"],
                        results["bah_portfolio"],
                        where  = results["agent_portfolio"] < results["bah_portfolio"],
                        alpha  = 0.15, color="#F44336")
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value ($)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.xticks(rotation=45)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"${x:,.0f}")
        )
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("backtest_results.csv not found locally.")
except Exception as e:
    st.error(f"Could not load equity curve: {e}")

st.divider()

# SECTION 5 — TRADE HISTORY

st.header("📋 Trade History")

if trades_data and len(trades_data) > 0:
    trades_df = pd.DataFrame(trades_data)

    # Color profitable trades
    def highlight_profitable(row):
        color = "#e8f5e9" if row["profitable"] else "#ffebee"
        return [f"background-color: {color}"] * len(row)

    trades_df["pnl_pct"] = trades_df["pnl_pct"].apply(
        lambda x: f"+{x:.2f}%" if x > 0 else f"{x:.2f}%"
    )
    trades_df["pnl"] = trades_df["pnl"].apply(
        lambda x: f"+${x:.2f}" if x > 0 else f"-${abs(x):.2f}"
    )

    display_cols = ["entry_date", "exit_date", "entry_price",
                    "exit_price", "pnl", "pnl_pct", "profitable"]
    st.dataframe(
        trades_df[display_cols].style.apply(highlight_profitable, axis=1),
        use_container_width=True
    )

    # P&L bar chart
    trades_raw = pd.DataFrame(trades_data)
    fig, ax    = plt.subplots(figsize=(12, 4))
    colors     = ["#4CAF50" if p else "#F44336"
                  for p in trades_raw["profitable"]]
    ax.bar(range(len(trades_raw)), trades_raw["pnl_pct"],
           color=colors, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Trade #")
    ax.set_ylabel("P&L (%)")
    ax.set_title("Individual Trade P&L (%)")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    st.pyplot(fig)

st.divider()

# SECTION 6 — ABOUT

st.header("ℹ️ About")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **What this is:**
    An algorithmic trading simulation that uses machine learning
    to decide whether to BUY, SELL, or HOLD Apple stock each day.

    **How it works:**
    1. Downloads 10 years of AAPL price data + VIX
    2. Engineers 32 technical features (RSI, MACD, Bollinger Bands, etc.)
    3. Trains an MLP classifier to predict next-day price direction
    4. Applies heuristic rules for rational decision making
    5. Simulates trading with $10,000 virtual capital

    **Important:** This is not financial advice.
    Built for educational purposes as an AI course project.
    """)

with col2:
    st.markdown("""
    **Results (2023–2024 backtest):**
    - Agent return: +38.41%
    - Buy-and-Hold return: +103.87%
    - Win rate: 76.5% (13/17 trades profitable)
    - Sharpe ratio: 1.334
    - Max drawdown: -14.15%

    **Tech stack:**
    Python, scikit-learn, FastAPI, Docker,
    Render, GitHub Actions, Streamlit, yfinance
    """)

st.caption("Built by Falah Hilal")
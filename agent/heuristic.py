# agent/heuristic.py
# Heuristic Decision Agent
# What it does:
#   Loads the trained model and wraps it in a stateful agent
#   that makes BUY/SELL/HOLD decisions using:
#     1. Model prediction (UP/DOWN/NEUTRAL)
#     2. Confidence threshold — only act when sure enough
#     3. RSI filter — don't buy overbought, don't sell oversold
#     4. VIX filter — don't buy when market fear is very high
#     5. Position tracking — can't sell what you don't own
#
# The heuristic layer is what makes this an intelligent AGENT
# rather than just a classifier. It models rational behavior:
# only act when expected utility of acting > doing nothing.

import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# TRADING AGENT CLASS
class TradingAgent:
    """
    A stateful intelligent trading agent.

    Stateful means it remembers:
      - Whether it currently holds shares (position)
      - Every decision it has made and why (trade_log)

    This matters because:
      - You can't SELL if you don't own shares
      - You shouldn't BUY if you already hold a position
      - The log lets you analyze every decision after the fact
    """

    def __init__(self, model, scaler, feature_cols):
        """
        Args:
            model:        trained sklearn classifier
            scaler:       fitted StandardScaler (must match training)
            feature_cols: list of feature column names in correct order
        """
        self.model        = model
        self.scaler       = scaler
        self.feature_cols = feature_cols

        # State
        self.position  = 0      # 0 = no shares held, 1 = holding shares
        self.trade_log = []     # every decision recorded here

        # Thresholds from config
        self.buy_conf      = config.BUY_CONFIDENCE     # 0.65
        self.sell_conf     = config.SELL_CONFIDENCE    # 0.60
        self.rsi_oversold  = config.RSI_OVERSOLD       # 40
        self.rsi_overbought= config.RSI_OVERBOUGHT     # 60
        self.vix_fear_high = 30   # VIX above 30 = high market fear

    def decide(self, date, features_row):
        """
        Makes a BUY/SELL/HOLD decision for a single day.

        Args:
            date:         the date of this decision (for logging)
            features_row: pd.Series with all feature values for this day

        Returns:
            action: "BUY", "SELL", or "HOLD" as a string
        """
        # Extract key signals for heuristic rules
        rsi       = features_row.get("rsi",       50)
        vix       = features_row.get("vix_close", 20)

        # Scale features exactly as during training
        X = features_row[self.feature_cols].values.reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        # Get model prediction and confidence
        prediction  = self.model.predict(X_scaled)[0]
        proba       = self.model.predict_proba(X_scaled)[0]
        confidence  = proba.max()

        # HEURISTIC DECISION RULES
        # These rules implement rational agent behavior:
        # act only when multiple signals agree

        action = "HOLD"   # default — do nothing when uncertain
        reason = "default hold"

        # --- BUY CONDITIONS ---
        # All of these must be true to BUY:
        # 1. Model predicts UP
        # 2. Confidence above threshold (65%)
        # 3. RSI below oversold threshold (40) — room to rise
        # 4. VIX not in extreme fear territory (< 30)
        # 5. Not already holding a position
        if (
            prediction  == "UP"
            and confidence  >= self.buy_conf
            and rsi         <= self.rsi_oversold
            and vix         <  self.vix_fear_high
            and self.position == 0
        ):
            action = "BUY"
            reason = (f"UP predicted (conf={confidence:.2f}), "
                      f"RSI={rsi:.1f} oversold, VIX={vix:.1f} calm")

        # --- SELL CONDITIONS ---
        # All of these must be true to SELL:
        # 1. Model predicts DOWN
        # 2. Confidence above threshold (60%)
        # 3. Currently holding a position (can't sell what you don't have)
        # OR: RSI is very overbought and holding — take profit signal
        elif (
            prediction  == "DOWN"
            and confidence  >= self.sell_conf
            and self.position == 1
        ):
            action = "SELL"
            reason = (f"DOWN predicted (conf={confidence:.2f}), "
                      f"RSI={rsi:.1f}")

        # --- PANIC SELL: VIX spike while holding ---
        # If market fear suddenly spikes above 40 while holding,
        # exit position regardless of model prediction.
        # This is a risk management rule — protect capital first.
        elif (
            self.position == 1
            and vix >= 40
        ):
            action = "SELL"
            reason = f"VIX panic sell — VIX={vix:.1f} extreme fear"

        # --- TAKE PROFIT: RSI overbought while holding ---
        # If RSI goes very high while holding, take profit.
        elif (
            self.position == 1
            and rsi >= 75
            and confidence >= 0.55
        ):
            action = "SELL"
            reason = (f"Take profit — RSI={rsi:.1f} overbought, "
                      f"conf={confidence:.2f}")

        else:
            reason = (f"HOLD — pred={prediction}, "
                      f"conf={confidence:.2f}, "
                      f"RSI={rsi:.1f}, VIX={vix:.1f}, "
                      f"position={self.position}")

        # Update position state
        if action == "BUY":
            self.position = 1
        elif action == "SELL":
            self.position = 0

        # Log the decision
        self.trade_log.append({
            "date"        : date,
            "prediction"  : prediction,
            "confidence"  : round(confidence, 4),
            "rsi"         : round(rsi, 2),
            "vix"         : round(vix, 2),
            "position_before": 1 if (action == "SELL") else (0 if action == "BUY" else self.position),
            "action"      : action,
            "reason"      : reason,
        })

        return action

    def run(self, df):
        """
        Runs the agent over an entire DataFrame of daily data.
        Calls decide() for each row in chronological order.

        Args:
            df: DataFrame with DatetimeIndex and all feature columns

        Returns:
            decisions: pd.Series of actions indexed by date
        """
        decisions = {}

        for date, row in df.iterrows():
            action = self.decide(date, row)
            decisions[date] = action

        return pd.Series(decisions, name="action")

    def get_trade_log(self):
        """Returns trade log as a clean DataFrame."""
        return pd.DataFrame(self.trade_log)

    def reset(self):
        """Resets agent state — useful for running multiple simulations."""
        self.position  = 0
        self.trade_log = []

    def summary(self):
        """Prints a summary of all decisions made."""
        log = self.get_trade_log()
        if len(log) == 0:
            print("[agent] No decisions made yet.")
            return

        total  = len(log)
        buys   = (log["action"] == "BUY").sum()
        sells  = (log["action"] == "SELL").sum()
        holds  = (log["action"] == "HOLD").sum()

        print("\n" + "="*55)
        print("AGENT DECISION SUMMARY")
        print("="*55)
        print(f"Total days evaluated : {total}")
        print(f"BUY  decisions       : {buys}  ({100*buys/total:.1f}%)")
        print(f"SELL decisions       : {sells} ({100*sells/total:.1f}%)")
        print(f"HOLD decisions       : {holds} ({100*holds/total:.1f}%)")
        print(f"Final position       : {'HOLDING' if self.position == 1 else 'CASH'}")

        # Show all non-hold decisions
        trades = log[log["action"] != "HOLD"]
        if len(trades) > 0:
            print(f"\nAll trades ({len(trades)} total):")
            print(trades[["date", "action", "confidence",
                           "rsi", "vix", "reason"]].to_string(index=False))

# LOAD MODEL HELPER

def load_agent():
    """
    Loads the saved model, scaler, and feature columns
    and returns a ready-to-use TradingAgent.
    """
    if not os.path.exists(config.MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"No saved model found at {config.MODEL_SAVE_PATH}. "
            f"Run agent/classifier.py first."
        )

    with open(config.MODEL_SAVE_PATH, "rb") as f:
        payload = pickle.load(f)

    agent = TradingAgent(
        model        = payload["model"],
        scaler       = payload["scaler"],
        feature_cols = payload["feature_cols"],
    )

    print(f"[agent] Model loaded from {config.MODEL_SAVE_PATH}")
    print(f"[agent] Model type    : {type(payload['model']).__name__}")
    print(f"[agent] Feature count : {len(payload['feature_cols'])}")

    return agent


# TEST — run directly to verify agent decisions

if __name__ == "__main__":
    print("\n" + "="*55)
    print("PHASE 4 — HEURISTIC AGENT")
    print("="*55)

    # Load agent
    agent = load_agent()

    # Load test data (last 20% — same test period as classifier)
    df = pd.read_csv(
        config.FEATURES_PATH,
        index_col   = "Date",
        parse_dates = True
    )

    split_idx = int(len(df) * (1 - config.TEST_SPLIT))
    test_df   = df.iloc[split_idx:]

    print(f"\n[agent] Running agent over test period...")
    print(f"[agent] Test period: {test_df.index[0].date()} "
          f"→ {test_df.index[-1].date()}")
    print(f"[agent] Test rows  : {len(test_df)}")

    # Run agent over test period
    decisions = agent.run(test_df)

    # Print summary
    agent.summary()

    # Save trade log
    log_path = os.path.join(config.OUTPUT_DIR, "trade_log.csv")
    agent.get_trade_log().to_csv(log_path, index=False)
    print(f"\n[agent] Trade log saved to {log_path}")

    # Show action distribution
    print(f"\n[agent] Action distribution:")
    print(decisions.value_counts())
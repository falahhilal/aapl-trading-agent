# agent/backtester.py
# Portfolio Simulation Engine
#
# What it does:
#   Takes the agent's BUY/SELL/HOLD decisions and simulates
#   actual trading with real AAPL prices.
#
#   Tracks day by day:
#     - Cash available
#     - Shares held
#     - Total portfolio value (cash + shares × price)
#     - Every trade with entry/exit prices and profit/loss
#
#   Compares against Buy-and-Hold baseline:
#     - Buy on day 1, sell on last day, do nothing in between
#     - This is the benchmark agent must beat

import os
import warnings
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from agent.heuristic import load_agent

# BACKTESTER CLASS

class Backtester:
    """
    Simulates portfolio performance given a sequence of
    BUY/SELL/HOLD decisions over a historical price series.

    Transaction cost of 0.1% is applied to every trade.
    This models realistic brokerage fees and prevents the
    simulation from looking unrealistically profitable.
    """

    def __init__(self):
        self.initial_capital = config.INITIAL_CAPITAL      # $10,000
        self.transaction_cost= config.TRANSACTION_COST     # 0.001 = 0.1%

        # Portfolio state
        self.cash            = self.initial_capital
        self.shares          = 0
        self.portfolio_values= []    # total value each day
        self.dates           = []

        # Trade tracking
        self.trades          = []    # completed round-trip trades
        self.entry_price     = None  # price paid on last BUY
        self.entry_date      = None  # date of last BUY

    def run(self, decisions, price_series):
        """
        Runs the backtest simulation.

        Args:
            decisions:    pd.Series of "BUY"/"SELL"/"HOLD" indexed by date
            price_series: pd.Series of closing prices indexed by date

        Returns:
            pd.Series of daily portfolio values indexed by date
        """
        # Align decisions and prices on the same dates
        common_dates = decisions.index.intersection(price_series.index)
        decisions    = decisions[common_dates]
        prices       = price_series[common_dates]

        print(f"[backtester] Running simulation over {len(common_dates)} days...")
        print(f"[backtester] Starting capital: ${self.initial_capital:,.2f}")

        for date, action in decisions.items():
            price = prices[date]

            if action == "BUY" and self.cash >= price:
                # Buy as many whole shares as possible
                shares_to_buy    = int(self.cash // price)
                cost             = shares_to_buy * price
                transaction_fee  = cost * self.transaction_cost
                total_cost       = cost + transaction_fee

                self.shares     += shares_to_buy
                self.cash       -= total_cost
                self.entry_price = price
                self.entry_date  = date

                print(f"[backtester]   BUY  {date.date()} | "
                      f"${price:.2f}/share × {shares_to_buy} shares | "
                      f"fee=${transaction_fee:.2f} | "
                      f"cash left=${self.cash:.2f}")

            elif action == "SELL" and self.shares > 0:
                # Sell all shares
                proceeds         = self.shares * price
                transaction_fee  = proceeds * self.transaction_cost
                net_proceeds     = proceeds - transaction_fee

                # Calculate profit/loss on this round trip
                if self.entry_price is not None:
                    pnl         = net_proceeds - (self.shares * self.entry_price)
                    pnl_pct     = (price - self.entry_price) / self.entry_price * 100
                else:
                    pnl     = 0
                    pnl_pct = 0

                self.trades.append({
                    "entry_date"  : self.entry_date,
                    "exit_date"   : date,
                    "entry_price" : self.entry_price,
                    "exit_price"  : price,
                    "shares"      : self.shares,
                    "pnl"         : round(pnl, 2),
                    "pnl_pct"     : round(pnl_pct, 2),
                    "profitable"  : pnl > 0,
                })

                self.cash   += net_proceeds
                self.shares  = 0

                pnl_sign = "+" if pnl >= 0 else ""
                print(f"[backtester]   SELL {date.date()} | "
                      f"${price:.2f}/share | "
                      f"P&L={pnl_sign}${pnl:.2f} ({pnl_sign}{pnl_pct:.1f}%) | "
                      f"cash=${self.cash:.2f}")

            # Record portfolio value for this day
            portfolio_value = self.cash + self.shares * price
            self.portfolio_values.append(portfolio_value)
            self.dates.append(date)

        portfolio_series = pd.Series(
            self.portfolio_values,
            index = self.dates,
            name  = "portfolio_value"
        )

        return portfolio_series

    def get_trades(self):
        """Returns completed trades as a DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame(self.trades)

    def summary(self):
        """Prints a summary of backtest results."""
        if not self.portfolio_values:
            print("[backtester] No simulation run yet.")
            return

        final_value  = self.portfolio_values[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital * 100

        trades_df    = self.get_trades()
        n_trades     = len(trades_df)
        n_profitable = trades_df["profitable"].sum() if n_trades > 0 else 0
        win_rate     = (n_profitable / n_trades * 100) if n_trades > 0 else 0

        print("\n" + "="*55)
        print("BACKTEST SUMMARY — AGENT")
        print("="*55)
        print(f"Starting capital  : ${self.initial_capital:>10,.2f}")
        print(f"Final value       : ${final_value:>10,.2f}")
        print(f"Total return      : {total_return:>+.2f}%")
        print(f"Total trades      : {n_trades}")
        print(f"Profitable trades : {n_profitable} / {n_trades}")
        print(f"Win rate          : {win_rate:.1f}%")

        if n_trades > 0:
            print(f"\nTrade breakdown:")
            print(trades_df[["entry_date", "exit_date",
                              "entry_price", "exit_price",
                              "pnl", "pnl_pct", "profitable"]].to_string(index=False))

# BUY AND HOLD BASELINE

def buy_and_hold(price_series, initial_capital):
    """
    Simulates a passive Buy-and-Hold strategy.
    Buy on day 1, hold forever, sell on last day.
    This is the benchmark the agent must beat.

    Returns pd.Series of daily portfolio values.
    """
    first_price  = price_series.iloc[0]
    shares       = int(initial_capital // first_price)
    cash_left    = initial_capital - shares * first_price

    portfolio    = cash_left + shares * price_series
    final_value  = portfolio.iloc[-1]
    total_return = (final_value - initial_capital) / initial_capital * 100

    print(f"\n[backtester] Buy-and-Hold baseline:")
    print(f"[backtester]   Bought {shares} shares at ${first_price:.2f} on {price_series.index[0].date()}")
    print(f"[backtester]   Final value  : ${final_value:,.2f}")
    print(f"[backtester]   Total return : {total_return:+.2f}%")

    return portfolio


# MAIN FUNCTION

def run_backtest():
    """
    Runs the full backtest pipeline:
    1. Loads agent and data
    2. Runs agent over test period
    3. Simulates portfolio
    4. Computes Buy-and-Hold baseline
    5. Saves results for metrics.py
    """
    print("\n" + "="*55)
    print("PHASE 5 — BACKTESTING")
    print("="*55)

    # Load data
    df = pd.read_csv(
        config.FEATURES_PATH,
        index_col   = "Date",
        parse_dates = True
    )

    # Use test period only (last 20%)
    split_idx = int(len(df) * (1 - config.TEST_SPLIT))
    test_df   = df.iloc[split_idx:]
    prices    = test_df["Close"]

    print(f"[backtester] Test period : {test_df.index[0].date()} → {test_df.index[-1].date()}")
    print(f"[backtester] Test rows   : {len(test_df)}")

    # Load agent and run decisions
    agent     = load_agent()
    agent.reset()
    decisions = agent.run(test_df)

    # Run backtest simulation
    backtester       = Backtester()
    portfolio_series = backtester.run(decisions, prices)

    # Buy-and-Hold baseline
    bah_series = buy_and_hold(prices, config.INITIAL_CAPITAL)

    # Print agent summary
    backtester.summary()

    # Save results to disk for metrics.py
    results = pd.DataFrame({
        "agent_portfolio"  : portfolio_series,
        "bah_portfolio"    : bah_series,
        "close_price"      : prices,
        "action"           : decisions,
    })

    results.index.name = "Date"

    results_path = os.path.join(config.OUTPUT_DIR, "backtest_results.csv")
    results.to_csv(results_path)
    print(f"\n[backtester] Results saved to {results_path}")

    # Save trades
    trades_path = os.path.join(config.OUTPUT_DIR, "trades.csv")
    backtester.get_trades().to_csv(trades_path, index=False)
    print(f"[backtester] Trades saved to {trades_path}")

    return portfolio_series, bah_series, backtester.get_trades()

# TEST

if __name__ == "__main__":
    portfolio_series, bah_series, trades_df = run_backtest()
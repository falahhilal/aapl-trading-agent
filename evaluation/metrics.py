# evaluation/metrics.py
# Performance Metrics & Visualizations
#
# Computes:
#   1. Total Return %
#   2. Sharpe Ratio
#   3. Maximum Drawdown
#   4. Win Rate
#   5. Equity curve plot (agent vs Buy-and-Hold)
#   6. Drawdown chart
#   7. Trade P&L bar chart
#   8. Final summary table

import os
import warnings
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd
import matplotlib.pyplot  as plt
import matplotlib.dates   as mdates
import seaborn as sns

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# METRIC CALCULATIONS

def total_return(portfolio_series, initial_capital):
    """
    Total return as a percentage.
    (final value - initial) / initial × 100
    """
    final = portfolio_series.iloc[-1]
    return (final - initial_capital) / initial_capital * 100


def sharpe_ratio(portfolio_series, risk_free_rate=0.0):
    """
    Sharpe Ratio — risk-adjusted return.

    Measures return per unit of risk taken.
    Higher is better. Above 1.0 is good, above 2.0 is great.

    Formula:
        Sharpe = (mean daily return - risk free rate) /
                  std of daily returns × sqrt(252)

    Annualized by multiplying by sqrt(252) trading days.
    Risk-free rate default = 0 for simplicity.
    """
    daily_returns = portfolio_series.pct_change().dropna()

    if daily_returns.std() == 0:
        return 0.0

    sharpe = (daily_returns.mean() - risk_free_rate) / daily_returns.std()
    return sharpe * np.sqrt(252)


def maximum_drawdown(portfolio_series):
    """
    Maximum Drawdown — worst peak-to-trough loss.

    Measures the largest single loss from any peak
    to any subsequent trough during the test period.

    Example: portfolio goes $10k → $12k → $9k
    Drawdown = (9k - 12k) / 12k = -25%

    Lower (more negative) is worse.
    """
    rolling_max = portfolio_series.cummax()
    drawdown    = (portfolio_series - rolling_max) / rolling_max
    return drawdown.min() * 100


def drawdown_series(portfolio_series):
    """Returns the full drawdown series over time (for plotting)."""
    rolling_max = portfolio_series.cummax()
    return ((portfolio_series - rolling_max) / rolling_max) * 100


def win_rate(trades_df):
    """
    Percentage of trades that were profitable.
    Only meaningful if there are enough trades.
    """
    if len(trades_df) == 0:
        return 0.0
    return trades_df["profitable"].mean() * 100


def compute_all_metrics(portfolio_series, bah_series, trades_df):
    """
    Computes all metrics for both agent and Buy-and-Hold.
    Returns a clean summary dictionary.
    """
    metrics = {
        "Agent": {
            "Final Value ($)"    : round(portfolio_series.iloc[-1], 2),
            "Total Return (%)"   : round(total_return(portfolio_series, config.INITIAL_CAPITAL), 2),
            "Sharpe Ratio"       : round(sharpe_ratio(portfolio_series), 3),
            "Max Drawdown (%)"   : round(maximum_drawdown(portfolio_series), 2),
            "Win Rate (%)"       : round(win_rate(trades_df), 1),
            "Total Trades"       : len(trades_df),
        },
        "Buy and Hold": {
            "Final Value ($)"    : round(bah_series.iloc[-1], 2),
            "Total Return (%)"   : round(total_return(bah_series, config.INITIAL_CAPITAL), 2),
            "Sharpe Ratio"       : round(sharpe_ratio(bah_series), 3),
            "Max Drawdown (%)"   : round(maximum_drawdown(bah_series), 2),
            "Win Rate (%)"       : "N/A",
            "Total Trades"       : 1,
        }
    }
    return metrics

# VISUALIZATIONS

def plot_equity_curve(portfolio_series, bah_series):
    """
    Plot 1: Equity curve — agent portfolio vs Buy-and-Hold.
    This is the headline chart of the entire project.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(portfolio_series.index, portfolio_series.values,
            color="#2196F3", linewidth=1.5, label="Agent", zorder=3)
    ax.plot(bah_series.index, bah_series.values,
            color="#FF9800", linewidth=1.5, label="Buy and Hold",
            linestyle="--", zorder=2)

    ax.axhline(config.INITIAL_CAPITAL, color="gray",
               linestyle=":", linewidth=1, alpha=0.7, label="Starting Capital")

    # Shade region where agent beats Buy-and-Hold
    ax.fill_between(
        portfolio_series.index,
        portfolio_series.values,
        bah_series.values,
        where  = portfolio_series.values >= bah_series.values,
        alpha  = 0.15,
        color  = "#4CAF50",
        label  = "Agent ahead"
    )
    ax.fill_between(
        portfolio_series.index,
        portfolio_series.values,
        bah_series.values,
        where  = portfolio_series.values < bah_series.values,
        alpha  = 0.15,
        color  = "#F44336",
        label  = "Agent behind"
    )

    ax.set_title("Portfolio Equity Curve — Agent vs Buy-and-Hold",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend(fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x:,.0f}")
    )

    plt.tight_layout()
    path = os.path.join(config.OUTPUT_DIR, "equity_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[metrics] Saved: {path}")


def plot_drawdown(portfolio_series, bah_series):
    """
    Plot 2: Drawdown chart — shows worst loss periods.
    """
    agent_dd = drawdown_series(portfolio_series)
    bah_dd   = drawdown_series(bah_series)

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.fill_between(agent_dd.index, agent_dd.values, 0,
                    alpha=0.4, color="#F44336", label="Agent Drawdown")
    ax.fill_between(bah_dd.index, bah_dd.values, 0,
                    alpha=0.2, color="#FF9800", label="Buy-and-Hold Drawdown")

    ax.plot(agent_dd.index, agent_dd.values,
            color="#F44336", linewidth=1)
    ax.plot(bah_dd.index, bah_dd.values,
            color="#FF9800", linewidth=1, linestyle="--")

    ax.set_title("Drawdown Over Time", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(config.OUTPUT_DIR, "drawdown.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[metrics] Saved: {path}")


def plot_trade_pnl(trades_df):
    """
    Plot 3: Individual trade P&L bar chart.
    Green = profitable trade, red = losing trade.
    """
    if len(trades_df) == 0:
        print("[metrics] No trades to plot.")
        return

    fig, ax = plt.subplots(figsize=(14, 5))

    colors = ["#4CAF50" if p else "#F44336"
              for p in trades_df["profitable"]]
    x      = range(len(trades_df))

    bars = ax.bar(x, trades_df["pnl_pct"], color=colors,
                  edgecolor="white", linewidth=0.5)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Individual Trade P&L (%)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Trade #")
    ax.set_ylabel("P&L (%)")

    # Label each bar with entry date
    ax.set_xticks(list(x))
    ax.set_xticklabels(
        [str(d)[:10] if d else "" for d in trades_df["entry_date"]],
        rotation=45, ha="right", fontsize=7
    )
    ax.grid(True, alpha=0.3, axis="y")

    # Legend
    from matplotlib.patches import Patch
    legend = [Patch(color="#4CAF50", label="Profitable"),
              Patch(color="#F44336", label="Loss")]
    ax.legend(handles=legend, fontsize=10)

    plt.tight_layout()
    path = os.path.join(config.OUTPUT_DIR, "trade_pnl.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[metrics] Saved: {path}")


def plot_price_with_trades(price_series, trades_df):
    """
    Plot 4: AAPL price chart with BUY/SELL markers overlaid.
    Shows exactly when the agent entered and exited positions.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(price_series.index, price_series.values,
            color="#607D8B", linewidth=1, alpha=0.8, label="AAPL Close")

    if len(trades_df) > 0:
        # BUY markers
        buy_dates  = pd.to_datetime(trades_df["entry_date"])
        buy_prices = trades_df["entry_price"].values
        ax.scatter(buy_dates, buy_prices,
                   marker="^", color="#4CAF50", s=120,
                   zorder=5, label="BUY", edgecolors="white", linewidths=0.5)

        # SELL markers
        sell_dates  = pd.to_datetime(trades_df["exit_date"])
        sell_prices = trades_df["exit_price"].values
        ax.scatter(sell_dates, sell_prices,
                   marker="v", color="#F44336", s=120,
                   zorder=5, label="SELL", edgecolors="white", linewidths=0.5)

    ax.set_title("AAPL Price with Agent BUY/SELL Signals",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    ax.legend(fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x:.0f}")
    )

    plt.tight_layout()
    path = os.path.join(config.OUTPUT_DIR, "price_with_trades.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[metrics] Saved: {path}")

# PRINT SUMMARY TABLE

def print_summary_table(metrics):
    """Prints the final comparison table."""
    print("\n" + "="*55)
    print("FINAL PERFORMANCE SUMMARY")
    print("="*55)

    rows = list(metrics["Agent"].keys())
    col1 = [str(metrics["Agent"][r])       for r in rows]
    col2 = [str(metrics["Buy and Hold"][r]) for r in rows]

    # Header
    print(f"{'Metric':<22} {'Agent':>14} {'Buy and Hold':>14}")
    print("-" * 52)
    for row, v1, v2 in zip(rows, col1, col2):
        print(f"{row:<22} {v1:>14} {v2:>14}")

    print()

    # Verdict
    agent_return = metrics["Agent"]["Total Return (%)"]
    bah_return   = metrics["Buy and Hold"]["Total Return (%)"]

    if agent_return > bah_return:
        print(f"RESULT: Agent BEATS Buy-and-Hold "
              f"by {agent_return - bah_return:.2f}%")
    else:
        print(f"RESULT: Agent TRAILS Buy-and-Hold "
              f"by {bah_return - agent_return:.2f}%")
    print("(Both outcomes are valid scientific results)")


# MAIN FUNCTION

def run_metrics():
    """
    Loads backtest results and computes all metrics and plots.
    """
    print("\n" + "="*55)
    print("PHASE 5 — EVALUATION & METRICS")
    print("="*55)

    # Load backtest results
    results_path = os.path.join(config.OUTPUT_DIR, "backtest_results.csv")
    if not os.path.exists(results_path):
        raise FileNotFoundError(
            "backtest_results.csv not found. "
            "Run agent/backtester.py first."
        )

    results = pd.read_csv(results_path, index_col="Date", parse_dates=True)

    portfolio_series = results["agent_portfolio"]
    bah_series       = results["bah_portfolio"]
    price_series     = results["close_price"]

    # Load trades
    trades_path = os.path.join(config.OUTPUT_DIR, "trades.csv")
    trades_df   = pd.read_csv(trades_path) if os.path.exists(trades_path) \
                  else pd.DataFrame()

    # Compute metrics
    metrics = compute_all_metrics(portfolio_series, bah_series, trades_df)

    # Print summary table
    print_summary_table(metrics)

    # Generate all plots
    print("\n[metrics] Generating plots...")
    plot_equity_curve(portfolio_series, bah_series)
    plot_drawdown(portfolio_series, bah_series)
    plot_trade_pnl(trades_df)
    plot_price_with_trades(price_series, trades_df)

    print("\n[metrics] Phase 5 complete.")
    print(f"[metrics] All plots saved to {config.OUTPUT_DIR}")

    return metrics

# TEST

if __name__ == "__main__":
    metrics = run_metrics()
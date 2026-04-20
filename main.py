# main.py
# Runs the entire AAPL Trading Agent pipeline end to end.

import warnings
warnings.filterwarnings("ignore")

from data.collector        import collect_all
from data.preprocessor     import run_preprocessing
from features.technical    import build_features
from agent.classifier      import run_training
from agent.backtester      import run_backtest
from evaluation.metrics    import run_metrics

def main():
    print("\n" + "="*55)
    print("AAPL TRADING AGENT — FULL PIPELINE")
    print("="*55)

    print("\n[main] Phase 1 — Data Collection")
    collect_all()

    print("\n[main] Phase 1 — Preprocessing")
    run_preprocessing()

    print("\n[main] Phase 2 — Feature Engineering")
    build_features()

    print("\n[main] Phase 3 — Model Training")
    run_training()

    print("\n[main] Phase 5 — Backtesting")
    run_backtest()

    print("\n[main] Phase 5 — Metrics & Evaluation")
    metrics = run_metrics()

    print("\n" + "="*55)
    print("PIPELINE COMPLETE")
    print("="*55)
    print("All outputs saved to outputs/")
    print("Trained model saved to models/")

if __name__ == "__main__":
    main()
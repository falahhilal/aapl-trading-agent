# agent/classifier.py
# Model Training & Selection
#
# What it does:
#   1. Loads aapl_features.csv
#   2. Defines exactly which columns are features vs labels
#   3. Chronological train/test split 
#   4. Trains 5 models: LogReg, KNN, SVM, Random Forest, MLP
#   5. Walk-forward validation across years
#   6. Compares all models on F1-score
#   7. Saves best model to models/best_model.pkl

import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model    import LogisticRegression
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.svm             import SVC
from sklearn.ensemble        import RandomForestClassifier
from sklearn.neural_network  import MLPClassifier
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import (
    f1_score, accuracy_score, classification_report, confusion_matrix
)

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# FEATURE COLUMNS
# Not using Close/High/Low/Open/Volume directly as features —
# they are price levels not returns, and leak scale information.


FEATURE_COLS = [
    # Basic price-derived
    "daily_return", "log_return", "high_low_range",
    "close_open_gap", "prev_close",

    # VIX macro features
    "vix_close", "vix_change",

    # RSI
    "rsi",

    # MACD
    "macd", "macd_signal", "macd_histogram",

    # Bollinger Bands
    "bb_width", "bb_position",

    # Rolling features
    "rolling_std_5",  "price_vs_mean_5",
    "rolling_std_10", "price_vs_mean_10",
    "rolling_std_20", "price_vs_mean_20",

    # Lag returns (not raw prices — returns are stationary)
    "lag_return_1", "lag_return_2",
    "lag_return_3", "lag_return_5",

    # Volume
    "volume_ratio", "log_volume", "volume_change",

    # Cyclic date
    "dow_sin", "dow_cos", "month_sin", "month_cos",

    # Momentum
    "roc_5", "roc_10",
]

LABEL_COL = "label"

# PART 1 — LOAD DATA

def load_data():
    """
    Loads the feature-engineered dataset.
    Returns X (features), y (labels), and the full DataFrame.
    """
    print("[classifier] Loading data...")

    df = pd.read_csv(
        config.FEATURES_PATH,
        index_col   = "Date",
        parse_dates = True
    )

    # Verify all feature columns exist
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X = df[FEATURE_COLS]
    y = df[LABEL_COL]

    print(f"[classifier] Loaded {len(df)} rows.")
    print(f"[classifier] Features : {len(FEATURE_COLS)}")
    print(f"[classifier] Labels   : {y.value_counts().to_dict()}")

    return X, y, df


# PART 2 — CHRONOLOGICAL TRAIN/TEST SPLIT

def train_test_split_chronological(X, y):
    """
    Splits data chronologically — last 20% is test set.
    NEVER shuffle time series data. Future data must never
    appear in the training set.
    """
    split_idx = int(len(X) * (1 - config.TEST_SPLIT))

    X_train = X.iloc[:split_idx]
    X_test  = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test  = y.iloc[split_idx:]

    print(f"\n[classifier] Train/test split:")
    print(f"[classifier]   Train: {len(X_train)} rows "
          f"({X_train.index[0].date()} → {X_train.index[-1].date()})")
    print(f"[classifier]   Test : {len(X_test)} rows "
          f"({X_test.index[0].date()} → {X_test.index[-1].date()})")

    return X_train, X_test, y_train, y_test


# PART 3 — MODEL DEFINITIONS

def get_models():
    """
    Returns all 5 models as a dictionary.
    All models use their most reasonable default settings.
    Random state fixed for reproducibility.

    Note: SVM and LogReg need scaled features (StandardScaler).
          KNN, RandomForest, MLP also benefit from scaling.
          We scale everything for consistency.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter     = 1000,
            random_state = 42,
            class_weight = "balanced"   # handles class imbalance
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors = 10,
            weights     = "distance"    # closer neighbors count more
        ),
        "SVM": SVC(
            kernel       = "rbf",
            probability  = True,        # needed for confidence scores
            random_state = 42,
            class_weight = "balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators = 200,
            max_depth    = 10,
            random_state = 42,
            class_weight = "balanced",
            n_jobs       = -1           # use all CPU cores
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes = (128, 64),
            max_iter           = 500,
            random_state       = 42,
            early_stopping     = True,  # stop if validation stops improving
            validation_fraction= 0.1
        ),
    }
    return models


# PART 4 — TRAIN AND EVALUATE ALL MODELS

def train_and_evaluate(X_train, X_test, y_train, y_test):
    """
    Trains all 5 models on training data.
    Evaluates each on test data.
    Returns results dictionary and all trained models.
    """
    print("\n" + "="*55)
    print("TRAINING ALL MODELS")
    print("="*55)

    # Scale features — fit ONLY on training data
    # then transform both train and test
    # If you fit on all data, test data leaks into scaling = data leakage
    scaler  = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    models  = get_models()
    results = {}

    for name, model in models.items():
        print(f"\n[classifier] Training {name}...")

        # Train
        model.fit(X_train_scaled, y_train)

        # Predict on test set
        y_pred = model.predict(X_test_scaled)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average="macro", zero_division=0)

        results[name] = {
            "model"    : model,
            "accuracy" : acc,
            "f1_macro" : f1,
            "y_pred"   : y_pred,
        }

        print(f"[classifier]   Accuracy : {acc:.4f}")
        print(f"[classifier]   F1 Macro : {f1:.4f}")

    # Store scaler with results so we can use it later
    results["_scaler"] = scaler

    return results


# PART 5 — WALK-FORWARD VALIDATION

def walk_forward_validation(X, y):
    """
    Walk-forward validation — the correct way to validate
    trading strategies.

    Instead of one fixed train/test split, we roll forward:
      Train on 2015–2018, test on 2019
      Train on 2015–2019, test on 2020
      Train on 2015–2020, test on 2021
      Train on 2015–2021, test on 2022
      Train on 2015–2022, test on 2023

    This simulates how a real trader retrains their model
    each year on all available history.

    We only run this on Random Forest — the likely best model —
    to keep runtime reasonable.
    """
    print("\n" + "="*55)
    print("WALK-FORWARD VALIDATION (Random Forest)")
    print("="*55)

    scaler    = StandardScaler()
    wf_results = []

    start_year = X.index[0].year
    end_year   = X.index[-1].year

    # Need at least TRAIN_YEARS of data before first test year
    test_years = range(
        start_year + config.TRAIN_YEARS,
        end_year             # last year kept for final test
    )

    for test_year in test_years:
        # All data before test year = training
        train_mask = X.index.year < test_year
        test_mask  = X.index.year == test_year

        X_tr = X[train_mask]
        y_tr = y[train_mask]
        X_te = X[test_mask]
        y_te = y[test_mask]

        if len(X_te) == 0:
            continue

        # Scale
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_te_scaled = scaler.transform(X_te)

        # Train Random Forest
        model = RandomForestClassifier(
            n_estimators = 200,
            max_depth    = 10,
            random_state = 42,
            class_weight = "balanced",
            n_jobs       = -1
        )
        model.fit(X_tr_scaled, y_tr)
        y_pred = model.predict(X_te_scaled)

        f1  = f1_score(y_te, y_pred, average="macro", zero_division=0)
        acc = accuracy_score(y_te, y_pred)

        wf_results.append({
            "test_year"  : test_year,
            "train_rows" : len(X_tr),
            "test_rows"  : len(X_te),
            "f1_macro"   : f1,
            "accuracy"   : acc,
        })

        print(f"[classifier]   {test_year}: "
              f"F1={f1:.3f}  Acc={acc:.3f}  "
              f"(trained on {len(X_tr)} rows, tested on {len(X_te)} rows)")

    wf_df = pd.DataFrame(wf_results)

    print(f"\n[classifier] Walk-forward summary:")
    print(f"[classifier]   Mean F1  : {wf_df['f1_macro'].mean():.3f}")
    print(f"[classifier]   Std F1   : {wf_df['f1_macro'].std():.3f}")
    print(f"[classifier]   Min F1   : {wf_df['f1_macro'].min():.3f}")
    print(f"[classifier]   Max F1   : {wf_df['f1_macro'].max():.3f}")

    return wf_df


# PART 6 — COMPARE MODELS AND SELECT BEST

def select_best_model(results):
    """
    Compares all models by F1 macro score.
    Returns the name and model object of the best performer.
    """
    print("\n" + "="*55)
    print("MODEL COMPARISON")
    print("="*55)

    # Build comparison table
    comparison = []
    for name, res in results.items():
        if name.startswith("_"):
            continue
        comparison.append({
            "Model"    : name,
            "Accuracy" : res["accuracy"],
            "F1 Macro" : res["f1_macro"],
        })

    comp_df = pd.DataFrame(comparison).sort_values(
        "F1 Macro", ascending=False
    ).reset_index(drop=True)

    print(comp_df.to_string(index=False))

    best_name = comp_df.iloc[0]["Model"]
    print(f"\n[classifier] Best model: {best_name} "
          f"(F1={comp_df.iloc[0]['F1 Macro']:.4f})")

    return best_name, results[best_name]["model"]

# PART 7 — DETAILED REPORT FOR BEST MODEL

def print_detailed_report(best_name, results, y_test):
    """
    Prints full classification report and confusion matrix
    for the best model.
    """
    print(f"\n{'='*55}")
    print(f"DETAILED REPORT — {best_name}")
    print("="*55)

    y_pred = results[best_name]["y_pred"]

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=["UP", "DOWN", "NEUTRAL"])
    print("Confusion Matrix (rows=actual, cols=predicted):")
    cm_df = pd.DataFrame(
        cm,
        index   = ["Actual UP", "Actual DOWN", "Actual NEUTRAL"],
        columns = ["Pred UP",   "Pred DOWN",   "Pred NEUTRAL"]
    )
    print(cm_df)


# PART 8 — SAVE PLOTS

def save_plots(results, wf_df, y_test, best_name):
    """
    Saves two plots to outputs/:
    1. Model comparison bar chart
    2. Walk-forward F1 over years
    """
    # --- Plot 1: Model comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Model Comparison", fontsize=13, fontweight="bold")

    names  = [n for n in results if not n.startswith("_")]
    f1s    = [results[n]["f1_macro"] for n in names]
    accs   = [results[n]["accuracy"] for n in names]
    colors = ["#2196F3" if n != best_name else "#4CAF50" for n in names]

    axes[0].barh(names, f1s, color=colors)
    axes[0].set_xlabel("F1 Macro Score")
    axes[0].set_title("F1 Macro (green = best)")
    axes[0].set_xlim(0, 1)
    for i, v in enumerate(f1s):
        axes[0].text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)

    axes[1].barh(names, accs, color=colors)
    axes[1].set_xlabel("Accuracy")
    axes[1].set_title("Accuracy (green = best)")
    axes[1].set_xlim(0, 1)
    for i, v in enumerate(accs):
        axes[1].text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    path1 = os.path.join(config.OUTPUT_DIR, "model_comparison.png")
    plt.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[classifier] Saved: {path1}")

    # --- Plot 2: Walk-forward F1 over years ---
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(wf_df["test_year"], wf_df["f1_macro"],
            marker="o", color="#2196F3", linewidth=2, markersize=7)
    ax.axhline(wf_df["f1_macro"].mean(), color="red",
               linestyle="--", linewidth=1, label=f"Mean F1: {wf_df['f1_macro'].mean():.3f}")
    ax.set_xlabel("Test Year")
    ax.set_ylabel("F1 Macro Score")
    ax.set_title("Walk-Forward Validation — Random Forest F1 by Year")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    path2 = os.path.join(config.OUTPUT_DIR, "walk_forward_validation.png")
    plt.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[classifier] Saved: {path2}")

# PART 9 — SAVE BEST MODEL

def save_model(model, scaler, feature_cols):
    """
    Saves the best model + scaler + feature columns together.
    We save all three because:
    - model:        the trained classifier
    - scaler:       must use the SAME scaler at inference time
    - feature_cols: must use the SAME features in the SAME order
    """
    payload = {
        "model"        : model,
        "scaler"       : scaler,
        "feature_cols" : feature_cols,
        "label_classes": ["UP", "DOWN", "NEUTRAL"],
    }
    with open(config.MODEL_SAVE_PATH, "wb") as f:
        pickle.dump(payload, f)

    print(f"\n[classifier] Model saved to {config.MODEL_SAVE_PATH}")

# MAIN FUNCTION

def run_training():
    """
    Runs the full training pipeline.
    Returns best model name, model object, and scaler.
    """
    print("\n" + "="*55)
    print("PHASE 3 — MODEL TRAINING")
    print("="*55)

    # Load data
    X, y, df = load_data()

    # Chronological split
    X_train, X_test, y_train, y_test = train_test_split_chronological(X, y)

    # Train all models
    results = train_and_evaluate(X_train, X_test, y_train, y_test)

    # Walk-forward validation
    wf_df = walk_forward_validation(X, y)

    # Select best
    best_name, best_model = select_best_model(results)

    # Detailed report
    print_detailed_report(best_name, results, y_test)

    # Save plots
    save_plots(results, wf_df, y_test, best_name)

    # Save model
    scaler = results["_scaler"]
    save_model(best_model, scaler, FEATURE_COLS)

    print("\n[classifier] Phase 3 complete.")
    print(f"[classifier] Best model : {best_name}")
    print(f"[classifier] Saved to   : {config.MODEL_SAVE_PATH}")

    return best_name, best_model, scaler

# TEST

if __name__ == "__main__":
    best_name, best_model, scaler = run_training()
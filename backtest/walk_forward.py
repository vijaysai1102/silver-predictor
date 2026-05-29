"""
Walk-forward (expanding-window) backtest for the QuantEnsemble.
- Train on all data before the test fold, predict one step ahead.
- No data from the test fold ever touches the training set.
- Reports: directional accuracy vs naive baseline, MAE, RMSE, interval coverage,
  and a simple reliability/calibration table.
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


def run_walk_forward(
    df: pd.DataFrame,
    feature_cols: list[str],
    min_train_rows: int = 500,
    step: int = 1,
) -> pd.DataFrame:
    """
    Expanding-window walk-forward backtest.

    df           : feature matrix with 'target_return' and 'target_close' columns.
    feature_cols : columns to use as predictors.
    min_train_rows: minimum number of rows before we start predicting.
    step         : predict every `step` rows (1 = every day).

    Returns a DataFrame with columns:
        date, actual_return, actual_close, pred_return, pred_close,
        ci_lower_80, ci_upper_80, direction_prob, direction_pred, direction_actual
    """
    from quant.ensemble import QuantEnsemble

    X = df[feature_cols].values
    y_ret   = df["target_return"].values
    y_close = df["target_close"].values
    dates   = df.index

    results = []
    n = len(df)

    for i in range(min_train_rows, n, step):
        X_train = X[:i]
        y_train = y_ret[:i]

        # Current close = last close in training window (i-1)
        current_close = y_close[i - 1]   # close of day i-1 → predicting close of day i

        model = QuantEnsemble()
        model.fit(
            pd.DataFrame(X_train, columns=feature_cols),
            pd.Series(y_train),
            last_close=current_close,
        )

        X_test_row = pd.DataFrame([X[i]], columns=feature_cols)
        pred = model.predict(X_test_row, current_close=current_close)

        actual_ret   = float(y_ret[i])
        actual_close = float(y_close[i])
        direction_actual = "up" if actual_ret >= 0 else "down"

        results.append({
            "date":             dates[i],
            "actual_return":    actual_ret,
            "actual_close":     actual_close,
            "pred_return":      pred["pred_return"],
            "pred_close":       pred["pred_close"],
            "ci_lower_80":      pred["ci_lower_80"],
            "ci_upper_80":      pred["ci_upper_80"],
            "direction_prob":   pred["direction_prob"],
            "direction_pred":   pred["direction"],
            "direction_actual": direction_actual,
            "residual_std":     pred["residual_std"],
        })

        done = i - min_train_rows
        total = n - min_train_rows
        if done % 100 == 0:
            logger.info("Walk-forward: %d/%d (%.0f%%)", done, total, 100 * done / max(total, 1))

    return pd.DataFrame(results).set_index("date")


def naive_baseline(df_bt: pd.DataFrame) -> dict:
    """'Tomorrow = today' naive baseline stats for comparison."""
    correct = (df_bt["direction_pred"] == df_bt["direction_actual"]).sum()
    return {
        "name": "Naive (tomorrow=today)",
        "directional_accuracy": 0.5,   # random walk ≈ 50% up/down
        "MAE":  float(df_bt["actual_close"].diff().abs().mean()),
        "RMSE": float(np.sqrt((df_bt["actual_close"].diff() ** 2).mean())),
    }


def compute_metrics(df_bt: pd.DataFrame) -> dict:
    """Compute all backtest metrics from the results DataFrame."""
    # Directional accuracy
    dir_acc = (df_bt["direction_pred"] == df_bt["direction_actual"]).mean()

    # Price errors
    mae  = (df_bt["pred_close"] - df_bt["actual_close"]).abs().mean()
    rmse = np.sqrt(((df_bt["pred_close"] - df_bt["actual_close"]) ** 2).mean())

    # 80% interval coverage
    in_ci = (
        (df_bt["actual_close"] >= df_bt["ci_lower_80"]) &
        (df_bt["actual_close"] <= df_bt["ci_upper_80"])
    ).mean()

    # Naive baseline
    naive_dir = 0.5
    naive_mae = (df_bt["actual_close"].diff().abs().mean())
    naive_rmse = np.sqrt((df_bt["actual_close"].diff() ** 2).mean())

    # Calibration: bucket direction_prob into deciles and check hit rate
    bins   = np.arange(0, 1.1, 0.1)
    labels = [f"{int(b*100)}-{int(b*100+10)}%" for b in bins[:-1]]
    df_bt  = df_bt.copy()
    df_bt["prob_bucket"] = pd.cut(df_bt["direction_prob"], bins=bins, labels=labels, right=False)
    df_bt["dir_hit"] = (df_bt["direction_pred"] == df_bt["direction_actual"]).astype(int)
    calibration = (
        df_bt.groupby("prob_bucket", observed=False)["dir_hit"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "hit_rate", "count": "n"})
    )

    return {
        "n_predictions":       len(df_bt),
        "directional_accuracy": round(float(dir_acc), 4),
        "MAE":                 round(float(mae), 4),
        "RMSE":                round(float(rmse), 4),
        "ci_80_coverage":      round(float(in_ci), 4),
        "naive_dir_accuracy":  round(float(naive_dir), 4),
        "naive_MAE":           round(float(naive_mae), 4),
        "naive_RMSE":          round(float(naive_rmse), 4),
        "beats_baseline_dir":  bool(dir_acc > naive_dir),
        "beats_baseline_mae":  bool(mae < naive_mae),
        "calibration":         calibration,
    }


def print_report(metrics: dict):
    print("\n" + "=" * 60)
    print("  BACKTEST REPORT")
    print("=" * 60)
    print(f"  Predictions:          {metrics['n_predictions']}")
    print()
    print(f"  Directional accuracy: {metrics['directional_accuracy']:.1%}  "
          f"(naive: {metrics['naive_dir_accuracy']:.1%})  "
          f"{'BEATS' if metrics['beats_baseline_dir'] else 'DOES NOT BEAT'} baseline")
    print()
    print(f"  MAE:                  {metrics['MAE']:.4f}  "
          f"(naive: {metrics['naive_MAE']:.4f})  "
          f"{'BEATS' if metrics['beats_baseline_mae'] else 'DOES NOT BEAT'} baseline")
    print(f"  RMSE:                 {metrics['RMSE']:.4f}  "
          f"(naive: {metrics['naive_RMSE']:.4f})")
    print()
    print(f"  80% CI coverage:      {metrics['ci_80_coverage']:.1%}  (target: 80%)")
    print()
    print("  Calibration table (direction probability buckets vs hit rate):")
    cal = metrics["calibration"]
    cal_filtered = cal[cal["n"] > 0]
    if len(cal_filtered):
        for bucket, row in cal_filtered.iterrows():
            bar = "#" * int(row["hit_rate"] * 20)
            print(f"    {bucket:<12} n={int(row['n']):>4}  hit={row['hit_rate']:.2f}  {bar}")
    print()

    if not metrics["beats_baseline_dir"]:
        print("  [NOTE] Model does NOT beat directional baseline.")
        print("         Consider: more features, longer history, or different model.")
    print("=" * 60 + "\n")

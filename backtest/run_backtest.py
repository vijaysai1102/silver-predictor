"""
Run the full walk-forward backtest and save results.
Usage: python backtest/run_backtest.py
"""

import sys
import os
import json
import logging
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

from data.fetchers import fetch_all, latest_values
from quant.features import build_feature_matrix, get_feature_cols
from backtest.walk_forward import run_walk_forward, compute_metrics, print_report


def main():
    fred_key = os.environ.get("FRED_API_KEY", "")
    logger.info("Fetching data for backtest ...")
    all_data = fetch_all(fred_api_key=fred_key or None)

    logger.info("Building feature matrix ...")
    df = build_feature_matrix(all_data)
    feature_cols = get_feature_cols(df)
    logger.info("Feature matrix: %d rows, %d features", len(df), len(feature_cols))

    # Use last 3 years as the test window; train on everything before that
    three_years_ago = df.index[-1] - pd.DateOffset(years=3)
    min_train = max(500, len(df[df.index < three_years_ago]))
    logger.info(
        "Walk-forward: min_train=%d, test window=%d rows",
        min_train, len(df) - min_train,
    )

    logger.info("Running walk-forward backtest (this may take a few minutes) ...")
    df_bt = run_walk_forward(df, feature_cols, min_train_rows=min_train)

    metrics = compute_metrics(df_bt)
    print_report(metrics)

    # Compute out-of-sample CI calibration parameters
    oos_return_errors = df_bt["pred_return"] - df_bt["actual_return"]
    oos_std = float(oos_return_errors.std())
    # Empirical 80th-pct of |error| gives a properly calibrated 80% CI half-width
    ci_half_width_return = float(oos_return_errors.abs().quantile(0.80))
    logger.info(
        "OOS residual std: %.4f | empirical 80th-pct |error|: %.4f",
        oos_std, ci_half_width_return,
    )

    # Save results
    out_dir = Path(__file__).parent
    df_bt.reset_index().to_csv(out_dir / "backtest_results.csv", index=False)

    # Save metrics as JSON (exclude calibration table — not JSON serializable)
    metrics_json = {k: v for k, v in metrics.items() if k != "calibration"}
    metrics_json["oos_residual_std"] = round(oos_std, 6)
    metrics_json["ci_half_width_return"] = round(ci_half_width_return, 6)
    with open(out_dir / "backtest_metrics.json", "w") as f:
        json.dump(metrics_json, f, indent=2)

    logger.info("Results saved to backtest/backtest_results.csv and backtest_metrics.json")

    # Train final model on all data with calibrated CI
    from quant.ensemble import QuantEnsemble
    final_model = QuantEnsemble()
    X_all = df[feature_cols]
    y_all = df["target_return"]
    final_model.fit(X_all, y_all, last_close=float(df["target_close"].iloc[-2]))
    final_model.calibrate_ci(oos_std, ci_half_width_return)

    imp = final_model.feature_importance().head(15)
    print("\nTop 15 feature importances (full training set):")
    for feat, score in imp.items():
        bar = "#" * int(score / imp.max() * 30)
        print(f"  {feat:<30} {score:.4f}  {bar}")

    ci_dollar = 75 * ci_half_width_return
    print(f"\nCI calibration: in-sample std={final_model.residual_std:.4f}, oos std={oos_std:.4f}")
    print(f"Empirical 80th-pct error: {ci_half_width_return:.4f} -> +/-${ci_dollar:.2f} on a $75 silver price")
    print(f"(This gives a properly calibrated 80% CI in live predictions)")


if __name__ == "__main__":
    main()

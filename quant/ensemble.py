"""
Gradient-boosting ensemble for next-day silver price prediction.
Uses XGBoost with walk-forward (expanding-window) cross-validation.
No look-ahead bias: features for day T predict day T+1 close.
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)


def _make_model(n_estimators: int = 120, seed: int = 42):
    try:
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.7,
            colsample_bytree=0.6,
            reg_alpha=0.5,
            reg_lambda=3.0,
            min_child_weight=10,
            random_state=seed,
            verbosity=0,
            n_jobs=-1,
        )
    except ImportError:
        from lightgbm import LGBMRegressor
        logger.warning("xgboost not available, falling back to lightgbm")
        return LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.7,
            colsample_bytree=0.6,
            reg_alpha=0.5,
            reg_lambda=3.0,
            min_child_weight=10,
            random_state=seed,
            verbose=-1,
            n_jobs=-1,
        )


class QuantEnsemble:
    """
    Wrapper around an XGBoost model.
    predict() returns: point_estimate, lower_80, upper_80, direction_prob
    """

    def __init__(self):
        self.model = None
        self.feature_cols: list[str] = []
        self.residual_std: float = 0.0         # in-sample residual std (optimistic)
        self.calibrated_std: float = 0.0       # out-of-sample std set after backtest
        self.ci_half_width_return: float | None = None  # empirical 80th-pct error (best CI estimator)
        self.is_fitted: bool = False
        self.last_close: float = 0.0           # the most recent silver close we trained on

    def fit(self, X: pd.DataFrame, y: pd.Series, last_close: float):
        self.model = _make_model()
        self.feature_cols = list(X.columns)
        self.model.fit(X, y)
        preds = self.model.predict(X)
        self.residual_std = float(np.std(y.values - preds))
        self.calibrated_std = self.residual_std   # will be overwritten if backtest data available
        self.is_fitted = True
        self.last_close = last_close
        logger.debug(
            "QuantEnsemble fitted on %d rows; residual_std=%.4f",
            len(X), self.residual_std,
        )

    def calibrate_ci(self, oos_residual_std: float, ci_half_width_return: Optional[float] = None):
        """
        Set CI parameters from out-of-sample backtest residuals.
        oos_residual_std       : std of (pred_return - actual_return) across backtest
        ci_half_width_return   : empirical 80th-pct of |pred_return - actual_return|
                                 If provided, used directly for the CI instead of z*sigma.
        """
        self.calibrated_std = float(oos_residual_std)
        self.ci_half_width_return = float(ci_half_width_return) if ci_half_width_return is not None else None
        logger.info(
            "CI calibrated: oos_std=%.4f, empirical_80pct=%.4f (in-sample=%.4f)",
            oos_residual_std,
            ci_half_width_return or (1.2816 * oos_residual_std),
            self.residual_std,
        )

    def predict(
        self,
        X_row: pd.DataFrame,
        current_close: Optional[float] = None,
    ) -> dict:
        """
        Predict next-day return, convert to price, derive 80% CI and directional prob.
        X_row: single-row DataFrame with the same feature columns used in training.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet")

        # Align columns — fill any missing with 0
        X_aligned = X_row.reindex(columns=self.feature_cols, fill_value=0.0)
        pred_return = float(self.model.predict(X_aligned)[0])

        close = current_close if current_close is not None else self.last_close
        pred_close = close * (1 + pred_return)

        # 80% CI: prefer empirical 80th-pct half-width from backtest (best calibration)
        if self.ci_half_width_return is not None:
            half = self.ci_half_width_return
        else:
            ci_std = self.calibrated_std if self.calibrated_std > self.residual_std else self.residual_std
            half = 1.2816 * ci_std
        lower = close * (1 + pred_return - half)
        upper = close * (1 + pred_return + half)

        # Directional probability using normal CDF
        from scipy.stats import norm
        ci_std = self.calibrated_std if self.calibrated_std > self.residual_std else self.residual_std
        direction_prob = float(norm.sf(0, loc=pred_return, scale=ci_std + 1e-9))

        return {
            "pred_return":    pred_return,
            "pred_close":     round(pred_close, 4),
            "ci_lower_80":    round(lower, 4),
            "ci_upper_80":    round(upper, 4),
            "direction_prob": round(direction_prob, 4),
            "direction":      "up" if pred_return >= 0 else "down",
            "residual_std":   round(self.residual_std, 6),
        }

    def feature_importance(self) -> pd.Series:
        if not self.is_fitted:
            return pd.Series(dtype=float)
        imp = self.model.feature_importances_
        return pd.Series(imp, index=self.feature_cols).sort_values(ascending=False)

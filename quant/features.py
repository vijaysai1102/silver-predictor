"""
Feature engineering for the silver-price prediction model.
All features are constructed strictly from data available at close of day T,
predicting the next-day (T+1) close.  No look-ahead bias.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def _safe_pct(series: pd.Series) -> pd.Series:
    return series.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    hl  = high - low
    hc  = (high - close.shift(1)).abs()
    lc  = (low  - close.shift(1)).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def build_feature_matrix(
    all_data: dict,
    target_col: str = "silver",
) -> pd.DataFrame:
    """
    Merge all data sources, engineer features, attach the next-day target.
    Returns a DataFrame with features + 'target_return' + 'target_close'.
    Rows with NaN targets or features are dropped.
    """
    # ── 1. Pull close series ─────────────────────────────────────────────────
    series = {}
    for name, rec in all_data.items():
        if rec["status"] != "ok":
            continue
        d = rec["data"]
        if isinstance(d, pd.DataFrame) and "Close" in d.columns:
            series[name] = d["Close"].rename(name)
        elif isinstance(d, pd.Series):
            series[name] = d.rename(name)

    if target_col not in series:
        raise ValueError(f"Target '{target_col}' not found in fetched data")

    # Align on the union of indices, then restrict to silver's actual trading days.
    # FRED series (e.g., monthly CPI) introduce non-trading dates that break rolling windows.
    df = pd.concat(series.values(), axis=1)
    df.sort_index(inplace=True)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    # Only keep rows where silver has a value (i.e., market was open)
    df = df[df[target_col].notna()].copy()

    silver = df[target_col]

    # ── 2. Target: next-day return & close (shift -1) ────────────────────────
    df["target_return"] = silver.pct_change(fill_method=None).shift(-1)   # return on day T+1
    df["target_close"]  = silver.shift(-1)                # actual close on T+1

    # ── 3. Silver-specific features ──────────────────────────────────────────
    df["silver_ret1"]   = _safe_pct(silver)
    df["silver_ret2"]   = silver.pct_change(2, fill_method=None)
    df["silver_ret5"]   = silver.pct_change(5, fill_method=None)
    df["silver_ret10"]  = silver.pct_change(10, fill_method=None)
    df["silver_ret21"]  = silver.pct_change(21, fill_method=None)

    df["silver_sma20"]  = _sma(silver, 20)
    df["silver_sma50"]  = _sma(silver, 50)
    df["silver_sma200"] = _sma(silver, 200)

    df["silver_vs_sma20"]  = silver / df["silver_sma20"] - 1
    df["silver_vs_sma50"]  = silver / df["silver_sma50"] - 1
    df["silver_vs_sma200"] = silver / df["silver_sma200"] - 1

    df["silver_rsi14"]  = _rsi(silver, 14)
    df["silver_vol10"]  = _safe_pct(silver).rolling(10).std()
    df["silver_vol21"]  = _safe_pct(silver).rolling(21).std()

    if "silver" in all_data and all_data["silver"]["status"] == "ok":
        sv_df = all_data["silver"]["data"]
        if isinstance(sv_df, pd.DataFrame) and "High" in sv_df.columns:
            sv_df.index = pd.to_datetime(sv_df.index).tz_localize(None)
            sv_df = sv_df.reindex(df.index)
            df["silver_atr14"] = _atr(sv_df["High"], sv_df["Low"], sv_df["Close"])

    # Momentum z-score: (current - 21d mean) / 21d std
    r1 = _safe_pct(silver)
    df["silver_mom_z21"] = (r1 - r1.rolling(21).mean()) / r1.rolling(21).std().replace(0, np.nan)

    # ── 4. Gold/silver ratio ──────────────────────────────────────────────────
    if "gold" in series:
        gold = df["gold"]
        df["gs_ratio"]        = gold / silver.replace(0, np.nan)
        df["gs_ratio_vs20"]   = df["gs_ratio"] / df["gs_ratio"].rolling(20).mean() - 1
        df["gold_ret1"]       = _safe_pct(gold)
        df["gold_ret5"]       = gold.pct_change(5, fill_method=None)

    # ── 5. Dollar (DXY) ───────────────────────────────────────────────────────
    if "dxy" in series:
        dxy = df["dxy"]
        df["dxy_ret1"]        = _safe_pct(dxy)
        df["dxy_ret5"]        = dxy.pct_change(5, fill_method=None)
        df["dxy_vs_sma20"]    = dxy / _sma(dxy, 20) - 1

    # ── 6. Interest rate proxies (prefer FRED, fall back to us10y) ────────────
    rate_source = None
    if "dfii10" in series:
        rate_source = df["dfii10"]
        df["real_rate"]       = rate_source
        df["real_rate_chg1"]  = rate_source.diff(1)
        df["real_rate_chg5"]  = rate_source.diff(5)
    elif "dgs10" in series:
        rate_source = df["dgs10"]
        df["real_rate"]       = rate_source
        df["real_rate_chg1"]  = rate_source.diff(1)
        df["real_rate_chg5"]  = rate_source.diff(5)
    elif "us10y" in series:
        rate_source = df["us10y"]
        df["nom_rate"]        = rate_source
        df["nom_rate_chg1"]   = rate_source.diff(1)
        df["nom_rate_chg5"]   = rate_source.diff(5)

    # Inflation expectations
    if "t10yie" in series:
        t10 = df["t10yie"]
        df["infl_exp"]        = t10
        df["infl_exp_chg5"]   = t10.diff(5)

    # ── 7. Industrial / macro proxies ─────────────────────────────────────────
    if "copper" in series:
        cu = df["copper"]
        df["copper_ret1"]     = _safe_pct(cu)
        df["copper_ret5"]     = cu.pct_change(5, fill_method=None)
        df["cu_vs_sma20"]     = cu / _sma(cu, 20) - 1
        # Copper/silver ratio — signals industrial demand relative to precious-metal demand
        df["cu_si_ratio"]     = cu / silver.replace(0, np.nan)
        df["cu_si_ratio_vs20"]= df["cu_si_ratio"] / df["cu_si_ratio"].rolling(20).mean() - 1

    if "sp500" in series:
        sp = df["sp500"]
        df["sp500_ret1"]      = _safe_pct(sp)
        df["sp500_ret5"]      = sp.pct_change(5, fill_method=None)
        df["sp500_vs_sma50"]  = sp / _sma(sp, 50) - 1

    # ── 8. Day-of-week dummies (Mon–Thu; Fri is reference) ───────────────────
    df["dow_mon"] = (df.index.dayofweek == 0).astype(int)
    df["dow_tue"] = (df.index.dayofweek == 1).astype(int)
    df["dow_wed"] = (df.index.dayofweek == 2).astype(int)
    df["dow_thu"] = (df.index.dayofweek == 3).astype(int)

    # ── 9. Drop non-feature columns that would leak the target ───────────────
    drop_raw = [
        "silver", "silver_etf", "gold", "dxy", "sp500", "copper",
        "us10y", "dgs10", "dfii10", "t10yie", "cpi",
    ]
    df.drop(columns=[c for c in drop_raw if c in df.columns], inplace=True)

    # ── 10. Drop rows where target is NaN (last row after shift) ─────────────
    df.dropna(subset=["target_return", "target_close"], inplace=True)

    # ── 11. Forward-fill sparse FRED series (monthly/weekly), then drop NaN features ──
    df.ffill(inplace=True)
    df.dropna(inplace=True)

    logger.info("Feature matrix: %d rows x %d features", len(df), df.shape[1] - 2)
    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return the list of feature column names (excludes target columns)."""
    return [c for c in df.columns if not c.startswith("target_")]

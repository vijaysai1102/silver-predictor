"""
Data fetchers for all market and macro inputs.
All functions return a dict with keys: data (DataFrame or scalar), status (ok/error/skipped),
source, and last_updated.  Failures never raise — they return status=error with a message.
"""

import os
import time
import logging
from datetime import date, timedelta
from typing import Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ── yfinance tickers ──────────────────────────────────────────────────────────
YFINANCE_TICKERS = {
    "silver":    "SI=F",
    "silver_etf": "SLV",
    "gold":      "GC=F",
    "dxy":       "DX-Y.NYB",
    "sp500":     "^GSPC",
    "copper":    "HG=F",
    "us10y":     "^TNX",
}

# ── FRED series ───────────────────────────────────────────────────────────────
FRED_SERIES = {
    "dgs10":   "DGS10",    # nominal 10-yr yield
    "dfii10":  "DFII10",   # real 10-yr yield (TIPS)
    "t10yie":  "T10YIE",   # 10-yr breakeven inflation
    "cpi":     "CPIAUCSL", # CPI all-urban consumers
}

_DEFAULT_PERIOD_DAYS = 365 * 5   # 5 years of history by default


def _retry(fn, retries: int = 3, backoff: float = 2.0):
    """Call fn(); on exception retry with exponential backoff."""
    last_exc = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt < retries - 1:
                sleep_s = backoff ** attempt
                logger.warning("Attempt %d failed: %s — retrying in %.1fs", attempt + 1, exc, sleep_s)
                time.sleep(sleep_s)
    raise last_exc


def fetch_yfinance(
    name: str,
    ticker: str,
    period_days: int = _DEFAULT_PERIOD_DAYS,
) -> dict[str, Any]:
    """Fetch OHLCV history for one ticker via yfinance."""
    try:
        import yfinance as yf

        start = (date.today() - timedelta(days=period_days)).isoformat()
        end   = (date.today() + timedelta(days=1)).isoformat()

        def _fetch():
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if df.empty:
                raise ValueError(f"No data returned for {ticker}")
            return df

        df = _retry(_fetch)
        df.index = pd.to_datetime(df.index)
        # Flatten multi-level columns that yfinance sometimes returns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.sort_index(inplace=True)
        last_date = df.index[-1].date().isoformat()
        return {
            "name": name,
            "ticker": ticker,
            "status": "ok",
            "source": "yfinance",
            "data": df,
            "last_updated": last_date,
            "rows": len(df),
        }
    except Exception as exc:
        logger.error("yfinance fetch failed for %s (%s): %s", name, ticker, exc)
        return {
            "name": name,
            "ticker": ticker,
            "status": "error",
            "source": "yfinance",
            "data": pd.DataFrame(),
            "last_updated": None,
            "error": str(exc),
        }


def fetch_all_yfinance(period_days: int = _DEFAULT_PERIOD_DAYS) -> dict[str, dict]:
    """Fetch all yfinance tickers. Returns dict keyed by logical name."""
    results = {}
    for name, ticker in YFINANCE_TICKERS.items():
        logger.info("Fetching %s (%s) …", name, ticker)
        results[name] = fetch_yfinance(name, ticker, period_days)
        time.sleep(0.5)   # be polite; avoid hammering yfinance
    return results


def fetch_fred(
    name: str,
    series_id: str,
    period_days: int = _DEFAULT_PERIOD_DAYS,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Fetch a FRED time series.  api_key falls back to FRED_API_KEY env var."""
    api_key = api_key or os.environ.get("FRED_API_KEY", "")
    if not api_key:
        return {
            "name": name,
            "series_id": series_id,
            "status": "skipped",
            "source": "FRED",
            "data": pd.Series(dtype=float),
            "last_updated": None,
            "error": "FRED_API_KEY not set",
        }
    try:
        from fredapi import Fred

        fred = Fred(api_key=api_key)
        start = (date.today() - timedelta(days=period_days)).isoformat()

        def _fetch():
            s = fred.get_series(series_id, observation_start=start)
            if s is None or s.empty:
                raise ValueError(f"No data for FRED series {series_id}")
            return s

        s = _retry(_fetch)
        s = s.dropna().sort_index()
        last_date = s.index[-1].date().isoformat()
        return {
            "name": name,
            "series_id": series_id,
            "status": "ok",
            "source": "FRED",
            "data": s,
            "last_updated": last_date,
            "rows": len(s),
        }
    except Exception as exc:
        logger.error("FRED fetch failed for %s (%s): %s", name, series_id, exc)
        return {
            "name": name,
            "series_id": series_id,
            "status": "error",
            "source": "FRED",
            "data": pd.Series(dtype=float),
            "last_updated": None,
            "error": str(exc),
        }


def fetch_all_fred(period_days: int = _DEFAULT_PERIOD_DAYS, api_key: str | None = None) -> dict[str, dict]:
    """Fetch all FRED series. Returns dict keyed by logical name."""
    results = {}
    for name, series_id in FRED_SERIES.items():
        logger.info("Fetching FRED %s (%s) …", name, series_id)
        results[name] = fetch_fred(name, series_id, period_days, api_key)
        time.sleep(0.3)
    return results


def fetch_all(period_days: int = _DEFAULT_PERIOD_DAYS, fred_api_key: str | None = None) -> dict[str, dict]:
    """Fetch everything. Returns combined dict keyed by logical name."""
    yf_data   = fetch_all_yfinance(period_days)
    fred_data = fetch_all_fred(period_days, fred_api_key)
    return {**yf_data, **fred_data}


def latest_values(all_data: dict[str, dict]) -> dict[str, float | None]:
    """
    Extract the most recent scalar value from each data source.
    For OHLCV frames, returns the latest Close.  For FRED series, the last value.
    """
    out = {}
    for name, rec in all_data.items():
        if rec["status"] != "ok":
            out[name] = None
            continue
        d = rec["data"]
        try:
            if isinstance(d, pd.DataFrame):
                out[name] = float(d["Close"].dropna().iloc[-1])
            elif isinstance(d, pd.Series):
                out[name] = float(d.dropna().iloc[-1])
            else:
                out[name] = None
        except Exception:
            out[name] = None
    return out

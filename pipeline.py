"""
pipeline.py — daily silver-price prediction pipeline.

Steps:
  1. Fetch latest market data
  2. Score yesterday's prediction against the actual close
  3. Build feature matrix, run quant ensemble
  4. Run 5 specialist agents (sequentially through Groq wrapper)
  5. Orchestrate final prediction
  6. Generate reporter commentary
  7. Export JSON/CSV for the site
  8. Log run metadata

Usage:
  python pipeline.py                   # full run
  python pipeline.py --quant-only      # disable LLM calls; quant-only mode
  python pipeline.py --dry-run         # run everything but don't write to DB or site/
"""

import sys
import os
import json
import argparse
import logging
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

EXPORT_DIR = Path(__file__).parent / "docs" / "data"


def next_trading_day(from_date: date) -> date:
    """Return the next weekday after from_date (skips weekends only; no holidays)."""
    d = from_date + timedelta(days=1)
    while d.weekday() >= 5:   # 5=Sat, 6=Sun
        d += timedelta(days=1)
    return d


def run(quant_only: bool = False, dry_run: bool = False) -> dict[str, Any]:
    start_ts = time.monotonic()
    today = date.today().isoformat()
    target_date = next_trading_day(date.today()).isoformat()

    logger.info("=" * 60)
    logger.info("Silver Predictor Pipeline — %s", today)
    logger.info("Predicting: %s | quant_only=%s | dry_run=%s", target_date, quant_only, dry_run)
    logger.info("=" * 60)

    from db.schema import (
        init_db, upsert_prediction, upsert_actual, score_predictions,
        rolling_accuracy, db_get_cache, db_store_cache,
        get_latest_prediction, get_prediction_history,
    )

    # ── 0. Init DB ────────────────────────────────────────────────────────────
    init_db()

    cache_fn  = db_get_cache  if not dry_run else None
    store_fn  = db_store_cache if not dry_run else None

    # ── 1. Fetch data ─────────────────────────────────────────────────────────
    logger.info("[1/8] Fetching market data ...")
    from data.fetchers import fetch_all, latest_values
    fred_key = os.environ.get("FRED_API_KEY", "")
    all_data = fetch_all(fred_api_key=fred_key or None)
    vals = latest_values(all_data)
    silver_close = vals.get("silver")
    if not silver_close:
        logger.error("Silver close unavailable — aborting")
        return {"status": "error", "error": "silver_close_unavailable"}
    slv_close = vals.get("silver_etf")

    logger.info("Silver (SI=F) close: $%.4f", silver_close)
    if slv_close:
        logger.info("Silver ETF (SLV) close: $%.4f", slv_close)

    # ── 2. Record today's actual, score yesterday's prediction ───────────────
    logger.info("[2/8] Storing actual close and scoring yesterday's prediction ...")
    if not dry_run:
        upsert_actual(today, silver_close, slv_actual_close=slv_close)
        scored = score_predictions()
        logger.info("Scored %d prediction(s)", len(scored))

    # ── 3. Feature engineering + quant ensemble ───────────────────────────────
    logger.info("[3/8] Building features and running quant ensemble ...")
    from quant.features import build_feature_matrix, get_feature_cols
    from quant.ensemble import QuantEnsemble
    import pandas as pd, json as _json

    try:
        df = build_feature_matrix(all_data)
    except Exception as e:
        logger.error("Feature matrix failed: %s — quant fallback", e)
        df = pd.DataFrame()

    # Load calibration from backtest if available
    bt_metrics_path = Path(__file__).parent / "backtest" / "backtest_metrics.json"
    oos_std = 0.03
    ci_half_width = None
    if bt_metrics_path.exists():
        with open(bt_metrics_path) as f:
            bt = _json.load(f)
            oos_std = bt.get("oos_residual_std", 0.03)
            ci_half_width = bt.get("ci_half_width_return")

    quant_model = QuantEnsemble()
    quant_prediction = None

    if not df.empty:
        feature_cols = get_feature_cols(df)
        X_train = df[feature_cols]
        y_train = df["target_return"]
        last_close_train = float(df["target_close"].iloc[-2]) if len(df) > 1 else silver_close

        quant_model.fit(X_train, y_train, last_close=last_close_train)
        quant_model.calibrate_ci(oos_std, ci_half_width)

        X_latest = df[feature_cols].iloc[[-1]]
        quant_prediction = quant_model.predict(X_latest, current_close=silver_close)
        logger.info(
            "Quant: close=$%.4f dir=%s prob=%.1f%% CI=[%.2f, %.2f]",
            quant_prediction["pred_close"],
            quant_prediction["direction"],
            quant_prediction["direction_prob"] * 100,
            quant_prediction["ci_lower_80"],
            quant_prediction["ci_upper_80"],
        )
    else:
        # Flat fallback: predict no change
        quant_prediction = {
            "pred_return":    0.0,
            "pred_close":     silver_close,
            "ci_lower_80":    silver_close * (1 - 1.2816 * oos_std),
            "ci_upper_80":    silver_close * (1 + 1.2816 * oos_std),
            "direction_prob": 0.5,
            "direction":      "up",
            "residual_std":   oos_std,
        }
        logger.warning("Feature matrix empty — using flat quant fallback")

    # ── 3b. SLV quant model ───────────────────────────────────────────────────
    slv_quant_prediction = None
    if slv_close:
        try:
            df_slv = build_feature_matrix(all_data, target_col="silver_etf")
        except Exception as e:
            logger.warning("SLV feature matrix failed: %s", e)
            df_slv = pd.DataFrame()

        slv_model = QuantEnsemble()
        if not df_slv.empty:
            slv_feature_cols = get_feature_cols(df_slv)
            X_slv_train = df_slv[slv_feature_cols]
            y_slv_train = df_slv["target_return"]
            slv_last_close = float(df_slv["target_close"].iloc[-2]) if len(df_slv) > 1 else slv_close
            slv_model.fit(X_slv_train, y_slv_train, last_close=slv_last_close)
            slv_model.calibrate_ci(oos_std, ci_half_width)
            X_slv_latest = df_slv[slv_feature_cols].iloc[[-1]]
            slv_quant_prediction = slv_model.predict(X_slv_latest, current_close=slv_close)
            logger.info(
                "SLV Quant: close=$%.4f dir=%s CI=[%.2f, %.2f]",
                slv_quant_prediction["pred_close"],
                slv_quant_prediction["direction"],
                slv_quant_prediction["ci_lower_80"],
                slv_quant_prediction["ci_upper_80"],
            )
        else:
            slv_quant_prediction = {
                "pred_return":    0.0,
                "pred_close":     slv_close,
                "ci_lower_80":    slv_close * (1 - 1.2816 * oos_std),
                "ci_upper_80":    slv_close * (1 + 1.2816 * oos_std),
                "direction_prob": 0.5,
                "direction":      "up",
                "residual_std":   oos_std,
            }
            logger.warning("SLV feature matrix empty — using flat quant fallback")

    # ── 4. Run specialist agents ──────────────────────────────────────────────
    logger.info("[4/8] Running specialist agents ...")
    specialist_signals: list[dict] = []

    if quant_only or not os.environ.get("GROQ_API_KEY"):
        logger.info("QUANT-ONLY MODE — skipping all LLM agents")
    else:
        from agents.specialists import (
            precious_metals_agent, dollar_rates_agent, industrial_demand_agent,
            macro_sentiment_agent, technical_agent,
        )
        from quant.features import build_feature_matrix

        # Extract latest feature values for agent prompts
        def _v(key, default=None):
            return vals.get(key) or default

        silver_sma20  = float(df["silver_sma20"].iloc[-1])  if not df.empty and "silver_sma20"  in df.columns else None
        silver_vs_sma20  = float(df["silver_vs_sma20"].iloc[-1])  if not df.empty else 0.0
        silver_vs_sma50  = float(df["silver_vs_sma50"].iloc[-1])  if not df.empty else 0.0
        silver_vs_sma200 = float(df["silver_vs_sma200"].iloc[-1]) if not df.empty else 0.0
        silver_rsi14     = float(df["silver_rsi14"].iloc[-1])     if not df.empty else 50.0
        silver_ret1      = float(df["silver_ret1"].iloc[-1])      if not df.empty else 0.0
        silver_ret5      = float(df["silver_ret5"].iloc[-1])      if not df.empty else 0.0
        silver_ret10     = float(df["silver_ret10"].iloc[-1])     if not df.empty else 0.0
        silver_vol10     = float(df["silver_vol10"].iloc[-1])     if not df.empty else 0.0
        silver_vol21     = float(df["silver_vol21"].iloc[-1])     if not df.empty else 0.0
        silver_atr14     = float(df["silver_atr14"].iloc[-1])     if not df.empty and "silver_atr14" in df.columns else None
        gs_ratio         = float(df["gs_ratio"].iloc[-1])         if not df.empty and "gs_ratio"     in df.columns else None
        gs_ratio_vs20    = float(df["gs_ratio_vs20"].iloc[-1])    if not df.empty and "gs_ratio_vs20" in df.columns else 0.0
        gold_ret5        = float(df["gold_ret5"].iloc[-1])        if not df.empty and "gold_ret5"    in df.columns else 0.0
        gold_ret1        = float(df["gold_ret1"].iloc[-1])        if not df.empty and "gold_ret1"    in df.columns else 0.0
        dxy_ret5         = float(df["dxy_ret5"].iloc[-1])         if not df.empty and "dxy_ret5"     in df.columns else 0.0
        dxy_vs_sma20     = float(df["dxy_vs_sma20"].iloc[-1])     if not df.empty and "dxy_vs_sma20" in df.columns else 0.0
        real_rate        = float(df["real_rate"].iloc[-1])        if not df.empty and "real_rate"    in df.columns else None
        real_rate_chg1   = float(df["real_rate_chg1"].iloc[-1])   if not df.empty and "real_rate_chg1" in df.columns else None
        real_rate_chg5   = float(df["real_rate_chg5"].iloc[-1])   if not df.empty and "real_rate_chg5" in df.columns else None
        nom_rate         = float(df["nom_rate"].iloc[-1])         if not df.empty and "nom_rate"     in df.columns else None
        copper_ret5      = float(df["copper_ret5"].iloc[-1])      if not df.empty and "copper_ret5"  in df.columns else 0.0
        cu_vs_sma20      = float(df["cu_vs_sma20"].iloc[-1])      if not df.empty and "cu_vs_sma20"  in df.columns else 0.0
        cu_si_ratio      = float(df["cu_si_ratio"].iloc[-1])      if not df.empty and "cu_si_ratio"  in df.columns else 0.0
        cu_si_ratio_vs20 = float(df["cu_si_ratio_vs20"].iloc[-1]) if not df.empty and "cu_si_ratio_vs20" in df.columns else 0.0
        sp500_ret5       = float(df["sp500_ret5"].iloc[-1])       if not df.empty and "sp500_ret5"   in df.columns else 0.0
        sp500_vs_sma50   = float(df["sp500_vs_sma50"].iloc[-1])   if not df.empty and "sp500_vs_sma50" in df.columns else 0.0
        infl_exp         = float(df["infl_exp"].iloc[-1])         if not df.empty and "infl_exp"     in df.columns else None
        infl_exp_chg5    = float(df["infl_exp_chg5"].iloc[-1])    if not df.empty and "infl_exp_chg5" in df.columns else None

        ck = {"run_date": today, "db_cache_fn": cache_fn, "db_store_fn": store_fn}

        s1 = precious_metals_agent(
            silver_close=silver_close,
            gold_close=_v("gold", 0),
            gs_ratio=gs_ratio or (_v("gold", 0) / silver_close if silver_close else 0),
            gs_ratio_vs20=gs_ratio_vs20,
            gold_ret5=gold_ret5,
            silver_vs_sma50=silver_vs_sma50,
            **ck,
        )
        specialist_signals.append(s1)
        logger.info("PreciousMetalsAgent: %s", s1.get("status"))

        s2 = dollar_rates_agent(
            dxy_close=_v("dxy", 0),
            dxy_ret5=dxy_ret5,
            dxy_vs_sma20=dxy_vs_sma20,
            real_rate=real_rate,
            real_rate_chg5=real_rate_chg5,
            nom_rate=nom_rate,
            **ck,
        )
        specialist_signals.append(s2)
        logger.info("DollarRatesAgent: %s", s2.get("status"))

        s3 = industrial_demand_agent(
            copper_close=_v("copper", 0),
            copper_ret5=copper_ret5,
            cu_vs_sma20=cu_vs_sma20,
            cu_si_ratio=cu_si_ratio,
            cu_si_ratio_vs20=cu_si_ratio_vs20,
            sp500_ret5=sp500_ret5,
            sp500_vs_sma50=sp500_vs_sma50,
            **ck,
        )
        specialist_signals.append(s3)
        logger.info("IndustrialDemandAgent: %s", s3.get("status"))

        s4 = macro_sentiment_agent(
            infl_exp=infl_exp,
            infl_exp_chg5=infl_exp_chg5,
            sp500_ret5=sp500_ret5,
            sp500_vs_sma50=sp500_vs_sma50,
            silver_vol21=silver_vol21,
            **ck,
        )
        specialist_signals.append(s4)
        logger.info("MacroSentimentAgent: %s", s4.get("status"))

        s5 = technical_agent(
            silver_close=silver_close,
            silver_ret1=silver_ret1,
            silver_ret5=silver_ret5,
            silver_ret10=silver_ret10,
            silver_rsi14=silver_rsi14,
            silver_vs_sma20=silver_vs_sma20,
            silver_vs_sma50=silver_vs_sma50,
            silver_vs_sma200=silver_vs_sma200,
            silver_vol10=silver_vol10,
            silver_atr14=silver_atr14,
            **ck,
        )
        specialist_signals.append(s5)
        logger.info("TechnicalAgent: %s", s5.get("status"))

    ok_count   = sum(1 for s in specialist_signals if s.get("status") == "ok")
    skip_count = len(specialist_signals) - ok_count
    logger.info("Agents: %d OK, %d skipped", ok_count, skip_count)

    # ── 5. Orchestrate ────────────────────────────────────────────────────────
    logger.info("[5/8] Orchestrating final prediction ...")
    if quant_only or not specialist_signals or not os.environ.get("GROQ_API_KEY"):
        from agents.orchestrator import _quant_only_prediction
        final_pred = _quant_only_prediction(quant_prediction, silver_close)
    else:
        from agents.orchestrator import orchestrate
        final_pred = orchestrate(
            specialist_signals=specialist_signals,
            quant_prediction=quant_prediction,
            current_close=silver_close,
            run_date=today,
            db_cache_fn=cache_fn,
            db_store_fn=store_fn,
        )

    final_pred["prediction_date"] = today
    final_pred["target_date"]     = target_date
    logger.info(
        "Final SI=F: $%.4f %s (%.1f%%) CI=[%.2f, %.2f]",
        final_pred["predicted_close"],
        final_pred["direction"].upper(),
        final_pred["direction_prob"] * 100,
        final_pred["ci_lower_80"],
        final_pred["ci_upper_80"],
    )

    # Apply same % LLM adjustment to SLV (agents analyse macro/direction, not price level)
    if slv_quant_prediction:
        quant_anchor = final_pred.get("quant_anchor") or quant_prediction["pred_close"]
        if not final_pred.get("quant_only_mode", False) and quant_anchor:
            llm_ratio = (final_pred["predicted_close"] - quant_anchor) / quant_anchor
        else:
            llm_ratio = 0.0
        slv_anchor = slv_quant_prediction["pred_close"]
        slv_ci_half = (slv_quant_prediction["ci_upper_80"] - slv_quant_prediction["ci_lower_80"]) / 2
        slv_final_close = slv_anchor * (1 + llm_ratio)
        final_pred["slv_predicted_close"] = round(slv_final_close, 4)
        final_pred["slv_ci_lower_80"]     = round(slv_final_close - slv_ci_half, 4)
        final_pred["slv_ci_upper_80"]     = round(slv_final_close + slv_ci_half, 4)
        logger.info(
            "Final SLV:  $%.4f CI=[%.2f, %.2f] (llm_adj=%.2f%%)",
            slv_final_close,
            final_pred["slv_ci_lower_80"],
            final_pred["slv_ci_upper_80"],
            llm_ratio * 100,
        )
    else:
        final_pred["slv_predicted_close"] = None
        final_pred["slv_ci_lower_80"]     = None
        final_pred["slv_ci_upper_80"]     = None

    # ── 6. Generate commentary ────────────────────────────────────────────────
    logger.info("[6/8] Generating reporter commentary ...")
    if quant_only or not os.environ.get("GROQ_API_KEY"):
        commentary = {
            "status":     "skipped",
            "commentary": "Quant-only mode: no commentary available.",
            "one_liner":  f"Silver predicted {final_pred['direction']} to ${final_pred['predicted_close']:.2f} (quant-only).",
            "watch_list": [],
        }
    else:
        from agents.reporter import generate_report
        commentary = generate_report(
            final_prediction=final_pred,
            specialist_signals=specialist_signals,
            current_close=silver_close,
            run_date=today,
            db_cache_fn=cache_fn,
            db_store_fn=store_fn,
        )

    # ── 7. Store in DB ────────────────────────────────────────────────────────
    logger.info("[7/8] Writing to DB ...")
    run_metadata = {
        "quant_only":       quant_only or not bool(os.environ.get("GROQ_API_KEY")),
        "agents_ok":        ok_count,
        "agents_skipped":   skip_count,
        "silver_close":     silver_close,
        "slv_close":        slv_close,
        "elapsed_s":        round(time.monotonic() - start_ts, 1),
    }
    if not dry_run:
        upsert_prediction(final_pred, commentary, run_metadata)

    # ── 8. Export static JSON for site ───────────────────────────────────────
    logger.info("[8/8] Exporting site data ...")
    acc = rolling_accuracy(last_n=90)
    history = get_prediction_history(limit=90)
    bt_metrics = {}
    if bt_metrics_path.exists():
        with open(bt_metrics_path) as f:
            bt_metrics = _json.load(f)

    export = {
        "latest_prediction": {
            **final_pred,
            "commentary":  commentary.get("commentary", ""),
            "one_liner":   commentary.get("one_liner", ""),
            "watch_list":  commentary.get("watch_list", []),
        },
        "agent_signals": [
            {k: v for k, v in s.items() if k not in ("db_cache_fn", "db_store_fn")}
            for s in specialist_signals
        ],
        "rolling_accuracy":  acc,
        "backtest_metrics":  bt_metrics,
        "history":           history,
        "last_updated":      today,
        "run_metadata":      run_metadata,
    }

    if not dry_run:
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        with open(EXPORT_DIR / "predictions.json", "w") as f:
            json.dump(export, f, indent=2, default=str)
        logger.info("Exported to %s", EXPORT_DIR / "predictions.json")

    elapsed = round(time.monotonic() - start_ts, 1)
    mode_str = "QUANT-ONLY" if run_metadata["quant_only"] else "FULL MULTI-AGENT"
    logger.info("Pipeline complete in %.1fs — mode=%s", elapsed, mode_str)

    return {"status": "ok", "export": export, "elapsed_s": elapsed}


def main():
    parser = argparse.ArgumentParser(description="Silver price prediction pipeline")
    parser.add_argument("--quant-only", action="store_true",
                        help="Disable LLM agents; use quant model only")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run pipeline but don't write to DB or site/")
    args = parser.parse_args()
    result = run(quant_only=args.quant_only, dry_run=args.dry_run)
    if result["status"] != "ok":
        sys.exit(1)


if __name__ == "__main__":
    main()

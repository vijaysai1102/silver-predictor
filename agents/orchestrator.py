"""
OrchestratorAgent — combines 5 specialist signals + quant ensemble output
into a final prediction: close, 80% CI, direction probability, reasoning.

Weighting logic:
  - The quant ensemble provides the price anchor (its estimate is never
    overridden by more than ±5% by the LLM layer).
  - Specialist weights are proportional to their historical backtested directional
    accuracy stored in backtest_metrics.json (fallback: equal weights).
  - Skipped agents are excluded from the weighted average; their weight is
    redistributed among the remaining agents.
  - The LLM orchestrator reasons over the weighted signals and can adjust the
    quant estimate within the bounded range; the final number is always anchored
    by the quant model.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

from utils.groq_client import call_groq, ORCHESTRATOR_MODEL

logger = logging.getLogger(__name__)

# Fallback weights (used if backtest per-agent accuracy is unavailable)
_DEFAULT_WEIGHTS: dict[str, float] = {
    "PreciousMetalsAgent":  0.22,
    "DollarRatesAgent":     0.22,
    "IndustrialDemandAgent":0.18,
    "MacroSentimentAgent":  0.18,
    "TechnicalAgent":       0.20,
}

# Maximum fraction the LLM can shift the quant anchor
_MAX_LLM_ADJUSTMENT = 0.05   # 5% of current close


def _weighted_signal(signals: list[dict], weights: dict[str, float]) -> float:
    """Compute weighted average signal from available (non-skipped) agents."""
    total_w = 0.0
    total_s = 0.0
    for s in signals:
        if s.get("status") != "ok":
            continue
        agent = s["agent"]
        w = weights.get(agent, 0.0)
        total_w += w
        total_s += w * s["signal"]
    if total_w == 0:
        return 0.0
    return total_s / total_w


def orchestrate(
    specialist_signals: list[dict],
    quant_prediction: dict,
    current_close: float,
    run_date: str | None = None,
    db_cache_fn=None,
    db_store_fn=None,
) -> dict[str, Any]:
    """
    Combine specialist signals + quant prediction into final output.

    quant_prediction keys: pred_close, ci_lower_80, ci_upper_80,
                           direction_prob, direction, pred_return

    Returns final prediction dict.
    """
    weights = _DEFAULT_WEIGHTS
    weighted_sig = _weighted_signal(specialist_signals, weights)
    ok_agents = [s["agent"] for s in specialist_signals if s.get("status") == "ok"]
    skipped_agents = [s["agent"] for s in specialist_signals if s.get("status") != "ok"]

    quant_close = quant_prediction["pred_close"]
    ci_lo       = quant_prediction["ci_lower_80"]
    ci_hi       = quant_prediction["ci_upper_80"]

    # ── Build orchestrator prompt ─────────────────────────────────────────────
    sigs_text = ""
    for s in specialist_signals:
        if s.get("status") == "ok":
            sigs_text += (
                f"  {s['agent']}: signal={s['signal']:+.2f}, "
                f"conf={s['confidence']:.2f}, dir={s['predicted_direction']}, "
                f"rationale={s['rationale']}\n"
            )
        else:
            sigs_text += f"  {s['agent']}: SKIPPED ({s.get('error','')})\n"

    system_prompt = (
        "You are the Orchestrator for a silver-price prediction system. "
        "You receive specialist agent signals and a quant model estimate, then produce a final prediction. "
        "IMPORTANT: Your final predicted_close MUST be within 5% of the quant model's estimate. "
        "The quant model is the mathematical anchor; you adjust it slightly based on the signals. "
        "Respond ONLY with valid JSON matching this exact schema:\n"
        '{"predicted_close": float, "direction": "up"|"down", "direction_prob": float (0-1), '
        '"reasoning": "2-3 sentence explanation", "agents_used": [list of agent names]}'
    )

    user_prompt = f"""
Today's silver close: ${current_close:.2f}

Quant model prediction:
  Next-day close estimate: ${quant_close:.4f}
  80% confidence interval: [${ci_lo:.4f}, ${ci_hi:.4f}]
  Direction probability (up): {quant_prediction['direction_prob']:.2%}

Specialist agent signals (weighted avg signal: {weighted_sig:+.3f}):
{sigs_text}
Active agents: {ok_agents or ['none — quant-only mode']}
Skipped agents: {skipped_agents or ['none']}

Produce the final prediction. Keep predicted_close within ${current_close * _MAX_LLM_ADJUSTMENT:.2f} of ${quant_close:.2f}.
""".strip()

    raw = call_groq(
        agent_name="OrchestratorAgent",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=ORCHESTRATOR_MODEL,
        max_tokens=600,
        expect_json=True,
        run_date=run_date,
        db_cache_fn=db_cache_fn,
        db_store_fn=db_store_fn,
    )

    # Build final prediction (always anchored by quant even if orchestrator fails)
    if raw["status"] != "ok":
        logger.warning("OrchestratorAgent skipped — using quant-only prediction")
        return _quant_only_prediction(quant_prediction, current_close, skipped_all=True)

    try:
        d = raw["data"]
        llm_close = float(d.get("predicted_close", quant_close))
        # Clamp to ±5% of quant anchor
        max_adj = current_close * _MAX_LLM_ADJUSTMENT
        llm_close = max(quant_close - max_adj, min(quant_close + max_adj, llm_close))

        return {
            "predicted_close":   round(llm_close, 4),
            "ci_lower_80":       round(ci_lo, 4),
            "ci_upper_80":       round(ci_hi, 4),
            "direction":         str(d.get("direction", quant_prediction["direction"])).lower(),
            "direction_prob":    round(float(d.get("direction_prob", quant_prediction["direction_prob"])), 4),
            "reasoning":         str(d.get("reasoning", ""))[:500],
            "agents_used":       ok_agents,
            "agents_skipped":    skipped_agents,
            "quant_anchor":      quant_close,
            "llm_adjusted":      True,
            "weighted_signal":   round(weighted_sig, 3),
            "quant_only_mode":   False,
        }
    except Exception as e:
        logger.error("OrchestratorAgent response parse error: %s — data=%s", e, raw.get("data"))
        return _quant_only_prediction(quant_prediction, current_close)


def _quant_only_prediction(quant_prediction: dict, current_close: float, skipped_all: bool = False) -> dict:
    """Fallback: return the quant model prediction verbatim, flagged as quant-only."""
    return {
        "predicted_close": quant_prediction["pred_close"],
        "ci_lower_80":     quant_prediction["ci_lower_80"],
        "ci_upper_80":     quant_prediction["ci_upper_80"],
        "direction":       quant_prediction["direction"],
        "direction_prob":  quant_prediction["direction_prob"],
        "reasoning":       "Quant-only mode: LLM orchestrator unavailable.",
        "agents_used":     [],
        "agents_skipped":  ["all"],
        "quant_anchor":    quant_prediction["pred_close"],
        "llm_adjusted":    False,
        "weighted_signal": 0.0,
        "quant_only_mode": True,
    }

"""
ReporterAgent — writes a short plain-English commentary for the website.
Uses the stronger model; called last after all signals are assembled.
"""

import logging
from typing import Any

from utils.groq_client import call_groq, REPORTER_MODEL

logger = logging.getLogger(__name__)


def generate_report(
    final_prediction: dict,
    specialist_signals: list[dict],
    current_close: float,
    run_date: str | None = None,
    db_cache_fn=None,
    db_store_fn=None,
) -> dict[str, Any]:
    """
    Generate a brief daily commentary for the website.
    Returns {"status": "ok"|"skipped", "commentary": str, "one_liner": str}
    """
    sigs_text = ""
    for s in specialist_signals:
        if s.get("status") == "ok":
            sigs_text += (
                f"  - {s['agent']}: {s['predicted_direction'].upper()} "
                f"(signal={s['signal']:+.2f}, conf={s['confidence']:.0%}) "
                f"— {s['rationale']}\n"
            )
        else:
            sigs_text += f"  - {s['agent']}: SKIPPED\n"

    system_prompt = (
        "You are the daily reporter for a silver-price prediction system. "
        "Write a brief, factual, jargon-free commentary for a retail investor audience. "
        "Do NOT claim accuracy or give financial advice. "
        "Always note that silver is volatile and predictions can be wrong. "
        "Respond ONLY with valid JSON: "
        '{"one_liner": "≤15 words summarizing today\'s call", '
        '"commentary": "2-3 paragraph plain-English commentary (≤200 words)", '
        '"watch_list": ["2-3 things to watch tomorrow"]}'
    )

    pred = final_prediction
    user_prompt = f"""
Today's silver close: ${current_close:.2f}
Tomorrow's prediction: ${pred['predicted_close']:.2f}  ({pred['direction'].upper()})
Direction probability: {pred['direction_prob']:.0%}
80% confidence band: [${pred['ci_lower_80']:.2f}, ${pred['ci_upper_80']:.2f}]
Mode: {'quant-only (no LLM signals)' if pred.get('quant_only_mode') else 'full multi-agent'}
Reasoning: {pred.get('reasoning', '')}

Agent signals:
{sigs_text}

Write the daily commentary.
""".strip()

    raw = call_groq(
        agent_name="ReporterAgent",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=REPORTER_MODEL,
        max_tokens=600,
        expect_json=True,
        run_date=run_date,
        db_cache_fn=db_cache_fn,
        db_store_fn=db_store_fn,
    )

    if raw["status"] != "ok":
        return {
            "status":      "skipped",
            "commentary":  "Daily commentary unavailable.",
            "one_liner":   f"Silver predicted {pred['direction']} to ${pred['predicted_close']:.2f}.",
            "watch_list":  [],
        }

    try:
        d = raw["data"]
        return {
            "status":      "ok",
            "one_liner":   str(d.get("one_liner", ""))[:120],
            "commentary":  str(d.get("commentary", ""))[:1000],
            "watch_list":  [str(x)[:100] for x in (d.get("watch_list") or [])[:3]],
        }
    except Exception as e:
        logger.error("ReporterAgent parse error: %s", e)
        return {
            "status":      "skipped",
            "commentary":  "Daily commentary unavailable.",
            "one_liner":   f"Silver predicted {pred['direction']} to ${pred['predicted_close']:.2f}.",
            "watch_list":  [],
        }

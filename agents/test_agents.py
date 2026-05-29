"""
Integration tests for all agents.  Uses mocked Groq to avoid burning API quota.
Usage: python agents/test_agents.py
"""

import sys
import os
import json
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.WARNING, format="%(levelname)s  %(message)s")

import utils.groq_client as gc


MOCK_SIGNAL = {
    "signal": 0.4,
    "confidence": 0.7,
    "predicted_direction": "up",
    "rationale": "Bullish technicals with gold leading.",
    "key_numbers": {"gs_ratio": 85.1},
}

MOCK_ORCHESTRATOR = {
    "predicted_close": 75.9,
    "direction": "up",
    "direction_prob": 0.62,
    "reasoning": "Quant model signals slight upward bias; technicals confirm.",
    "agents_used": ["TechnicalAgent"],
}

MOCK_REPORTER = {
    "one_liner": "Silver expected up tomorrow on bullish technicals.",
    "commentary": "Silver showed strength today. The technical model signals upward momentum. "
                  "As always, silver is volatile and this is not financial advice.",
    "watch_list": ["DXY movement", "Gold futures", "Copper trend"],
}


def _make_mock(payload: dict):
    msg = MagicMock()
    msg.content = json.dumps(payload)
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _mock_client(payload):
    client = MagicMock()
    client.chat.completions.create.return_value = _make_mock(payload)
    return client


def run_with_mock(payload, fn, *args, **kwargs):
    """Call fn with mocked Groq. run_date is passed only if fn accepts it."""
    gc._in_memory_cache.clear()
    with patch.dict(os.environ, {"GROQ_API_KEY": "fake_key"}):
        with patch("utils.groq_client._get_groq_client", return_value=_mock_client(payload)):
            with patch("utils.groq_client._sleep_for_tpm"):
                return fn(*args, **kwargs)


def test_all_specialists():
    from agents.specialists import (
        precious_metals_agent, dollar_rates_agent, industrial_demand_agent,
        macro_sentiment_agent, technical_agent,
    )

    tests = [
        ("PreciousMetalsAgent", lambda: precious_metals_agent(
            silver_close=75.0, gold_close=4500.0, gs_ratio=60.0,
            gs_ratio_vs20=0.05, gold_ret5=0.01, silver_vs_sma50=0.02)),
        ("DollarRatesAgent", lambda: dollar_rates_agent(
            dxy_close=99.0, dxy_ret5=-0.005, dxy_vs_sma20=-0.01,
            real_rate=2.1, real_rate_chg5=-0.05, nom_rate=None)),
        ("IndustrialDemandAgent", lambda: industrial_demand_agent(
            copper_close=6.4, copper_ret5=0.02, cu_vs_sma20=0.03,
            cu_si_ratio=0.085, cu_si_ratio_vs20=0.01,
            sp500_ret5=0.01, sp500_vs_sma50=0.04)),
        ("MacroSentimentAgent", lambda: macro_sentiment_agent(
            infl_exp=2.3, infl_exp_chg5=0.05,
            sp500_ret5=0.01, sp500_vs_sma50=0.04,
            silver_vol21=0.015)),
        ("TechnicalAgent", lambda: technical_agent(
            silver_close=75.0, silver_ret1=0.005, silver_ret5=0.01,
            silver_ret10=0.015, silver_rsi14=55.0,
            silver_vs_sma20=0.02, silver_vs_sma50=0.02, silver_vs_sma200=0.04,
            silver_vol10=0.012, silver_atr14=1.2)),
    ]

    for name, fn in tests:
        result = run_with_mock(MOCK_SIGNAL, fn)
        assert result["status"] == "ok", f"{name} returned status={result['status']}"
        assert -1.0 <= result["signal"] <= 1.0, f"{name} signal out of range"
        assert result["predicted_direction"] in ("up", "down"), f"{name} bad direction"
        print(f"  [OK] {name}: signal={result['signal']:+.2f}, dir={result['predicted_direction']}")


def test_orchestrator():
    from agents.orchestrator import orchestrate

    signals = [
        {"status": "ok", "agent": "TechnicalAgent",        "signal": 0.4, "confidence": 0.7,
         "predicted_direction": "up", "rationale": "Bullish MA cross"},
        {"status": "skipped", "agent": "PreciousMetalsAgent", "error": "timeout"},
    ]
    quant = {
        "pred_close": 75.9, "ci_lower_80": 73.5, "ci_upper_80": 78.3,
        "direction_prob": 0.62, "direction": "up", "pred_return": 0.008,
    }

    result = run_with_mock(MOCK_ORCHESTRATOR, orchestrate,
                           specialist_signals=signals,
                           quant_prediction=quant,
                           current_close=75.0)
    assert result["predicted_close"] > 0, f"Bad predicted close: {result}"
    assert result["direction"] in ("up", "down")
    # CI must come from quant (LLM can't override it)
    assert result["ci_lower_80"] == quant["ci_lower_80"]
    assert result["ci_upper_80"] == quant["ci_upper_80"]
    # Skipped agent recorded
    assert "PreciousMetalsAgent" in result["agents_skipped"]
    print(f"  [OK] OrchestratorAgent: close=${result['predicted_close']:.2f}, "
          f"dir={result['direction']}, skipped={result['agents_skipped']}")


def test_orchestrator_quant_only():
    """If orchestrator LLM fails, must fall back to quant-only cleanly."""
    from agents.orchestrator import orchestrate

    quant = {
        "pred_close": 75.9, "ci_lower_80": 73.5, "ci_upper_80": 78.3,
        "direction_prob": 0.62, "direction": "up", "pred_return": 0.008,
    }

    # Force GROQ_API_KEY missing so orchestrator skips
    gc._in_memory_cache.clear()
    with patch.dict(os.environ, {"GROQ_API_KEY": ""}):
        result = orchestrate(
            specialist_signals=[],
            quant_prediction=quant,
            current_close=75.0,
            run_date="2099-06-02",
        )
    assert result["quant_only_mode"] is True
    assert result["predicted_close"] == quant["pred_close"]
    print(f"  [OK] Quant-only fallback: close=${result['predicted_close']:.2f}, mode=quant-only")


def test_reporter():
    from agents.reporter import generate_report

    pred = {
        "predicted_close": 75.9, "ci_lower_80": 73.5, "ci_upper_80": 78.3,
        "direction": "up", "direction_prob": 0.62, "quant_only_mode": False,
        "reasoning": "Bullish signals dominate.",
    }
    signals = [
        {"status": "ok", "agent": "TechnicalAgent", "signal": 0.4,
         "confidence": 0.7, "predicted_direction": "up", "rationale": "Bullish MAs"},
    ]

    result = run_with_mock(MOCK_REPORTER, generate_report,
                           final_prediction=pred,
                           specialist_signals=signals,
                           current_close=75.0,
                           run_date="2099-06-01")
    assert result["status"] == "ok"
    assert len(result["commentary"]) > 10
    assert len(result["one_liner"]) > 5
    print(f"  [OK] ReporterAgent: one_liner='{result['one_liner']}'")


def test_reporter_skip_graceful():
    """Reporter skip must still return a usable stub."""
    from agents.reporter import generate_report

    pred = {"predicted_close": 75.9, "direction": "up", "direction_prob": 0.62,
            "quant_only_mode": False, "reasoning": "",
            "ci_lower_80": 73.5, "ci_upper_80": 78.3}
    gc._in_memory_cache.clear()
    with patch.dict(os.environ, {"GROQ_API_KEY": ""}):
        result = generate_report(pred, [], current_close=75.0, run_date="2099-06-03")
    assert result["status"] == "skipped"
    assert len(result["one_liner"]) > 0   # stub must be non-empty
    print(f"  [OK] Reporter graceful skip: '{result['one_liner']}'")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  AGENTS INTEGRATION TESTS")
    print("=" * 60)

    failed = []
    tests = [
        ("Specialist agents",          test_all_specialists),
        ("Orchestrator",               test_orchestrator),
        ("Orchestrator quant-only",    test_orchestrator_quant_only),
        ("Reporter",                   test_reporter),
        ("Reporter graceful skip",     test_reporter_skip_graceful),
    ]
    for label, fn in tests:
        print(f"\n  {label} ...")
        try:
            fn()
        except AssertionError as e:
            print(f"  FAIL: {e}")
            failed.append(label)
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
            failed.append(label)

    print()
    if not failed:
        print("[OK] All agent tests passed -- Step 4 complete.\n")
    else:
        print(f"[FAIL] {len(failed)} test(s) failed: {failed}\n")
        sys.exit(1)

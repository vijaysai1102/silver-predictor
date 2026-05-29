"""
5 specialist agents — each analyzes ONE driver and returns a signal dict.
All calls are routed through the shared groq_client wrapper.
"""

import logging
from typing import Any

from utils.groq_client import call_groq, SPECIALIST_MODEL
from agents.base import build_system_prompt, validate_signal

logger = logging.getLogger(__name__)


def _run_agent(
    agent_name: str,
    domain: str,
    user_prompt: str,
    run_date: str | None = None,
    db_cache_fn=None,
    db_store_fn=None,
) -> dict[str, Any]:
    """Call Groq, validate JSON, return signal dict with status."""
    system = build_system_prompt(agent_name, domain)
    raw = call_groq(
        agent_name=agent_name,
        system_prompt=system,
        user_prompt=user_prompt,
        model=SPECIALIST_MODEL,
        max_tokens=500,
        expect_json=True,
        run_date=run_date,
        db_cache_fn=db_cache_fn,
        db_store_fn=db_store_fn,
    )

    if raw["status"] != "ok":
        return {"status": "skipped", "agent": agent_name, "error": raw.get("error", "unknown")}

    try:
        signal = validate_signal(raw["data"])
        signal["status"] = "ok"
        signal["agent"] = agent_name
        return signal
    except (ValueError, KeyError, TypeError) as e:
        logger.error("[%s] Signal validation failed: %s — data=%s", agent_name, e, raw["data"])
        return {"status": "skipped", "agent": agent_name, "error": f"validation: {e}"}


# ── 1. PreciousMetalsAgent ────────────────────────────────────────────────────

def precious_metals_agent(
    silver_close: float,
    gold_close: float,
    gs_ratio: float,
    gs_ratio_vs20: float,    # % above/below 20-day mean
    gold_ret5: float,        # gold 5-day return
    silver_vs_sma50: float,
    run_date: str | None = None,
    **cache_kwargs,
) -> dict:
    prompt = f"""
Silver/Gold market data (latest close):
  Silver close:         ${silver_close:.2f}
  Gold close:           ${gold_close:.2f}
  Gold/Silver ratio:    {gs_ratio:.1f}  ({gs_ratio_vs20:+.1%} vs 20-day avg)
  Gold 5-day return:    {gold_ret5:+.2%}
  Silver vs 50-day MA:  {silver_vs_sma50:+.2%}

Analyze whether these precious-metals signals are bullish or bearish for silver's NEXT trading day close.
""".strip()
    return _run_agent(
        "PreciousMetalsAgent",
        "gold/silver ratio and precious metals trends",
        prompt,
        run_date=run_date,
        **cache_kwargs,
    )


# ── 2. DollarRatesAgent ───────────────────────────────────────────────────────

def dollar_rates_agent(
    dxy_close: float,
    dxy_ret5: float,
    dxy_vs_sma20: float,
    real_rate: float | None,      # DFII10 or DGS10 (%)
    real_rate_chg5: float | None,  # 5-day change in real rate
    nom_rate: float | None,
    run_date: str | None = None,
    **cache_kwargs,
) -> dict:
    rate_str = ""
    if real_rate is not None:
        rate_str = f"  Real 10yr yield (TIPS): {real_rate:.2f}%  (5d chg: {real_rate_chg5:+.2f}%p)"
    elif nom_rate is not None:
        rate_str = f"  Nominal 10yr yield:     {nom_rate:.2f}%"

    prompt = f"""
US Dollar & Interest Rate data:
  DXY close:           {dxy_close:.2f}
  DXY 5-day return:    {dxy_ret5:+.2%}
  DXY vs 20-day MA:    {dxy_vs_sma20:+.2%}
{rate_str}

Silver is priced in USD and is non-yielding. Strong dollar and high real rates are bearish for silver.
Analyze whether the dollar and rates signal bullish or bearish for silver's NEXT trading day.
""".strip()
    return _run_agent(
        "DollarRatesAgent",
        "US dollar index and real interest rates",
        prompt,
        run_date=run_date,
        **cache_kwargs,
    )


# ── 3. IndustrialDemandAgent ──────────────────────────────────────────────────

def industrial_demand_agent(
    copper_close: float,
    copper_ret5: float,
    cu_vs_sma20: float,
    cu_si_ratio: float,
    cu_si_ratio_vs20: float,
    sp500_ret5: float,
    sp500_vs_sma50: float,
    run_date: str | None = None,
    **cache_kwargs,
) -> dict:
    prompt = f"""
Industrial demand proxy data:
  Copper close:           ${copper_close:.3f}/lb
  Copper 5-day return:    {copper_ret5:+.2%}
  Copper vs 20-day MA:    {cu_vs_sma20:+.2%}
  Copper/Silver ratio:    {cu_si_ratio:.2f}  ({cu_si_ratio_vs20:+.2%} vs 20d avg)
  S&P 500 5-day return:   {sp500_ret5:+.2%}
  S&P 500 vs 50-day MA:   {sp500_vs_sma50:+.2%}

Silver demand is ~50% industrial. Strong copper and equities suggest higher industrial activity.
Analyze whether industrial signals are bullish or bearish for silver's NEXT trading day.
""".strip()
    return _run_agent(
        "IndustrialDemandAgent",
        "industrial commodity proxies (copper, equities)",
        prompt,
        run_date=run_date,
        **cache_kwargs,
    )


# ── 4. MacroSentimentAgent ────────────────────────────────────────────────────

def macro_sentiment_agent(
    infl_exp: float | None,       # T10YIE breakeven inflation (%)
    infl_exp_chg5: float | None,
    sp500_ret5: float,
    sp500_vs_sma50: float,
    silver_vol21: float,          # 21-day realized volatility
    run_date: str | None = None,
    **cache_kwargs,
) -> dict:
    infl_str = ""
    if infl_exp is not None:
        chg = f"  5d chg: {infl_exp_chg5:+.2f}%p" if infl_exp_chg5 is not None else ""
        infl_str = f"  10yr Breakeven Inflation: {infl_exp:.2f}% {chg}"

    prompt = f"""
Macro / sentiment data:
{infl_str if infl_str else '  Inflation expectations: not available'}
  S&P 500 5-day return:   {sp500_ret5:+.2%}
  S&P 500 vs 50-day MA:   {sp500_vs_sma50:+.2%}
  Silver 21-day realized vol: {silver_vol21:.2%} per day

Higher inflation expectations and risk-off sentiment are generally bullish for silver (safe-haven + inflation hedge).
Analyze whether macro/sentiment signals are bullish or bearish for silver's NEXT trading day.
""".strip()
    return _run_agent(
        "MacroSentimentAgent",
        "inflation expectations, risk sentiment, and macro indicators",
        prompt,
        run_date=run_date,
        **cache_kwargs,
    )


# ── 5. TechnicalAgent ────────────────────────────────────────────────────────

def technical_agent(
    silver_close: float,
    silver_ret1: float,
    silver_ret5: float,
    silver_ret10: float,
    silver_rsi14: float,
    silver_vs_sma20: float,
    silver_vs_sma50: float,
    silver_vs_sma200: float,
    silver_vol10: float,
    silver_atr14: float | None,
    run_date: str | None = None,
    **cache_kwargs,
) -> dict:
    atr_str = f"  14-day ATR (avg true range): {silver_atr14:.3f}" if silver_atr14 is not None else ""
    prompt = f"""
Silver technical analysis:
  Close:                  ${silver_close:.2f}
  1-day return:           {silver_ret1:+.2%}
  5-day return:           {silver_ret5:+.2%}
  10-day return:          {silver_ret10:+.2%}
  RSI (14):               {silver_rsi14:.1f}
  vs 20-day MA:           {silver_vs_sma20:+.2%}
  vs 50-day MA:           {silver_vs_sma50:+.2%}
  vs 200-day MA:          {silver_vs_sma200:+.2%}
  10-day realized vol:    {silver_vol10:.2%}/day
{atr_str}

RSI above 70 = overbought (bearish signal), below 30 = oversold (bullish).
Price above key MAs = uptrend (bullish). Price well below MAs = mean-reversion potential.
Analyze whether technical signals are bullish or bearish for silver's NEXT trading day.
""".strip()
    return _run_agent(
        "TechnicalAgent",
        "price action, moving averages, RSI, and momentum",
        prompt,
        run_date=run_date,
        **cache_kwargs,
    )

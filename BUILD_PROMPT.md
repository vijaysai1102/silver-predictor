# BUILD PROMPT — Daily Silver-Price Multi-Agent Prediction System
---

## ROLE & OBJECTIVE

You are a senior ML/quant engineer. Build a fully automated, **multi-agent daily silver-price prediction system** with a public website that updates itself every day with **zero human interaction**. The system predicts the **next trading day's silver (XAG/USD) closing price**, shows a point estimate **with a confidence interval**, states a **directional call (up/down) with a probability**, and **logs every prediction against the actual outcome** so accuracy is tracked honestly over time.

Do not promise or imply "perfect" accuracy. Silver is a noisy financial asset. Your job is a system that is **calibrated, backtested, and measurably better than a naive baseline** — and transparent about its own error.

## REALISTIC SUCCESS CRITERIA (build to these, not to "perfect")

1. **Directional accuracy** on a held-out backtest that beats the naive "tomorrow = today" baseline (target 55–62% on next-day direction).
2. **Point-estimate error**: report MAE and RMSE; aim to beat a random-walk baseline.
3. **Calibration**: when the model says "70% chance up," it should be right ~70% of the time. Include a reliability check.
4. **Confidence interval** that contains the actual close ~80% of the time (well-calibrated 80% band).
5. The site must visibly display these live accuracy stats so the system is honest about itself.

## GROQ FREE-TIER RESILIENCE (critical — the site must never crash on rate limits)

Context: the pipeline runs **once per weekday** and makes only **7 LLM calls total** (5 specialists + orchestrator + reporter). That is trivial for Groq's free **requests-per-day** budget. The real free-tier constraint is **tokens-per-minute (~6,000 TPM on most models)** and the per-minute request cap (~30 RPM). Because this is a daily batch job and is **not** latency-sensitive, design for safety over speed:

1. **Run agents sequentially, never in parallel.** Add a deliberate pause between LLM calls (default ~20–30s, configurable) so total tokens in any rolling 60-second window stay under the TPM cap. Slow is fine — the job has all night.
2. **Lean prompts = low tokens.** Pass each agent only the small set of numbers it needs (latest values + a few summary stats), never raw multi-year data dumps. Cap `max_tokens` on every response (e.g., 400–600). This keeps each call far under the per-minute budget.
3. **Token budgeting + estimation.** Before each call, estimate input+output tokens; track a running per-minute total and auto-extend the pause if a call would breach the window.
4. **Robust 429 handling.** Wrap every Groq call in retry-with-exponential-backoff (e.g., 3–5 attempts, jittered). On HTTP 429, **read Groq's `retry-after` / rate-limit response headers** and sleep exactly that long before retrying.
5. **Graceful degradation, agent by agent.** If an agent still fails after retries, **skip it and continue** — record `status: "skipped"` for that agent and let the orchestrator weight the remaining signals. One dead agent must never abort the run.
6. **THE QUANT MODEL IS THE SAFETY NET.** The gradient-boosting ensemble uses **zero API calls**. If *every* LLM call fails, the pipeline must still output a complete prediction from the quant model alone, flagged as "quant-only mode." **The system is therefore architecturally incapable of crashing due to Groq limits.**
7. **Idempotent + cached per day.** If the job re-runs on the same date (retry/manual trigger), reuse already-computed agent outputs from `predictions.db` instead of re-calling the API.
8. **The site reads only committed static JSON.** It never calls Groq itself. If a daily run fails entirely, the site keeps showing the last successful prediction plus a visible "last updated" date — it does not break.
9. **Surface API health.** Log per-run how many agents succeeded vs. skipped and whether quant-only mode was used; optionally show a small status indicator on the site.

## TECH STACK (use exactly this unless you justify a swap)

- **Language:** Python 3.11+
- **LLM orchestration:** **Groq free API** (OpenAI-compatible endpoint, `groq` Python SDK). Groq runs open-source models only. Model assignments:
  - The 5 specialist agents → a small/fast, high-quota model (`llama-3.1-8b-instant`) to stay well within limits.
  - The orchestrator + reporter → a stronger reasoning model (`llama-3.3-70b-versatile`).
  - **At build time, query Groq's current model list and rate-limit docs and confirm these model IDs and the free-tier limits still exist; substitute the closest current equivalents if not.** Request strict JSON outputs from each agent (use Groq's JSON/structured-output mode where available).
- **Data sources (all free):**
  - `yfinance` for silver futures `SI=F` / `SLV` ETF, gold `GC=F`, US Dollar Index `DX-Y.NYB`, S&P 500 `^GSPC`, copper `HG=F`, US 10Y yield `^TNX`.
  - **FRED API** (`fredapi`) for real interest rates (DGS10, DFII10), inflation expectations (T10YIE), CPI.
  - Optional fallback scraping only if an API fails; never hard-fail the pipeline.
  - At the start, **verify each data source still works** and report any that are deprecated, then proceed with working ones.
- **Numerical/ML:** pandas, numpy, scikit-learn, and a gradient-boosting model (`xgboost` or `lightgbm`) as the quantitative ensemble layer.
- **Storage:** SQLite (`predictions.db`) for prediction history + actuals + accuracy; CSV exports for the site.
- **Backend (optional, only if needed):** FastAPI. Otherwise the pipeline writes static JSON the site reads.
- **Frontend:** Single static site (plain HTML + Tailwind via CDN + vanilla JS, OR Next.js if you prefer) that reads generated JSON. Must be mobile-friendly and look clean.
- **Scheduling / automation (the "no human" part):** **GitHub Actions** cron workflow that runs once per weekday after US market close, executes the full pipeline, commits results, and redeploys the site.
- **Deployment:** GitHub Pages (or Vercel/Netlify) for the site; GitHub Actions for the daily job. Everything must run on free tiers.

## MULTI-AGENT ARCHITECTURE

Build **5 specialist agents + 1 orchestrator + 1 quant ensemble + 1 reporter**. Each specialist independently analyzes ONE driver, fetches its own data, and returns strict JSON: `{ signal: -1.0..+1.0, confidence: 0..1, predicted_direction, rationale, key_numbers }`.

1. **PreciousMetalsAgent** — tracks the gold price and the **gold/silver ratio**; reasons about whether silver is rich/cheap vs gold and gold's own trend.
2. **DollarRatesAgent** — tracks **DXY (US Dollar Index)** and **real interest rates** (DFII10). Silver is priced in USD and is non-yielding, so a strong dollar / high real rates pressure it.
3. **IndustrialDemandAgent** — tracks **industrial proxies** (copper price, equities/PMI trend, and notes on solar/EV/electronics demand — silver is ~50% industrial). Higher industrial activity supports silver.
4. **MacroSentimentAgent** — tracks **inflation expectations (T10YIE)**, risk sentiment (VIX if available, equity trend), and any safe-haven flows.
5. **TechnicalAgent** — tracks **price-action signals** on silver itself: moving averages (e.g., 20/50/200-day), RSI, momentum, recent volatility, and support/resistance.

**QuantEnsemble (not an LLM):** a gradient-boosting model trained on engineered features (all the above metrics + lags + technicals) producing its own next-day return prediction and prediction interval. This is the statistical backbone.

**OrchestratorAgent:** takes the 5 specialist JSON outputs **plus** the QuantEnsemble output, combines them into a final next-day predicted close, a directional probability, and an 80% confidence interval. Weights for the specialists should be **learned/tuned from the backtest** (which drivers historically predicted best), not picked arbitrarily. Document the weighting logic. The LLM orchestrator explains the reasoning; the final number must be anchored by the quant ensemble so the LLM can't hallucinate a price.

**ReporterAgent:** writes a short, plain-English daily commentary (what each driver is saying, what to watch) for the website.

## PREDICTION METHODOLOGY

- Engineer features from all metrics (levels, daily returns, multi-day lags, ratios, technical indicators).
- Train QuantEnsemble on historical data with **proper time-series cross-validation** (no look-ahead; walk-forward / expanding window). Never shuffle time-series rows.
- Final prediction = quant ensemble anchor, adjusted by the weighted specialist signals within a bounded range; produce point estimate + 80% interval + direction probability.
- **Strictly forbid look-ahead bias.** Features for predicting day T+1 may only use data available at the close of day T.

## BACKTESTING (required before going live)

- Walk-forward backtest over at least the last 2–3 years.
- Report: directional accuracy, MAE, RMSE, vs the naive random-walk baseline; interval coverage; a calibration/reliability summary.
- Save a backtest report (markdown + chart) and surface the headline numbers on the website.
- If the model does NOT beat the baseline, **say so honestly** and suggest improvements rather than faking good numbers.

## DAILY AUTOMATION FLOW (GitHub Actions, weekdays after close)

1. Fetch latest market data for all metrics (with retries + graceful handling of missing sources).
2. **Score yesterday's prediction**: pull the actual close that just occurred, compare to what was predicted, write the result to `predictions.db`, update rolling accuracy.
3. Run the 5 specialist agents + quant ensemble + orchestrator to produce **today's prediction for the next trading day**.
4. Generate the reporter commentary.
5. Export everything the site needs as JSON/CSV (latest prediction, history, rolling accuracy, per-agent signals, commentary).
6. Commit results to the repo and trigger site redeploy.
7. Log success/failure; on failure, the site should show "last successful update: <date>" rather than break.

## WEBSITE SPEC

A clean dashboard showing:
- **Today's headline**: predicted next-day close, 80% confidence band, up/down call + probability, and a one-line summary.
- **Per-agent panel**: each of the 5 agents' signal, confidence, and one-line rationale.
- **Accuracy panel (prominent)**: rolling directional accuracy, MAE/RMSE, interval coverage, vs baseline — so the system is honest.
- **History chart**: predicted vs actual close over time.
- **Commentary** from the ReporterAgent.
- A clear disclaimer: *educational, not financial advice; markets are unpredictable.*
- "Last updated" timestamp pulled from the data, not hardcoded.

## DELIVERABLES / REPO STRUCTURE

Produce a complete, runnable repo:

```
silver-predictor/
  agents/            # 5 specialists + orchestrator + reporter
  quant/             # feature engineering, ensemble model, training
  data/              # fetchers for yfinance + FRED, with retry/fallback
  backtest/          # walk-forward backtest + report generator
  pipeline.py        # the full daily run (score yesterday -> predict today -> export)
  db/                # SQLite schema + access layer
  site/              # static dashboard (HTML/JS or Next.js) reading exported JSON
  .github/workflows/daily.yml   # cron automation
  requirements.txt
  README.md          # setup, API keys, deploy steps, how accuracy is measured
  .env.example       # GROQ_API_KEY, FRED_API_KEY
  utils/groq_client.py  # wrapper: sequential calls, token budgeting, 429 backoff, per-day cache
```

## CONSTRAINTS & GUARDRAILS

- Handle API failures gracefully; the daily job must not crash if one data source or one LLM agent is down.
- **All Groq calls go through one shared client wrapper** that enforces sequential execution, inter-call pauses, token budgeting, and 429 backoff (see the Groq Resilience section). No agent calls Groq directly.
- The pipeline must complete and produce a prediction even in **quant-only mode** (zero successful LLM calls). Test this path explicitly.
- Store the Groq key in `GROQ_API_KEY` (GitHub Secrets in CI); never hardcode it.
- Keep all secrets in env vars / GitHub Secrets (`GROQ_API_KEY`, `FRED_API_KEY`); never hardcode keys.
- The LLM agents must return **strict JSON** (validate it; retry on malformed output).
- The final price must be **anchored by the quant model**, not invented by an LLM.
- No look-ahead bias anywhere. No claims of perfect or guaranteed accuracy anywhere in code, output, or site.
- Everything must run on free tiers and require no manual step after initial deploy.

## BUILD ORDER (do these in sequence and show me each)

1. Data fetchers + verify all sources work today.
2. Feature engineering + quant ensemble + walk-forward backtest (show me the honest numbers).
3. **Build and unit-test the shared Groq client wrapper** (sequential calls, inter-call pause, token budgeting, 429 backoff with `retry-after`, per-day caching) and confirm Groq's current free-tier limits and model IDs. Simulate a 429 and confirm it backs off instead of crashing.
4. The 5 specialist agents + orchestrator + reporter, all routed through the Groq wrapper, with JSON validation and per-agent skip-on-failure.
5. `pipeline.py` end-to-end (score yesterday, predict tomorrow, export) — and explicitly test **quant-only mode** with the LLM disabled to prove the site still produces a prediction.
6. SQLite schema + accuracy tracking.
7. The static dashboard reading exported JSON (including an API-health / "last updated" indicator).
8. The GitHub Actions cron workflow + README with full deploy instructions.

Start with step 1 now. After each step, briefly confirm it works before moving on.
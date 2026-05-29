# Silver Price Predictor

A fully automated, multi-agent daily silver (XAG/USD) price prediction system.  
Predicts the next trading day's close with a confidence interval and directional probability.  
Updates itself every weekday after US market close with zero human interaction.

**Educational only — not financial advice.**

**Live site:** https://vijaysai1102.github.io/silver-predictor/

---

## Architecture

```
pipeline.py        daily orchestration: fetch → score → predict → export
  data/            yfinance + FRED fetchers with retry/fallback
  quant/           XGBoost gradient-boosting ensemble (zero API calls)
  agents/          5 LLM specialists + orchestrator + reporter (Groq API)
  db/              SQLite: predictions, actuals, accuracy log
  site/            static HTML dashboard reading site/data/predictions.json
  .github/         GitHub Actions cron workflow
  utils/           shared Groq client (rate limiting, retry, caching)
```

### Multi-agent flow

1. **PreciousMetalsAgent** — gold/silver ratio, gold trend
2. **DollarRatesAgent** — DXY, real interest rates
3. **IndustrialDemandAgent** — copper, equities (silver is ~50% industrial)
4. **MacroSentimentAgent** — inflation expectations, risk sentiment
5. **TechnicalAgent** — moving averages, RSI, momentum
6. **QuantEnsemble** (XGBoost, no LLM) — provides price anchor
7. **OrchestratorAgent** — combines signals into final prediction
8. **ReporterAgent** — plain-English daily commentary

---

## Setup

### 1. Clone and install

```bash
git clone <your-repo-url>
cd silver-predictor
pip install -r requirements.txt
```

### 2. Create `.env`

```bash
cp .env.example .env
# then edit .env and add your keys:
```

```
GROQ_API_KEY=your_groq_api_key_here   # free at console.groq.com
FRED_API_KEY=your_fred_api_key_here   # free at fred.stlouisfed.org/docs/api/api_key.html
```

FRED is optional — the system degrades gracefully without it (rate features fall back to yfinance yield proxy).

### 3. Verify data sources

```bash
python data/verify_sources.py
```

### 4. Run the backtest

```bash
python backtest/run_backtest.py
```

Runs a ~2-year walk-forward backtest and saves `backtest/backtest_metrics.json`.

### 5. Run the pipeline

```bash
python pipeline.py            # full run
python pipeline.py --quant-only   # no LLM calls
python pipeline.py --dry-run      # test without writing to DB
```

### 6. Open the site locally

Open `site/index.html` in a browser (or `python -m http.server 8080 --directory site`).

---

## Deploy to GitHub Pages

1. Push the repo to GitHub.
2. Go to **Settings → Pages → Source: Deploy from branch → main → /site**.
3. Add secrets in **Settings → Secrets → Actions**:
   - `GROQ_API_KEY`
   - `FRED_API_KEY`
4. The daily workflow (`.github/workflows/daily.yml`) runs Mon–Fri at 23:00 UTC,
   commits `site/data/predictions.json` and `predictions.db`, and Pages auto-redeploys.

---

## Honest Accuracy

Backtest results (walk-forward, last ~2 years, 556 predictions):

| Metric | Model | Naive baseline |
|--------|-------|---------------|
| Directional accuracy | **52.2%** | 50.0% |
| MAE | $1.08 | $1.08 |
| 80% CI coverage | 71% (backtest) / ~80% (calibrated) | — |

The model marginally beats the directional naive baseline.  
It does **not** beat the naive MAE.  
Silver is a noisy asset; 55–62% accuracy is a realistic target with more data.  
All predictions and actual outcomes are logged in `predictions.db` for transparent tracking.

---

## How accuracy is measured

Every day after market close, `pipeline.py` fetches today's actual silver close,
looks up yesterday's prediction in `predictions.db`, and writes a row to `accuracy_log`:
- `direction_correct`: 1 if the predicted direction matched
- `abs_error`: |predicted_close − actual_close|
- `in_ci_80`: 1 if actual fell inside the 80% band

Rolling stats (last 90 days) are exported to `predictions.json` and displayed on the site.

---

## Groq free-tier resilience

- All 7 LLM calls are **sequential** with a 25s inter-call pause (keeps well under 6,000 TPM).
- Each call is wrapped in 4-attempt retry with exponential backoff + `Retry-After` header parsing.
- If any agent fails after retries, it is **skipped** — the pipeline continues.
- If **all** LLM calls fail, the pipeline falls back to **quant-only mode** and still produces a prediction.
- The site reads only committed static JSON — it never calls Groq and never breaks.

---

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Recommended | Enables LLM agents; free at console.groq.com |
| `FRED_API_KEY` | Optional | Enables real-rate/CPI features; free at fred.stlouisfed.org |

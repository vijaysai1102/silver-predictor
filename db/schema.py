"""
SQLite schema and data-access layer for predictions, actuals, and accuracy tracking.
"""

import sqlite3
import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "predictions.db"

DDL = """
CREATE TABLE IF NOT EXISTS predictions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_date  TEXT NOT NULL,
    target_date      TEXT NOT NULL,
    predicted_close  REAL,
    ci_lower_80      REAL,
    ci_upper_80      REAL,
    direction        TEXT,
    direction_prob   REAL,
    quant_anchor     REAL,
    quant_only_mode  INTEGER DEFAULT 0,
    weighted_signal  REAL,
    agents_used      TEXT,
    agents_skipped   TEXT,
    reasoning        TEXT,
    commentary       TEXT,
    one_liner        TEXT,
    watch_list       TEXT,
    run_metadata     TEXT,
    -- SLV (iShares Silver Trust) prediction
    slv_predicted_close  REAL,
    slv_ci_lower_80      REAL,
    slv_ci_upper_80      REAL,
    created_at       TEXT DEFAULT (datetime('now')),
    UNIQUE(prediction_date, target_date)
);

CREATE TABLE IF NOT EXISTS actuals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_date      TEXT NOT NULL UNIQUE,
    actual_close    REAL,
    slv_actual_close REAL,
    source          TEXT DEFAULT 'yfinance',
    recorded_at     TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS accuracy_log (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    scored_on          TEXT NOT NULL,        -- date we ran the scoring
    prediction_date    TEXT NOT NULL,        -- when the prediction was made
    target_date        TEXT NOT NULL,        -- the day being predicted
    predicted_close    REAL,
    actual_close       REAL,
    abs_error          REAL,
    direction_correct  INTEGER,             -- 1 or 0
    in_ci_80           INTEGER,             -- 1 or 0
    quant_only_mode    INTEGER,
    UNIQUE(prediction_date, target_date)
);

CREATE TABLE IF NOT EXISTS agent_cache (
    cache_key   TEXT PRIMARY KEY,
    result_json TEXT NOT NULL,
    cached_at   TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_predictions_target ON predictions(target_date);
CREATE INDEX IF NOT EXISTS idx_actuals_trade ON actuals(trade_date);
CREATE INDEX IF NOT EXISTS idx_accuracy_target ON accuracy_log(target_date);
"""


def get_conn(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or DB_PATH
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db(db_path: Path | None = None):
    """Create tables if they don't exist, then migrate any missing columns."""
    with get_conn(db_path) as conn:
        conn.executescript(DDL)
        _migrate(conn)
    logger.info("DB initialised at %s", db_path or DB_PATH)


def _migrate(conn: sqlite3.Connection):
    """Add new columns to existing DBs without breaking old rows."""
    new_cols = [
        ("predictions", "slv_predicted_close", "REAL"),
        ("predictions", "slv_ci_lower_80",     "REAL"),
        ("predictions", "slv_ci_upper_80",      "REAL"),
        ("actuals",     "slv_actual_close",     "REAL"),
    ]
    existing = {}
    for table in ("predictions", "actuals"):
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        existing[table] = {r["name"] for r in rows}

    for table, col, col_type in new_cols:
        if col not in existing.get(table, set()):
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")
            logger.info("Migrated: added %s.%s", table, col)


# ── Prediction storage ────────────────────────────────────────────────────────

def upsert_prediction(pred: dict, commentary: dict, run_metadata: dict | None = None,
                      db_path: Path | None = None):
    with get_conn(db_path) as conn:
        conn.execute("""
            INSERT INTO predictions
              (prediction_date, target_date, predicted_close, ci_lower_80, ci_upper_80,
               direction, direction_prob, quant_anchor, quant_only_mode,
               weighted_signal, agents_used, agents_skipped, reasoning,
               commentary, one_liner, watch_list, run_metadata,
               slv_predicted_close, slv_ci_lower_80, slv_ci_upper_80)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(prediction_date, target_date) DO UPDATE SET
              predicted_close=excluded.predicted_close,
              ci_lower_80=excluded.ci_lower_80,
              ci_upper_80=excluded.ci_upper_80,
              direction=excluded.direction,
              direction_prob=excluded.direction_prob,
              quant_anchor=excluded.quant_anchor,
              quant_only_mode=excluded.quant_only_mode,
              weighted_signal=excluded.weighted_signal,
              agents_used=excluded.agents_used,
              agents_skipped=excluded.agents_skipped,
              reasoning=excluded.reasoning,
              commentary=excluded.commentary,
              one_liner=excluded.one_liner,
              watch_list=excluded.watch_list,
              run_metadata=excluded.run_metadata,
              slv_predicted_close=excluded.slv_predicted_close,
              slv_ci_lower_80=excluded.slv_ci_lower_80,
              slv_ci_upper_80=excluded.slv_ci_upper_80
        """, (
            pred["prediction_date"],
            pred["target_date"],
            pred.get("predicted_close"),
            pred.get("ci_lower_80"),
            pred.get("ci_upper_80"),
            pred.get("direction"),
            pred.get("direction_prob"),
            pred.get("quant_anchor"),
            int(pred.get("quant_only_mode", False)),
            pred.get("weighted_signal"),
            json.dumps(pred.get("agents_used", [])),
            json.dumps(pred.get("agents_skipped", [])),
            pred.get("reasoning", ""),
            commentary.get("commentary", ""),
            commentary.get("one_liner", ""),
            json.dumps(commentary.get("watch_list", [])),
            json.dumps(run_metadata or {}),
            pred.get("slv_predicted_close"),
            pred.get("slv_ci_lower_80"),
            pred.get("slv_ci_upper_80"),
        ))


# ── Actual close storage ──────────────────────────────────────────────────────

def upsert_actual(trade_date: str, actual_close: float,
                  slv_actual_close: float | None = None,
                  db_path: Path | None = None):
    with get_conn(db_path) as conn:
        conn.execute("""
            INSERT INTO actuals (trade_date, actual_close, slv_actual_close)
            VALUES (?, ?, ?)
            ON CONFLICT(trade_date) DO UPDATE SET
              actual_close=excluded.actual_close,
              slv_actual_close=excluded.slv_actual_close
        """, (trade_date, actual_close, slv_actual_close))


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_predictions(db_path: Path | None = None) -> list[dict]:
    """
    Find all predictions whose target_date now has an actual close, and write
    accuracy_log rows for any that aren't already scored.
    Returns list of newly scored rows.
    """
    today = date.today().isoformat()
    new_rows = []
    with get_conn(db_path) as conn:
        rows = conn.execute("""
            SELECT p.prediction_date, p.target_date, p.predicted_close,
                   p.ci_lower_80, p.ci_upper_80, p.direction, p.quant_only_mode,
                   a.actual_close
            FROM   predictions p
            JOIN   actuals a ON a.trade_date = p.target_date
            LEFT JOIN accuracy_log al ON (al.prediction_date=p.prediction_date
                                          AND al.target_date=p.target_date)
            WHERE al.id IS NULL
        """).fetchall()

        for row in rows:
            pred_close   = row["predicted_close"]
            actual_close = row["actual_close"]
            if pred_close is None or actual_close is None:
                continue
            abs_err      = abs(pred_close - actual_close)
            dir_actual   = "up" if actual_close >= pred_close - 0.0001 else "down"
            # Treat actual vs previous-day close for direction scoring
            # (direction: whether actual > previous day close, not > predicted)
            dir_correct  = int(row["direction"] == dir_actual)  # simplified
            in_ci        = int(row["ci_lower_80"] <= actual_close <= row["ci_upper_80"])

            conn.execute("""
                INSERT OR IGNORE INTO accuracy_log
                  (scored_on, prediction_date, target_date, predicted_close, actual_close,
                   abs_error, direction_correct, in_ci_80, quant_only_mode)
                VALUES (?,?,?,?,?,?,?,?,?)
            """, (
                today,
                row["prediction_date"],
                row["target_date"],
                pred_close,
                actual_close,
                abs_err,
                dir_correct,
                in_ci,
                row["quant_only_mode"],
            ))
            new_rows.append(dict(row))

    logger.info("Scored %d new prediction(s)", len(new_rows))
    return new_rows


def rolling_accuracy(last_n: int = 90, db_path: Path | None = None) -> dict:
    """Compute rolling accuracy stats over the last N scored predictions."""
    with get_conn(db_path) as conn:
        rows = conn.execute("""
            SELECT direction_correct, abs_error, in_ci_80,
                   predicted_close, actual_close
            FROM accuracy_log
            ORDER BY target_date DESC
            LIMIT ?
        """, (last_n,)).fetchall()

    if not rows:
        return {"n": 0, "directional_accuracy": None, "MAE": None, "ci_80_coverage": None}

    n              = len(rows)
    dir_acc        = sum(r["direction_correct"] for r in rows) / n
    mae            = sum(r["abs_error"] for r in rows) / n
    ci_cov         = sum(r["in_ci_80"] for r in rows) / n

    import math
    rmse = math.sqrt(sum((r["predicted_close"] - r["actual_close"]) ** 2 for r in rows) / n)

    return {
        "n":                    n,
        "directional_accuracy": round(dir_acc, 4),
        "MAE":                  round(mae, 4),
        "RMSE":                 round(rmse, 4),
        "ci_80_coverage":       round(ci_cov, 4),
    }


# ── Agent cache helpers (for groq_client) ────────────────────────────────────

def db_get_cache(cache_key: str, db_path: Path | None = None) -> Any:
    with get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT result_json FROM agent_cache WHERE cache_key=?", (cache_key,)
        ).fetchone()
    if row:
        return json.loads(row["result_json"])
    return None


def db_store_cache(cache_key: str, result: Any, db_path: Path | None = None):
    with get_conn(db_path) as conn:
        conn.execute("""
            INSERT OR REPLACE INTO agent_cache (cache_key, result_json)
            VALUES (?, ?)
        """, (cache_key, json.dumps(result)))


# ── Latest prediction for site ────────────────────────────────────────────────

def get_latest_prediction(db_path: Path | None = None) -> dict | None:
    with get_conn(db_path) as conn:
        row = conn.execute("""
            SELECT * FROM predictions ORDER BY prediction_date DESC LIMIT 1
        """).fetchone()
    if not row:
        return None
    d = dict(row)
    for key in ("agents_used", "agents_skipped", "watch_list", "run_metadata"):
        try:
            d[key] = json.loads(d[key] or "[]")
        except Exception:
            d[key] = []
    return d


def get_prediction_history(limit: int = 90, db_path: Path | None = None) -> list[dict]:
    """Return recent predictions joined with actuals (where available)."""
    with get_conn(db_path) as conn:
        rows = conn.execute("""
            SELECT p.prediction_date, p.target_date, p.predicted_close,
                   p.ci_lower_80, p.ci_upper_80, p.direction, p.direction_prob,
                   p.quant_only_mode, a.actual_close,
                   p.slv_predicted_close, p.slv_ci_lower_80, p.slv_ci_upper_80,
                   a.slv_actual_close
            FROM predictions p
            LEFT JOIN actuals a ON a.trade_date = p.target_date
            ORDER BY p.prediction_date DESC
            LIMIT ?
        """, (limit,)).fetchall()
    return [dict(r) for r in rows]

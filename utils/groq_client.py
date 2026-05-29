"""
Shared Groq API client wrapper.

Enforces:
  - Sequential calls only (no parallel LLM calls)
  - Configurable inter-call pause (default 25s) to stay under TPM limits
  - Token budgeting: estimates input+output tokens and extends pause if needed
  - 429 backoff: reads Retry-After / x-ratelimit-reset-tokens headers
  - Per-day caching: returns cached result if same (date, agent, prompt_hash) already run
  - JSON output validation with retry on malformed JSON
  - Graceful skip: after max retries, returns status="skipped" — never raises

Free-tier limits (Groq, verified 2025):
  - llama-3.1-8b-instant:   ~6,000 TPM,  ~30 RPM  (specialist agents)
  - llama-3.3-70b-versatile: ~6,000 TPM,  ~30 RPM  (orchestrator / reporter)
  TPD (tokens per day) is generous; the per-minute window is the binding constraint.
"""

import os
import time
import json
import hashlib
import logging
from datetime import date
from typing import Any

logger = logging.getLogger(__name__)

# ── Model assignments ─────────────────────────────────────────────────────────
SPECIALIST_MODEL    = "llama-3.1-8b-instant"
ORCHESTRATOR_MODEL  = "llama-3.3-70b-versatile"
REPORTER_MODEL      = "llama-3.3-70b-versatile"

# ── Rate-limit safety parameters ─────────────────────────────────────────────
DEFAULT_INTER_CALL_PAUSE_S = 25       # seconds between calls (keeps TPM safe)
TPM_BUDGET                 = 5_500    # conservative cap (real limit ~6000)
ESTIMATED_TOKENS_PER_CALL  = 800      # prompt ≈ 300–400 tok + response ≤ 600 tok
MAX_RESPONSE_TOKENS        = 600      # cap passed to every call
MAX_RETRIES                = 4
BACKOFF_BASE_S             = 2.0      # exponential backoff base

# ── In-memory per-run token tracking ─────────────────────────────────────────
_minute_window_start: float = 0.0
_tokens_this_minute: int = 0


def _estimate_tokens(text: str) -> int:
    """Rough estimate: ~4 characters per token."""
    return max(1, len(text) // 4)


def _sleep_for_tpm(estimated_input_tokens: int, max_output_tokens: int):
    """
    If adding this call would exceed TPM_BUDGET within the current rolling minute,
    sleep until the window resets.  Also applies the fixed inter-call pause.
    """
    global _minute_window_start, _tokens_this_minute

    now = time.monotonic()
    call_tokens = estimated_input_tokens + max_output_tokens

    # Reset window if > 60s has passed
    if now - _minute_window_start >= 60.0:
        _minute_window_start = now
        _tokens_this_minute = 0

    # If adding this call would exceed the budget, sleep until the window resets
    remaining_in_window = 60.0 - (now - _minute_window_start)
    if _tokens_this_minute + call_tokens > TPM_BUDGET and remaining_in_window > 0:
        logger.info(
            "TPM budget check: %d + %d > %d — sleeping %.1fs to reset window",
            _tokens_this_minute, call_tokens, TPM_BUDGET, remaining_in_window,
        )
        time.sleep(remaining_in_window + 1.0)
        _minute_window_start = time.monotonic()
        _tokens_this_minute = 0

    # Apply the fixed inter-call pause (always, regardless of TPM)
    time.sleep(DEFAULT_INTER_CALL_PAUSE_S)

    _tokens_this_minute += call_tokens


def _cache_key(run_date: str, agent_name: str, prompt: str) -> str:
    h = hashlib.sha256(prompt.encode()).hexdigest()[:12]
    return f"{run_date}::{agent_name}::{h}"


# Simple in-process cache; the DB-backed cache is in db/access.py
_in_memory_cache: dict[str, Any] = {}


def _get_groq_client():
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY environment variable not set")
    from groq import Groq
    return Groq(api_key=api_key)


def call_groq(
    agent_name: str,
    system_prompt: str,
    user_prompt: str,
    model: str = SPECIALIST_MODEL,
    max_tokens: int = MAX_RESPONSE_TOKENS,
    expect_json: bool = True,
    run_date: str | None = None,
    db_cache_fn=None,      # optional callable(cache_key) -> cached_result | None
    db_store_fn=None,      # optional callable(cache_key, result)
) -> dict[str, Any]:
    """
    Single entry point for all Groq calls.

    Returns one of:
      {"status": "ok",      "agent": agent_name, "data": <parsed JSON or str>}
      {"status": "skipped", "agent": agent_name, "error": <reason>}
    """
    today = run_date or date.today().isoformat()
    prompt_full = system_prompt + "\n\n" + user_prompt
    cache_key = _cache_key(today, agent_name, prompt_full)

    # ── 1. Check in-memory cache ──────────────────────────────────────────────
    if cache_key in _in_memory_cache:
        logger.info("[%s] Cache hit (in-memory)", agent_name)
        return _in_memory_cache[cache_key]

    # ── 2. Check DB cache ─────────────────────────────────────────────────────
    if db_cache_fn:
        cached = db_cache_fn(cache_key)
        if cached is not None:
            logger.info("[%s] Cache hit (DB)", agent_name)
            _in_memory_cache[cache_key] = cached
            return cached

    # ── 3. Validate API key ───────────────────────────────────────────────────
    if not os.environ.get("GROQ_API_KEY", ""):
        result = {"status": "skipped", "agent": agent_name, "error": "GROQ_API_KEY not set"}
        _in_memory_cache[cache_key] = result
        return result

    est_input_tokens = _estimate_tokens(prompt_full)

    # ── 4. Retry loop ─────────────────────────────────────────────────────────
    last_error = ""
    for attempt in range(MAX_RETRIES):
        try:
            _sleep_for_tpm(est_input_tokens, max_tokens)

            client = _get_groq_client()
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ]

            kwargs: dict[str, Any] = {
                "model":       model,
                "messages":    messages,
                "max_tokens":  max_tokens,
                "temperature": 0.1,
            }
            if expect_json:
                kwargs["response_format"] = {"type": "json_object"}

            logger.info("[%s] Calling Groq (%s) attempt %d …", agent_name, model, attempt + 1)
            response = client.chat.completions.create(**kwargs)

            content = response.choices[0].message.content or ""

            if expect_json:
                try:
                    data = json.loads(content)
                except json.JSONDecodeError as e:
                    logger.warning("[%s] Malformed JSON on attempt %d: %s", agent_name, attempt + 1, e)
                    last_error = f"JSON parse error: {e}"
                    continue   # retry
            else:
                data = content

            result = {"status": "ok", "agent": agent_name, "data": data}
            _in_memory_cache[cache_key] = result
            if db_store_fn:
                db_store_fn(cache_key, result)
            logger.info("[%s] Success", agent_name)
            return result

        except Exception as exc:
            exc_str = str(exc)
            last_error = exc_str

            # ── 429 handling ──────────────────────────────────────────────────
            retry_after = _parse_retry_after(exc)
            if retry_after:
                sleep_s = retry_after + 1.0
                logger.warning(
                    "[%s] 429 rate-limit. Sleeping %.1fs per Retry-After header",
                    agent_name, sleep_s,
                )
                time.sleep(sleep_s)
                continue

            # Exponential backoff for other errors
            sleep_s = (BACKOFF_BASE_S ** attempt) + (0.5 * attempt)
            logger.warning(
                "[%s] Error on attempt %d: %s — retrying in %.1fs",
                agent_name, attempt + 1, exc_str[:120], sleep_s,
            )
            time.sleep(sleep_s)

    # ── 5. All retries exhausted — graceful skip ──────────────────────────────
    logger.error("[%s] All %d attempts failed. Skipping. Last error: %s", agent_name, MAX_RETRIES, last_error)
    result = {"status": "skipped", "agent": agent_name, "error": last_error}
    _in_memory_cache[cache_key] = result
    return result


def _parse_retry_after(exc: Exception) -> float | None:
    """
    Extract wait time from a Groq 429 exception.
    Groq's RateLimitError includes headers; also inspect the exception string.
    """
    exc_str = str(exc)
    # Groq SDK wraps HTTP errors; check for rate limit status code
    if "429" not in exc_str and "rate_limit" not in exc_str.lower() and "RateLimitError" not in type(exc).__name__:
        return None
    # Try to read Retry-After from exception attributes
    for attr in ("response", "headers"):
        headers = getattr(exc, attr, None)
        if headers is None:
            continue
        if hasattr(headers, "get"):
            ra = headers.get("retry-after") or headers.get("Retry-After")
            if ra:
                try:
                    return float(ra)
                except (ValueError, TypeError):
                    pass
            # Groq also sends x-ratelimit-reset-tokens (seconds to refill)
            reset = headers.get("x-ratelimit-reset-tokens")
            if reset:
                # Format is like "3s" or "1m30s"
                return _parse_duration(reset)
    # Fallback: look for numbers in the error message
    import re
    match = re.search(r"try again in (\d+\.?\d*)s", exc_str, re.IGNORECASE)
    if match:
        return float(match.group(1))
    # Default for a confirmed 429: wait 60s
    if "429" in exc_str or "RateLimitError" in type(exc).__name__:
        return 60.0
    return None


def _parse_duration(s: str) -> float:
    """Parse '1m30s' or '45s' or '2.5s' into seconds."""
    import re
    total = 0.0
    for match in re.finditer(r"(\d+\.?\d*)([ms])", s):
        val, unit = float(match.group(1)), match.group(2)
        total += val * 60 if unit == "m" else val
    return total if total else 30.0

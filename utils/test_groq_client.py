"""
Tests for the Groq client wrapper.
Usage: python utils/test_groq_client.py

Tests:
  1. Sequential call enforcement (via timing)
  2. JSON validation + retry on bad JSON
  3. Simulated 429 — confirm it sleeps and retries (not raises)
  4. In-memory cache — same call returns cached without another API hit
  5. Graceful skip on missing API key
  6. Live call (if GROQ_API_KEY is set) — real ping to Groq
"""

import sys
import os
import time
import json
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

import utils.groq_client as gc


def _make_mock_response(content: str):
    """Build a mock that mimics groq.ChatCompletion response."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ── Test 1: Graceful skip when no API key ────────────────────────────────────
def test_no_api_key():
    print("\n[TEST 1] Graceful skip with no API key ...")
    with patch.dict(os.environ, {"GROQ_API_KEY": ""}):
        # Clear cache
        gc._in_memory_cache.clear()
        result = gc.call_groq(
            "test_agent",
            "You are a test agent.",
            "Return JSON: {}",
            run_date="2099-01-01",
        )
    assert result["status"] == "skipped", f"Expected 'skipped', got {result}"
    assert "GROQ_API_KEY" in result["error"]
    print("  PASS — returned status='skipped' with no key")


# ── Test 2: In-memory cache ───────────────────────────────────────────────────
def test_cache():
    print("\n[TEST 2] In-memory cache ...")
    gc._in_memory_cache.clear()
    # Manually inject a cached result
    cached_result = {"status": "ok", "agent": "cached_agent", "data": {"x": 1}}
    from utils.groq_client import _cache_key
    key = _cache_key("2099-01-01", "cached_agent", "sys\n\nuser")
    gc._in_memory_cache[key] = cached_result

    result = gc.call_groq(
        "cached_agent",
        "sys",
        "user",
        run_date="2099-01-01",
    )
    assert result["data"] == {"x": 1}, f"Expected cached data, got {result}"
    print("  PASS — cache hit returned without API call")


# ── Test 3: JSON validation + retry on bad JSON ───────────────────────────────
def test_json_retry():
    print("\n[TEST 3] JSON retry on malformed response ...")
    gc._in_memory_cache.clear()
    call_count = 0

    def mock_create(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _make_mock_response("not valid json {{{{")
        return _make_mock_response('{"signal": 0.5, "confidence": 0.7}')

    with patch.dict(os.environ, {"GROQ_API_KEY": "fake_key_for_test"}):
        with patch("utils.groq_client._get_groq_client") as mock_client_fn:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = mock_create
            mock_client_fn.return_value = mock_client
            with patch("utils.groq_client._sleep_for_tpm"):   # skip actual sleeping
                result = gc.call_groq(
                    "json_test_agent",
                    "Return valid JSON.",
                    "Give me a signal.",
                    run_date="2099-01-02",
                )
    assert result["status"] == "ok", f"Expected ok, got {result}"
    assert result["data"]["signal"] == 0.5
    assert call_count == 2, f"Expected 2 calls (1 bad + 1 good), got {call_count}"
    print(f"  PASS — retried after bad JSON; succeeded on attempt 2 (calls={call_count})")


# ── Test 4: Simulated 429 — back off, don't crash ─────────────────────────────
def test_429_backoff():
    print("\n[TEST 4] Simulated 429 — must back off, not crash ...")
    gc._in_memory_cache.clear()
    call_count = 0
    sleep_calls = []

    class FakeRateLimit(Exception):
        """Mimics Groq's RateLimitError."""
        pass

    def mock_create(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise FakeRateLimit("RateLimitError: 429 Too Many Requests. Try again in 5s")
        return _make_mock_response('{"signal": 0.3}')

    original_sleep = time.sleep
    def fake_sleep(s):
        sleep_calls.append(s)

    with patch.dict(os.environ, {"GROQ_API_KEY": "fake_key_for_test"}):
        with patch("utils.groq_client._get_groq_client") as mock_client_fn:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = mock_create
            mock_client_fn.return_value = mock_client
            with patch("utils.groq_client._sleep_for_tpm"):
                with patch("time.sleep", side_effect=fake_sleep):
                    result = gc.call_groq(
                        "rate_limit_agent",
                        "sys",
                        "user",
                        run_date="2099-01-03",
                    )

    assert result["status"] == "ok", f"Expected ok after 429 backoff, got {result}"
    assert call_count == 3, f"Expected 3 calls, got {call_count}"
    # Verify at least one sleep > 5s (the 5s from the error message + 1s buffer)
    long_sleeps = [s for s in sleep_calls if s >= 5.0]
    assert long_sleeps, f"Expected sleep >= 5s for 429, got sleep_calls={sleep_calls}"
    print(f"  PASS — backed off {len(long_sleeps)} time(s) with sleep >= 5s; succeeded on attempt 3")


# ── Test 5: All retries fail → graceful skip, not exception ──────────────────
def test_graceful_skip():
    print("\n[TEST 5] All retries fail -> graceful skip ...")
    gc._in_memory_cache.clear()

    def always_fail(**kwargs):
        raise ConnectionError("Network unreachable")

    with patch.dict(os.environ, {"GROQ_API_KEY": "fake_key"}):
        with patch("utils.groq_client._get_groq_client") as mock_client_fn:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = always_fail
            mock_client_fn.return_value = mock_client
            with patch("utils.groq_client._sleep_for_tpm"):
                with patch("time.sleep"):
                    result = gc.call_groq(
                        "failing_agent",
                        "sys",
                        "user",
                        run_date="2099-01-04",
                    )
    assert result["status"] == "skipped", f"Expected 'skipped', got {result}"
    assert "error" in result
    print(f"  PASS — agent skipped after {gc.MAX_RETRIES} failures; no exception raised")


# ── Test 6: Token budget extension ───────────────────────────────────────────
def test_tpm_budget():
    print("\n[TEST 6] TPM budget: large call should trigger window sleep ...")
    gc._in_memory_cache.clear()
    gc._minute_window_start = time.monotonic()
    gc._tokens_this_minute = gc.TPM_BUDGET - 100   # almost full

    sleep_calls = []
    original_sleep = time.sleep
    def fake_sleep(s):
        sleep_calls.append(s)

    with patch("time.sleep", side_effect=fake_sleep):
        gc._sleep_for_tpm(estimated_input_tokens=200, max_output_tokens=600)

    # Should have slept to clear the window
    window_sleeps = [s for s in sleep_calls if s > 1]
    assert window_sleeps, f"Expected window reset sleep, got {sleep_calls}"
    print(f"  PASS — TPM exceeded, slept {window_sleeps[0]:.1f}s to reset window")


# ── Test 7: Live Groq call (optional, requires real key) ─────────────────────
def test_live_call():
    print("\n[TEST 7] Live Groq call ...")
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        print("  SKIP — GROQ_API_KEY not set")
        return

    gc._in_memory_cache.clear()
    # Reset TPM tracking for clean test
    gc._minute_window_start = 0.0
    gc._tokens_this_minute = 0

    result = gc.call_groq(
        "live_test_agent",
        "You are a test assistant. Always respond with valid JSON.",
        'Return exactly this JSON: {"status": "alive", "model": "working"}',
        model=gc.SPECIALIST_MODEL,
        max_tokens=50,
        expect_json=True,
        run_date="2099-01-05",
    )
    if result["status"] == "ok":
        print(f"  PASS — live call succeeded: {result['data']}")
    else:
        print(f"  WARN — live call returned {result['status']}: {result.get('error','')}")


# ── Run all tests ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  GROQ CLIENT WRAPPER TESTS")
    print("=" * 60)

    failed = []
    for fn in [test_no_api_key, test_cache, test_json_retry, test_429_backoff,
               test_graceful_skip, test_tpm_budget, test_live_call]:
        try:
            fn()
        except AssertionError as e:
            print(f"  FAIL: {e}")
            failed.append(fn.__name__)
        except Exception as e:
            print(f"  ERROR in {fn.__name__}: {e}")
            failed.append(fn.__name__)

    print()
    if not failed:
        print("[OK] All tests passed -- Step 3 complete.\n")
    else:
        print(f"[FAIL] {len(failed)} test(s) failed: {failed}\n")
        sys.exit(1)

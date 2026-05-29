"""
Run this once to confirm every data source is reachable and returning data.
Usage: python data/verify_sources.py
"""

import sys
import os
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

from data.fetchers import fetch_all, latest_values, YFINANCE_TICKERS, FRED_SERIES


def main():
    print("\n" + "=" * 60)
    print("  DATA SOURCE VERIFICATION")
    print("=" * 60 + "\n")

    fred_key = os.environ.get("FRED_API_KEY", "")
    if not fred_key:
        print("[WARN] FRED_API_KEY not set - FRED sources will be skipped.")
        print("       Set it in .env or as environment variable.\n")
    else:
        print(f"[OK]   FRED_API_KEY found ({fred_key[:4]}...)\n")

    print("Fetching ~5 years of history for all sources ...\n")
    all_data = fetch_all(fred_api_key=fred_key or None)

    ok = []
    skipped = []
    errors = []

    for name, rec in all_data.items():
        status  = rec["status"]
        rows    = rec.get("rows", 0)
        last    = rec.get("last_updated", "N/A")
        err_msg = rec.get("error", "")

        if status == "ok":
            ok.append(name)
            tag = rec.get("ticker") or rec.get("series_id", "")
            print(f"  [OK]   {name:<18} ({tag:<12}) | {rows:>5} rows | last: {last}")
        elif status == "skipped":
            skipped.append(name)
            print(f"  [SKIP] {name:<18} SKIPPED - {err_msg}")
        else:
            errors.append(name)
            print(f"  [FAIL] {name:<18} ERROR   - {err_msg}")

    print()
    print("=" * 60)
    print(f"  OK: {len(ok)}   SKIPPED: {len(skipped)}   ERRORS: {len(errors)}")
    print("=" * 60)

    if ok:
        print("\nLatest values:")
        vals = latest_values(all_data)
        for name, val in vals.items():
            if val is not None:
                print(f"  {name:<18} = {val:.4f}")

    if errors:
        print(f"\n[WARN] {len(errors)} source(s) failed. Check network/API keys.")

    if not errors and not skipped:
        print("\n[OK] All sources OK -- Step 1 complete.\n")
    elif not errors:
        print("\n[OK] All available sources OK (FRED skipped - add API key to enable).\n")
    else:
        print("\n[FAIL] Some sources failed - review errors above.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()

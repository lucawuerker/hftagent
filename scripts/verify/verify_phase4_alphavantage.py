"""Phase 4 verification — REAL AlphaVantage fetch (needs ALPHAVANTAGE_API_KEY).

Run: PYTHONPATH=. ./venv/bin/python scripts/verify/verify_phase4_alphavantage.py

Frugal by design: only 2 tickers (free tier is ~25 requests/day, ~5/min). The
shared HTTP helper throttles to ~13s/request and the cache makes re-runs free.
Skips cleanly (exit 0) on missing key, no network, or rate-limit.
"""

import os
import tempfile
from datetime import date, timedelta

from dotenv import load_dotenv

load_dotenv()

from quant_fund_agent.config import DataSettings, Settings
from quant_fund_agent.data import load_panel
from quant_fund_agent.data.frequency import periods_per_year_from_index
from quant_fund_agent.data.providers._http import RateLimited

TICKERS = ["AAPL", "MSFT"]  # kept tiny to respect the 25/day free limit


def main():
    if not os.getenv("ALPHAVANTAGE_API_KEY"):
        print("SKIP — ALPHAVANTAGE_API_KEY not set in .env.")
        return

    # Free tier returns only the last ~100 daily bars (outputsize=compact), so
    # keep the window inside that to avoid a never-covered cache.
    end = date.today().isoformat()
    start = (date.today() - timedelta(days=80)).isoformat()
    tmp = tempfile.mkdtemp()
    settings = Settings(data=DataSettings(
        provider="alphavantage", tickers=TICKERS, start=start, end=end,
        frequency="1d", cache_dir=tmp))

    print(f"===== live AlphaVantage fetch: {TICKERS}  (~13s/request throttle) =====")
    print("NOTE: free tier prices are UNADJUSTED (splits cause jumps).")
    try:
        panel = load_panel(settings=settings)
    except RateLimited as e:
        print(f"SKIP — AlphaVantage rate-limited (free tier 25/day): {e}")
        return
    except Exception as e:  # noqa: BLE001
        print(f"SKIP — AlphaVantage fetch failed (network/key): {type(e).__name__}: {e}")
        return

    close = panel.get("close")
    if close is None or close.empty:
        print("SKIP — AlphaVantage returned no data.")
        return

    print("fields:", sorted(panel))
    print("tickers loaded:", list(close.columns))
    print("shape:", close.shape, "| range:", str(close.index.min()), "→", str(close.index.max()))
    print("periods/year inferred:", periods_per_year_from_index(close.index))
    print("\nlast 3 (raw) closes:")
    print(close.tail(3).to_string())
    print("\nLIVE ALPHAVANTAGE FETCH OK")


if __name__ == "__main__":
    main()

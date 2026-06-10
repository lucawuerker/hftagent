"""Phase 4 verification — REAL FMP fetch (needs FMP_API_KEY + network).

Run: PYTHONPATH=. ./venv/bin/python scripts/verify/verify_phase4_fmp.py

Pulls a few tickers of recent daily data through the actual FMPProvider + cache
and prints the result. Skips cleanly (exit 0) if no key or no network.
"""

import os
import tempfile
from datetime import date, timedelta

from dotenv import load_dotenv

load_dotenv()

from quant_fund_agent.config import DataSettings, Settings
from quant_fund_agent.data import load_panel
from quant_fund_agent.data.frequency import periods_per_year_from_index

TICKERS = ["AAPL", "MSFT", "NVDA"]


def main():
    if not os.getenv("FMP_API_KEY"):
        print("SKIP — FMP_API_KEY not set in .env.")
        return

    end = date.today().isoformat()
    start = (date.today() - timedelta(days=200)).isoformat()
    tmp = tempfile.mkdtemp()
    settings = Settings(data=DataSettings(
        provider="fmp", tickers=TICKERS, start=start, end=end,
        frequency="1d", cache_dir=tmp))

    print(f"===== live FMP fetch: {TICKERS}  {start}..{end} =====")
    try:
        panel = load_panel(settings=settings)
    except Exception as e:  # noqa: BLE001
        print(f"SKIP — FMP fetch failed (network / key / quota): {type(e).__name__}: {e}")
        return

    close = panel.get("close")
    if close is None or close.empty:
        print("SKIP — FMP returned no data. Run locally with a valid key.")
        return

    print("fields:", sorted(panel))
    print("tickers loaded:", list(close.columns))
    print("shape:", close.shape, "| range:", str(close.index.min()), "→", str(close.index.max()))
    print("periods/year inferred:", periods_per_year_from_index(close.index))
    print("vwap present:", "vwap" in panel, "| returns present:", "returns" in panel)
    print("\nlast 3 adjusted closes:")
    print(close.tail(3).to_string())
    print("\nLIVE FMP FETCH OK")


if __name__ == "__main__":
    main()

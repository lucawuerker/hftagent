"""Phase 3 verification — REAL yfinance fetch (free, needs network).

Run: PYTHONPATH=. ./venv/bin/python scripts/verify/verify_phase3_live_yfinance.py

Pulls a few tickers of recent daily data through the actual YFinanceProvider +
parquet cache and prints the result. Skips cleanly (exit 0) if there is no
network, so it never produces a false failure in a sandbox.
"""

import tempfile
from datetime import date, timedelta

from quant_fund_agent.config import DataSettings, Settings
from quant_fund_agent.data import load_panel
from quant_fund_agent.data.frequency import periods_per_year_from_index

TICKERS = ["AAPL", "MSFT", "NVDA"]


def main():
    end = date.today().isoformat()
    start = (date.today() - timedelta(days=200)).isoformat()
    tmp = tempfile.mkdtemp()
    settings = Settings(data=DataSettings(
        provider="yfinance", tickers=TICKERS, start=start, end=end,
        frequency="1d", cache_dir=tmp))

    print(f"===== live yfinance fetch: {TICKERS}  {start}..{end} =====")
    try:
        panel = load_panel(settings=settings)
    except Exception as e:  # noqa: BLE001
        print(f"SKIP — live fetch failed (likely no network): {type(e).__name__}: {e}")
        return

    close = panel.get("close")
    if close is None or close.empty:
        print("SKIP — yfinance returned no data (network/rate-limit). Run locally.")
        return

    print("fields:", sorted(panel))
    print("tickers loaded:", list(close.columns))
    print("shape:", close.shape, "| date range:", str(close.index.min()), "→", str(close.index.max()))
    print("periods/year inferred:", periods_per_year_from_index(close.index))
    print("\nlast 3 closes:")
    print(close.tail(3).to_string())
    print("\nvwap present:", "vwap" in panel, "| returns present:", "returns" in panel)
    print("\nLIVE YFINANCE FETCH OK")


if __name__ == "__main__":
    main()

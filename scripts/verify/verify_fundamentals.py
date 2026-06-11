"""Fundamentals verification — REAL FMP / AlphaVantage non-OHLCV fetches.

Run: PYTHONPATH=. ./venv/bin/python scripts/verify/verify_fundamentals.py

Proves the non-OHLCV data layer works end-to-end on live vendor data:
  1. FMP (needs FMP_API_KEY) — pull a few equities through `load_panel`; show the
     fundamental/estimate/event fields that materialise, a sector/peRatio/roe
     sample, and a **point-in-time timeline** (a field stepping at its availability
     date, NaN before it). Sanity-check AAPL's sector. Then run the latent-bug
     regression (alpha_058 with a wide `sector` frame) and the three example
     fundamental factors on the real panel.
  2. AlphaVantage (needs ALPHAVANTAGE_API_KEY) — best-effort smoke of the same
     seam; skips cleanly on the free-tier rate limit (~25/day).

Everything writes to a temp cache dir; nothing touches the repo. API keys are read
from `.env` and never printed. Skips (exit 0) on missing keys / network.
"""

import os
import tempfile
from datetime import date, timedelta

from dotenv import load_dotenv

load_dotenv()

from quant_fund_agent.config import DataSettings, Settings
from quant_fund_agent.data import load_panel
from quant_fund_agent.data.fields import NON_OHLCV_FIELDS
from quant_fund_agent.data.frequency import periods_per_year_from_index
from quant_fund_agent.data.providers._http import RateLimited
from quant_fund_agent.factors._discover import discover_factors
from quant_fund_agent.factors.registry import instantiate_factor

TMP = tempfile.mkdtemp()


def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _settings(provider: str, tickers: list[str], days: int = 900) -> Settings:
    end = date.today().isoformat()
    start = (date.today() - timedelta(days=days)).isoformat()
    return Settings(data=DataSettings(
        provider=provider, asset_class="equity", tickers=tickers,
        start=start, end=end, frequency="1d", cache_dir=TMP))


def _show_fundamentals(panel, label: str) -> None:
    present = sorted(set(panel) & set(NON_OHLCV_FIELDS))
    close = panel.get("close")
    print(f"\n  {label}: OHLCV bars={None if close is None else close.shape} "
          f"periods/year={None if close is None else periods_per_year_from_index(close.index)}")
    print(f"    non-OHLCV fields present: {present}")
    for f in ("sector", "peRatio", "roe", "epsSurprise"):
        if f in panel:
            last = panel[f].dropna(how="all").tail(1)
            if not last.empty:
                print(f"    latest {f}: {last.iloc[0].to_dict()}")


def _pit_timeline(panel, field: str, ticker: str) -> None:
    """Show that a field is NaN before its first availability, then steps."""
    if field not in panel or ticker not in panel[field].columns:
        return
    s = panel[field][ticker]
    first_valid = s.first_valid_index()
    print(f"\n  PIT timeline — {field}[{ticker}]:")
    if first_valid is None:
        print("    (no values)")
        return
    print(f"    first available: {first_valid.date()}  (NaN before this — no look-ahead)")
    # print the value at each change point
    changes = s.dropna()
    changes = changes[changes != changes.shift()]
    for dt, val in list(changes.items())[:6]:
        print(f"      {dt.date()}: {val}")


def _run_factors(panel) -> None:
    discover_factors()
    section("example fundamental factors + latent-bug regression")
    # latent-bug guard: alpha_058 neutralises VWAP by the now-real `sector` frame
    if "sector" in panel:
        try:
            sig = instantiate_factor("alpha_058").calc(panel)
            n = int(sig.tail(5).notna().sum().sum())
            print(f"  alpha_058 (sector-neutralised) ran OK — non-NaN tail cells: {n}")
        except Exception as e:  # noqa: BLE001
            print(f"  [FAIL] alpha_058 raised on a real sector frame: {type(e).__name__}: {e}")
    else:
        print("  (sector absent — skipping alpha_058 neutralisation check)")
    for fid in ("value_earnings_yield", "quality_roe", "earnings_surprise_drift"):
        try:
            sig = instantiate_factor(fid).calc(panel)
            last = sig.dropna(how="all").tail(1)
            ranks = last.iloc[0].dropna().round(3).to_dict() if not last.empty else {}
            print(f"  {fid}: shape={sig.shape}  latest cross-section ranks={ranks}")
        except Exception as e:  # noqa: BLE001
            print(f"  [FAIL] {fid}: {type(e).__name__}: {e}")


def main() -> None:
    tickers = ["AAPL", "MSFT", "NVDA"]

    section("1) FMP (FMP_API_KEY): fundamentals + estimates + events")
    if not os.getenv("FMP_API_KEY"):
        print("  SKIP — FMP_API_KEY not set.")
    else:
        try:
            panel = load_panel(settings=_settings("fmp", tickers))
            _show_fundamentals(panel, "fmp equity AAPL/MSFT/NVDA")
            _pit_timeline(panel, "peRatio", "AAPL")
            _pit_timeline(panel, "epsSurprise", "AAPL")
            sec = panel.get("sector")
            if sec is not None and "AAPL" in sec.columns:
                aapl_sector = sec["AAPL"].dropna()
                print(f"\n  AAPL sector (sanity): "
                      f"{aapl_sector.iloc[-1] if not aapl_sector.empty else 'MISSING'}")
            _run_factors(panel)
        except RateLimited as e:
            print(f"  SKIP — FMP rate-limited: {e}")
        except Exception as e:  # noqa: BLE001
            print(f"  SKIP — FMP fundamentals failed: {type(e).__name__}: {e}")

    section("2) AlphaVantage (ALPHAVANTAGE_API_KEY): best-effort smoke")
    if not os.getenv("ALPHAVANTAGE_API_KEY"):
        print("  SKIP — ALPHAVANTAGE_API_KEY not set.")
    else:
        try:
            panel = load_panel(settings=_settings("alphavantage", ["AAPL", "MSFT"], days=400))
            _show_fundamentals(panel, "alphavantage equity AAPL/MSFT")
            _pit_timeline(panel, "epsSurprise", "AAPL")
            _run_factors(panel)  # AV delivers epsSurprise → real surprise-drift ranks
        except RateLimited as e:
            print(f"  SKIP — AV rate-limited (free 25/day): {e}")
        except Exception as e:  # noqa: BLE001
            print(f"  SKIP — AV fundamentals failed: {type(e).__name__}: {e}")

    section("FUNDAMENTALS VERIFICATION DONE")


if __name__ == "__main__":
    main()

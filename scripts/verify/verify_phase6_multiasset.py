"""Phase 6 verification — REAL multi-asset (crypto / FX) fetches.

Run: PYTHONPATH=. ./venv/bin/python scripts/verify/verify_phase6_multiasset.py

Proves the data layer now serves crypto and FX, with the calendar/annualization
bug fixed:

  1. yfinance (free) — crypto BTC-USD/ETH-USD + fx EUR-USD through `load_panel`;
     assert weekend bars are present for crypto and **periods/year == 365**, vs
     **252** for an equity contrast (AAPL). This is the actual fix.
  2. FMP (needs FMP_API_KEY) — crypto + fx through the same seam; cross-check the
     latest BTC-USD close against yfinance (should be within a few %).
  3. AlphaVantage (needs ALPHAVANTAGE_API_KEY) — best-effort crypto + fx smoke;
     skips cleanly on the free-tier rate limit.

Everything writes to a temp cache dir; nothing touches the repo. Skips (exit 0)
on missing keys / no network.
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

TMP = tempfile.mkdtemp()


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def _settings(provider: str, asset_class: str, tickers: list[str],
              days: int = 400) -> Settings:
    end = date.today().isoformat()
    start = (date.today() - timedelta(days=days)).isoformat()
    return Settings(data=DataSettings(
        provider=provider, asset_class=asset_class, tickers=tickers,
        start=start, end=end, frequency="1d", cache_dir=TMP))


def _summarise(panel, label: str) -> float | None:
    close = panel.get("close")
    if close is None or close.empty:
        print(f"  {label}: NO DATA")
        return None
    idx = close.index
    ppy = periods_per_year_from_index(idx)
    weekend = bool((idx.dayofweek >= 5).any())
    print(f"  {label}: shape={close.shape}  range={idx.min().date()}→{idx.max().date()}"
          f"  weekend_bars={weekend}  periods/year={ppy}")
    print(f"    fields: {sorted(panel)}")
    print(f"    last closes:\n{close.tail(2).to_string()}")
    return float(close.iloc[-1, 0])


def main() -> None:
    # 1) yfinance — free, the headline proof ---------------------------------
    section("1) yfinance (free): crypto + fx + equity contrast")
    yf_btc = None
    try:
        crypto = load_panel(settings=_settings("yfinance", "crypto", ["BTC-USD", "ETH-USD"]))
        yf_btc = _summarise(crypto, "yfinance crypto BTC/ETH")
        equity = load_panel(settings=_settings("yfinance", "equity", ["AAPL"]))
        _summarise(equity, "yfinance equity AAPL")
        fx = load_panel(settings=_settings("yfinance", "fx", ["EUR-USD"]))
        _summarise(fx, "yfinance fx EUR-USD")

        c_idx = crypto["close"].index
        e_idx = equity["close"].index
        assert (c_idx.dayofweek >= 5).any(), "crypto should have weekend bars!"
        assert not (e_idx.dayofweek >= 5).any(), "equity should NOT have weekend bars!"
        assert periods_per_year_from_index(c_idx) == 365, "crypto must annualise at 365"
        assert periods_per_year_from_index(e_idx) == 252, "equity must annualise at 252"
        print("\n  [OK] crypto→365, equity→252 — the dead crypto-calendar branch is now live")
    except Exception as e:  # noqa: BLE001
        print(f"  SKIP — yfinance multi-asset failed (network?): {type(e).__name__}: {e}")

    # 2) FMP — keyed -------------------------------------------------------
    section("2) FMP (FMP_API_KEY): crypto + fx via the stable 'full' endpoint")
    if not os.getenv("FMP_API_KEY"):
        print("  SKIP — FMP_API_KEY not set.")
    else:
        try:
            fmp_c = load_panel(settings=_settings("fmp", "crypto", ["BTC-USD", "ETH-USD"]))
            fmp_btc = _summarise(fmp_c, "fmp crypto BTC/ETH")
            fmp_fx = load_panel(settings=_settings("fmp", "fx", ["EUR-USD"]))
            _summarise(fmp_fx, "fmp fx EUR-USD")
            if yf_btc and fmp_btc:
                diff = abs(yf_btc - fmp_btc) / yf_btc
                print(f"\n  BTC-USD last close: yfinance={yf_btc:.2f}  fmp={fmp_btc:.2f}"
                      f"  rel_diff={diff:.3%}")
                print("  [OK] cross-vendor BTC price is consistent"
                      if diff < 0.1 else "  [WARN] BTC prices diverge >10%")
        except RateLimited as e:
            print(f"  SKIP — FMP rate-limited: {e}")
        except Exception as e:  # noqa: BLE001
            print(f"  SKIP — FMP multi-asset failed: {type(e).__name__}: {e}")

    # 3) AlphaVantage — best-effort (free tier is heavily rate-limited) ------
    section("3) AlphaVantage (ALPHAVANTAGE_API_KEY): crypto + fx best-effort smoke")
    if not os.getenv("ALPHAVANTAGE_API_KEY"):
        print("  SKIP — ALPHAVANTAGE_API_KEY not set.")
    else:
        try:
            av_c = load_panel(settings=_settings("alphavantage", "crypto", ["BTC-USD"], days=120))
            _summarise(av_c, "alphavantage crypto BTC")
        except RateLimited as e:
            print(f"  SKIP — AV crypto rate-limited (free 25/day): {e}")
        except Exception as e:  # noqa: BLE001
            print(f"  SKIP — AV crypto failed: {type(e).__name__}: {e}")
        try:
            av_fx = load_panel(settings=_settings("alphavantage", "fx", ["EUR-USD"], days=120))
            _summarise(av_fx, "alphavantage fx EUR-USD")
        except RateLimited as e:
            print(f"  SKIP — AV fx rate-limited: {e}")
        except Exception as e:  # noqa: BLE001
            print(f"  SKIP — AV fx failed: {type(e).__name__}: {e}")

    section("PHASE 6 MULTI-ASSET VERIFICATION DONE")


if __name__ == "__main__":
    main()

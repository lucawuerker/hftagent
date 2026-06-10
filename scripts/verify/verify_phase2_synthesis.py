"""Phase 2 verification — vwap/returns synthesis on live LOBSTER data.

Run: PYTHONPATH=. ./venv/bin/python scripts/verify/verify_phase2_synthesis.py

Shows that requesting derived fields the LOBSTER loader doesn't natively produce
returns them, computed from OHLCV (vwap ≈ (H+L+C)/3, returns = close.pct_change()).
"""

import pandas as pd

from quant_fund_agent.data import load_panel

DATA_DIR = "ticker_data"


def main():
    print("===== load a small live panel asking for vwap/returns + OHLC =====")
    panel = load_panel(DATA_DIR, n_tickers=2,
                       fields=["open", "high", "low", "close", "vwap", "returns"])
    print("fields returned:", sorted(panel))
    print("tickers:", list(panel["close"].columns), "| shape:", panel["close"].shape)

    print("\n===== verify synthesis formulas =====")
    exp_vwap = (panel["high"] + panel["low"] + panel["close"]) / 3.0
    exp_ret = panel["close"].pct_change()
    vwap_ok = bool((panel["vwap"].fillna(0) - exp_vwap.fillna(0)).abs().max().max() < 1e-6)
    ret_ok = bool((panel["returns"].fillna(0) - exp_ret.fillna(0)).abs().max().max() < 1e-9)
    print("vwap == (H+L+C)/3      :", vwap_ok)
    print("returns == close.pct_change():", ret_ok)

    print("\n===== sample rows (first ticker) =====")
    t = panel["close"].columns[0]
    sample = pd.DataFrame({
        "high": panel["high"][t], "low": panel["low"][t], "close": panel["close"][t],
        "vwap": panel["vwap"][t], "returns": panel["returns"][t],
    }).head(4)
    print(sample.to_string())

    print("\n===== targeted load trims to exactly what was requested =====")
    trimmed = load_panel(DATA_DIR, n_tickers=2, fields=["vwap", "close"])
    print("requested ['vwap','close'] -> returned:", sorted(trimmed),
          "(high/low loaded as deps then trimmed)")

    assert {"vwap", "returns"} <= set(panel)
    assert vwap_ok and ret_ok
    assert set(trimmed) == {"vwap", "close"}
    print("\nALL PHASE-2 SYNTHESIS ASSERTIONS PASSED")


if __name__ == "__main__":
    main()

"""Phase 3 verification — onboarding wizard writes a loadable config.

Run: PYTHONPATH=. ./venv/bin/python scripts/verify/verify_phase3_wizard.py

Drives `quant_fund_agent.setup` non-interactively (flags + --no-validate, no
network) to write a temp quant.config.yaml, then loads it back through Settings
and asserts the values round-trip.
"""

import os
import tempfile

from quant_fund_agent import setup as wizard
from quant_fund_agent.config import Settings


def main():
    out = os.path.join(tempfile.mkdtemp(), "quant.config.yaml")
    argv = [
        "--provider", "yfinance", "--preset", "demo",
        "--start", "2023-01-01", "--end", "2025-01-01",
        "--freq", "1d", "--n-tickers", "5",
        "--output", out, "--no-validate", "--yes",
    ]
    print("===== running wizard non-interactively =====")
    print("argv:", " ".join(argv))
    wizard.main(argv)

    print("\n===== written file =====")
    with open(out) as fh:
        print(fh.read())

    print("===== load back via Settings and check round-trip =====")
    s = Settings.load(out)
    print("provider :", s.data.provider)
    print("preset   :", s.data.universe_preset)
    print("freq     :", s.data.frequency)
    print("start/end:", s.data.start, "/", s.data.end)
    print("n_tickers:", s.data.n_tickers)

    assert s.data.provider == "yfinance"
    assert s.data.universe_preset == "demo"
    assert s.data.frequency == "1d"
    assert s.data.start == "2023-01-01" and s.data.end == "2025-01-01"
    assert s.data.n_tickers == 5
    print("\nWIZARD ROUND-TRIP OK")


if __name__ == "__main__":
    main()

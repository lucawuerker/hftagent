"""Guided onboarding wizard — write ``quant.config.yaml`` for a fresh clone.

    python -m quant_fund_agent.setup

Detects which data providers you can use (yfinance needs no key; FMP /
AlphaVantage read keys from ``.env``), asks for a universe / timespan / frequency,
optionally does a small validation fetch, and writes ``quant.config.yaml`` that
the whole agent pipeline then reads (see ``docs/data-layer/ONBOARDING.md``).

Every prompt has a matching CLI flag, so the wizard is fully scriptable and
non-interactive (`--yes` accepts defaults for anything not supplied):

    python -m quant_fund_agent.setup --provider yfinance --preset demo \
        --start 2023-01-01 --end 2025-01-01 --freq 1d --yes

The optional LLM ``--assist`` mode is Phase 5.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, timedelta

from quant_fund_agent.data.providers import PROVIDERS
from quant_fund_agent.data.universe import available_presets

CONFIG_PATH = "quant.config.yaml"


# ── key / provider detection ───────────────────────────────────────────────

def detect_providers() -> dict[str, bool]:
    """Which registered providers are usable right now (key present where needed)."""
    usable: dict[str, bool] = {}
    for name in PROVIDERS:
        if name == "yfinance":
            usable[name] = True            # no key required
        elif name == "lobster":
            usable[name] = True            # uses local CSVs
        elif name == "fmp":
            usable[name] = bool(os.getenv("FMP_API_KEY"))
        elif name == "alphavantage":
            usable[name] = bool(os.getenv("ALPHAVANTAGE_API_KEY"))
        else:
            usable[name] = True
    return usable


# ── prompting helpers ──────────────────────────────────────────────────────

def _ask(prompt: str, default: str, *, interactive: bool) -> str:
    if not interactive:
        return default
    raw = input(f"{prompt} [{default}]: ").strip()
    return raw or default


# ── config assembly ────────────────────────────────────────────────────────

def build_config(args: argparse.Namespace, *, interactive: bool) -> dict:
    usable = detect_providers()
    available = [p for p, ok in usable.items() if ok]

    provider = args.provider or _ask(
        f"Data provider {available}", "yfinance", interactive=interactive)
    if provider not in PROVIDERS:
        raise SystemExit(f"Unknown provider {provider!r}; choose from {list(PROVIDERS)}.")
    if not usable.get(provider, False):
        raise SystemExit(
            f"Provider {provider!r} needs an API key in .env (see .env.example).")

    data: dict = {"provider": provider}

    if provider == "lobster":
        data["data_dir"] = args.data_dir or _ask(
            "LOBSTER data_dir", "ticker_data", interactive=interactive)
        data["frequency"] = args.freq or "10s"
        data["asset_class"] = args.asset_class or "equity"
        return {"data": data}

    # API providers (yfinance / fmp / alphavantage)
    data["asset_class"] = args.asset_class or _ask(
        "Asset class", "equity", interactive=interactive)
    data["frequency"] = args.freq or _ask(
        "Frequency (1d/1h/5m/1m)", "1d", interactive=interactive)

    default_end = date.today().isoformat()
    default_start = (date.today() - timedelta(days=365 * 2)).isoformat()
    data["start"] = args.start or _ask("Start date (YYYY-MM-DD)", default_start,
                                        interactive=interactive)
    data["end"] = args.end or _ask("End date (YYYY-MM-DD)", default_end,
                                   interactive=interactive)
    data["cache_dir"] = args.cache_dir or "data/market"

    # Universe: explicit tickers win over a preset.
    if args.tickers:
        data["tickers"] = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        preset = args.preset or _ask(
            f"Universe preset {available_presets()} (or pass --tickers)",
            "demo", interactive=interactive)
        data["universe_preset"] = preset
    if args.n_tickers:
        data["n_tickers"] = args.n_tickers

    return {"data": data}


def validate_fetch(config: dict) -> tuple[bool, str]:
    """Try a tiny fetch (1 ticker) to confirm the provider/universe resolves."""
    try:
        import dataclasses

        from quant_fund_agent.config import DataSettings, Settings
        from quant_fund_agent.data import load_panel

        d = dict(config["data"])
        d["n_tickers"] = 1
        settings = Settings(data=DataSettings(**{
            k: v for k, v in d.items() if k in {f.name for f in dataclasses.fields(DataSettings)}
        }))
        panel = load_panel(fields=["close"], settings=settings)
        close = panel.get("close")
        if close is None or close.empty:
            return False, "fetch returned no data"
        return True, f"ok — {close.shape[0]} bars × {close.shape[1]} ticker(s)"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def write_config(config: dict, path: str) -> None:
    import yaml

    header = (
        "# Written by `python -m quant_fund_agent.setup`. Secrets stay in .env.\n"
        "# Edit freely; env vars (DATA_DIR, ARCHITECT_N_TICKERS, QF_DATA_PROVIDER)\n"
        "# still override these values. See docs/data-layer/ONBOARDING.md.\n"
    )
    with open(path, "w") as fh:
        fh.write(header)
        yaml.safe_dump(config, fh, sort_keys=False, default_flow_style=False)


# ── entry point ────────────────────────────────────────────────────────────

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QuantFundAgent onboarding wizard.")
    p.add_argument("--provider", choices=list(PROVIDERS))
    p.add_argument("--asset-class", dest="asset_class")
    p.add_argument("--freq")
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--preset", help=f"universe preset {available_presets()}")
    p.add_argument("--tickers", help="comma-separated tickers (overrides --preset)")
    p.add_argument("--n-tickers", type=int, dest="n_tickers")
    p.add_argument("--data-dir", dest="data_dir", help="LOBSTER provider only")
    p.add_argument("--cache-dir", dest="cache_dir")
    p.add_argument("--output", default=CONFIG_PATH, help="config file to write")
    p.add_argument("--no-validate", action="store_true",
                   help="skip the small validation fetch")
    p.add_argument("--yes", "-y", action="store_true",
                   help="non-interactive: accept defaults for anything not given")
    return p.parse_args(argv)


def main(argv=None) -> None:
    from dotenv import load_dotenv

    load_dotenv()
    args = _parse_args(argv)
    interactive = sys.stdin.isatty() and not args.yes

    print("QuantFundAgent setup — detecting providers…")
    for name, ok in detect_providers().items():
        print(f"  {name:<14s} {'available' if ok else 'needs API key (.env)'}")
    print()

    config = build_config(args, interactive=interactive)

    if not args.no_validate:
        print("Validating with a small fetch…")
        ok, msg = validate_fetch(config)
        print(f"  {'OK' if ok else 'WARNING'}: {msg}")
        if not ok and interactive:
            if input("Fetch failed — write config anyway? [y/N]: ").strip().lower() != "y":
                raise SystemExit("Aborted; no config written.")

    write_config(config, args.output)
    print(f"\nWrote {args.output}:")
    import yaml
    print(yaml.safe_dump(config, sort_keys=False).rstrip())
    print("\nNext: ./venv/bin/python run_fund.py --n-strategies 1")


if __name__ == "__main__":
    main()

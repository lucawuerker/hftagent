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

Optional ``--assist`` adds an LLM layer that turns a plain-English description
into a *proposed* config you then confirm field by field (needs an LLM key, e.g.
``OPENAI_API_KEY``):

    python -m quant_fund_agent.setup --assist "tech mega-caps, last 18 months, daily"

The proposal is only ever a set of suggested defaults — precedence stays
**CLI flag > LLM proposal > built-in default** — and the wizard always works
without an LLM (see ``setup_assist.py``).
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, timedelta

from quant_fund_agent.data.providers import PROVIDERS, get_provider_class
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

def build_config(
    args: argparse.Namespace, *, interactive: bool, proposal: dict | None = None
) -> dict:
    proposal = proposal or {}
    usable = detect_providers()
    available = [p for p, ok in usable.items() if ok]

    # Precedence per field: explicit CLI flag > LLM proposal (shown as the prompt
    # default the user confirms/overrides) > built-in default.
    def pick(cli_value, key: str, prompt: str, default: str) -> str:
        if cli_value is not None:
            return cli_value
        return _ask(prompt, str(proposal.get(key, default)), interactive=interactive)

    provider = pick(args.provider, "provider", f"Data provider {available}", "yfinance")
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

    # API providers (yfinance / fmp / alphavantage) — multi-asset.
    classes = list(get_provider_class(provider).asset_classes)
    asset_class = pick(args.asset_class, "asset_class", f"Asset class {classes}", classes[0])
    if asset_class not in classes:
        raise SystemExit(
            f"Provider {provider!r} serves {classes}, not {asset_class!r}.")
    data["asset_class"] = asset_class
    data["frequency"] = pick(args.freq, "frequency", "Frequency (1d/1h/5m/1m)", "1d")

    default_end = date.today().isoformat()
    default_start = (date.today() - timedelta(days=365 * 2)).isoformat()
    data["start"] = pick(args.start, "start", "Start date (YYYY-MM-DD)", default_start)
    data["end"] = pick(args.end, "end", "End date (YYYY-MM-DD)", default_end)
    data["cache_dir"] = args.cache_dir or "data/market"

    # Universe: explicit tickers (CLI flag, then LLM proposal) win over a preset.
    # The default preset matches the asset class (crypto→crypto_demo, fx→fx_demo).
    if args.tickers:
        data["tickers"] = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    elif proposal.get("tickers"):
        data["tickers"] = list(proposal["tickers"])
    else:
        preset_default = {"crypto": "crypto_demo", "fx": "fx_demo"}.get(asset_class, "demo")
        if preset_default not in available_presets():
            preset_default = available_presets()[0]
        preset = pick(args.preset, "preset",
                      f"Universe preset {available_presets()} (or pass --tickers)",
                      preset_default)
        data["universe_preset"] = preset
    n_tickers = args.n_tickers or proposal.get("n_tickers")
    if n_tickers:
        data["n_tickers"] = int(n_tickers)

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
    p.add_argument("--assist", nargs="?", const="", metavar="DESCRIPTION",
                   help="LLM-assisted setup: describe your fund in plain English and "
                        "an LLM proposes a config you then confirm (needs an LLM key). "
                        "Pass the text inline, or use the flag alone to be prompted.")
    p.add_argument("--no-validate", action="store_true",
                   help="skip the small validation fetch")
    p.add_argument("--yes", "-y", action="store_true",
                   help="non-interactive: accept defaults for anything not given")
    return p.parse_args(argv)


def _run_assist(
    args: argparse.Namespace, usable: dict[str, bool], *, interactive: bool
) -> dict:
    """If ``--assist`` was given, return a validated LLM proposal (``{}`` otherwise)."""
    if args.assist is None:
        return {}

    description = args.assist
    if not description and interactive:
        description = input(
            "Describe your fund (universe, timespan, frequency): ").strip()
    if not description:
        print("  (no description given — falling back to the standard wizard)\n")
        return {}

    from quant_fund_agent.setup_assist import propose_config

    available = [p for p, ok in usable.items() if ok]
    print("Asking the assistant to draft a config…")
    proposal = propose_config(
        description, available=available, presets=available_presets())
    if proposal:
        print("  assistant proposal (edit/confirm below):")
        for key, value in proposal.items():
            print(f"    {key}: {value}")
    else:
        print("  (assistant unavailable or produced nothing usable — using defaults)")
    print()
    return proposal


def main(argv=None) -> None:
    from dotenv import load_dotenv

    load_dotenv()
    args = _parse_args(argv)
    interactive = sys.stdin.isatty() and not args.yes

    print("QuantFundAgent setup — detecting providers…")
    usable = detect_providers()
    for name, ok in usable.items():
        print(f"  {name:<14s} {'available' if ok else 'needs API key (.env)'}")
    print()

    proposal = _run_assist(args, usable, interactive=interactive)

    config = build_config(args, interactive=interactive, proposal=proposal)

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

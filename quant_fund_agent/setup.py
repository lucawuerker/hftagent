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


def _menu_select(prompt: str, options: list[str], default: str) -> str:
    """Arrow-key single-select menu (Claude-Code style).

    Renders ``options`` as rows; ↑/↓ (or k/j) move the highlight, Enter
    confirms and Ctrl-C aborts. Degrades gracefully to a typed ``_ask`` prompt
    when the terminal can't deliver raw keystrokes (e.g. piped stdin), so the
    wizard still works over a pipe or in a dumb terminal.
    """
    options = list(options)
    if not options:
        return default
    try:
        import termios
        import tty

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
    except Exception:  # noqa: BLE001 — no real TTY; fall back to typing
        return _ask(f"{prompt} {options}", default, interactive=True)

    idx = options.index(default) if default in options else 0

    def _render(first: bool) -> None:
        if not first:
            sys.stdout.write(f"\x1b[{len(options)}A")   # cursor up to the first row
        for i, opt in enumerate(options):
            pointer = "❯" if i == idx else " "
            row = f" {pointer} {opt}"
            if i == idx:
                row = f"\x1b[36m{row}\x1b[0m"            # cyan highlight
            sys.stdout.write("\x1b[2K" + row + "\r\n")   # clear line, then draw
        sys.stdout.flush()

    print(f"{prompt}  (↑/↓ to move, Enter to select)")
    try:
        tty.setraw(fd)
        _render(first=True)
        while True:
            ch = sys.stdin.read(1)
            if ch in ("\r", "\n"):
                break
            if ch == "\x03":                              # Ctrl-C
                raise KeyboardInterrupt
            if ch == "\x1b" and sys.stdin.read(1) == "[":  # arrow-key escape seq
                arrow = sys.stdin.read(1)
                if arrow == "A":
                    idx = (idx - 1) % len(options)
                elif arrow == "B":
                    idx = (idx + 1) % len(options)
            elif ch == "k":
                idx = (idx - 1) % len(options)
            elif ch == "j":
                idx = (idx + 1) % len(options)
            _render(first=False)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    print(f"  → {options[idx]}\n")
    return options[idx]


def _choose_universe(
    *, cli_preset, cli_tickers, proposal: dict, presets: list[str],
    default_preset: str, interactive: bool,
) -> dict:
    """Resolve the universe into ``{'tickers': [...]}`` or ``{'universe_preset': name}``.

    Explicit tickers (CLI flag, then LLM proposal) win over a preset. Otherwise
    the user picks a preset from an arrow-key menu, or chooses the manual-entry
    row to type a comma-separated ticker list.
    """
    if cli_tickers:
        return {"tickers": [t.strip().upper() for t in cli_tickers.split(",") if t.strip()]}
    if proposal.get("tickers"):
        return {"tickers": list(proposal["tickers"])}
    if cli_preset is not None:
        return {"universe_preset": cli_preset}

    default = str(proposal.get("preset", default_preset))
    if not interactive:
        return {"universe_preset": default}

    manual = "⌨  type tickers manually"
    choice = _menu_select("Universe preset", presets + [manual], default)
    if choice == manual:
        raw = _ask("Comma-separated tickers", "", interactive=True)
        tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]
        if tickers:
            return {"tickers": tickers}
        return {"universe_preset": default}    # empty entry → keep the default preset
    return {"universe_preset": choice}


def _ask_yes_no(prompt: str, default: bool, *, interactive: bool) -> bool:
    """Yes/no question via the arrow-key menu (default highlighted)."""
    default_label = "yes" if default else "no"
    if not interactive:
        return default
    return _menu_select(prompt, ["yes", "no"], default_label) == "yes"


# Human-readable LOBSTER level menu → the integer written to config.
_LOBSTER_LEVEL_LABELS = {
    "level 3 — full message stream (trades, order flow, hidden, auctions)": 3,
    "level 2 — order book only (prices/sizes, spread, depth, imbalance + volume)": 2,
}


def _choose_lobster_level(
    cli_value: int | None, proposal: dict, *, interactive: bool
) -> int:
    """Resolve the LOBSTER order-book level (CLI > proposal > menu/default 3)."""
    if cli_value is not None:
        return int(cli_value)
    default = int(proposal.get("lobster_level", 3) or 3)
    if not interactive:
        return default
    labels = list(_LOBSTER_LEVEL_LABELS)
    default_label = next(
        (lbl for lbl, lvl in _LOBSTER_LEVEL_LABELS.items() if lvl == default), labels[0]
    )
    choice = _menu_select("LOBSTER data level", labels, default_label)
    return _LOBSTER_LEVEL_LABELS[choice]


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

    def pick_menu(cli_value, key: str, prompt: str, options: list[str], default: str) -> str:
        """Same precedence as ``pick`` but offers an arrow-key menu interactively."""
        if cli_value is not None:
            return cli_value
        default = str(proposal.get(key, default))
        if not interactive:
            return default
        return _menu_select(prompt, options, default)

    provider_default = "yfinance" if "yfinance" in available else (
        available[0] if available else "yfinance")
    provider = pick_menu(args.provider, "provider", "Data provider", available, provider_default)
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

        # Order-book level → which fields the Factor Researcher may use.  Level 2
        # is the order book only (+ traded volume); level 3 adds the trade /
        # message-stream fields (hidden, orderFlow, …) that can't be derived
        # from the book.  Default 3 (full feed) keeps prior behaviour.
        data["lobster_level"] = _choose_lobster_level(
            args.lobster_level, proposal, interactive=interactive)

        # Universe: explicit tickers (CLI flag or LLM proposal) win over a preset;
        # default preset is "lobster" (all 10 supported ETFs). Interactively the
        # user picks from a menu or chooses the manual-entry row to type tickers.
        lobster_presets = available_presets()
        preset_default = "lobster" if "lobster" in lobster_presets else lobster_presets[0]
        data.update(_choose_universe(
            cli_preset=args.preset, cli_tickers=args.tickers, proposal=proposal,
            presets=lobster_presets, default_preset=preset_default,
            interactive=interactive,
        ))
        if args.n_tickers or proposal.get("n_tickers"):
            data["n_tickers"] = int(args.n_tickers or proposal["n_tickers"])

        return {"data": data}

    # API providers (yfinance / fmp / alphavantage) — multi-asset.
    classes = list(get_provider_class(provider).asset_classes)
    asset_class = pick_menu(args.asset_class, "asset_class", "Asset class", classes, classes[0])
    if asset_class not in classes:
        raise SystemExit(
            f"Provider {provider!r} serves {classes}, not {asset_class!r}.")
    data["asset_class"] = asset_class
    data["frequency"] = pick_menu(
        args.freq, "frequency", "Frequency", ["1d", "1h", "5m", "1m"], "1d")

    # Non-OHLCV (fundamentals / estimates / events) are equity-only and served
    # only by FMP / AlphaVantage.  Ask whether to use them so the Factor
    # Researcher only builds fundamental factors when the data is actually there.
    if asset_class == "equity" and provider in {"fmp", "alphavantage"}:
        if args.fundamentals is not None:
            data["fundamentals"] = args.fundamentals == "yes"
        else:
            data["fundamentals"] = _ask_yes_no(
                "Use fundamental data (sector / peRatio / roe / epsSurprise / …)?",
                bool(proposal.get("fundamentals", True)),
                interactive=interactive,
            )

    default_end = date.today().isoformat()
    default_start = (date.today() - timedelta(days=365 * 2)).isoformat()
    data["start"] = pick(args.start, "start", "Start date (YYYY-MM-DD)", default_start)
    data["end"] = pick(args.end, "end", "End date (YYYY-MM-DD)", default_end)
    data["cache_dir"] = args.cache_dir or "data/market"

    # Universe: explicit tickers (CLI flag, then LLM proposal) win over a preset.
    # The default preset matches the asset class (crypto→crypto_demo, fx→fx_demo).
    preset_default = {"crypto": "crypto_demo", "fx": "fx_demo"}.get(asset_class, "demo")
    if preset_default not in available_presets():
        preset_default = available_presets()[0]
    data.update(_choose_universe(
        cli_preset=args.preset, cli_tickers=args.tickers, proposal=proposal,
        presets=available_presets(), default_preset=preset_default,
        interactive=interactive,
    ))
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
    p.add_argument("--lobster-level", dest="lobster_level", type=int, choices=[2, 3],
                   help="LOBSTER order-book level available: 2 = order book only "
                        "(prices/sizes, spread, depth, imbalance + volume), 3 = full "
                        "message stream (trades, order flow, hidden, auctions). "
                        "Gates which fields the Factor Researcher may use.")
    p.add_argument("--fundamentals", choices=["yes", "no"], dest="fundamentals",
                   help="Use non-OHLCV fundamental/estimate/event data (equity FMP / "
                        "AlphaVantage only). Default: yes where the provider serves it.")
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

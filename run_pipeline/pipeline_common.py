"""Shared setup + pretty-printing for the full-pipeline runner scripts.

The scripts in this folder run the quant-fund pipeline **for real** on the local
order-book / price panel (``ticker_data/``): the Factor Researcher's IC backtest,
the Architect's fit + backtest refinement loop, the Statistician's out-of-sample
tests and the Portfolio Manager's allocation all actually execute, and the
results are written to the **real** databases (``data/factors``,
``data/strategies``, ``data/portfolio``).

This is the up-to-date sibling of two existing example sets:

* ``prelim_files/`` — older standalone scripts that predate
  :mod:`quant_fund_agent.pipeline`; these scripts replace them and drive the
  agents through that reusable seam instead of re-implementing orchestration.
* ``showcase_pipeline/`` — clean per-step scripts that deliberately *stop before*
  any market-data step and write to sandbox DB copies (so the public repo needs
  no proprietary data). These scripts go all the way through, on real data.

Importing this module (it must be the FIRST import in every script) configures
the environment so the agents run with no external infrastructure by default:

* ``QF_USE_MCP=0`` — run every agent tool in-process, so no MCP stdio servers
  need to be spawned.  Pass ``--mcp`` on any script to flip this back to ``1``
  (the production path) via :func:`configure_runtime`.
* Working directory → repository root, so all the relative data paths resolve.
* ``.env`` is loaded (for ``OPENAI_API_KEY``).

Unlike ``showcase_common`` it does **not** redirect ``FACTOR_DB_PATH``: the real
factor / strategy / portfolio databases are used.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ── Repository layout ─────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]


def _configure_environment() -> None:
    # Make ``quant_fund_agent`` importable when a script is run directly
    # (``python run_pipeline/2_selector.py`` only puts the script's own folder
    # on sys.path, not the repo root).
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    # Run from the repo root so all the relative data paths resolve.
    os.chdir(REPO_ROOT)

    # In-process tools by default: no MCP servers required.  ``--mcp`` flips
    # this via configure_runtime() *before* any agent module is imported.
    os.environ.setdefault("QF_USE_MCP", "0")

    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
    except Exception:  # python-dotenv missing is non-fatal; env may already be set
        pass


_configure_environment()


# ── Runtime configuration (call BEFORE importing any agent module) ────────
def configure_runtime(
    n_tickers: int | None = None,
    data_dir: str | None = None,
    use_mcp: bool = False,
) -> None:
    """Set the env knobs the agents read at import time.

    ``ARCHITECT_N_TICKERS`` (shared by the architect graph and the modeling
    service) and ``DATA_DIR`` must be set *before* those modules are imported,
    so every script calls this right after parsing args and before its lazy
    ``from quant_fund_agent...`` imports.

    * ``n_tickers`` — universe cap; ``None`` or ``0`` leaves it unset (full
      universe).  ``ticker_data/`` is several GB across 50+ tickers, so a small
      cap (e.g. 8–15) keeps memory and runtime sane for a demo.
    * ``data_dir`` — overrides the panel location (default ``ticker_data``).
    * ``use_mcp`` — when True, run agent tools over their MCP stdio servers
      instead of in-process.
    """
    if n_tickers:  # truthy → positive cap
        os.environ["ARCHITECT_N_TICKERS"] = str(n_tickers)
    if data_dir:
        os.environ["DATA_DIR"] = data_dir
    if use_mcp:
        os.environ["QF_USE_MCP"] = "1"


# ── Pre-flight checks ─────────────────────────────────────────────────────
def require_openai_key() -> None:
    """Exit early with a helpful message if no OpenAI key is configured."""
    if not os.getenv("OPENAI_API_KEY"):
        print(
            "ERROR: OPENAI_API_KEY is not set.\n\n"
            "These scripts call an LLM (OpenAI) to brainstorm factors, select\n"
            "factors, design strategies and (optionally) moderate the PM\n"
            "committee.  Put your key in a .env file at the repository root:\n\n"
            "    OPENAI_API_KEY=sk-...\n",
            file=sys.stderr,
        )
        sys.exit(1)


def require_market_data(data_dir: str = "ticker_data") -> None:
    """Exit early if the local price/order-book panel is missing.

    Unlike the showcase scripts, everything here needs the proprietary panel
    (IC backtest, architect fit, OOS tests).  Fail fast with a clear message
    rather than deep inside a data loader.
    """
    p = Path(data_dir)
    if not p.is_dir() or not any(p.iterdir()):
        print(
            f"ERROR: market-data directory '{data_dir}' is missing or empty.\n\n"
            "The full pipeline runs the IC backtest, the Architect's fit +\n"
            "backtest loop and the Statistician's OOS tests, all of which need\n"
            "the local order-book / price panel (not committed to the repo).\n"
            "Point --data-dir at your ticker_data root, or set DATA_DIR.\n",
            file=sys.stderr,
        )
        sys.exit(1)


# ── Shared argparse options ───────────────────────────────────────────────
def add_common_args(parser, *, with_data: bool = True) -> None:
    """Add flags shared across the scripts so they stay consistent.

    ``--mcp`` everywhere; ``--n-tickers`` / ``--data-dir`` on the scripts that
    touch the market-data panel (``with_data=True``).
    """
    parser.add_argument(
        "--mcp", action="store_true",
        help="Run agent tools over MCP stdio servers instead of in-process "
             "(default: in-process, QF_USE_MCP=0).",
    )
    if with_data:
        parser.add_argument(
            "--n-tickers", type=int, default=15,
            help="Universe cap for the market-data backtests "
                 "(sets ARCHITECT_N_TICKERS).  0 = full universe.",
        )
        parser.add_argument(
            "--data-dir", default=os.getenv("DATA_DIR", "ticker_data"),
            help="Path to the ticker_data root (the panel source).",
        )


# ── Pretty printing ───────────────────────────────────────────────────────
RULE = "=" * 80


def banner(title: str) -> None:
    print("\n" + RULE)
    print(title)
    print(RULE)


def section(title: str) -> None:
    print("\n" + "-" * 80)
    print(title)
    print("-" * 80)


def fmt(x, nd: int = 3) -> str:
    """Format a float KPI compactly; '-' for missing/non-numeric."""
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "-"

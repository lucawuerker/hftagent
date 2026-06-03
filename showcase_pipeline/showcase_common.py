"""Shared setup + pretty-printing for the GitHub showcase scripts.

The showcase scripts in this folder demonstrate the *creative* stages of the
quant-fund pipeline — Factor Researcher → Selector → Architect — running them
exactly as the real agents do, but stopping right before the point where they
would need the proprietary order-book / stock-price panel (``ticker_data/``),
which is **not** checked into this repository.

Importing this module (it must be the FIRST import in every showcase script)
configures the environment so the agents run with zero external infrastructure:

* ``QF_USE_MCP=0`` — run every agent tool in-process, so no MCP stdio servers
  need to be started.
* ``FACTOR_DB_PATH`` — point the Factor Researcher (writes) and the Selector
  (reads) at a *separate showcase copy* of the factor database, leaving the
  real ``data/factors/factor_db.json`` untouched.

It also changes the working directory to the repository root so the scripts can
be launched from anywhere, and loads the ``.env`` file (for ``OPENAI_API_KEY``).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

# ── Repository layout ─────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]

# Canonical (real) databases — never written to by the showcase scripts.
FACTOR_DB_REAL = Path("data/factors/factor_db.json")
STRATEGY_DB_REAL = Path("data/strategies/strategy_db.json")

# Separate showcase copies the scripts read from and write to.
FACTOR_DB_SHOWCASE = Path("data/factors/showcase_factor_db.json")
STRATEGY_DB_SHOWCASE = Path("data/strategies/showcase_strategy_db.json")

# Where the Factor Researcher materialises generated factor subclasses.
RESEARCHER_FACTOR_DIR = Path("quant_fund_agent/factors/researcher")


# ── Environment setup (runs once, at import, before any agent import) ─────
def _configure_environment() -> None:
    # Make ``quant_fund_agent`` importable when these scripts are run directly
    # (``python showcase_pipeline/1_factor_research.py`` only puts the script's
    # own folder on sys.path, not the repo root).
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    # Run from the repo root so all the relative data paths resolve.
    os.chdir(REPO_ROOT)

    # In-process tools: no MCP servers required for the supervisor to run this.
    os.environ.setdefault("QF_USE_MCP", "0")

    # Redirect factor-DB reads/writes to the showcase copy (see module docstring).
    os.environ.setdefault("FACTOR_DB_PATH", str(FACTOR_DB_SHOWCASE))

    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
    except Exception:  # python-dotenv missing is non-fatal; env may already be set
        pass


_configure_environment()


# ── Showcase-DB seeding ───────────────────────────────────────────────────
def ensure_showcase_factor_db(reset: bool = False) -> None:
    """Make sure the showcase factor DB exists.

    On first use it is seeded from the real ``factor_db.json`` so the Selector
    sees the full catalog (seed factors + anything the Factor Researcher has
    already added this session).  ``reset=True`` rebuilds it from the real DB.
    """
    FACTOR_DB_SHOWCASE.parent.mkdir(parents=True, exist_ok=True)
    if reset and FACTOR_DB_SHOWCASE.exists():
        FACTOR_DB_SHOWCASE.unlink()

    if FACTOR_DB_SHOWCASE.exists():
        return

    if FACTOR_DB_REAL.exists():
        shutil.copyfile(FACTOR_DB_REAL, FACTOR_DB_SHOWCASE)
        print(f"  (seeded showcase factor DB from {FACTOR_DB_REAL})")
    else:
        FACTOR_DB_SHOWCASE.write_text(json.dumps({"factors": []}, indent=2))
        print(f"  (created empty showcase factor DB at {FACTOR_DB_SHOWCASE})")


# ── Pre-flight checks ─────────────────────────────────────────────────────
def require_openai_key() -> None:
    """Exit early with a helpful message if no OpenAI key is configured."""
    if not os.getenv("OPENAI_API_KEY"):
        print(
            "ERROR: OPENAI_API_KEY is not set.\n\n"
            "These showcase scripts call an LLM (OpenAI) to brainstorm factors,\n"
            "select factors and design strategies.  Put your key in a .env file\n"
            "at the repository root:\n\n"
            "    OPENAI_API_KEY=sk-...\n",
            file=sys.stderr,
        )
        sys.exit(1)


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


def note_skipped_data_step(what: str) -> None:
    """Announce the boundary where the real agent would hit the market data."""
    print("\n" + "·" * 80)
    print(f"SKIPPED (needs market data not in this repo): {what}")
    print("·" * 80)

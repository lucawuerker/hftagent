"""Phase 5 verification — REAL LLM-assisted onboarding (needs an LLM key).

Run: PYTHONPATH=. ./venv/bin/python scripts/verify/verify_phase5_assist.py

Exercises the `--assist` layer end-to-end against a real LLM (OPENAI_API_KEY in
.env, gpt-4o-mini by default):

  1. Live proposals  — three plain-English prompts → validated config proposals,
     proving the LLM output is constrained to legal providers/presets/dates.
  2. Full wizard     — `setup.main(["--assist", "...", "--yes", "--no-validate"])`
     writes a real quant.config.yaml from free text (no network fetch).
  3. Graceful fallback — with the LLM forced to fail, the wizard still produces a
     valid deterministic config (the optional layer never breaks onboarding).

Writes only to a temp dir; never touches the repo's quant.config.yaml. Skips
cleanly (exit 0) if no LLM key is set.
"""

import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from quant_fund_agent.data.universe import available_presets
from quant_fund_agent.setup import detect_providers
from quant_fund_agent.setup_assist import propose_config


def _usable_providers() -> list[str]:
    return [p for p, ok in detect_providers().items() if ok]


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("SKIP — OPENAI_API_KEY not set; --assist needs an LLM key.")
        return

    available = _usable_providers()
    presets = available_presets()
    print(f"usable providers: {available}")
    print(f"universe presets: {presets}")

    # 1) Live proposals from free text ---------------------------------------
    section("1) LIVE LLM PROPOSALS (free text → validated config)")
    prompts = [
        "Apple, Microsoft and Nvidia over the last 18 months, daily bars, use yfinance.",
        "A broad large-cap US equity universe for the past 3 years on daily data.",
        "Mid-cap biotech names, weekly — and pull from Bloomberg.",  # invalid provider/freq
    ]
    for text in prompts:
        print(f"\nPROMPT: {text}")
        proposal = propose_config(text, available=available, presets=presets)
        print(f"  → proposal: {proposal}")
        # Invariants: nothing illegal can survive validation.
        if "provider" in proposal:
            assert proposal["provider"] in available, "leaked an unusable provider!"
        if "preset" in proposal:
            assert proposal["preset"] in presets, "leaked a non-existent preset!"
        if "start" in proposal and "end" in proposal:
            assert proposal["start"] < proposal["end"], "incoherent date range!"
    print("\n  [OK] every live proposal stayed within legal providers/presets/dates")

    # 2) Full wizard writes a config from free text --------------------------
    section("2) FULL WIZARD: --assist writes quant.config.yaml from free text")
    from quant_fund_agent import setup as wizard

    tmp = Path(tempfile.mkdtemp())
    out = tmp / "quant.config.yaml"
    wizard.main([
        "--assist", "AAPL, MSFT, NVDA over the last year on daily data via yfinance",
        "--yes", "--no-validate", "--output", str(out),
    ])
    print("\n----- written config -----")
    print(out.read_text().rstrip())
    written = out.read_text()
    assert "provider: yfinance" in written, "assist did not drive the provider"
    print("\n  [OK] wizard produced a config from a plain-English description")

    # 3) Graceful fallback when the LLM is unavailable -----------------------
    section("3) GRACEFUL FALLBACK: LLM failure → deterministic config still written")
    import quant_fund_agent.llm as llm_mod

    original = llm_mod.make_chat_llm
    llm_mod.make_chat_llm = lambda **k: (_ for _ in ()).throw(RuntimeError("forced LLM outage"))
    try:
        out2 = tmp / "fallback.config.yaml"
        wizard.main([
            "--assist", "anything at all",
            "--provider", "yfinance", "--preset", "demo",
            "--yes", "--no-validate", "--output", str(out2),
        ])
        print("\n----- written config (fallback) -----")
        print(out2.read_text().rstrip())
        assert "provider: yfinance" in out2.read_text()
        print("\n  [OK] LLM outage did not break onboarding — deterministic config written")
    finally:
        llm_mod.make_chat_llm = original

    section("PHASE 5 ASSIST VERIFICATION OK")


if __name__ == "__main__":
    main()

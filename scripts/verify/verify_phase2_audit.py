"""Phase 2 verification — surface the known misdeclared-inputs factors.

Run: PYTHONPATH=. ./venv/bin/python scripts/verify/verify_phase2_audit.py

Gating trusts each factor's DECLARED `inputs`. This audit lists factors whose
code accesses fields not in their declared inputs — a small, known set we chose
NOT to auto-fix (decision D2): they stay `standard` and degrade gracefully, as
they already do on LOBSTER. Listed here so the limitation is visible, not hidden.
"""

import inspect
import re

from quant_fund_agent.data.tiers import all_known_fields
from quant_fund_agent.factors import discover_factors
from quant_fund_agent.factors.registry import get_all_factor_classes


def main():
    discover_factors()
    classes = get_all_factor_classes()
    known = set(all_known_fields())

    bad = []
    for fid, cls in classes.items():
        try:
            src = inspect.getsource(cls)
        except Exception:
            continue
        accessed = set(re.findall(r"""data\[['\"]([a-zA-Z_]+)['\"]\]""", src))
        declared = set(getattr(cls, "inputs", ["close"]) or ["close"])
        undeclared = (accessed & known) - declared
        if undeclared:
            bad.append((fid, sorted(undeclared), sorted(declared)))

    print(f"===== misdeclared-inputs audit: {len(bad)}/{len(classes)} factors =====")
    for fid, und, dec in bad:
        print(f"  {fid}: accesses {und} not in declared {dec}")
    print("\nThese 4 are seed alphas needing fundamental classification data that")
    print("LOBSTER lacks; they degrade gracefully today and are left as-is (D2).")
    print("Revisit when a provider (FMP) supplies the `fundamental` tier.")


if __name__ == "__main__":
    main()

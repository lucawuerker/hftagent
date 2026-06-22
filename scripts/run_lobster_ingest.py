"""Autonomous LOBSTER ingest: download → convert → delete, chunk by chunk.

Drives :class:`...lobster_ingest.orchestrator.Orchestrator` from a YAML config
(see ``docs/lobster-ingestion/ingest.config.example.yaml``).  Keeps raw scratch
under a disk budget by deleting each chunk's raw files after conversion.

Typical sequence
----------------
::

    # 1. Plan only — chunk list + disk math, no network (safe to run anytime).
    ./venv/bin/python scripts/run_lobster_ingest.py --config my.ingest.yaml --dry-run

    # 2. One-time browser login; saves a cookie session for headless runs.
    ./venv/bin/python scripts/run_lobster_ingest.py --config my.ingest.yaml --recon

    # 3. Run the full pull (Playwright portal). Resumable: re-run to continue.
    ./venv/bin/python scripts/run_lobster_ingest.py --config my.ingest.yaml

    # Manual fallback: you download archives into a drop folder; it auto-processes.
    ./venv/bin/python scripts/run_lobster_ingest.py --config my.ingest.yaml \
        --manual --drop-dir data/lobster_drop
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quant_fund_agent.data.lobster_ingest.orchestrator import (  # noqa: E402
    IngestConfig,
    Orchestrator,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="ingest YAML config")
    ap.add_argument("--dry-run", action="store_true", help="plan chunks + disk math only (no network)")
    ap.add_argument("--recon", action="store_true", help="one-time browser login to save the cookie session")
    ap.add_argument("--place-raw", action="store_true",
                    help="submit one RAW requestdata.php request per ticker (message+orderbook "
                         "→ ALL 19 columns). This is the product for the full microstructure panel.")
    ap.add_argument("--place-bulk", action="store_true",
                    help="submit a bulk request — SAMPLED order book only (no flow/trade columns). "
                         "Rarely what you want; use --place-raw for all columns.")
    ap.add_argument("--ingest", action="store_true",
                    help="download + stream-convert + delete every finished order on mydata.php (default action)")
    ap.add_argument("--run-live", action="store_true",
                    help="request planned RAW chunks, ingest each when ready, delete server archives, and continue")
    ap.add_argument("--only-raw", action="store_true",
                    help="with --ingest, skip sampled (interval>0) orders")
    ap.add_argument("--interval", type=int, default=None,
                    help="override the config snapshot interval for --place-bulk (0 = raw)")
    ap.add_argument("--manual", action="store_true", help="use the watch-folder portal instead of Playwright")
    ap.add_argument("--drop-dir", default="data/lobster_drop", help="drop folder for --manual mode")
    ap.add_argument("-v", "--verbose", action="store_true", help="DEBUG logging")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = IngestConfig.from_yaml(args.config)

    if args.recon:
        from quant_fund_agent.data.lobster_ingest.lobster_portal import (
            LobsterPortalConfig,
            PlaywrightLobsterPortal,
        )

        PlaywrightLobsterPortal(LobsterPortalConfig(storage_state=f"{cfg.scratch_dir}/.lobster_state.json")).recon()
        return

    if args.dry_run:
        Orchestrator(cfg).dry_run()
        return

    if args.manual:
        from quant_fund_agent.data.lobster_ingest.lobster_portal import WatchFolderPortal

        portal = WatchFolderPortal(
            drop_dir=args.drop_dir,
            poll_interval_s=cfg.poll_interval_s,
            poll_timeout_s=cfg.poll_timeout_s,
        )
        Orchestrator(cfg, portal=portal).run()
        return

    # Live Playwright flow against the real LOBSTER site.
    from quant_fund_agent.data.lobster_ingest.lobster_portal import (
        LobsterPortalConfig,
        PlaywrightLobsterPortal,
    )

    portal = PlaywrightLobsterPortal(
        LobsterPortalConfig(
            storage_state=f"{cfg.scratch_dir}/.lobster_state.json",
            headless=True,
        ),
        poll_interval_s=cfg.poll_interval_s,
        poll_timeout_s=cfg.poll_timeout_s,
    )
    orch = Orchestrator(cfg, portal=portal)

    if args.run_live:
        orch.run_live(portal)
        return

    if args.place_raw:
        orch.place_raw(portal)
        print("Raw request(s) submitted. LOBSTER prepares data asynchronously; "
              "re-run with --ingest (or no flag) once they appear on mydata.php.")
        return

    if args.place_bulk:
        orch.place_bulk(portal, interval=args.interval)
        print("Bulk (sampled, book-only) request submitted. Use --place-raw for all columns.")
        return

    # default action: collect whatever is ready on mydata.php
    orch.ingest_ready(portal, only_raw=args.only_raw)


if __name__ == "__main__":
    main()

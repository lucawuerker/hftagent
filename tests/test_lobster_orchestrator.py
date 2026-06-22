"""Offline tests for the LOBSTER ingest orchestrator (no network/Playwright).

Covers chunk planning, the disk governor, manifest resumability, and the full
download→convert→delete loop driven by a fake portal that serves the synthetic
day built in ``test_lobster_converter``.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from quant_fund_agent.data.lobster_ingest.orchestrator import (
    Chunk,
    DiskGovernor,
    IngestConfig,
    Manifest,
    Orchestrator,
    plan_chunks,
)
from tests.test_lobster_converter import _write_synthetic_day


def _cfg(tmp: Path, **kw) -> IngestConfig:
    base = dict(
        tickers=["AAA", "BBB"],
        start="2024-01-01",
        end="2024-02-29",
        levels=5,
        chunk="month",
        scratch_dir=str(tmp / "raw"),
        out_dir=str(tmp / "ticker_data"),
        manifest_path=str(tmp / "raw" / "manifest.json"),
    )
    base.update(kw)
    return IngestConfig(**base)


def test_plan_chunks_month():
    cfg = _cfg(Path("/tmp/unused"))
    chunks = plan_chunks(cfg)
    assert len(chunks) == 2 * 2  # 2 tickers × {Jan, Feb}
    jan = next(c for c in chunks if c.ticker == "AAA" and c.start == "2024-01-01")
    assert jan.end == "2024-01-31"
    assert jan.est_gb(cfg) > 0


def test_disk_governor_blocks_when_tight(tmp_path: Path):
    gov = DiskGovernor(tmp_path / "raw", budget_gb=10.0, min_free_gb=5.0)
    ok, _ = gov.check(need_gb=1.0)
    assert ok
    # asking for more than the budget is refused
    blocked, msg = gov.check(need_gb=999.0)
    assert not blocked and "budget" in msg or "floor" in msg


def test_manifest_resumable(tmp_path: Path):
    cfg = _cfg(tmp_path)
    m = Manifest(cfg.manifest_path)
    m.sync(plan_chunks(cfg))
    # mark one done, persist, reload — status survives.
    first = next(iter(m.chunks.values()))
    first.status = "done"
    m.save()
    reloaded = Manifest(cfg.manifest_path)
    assert reloaded.chunks[first.key].status == "done"
    assert first.key not in {c.key for c in reloaded.pending()}


class _FakePortal:
    """Serves the synthetic day for every chunk, ignoring the date range."""

    def __init__(self, tmp: Path):
        self.tmp = tmp

    def fetch(self, ticker, start, end, levels, dest: Path) -> Path:
        dest.mkdir(parents=True, exist_ok=True)
        # Build one LOBSTER-named day pair inside dest.
        msg, ob = _write_synthetic_day(dest)
        msg.rename(dest / f"{ticker}_{start.isoformat()}_34200000_57600000_message_{levels}.csv")
        ob.rename(dest / f"{ticker}_{start.isoformat()}_34200000_57600000_orderbook_{levels}.csv")
        return dest


def test_full_loop_converts_and_deletes_raw(tmp_path: Path):
    cfg = _cfg(tmp_path, tickers=["AAA"], start="2024-01-01", end="2024-01-31")
    orch = Orchestrator(cfg, portal=_FakePortal(tmp_path))
    orch.run()

    # output written, every chunk done, raw scratch cleaned up.
    out = Path(cfg.out_dir) / "AAA" / "bin202401.csv"
    assert out.exists()
    df = pd.read_csv(out)
    assert (df["stock"] == "AAA").all() and len(df) == 2340
    assert all(c.status == "done" for c in orch.manifest.chunks.values())
    chunk_raw = Path(cfg.scratch_dir) / "AAA_2024-01-01_2024-01-31"
    assert not chunk_raw.exists()  # raw deleted after convert

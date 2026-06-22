"""Tests for the raw-LOBSTER → ``ticker_data`` converter.

Two layers:

* a fully deterministic *synthetic* test that hand-builds a tiny message +
  orderbook pair with known events and asserts every column of the first 10s
  bar — this runs anywhere, no large fixture needed;
* an *integration* test against the on-disk GLD level-3 sample (skipped when
  the gitignored sample is absent) that checks schema/shape/ranges and that the
  project loader ingests the result.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_fund_agent.data.lobster_ingest.converter import (
    OUTPUT_COLUMNS,
    convert_archive,
    convert_day,
    convert_folder,
    convert_sampled_day,
    level_field_names,
    output_columns,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
GLD_SAMPLE = REPO_ROOT / "GLD_2026-06-01_2026-06-11_3"

# LOBSTER sentinels for an empty book level.
ASK_EMPTY, BID_EMPTY = 9_999_999_999, -9_999_999_999


def _write_synthetic_day(tmp: Path) -> tuple[Path, Path]:
    """Five messages in the 09:30:00 bar + a row-aligned 2-level book.

    Columns: message = time,event,order_id,size,price×1e4,direction(,null);
    orderbook (2 levels) = AskP1,AskV1,BidP1,BidV1,AskP2,AskV2,BidP2,BidV2.
    """
    messages = [
        # time,   ev, id, size, price,    dir
        (34200.0, 1, 1, 100, 4_000_000, 1),    # buy submit  @ $400.00
        (34201.0, 1, 2, 100, 4_001_000, -1),   # sell submit @ $400.10
        (34202.0, 4, 1, 50, 4_000_000, 1),     # exec of buy-limit → seller-initiated → trade −50
        (34203.0, 5, 0, 30, 4_000_500, 1),     # hidden exec, size 30
        (34205.0, 6, 0, 1000, 4_000_500, 1),   # auction/cross, size 1000
    ]
    books = [
        # AskP1,AskV1, BidP1,BidV1,  AskP2,AskV2, BidP2,BidV2
        (ASK_EMPTY, 0, 4_000_000, 100, ASK_EMPTY, 0, BID_EMPTY, 0),
        (4_001_000, 100, 4_000_000, 100, ASK_EMPTY, 0, BID_EMPTY, 0),
        (4_001_000, 100, 4_000_000, 50, ASK_EMPTY, 0, BID_EMPTY, 0),
        (4_001_000, 100, 4_000_000, 50, ASK_EMPTY, 0, BID_EMPTY, 0),
        (4_001_000, 100, 4_000_000, 50, ASK_EMPTY, 0, BID_EMPTY, 0),
    ]
    msg_path = tmp / "TEST_2024-01-03_34200000_57600000_message_2.csv"
    ob_path = tmp / "TEST_2024-01-03_34200000_57600000_orderbook_2.csv"
    pd.DataFrame(messages).to_csv(msg_path, header=False, index=False)
    pd.DataFrame(books).to_csv(ob_path, header=False, index=False)
    return msg_path, ob_path


def test_convert_day_synthetic(tmp_path: Path):
    msg, ob = _write_synthetic_day(tmp_path)
    df = convert_day(msg, ob, ticker="TEST", date=dt.date(2024, 1, 3))

    # shape + grid — 19 aggregates followed by the 2-level book columns
    assert list(df.columns) == output_columns(2)
    assert len(df) == 2340  # full regular session, 10s bars
    assert df["time"].iloc[0] == "09:30:00"
    assert df["time"].iloc[-1] == "15:59:50"
    assert (df["stock"] == "TEST").all()

    bar0 = df.iloc[0]
    # Per-level book columns (event-time mean over the 5 rows in bar 0).
    # L1 ask price is NaN on row0 (empty level) → mean of the 4 valid $400.10s.
    assert bar0["askPrice1"] == pytest.approx(400.10)
    assert bar0["askDepth1"] == pytest.approx((0 + 100 * 4) / 5)   # 80
    assert bar0["bidPrice1"] == pytest.approx(400.00)
    assert bar0["bidDepth1"] == pytest.approx((100 + 100 + 50 + 50 + 50) / 5)  # 70
    # L2 stays empty across the whole synthetic day.
    assert np.isnan(bar0["askPrice2"]) and np.isnan(bar0["bidPrice2"])
    assert bar0["askDepth2"] == 0 and bar0["bidDepth2"] == 0
    assert bar0["trade"] == -50          # seller-initiated exec of a buy limit
    assert bar0["orderFlow"] == 0        # +100 buy submit − 100 sell submit
    assert bar0["ofLiq"] == 200          # gross submitted volume
    assert bar0["trdLiq"] == 50
    assert bar0["nbTrades"] == 1
    assert bar0["hidden"] == 30
    assert bar0["nbHidden"] == 1
    assert bar0["auction"] == 1000
    assert bar0["nbEvents"] == 5
    assert bar0["mid"] == pytest.approx(400.05)
    assert bar0["midEnd"] == pytest.approx(400.05)
    assert bar0["spread"] == pytest.approx(0.10, abs=1e-6)
    assert bar0["effSpread"] == pytest.approx(0.10, abs=1e-6)  # 2·|400.00−400.05|
    assert bar0["lobImb"] == pytest.approx(0.0, abs=1e-6)
    assert bar0["effLobImb"] == pytest.approx(0.0, abs=1e-6)
    assert bar0["depth"] == pytest.approx(150.0, abs=1e-6)

    # empty bars after the only events: book state forward-filled, flows zeroed.
    bar1 = df.iloc[1]
    assert bar1["mid"] == pytest.approx(400.05)
    assert bar1["trade"] == 0 and bar1["nbEvents"] == 0


def test_imbalances_bounded_synthetic(tmp_path: Path):
    msg, ob = _write_synthetic_day(tmp_path)
    df = convert_day(msg, ob, ticker="TEST", date=dt.date(2024, 1, 3))
    assert df["lobImb"].between(-1, 1).all()
    assert df["effLobImb"].between(-1, 1).all()
    assert (df["spread"] >= 0).all()
    assert df[["mid", "midEnd"]].notna().all().all()


def _write_synthetic_sampled_day(tmp: Path) -> Path:
    """A 2-level sampled-book CSV (header + 10s snapshots over the first minute)."""
    rows = []
    for i, sec in enumerate(range(34_200, 34_260, 10)):  # 09:30:00 … 09:30:50
        t = f"{sec // 3600:02d}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"
        # ask_1, ask_size_1, bid_1, bid_size_1, ask_2, ask_size_2, bid_2, bid_size_2 (×1e4)
        rows.append([t, 4_001_000 + i * 100, 100, 4_000_000, 50 + i,
                     4_002_000, 80, 3_999_000, 90])
    path = tmp / "ZZZ_2024-01-03_34200000_57600000_2_10.csv"
    pd.DataFrame(
        rows,
        columns=["time", "ask_1", "ask_size_1", "bid_1", "bid_size_1",
                 "ask_2", "ask_size_2", "bid_2", "bid_size_2"],
    ).to_csv(path, index=False)
    return path


def test_convert_sampled_day(tmp_path: Path):
    path = _write_synthetic_sampled_day(tmp_path)
    df = convert_sampled_day(path, ticker="ZZZ", date=dt.date(2024, 1, 3))
    assert list(df.columns) == output_columns(2)   # 2-level sampled book
    assert len(df) == 2340 and df["time"].iloc[0] == "09:30:00"
    # book columns (incl. per-level) populated, flow columns empty (no message data)
    for c in ("mid", "midEnd", "spread", "lobImb", "effLobImb", "depth",
              *level_field_names(2)):
        assert df[c].notna().all(), c
    for c in ("trade", "orderFlow", "hidden", "auction", "effSpread",
              "trdLiq", "ofLiq", "nbEvents", "nbHidden", "nbTrades"):
        assert df[c].isna().all(), c
    b0 = df.iloc[0]
    assert b0["mid"] == pytest.approx((400.10 + 400.00) / 2)        # ask_1/bid_1 at 09:30:00
    assert b0["midEnd"] == pytest.approx((400.11 + 400.00) / 2)     # next snapshot's mid
    # per-level prices/depths from the 09:30:00 snapshot
    assert b0["askPrice1"] == pytest.approx(400.10) and b0["bidPrice1"] == pytest.approx(400.00)
    assert b0["askDepth1"] == 100 and b0["bidDepth1"] == 50
    assert b0["askPrice2"] == pytest.approx(400.20) and b0["bidPrice2"] == pytest.approx(399.90)
    assert b0["askDepth2"] == 80 and b0["bidDepth2"] == 90
    assert df["lobImb"].between(-1, 1).all()


def test_convert_archive_streams_raw(tmp_path: Path):
    py7zr = pytest.importorskip("py7zr")
    # Build a 2-day raw archive from synthetic message/orderbook pairs.
    staging = tmp_path / "staging"
    staging.mkdir()
    for day in ("2024-01-03", "2024-01-04"):
        msg, ob = _write_synthetic_day(staging)
        msg.rename(staging / f"AAA_{day}_34200000_57600000_message_2.csv")
        ob.rename(staging / f"AAA_{day}_34200000_57600000_orderbook_2.csv")
    archive = tmp_path / "AAA_2024-01-03_2024-01-04_2.7z"
    with py7zr.SevenZipFile(archive, "w") as z:
        for f in staging.glob("*.csv"):
            z.write(f, arcname=f.name)

    summary = convert_archive(archive, ticker="AAA", out_dir=tmp_path / "out")
    assert summary.days == 2
    df = pd.read_csv(tmp_path / "out" / "AAA" / "bin202401.csv")
    assert list(df.columns) == output_columns(2)
    assert (df.groupby("date").size() == 2340).all()
    assert df["trade"].notna().all()          # raw → flow columns populated
    assert df["nbTrades"].sum() > 0
    assert df["askDepth1"].notna().all()      # per-level book columns present


@pytest.mark.skipif(not GLD_SAMPLE.is_dir(), reason="GLD LOBSTER sample not present")
def test_convert_folder_gld_sample(tmp_path: Path):
    summary = convert_folder(GLD_SAMPLE, ticker="GLD", out_dir=tmp_path)
    assert summary.days == 9
    out = tmp_path / "GLD" / "bin202606.csv"
    assert out.exists()

    df = pd.read_csv(out)
    assert list(df.columns) == output_columns(3)   # GLD sample is level-3
    assert (df.groupby("date").size() == 2340).all()
    assert df["mid"].notna().all() and df["midEnd"].notna().all()
    assert df["lobImb"].between(-1, 1).all()
    assert df["effLobImb"].between(-1, 1).all()
    assert (df["spread"] >= 0).all()
    assert (df["effSpread"] >= 0).all()
    # best-level price columns are well-formed: ask ≥ bid wherever both exist
    both = df["askPrice1"].notna() & df["bidPrice1"].notna()
    assert (df.loc[both, "askPrice1"] >= df.loc[both, "bidPrice1"]).all()

    # The project loader must ingest it as a clean single-symbol panel, and the
    # per-level columns must surface as their own panel fields.
    from quant_fund_agent.backtesting.data_loader import load_panel

    panel = load_panel(tmp_path, tickers=["GLD"])
    assert "GLD" in panel["close"].columns
    o, h, l, c = (panel[k]["GLD"] for k in ("open", "high", "low", "close"))
    assert (h >= np.maximum(o, c)).all()
    assert (l <= np.minimum(o, c)).all()
    assert "askPrice1" in panel and "GLD" in panel["askPrice1"].columns
    # a targeted level-field request also works
    sub = load_panel(tmp_path, tickers=["GLD"], fields=["close", "bidDepth2"])
    assert "bidDepth2" in sub and "GLD" in sub["bidDepth2"].columns

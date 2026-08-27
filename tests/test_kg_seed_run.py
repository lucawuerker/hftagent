"""End-to-end tests for ``scripts/run_kg_seed_run.py`` (the seeding-only KG
breadth-campaign entrypoint).

Reuses the fake-LLM / monkeypatched ``mechanism_group_specs`` patterns from
``test_evolution_mechanism_groups.py``: everything runs in-process
(``QF_USE_MCP=0``) with the graph, embedding store, retrieval and codegen LLM
round-trips faked, so ``main()`` exercises the real ``seed_programs`` seeding
path, the dedup, the persist and the link-back wiring end-to-end.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from quant_fund_agent.agents.factor_research.evolution import loop as loop_mod
from quant_fund_agent.agents.factor_research.evolution.genome import FactorProgram

REPO_ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "run_kg_seed_run", REPO_ROOT / "scripts" / "run_kg_seed_run.py")
kg_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kg_mod)


def _factor_code(fid: str, body: str, horizon: int = 6) -> str:
    return f'''\
"""Test factor {fid}."""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import stddev, ts_mean, ts_rank
from quant_fund_agent.factors.registry import register_factor


@register_factor
class F_{fid}(BaseFactor):
    factor_id = "{fid}"
    name = "{fid}"
    category = "momentum"
    inputs = ["close"]
    prediction_horizon = {horizon}

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        return {body}
'''


MOM_BODY = "close.pct_change().fillna(0.0)"
VOL_BODY = "stddev(close.pct_change(), 12).fillna(0.0)"
SMOOTH_BODY = "ts_mean(close.pct_change(), 3).fillna(0.0)"
RANK_BODY = "(ts_rank(close, 8) - 0.5).fillna(0.0)"

# One structurally distinct body per seeded idea; kg_mom_a is an EXACT canonical
# clone of the pre-existing cumulative-book member (window jitter only — the
# canonical AST turns numeric literals into typed placeholders).
BODIES = {
    "kg_mom_a": MOM_BODY,
    "kg_mom_b": VOL_BODY,
    "kg_liq_a": SMOOTH_BODY,
    "kg_liq_b": RANK_BODY,
}

IDEAS_BY_FOCUS = {
    "momentum spillover": ["kg_mom_a", "kg_mom_b"],
    "liquidity provision": ["kg_liq_a", "kg_liq_b"],
}


class _FakeLLM:
    """Stands in for every chat model; the patched retrieval/codegen seams mean
    it should never actually be invoked — fail loudly if it is."""

    def invoke(self, prompt):
        raise AssertionError("the fake LLM must not be invoked directly")


class _SpyGraph:
    """The 'live' knowledge graph object; only identity matters here."""


@pytest.fixture()
def campaign(tmp_path, monkeypatch):
    """A fully faked 2-group × 2-idea campaign environment; returns run kwargs."""
    monkeypatch.setenv("QF_USE_MCP", "0")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    # main() writes these envs — setenv first so monkeypatch restores them
    for key in ("FACTOR_RESEARCH_LLM_MODEL", "FACTOR_RESEARCH_LLM_PROVIDER",
                "QF_MAX_LLM_COST_USD", "DATA_DIR"):
        monkeypatch.setenv(key, "sentinel")

    # ── graph layer: 2 usable mechanism groups from the "live" graph ──
    import quant_fund_agent.knowledge.graph_query as gq

    spy_graph = _SpyGraph()
    monkeypatch.setattr(gq, "mechanism_group_specs",
                        lambda graph, n, fields: [
                            {"mechanism_group_id": 0, "community_id": 1,
                             "focus": "momentum spillover",
                             "mechanisms": ["mom_mech"]},
                            {"mechanism_group_id": 1, "community_id": 2,
                             "focus": "liquidity provision",
                             "mechanisms": ["liq_mech"]}])
    monkeypatch.setattr(
        "quant_fund_agent.knowledge.graph_store.KnowledgeGraph.load",
        staticmethod(lambda *a, **k: spy_graph))
    monkeypatch.setattr(
        "quant_fund_agent.knowledge.embed_store.EmbedStore", lambda: object())

    # ── retrieval: 2 mechanism-tagged ideas per group ──
    def fake_retrieve(llm, store, **kwargs):
        focus = kwargs.get("focus") or ""
        gaps = kwargs.get("gaps") or ["mech"]
        return [{"factor_id": fid, "name": fid, "category": "momentum",
                 "trading_idea": f"idea behind {fid}",
                 "description": f"{fid} description",
                 "prediction_horizon": 6, "expected_sign": 1,
                 "mechanism": gaps[0]}
                for fid in IDEAS_BY_FOCUS.get(focus, [])]

    monkeypatch.setattr(
        "quant_fund_agent.knowledge.retrieval.retrieve_and_brainstorm",
        fake_retrieve)

    # ── LLM seams: no real chat model, deterministic codegen ──
    fake = _FakeLLM()
    monkeypatch.setattr(loop_mod, "_get_llm", lambda temperature, role=None: fake)

    def fake_codegen(llm, idea, data_context, expected_prediction_horizon=None):
        return FactorProgram(
            factor_id=idea.factor_id,
            code=_factor_code(idea.factor_id, BODIES[idea.factor_id]),
            name=idea.name, category=idea.category,
            trading_idea=idea.trading_idea, description=idea.description,
            prediction_horizon=idea.prediction_horizon,
            expected_sign=idea.expected_sign)

    monkeypatch.setattr(loop_mod, "_codegen_program", fake_codegen)

    # ── link-back spy (also swallows seed_programs' internal readonly call) ──
    calls: list[dict] = []
    monkeypatch.setattr(
        loop_mod, "link_programs_into_graph",
        lambda graph, programs, mech_by_fid, *, readonly=False: calls.append(
            {"graph": graph, "programs": list(programs),
             "mech_by_fid": dict(mech_by_fid), "readonly": readonly}))

    # ── isolate the persist targets ──
    researcher_dir = tmp_path / "researcher"
    monkeypatch.setattr(
        "quant_fund_agent.agents.factor_research.codegen.RESEARCHER_DIR",
        researcher_dir)
    monkeypatch.setattr(kg_mod, "_data_context_and_fields",
                        lambda cfg: ("CTX", ["close"]))
    monkeypatch.setattr(
        "quant_fund_agent.mcp.research_client.existing_factor_ids",
        lambda scope="package": ["existing_pkg_factor"])

    campaign_dir = tmp_path / "kg_campaign"
    workspace_root = tmp_path / "preruns"
    campaign_dir.mkdir()
    # Pre-existing cumulative book: kg_prior is the exact structural clone
    # (id-masked, window-masked) of what kg_mom_a will generate.
    prior_fp = kg_mod._canonical_fp(_factor_code("kg_prior", MOM_BODY))
    assert prior_fp is not None
    (campaign_dir / "cumulative_book.json").write_text(json.dumps([
        {"factor_id": "kg_prior", "run": 0, "code_path": "kg_prior.py",
         "canonical_fp": prior_fp}]))

    argv = ["--run-index", "1", "--seed-ideas-per-group", "2",
            "--mechanism-groups", "2",
            "--campaign-dir", str(campaign_dir),
            "--workspace-root", str(workspace_root)]
    return dict(argv=argv, calls=calls, spy_graph=spy_graph,
                campaign_dir=campaign_dir, workspace_root=workspace_root,
                researcher_dir=researcher_dir)


def test_kg_seed_run_end_to_end(campaign):
    rc = kg_mod.main(campaign["argv"])
    assert rc == 0

    kept_ids = {"kg_mom_b", "kg_liq_a", "kg_liq_b"}

    # factor files written for the survivors, NOT for the structural clone
    for fid in kept_ids:
        assert (campaign["researcher_dir"] / f"{fid}.py").exists()
    assert not (campaign["researcher_dir"] / "kg_mom_a.py").exists()

    # prerun factor-DB rows carry the campaign provenance
    db = json.loads((campaign["workspace_root"] / "KG01_terra_s0" / "factors"
                     / "factor_db.json").read_text())
    rows = {row["id"]: row for row in db["factors"]}
    assert set(rows) == kept_ids
    for row in rows.values():
        assert row["metadata"]["kg_campaign_run"] == 1
        assert row["metadata"]["mechanism"] in ("mom_mech", "liq_mech")
        assert row["metadata"]["mechanism_group_id"] in (0, 1)
        assert row["code_path"].endswith(f"{row['id']}.py")
        assert row["required_inputs"] == ["close"]
        assert row["prediction_horizon"] == 6
    assert rows["kg_liq_a"]["metadata"]["mechanism_group_id"] == 1

    # cumulative book: prior member kept, survivors appended with fingerprints
    book = json.loads(
        (campaign["campaign_dir"] / "cumulative_book.json").read_text())
    assert [r["factor_id"] for r in book] == [
        "kg_prior", "kg_mom_b", "kg_liq_a", "kg_liq_b"]
    for row in book[1:]:
        assert row["run"] == 1
        assert row["canonical_fp"]

    # graph link-back: exactly one writable call, on the DEDUPED persisted set,
    # against the live graph object (seed_programs' internal per-group calls
    # ride through the spy as readonly=True and must not write)
    writable = [c for c in campaign["calls"] if c["readonly"] is False]
    assert len(writable) == 1
    assert writable[0]["graph"] is campaign["spy_graph"]
    assert {p.factor_id for p in writable[0]["programs"]} == kept_ids
    assert writable[0]["mech_by_fid"] == {
        "kg_mom_b": "mom_mech", "kg_liq_a": "liq_mech", "kg_liq_b": "liq_mech"}
    assert all(c["readonly"] is True for c in campaign["calls"]
               if c is not writable[0])

    # run summary: validated 4, one clone deduped, three persisted
    summary = json.loads(
        (campaign["campaign_dir"] / "run_01_summary.json").read_text())
    assert summary == {
        "run": 1, "n_ideas_requested": 4, "n_validated": 4, "n_deduped": 1,
        "n_persisted": 3, "llm_cost_usd": summary["llm_cost_usd"],
        "group_ids": [0, 1]}
    assert "timestamp" not in summary


def test_zero_persisted_exits_3(campaign, monkeypatch):
    monkeypatch.setattr(
        "quant_fund_agent.knowledge.retrieval.retrieve_and_brainstorm",
        lambda llm, store, **kwargs: [])
    rc = kg_mod.main(campaign["argv"])
    assert rc == 3
    summary = json.loads(
        (campaign["campaign_dir"] / "run_01_summary.json").read_text())
    assert summary["n_persisted"] == 0
    assert not [c for c in campaign["calls"] if c["readonly"] is False]


def test_canonical_fp_is_an_exact_clone_check_not_a_novelty_score():
    """Window jitter canonicalises to the SAME fingerprint (clone); a different
    operator pipeline gets a different one (kept)."""
    a = kg_mod._canonical_fp(_factor_code("f_a", "ts_mean(close.pct_change(), 20).fillna(0.0)"))
    b = kg_mod._canonical_fp(_factor_code("f_b", "ts_mean(close.pct_change(), 21).fillna(0.0)"))
    c = kg_mod._canonical_fp(_factor_code("f_c", "stddev(close.pct_change(), 20).fillna(0.0)"))
    assert a is not None and a == b
    assert c is not None and c != a

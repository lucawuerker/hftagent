"""Unit tests for the canonical AST weighted-subtree novelty metric.

Covers the properties the ``research_eval.ast_novelty`` module must guarantee:
identity, symmetry/bounds, and invariance to formatting, comments, docstrings,
metadata, local variable names, safe temporary inlining and numeric window
constants — while staying sensitive to changed data fields and operators.
"""

from __future__ import annotations

import ast

import pytest

from quant_fund_agent.research_eval.ast_novelty import (
    NOVELTY_METRIC,
    ast_subtree_distance,
    ast_subtree_similarity,
    canonical_factor_tree,
    subtree_profile,
)

# ── shared fixtures ───────────────────────────────────────────────────────────

MOM = (
    "def calc(self):\n"
    "    return (self.close - self.close.shift(5)) / self.close.shift(5)"
)
# same computation, only the window constant changed (5 → 10)
MOM_WINDOW = (
    "def calc(self):\n"
    "    return (self.close - self.close.shift(10)) / self.close.shift(10)"
)
# same computation, reformatted + commented + docstring
MOM_MESSY = (
    "def calc(self):\n"
    '    """A momentum factor."""\n'
    "    # normalised change\n"
    "    return (self.close-self.close.shift(5))  /  self.close.shift(5)\n"
)
# same computation via a renamed local temporary
MOM_LOCAL = (
    "def calc(self):\n"
    "    momentum = self.close - self.close.shift(5)\n"
    "    return momentum / self.close.shift(5)"
)
# different field + different operator structure
VOL = (
    "def calc(self):\n"
    "    v = self.volume.rolling(20).mean()\n"
    "    return (v - v.shift(1)) / (v.shift(1) + 1e-8)"
)


# ── A. identity ───────────────────────────────────────────────────────────────

def test_identity_similarity_one_distance_zero():
    assert ast_subtree_similarity(MOM, MOM) == 1.0
    assert ast_subtree_distance(MOM, MOM) == 0.0


# ── B. symmetry and bounds ────────────────────────────────────────────────────

@pytest.mark.parametrize("a,b", [(MOM, VOL), (MOM, MOM_WINDOW), (VOL, MOM_LOCAL)])
def test_symmetry_and_bounds(a, b):
    dab = ast_subtree_distance(a, b)
    dba = ast_subtree_distance(b, a)
    assert dab == pytest.approx(dba, abs=1e-12)      # symmetric
    for x in (dab, dba, ast_subtree_similarity(a, b)):
        assert 0.0 <= x <= 1.0                        # bounded


# ── C. formatting invariance ──────────────────────────────────────────────────

def test_whitespace_and_linebreaks_do_not_change_distance():
    dense = "def calc(self):\n    return self.close-self.close.shift(5)"
    spaced = "def calc(self):\n    return self.close  -  self.close.shift( 5 )\n\n"
    assert ast_subtree_distance(dense, spaced) == 0.0


# ── D. comment / docstring invariance ─────────────────────────────────────────

def test_comments_and_docstrings_are_ignored():
    assert ast_subtree_distance(MOM, MOM_MESSY) == 0.0


# ── E. metadata invariance ────────────────────────────────────────────────────

def test_class_name_id_and_metadata_do_not_change_calc_structure():
    a = (
        "class Alpha(BaseFactor):\n"
        "    factor_id = 'alpha_one'\n"
        "    name = 'Alpha One'\n"
        "    category = 'momentum'\n"
        "    inputs = ['close']\n"
        "    prediction_horizon = 6\n"
        "    def calc(self, data):\n"
        "        return data['close'].pct_change()\n"
    )
    b = (
        "import numpy as np\n"
        "class Beta(BaseFactor):\n"
        "    factor_id = 'totally_different_id'\n"
        "    name = 'Beta Two'\n"
        "    category = 'reversal'\n"
        "    inputs = ['close', 'volume']\n"
        "    prediction_horizon = 24\n"
        "    def calc(self, panel):\n"
        "        return panel['close'].pct_change()\n"
    )
    assert ast_subtree_distance(a, b) == 0.0


# ── F. local-variable invariance ──────────────────────────────────────────────

def test_local_variable_renaming_is_invariant():
    a = (
        "def calc(self):\n"
        "    momentum = self.close - self.close.shift(5)\n"
        "    volatility = self.close.shift(5)\n"
        "    return momentum / volatility"
    )
    b = (
        "def calc(self):\n"
        "    x = self.close - self.close.shift(5)\n"
        "    y = self.close.shift(5)\n"
        "    return x / y"
    )
    assert ast_subtree_distance(a, b) == 0.0


# ── G. safe temporary inlining ────────────────────────────────────────────────

def test_inline_return_equals_straightline_temporaries():
    inline = "def calc(self):\n    return ts_mean(close, 20) - ts_mean(close, 100)"
    temps = (
        "def calc(self):\n"
        "    fast = ts_mean(close, 20)\n"
        "    slow = ts_mean(close, 100)\n"
        "    return fast - slow"
    )
    assert ast_subtree_distance(inline, temps) == 0.0


# ── H. numeric-constant (parameter) invariance ────────────────────────────────

def test_window_constant_change_is_a_structural_clone():
    assert ast_subtree_distance(MOM, MOM_WINDOW) == 0.0
    a = "def calc(self):\n    return ts_mean(close, 20)"
    b = "def calc(self):\n    return ts_mean(close, 21)"
    assert ast_subtree_distance(a, b) == 0.0


def test_booleans_none_and_strings_are_not_placeholdered():
    # a bool literal is preserved: flipping it is a structural change, unlike ints
    a = "def calc(self):\n    return self.close.rolling(5).mean(skipna=True)"
    b = "def calc(self):\n    return self.close.rolling(5).mean(skipna=False)"
    assert ast_subtree_distance(a, b) > 0.0


# ── I. field sensitivity ──────────────────────────────────────────────────────

def test_field_change_is_non_zero_distance():
    close_f = "def calc(self):\n    return self.close.shift(1)"
    volume_f = "def calc(self):\n    return self.volume.shift(1)"
    assert ast_subtree_distance(close_f, volume_f) > 0.0
    # string-key fields too
    a = "def calc(self, data):\n    return data['close'].pct_change()"
    b = "def calc(self, data):\n    return data['volume'].pct_change()"
    assert ast_subtree_distance(a, b) > 0.0


# ── J. operator sensitivity ───────────────────────────────────────────────────

def test_operator_change_is_non_zero_distance():
    add = "def calc(self):\n    return self.close + self.close.shift(1)"
    sub = "def calc(self):\n    return self.close - self.close.shift(1)"
    assert ast_subtree_distance(add, sub) > 0.0
    mean = "def calc(self):\n    return self.close.rolling(20).mean()"
    std = "def calc(self):\n    return self.close.rolling(20).std()"
    assert ast_subtree_distance(mean, std) > 0.0


def test_commutative_add_and_mult_are_canonicalised():
    assert ast_subtree_distance(
        "def calc(self):\n    return self.close + self.volume",
        "def calc(self):\n    return self.volume + self.close") == 0.0
    assert ast_subtree_distance(
        "def calc(self):\n    return self.close * self.volume",
        "def calc(self):\n    return self.volume * self.close") == 0.0
    # subtraction is NOT commutative — order matters
    assert ast_subtree_distance(
        "def calc(self):\n    return self.close - self.volume",
        "def calc(self):\n    return self.volume - self.close") > 0.0


# ── K. structural ranking ─────────────────────────────────────────────────────

def test_window_clone_is_closer_to_reference_than_a_different_program():
    ref = "def calc(self):\n    return self.close - self.close.shift(20)"
    window_clone = "def calc(self):\n    return self.close - self.close.shift(60)"
    other = "def calc(self):\n    return self.volume.rolling(30).std()"
    d_clone = ast_subtree_distance(window_clone, ref)
    d_other = ast_subtree_distance(other, ref)
    assert d_clone < d_other
    assert d_clone == 0.0                     # a pure window change is a clone
    # sharing a larger computational subtree yields more similarity than sharing
    # only trivial leaves
    assert ast_subtree_similarity(window_clone, ref) > ast_subtree_similarity(other, ref)


# ── N. robustness on empty / invalid source ───────────────────────────────────

@pytest.mark.parametrize("bad", ["", "   ", "\n", "this is not python ("])
def test_empty_or_invalid_source_returns_none(bad):
    assert subtree_profile(bad) is None
    assert canonical_factor_tree(bad) is None
    assert ast_subtree_similarity(bad, MOM) is None
    assert ast_subtree_distance(bad, MOM) is None
    assert ast_subtree_distance(MOM, bad) is None


# ── profile / API shape ───────────────────────────────────────────────────────

def test_profile_is_immutable_and_multiset_and_metric_named():
    prof = subtree_profile(MOM)
    assert prof is not None
    assert prof.n_nodes >= prof.n_unique >= 1
    # a Sub over two identical operands makes at least one subtree appear twice
    dup = "def calc(self):\n    return ts_mean(close, 20) - ts_mean(close, 20)"
    dprof = subtree_profile(dup)
    assert any(mult >= 2 for _, mult, _ in dprof.items)   # genuine multiset
    # frozen dataclass — immutable
    with pytest.raises(Exception):
        prof.n_nodes = 0  # type: ignore[misc]
    assert NOVELTY_METRIC == "canonical_ast_weighted_subtree_jaccard"


def test_canonical_tree_is_an_ast_and_standalone_calc_and_fallback_agree():
    assert isinstance(canonical_factor_tree(MOM), ast.AST)
    # a standalone `calc` function (no class) and a bare top-level statement both
    # extract to the same canonical computation as the method form
    standalone = "def calc():\n    return close.pct_change()"
    toplevel = "signal = close.pct_change()"
    method = "class F(BaseFactor):\n    def calc(self, data):\n        return close.pct_change()"
    assert ast_subtree_distance(standalone, toplevel) == 0.0
    assert ast_subtree_distance(standalone, method) == 0.0

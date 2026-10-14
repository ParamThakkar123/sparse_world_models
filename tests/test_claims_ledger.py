"""Structural checks on the paper's claim ledger.

The ledger's *values* are checked against run artifacts by ``python paper/claims.py``, which
needs ``experiments/runs/`` and therefore cannot run on a fresh clone. What is checked here is
the ledger itself: that it is well formed, that it has not silently lost the claims the paper's
argument rests on, and that no entry has been left pointing at nothing.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("paper_claims", ROOT / "paper" / "claims.py")
assert SPEC and SPEC.loader
claims = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(claims)

VALID_KINDS = {"audit", "literature", "lit_range", "lit_diff", "planning", "pixel", "json", "json_diff", "onset_rate", "derived",
               "constant", "ledger"}


def test_every_entry_is_well_formed():
    for value, where, source in claims.CLAIMS:
        assert isinstance(value, str) and float(value) == float(value), where
        assert where and isinstance(where, str)
        assert source[0] in VALID_KINDS, f"{where}: unknown source kind {source[0]!r}"
        if source[0] == "audit":
            assert len(source) in (5, 6, 7), f"{where}: audit source needs run/benchmark/rule/metric"
        elif source[0] == "literature":
            assert len(source) == 5, f"{where}: literature source needs benchmark/model/metric/count"
        elif source[0] == "ledger":
            assert len(source) == 3 and source[1], f"{where}: ledger source needs a RESULTS.md section"
        elif source[0] in ("derived", "constant"):
            assert len(source) == 2 and source[1], f"{where}: {source[0]} needs an expression/reason"


def test_no_duplicate_claim_descriptions():
    seen = [where for _, where, _ in claims.CLAIMS]
    assert len(seen) == len(set(seen)), "two ledger entries share a description"


def test_the_load_bearing_claims_are_present():
    """The four best-trivial numbers are the paper's spine; losing one silently is the risk."""
    values = {v for v, _, _ in claims.CLAIMS}
    for headline in ("0.927", "0.961", "0.955", "0.821"):
        assert headline in values, f"best-trivial value {headline} dropped out of the ledger"
    # And the pairing that was wrong once already: the velocity rule is 0.866, NOT 0.927.
    velocity = [v for v, w, s in claims.CLAIMS
                if s[0] == "audit" and s[3] == "already_moving" and s[2] == "motion"
                and s[4] == "f1"]
    assert velocity == ["0.866"], f"velocity-on-motion should be 0.866, ledger says {velocity}"


def test_every_ledger_only_claim_names_a_results_section():
    for value, where, source in claims.CLAIMS:
        if source[0] == "ledger":
            assert len(source[1]) > 5, f"{where}: RESULTS.md pointer is too vague to follow"


@pytest.mark.parametrize("kind", sorted(VALID_KINDS - {"ledger"}))
def test_each_source_kind_is_actually_used(kind):
    """A kind that no entry uses is dead code in the checker; a kind that every entry uses
    means the ledger has stopped distinguishing verified from unverifiable. ``ledger``
    (RESULTS.md-only) is exempt: the goal is for no entry to need it."""
    used = [w for _, w, s in claims.CLAIMS if s[0] == kind]
    assert used, f"no ledger entry uses source kind {kind!r}"
    assert len(used) < len(claims.CLAIMS), f"every entry is kind {kind!r}"


def test_derived_entries_evaluate_to_their_printed_value():
    for printed, where, source in claims.CLAIMS:
        if source[0] == "derived":
            actual = claims.recompute(source)
            assert abs(actual - float(printed)) <= claims.tolerance(printed, True), where


def test_presence_check_matches_whole_tokens_only():
    """A substring match let 0.41 stand in for 0.413 once; the check must not."""
    body = "the floor is $0.413$ and the ceiling $0.982$"
    assert claims.token_present("0.413", body)
    assert not claims.token_present("0.41", body)
    assert not claims.token_present("0.98", body)
    assert claims.token_present("-0.413", body)


def test_coverage_scan_ignores_comments_and_figure_options(tmp_path):
    tex = tmp_path / "m.tex"
    tex.write_text("% 0.123 in a comment\n"
                   r"\includegraphics[width=0.92\textwidth]{f}" "\n"
                   "real 0.456\n", encoding="utf-8")
    assert claims.NUMBER.findall(claims.manuscript_body(tex)) == ["0.456"]

"""Tests for the manuscript builder machinery (manuscript/build_manuscript.py)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "manuscript"))
import build_manuscript as B  # noqa: E402


def test_lint_flags_typed_numbers_and_allows_exceptions():
    assert B.lint("The cohort has 48 lesions.")                         # typed number
    ok = ("Figures 2 and 3, Table 1, Section 2.1, [3,5–7], 2D and 3D, D8, AT1, P2 elements, "
          "C-A180-T0.2-SUP, inline math $E = 2 \\mu$.")
    assert B.lint(ok) == []
    assert B.lint("# 2. Methods") == []                                  # headings are skipped


def test_ledger_numbers_pass_lint_and_are_recorded():
    ctx = B.Ctx()
    t = ctx.v("results/cohort_summary.json", "n_per_set.core", fmt="d")
    assert B.lint(f"The cohort has {t} lesions.") == []
    assert B.strip_marks(t) == str(ctx.get("results/cohort_summary.json", "n_per_set.core"))
    row = ctx.ledger[-1]
    assert row["source"] == "results/cohort_summary.json" and row["key"] == "n_per_set.core"


def test_missing_value_raises():
    ctx = B.Ctx()
    with pytest.raises(B.BuildError):
        ctx.num(None, src="x", key="y")
    with pytest.raises(B.BuildError):
        ctx.num(True, src="x", key="y")
    with pytest.raises(KeyError):
        ctx.get("results/cohort_summary.json", "no.such.key")


def test_citation_numbering_by_first_appearance():
    ctx = B.Ctx()
    keys = list(ctx.bib)[:5]
    assert ctx.cite(keys[2]) == "[1]"
    assert ctx.cite(keys[0], keys[2]) == "[1,2]"
    assert ctx.cite(keys[1], keys[3], keys[4]) == "[3–5]"
    with pytest.raises(B.BuildError):
        ctx.cite("not-a-key")


def test_reference_format():
    e = {"author": "Hill, J. M. and others", "title": "A {Title}", "journal": "J. X", "year": "2020",
         "volume": "76", "pages": "2635--2646", "doi": "10.1/abc"}
    s = B.format_reference(e)
    assert s.startswith("Hill, J. M.; et al. A Title.") and "**2020**" in s and "https://doi.org/10.1/abc" in s


def test_validation_wording_rule():
    import re
    pat = re.compile(r"(?<!cross-)\bvalidat\w*", re.I)
    assert pat.search("The model was validated against patients.")
    assert pat.search("Validation of the model")
    assert not pat.search("leave-one-out cross-validation (LOOCV)")


def test_bibtex_markup_is_rendered():
    assert B._clean(r"Tann{\'e}, E. and G{\"u}ltekin, O.") == "Tanné, E. and Gültekin, O."
    assert B._clean(r"Circular Angle 180\textdegree") == "Circular Angle 180°"
    assert B._clean(r"A `Digital Twin' for") == "A ‘Digital Twin’ for"
    ref = B.format_reference({"author": "Gharaibeh, Y. and others and Wilson, D. L.", "title": "T", "journal": "J",
                              "year": "2023"})
    assert ref.startswith("Gharaibeh, Y.; et al. T.") and "Wilson" not in ref

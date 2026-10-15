"""Tests for the results figures 4 (damage fields) and 5 (core outcomes): they build from the committed
partial data and from frames with missing lesions and protocols, and write PNGs at figstyle.DPI."""
import json
import os
import shutil

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from dtivl import params as P
from dtivl.drivers import figstyle
from dtivl.drivers.figs import fig4_damage_fields as F4
from dtivl.drivers.figs import fig5_outcomes as F5
from dtivl.drivers.stage_fields import FIELDS_DIR, field_lesions


@pytest.fixture
def fig_dir(tmp_path, monkeypatch):
    """Figures go to a temporary directory, never to figures/."""
    monkeypatch.setattr(figstyle, "FIG_DIR", tmp_path)
    return tmp_path


def _check_png(path: str) -> None:
    assert path.endswith(".png") and os.path.exists(path)
    assert os.path.exists(path[:-4] + ".pdf")
    dpi = Image.open(path).info["dpi"]
    assert round(dpi[0]) == figstyle.DPI and round(dpi[1]) == figstyle.DPI


# ----------------------------------------------------------------------------- figure 4
def _committed_fields():
    lids = [lid for lid in field_lesions() if (FIELDS_DIR / f"{lid}_ivl.npz").exists()]
    if not lids:
        pytest.skip("no field files on disk")
    return lids


def test_fig4_builds_on_committed_fields(fig_dir):
    _committed_fields()
    _check_png(F4.make())


def test_fig4_builds_with_missing_lesions_and_protocols(fig_dir, tmp_path, monkeypatch):
    lids = _committed_fields()
    part = tmp_path / "fields_partial"
    part.mkdir()
    # first lesion: IVL phase field only; one extra core lesion without any field file; no summary entry for it
    shutil.copy(FIELDS_DIR / f"{lids[0]}_ivl.npz", part)
    summary = json.loads((FIELDS_DIR / "fields_summary.json").read_text())
    (part / "fields_summary.json").write_text(json.dumps({lids[0]: {"ivl": summary[lids[0]]["ivl"]}}))
    monkeypatch.setattr(F4, "FIELDS_DIR", part)
    monkeypatch.setattr(F4, "field_lesions", lambda: [lids[0], "C-A360-T0.8-NOD"])
    assert F4.load_panel(lids[0], "ivl_erosion") is None and F4.load_panel(lids[0], "ivl") is not None
    _check_png(F4.make("fig4_partial"))


def test_fig4_without_any_field_file(fig_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(F4, "FIELDS_DIR", tmp_path / "empty")
    _check_png(F4.make("fig4_empty"))


def test_fig4_rotation_puts_calcium_centroid_at_orientation():
    lid = _committed_fields()[0]
    p = F4.prepare(F4.load_panel(lid, "ivl"))
    tri = p["triangles"][p["ca"]]
    cen = p["ref"][tri].mean(axis=1).mean(axis=0)
    ang = np.degrees(np.arctan2(cen[1], cen[0]))
    assert abs(ang - P.get("figures.fig4_damage_fields.orientation_deg").value) < 2.0   # area vs plain mean
    # the lumen outline is the inner boundary loop: its nodes are the boundary nodes nearest the centre
    r_lumen = np.hypot(*p["points"][np.unique(p["lumen_edges"])].T)
    r_all = np.hypot(*p["points"][np.unique(F4._boundary_edges(p["triangles"]))].T)
    assert r_lumen.max() < r_all.max()


def test_fig4_scale_bar_and_annotation_from_file():
    frac = P.get("figures.fig4_damage_fields.scale_bar_frac").value
    for w in (0.37, 1.0, 2.9, 4.4, 13.0):
        L = F4.nice_length(w)
        assert L <= frac * w + 1e-12 and L > 0.4 * frac * w           # 1-2-5 steps are at most 2.5x apart
        assert round(L / 10.0 ** np.floor(np.log10(L)), 9) in (1.0, 2.0, 5.0)
    summary = json.loads((FIELDS_DIR / "fields_summary.json").read_text())
    for lid, rec in summary.items():
        for proto, r in rec.items():
            txt = F4.outcome_text(r)
            assert ("through-thickness: yes" in txt) == (r["fracture"] is True)
            if r["onset_load"] is not None:
                assert f"{r['onset_load']:#.3g} MPa" in txt
            else:
                assert "no damage onset" in txt
    assert F4.outcome_text(None) == "outcome not recorded"


# ----------------------------------------------------------------------------- figure 5
def _outcomes():
    if not F5.OUTCOMES_CSV.exists():
        pytest.skip("results/cohort_outcomes.csv not built yet")
    return F5.read_outcomes()


def test_fig5_builds_on_committed_outcomes(fig_dir):
    _outcomes()
    _check_png(F5.make())


def test_fig5_builds_with_missing_protocols(fig_dir, monkeypatch):
    df = _outcomes()
    drop = [c for c in df.columns if c.startswith(("balloon_10atm_", "stent_no_ivl_")) or c == "ivl_p_top"]
    syn = df.drop(columns=drop).iloc[::2].copy()                       # half the lesions, two protocols gone
    syn.loc[syn.index[:2], "ivl_status"] = "failed"
    monkeypatch.setattr(F5, "read_outcomes", lambda: syn)
    _check_png(F5.make("fig5_partial"))
    # no IVL and no stent columns at all: the panels say 'not available' but the figure builds
    bare = df[[c for c in df.columns if not c.startswith(("ivl", "stent_"))]].copy()
    monkeypatch.setattr(F5, "read_outcomes", lambda: bare)
    _check_png(F5.make("fig5_bare"))


def test_damage_class_rule():
    df = pd.DataFrame({
        "x_status": ["ok", "ok", "ok", "ok", "ok", "failed", None, "ok"],
        "x_fracture": [True, False, False, False, False, True, None, None],
        "x_onset": [0.5, 0.6, np.nan, np.nan, np.nan, 0.5, np.nan, np.nan],
        "x_n_partial": [0, 0, 2, 0, 0, 0, np.nan, 0],
        "x_max_depth_frac": [1.0, 0.3, 0.4, 0.0, 0.0, 1.0, np.nan, 0.0],
        "x_damaged_fraction": [1.0, 0.1, 0.1, 0.05, 0.0, 1.0, np.nan, 0.0],
    })
    assert F5.damage_class(df, "x").tolist() == ["through", "partial", "partial", "partial", "none", "na", "na",
                                                 "na"]
    # an absent evidence column is skipped; an absent protocol is 'not run'
    assert F5.damage_class(df.drop(columns=["x_onset"]), "x").tolist()[:5] == ["through", "partial", "partial",
                                                                               "partial", "none"]
    assert set(F5.damage_class(df, "y")) == {"na"}


def test_onset_table_censoring_and_peak_load_fallback():
    df = _outcomes()
    t = df[df["ivl_status"] == "ok"].copy()
    if t.empty:
        pytest.skip("no IVL run yet")
    ref = F5.onset_table(t)
    # the peak load comes from the json record where the table lacks ivl_p_top
    no_top = F5.onset_table(t.drop(columns=["ivl_p_top"]))
    np.testing.assert_allclose(no_top["value"].to_numpy(float), ref["value"].to_numpy(float), rtol=0, atol=0)
    cens = ref[~ref["event"]]
    np.testing.assert_array_equal(cens["value"].to_numpy(float), pd.to_numeric(cens["ivl_p_top"]).to_numpy(float))
    ev = ref[ref["event"] & ~ref["after_peak"]]
    np.testing.assert_array_equal(ev["value"].to_numpy(float), pd.to_numeric(ev["ivl_onset"]).to_numpy(float))
    # an onset recorded after the loading phases is drawn at the peak load (metrics-stage rule)
    late = t.head(1).copy()
    late["ivl_onset"], late["ivl_onset_phase"] = 0.123, "stent_expand"
    r = F5.onset_table(late)
    assert bool(r["after_peak"].iloc[0]) and r["value"].iloc[0] == pd.to_numeric(late["ivl_p_top"]).iloc[0]


def test_expansion_reference_and_labels_come_from_params():
    ref = F5.expansion_reference()
    d_s = P.get("geometry.stent_target_diameter_mm").value
    d_r = P.get("geometry.reference_diameter_mm").value
    assert ref == pytest.approx(100.0 * (d_s / d_r) ** 2)
    row = P.load_literature()[P.raw("loads.ivl_balloon_pressure_atm")["lit"]]
    assert f"{float(row.value_str):g} {row.unit}" in F5.protocol_label("balloon_only")

"""Tests for dtivl.geometry: build, rasterize, descriptors, fujino, cohort."""
from __future__ import annotations

import copy
import math

import numpy as np
import pandas as pd
import pytest
import shapely
from scipy.integrate import quad

from dtivl import params
from dtivl.geometry import build, cohort
from dtivl.geometry.build import (calcium_inner_radius, calcium_radii, cap_half_angle, lesion_polygons,
                                  nominal_descriptors)
from dtivl.geometry.descriptors import _max_circular_run, count_components_wrapped, descriptors
from dtivl.geometry.fujino import fujino_criteria, fujino_max_score, fujino_score
from dtivl.geometry.labels import LABEL_CALCIUM, LABEL_LUMEN, LABEL_OTHER
from dtivl.geometry.polar import PolarLabelMap
from dtivl.geometry.rasterize import ray_circle_radius, rasterize_polar
from dtivl.geometry.spec import MORPHOLOGIES

AREA_RTOL = 1e-4
ARCS = [90, 180, 270, 360]
THICKNESSES = [0.2, 0.4]


@pytest.fixture(scope="module")
def geo():
    """Core reference geometry (PENDING placeholders allowed in tests only)."""
    return cohort.fixed_geometry(allow_pending=True)


def _spec(geo, morph, arc, t, **kw):
    return cohort.make_spec(f"T-{morph}-{arc}-{t}", morph, arc, t, geo, set_name="test", **kw)


def _deposit_area(spec, width_deg):
    r1, r2 = calcium_radii(spec)
    rho, r_m = 0.5 * (r2 - r1), 0.5 * (r1 + r2)
    return 2 * rho * r_m * (math.radians(width_deg) - 2 * cap_half_angle(spec)) + math.pi * rho ** 2


# ------------------------------------------------------------------ build
@pytest.mark.parametrize("morph", ["superficial_sheet", "deep_sheet"])
@pytest.mark.parametrize("arc", ARCS)
@pytest.mark.parametrize("t", THICKNESSES)
def test_sheet_areas_match_analytic(geo, morph, arc, t):
    s = _spec(geo, morph, arc, t)
    g = lesion_polygons(s)
    r1, r2 = calcium_radii(s)
    exact = math.pi * (r2 ** 2 - r1 ** 2) if arc == 360 else _deposit_area(s, arc)
    assert g["calcium"].area == pytest.approx(exact, rel=AREA_RTOL)
    assert len(g["calcium"].geoms) == 1
    assert g["lumen"].area == pytest.approx(math.pi * s.r_lumen ** 2, rel=AREA_RTOL)
    assert g["lumen"].area == pytest.approx(geo["lumen_area"], rel=AREA_RTOL)
    assert g["iel"].area == pytest.approx(math.pi * build.iel_radius(s) ** 2, rel=AREA_RTOL)
    assert g["outer"].area == pytest.approx(math.pi * build.outer_radius(s) ** 2, rel=AREA_RTOL)
    assert g["lipid"].is_empty
    assert len(g["lumen"].exterior.coords) - 1 >= 1440


def test_rounded_end_extent_is_exact(geo):
    """Caps are tangent to the rays bounding the arc."""
    s = _spec(geo, "superficial_sheet", 90, 0.4)
    ca = lesion_polygons(s)["calcium"]
    xy = np.vstack([np.asarray(p.exterior.coords) for p in ca.geoms])
    ang = np.degrees(np.arctan2(xy[:, 1], xy[:, 0]))
    assert ang.max() == pytest.approx(45.0, abs=1e-3)
    assert ang.min() == pytest.approx(-45.0, abs=1e-3)


@pytest.mark.parametrize("arc", ARCS)
def test_fragmented_areas(geo, arc):
    s = _spec(geo, "fragmented", arc, 0.2)
    g = lesion_polygons(s)
    width = s.fragment_fill * arc / s.n_fragments
    assert len(g["calcium"].geoms) == s.n_fragments
    assert g["calcium"].area == pytest.approx(s.n_fragments * _deposit_area(s, width), rel=AREA_RTOL)


@pytest.mark.parametrize("arc", [180, 360])
def test_nodule_area_and_cover(geo, arc):
    s = _spec(geo, "nodule", arc, 0.2)
    g = lesion_polygons(s)
    r1, r2 = calcium_radii(s)
    half = math.radians(s.nodule_base_arc_deg) / 2
    mound = quad(lambda p: 0.5 * (r1 ** 2 - float(calcium_inner_radius(s, np.array(p))) ** 2),
                 -half, half, points=[0.0])[0]
    sheet = math.pi * (r2 ** 2 - r1 ** 2) if arc == 360 else _deposit_area(s, arc)
    assert g["calcium"].area == pytest.approx(sheet + mound, rel=AREA_RTOL)
    # calcium protrudes by nodule_height at the arc centre
    assert float(calcium_inner_radius(s, np.array(0.0))) == pytest.approx(r1 - s.nodule_height)
    # covering layer of uniform normal thickness ca_depth
    assert g["lumen"].intersection(g["calcium"]).area == 0.0
    assert g["lumen"].distance(g["calcium"]) == pytest.approx(s.ca_depth, abs=1e-4)
    pts = np.asarray(g["lumen"].exterior.coords)
    over = np.hypot(pts[:, 0], pts[:, 1]) < s.r_lumen - 1e-6
    assert over.sum() > 100
    d = shapely.distance(shapely.points(pts[over]), g["calcium"])
    assert np.allclose(d, s.ca_depth, atol=1e-4)
    assert g["lumen"].area < math.pi * s.r_lumen ** 2


def test_lipid_and_offset_plaque(geo):
    s = _spec(geo, "superficial_sheet", 90, 0.2, plaque_offset=0.2, lipid_arc_deg=90,
              lipid_thickness=0.2, lipid_cap_thickness=0.1, lipid_center_deg=180)
    g = lesion_polygons(s)
    ra = s.r_lumen + s.lipid_cap_thickness
    exact = 0.5 * math.radians(90) * ((ra + 0.2) ** 2 - ra ** 2)
    assert g["lipid"].area == pytest.approx(exact, rel=AREA_RTOL)
    plm = rasterize_polar(s)
    d = descriptors(plm)
    assert abs(d["lipid_arc_deg"] - 90) <= 2 * 360 / plm.n_theta
    assert abs(d["cap_thickness_min"] - 0.1) <= plm.dr + 1e-9
    # intima thickness r_iel - r_lumen varies between P - offset and P + offset
    P, o = s.plaque_thickness, s.plaque_offset
    assert d["plaque_ecc_index"] == pytest.approx(2 * o / (P + o), abs=2 * plm.dr / (P + o))


def test_invalid_geometry_raises(geo):
    with pytest.raises(ValueError):
        _spec(geo, "deep_sheet", 90, 0.4, ca_depth=geo["plaque_thickness"])
    s = _spec(geo, "superficial_sheet", 90, 0.2, lipid_arc_deg=120, lipid_thickness=0.1,
              lipid_center_deg=180)
    s.lipid_center_deg = 80.0   # overlaps the calcium arc
    with pytest.raises(ValueError):
        lesion_polygons(s)


def test_zero_depth_shares_boundary(geo):
    for morph in ("superficial_sheet", "nodule"):
        s = _spec(geo, morph, 180, 0.2, ca_depth=0.0)
        g = lesion_polygons(s)
        assert g["lumen"].intersection(g["calcium"]).area < 1e-8
        assert descriptors(rasterize_polar(s))["ca_depth_min"] == pytest.approx(0.0)


# -------------------------------------------------------------- rasterize
def test_ray_circle_radius_exact():
    th = np.linspace(0, 2 * np.pi, 37)
    c, R = (0.3, -0.2), 1.7
    s = ray_circle_radius(th, c, R)
    x, y = s * np.cos(th), s * np.sin(th)
    assert np.allclose(np.hypot(x - c[0], y - c[1]), R, atol=1e-12)
    assert np.all(s > 0)
    with pytest.raises(ValueError):
        ray_circle_radius(th, (2.0, 0.0), 1.0)


def test_rasterize_defaults_and_validity(geo):
    s = _spec(geo, "superficial_sheet", 180, 0.2, plaque_offset=0.15)
    plm = rasterize_polar(s)
    n_theta, dr = int(params.raw("discretisation.polar_n_theta")), float(params.raw("discretisation.polar_dr_mm"))
    assert plm.n_theta == n_theta and plm.dr == dr
    plm.validate()
    assert plm.r[-1] > plm.r_out.max()
    assert set(np.unique(plm.labels)) <= {LABEL_OTHER, LABEL_LUMEN, LABEL_CALCIUM}
    # IEL exact intersection: thickest plaque toward the calcium (theta = 0)
    assert plm.r_iel[0] == pytest.approx(s.r_lumen + s.plaque_thickness + s.plaque_offset)
    assert plm.r_out[n_theta // 2] == pytest.approx(
        s.r_lumen + s.plaque_thickness + s.wall_thickness - s.plaque_offset)


@pytest.mark.parametrize("morph", MORPHOLOGIES)
@pytest.mark.parametrize("arc", ARCS)
@pytest.mark.parametrize("t", THICKNESSES)
def test_descriptors_match_spec(geo, morph, arc, t):
    """Rasterised descriptors within one sample of the spec (per deposit end for the arc)."""
    s = _spec(geo, morph, arc, t)
    plm = rasterize_polar(s)
    d = descriptors(plm)
    nom = nominal_descriptors(s)
    dth = 360.0 / plm.n_theta
    eps = 1e-9
    assert d["n_deposits"] == nom["n_deposits"]
    assert abs(d["ca_arc_max_contig_deg"] - nom["ca_arc_max_contig_deg"]) <= dth + eps
    assert abs(d["ca_arc_total_deg"] - nom["ca_arc_total_deg"]) <= nom["n_deposits"] * dth + eps
    assert abs(d["ca_thickness_max"] - nom["ca_thickness_max"]) <= plm.dr + eps
    assert abs(d["ca_depth_min"] - nom["ca_depth_min"]) <= plm.dr + eps
    if morph != "nodule":
        assert abs(d["lumen_area"] - math.pi * s.r_lumen ** 2) <= 2 * math.pi * s.r_lumen * plm.dr
    ca_exact = lesion_polygons(s)["calcium"].area
    assert d["ca_area"] == pytest.approx(ca_exact, rel=0.05)
    # concentric plaque: eccentricity only from the nodule's lumen protrusion
    h = s.nodule_height if morph == "nodule" else 0.0
    assert d["plaque_ecc_index"] == pytest.approx(h / (s.plaque_thickness + h),
                                                  abs=2 * plm.dr / s.plaque_thickness)
    assert math.isnan(d["cap_thickness_min"]) and d["lipid_arc_deg"] == 0
    assert d["fujino_max"] == fujino_max_score(False)


# ------------------------------------------------------------ descriptors
def test_circular_run_and_wrapped_components():
    assert _max_circular_run(np.array([1, 1, 0, 0, 1, 1, 1], bool)) == 5
    assert _max_circular_run(np.zeros(5, bool)) == 0
    assert _max_circular_run(np.ones(5, bool)) == 5
    m = np.zeros((12, 6), bool)
    m[0:2, 2:4] = True     # crosses the seam with ...
    m[10:12, 3:5] = True   # ... this block
    m[5:7, 1:3] = True     # separate deposit
    assert count_components_wrapped(m) == 2
    m[11, :] = False
    m[11, 1] = True        # diagonal neighbour of (0, 2) across the seam
    m[10, :] = False
    assert count_components_wrapped(m) == 2


def test_descriptors_hand_made_mask():
    """Real-segmentation path: labels {0, 1, 4}, no IEL, wrap-around deposit."""
    n_theta, n_r, dr = 36, 50, 0.02
    lab = np.zeros((n_theta, n_r), np.uint8)
    lab[:, :20] = LABEL_LUMEN
    rows = [34, 35, 0, 1, 2]                 # 5 A-lines across the seam
    lab[np.ix_(rows, range(25, 35))] = LABEL_CALCIUM
    plm = PolarLabelMap(lab, dr, r_out=np.full(n_theta, 0.95), r_iel=None)
    d = descriptors(plm, length_mm=6.0)
    assert d["ca_arc_total_deg"] == pytest.approx(50.0)
    assert d["ca_arc_max_contig_deg"] == pytest.approx(50.0)
    assert d["n_deposits"] == 1
    assert d["ca_thickness_max"] == pytest.approx(10 * dr)
    assert d["ca_depth_min"] == pytest.approx(5 * dr)
    r = plm.r
    assert d["ca_area"] == pytest.approx(5 * (2 * np.pi / n_theta) * dr * r[25:35].sum())
    assert d["lumen_area"] == pytest.approx(np.pi * (20 * dr) ** 2)
    assert math.isnan(d["plaque_ecc_index"])
    assert d["fujino_score"] == fujino_criteria()["length"].points
    assert d["fujino_max"] == fujino_max_score(True)


# ------------------------------------------------------------------ fujino
def test_fujino_edge_cases():
    c = fujino_criteria()
    a, t, L = c["angle"].threshold, c["thickness"].threshold, c["length"].threshold
    assert fujino_score(a, t, L) == 0                       # exactly at thresholds: 0
    assert fujino_score(a, 0.0, None) == 0                  # exactly 180 deg scores 0
    assert fujino_score(a + 1e-9, 0.0) == c["angle"].points
    assert fujino_score(0.0, t + 1e-9) == c["thickness"].points
    assert fujino_score(0.0, 0.0, L + 1e-9) == c["length"].points
    assert fujino_score(a + 1, t + 0.1, None) == fujino_max_score(False)
    assert fujino_score(a + 1, t + 0.1, L + 1) == fujino_max_score(True)
    assert fujino_score(360.0, float("nan"), None) == c["angle"].points
    # values come from the literature table
    assert a == params.lit("fujino_angle_threshold").value
    assert (c["angle"].points, c["thickness"].points, c["length"].points) == (2, 1, 1)
    assert (fujino_max_score(False), fujino_max_score(True)) == (3, 4)


# ------------------------------------------------------------------ cohort
def _patched_config(monkeypatch):
    cfg = copy.deepcopy(params.load_config())
    monkeypatch.setattr(params, "load_config", lambda *a, **k: cfg)
    return cfg


def test_pending_refused_unless_allowed(monkeypatch):
    cfg = _patched_config(monkeypatch)
    cfg["geometry"]["plaque_thickness_mm"] = {"value": 0.8, "basis": "test placeholder", "status": "PENDING"}
    with pytest.raises(ValueError, match="PENDING"):
        cohort.core_specs()
    specs = cohort.core_specs(allow_pending=True)
    assert specs[0].plaque_thickness == 0.8
    assert specs[0].notes["pending_inputs"] == ["geometry.plaque_thickness_mm"]


def test_core_specs():
    specs = cohort.core_specs(allow_pending=True)
    f = params.raw("design.core_factorial")
    assert len(specs) == len(f["arc_deg"]) * len(f["thickness_mm"]) * len(f["morphology"]) == 32
    ids = [s.lesion_id for s in specs]
    assert len(set(ids)) == 32
    assert "C-A270-T0.4-NOD" in ids
    combos = {(s.ca_arc_deg, s.ca_thickness, s.morphology) for s in specs}
    assert len(combos) == 32
    geo = cohort.fixed_geometry(allow_pending=True)
    for s in specs:
        s.validate()
        lesion_polygons(s)
        assert s.set_name == "core" and not s.is_3d
        assert s.r_lumen == pytest.approx(math.sqrt(params.get("geometry.lumen_area_mm2").value / math.pi))
        assert s.ca_depth == (geo["depth_deep"] if s.morphology == "deep_sheet" else geo["depth_superficial"])
        assert (s.nodule_height > 0) == (s.morphology == "nodule")
        assert s.plaque_offset == 0 and s.lipid_arc_deg == 0


def test_pilot_and_example3d():
    pilots = cohort.pilot_specs(allow_pending=True)
    assert len(pilots) == len(params.raw("design.pilot_2d"))
    ex = cohort.example3d_spec(allow_pending=True)
    e = params.raw("design.fujino4_example_3d")
    assert ex.is_3d and ex.ca_length == e["length_mm"]
    assert ex.lesion_id == f"X3D-A{e['arc_deg']}-T{e['thickness_mm']}-L{e['length_mm']}-SUP"
    d = descriptors(rasterize_polar(ex), length_mm=ex.ca_length)
    assert d["fujino_score"] == 4 == fujino_max_score(True)


def test_holdout_requires_ranges(monkeypatch):
    cfg = _patched_config(monkeypatch)
    cfg["design"]["holdout_lhs"].pop("ranges", None)
    with pytest.raises(KeyError, match="ranges"):
        cohort.holdout_specs(allow_pending=True)


def test_holdout_from_config():
    """Hold-out set from the configured ranges: balanced, valid, inside the ranges."""
    cfg = params.raw("design.holdout_lhs")
    specs = cohort.holdout_specs(allow_pending=True)
    assert len(specs) == cfg["n"] and len({s.lesion_id for s in specs}) == cfg["n"]
    counts = pd.Series([s.morphology for s in specs]).value_counts()
    assert set(counts) == {cfg["n"] // 4}
    rng = cohort._parse_range("arc_deg", cfg["ranges"]["arc_deg"])
    arcs = np.array([s.ca_arc_deg for s in specs])
    assert arcs.min() >= rng[1] and arcs.max() <= rng[2]
    # Latin hypercube: one arc per stratum
    assert sorted(np.floor((arcs - rng[1]) / (rng[2] - rng[1]) * len(specs)).astype(int)) == list(range(len(specs)))
    clearance = cohort._depth_clearance(cfg["ranges"]["depth_rule"])
    for s in specs:
        s.validate()
        lesion_polygons(s)
        t_min = cohort._min_local_plaque(s.r_lumen, s.plaque_thickness, s.plaque_offset, s.ca_arc_deg,
                                         s.ca_center_deg)
        assert s.ca_depth + s.ca_thickness + clearance <= t_min + 1e-9
        assert (s.lipid_arc_deg > 0) == s.notes["has_lipid"]
    assert any(s.notes["has_lipid"] for s in specs)
    again = cohort.holdout_specs(allow_pending=True)
    assert [s.to_dict() for s in again] == [s.to_dict() for s in specs]


def test_holdout_ecc_index_and_custom_ranges(monkeypatch):
    cfg = _patched_config(monkeypatch)
    cfg["design"]["holdout_lhs"]["ranges"] = {
        "arc_deg": [60, 180], "thickness_mm": {"min": 0.2, "max": 0.3}, "ecc_index": [0.2, 0.4],
        "morphology": ["superficial_sheet", "fragmented"],
    }
    specs = cohort.holdout_specs(n=4, seed=7, allow_pending=True)
    assert {s.morphology for s in specs} <= {"superficial_sheet", "fragmented"}
    for s in specs:
        d = descriptors(rasterize_polar(s))
        assert d["plaque_ecc_index"] == pytest.approx(s.notes["ecc_index_target"], abs=0.03)
    cfg["design"]["holdout_lhs"]["ranges"]["bogus_key"] = [0, 1]
    with pytest.raises(ValueError, match="bogus_key"):
        cohort.holdout_specs(n=2, allow_pending=True)


def test_build_cohort(tmp_path):
    specs = cohort.pilot_specs(allow_pending=True) + [cohort.example3d_spec(allow_pending=True)]
    df = cohort.build_cohort(specs, tmp_path)
    assert (tmp_path / "cohort.csv").is_file()
    csv = pd.read_csv(tmp_path / "cohort.csv")
    assert len(df) == len(csv) == len(specs)
    for col in ("lesion_id", "morphology", "ca_arc_deg", "ca_thickness", "ca_arc_total_deg",
                "ca_arc_max_contig_deg", "ca_thickness_max", "ca_depth_min", "n_deposits",
                "lumen_area", "plaque_ecc_index", "fujino_score", "spec_lipid_arc_deg", "lipid_arc_deg"):
        assert col in csv.columns
    for s in specs:
        mask = tmp_path / "masks" / f"{s.lesion_id}.npz"
        png = tmp_path / "previews" / f"{s.lesion_id}.png"
        assert mask.is_file() and png.is_file() and png.stat().st_size > 0
        plm = PolarLabelMap.load(mask)
        plm.validate()
        assert np.array_equal(plm.labels, rasterize_polar(s).labels)
    row = csv.set_index("lesion_id").loc[specs[-1].lesion_id]
    assert row["fujino_score"] == 4
    from PIL import Image
    img = np.asarray(Image.open(tmp_path / "previews" / f"{specs[0].lesion_id}.png"))
    assert img.ndim == 3 and img.shape[2] == 3
    with pytest.raises(ValueError):
        cohort.build_cohort(specs + specs[:1], tmp_path / "dup")


# ------------------------------------------------------- review regressions
@pytest.mark.parametrize("arc", [40, 60])
def test_nodule_wider_than_calcium_raises(geo, arc):
    """A protrusion wider than the calcium inner arc would indent the lumen over no calcium."""
    s = _spec(geo, "nodule", 90, 0.2)
    s.ca_arc_deg = float(arc)
    assert s.nodule_base_arc_deg > arc - 2 * math.degrees(cap_half_angle(s))
    with pytest.raises(ValueError, match="nodule base arc"):
        lesion_polygons(s)
    # a closed ring always accommodates the bump; the core arcs fit
    for a in ARCS:
        build.check_nodule_fits(_spec(geo, "nodule", a, 0.4))


def test_holdout_depth_rule_never_silently_violated(monkeypatch):
    """If the calcium cannot meet the depth rule even at zero plaque offset, the sample is invalid."""
    cfg = _patched_config(monkeypatch)
    rule = params.raw("design.holdout_lhs.ranges.depth_rule")
    P = cohort.fixed_geometry(allow_pending=True)["plaque_thickness"]
    clearance = cohort._depth_clearance(rule)
    d_lo = 0.16  # numerical test setting (not a literature value)
    t_lo = P - d_lo - clearance + 0.005   # fits geometrically, but not with the clearance
    cfg["design"]["holdout_lhs"]["ranges"] = {
        "arc_deg": [340, 360], "thickness_mm": [t_lo, t_lo + 0.01], "depth_mm": [d_lo, d_lo + 0.04],
        "ecc_index": [0.3, 0.4], "morphology": ["superficial_sheet"], "depth_rule": rule}
    with pytest.raises(ValueError, match="depth_rule cannot be met"):
        cohort.holdout_specs(n=3, seed=1, allow_pending=True)
    assert cohort.holdout_specs(n=3, seed=1, allow_pending=True, on_invalid="skip") == []


def test_holdout_lipid_fraction_requires_lipid_arc(monkeypatch):
    cfg = _patched_config(monkeypatch)
    cfg["design"]["holdout_lhs"]["ranges"] = {"arc_deg": [90, 180], "thickness_mm": [0.2, 0.3],
                                              "lipid_fraction": 1.0, "lipid_thickness_mm": 0.2}
    with pytest.raises(KeyError, match="lipid_arc_deg"):
        cohort.holdout_specs(n=2, seed=1, allow_pending=True)

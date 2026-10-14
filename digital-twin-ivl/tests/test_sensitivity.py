"""Tests for the sensitivity stage (src/drivers/stage_sensitivity.py).

Only the last test runs the outcome model (transient + IVL + balloon alone on a small annulus with a
calcium sector, reduced increments); everything else checks scenario construction from the config,
the Morris design, the input mapping, censoring, transient reuse, claims, resume and the analysis on
synthetic outputs.
"""
import ast
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import fields

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pytest

from dtivl import params as P
from dtivl.drivers import stage_sensitivity as S
from dtivl.geometry.labels import LABEL_CALCIUM, LABEL_LUMEN, LABEL_OTHER, TAG_CALCIUM, TAG_WALL
from dtivl.geometry.polar import PolarLabelMap
from dtivl.materials.constitutive import MaterialSet

# small test lesion (as tests/test_nonlinear.py): lumen A, outer B, calcium sector R1..R2 over ARC degrees
A, B, R1, R2, ARC = 0.8, 1.5, 0.9, 1.1, 180.0
H, H_CA = 0.2, 0.08
FAST_STEPS = {"ivl": (2, 2, 2), "pressure": 2}
DENSITY_ROW = "density_calcified_plaque_model_input"


def sector_plm(a=A, b=B, r1=R1, r2=R2, arc=ARC, n_theta=360, dr=0.01) -> PolarLabelMap:
    th = 2 * np.pi * np.arange(n_theta) / n_theta
    r = (np.arange(int(np.ceil(b / dr))) + 0.5) * dr
    TH, R = np.meshgrid(th, r, indexing="ij")
    lab = np.full(TH.shape, LABEL_OTHER, np.uint8)
    lab[R < a] = LABEL_LUMEN
    dth = np.abs((np.degrees(TH) + 180.0) % 360.0 - 180.0)
    lab[(R >= r1) & (R < r2) & (dth <= arc / 2)] = LABEL_CALCIUM
    plm = PolarLabelMap(lab, dr, np.full(n_theta, b))
    plm.validate()
    return plm


@pytest.fixture(scope="module")
def annulus():
    from dtivl.solvers.nonlinear import annulus_meshdata

    md = annulus_meshdata(A, B, H, ca_sector=(R1, R2, ARC), h_ca=H_CA, tag=TAG_WALL)
    n = S._n_cells(md)
    info = {"n_nodes": None, "n_cells": n, "n_cells_per_tag": {}}
    return sector_plm(), info, md


def _own_and_ms(inp: S.ModelInputs) -> dict:
    d = {f"ms.{k}": getattr(inp.ms, k) for k in S._MS_FIELD_NAMES}
    d.update({f.name: getattr(inp, f.name) for f in fields(inp) if f.name not in ("ms", "changed")})
    return d


# ---------------------------------------------------------------------------
# literature parser: alternative value of a single-number range column (D15 density)
# ---------------------------------------------------------------------------
def test_alt_value_parsed_from_range_column():
    row = P.load_literature()[DENSITY_ROW]
    lead = float(re.match(r"\s*([-+]?\d+(?:\.\d+)?)", row.range_str).group(1))
    assert row.alt == pytest.approx(lead)
    assert row.lower is None and row.upper is None          # bounds semantics unchanged
    assert P.lit(DENSITY_ROW, "alt", allow_unverified=True).value == row.alt
    with pytest.raises(ValueError):                         # SECONDARY row: explicit allow_status needed
        P.lit(DENSITY_ROW, "alt")
    q = P.get("materials.density.calcium")
    assert q.value == row.value and q.sens == (row.alt,)
    for s in ("SD 0.5", "2-5", "IQR 0.1-0.2", "n=47", "47/180", "approx. 50 atm", ""):
        assert P._parse_alt(s, 1.0) is None
    with pytest.raises(ValueError):
        P.lit("calcification_fracture_stress", "alt")       # no single-number range column


# ---------------------------------------------------------------------------
# scenarios and inputs from the config (no typed values)
# ---------------------------------------------------------------------------
def test_no_float_literals_in_function_bodies():
    """Numbers in the driver come from the config / literature table; only named module-level
    numerical constants may carry float literals."""
    tree = ast.parse(open(S.__file__).read())
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Constant) and isinstance(sub.value, float):
                    bad.append((sub.lineno, sub.value))
    assert not bad, f"float literals in function bodies: {bad}"


def test_oat_scenarios_from_config():
    sc = S.oat_scenarios()
    base = S.ModelInputs.baseline()
    assert sc[0].name == S.BASELINE and sc[0].inputs == base
    by_key = {}
    for s in sc[1:]:
        by_key.setdefault(s.key, []).append(s)
        assert s.inputs.changed == (s.input,)
        a, b = _own_and_ms(s.inputs), _own_and_ms(base)
        diff = {k for k in a if a[k] != b[k]}
        assert diff == {f"ms.{s.input}" if s.input in S._MS_FIELD_NAMES else s.input}, s.name
        assert s.basis
    for key, inp, did in S.OAT_ENTRIES:
        node = S._raw_or_none(key)
        if not isinstance(node, dict) or "sens" not in node:
            assert key not in by_key
            continue
        if isinstance(node.get("value"), str):
            want = [v for v in node["sens"] if v != node["value"]]
        else:
            q = P.get(key)
            want = [v for v in q.sens if not math.isclose(v, q.value)]
        assert sorted(s.value for s in by_key[key]) == sorted(want), key
        if did:
            assert P.raw(f"decisions.{did}") in by_key[key][0].basis
    # the decisions named in the task are all present
    assert [s.value for s in by_key["fracture.crack_filler"]] == ["none"]                    # D9
    assert [s.value for s in by_key["materials.density.calcium"]] == [
        P.lit(DENSITY_ROW, "alt", allow_unverified=True).value]                              # D15
    for key in ("fracture.ell_over_h", "loads.ivl_pulse_scale", "loads.perivascular_k_MPa_per_mm",
                "materials.bulk_to_shear_soft"):
        assert by_key[key]
    st = by_key["failure_mode"]
    assert [s.value for s in st] == ["stress"]
    assert st[0].inputs.ms.failure_mode == "stress"
    assert st[0].inputs.ms.sigma_crit == pytest.approx(P.get("fracture.sigma_c_stress_controlled").value)
    names = [s.name for s in sc]
    assert len(set(names)) == len(names)
    assert S.n_oat_runs() == len(sc) * 3


def test_baseline_inputs_are_the_cohort_defaults():
    from dtivl.fracture import phasefield as PF
    from dtivl.solvers import transient as TR
    from dtivl.solvers.nonlinear import SolverSettings

    base = S.ModelInputs.baseline()
    assert base.ms == MaterialSet.baseline()
    kw = base.state_kwargs()
    assert kw["ell"] == pytest.approx(PF.regularisation_length())
    assert kw["k_found"] == P.get("loads.perivascular_k_MPa_per_mm").value
    assert kw["settings"].driving_force == SolverSettings().driving_force
    assert base.pulse_scale == P.get("loads.ivl_pulse_scale").value
    assert base.densities() == TR.default_densities()
    # set() returns a copy: the baseline and its provenance are untouched
    prov = dict(base.ms.provenance)
    other = base.set(ca_E=2 * base.ms.ca_E, pulse_scale=base.pulse_scale / 2)
    assert base.ms.provenance == prov and base.ms.ca_E == MaterialSet.baseline().ca_E
    assert other.ms.provenance["ca_E"] == "sensitivity" and other.changed == ("ca_E", "pulse_scale")
    with pytest.raises(KeyError):
        base.set(not_an_input=1)
    d = other.describe()
    json.dumps(d)
    assert d["ms"]["ca_E"] == other.ms.ca_E and d["k_floor"] == other.ms.k_floor


# ---------------------------------------------------------------------------
# Morris design
# ---------------------------------------------------------------------------
def test_morris_inputs_from_config_and_literature():
    from dtivl.drivers.stage_pilot import e_sweep

    inputs = {m.name: m for m in S.morris_inputs()}
    assert all(m.basis for m in inputs.values())
    assert len(inputs) == len(S.morris_inputs())
    base = MaterialSet.baseline()
    lo, hi = S.sweep_neighbours()
    pts = e_sweep()
    i = pts.index(lo)
    assert pts[i + 2] == hi and math.isclose(pts[i + 1], base.ca_E, rel_tol=S.SWEEP_MATCH_RTOL)
    assert inputs["log10_ca_E"].bounds == (math.log10(lo), math.log10(hi))
    assert inputs["log10_ca_E"].value(inputs["log10_ca_E"].bounds[1]) == pytest.approx(hi)
    row = P.load_literature()["calcification_fracture_stretch"]
    assert inputs["eps_f"].bounds == pytest.approx((row.lower - 1, row.upper - 1))
    assert inputs["eps_f"].bounds[0] < base.eps_f < inputs["eps_f"].bounds[1]

    def span(key):
        q = P.get(key)
        return min((q.value,) + q.sens), max((q.value,) + q.sens)

    assert inputs["ivl_pulse_scale"].bounds == span("loads.ivl_pulse_scale")
    assert inputs["ell_over_h"].bounds == span("fracture.ell_over_h")
    assert inputs["rho_calcium"].bounds == span("materials.density.calcium")
    assert inputs["k_over_mu_soft"].bounds == span("materials.bulk_to_shear_soft")
    k_lo, k_hi = span("loads.perivascular_k_MPa_per_mm")
    assert inputs["log10_k_perivascular"].bounds == (math.log10(k_lo), math.log10(k_hi))
    fib = [P.get("materials.fibrous.E").value] + [q.value for q in P.get_list("materials.fibrous.sens_E")]
    assert inputs["fibrous_E"].bounds == (min(fib), max(fib))
    for m in inputs.values():
        assert m.bounds[0] < m.bounds[1]
    cfg = S.morris_config()
    assert cfg["seed"] == int(P.raw("design.seed")) + int(P.raw("design.sensitivity")["seed_offset"])
    assert S.n_morris_runs() == cfg["r"] * (len(inputs) + 1) * len(cfg["lesions"])
    assert S.n_morris_runs() <= 250                       # task budget
    assert str(S.n_morris_runs()) in S.__doc__            # run count stated in the docstring


def test_morris_sampling_reproducible():
    cfg = S.morris_config()
    k = len(S.morris_inputs())
    X1, X2 = S.morris_samples(), S.morris_samples()
    assert X1.shape == (cfg["r"] * (k + 1), k)
    assert np.array_equal(X1, X2)
    assert not np.array_equal(X1, S.morris_samples(seed=cfg["seed"] + 1))
    assert S.morris_design()["X"] == X1.tolist()
    for c, m in enumerate(S.morris_inputs()):
        lo, hi = m.bounds
        u = (X1[:, c] - lo) / (hi - lo) * (cfg["levels"] - 1)
        assert np.allclose(u, np.round(u), atol=1e-9) and u.min() >= -1e-9 and u.max() <= cfg["levels"] - 1 + 1e-9
    for t in range(cfg["r"]):                             # one factor changes per step of a trajectory
        T = X1[t * (k + 1):(t + 1) * (k + 1)]
        changed = (np.abs(np.diff(T, axis=0)) > 0).sum(axis=1)
        assert np.all(changed == 1)
        assert sorted(np.argmax(np.abs(np.diff(T, axis=0)) > 0, axis=1)) == list(range(k))


def test_morris_point_maps_to_material_set_and_settings():
    from dtivl.solvers import transient as TR

    inputs = S.morris_inputs()
    x = S.morris_samples()[0]
    mi = S.morris_point(x)
    v = {m.name: m.value(xv) for m, xv in zip(inputs, x)}
    assert mi.ms.ca_E == pytest.approx(10 ** x[0]) == pytest.approx(v["log10_ca_E"])
    assert mi.ms.eps_f == v["eps_f"] and mi.ms.k_over_mu == v["k_over_mu_soft"] and mi.ms.fibrous_E == v["fibrous_E"]
    assert mi.pulse_scale == v["ivl_pulse_scale"] and mi.ell_over_h == v["ell_over_h"]
    assert mi.k_found == pytest.approx(v["log10_k_perivascular"]) and mi.rho_calcium == v["rho_calcium"]
    rho = mi.densities()
    base_rho = TR.default_densities()
    assert rho[TAG_CALCIUM] == v["rho_calcium"] * TR.KG_PER_M3_TO_N_US2_PER_MM4
    assert {t: r for t, r in rho.items() if t != TAG_CALCIUM} == {t: r for t, r in base_rho.items() if t != TAG_CALCIUM}
    kw = mi.state_kwargs()
    assert kw["ell"] == pytest.approx(v["ell_over_h"] * float(P.raw("discretisation.h_calcium_mm")))
    assert kw["k_found"] == mi.k_found and kw["settings"].driving_force == S.ModelInputs.baseline().driving_force
    # untouched MaterialSet fields keep the baseline
    base = MaterialSet.baseline()
    for f in ("wall_c10", "lipid_E", "ca_nu", "sigma_c", "failure_mode", "crack_filler", "k_res"):
        assert getattr(mi.ms, f) == getattr(base, f)
    # k_floor follows the sampled moduli
    assert mi.ms.k_floor == pytest.approx(max(base.k_res, mi.ms.mu_fibrous / mi.ms.ca_mu))
    with pytest.raises(ValueError):
        S.morris_point(x[:-1])


# ---------------------------------------------------------------------------
# outputs and censoring
# ---------------------------------------------------------------------------
def _prec(status="ok", onset=None, phase=None, fracture=False, p_top=0.6, failed_phase=None):
    summ = {"p_top": p_top}
    if failed_phase:
        summ["failed_phase"] = failed_phase
    return {"status": status, "summary": summ, "n_steps_unconverged_stagger": 0,
            "cracks": {"fracture": fracture, "onset_load": onset, "onset_phase": phase, "damaged_fraction": 0.0}}


def test_censored_onset_and_fracture_flags():
    c = S.censored_onset(_prec(onset=0.5, phase="ivl_pulse", fracture=True))
    assert c == {"value": 0.5, "censored": False, "after_peak": False}
    assert S.censored_onset(_prec(onset=0.3, phase="ivl_inflate"))["value"] == 0.3
    c = S.censored_onset(_prec(onset=0.2, phase="ivl_unload"))
    assert c == {"value": 0.6, "censored": False, "after_peak": True}
    assert S.censored_onset(_prec()) == {"value": 0.6, "censored": True, "after_peak": False}
    # failure before the peak without onset: unknown; after the peak: censored at p_top
    assert S.censored_onset(_prec("failed", failed_phase="ivl_pulse"))["value"] is None
    assert S.censored_onset(_prec("failed", failed_phase="ivl_unload"))["censored"] is True
    assert S.censored_onset(_prec("failed"))["censored"] is True            # release step failed
    assert S.censored_onset(_prec("failed", onset=0.45, phase="ivl_pulse", failed_phase="ivl_pulse"))["value"] == 0.45
    assert S.censored_onset({"status": "error", "error": "x"})["value"] is None
    # fracture: True once found, False only when decided
    assert S.fracture_flag(_prec("failed", fracture=True, failed_phase="ivl_pulse"), False) is True
    assert S.fracture_flag(_prec(), True) is False
    assert S.fracture_flag(_prec("failed", failed_phase="ivl_pulse"), False) is None
    assert S.fracture_flag({"status": "error"}, False) is None
    rec = {"transient": {"T_equivalent": 0.05},
           "ivl": _prec(onset=0.5, phase="ivl_pulse", fracture=True),
           "balloon_only": _prec("failed", p_top=None)}
    o = S.outputs(rec)
    assert o["ivl_fracture"] is True and o["ivl_onset_load"] == 0.5 and o["ivl_onset_censored"] is False
    assert o["balloon_fracture"] is None and o["balloon_status"] == "failed"
    assert o["T_equivalent"] == 0.05 and o["ivl_p_top"] == 0.6
    assert set(S.MORRIS_OUTPUTS) <= set(o)


# ---------------------------------------------------------------------------
# transient reuse, claims
# ---------------------------------------------------------------------------
def test_transient_key_and_reuse(annulus, tmp_path):
    _, _, md = annulus
    base = S.ModelInputs.baseline()
    k0 = S.transient_key(base, md)
    same = [base.set(pulse_scale=base.pulse_scale / 2), base.set(ell_over_h=2 * base.ell_over_h),
            base.set(eps_f=2 * base.ms.eps_f), base.set(crack_filler="none"), base.set(failure_mode="stress"),
            base.set(driving_force="rankine")]
    diff = [base.set(rho_calcium=base.rho_calcium / 2), base.set(k_found=10 * base.k_found),
            base.set(ca_E=2 * base.ms.ca_E), base.set(k_over_mu=2 * base.ms.k_over_mu),
            base.set(fibrous_E=2 * base.ms.fibrous_E)]
    assert all(S.transient_key(x, md) == k0 for x in same)
    assert len({S.transient_key(x, md) for x in diff} | {k0}) == len(diff) + 1
    assert S._cached_T(tmp_path, "L1", k0) == (None, None)
    (tmp_path / "L1__a.json").write_text(json.dumps({"lesion": "L1", "transient": {
        "T_equivalent": 0.07, "source": "computed", "key": k0}}))
    (tmp_path / "L2__b.json").write_text(json.dumps({"lesion": "L2", "transient": {
        "T_equivalent": 0.09, "source": "computed", "key": k0}}))
    (tmp_path / "L1__c.json").write_text(json.dumps({"lesion": "L1", "transient": {
        "T_equivalent": 0.08, "source": "reused from x", "key": "other"}}))
    assert S._cached_T(tmp_path, "L1", k0) == (0.07, "reused from L1__a")
    assert S._cached_T(tmp_path, "L1", "other") == (None, None)


def test_claim_and_stale_locks(tmp_path):
    lock = tmp_path / "locks" / "x.lock"
    assert S._claim(lock)
    assert not S._claim(lock)                 # held by this (live) process
    S._release(lock)
    p = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"], capture_output=True, text=True)
    lock.write_text(p.stdout.strip())         # pid of a finished process: stale
    assert S._claim(lock)
    lock.write_text("")                       # fresh empty lock: being claimed by another process
    assert not S._claim(lock)
    old = lock.stat().st_mtime - 2 * S.EMPTY_LOCK_GRACE_S
    os.utime(lock, (old, old))                # old empty lock: stale
    assert S._claim(lock)
    S._release(lock)
    assert not lock.exists()


# ---------------------------------------------------------------------------
# drivers with a stub outcome model (resume, files, analysis)
# ---------------------------------------------------------------------------
@pytest.fixture
def stub_model(monkeypatch, annulus):
    calls = []

    def fake_run_model(md, plm, inputs, *, T=None, T_source=None, n_steps=None, **kw):
        calls.append((inputs, T))
        key = S.transient_key(inputs, md)
        tr = ({"T_equivalent": 0.05, "source": "computed", "key": key} if T is None
              else {"T_equivalent": T, "source": T_source, "key": key})
        rec = {"inputs": inputs.describe(), "transient": tr,
               "ivl": _prec(onset=0.5, phase="ivl_pulse", fracture=True), "balloon_only": _prec(p_top=None)}
        rec["outputs"] = S.outputs(rec)
        return rec

    monkeypatch.setattr(S, "run_model", fake_run_model)
    monkeypatch.setattr(S, "lesion_mesh", lambda lid: annulus)
    return calls


def test_one_at_a_time_files_resume_and_reuse(stub_model, tmp_path):
    rho_alt = P.get("materials.density.calcium").sens[0]
    names = [S.BASELINE, "fracture.crack_filler=none", f"materials.density.calcium={rho_alt:g}"]
    out = S.one_at_a_time(["L1"], names, out_dir=tmp_path)
    assert len(stub_model) == 3
    files = sorted(p.name for p in (tmp_path / S.OAT_RUNS).glob("*.json"))
    assert files == sorted(f"L1__{n}.json" for n in names)
    # crack filler: transient inputs unchanged -> T reused from the baseline; density: recomputed
    by = {S.BASELINE: stub_model[0], "filler": stub_model[1], "density": stub_model[2]}
    assert by[S.BASELINE][1] is None and by["filler"][1] == 0.05 and by["density"][1] is None
    summary = json.loads((tmp_path / S.OAT_JSON).read_text())
    assert summary == json.loads(json.dumps(out))
    assert summary["n_runs_complete"] == 3 and [r["scenario"] for r in summary["table"]] == names
    row = summary["runs"]["L1"]["fracture.crack_filler=none"]
    assert row["T_source"] == f"reused from L1__{S.BASELINE}" and row["ivl_fracture_changed_vs_baseline"] is False
    assert row["ivl_onset_load_minus_baseline"] == 0.0
    S.one_at_a_time(["L1"], names, out_dir=tmp_path)            # resume: nothing re-run
    assert len(stub_model) == 3
    with pytest.raises(KeyError):
        S.one_at_a_time(["L1"], ["no-such-scenario"], out_dir=tmp_path)


def test_morris_driver_resume_reuse_and_design_check(stub_model, annulus, tmp_path):
    done = S.morris(["L1"], out_dir=tmp_path, max_runs=3)
    assert done == ["L1_000", "L1_001", "L1_002"]
    rec = json.loads((tmp_path / S.MORRIS_RUNS / "L1_001.json").read_text())
    X = S.morris_samples()
    assert rec["x"] == X[1].tolist() and rec["complete"] and rec["trajectory"] == 0 and rec["step"] == 1
    assert S.morris(["L1"], out_dir=tmp_path, max_runs=2) == ["L1_003", "L1_004"]
    assert len(stub_model) == 5
    # T_equivalent is recomputed exactly when the transient inputs differ from every earlier sample
    keys = [S.transient_key(S.morris_point(x), annulus[2]) for x in X[:5]]
    for i, (_, T) in enumerate(stub_model):
        assert (T is None) == (keys[i] not in keys[:i])
    # a changed design is refused
    dpath = tmp_path / S.MORRIS_RUNS / S.MORRIS_DESIGN
    d = json.loads(dpath.read_text())
    d["X"][0][0] += 1
    dpath.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        S.morris(["L1"], out_dir=tmp_path, max_runs=1)


def test_analyze_on_synthetic_outputs(tmp_path):
    """mu_star from known linear / step responses; a missing sample excludes its trajectory."""
    design = S.morris_design()
    run_dir = tmp_path / S.MORRIS_RUNS
    run_dir.mkdir()
    (run_dir / S.MORRIS_DESIGN).write_text(json.dumps(design))
    X = np.asarray(design["X"])
    lo = np.array([b[0] for b in design["problem"]["bounds"]])
    hi = np.array([b[1] for b in design["problem"]["bounds"]])
    U = (X - lo) / (hi - lo)                                     # normalised inputs
    k = X.shape[1]
    first, second = design["lesions"][:2]
    for lid in design["lesions"]:
        for i, u in enumerate(U):
            if lid == first and i == 0:
                continue                                         # missing sample -> trajectory 0 excluded
            onset = 0.3 + 0.2 * u[0] - 0.1 * u[3]
            out = {"ivl_onset_load": onset, "ivl_onset_censored": False, "ivl_fracture": bool(u[1] > 0.5),
                   "balloon_fracture": False, "ivl_status": "ok", "balloon_status": "ok"}
            if lid == second and i == k + 1:
                out["ivl_fracture"] = None                       # undecided -> trajectory 1 excluded for it
            (run_dir / f"{S._morris_stem(lid, i)}.json").write_text(json.dumps(
                {"lesion": lid, "index": i, "complete": True, "outputs": out}))
    res = S.analyze(tmp_path)
    assert json.loads((tmp_path / S.MORRIS_JSON).read_text()) == json.loads(json.dumps(res))
    r = design["r"]
    names = design["problem"]["names"]
    a = res["lesion_results"][first]["outputs"]["ivl_onset_load"]
    assert a["excluded_trajectories"] == [0] and a["n_trajectories"] == r - 1
    idx = a["indices"]
    assert idx[names[0]]["mu_star"] == pytest.approx(0.2) and idx[names[0]]["mu"] == pytest.approx(0.2)
    assert idx[names[3]]["mu_star"] == pytest.approx(0.1) and idx[names[3]]["mu"] == pytest.approx(-0.1)
    assert idx[names[0]]["sigma"] == pytest.approx(0.0, abs=1e-12)
    for n in names[1:3] + names[4:]:
        assert idx[n]["mu_star"] == pytest.approx(0.0, abs=1e-12)
    f2 = res["lesion_results"][second]["outputs"]["ivl_fracture"]
    assert f2["excluded_trajectories"] == [1]
    assert res["lesion_results"][second]["outputs"]["ivl_onset_load"]["excluded_trajectories"] == []
    bal = res["lesion_results"][second]["outputs"]["balloon_fracture"]["indices"]
    assert all(v["mu_star"] == 0.0 for v in bal.values())
    pooled = res["pooled"]["ivl_onset_load"]
    assert pooled["n_trajectories"] == r * len(design["lesions"]) - 1
    assert pooled["indices"][names[0]]["mu_star"] == pytest.approx(0.2)
    assert res["n_runs_complete"] == len(X) * len(design["lesions"]) - 1


# ---------------------------------------------------------------------------
# one small end-to-end run of the real outcome model
# ---------------------------------------------------------------------------
def test_end_to_end_tiny_annulus(monkeypatch, annulus, tmp_path):
    """Baseline scenario on a small annulus (Yeoh wall, 180-degree calcium sector) with reduced
    increments: transient T_equivalent, IVL and balloon alone, files and summary."""
    monkeypatch.setattr(S, "lesion_mesh", lambda lid: annulus)
    out = S.one_at_a_time(["TINY"], [S.BASELINE], out_dir=tmp_path, n_steps=FAST_STEPS, h_ca=H_CA)
    rec = json.loads((tmp_path / S.OAT_RUNS / f"TINY__{S.BASELINE}.json").read_text())
    assert rec["complete"] and not rec.get("error")
    tr = rec["transient"]
    assert tr["source"] == "computed" and tr["T_equivalent"] > 0 and not tr["peak_at_window_end"]
    o = rec["outputs"]
    assert o["ivl_status"] in ("ok", "failed") and o["balloon_status"] in ("ok", "failed", "limited")
    base = S.ModelInputs.baseline()
    p_top = P.get("loads.ivl_balloon_pressure_atm").value + tr["T_equivalent"] * P.get(
        "loads.ivl_peak_pressure_MPa").value * base.pulse_scale
    assert o["ivl_p_top"] == pytest.approx(p_top)
    if o["ivl_onset_censored"] is False and not o["ivl_onset_after_peak"]:
        assert o["ivl_onset_load"] == o["ivl_onset_load_raw"] <= p_top + 1e-12
    elif o["ivl_onset_censored"]:
        assert o["ivl_onset_load_raw"] is None and o["ivl_onset_load"] == pytest.approx(p_top)
    assert isinstance(o["ivl_fracture"], bool) or o["ivl_fracture"] is None
    for p in ("ivl", "balloon_only"):
        assert rec[p]["seconds"] > 0 and "n_steps_unconverged_stagger" in rec[p]
        assert 0.0 <= rec[p]["cracks"]["damaged_fraction"] <= 1.0
    assert rec["inputs"]["ms"]["ca_E"] == base.ms.ca_E and rec["inputs"]["changed"] == []
    assert rec["inputs"]["ell_mm"] == pytest.approx(base.ell(H_CA))
    assert out["runs"]["TINY"][S.BASELINE]["ivl_fracture"] == o["ivl_fracture"]

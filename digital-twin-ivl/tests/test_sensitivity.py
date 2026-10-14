"""Tests for the sensitivity stage (src/drivers/stage_sensitivity.py).

Only the last test runs the outcome model (transient + IVL + balloon alone on a small annulus with a
calcium sector, reduced increments); everything else checks scenario construction from the config,
the Morris design, the input mapping and its forwarding to the solvers (recorders in place of the
solvers), censoring and onset load steps, transient reuse, claims, resume (protocol errors) and the
analysis on synthetic outputs.
"""
import ast
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import fields
from types import SimpleNamespace

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
    for s in ("SD 0.5", "2-5", "IQR 0.1-0.2", "n=47", "47/180", "approx. 50 atm", "",
              "1450", "96 (score 0-3)", "155 (patients)", "3 (other balloon)", "SD 6 (other FE input)"):
        assert P._parse_alt(s, 1.0) is None, s                # only a remark marking an alternative value
    for s, v in (("1450 (other FE input)", 1450), (" 2.5 (alternative value) ", 2.5), ("7 (alt. input)", 7)):
        assert P._parse_alt(s, 1.0) == v and P._parse_alt(s, 2.0) == 2 * v
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
    # censored only if the protocol ran to its end: a failure before or after the peak without onset is
    # unknown (the onset could come in the phases that never ran)
    unknown = {"value": None, "censored": None, "after_peak": None}
    assert S.censored_onset(_prec("failed", failed_phase="ivl_pulse")) == unknown
    assert S.censored_onset(_prec("failed", failed_phase="ivl_unload")) == unknown
    assert S.censored_onset(_prec("failed")) == unknown                     # release step failed
    assert S.censored_onset(_prec("failed", onset=0.45, phase="ivl_pulse", failed_phase="ivl_pulse"))["value"] == 0.45
    assert S.censored_onset(_prec("failed", onset=0.45, phase="ivl_pulse", failed_phase="ivl_unload")) == {
        "value": 0.45, "censored": False, "after_peak": False}
    assert S.censored_onset(_prec("failed", onset=0.2, phase="ivl_unload", failed_phase="ivl_unload")) == {
        "value": 0.6, "censored": False, "after_peak": True}
    assert S.censored_onset({"status": "error", "error": "x"}) == unknown
    # failed after the peak: while unloading or at the release step, not in a loading phase
    assert S.ivl_failed_after_peak(_prec("failed", failed_phase="ivl_unload"))
    assert S.ivl_failed_after_peak(_prec("failed"))
    for prec in (_prec(), _prec("failed", failed_phase="ivl_pulse"), _prec("failed", failed_phase="ivl_inflate"),
                 {"status": "error", "error": "x"}):
        assert not S.ivl_failed_after_peak(prec)
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
    assert o["ivl_onset_load_uncensored"] == 0.5 and o["ivl_failed_after_peak"] is False
    assert o["balloon_fracture"] is None and o["balloon_status"] == "failed"
    assert o["T_equivalent"] == 0.05 and o["ivl_p_top"] == 0.6
    assert set(S.MORRIS_OUTPUTS) <= set(o)
    # IVL failed after the peak: fracture and onset decided only if found by then
    o = S.outputs({"ivl": _prec("failed", failed_phase="ivl_unload"), "balloon_only": _prec()})
    assert o["ivl_fracture"] is None and o["ivl_onset_load"] is None and o["ivl_onset_censored"] is None
    assert o["ivl_onset_load_uncensored"] is None and o["ivl_failed_after_peak"] is True
    assert o["balloon_fracture"] is False
    o = S.outputs({"ivl": _prec("failed", onset=0.5, phase="ivl_pulse", fracture=True, failed_phase="ivl_unload")})
    assert o["ivl_fracture"] is True and o["ivl_onset_load"] == 0.5 and o["ivl_failed_after_peak"] is True
    o = S.outputs({"ivl": _prec("failed", onset=0.5, phase="ivl_pulse", failed_phase="ivl_unload")})
    assert o["ivl_fracture"] is None and o["ivl_onset_load"] == 0.5
    # censored and after-peak values sit at p_top: missing in the uncensored output
    for prec in (_prec(), _prec(onset=0.2, phase="ivl_unload")):
        o = S.outputs({"ivl": prec})
        assert o["ivl_onset_load"] == 0.6 and o["ivl_onset_load_uncensored"] is None
        assert o["ivl_onset_load_lower"] is None and o["ivl_onset_step"] is None


def test_onset_bracket_and_load_step():
    """Load of the step before the onset and the step size, from the protocol history (same rule as
    cracks.onset_load); kept in the outputs only for an onset during inflation or pulse."""
    dc = float(P.raw("fracture.d_crack"))
    hist = [{"phase": "ivl_inflate", "load": 0.2, "increment": 0.2, "max_d": 0},
            {"phase": "ivl_inflate", "load": 0.4, "increment": 0.2, "max_d": dc / 2},
            {"phase": "ivl_pulse", "load": 0.45, "increment": 0.05, "max_d": dc},
            {"phase": "ivl_pulse", "load": 0.5, "increment": 0.05, "max_d": 1}]
    br = S.onset_bracket(hist)
    assert br["load"] == 0.45 and br["phase"] == "ivl_pulse"
    assert br["lower"] == 0.4 and br["step"] == pytest.approx(0.05)
    br0 = S.onset_bracket([{**hist[0], "max_d": dc}] + hist[1:])        # onset at the first step: ramp start
    assert br0["load"] == 0.2 and br0["lower"] == pytest.approx(0.0) and br0["step"] == pytest.approx(0.2)
    assert S.onset_bracket(hist[:2]) == {"load": None, "phase": None, "lower": None, "step": None}
    assert S.onset_bracket([]) == {"load": None, "phase": None, "lower": None, "step": None}
    assert S.onset_bracket(hist, d_crack=dc / 4)["load"] == 0.4
    ivl = {**_prec(onset=0.45, phase="ivl_pulse"), "onset_bracket": br}
    o = S.outputs({"ivl": ivl})
    assert o["ivl_onset_load"] == 0.45 == o["ivl_onset_load_uncensored"]
    assert o["ivl_onset_load_lower"] == 0.4 and o["ivl_onset_step"] == pytest.approx(0.05)
    # a bracket of another onset (or none) is not used
    o = S.outputs({"ivl": {**_prec(onset=0.5, phase="ivl_pulse"), "onset_bracket": br}})
    assert o["ivl_onset_load_lower"] is None and o["ivl_onset_step"] is None
    o = S.outputs({"ivl": _prec(onset=0.45, phase="ivl_pulse")})
    assert o["ivl_onset_step"] is None and o["ivl_onset_load"] == 0.45


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
# run_model forwards every input to the solvers (recorders in place of the FE solves)
# ---------------------------------------------------------------------------
def test_run_model_forwards_inputs_to_solvers(monkeypatch, annulus):
    from dtivl.drivers import stage_pilot
    from dtivl.solvers import nonlinear as NL
    from dtivl.solvers import transient as TR

    plm, _, md = annulus
    dc = float(P.raw("fracture.d_crack"))
    base = S.ModelInputs.baseline()
    ref, forces = S._sens_values("fracture.driving_force")
    other_force = next(v for v in forces if v != ref)
    rho_alt = P.get("materials.density.calcium").sens[0]
    inputs = base.set(ca_E=2 * base.ms.ca_E, rho_calcium=rho_alt, k_found=10 * base.k_found,
                      ell_over_h=2 * base.ell_over_h, driving_force=other_force, pulse_scale=base.pulse_scale / 2)
    for name in ("rho_calcium", "k_found", "ell_over_h", "driving_force", "pulse_scale"):
        assert getattr(inputs, name) != getattr(base, name), name
    assert inputs.ms.ca_E != base.ms.ca_E and inputs.densities() != base.densities()
    calls = {"tf": [], "state": [], "ivl": [], "pressure": []}
    T_eq = 0.07

    def fake_tf(md_, plm_, ms, densities=None, sound_speeds=None, **kw):
        calls["tf"].append({"md": md_, "plm": plm_, "ms": ms, "densities": densities, **kw})
        return {k: T_eq for k in S.TRANSIENT_KEYS}

    class FakeState:
        def __init__(self, md_, ms=None, **kw):
            calls["state"].append({"md": md_, "ms": ms, **kw})
            self.kw = kw
            self.history = []

    def fake_ivl(state, *, T, pulse_scale=None, n_steps=None, **kw):
        calls["ivl"].append({"state": state, "T": T, "pulse_scale": pulse_scale, "n_steps": n_steps, **kw})
        state.history = [{"phase": "ivl_inflate", "load": 0.3, "increment": 0.3, "max_d": 0},
                         {"phase": "ivl_pulse", "load": 0.5, "increment": 0.2, "max_d": dc}]
        return SimpleNamespace(state=state, history=state.history, status="ok", summary={"p_top": 0.6})

    def fake_pressure(state, p_target, *, cap_radius=None, n_steps=None, phase="pressure"):
        calls["pressure"].append({"state": state, "p_target": p_target, "cap_radius": cap_radius,
                                  "n_steps": n_steps, "phase": phase})
        return SimpleNamespace(state=state, history=state.history, status="ok", summary={})

    def fake_record(res, md_, plm_, **kw):
        on = next((h for h in res.state.history if h["max_d"] >= dc), None)
        return {"status": res.status, "summary": res.summary, "n_steps_unconverged_stagger": 0,
                "cracks": {"fracture": False, "onset_load": on and on["load"], "onset_phase": on and on["phase"],
                           "damaged_fraction": 0.0}}

    monkeypatch.setattr(TR, "transfer_factor", fake_tf)
    monkeypatch.setattr(NL, "NonlinearState", FakeState)
    monkeypatch.setattr(NL, "run_ivl", fake_ivl)
    monkeypatch.setattr(NL, "run_pressure", fake_pressure)
    monkeypatch.setattr(stage_pilot, "_protocol_record", fake_record)
    rec = S.run_model(md, plm, inputs, n_steps=FAST_STEPS, h_ca=H_CA)
    # transient: material set, densities (calcium density replaced) and foundation of these inputs
    (tf,) = calls["tf"]
    assert tf["md"] is md and tf["plm"] is plm and tf["ms"] is inputs.ms
    assert tf["densities"] == inputs.densities()
    assert tf["densities"][TAG_CALCIUM] == rho_alt * TR.KG_PER_M3_TO_N_US2_PER_MM4
    assert tf["k_found"] == inputs.k_found
    assert rec["transient"]["source"] == "computed" and rec["transient"]["T_equivalent"] == T_eq
    assert rec["transient"]["key"] == S.transient_key(inputs, md)
    # nonlinear states: material set, ell, foundation and driving force of these inputs, one per protocol
    assert [c["name"] for c in calls["state"]] == ["sens_ivl", "sens_bal"]
    for c in calls["state"]:
        assert c["md"] is md and c["ms"] is inputs.ms
        assert c["ell"] == pytest.approx(inputs.ell_over_h * H_CA) and c["ell"] == inputs.ell(H_CA)
        assert c["k_found"] == inputs.k_found and c["settings"].driving_force == other_force
    # IVL: T = T_equivalent, pulse scale and increments; balloon alone: IVL balloon pressure and cap
    (ivl,) = calls["ivl"]
    assert ivl["state"].kw["name"] == "sens_ivl" and ivl["T"] == T_eq
    assert ivl["pulse_scale"] == inputs.pulse_scale and ivl["n_steps"] == FAST_STEPS["ivl"]
    (bal,) = calls["pressure"]
    assert bal["state"].kw["name"] == "sens_bal" and bal["phase"] == "balloon_only"
    assert bal["p_target"] == P.get("loads.ivl_balloon_pressure_atm").value
    assert bal["cap_radius"] == P.get("geometry.ivl_balloon_diameter_mm").value / 2
    assert bal["n_steps"] == FAST_STEPS["pressure"]
    # record: inputs, onset bracket from the IVL history, outputs
    assert rec["inputs"] == inputs.describe(H_CA)
    assert rec["ivl"]["onset_bracket"] == {"load": 0.5, "phase": "ivl_pulse", "lower": 0.3,
                                           "step": pytest.approx(0.2)}
    o = rec["outputs"]
    assert o["ivl_onset_load"] == 0.5 and o["ivl_onset_load_lower"] == 0.3 and o["ivl_onset_step"] == pytest.approx(0.2)
    assert o["ivl_status"] == o["balloon_status"] == "ok" and o["T_equivalent"] == T_eq
    # a given T: no transient computation, the IVL protocol gets that T
    rec2 = S.run_model(md, plm, inputs, T=T_eq / 2, T_source="reused from x", n_steps=FAST_STEPS, h_ca=H_CA)
    assert len(calls["tf"]) == 1 and calls["ivl"][-1]["T"] == T_eq / 2
    assert rec2["transient"] == {"T_equivalent": T_eq / 2, "source": "reused from x", "key": S.transient_key(inputs, md)}


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


def test_protocol_error_leaves_run_incomplete_and_resume_retries(monkeypatch, annulus, tmp_path):
    """A protocol exception is recorded (status 'error') but the run is not complete, so resume runs it
    again; the transient result of the failed run is reused."""
    from dtivl.drivers import stage_pilot

    def boom():
        raise RuntimeError("boom")

    rec = S._run_protocol(boom, None, None)
    assert rec["status"] == "error" and rec["error"] == "RuntimeError: boom" and rec["seconds"] >= 0
    res = SimpleNamespace(state=SimpleNamespace(history=[]), history=[], status="ok", summary={})
    monkeypatch.setattr(stage_pilot, "_protocol_record", lambda r, md, plm: {"status": r.status})
    assert S._run_protocol(lambda: res, None, None)["onset_bracket"]["load"] is None
    calls = []

    def fake_run_model(md, plm, inputs, *, T=None, T_source=None, n_steps=None, **kw):
        calls.append(T)
        tr = ({"T_equivalent": 0.05, "source": "computed", "key": S.transient_key(inputs, md)} if T is None
              else {"T_equivalent": T, "source": T_source, "key": S.transient_key(inputs, md)})
        ivl = {"status": "error", "error": "RuntimeError: boom"} if len(calls) == 1 else _prec()
        rec = {"inputs": inputs.describe(), "transient": tr, "ivl": ivl, "balloon_only": _prec(p_top=None)}
        rec["outputs"] = S.outputs(rec)
        return rec

    monkeypatch.setattr(S, "run_model", fake_run_model)
    monkeypatch.setattr(S, "lesion_mesh", lambda lid: annulus)
    out = S.one_at_a_time(["L1"], [S.BASELINE], out_dir=tmp_path)
    path = tmp_path / S.OAT_RUNS / f"L1__{S.BASELINE}.json"
    rec = json.loads(path.read_text())
    assert not rec.get("complete") and rec["ivl"]["status"] == "error"
    assert "ivl" in rec["error"] and "boom" in rec["error"] and "balloon_only" not in rec["error"]
    assert out["n_runs_complete"] == 0 and out["n_runs_error"] == 1
    out = S.one_at_a_time(["L1"], [S.BASELINE], out_dir=tmp_path)          # resume retries it
    assert calls == [None, 0.05]                                           # T_equivalent of the failed run reused
    rec = json.loads(path.read_text())
    assert rec["complete"] and not rec.get("error") and rec["ivl"]["status"] == "ok"
    assert out["n_runs_complete"] == 1 and out["n_runs_error"] == 0
    S.one_at_a_time(["L1"], [S.BASELINE], out_dir=tmp_path)
    assert len(calls) == 2


def test_collect_oat_onset_difference_only_for_observed_onsets(tmp_path):
    """ivl_onset_load_minus_baseline only where both onsets are observed during inflation or pulse; a
    censored or after-peak value (p_top) is flagged instead."""
    sc = [s.name for s in S.oat_scenarios()]
    run_dir = tmp_path / S.OAT_RUNS
    run_dir.mkdir()
    ivl = {"observed": _prec(onset=0.4, phase="ivl_pulse"), "observed2": _prec(onset=0.5, phase="ivl_inflate"),
           "censored": _prec(p_top=0.7), "after_peak": _prec(onset=0.2, phase="ivl_unload", p_top=0.7),
           "unknown": _prec("failed", failed_phase="ivl_unload")}
    plan = {"L1": {sc[0]: "observed", sc[1]: "observed2", sc[2]: "censored", sc[3]: "after_peak", sc[4]: "unknown"},
            "L2": {sc[0]: "censored", sc[1]: "observed", sc[2]: "censored"}}
    for lid, by in plan.items():
        for name, kind in by.items():
            rec = {"lesion": lid, "scenario": name, "complete": True, "transient": {"T_equivalent": 0.05},
                   "ivl": ivl[kind], "balloon_only": _prec()}
            rec["outputs"] = S.outputs(rec)
            (run_dir / f"{S._oat_stem(lid, name)}.json").write_text(json.dumps(rec))
    runs = S.collect_oat(tmp_path)["runs"]
    r = runs["L1"]
    assert r[sc[1]]["ivl_onset_load_minus_baseline"] == pytest.approx(0.1)
    assert r[sc[1]]["ivl_onset_load_censored_vs_baseline"] is False
    for name in (sc[2], sc[3]):
        assert "ivl_onset_load_minus_baseline" not in r[name] and r[name]["ivl_onset_load_censored_vs_baseline"]
        assert r[name]["ivl_onset_load"] == 0.7                          # the value itself is kept (p_top)
    assert "ivl_onset_load_minus_baseline" not in r[sc[4]] and "ivl_onset_load_censored_vs_baseline" not in r[sc[4]]
    assert r[sc[4]]["ivl_fracture"] is None and "ivl_fracture_changed_vs_baseline" not in r[sc[4]]
    assert r[sc[4]]["ivl_failed_after_peak"] is True
    assert r[sc[3]]["T_equivalent_minus_baseline"] == 0.0               # other differences unchanged
    for name in (sc[1], sc[2]):                                         # censored baseline
        assert "ivl_onset_load_minus_baseline" not in runs["L2"][name]
        assert runs["L2"][name]["ivl_onset_load_censored_vs_baseline"] is True


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


def _ee_flag_counts(X, F, k, names, trajectories):
    """Reference count: elementary effects per factor with a flagged endpoint, trajectories all flagged."""
    per, n_all = dict.fromkeys(names, 0), 0
    for t in trajectories:
        T, Ft = X[t * (k + 1):(t + 1) * (k + 1)], F[t * (k + 1):(t + 1) * (k + 1)]
        for s in range(k):
            (c,) = np.nonzero(T[s + 1] != T[s])[0]
            per[names[c]] += bool(Ft[s] or Ft[s + 1])
        n_all += bool(Ft.all())
    return per, n_all


def test_analyze_censoring_counts_uncensored_and_load_step(tmp_path):
    """Censored onsets sit at p_top, which moves with the pulse scale: a lesion without any onset gets a
    nonzero mu_star for it; analyze counts the flagged elementary effects per input, the uncensored
    output drops trajectories with a censored or after-peak point, the median load step is reported and
    failed-after-peak / errored runs are counted."""
    design = S.morris_design()
    run_dir = tmp_path / S.MORRIS_RUNS
    run_dir.mkdir()
    (run_dir / S.MORRIS_DESIGN).write_text(json.dumps(design))
    X = np.asarray(design["X"])
    lo = np.array([b[0] for b in design["problem"]["bounds"]])
    hi = np.array([b[1] for b in design["problem"]["bounds"]])
    U = (X - lo) / (hi - lo)
    k, r = X.shape[1], design["r"]
    names = design["problem"]["names"]
    c_pulse = names.index("ivl_pulse_scale")
    observed, none, mixed, failing = design["lesions"]
    step = 0.02
    assert r >= 5
    # mixed lesion, per trajectory: 0, 1 censored where u0 > 0.5; 2 censored throughout; 3 one onset only
    # while unloading; the others observed throughout
    traj = np.arange(len(X)) // (k + 1)
    censored_rows = ((traj <= 1) & (U[:, 0] > 0.5)) | (traj == 2)
    after_peak_rows = np.zeros(len(X), bool)
    after_peak_rows[3 * (k + 1)] = True
    for lid in design["lesions"]:
        for i, u in enumerate(U):
            p_top = 1 + 0.5 * u[c_pulse]
            out = {"ivl_onset_load": 0.3 + 0.2 * u[0], "ivl_onset_censored": False, "ivl_onset_after_peak": False,
                   "ivl_onset_step": step, "ivl_fracture": False, "balloon_fracture": False,
                   "ivl_failed_after_peak": False, "ivl_status": "ok", "balloon_status": "ok"}
            if lid == none or (lid == mixed and censored_rows[i]):
                out.update(ivl_onset_load=p_top, ivl_onset_censored=True, ivl_onset_step=None)
            elif lid == mixed and after_peak_rows[i]:
                out.update(ivl_onset_load=p_top, ivl_onset_after_peak=True, ivl_onset_step=None)
            out["ivl_onset_load_uncensored"] = (out["ivl_onset_load"] if not (out["ivl_onset_censored"]
                                                or out["ivl_onset_after_peak"]) else None)
            rec = {"lesion": lid, "index": i, "complete": True, "outputs": out}
            if lid == failing and i == 0:                                # IVL failed after the peak, undecided
                out.update(ivl_onset_load=None, ivl_onset_load_uncensored=None, ivl_onset_censored=None,
                           ivl_onset_after_peak=None, ivl_onset_step=None, ivl_fracture=None,
                           ivl_failed_after_peak=True, ivl_status="failed")
            if lid == failing and i == k + 1:                            # protocol error: not complete
                rec = {"lesion": lid, "index": i, "error": "protocol error: ivl: RuntimeError: boom",
                       "outputs": out}
            (run_dir / f"{S._morris_stem(lid, i)}.json").write_text(json.dumps(rec))
    res = S.analyze(tmp_path)
    assert res["outputs"] == S.MORRIS_OUTPUTS and "ivl_onset_load_uncensored" in res["outputs"]
    assert res["n_runs_error"] == 1 and res["n_runs_complete"] == len(X) * len(design["lesions"]) - 1
    note = res["note"]
    assert "conditional" in note and f"r = {r}" in note and "unstable" in note and "p_top" in note
    LR = res["lesion_results"]
    zero = dict.fromkeys(names, 0)
    # every onset observed: the uncensored output is the onset load; no flagged elementary effects
    a = LR[observed]["outputs"]
    assert a["ivl_onset_load_uncensored"]["indices"] == a["ivl_onset_load"]["indices"]
    assert a["ivl_onset_load"]["median_ivl_onset_step"] == pytest.approx(step)
    assert a["ivl_onset_load"]["n_ivl_onset_step"] == len(X)
    for f in S.CENSORING_FLAGS["ivl_onset_load"] + (S.AT_P_TOP,):
        c = a["ivl_onset_load"]["censoring"][f]
        assert c["n_points"] == 0 and c["n_ee_with_flagged_endpoint"] == zero
        assert c["n_trajectories_all_points_flagged"] == 0
    # no onset at all: censored values at p_top give the pulse scale a mu_star although nothing happens
    b = LR[none]["outputs"]
    assert LR[none]["n_ivl_onset_censored"] == len(X)
    assert b["ivl_onset_load"]["indices"]["ivl_pulse_scale"]["mu_star"] == pytest.approx(0.5)
    c = b["ivl_onset_load"]["censoring"]["ivl_onset_censored"]
    assert c["n_points"] == len(X) and c["n_trajectories_all_points_flagged"] == r
    assert c["n_ee_with_flagged_endpoint"] == dict.fromkeys(names, r)
    assert b["ivl_onset_load_uncensored"]["indices"] is None and b["ivl_onset_load_uncensored"]["n_trajectories"] == 0
    assert b["ivl_onset_load"]["median_ivl_onset_step"] is None
    # mixed: counts per flag and either flag; uncensored drops every trajectory with a flagged point
    m = LR[mixed]["outputs"]
    assert 0 < censored_rows.sum() < len(X)
    any_rows = censored_rows | after_peak_rows
    for f, F in (("ivl_onset_censored", censored_rows), ("ivl_onset_after_peak", after_peak_rows),
                 (S.AT_P_TOP, any_rows)):
        c = m["ivl_onset_load"]["censoring"][f]
        per, n_all = _ee_flag_counts(X, F, k, names, range(r))
        assert c["n_points"] == F.sum() and c["n_ee_with_flagged_endpoint"] == per
        assert c["n_trajectories_all_points_flagged"] == n_all
    assert m["ivl_onset_load"]["censoring"]["ivl_onset_censored"]["n_trajectories_all_points_flagged"] == 1
    flagged_t = [t for t in range(r) if any_rows[t * (k + 1):(t + 1) * (k + 1)].any()]
    assert flagged_t == [0, 1, 2, 3]
    u = m["ivl_onset_load_uncensored"]
    assert u["excluded_trajectories"] == flagged_t and u["n_trajectories"] == r - len(flagged_t)
    assert u["indices"][names[0]]["mu_star"] == pytest.approx(0.2)
    assert u["indices"]["ivl_pulse_scale"]["mu_star"] == pytest.approx(0.0, abs=1e-12)
    assert m["ivl_onset_load"]["excluded_trajectories"] == []
    assert m["ivl_onset_load"]["median_ivl_onset_step"] == pytest.approx(step)
    assert m["ivl_onset_load"]["n_ivl_onset_step"] == (~any_rows).sum()
    # failed after the peak (undecided) and errored runs: missing values, counted
    d = LR[failing]
    assert d["n_ivl_failed_after_peak"] == 1 and d["n_complete"] == len(X) - 1
    for name in ("ivl_onset_load", "ivl_fracture"):
        assert d["outputs"][name]["excluded_trajectories"] == [0, 1]
    assert d["outputs"]["balloon_fracture"]["excluded_trajectories"] == [1]
    # pooled: the same counts summed over lesions
    pc = res["pooled"]["ivl_onset_load"]["censoring"][S.AT_P_TOP]
    assert pc["n_points"] == sum(LR[lid]["outputs"]["ivl_onset_load"]["censoring"][S.AT_P_TOP]["n_points"]
                                 for lid in design["lesions"])
    assert res["pooled"]["ivl_onset_load"]["median_ivl_onset_step"] == pytest.approx(step)


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
        br = rec[p]["onset_bracket"]                                    # same history and rule as onset_load
        assert br["load"] == rec[p]["cracks"]["onset_load"] and br["phase"] == rec[p]["cracks"]["onset_phase"]
        if br["load"] is not None:
            assert br["step"] == pytest.approx(br["load"] - br["lower"])
    if o["ivl_onset_censored"] is False and not o["ivl_onset_after_peak"]:
        assert o["ivl_onset_step"] > 0 and o["ivl_onset_load_lower"] < o["ivl_onset_load"]
    assert o["ivl_failed_after_peak"] is S.ivl_failed_after_peak(rec["ivl"])
    assert rec["inputs"]["ms"]["ca_E"] == base.ms.ca_E and rec["inputs"]["changed"] == []
    assert rec["inputs"]["ell_mm"] == pytest.approx(base.ell(H_CA))
    assert out["runs"]["TINY"][S.BASELINE]["ivl_fracture"] == o["ivl_fracture"]

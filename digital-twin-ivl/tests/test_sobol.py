"""Tests for the Sobol stage of the linear predictors (src/drivers/stage_sobol.py).

Only the last test runs the real predictor analysis (small annulus with a calcium sector, a tiny
design); everything else checks the inputs and their mapping from the config, the design, resume and
design-mismatch refusal, and the analysis on synthetic outputs (Ishigami function) with the model call
replaced.
"""
import ast
import json
import math
import os
from dataclasses import fields

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pytest

from dtivl import params as P
from dtivl.drivers import stage_sensitivity as S
from dtivl.drivers import stage_sobol as SB
from dtivl.drivers.stage_cohort_runs import PREDICTOR_KEYS
from dtivl.geometry.labels import LABEL_CALCIUM, LABEL_LUMEN, LABEL_OTHER, TAG_WALL
from dtivl.geometry.polar import PolarLabelMap
from dtivl.materials.constitutive import MaterialSet

# small test lesion (as tests/test_sensitivity.py): lumen A, outer B, calcium sector R1..R2 over ARC degrees
A, B, R1, R2, ARC = 0.8, 1.5, 0.9, 1.1, 180.0
H, H_CA = 0.2, 0.08
SMALL_N = 4                                   # base samples of the driver tests (power of 2)
ISHIGAMI_N = 512                              # base samples of the Ishigami test
ISHIGAMI_RESAMPLES = 100                      # bootstrap resamples there (speed; the intervals are not checked)
ISHIGAMI_A, ISHIGAMI_B = 7.0, 0.1
INCLUDED = {"log10_ca_E", "eps_f", "log10_k_perivascular", "k_over_mu_soft", "fibrous_E"}
EXCLUDED = {"ivl_pulse_scale", "ell_over_h", "rho_calcium"}


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
    info = {"n_nodes": None, "n_cells": S._n_cells(md), "n_cells_per_tag": {}}
    return sector_plm(), info, md


def _value_of(mi: S.ModelInputs, m: S.MorrisInput) -> float:
    """Sampled-variable value of input ``m`` read back from the model inputs (log10 for log inputs)."""
    v = getattr(mi.ms, m.input) if m.input in S._MS_FIELD_NAMES else getattr(mi, m.input)
    return math.log10(v) if m.scale == "log10" else float(v)


def _unit(mi: S.ModelInputs, inputs) -> np.ndarray:
    """Sample normalised to [0, 1] per input, from the model inputs."""
    return np.array([(_value_of(mi, m) - m.bounds[0]) / (m.bounds[1] - m.bounds[0]) for m in inputs])


# ---------------------------------------------------------------------------
# inputs and design from the config
# ---------------------------------------------------------------------------
def test_no_float_literals_in_function_bodies():
    """Numbers in the driver come from the config; only named module-level constants carry float literals."""
    tree = ast.parse(open(SB.__file__).read())
    bad = [(sub.lineno, sub.value) for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.ClassDef))
           for sub in ast.walk(node) if isinstance(sub, ast.Constant) and isinstance(sub.value, float)]
    assert not bad, f"float literals in function bodies: {bad}"


def test_inputs_are_the_linear_morris_inputs():
    use, excl = SB.sobol_inputs()
    morris = {m.name: m for m in S.morris_inputs()}
    assert {m.name for m in use} == INCLUDED and {e["name"] for e in excl} == EXCLUDED
    assert [m.name for m in use] == [n for n in morris if n in INCLUDED]          # Morris order
    for m in use:
        assert m == morris[m.name]                                                 # same range, scale, basis
    dec = P.raw("design.sensitivity.sobol.inputs")
    assert set(dec) == set(morris)
    for name, d in dec.items():
        assert d["basis"].startswith("DESIGN") and d["status"] == "DESIGN"
    assert all(e["basis"] == dec[e["name"]]["basis"] for e in excl)
    # a Morris input without a decision is refused
    extra = S.MorrisInput("new_input", "ca_nu", (0.1, 0.2), "linear", "test")
    orig = S.morris_inputs
    try:
        S.morris_inputs = lambda: orig() + [extra]
        with pytest.raises(ValueError):
            SB.sobol_inputs()
    finally:
        S.morris_inputs = orig


def test_config_seed_budget_and_docstring():
    cfg = SB.sobol_config()
    node = P.raw("design.sensitivity.sobol")
    assert node["status"] == "DESIGN" and node["basis"]
    assert cfg["lesions"] == [str(x) for x in P.raw("design.sensitivity.morris_lesions")]
    assert cfg["seed"] == int(P.raw("design.seed")) + int(node["seed_offset"]["value"])
    assert cfg["seed"] != S.morris_config()["seed"]
    n, k = cfg["n_base"], len(SB.sobol_inputs()[0])
    assert n & (n - 1) == 0
    assert SB.n_runs_per_lesion() == n * (k + 2) and SB.n_runs() == n * (k + 2) * len(cfg["lesions"])
    # N is the largest power of 2 whose analyses (baselines included) fit the one-core budget
    cost = SB.planned_cost()
    n_les, s = len(cfg["lesions"]), cfg["planning_seconds_per_analysis"]
    budget_s = cfg["budget_core_hours"] * SB.SECONDS_PER_HOUR
    assert cost["n_analyses"] == SB.n_runs() + n_les and cost["within_budget"] and cost["n_base_max"] == n
    assert cost["n_analyses"] * s <= budget_s < (2 * n * (k + 2) + 1) * n_les * s
    assert cost["core_hours"] == pytest.approx(cost["n_analyses"] * s / SB.SECONDS_PER_HOUR)
    over = SB.planned_cost(2 * n)
    assert not over["within_budget"] and over["n_base_max"] == n and over["n_analyses"] > cost["n_analyses"]
    for v in (SB.n_runs_per_lesion(), SB.n_runs(), n):
        assert str(v) in SB.__doc__                                                # counts stated in the docstring
    lo, hi = cfg["tau_range_pct"]
    assert 0 <= lo < hi <= 100 and 0 < cfg["conf_level"] < 1 and cfg["conf_resamples"] > 0
    # every Sobol design choice is a DESIGN row of model_assumptions.csv
    import csv

    with open(P.ASSUMPTIONS_CSV, newline="") as f:
        rows = {r["assumption"]: r for r in csv.DictReader(f)}
    for key in ("n_base", "scramble", "seed_offset", "conf_resamples", "conf_level", "tau_range_pct",
                "budget_core_hours", "planning_seconds_per_analysis"):
        assert rows[f"design.sensitivity.sobol.{key}"]["status"] == "DESIGN"
    for name in INCLUDED | EXCLUDED:
        assert rows[f"design.sensitivity.sobol.inputs.{name}"]["status"] == "DESIGN"


def test_design_reproducible_and_structured():
    cfg = SB.sobol_config()
    use = SB.sobol_inputs()[0]
    k = len(use)
    X1, X2 = SB.sobol_samples(SMALL_N), SB.sobol_samples(SMALL_N)
    assert X1.shape == (SMALL_N * (k + 2), k) and np.array_equal(X1, X2)
    assert not np.array_equal(X1, SB.sobol_samples(SMALL_N, seed=cfg["seed"] + 1))
    assert np.array_equal(SB.sobol_samples(), SB.sobol_samples(cfg["n_base"], cfg["seed"]))
    for c, m in enumerate(use):
        assert np.all(X1[:, c] >= m.bounds[0]) and np.all(X1[:, c] <= m.bounds[1])
    for g in range(SMALL_N):                       # groups: A, AB_1..AB_k (A with column i of B), B
        G = X1[g * (k + 2):(g + 1) * (k + 2)]
        a, b = G[0], G[-1]
        for i in range(k):
            want = a.copy()
            want[i] = b[i]
            assert np.array_equal(G[1 + i], want)
    with pytest.raises(ValueError):
        SB.sobol_samples(SMALL_N + 1)
    d = SB.sobol_design(SMALL_N)
    assert d["X"] == X1.tolist() and d["n_runs_per_lesion"] == len(X1) and d["rows_per_group"] == k + 2
    assert d["seed"] == cfg["seed"] and d["calc_second_order"] is False
    assert d["baseline"]["k_found"] == P.get("loads.perivascular_k_MPa_per_mm").value
    json.dumps(d)
    h = SB.design_hash(d)
    assert h == SB.design_hash(SB.sobol_design(SMALL_N))
    assert h != SB.design_hash(SB.sobol_design(2 * SMALL_N))
    assert h != SB.design_hash(SB.sobol_design(SMALL_N, seed=cfg["seed"] + 1))
    assert h == SB.design_hash({**d, "basis": "edited wording"})                 # texts do not enter the hash


def test_input_mapping_reaches_material_set_and_k_found(monkeypatch):
    use = SB.sobol_inputs()[0]
    base = S.ModelInputs.baseline()
    x = np.array([m.bounds[0] + 0.3 * (m.bounds[1] - m.bounds[0]) for m in use])
    mi = SB.sobol_point(x)
    v = {m.name: m.value(xv) for m, xv in zip(use, x)}
    assert mi.ms.ca_E == pytest.approx(v["log10_ca_E"]) and mi.ms.eps_f == v["eps_f"]
    assert mi.k_found == pytest.approx(v["log10_k_perivascular"])
    assert mi.ms.k_over_mu == v["k_over_mu_soft"] and mi.ms.fibrous_E == v["fibrous_E"]
    for f in ("ell_over_h", "pulse_scale", "rho_calcium", "driving_force"):        # excluded inputs stay baseline
        assert getattr(mi, f) == getattr(base, f)
    for f in ("wall_c10", "lipid_E", "ca_nu", "sigma_c", "failure_mode", "crack_filler"):
        assert getattr(mi.ms, f) == getattr(MaterialSet.baseline(), f)
    with pytest.raises(ValueError):
        SB.sobol_point(x[:-1])
    # the model call passes the sample's MaterialSet and k_found to predictor_analysis
    from dtivl.metrics import predictors as PR

    seen = []

    def fake_pa(md, plm, ms=None, *, k_found=None, **kw):
        seen.append((ms, k_found))
        return {k: (None if k == "scf_edge_nl" else 1.0) for k in PREDICTOR_KEYS}

    monkeypatch.setattr(PR, "predictor_analysis", fake_pa)
    y = SB.predictor_vector("md", "plm", mi)
    assert seen[-1][0] == mi.ms and seen[-1][1] == mi.k_found
    assert y.shape == (len(PREDICTOR_KEYS),) and np.isnan(y[PREDICTOR_KEYS.index("scf_edge_nl")])
    # each input on its own reaches exactly its MaterialSet field or k_found
    ref = seen[-1]
    ms_names = [f.name for f in fields(MaterialSet) if f.name != "provenance"]
    for c, m in enumerate(use):
        x2 = x.copy()
        x2[c] = m.bounds[1]
        SB.predictor_vector("md", "plm", SB.sobol_point(x2))
        ms2, k2 = seen[-1]
        changed = {f for f in ms_names if getattr(ms2, f) != getattr(ref[0], f)}
        changed |= {"k_found"} if k2 != ref[1] else set()
        assert changed == {m.input}, m.name


# ---------------------------------------------------------------------------
# driver with a stub model: resume, refusal of a changed design, failures
# ---------------------------------------------------------------------------
@pytest.fixture
def stub(monkeypatch):
    calls = []
    use = SB.sobol_inputs()[0]

    def fake_eval(md, plm, inputs):
        calls.append(inputs)
        u = _unit(inputs, use)
        return {k: float(md["j"] + u[c % len(u)]) for c, k in enumerate(PREDICTOR_KEYS)}

    monkeypatch.setattr(SB, "evaluate", fake_eval)
    monkeypatch.setattr(SB, "lesion_mesh", lambda lid: (None, {"n_cells": 1}, {"j": int(lid[1:])}))
    return calls


def test_resume_and_design_mismatch(stub, monkeypatch, tmp_path):
    k = len(SB.sobol_inputs()[0])
    n_rows = SMALL_N * (k + 2)
    path = tmp_path / SB.SOBOL_RUNS / "L1.npz"
    assert SB.run(["L1"], out_dir=tmp_path, n_base=SMALL_N, max_runs=5) == {"L1": 5}
    rec = SB.load_runs(path)
    assert rec["base_status"] == SB.STATUS_OK and list(rec["status"][:4]) == [SB.STATUS_OK] * 4
    assert np.all(rec["status"][4:] == SB.STATUS_PENDING) and np.all(np.isnan(rec["Y"][4:]))
    assert np.array_equal(rec["X"], SB.sobol_samples(SMALL_N)) and rec["keys"] == list(PREDICTOR_KEYS)
    assert rec["design_hash"] == SB.design_hash(SB.sobol_design(SMALL_N)) and rec["mesh"]["n_cells"] == 1
    base_u = _unit(S.ModelInputs.baseline(), SB.sobol_inputs()[0])
    assert rec["y_base"][0] == pytest.approx(1 + base_u[0])
    # resume: continues with row 4; nothing already done is re-run
    assert SB.run(["L1"], out_dir=tmp_path, n_base=SMALL_N, max_runs=3) == {"L1": 3}
    assert len(stub) == 8
    X = SB.sobol_samples(SMALL_N)
    for i, inp in enumerate(stub[5:8]):
        assert np.allclose(_unit(inp, SB.sobol_inputs()[0]), _unit(SB.sobol_point(X[4 + i]), SB.sobol_inputs()[0]))
    # a failing row is recorded and retried by the next run
    real = SB.evaluate

    def flaky(md, plm, inputs):
        if len(stub) == 9:
            stub.append(inputs)
            raise RuntimeError("solver failed")
        return real(md, plm, inputs)

    monkeypatch.setattr(SB, "evaluate", flaky)
    SB.run(["L1"], out_dir=tmp_path, n_base=SMALL_N, max_runs=2)
    rec = SB.load_runs(path)
    assert rec["status"][8] == SB.STATUS_ERROR and "solver failed" in rec["errors"][8] and np.all(np.isnan(rec["Y"][8]))
    assert not (tmp_path / SB.SOBOL_JSON).exists()
    out = SB.run(["L1"], out_dir=tmp_path, n_base=SMALL_N)                    # completes, retries row 8, analyses
    assert out == {"L1": n_rows - 9 + 1}
    rec = SB.load_runs(path)
    assert np.all(rec["status"] == SB.STATUS_OK) and rec["errors"] == {} and np.all(np.isfinite(rec["Y"]))
    summary = json.loads((tmp_path / SB.SOBOL_JSON).read_text())
    assert summary["design_matches_config"] is False and summary["n_base"] == SMALL_N   # N differs from the config
    n_calls = len(stub)
    assert SB.run(["L1"], out_dir=tmp_path, n_base=SMALL_N) == {"L1": 0} and len(stub) == n_calls
    # another design is refused before anything runs: N, seed, a stored hash, a changed mesh
    with pytest.raises(ValueError):
        SB.run(["L1"], out_dir=tmp_path, n_base=2 * SMALL_N)
    cfg = SB.sobol_config()
    monkeypatch.setattr(SB, "sobol_config", lambda: {**cfg, "seed": cfg["seed"] + 1})
    with pytest.raises(ValueError):
        SB.run(["L1"], out_dir=tmp_path, n_base=SMALL_N)
    monkeypatch.setattr(SB, "sobol_config", lambda: cfg)
    rec["design_hash"] = "0" * 64
    SB._save_runs(path, rec)
    with pytest.raises(ValueError):
        SB.run(["L1"], out_dir=tmp_path, n_base=SMALL_N)
    assert len(stub) == n_calls
    rec["design_hash"] = SB.design_hash(SB.sobol_design(SMALL_N))
    rec["status"][-1] = SB.STATUS_PENDING
    rec["mesh"]["n_cells"] = 2
    SB._save_runs(path, rec)
    with pytest.raises(ValueError):
        SB.run(["L1"], out_dir=tmp_path, n_base=SMALL_N)
    assert len(stub) == n_calls


def test_blocks_interleave_lesions_and_skip_locked(stub, tmp_path):
    k = len(SB.sobol_inputs()[0])
    step = k + 2
    n = 2 * SB.BLOCK_GROUPS
    SB.run(["L1", "L2"], out_dir=tmp_path, n_base=n, max_runs=2 * (1 + SB.BLOCK_GROUPS * step))
    for lid in ("L1", "L2"):                       # first block of every lesion before the second block of any
        st = SB.load_runs(tmp_path / SB.SOBOL_RUNS / f"{lid}.npz")["status"]
        assert np.all(st[:SB.BLOCK_GROUPS * step] == SB.STATUS_OK) and np.all(st[SB.BLOCK_GROUPS * step:] == 0)
    lock = tmp_path / SB.SOBOL_RUNS / "locks" / "L1.lock"
    assert S._claim(lock)                          # held by this (live) process: L1 is left to its holder
    try:
        assert SB.run(["L1", "L2"], out_dir=tmp_path, n_base=n) == {"L1": 0, "L2": SB.BLOCK_GROUPS * step}
    finally:
        S._release(lock)
    assert not (tmp_path / SB.SOBOL_JSON).exists()                             # L1 incomplete: no analysis


# ---------------------------------------------------------------------------
# analysis on a synthetic model (Ishigami) with known indices
# ---------------------------------------------------------------------------
def test_kendall_tau_b_matches_scipy():
    from scipy.stats import kendalltau

    rng = np.random.default_rng(0)
    for n in (2, 3, 4, 6):
        b = rng.integers(0, 3, n).astype(float)                                # with ties
        Y = np.concatenate([rng.integers(0, 3, (n, 20)), rng.normal(size=(n, 20)), np.ones((n, 1))], axis=1)
        t = SB.kendall_tau_b(b, Y)
        for c in range(Y.shape[1]):
            if np.ptp(b) == 0 or np.ptp(Y[:, c]) == 0:
                assert np.isnan(t[c])
            else:
                assert t[c] == pytest.approx(kendalltau(b, Y[:, c]).statistic, abs=1e-12)
    b = np.array([3.0, 1.0, 2.0])
    assert np.array_equal(SB.kendall_tau_b(b, np.stack([b, -b, 2 * b + 1], axis=1)), [1.0, -1.0, 1.0])


def _ishigami_reference(a=ISHIGAMI_A, b=ISHIGAMI_B):
    pi = math.pi
    v1 = 0.5 * (1 + b * pi ** 4 / 5) ** 2
    v2 = a ** 2 / 8
    v13 = 8 * b ** 2 * pi ** 8 / 225
    v = v1 + v2 + v13
    return {"mean": a / 2, "variance": v, "S1": [v1 / v, v2 / v, 0.0], "ST": [(v1 + v13) / v, v2 / v, v13 / v]}


def test_analyze_ishigami(monkeypatch, tmp_path):
    use = SB.sobol_inputs()[0]
    names = [m.name for m in use]
    k = len(use)

    def fake_eval(md, plm, inputs):
        u = _unit(inputs, use)
        z = -math.pi + 2 * math.pi * u[:3]
        j = md["j"]
        out = {key: 1.0 for key in PREDICTOR_KEYS}                             # constant: no indices
        out["sigma1_p95"] = math.sin(z[0]) + ISHIGAMI_A * math.sin(z[1]) ** 2 + ISHIGAMI_B * z[2] ** 4 * math.sin(z[0])
        out["sigma1_mean"] = 10.0 * j + u[3]                                   # lesion order never changes
        out["sed_mean"] = j * (u[0] - 0.25)                                    # order reverses for u0 < 0.25
        out["frac_eps_gt_epsf"] = 0.0
        out["scf_edge_nl"] = None if u[4] > 0.9 else u[4]                      # missing values exclude groups
        return out

    monkeypatch.setattr(SB, "evaluate", fake_eval)
    monkeypatch.setattr(SB, "lesion_mesh", lambda lid: (None, {"n_cells": 1}, {"j": int(lid[1:])}))
    cfg = SB.sobol_config()
    monkeypatch.setattr(SB, "sobol_config", lambda: {**cfg, "conf_resamples": ISHIGAMI_RESAMPLES})
    # cohort-run records: L1 with the baseline predictors (sigma1_mean 1 % off), L2 without predictors
    cohort_dir = tmp_path / SB.COHORT_RUNS
    cohort_dir.mkdir()
    pred = fake_eval({"j": 1}, None, S.ModelInputs.baseline())
    pred["sigma1_mean"] *= 1.01
    (cohort_dir / "L1.json").write_text(json.dumps({"lesion": "L1", "predictors": pred}))
    (cohort_dir / "L2.json").write_text(json.dumps({"lesion": "L2"}))
    SB.run(["L1", "L2", "L3"], out_dir=tmp_path, n_base=ISHIGAMI_N)              # complete: analysed
    assert json.loads((tmp_path / SB.SOBOL_JSON).read_text())["lesions"] == ["L1", "L2", "L3"]
    n_partial = 3                                                              # L4: baseline + 3 groups only
    SB.run(["L4"], out_dir=tmp_path, n_base=ISHIGAMI_N, max_runs=1 + n_partial * (k + 2))
    res = json.loads(json.dumps(SB.analyze(tmp_path)))
    assert res == json.loads((tmp_path / SB.SOBOL_JSON).read_text())
    assert res["lesions"] == ["L1", "L2", "L3", "L4"] and res["lesions_missing"] == SB.sobol_config()["lesions"]
    assert res["design_matches_config"] is (ISHIGAMI_N == cfg["n_base"])
    assert res["conf_resamples"] == ISHIGAMI_RESAMPLES
    assert res["n_runs_complete"] == 3 * ISHIGAMI_N * (k + 2) + n_partial * (k + 2)
    assert res["n_runs_planned"] == ISHIGAMI_N * (k + 2) * len(cfg["lesions"])
    assert res["planned_cost"] == SB.planned_cost(ISHIGAMI_N)
    assert res["core_hours_projected"] == pytest.approx(
        res["seconds_per_run_median"] * (ISHIGAMI_N * (k + 2) + 1) * len(cfg["lesions"]) / SB.SECONDS_PER_HOUR)
    cc = res["lesion_results"]["L1"]["baseline_vs_cohort_run"]
    assert cc["max_rel_diff_predictor"] == "sigma1_mean" and cc["max_rel_diff"] == pytest.approx(0.01 / 1.01)
    assert cc["n_compared"] == sum(v is not None for v in pred.values()) and cc["cohort_run"] == "cohort_runs/L1.json"
    assert all(res["lesion_results"][lid]["baseline_vs_cohort_run"] is None for lid in ("L2", "L3", "L4"))
    ref = _ishigami_reference()
    for lid in ("L1", "L2", "L3"):
        lr = res["lesion_results"][lid]
        assert lr["n_complete"] == ISHIGAMI_N * (k + 2) and lr["baseline_complete"] and lr["n_pending"] == 0
        ish = lr["predictors"]["sigma1_p95"]
        assert ish["n_groups"] == ISHIGAMI_N and ish["n_base_samples"] == 2 * ISHIGAMI_N
        assert ish["mean"] == pytest.approx(ref["mean"], abs=0.2)
        assert ish["variance"] == pytest.approx(ref["variance"], rel=0.05)
        for c in range(3):
            idx = ish["indices"][names[c]]
            assert idx["S1"] == pytest.approx(ref["S1"][c], abs=0.06), (lid, c)
            assert idx["ST"] == pytest.approx(ref["ST"][c], abs=0.06), (lid, c)
            assert idx["S1_conf"] > 0 and idx["ST_conf"] > 0
        for c in (3, 4):                                                       # inputs without effect
            assert abs(ish["indices"][names[c]]["S1"]) < 0.05 and ish["indices"][names[c]]["ST"] < 0.02
        assert ish["S1_sum"] == pytest.approx(sum(ref["S1"]), abs=0.1)
        lin = lr["predictors"]["sigma1_mean"]
        assert lin["indices"][names[3]]["S1"] == pytest.approx(1.0, abs=0.02)
        assert lin["indices"][names[3]]["ST"] == pytest.approx(1.0, abs=0.02)
        assert lin["variance"] == pytest.approx(1 / 12, rel=0.05)
        const = lr["predictors"]["frac_eps_gt_epsf"]
        assert const["indices"] is None and "constant" in const["note"] and const["variance"] == 0.0
        nan = lr["predictors"]["scf_edge_nl"]
        X = SB.sobol_samples(ISHIGAMI_N)
        u4 = (X[:, 4] - use[4].bounds[0]) / (use[4].bounds[1] - use[4].bounds[0])
        n_bad = int((u4 > 0.9).reshape(-1, k + 2).any(axis=1).sum())
        assert nan["n_groups_excluded"] == n_bad > 0 and nan["n_groups"] == ISHIGAMI_N - n_bad
    part = res["lesion_results"]["L4"]
    assert part["n_complete"] == n_partial * (k + 2) and part["predictors"]["sigma1_p95"]["n_groups"] == n_partial
    acr = res["across_lesions"]["sigma1_p95"][names[1]]
    assert acr["n_lesions"] == 4 and acr["ST_min"] <= acr["ST_max"]
    assert res["across_lesions"]["frac_eps_gt_epsf"] == {}
    # ranking stability: all four lesions on the rows run for every lesion (the partial lesion's rows)
    rs = res["ranking_stability"]["sigma1_mean"]
    assert rs["lesions"] == ["L1", "L2", "L3", "L4"] and rs["n_pairs"] == 6
    assert rs["n_samples"] == n_partial * (k + 2) and rs["tau_median"] == 1.0 and rs["fraction_order_unchanged"] == 1.0
    assert rs["baseline_order"] == ["L4", "L3", "L2", "L1"]
    rs = res["ranking_stability"]["sed_mean"]
    X = SB.sobol_samples(ISHIGAMI_N)[:n_partial * (k + 2)]
    u0 = (X[:, 0] - use[0].bounds[0]) / (use[0].bounds[1] - use[0].bounds[0])
    base_u0 = _unit(S.ModelInputs.baseline(), use)[0]
    assert base_u0 > 0.25                                                      # baseline order: L4 first
    assert rs["fraction_order_unchanged"] == pytest.approx(float(np.mean(u0 > 0.25)))
    assert rs["tau_min"] == (-1.0 if np.any(u0 < 0.25) else 1.0)
    assert rs["tau_median"] == float(np.median(np.where(u0 > 0.25, 1.0, -1.0)))
    assert res["ranking_stability"]["frac_eps_gt_epsf"]["note"].startswith("baseline values equal")


# ---------------------------------------------------------------------------
# one small run of the real predictor analysis
# ---------------------------------------------------------------------------
def test_smoke_tiny_annulus(monkeypatch, annulus, tmp_path):
    """Two base samples on a small annulus with a calcium sector: the real linear solve per row, files,
    baseline equal to the cohort predictor analysis, and the analysis summary."""
    from dtivl.metrics.predictors import predictor_analysis

    plm, info, md = annulus
    monkeypatch.setattr(SB, "lesion_mesh", lambda lid: annulus)
    n = 2
    k = len(SB.sobol_inputs()[0])
    assert SB.run(["TINY"], out_dir=tmp_path, n_base=n) == {"TINY": 1 + n * (k + 2)}
    rec = SB.load_runs(tmp_path / SB.SOBOL_RUNS / "TINY.npz")
    assert np.all(rec["status"] == SB.STATUS_OK) and rec["errors"] == {} and np.all(rec["seconds"] > 0)
    base = predictor_analysis(md, plm, MaterialSet.baseline())
    assert rec["y_base"] == pytest.approx([base[key] for key in PREDICTOR_KEYS], nan_ok=True)
    i = 3
    mi = SB.sobol_point(rec["X"][i])
    row = predictor_analysis(md, plm, mi.ms, k_found=mi.k_found)
    assert rec["Y"][i] == pytest.approx([row[key] for key in PREDICTOR_KEYS], nan_ok=True)
    s1 = rec["Y"][:, PREDICTOR_KEYS.index("sigma1_mean")]
    assert np.all(s1 > 0) and np.ptp(s1) > 0                                   # the inputs reach the solve
    res = json.loads((tmp_path / SB.SOBOL_JSON).read_text())                   # written by run() on completion
    lr = res["lesion_results"]["TINY"]
    assert lr["mesh"]["n_cells"] == info["n_cells"] and lr["n_complete"] == n * (k + 2)
    assert lr["predictors"]["sigma1_mean"]["indices"] is not None
    assert set(lr["predictors"]) == set(PREDICTOR_KEYS) and res["seconds_per_run_median"] > 0
    assert res["ranking_stability"]["sigma1_mean"]["note"].startswith("fewer than 2 lesions")

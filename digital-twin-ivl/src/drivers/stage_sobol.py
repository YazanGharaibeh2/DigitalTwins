"""Stage: Sobol indices of the pre-treatment linear predictors (study plan Phase 4).

Model (:func:`evaluate`)
------------------------
The cohort predictor analysis :func:`dtivl.metrics.predictors.predictor_analysis`
(linear-elastic plane-strain solve at the lumen pressure
``loads.ivl_balloon_pressure_atm``) with the sample's
:class:`~dtivl.materials.constitutive.MaterialSet` and perivascular foundation
``k_found``; the outputs are the predictors of
:data:`dtivl.drivers.stage_cohort_runs.PREDICTOR_KEYS`. No outcome-model input
(``T_equivalent``, IVL load) enters; nothing of the nonlinear damage runs is
computed here.

Inputs (:func:`sobol_inputs`)
-----------------------------
The Morris inputs of :func:`dtivl.drivers.stage_sensitivity.morris_inputs`
(same names, ranges, scales and bases) that can change a linear predictor,
selected per input in ``design.sensitivity.sobol.inputs`` (each decision with
its basis): log10 calcium modulus, failure strain ``eps_f`` (threshold
predictors only), log10 perivascular foundation, soft-tissue ``K/mu``,
fibrous modulus. Excluded: IVL pulse scale (``run_ivl`` only),
``ell_over_h`` (nonlinear model; the nonlocal predictors average over the
fixed nominal length) and the calcium density (transient ``T_equivalent``
only). A sample row is mapped to the model inputs exactly as a Morris row
(:func:`dtivl.drivers.stage_sensitivity.morris_point`: baseline
:class:`~dtivl.drivers.stage_sensitivity.ModelInputs` with every sampled
input set). The inputs are independent and uniform on their ranges (log10
scale for the modulus and the foundation).

Design (:func:`sobol_design`)
-----------------------------
SALib 1.6 ``SALib.sample.sobol.sample`` (Saltelli cross-sampling of a
scrambled Sobol sequence, ``calc_second_order=False``) with
``N = design.sensitivity.sobol.n_base`` (a power of 2) and seed
``design.seed + design.sensitivity.sobol.seed_offset``: ``N`` groups of
``k + 2`` rows (``A``, ``AB_1`` ... ``AB_k``, ``B``). The same sample matrix
is used for every lesion. Lesions: ``design.sensitivity.morris_lesions``.
``N`` is the largest power of 2 for which the lesions fit
``budget_core_hours`` on one core at ``planning_seconds_per_analysis``
(:func:`planned_cost`). With the configuration at the time of writing
(N = 512, k = 5 inputs): 512 x 7 = 3584 predictor analyses per lesion plus
one baseline analysis, 4 lesions: 14336 analyses (:func:`n_runs_per_lesion`,
:func:`n_runs`), just under 2 h at 0.5 s per analysis (N = 1024 would double
it).

Runs and resume (:func:`run`)
-----------------------------
Each lesion is meshed once per process (private mesh file, as
:func:`dtivl.drivers.stage_sensitivity.lesion_mesh`) and the mesh is reused
for every sample. The outputs are stored incrementally in
``results/sensitivity_sobol_runs/<lesion>.npz`` (sample matrix, outputs,
per-row status and wall time, baseline outputs, mesh size and a hash of the
design, :func:`design_hash`); a stored file whose design hash or mesh size
differs from the current one is refused (no mixing of designs). The rows are
run in blocks of :data:`BLOCK_GROUPS` groups, every lesion per block, so
that partial results hold complete groups for all lesions; the file is
written after every block. A lesion is claimed with a lock file per block
(:func:`dtivl.drivers.stage_sensitivity._claim`), so several worker
processes can run concurrently. A failed row is recorded (status, message)
and retried by the next :func:`run`.

Analysis (:func:`analyze`)
--------------------------
``results/sensitivity_sobol.json``: per lesion and predictor ``S1``,
``S1_conf``, ``ST``, ``ST_conf`` per input (SALib.analyze.sobol, bootstrap
``conf_resamples`` at ``conf_level``), and the mean and variance of the
predictor over the base samples; only groups with all ``k + 2`` values
present enter (partial data and missing lesions are allowed; the counts are
reported). A predictor constant over the samples has no indices. Per
predictor, the cross-lesion ranking stability: Kendall tau-b between the
lesions' baseline values and their values at each sample (median and the
``tau_range_pct`` percentiles, fraction of samples with the baseline order
unchanged). Per lesion, the baseline outputs against the predictors of the
lesion's cohort run (``results/cohort_runs/<lesion>.json``, where it exists;
:func:`cohort_consistency`), and overall the planned cost, the measured median
time per analysis and the one-core time it projects for the whole design.
Command line: ``python -m dtivl.drivers.stage_sobol {run,analyze}
[--lesions ...] [--max-runs n]``.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

import numpy as np

from dtivl import params as P
from dtivl.drivers import stage_sensitivity as S
from dtivl.drivers.common import RESULTS_DIR, Timer, _to_jsonable
from dtivl.drivers.stage_cohort_runs import PREDICTOR_KEYS

log = logging.getLogger(__name__)

SOBOL_RUNS = "sensitivity_sobol_runs"
SOBOL_JSON = "sensitivity_sobol.json"
#: cohort-run records (under the same results directory) compared with the baseline outputs
COHORT_RUNS = "cohort_runs"
CONFIG_KEY = "design.sensitivity.sobol"
METHOD = ("Sobol indices (SALib 1.6: SALib.sample.sobol Saltelli cross-sampling of a scrambled Sobol sequence, "
          "SALib.analyze.sobol; calc_second_order false)")

#: Saltelli groups (A, AB_1..AB_k, B rows of one base sample) per lesion between two file writes
BLOCK_GROUPS = 8
#: per-row status in the run files
STATUS_PENDING, STATUS_OK, STATUS_ERROR = 0, 1, -1
#: fewest complete groups for which indices are computed
MIN_GROUPS = 2
#: a Kendall tau within this of 1 counts as "baseline order unchanged"
TAU_ONE_ATOL = 1e-12
#: a predictor whose range over the samples is at most this times its largest magnitude is constant
CONST_RTOL = 1e-12
SECONDS_PER_HOUR = 3600.0


# ---------------------------------------------------------------------------
# Configuration, inputs, design
# ---------------------------------------------------------------------------
def _value(key: str):
    return P.raw(f"{CONFIG_KEY}.{key}")["value"]


def sobol_config() -> dict:
    """``design.sensitivity.sobol`` resolved: lesions (the Morris lesions), ``n_base`` (checked to be a
    power of 2), ``scramble``, ``seed`` (``design.seed + seed_offset``), bootstrap settings, the
    Kendall-tau percentiles and the planning budget (:func:`planned_cost`)."""
    n = int(_value("n_base"))
    if n < 1 or n & (n - 1):
        raise ValueError(f"{CONFIG_KEY}.n_base = {n} is not a power of 2")
    lo, hi = (float(v) for v in _value("tau_range_pct"))
    return {"lesions": S.morris_config()["lesions"], "n_base": n, "scramble": bool(_value("scramble")),
            "seed": int(P.raw("design.seed")) + int(_value("seed_offset")),
            "conf_resamples": int(_value("conf_resamples")), "conf_level": float(_value("conf_level")),
            "tau_range_pct": (lo, hi), "budget_core_hours": float(_value("budget_core_hours")),
            "planning_seconds_per_analysis": float(_value("planning_seconds_per_analysis")),
            "basis": P.raw(CONFIG_KEY)["basis"]}


def sobol_inputs() -> tuple[list[S.MorrisInput], list[dict]]:
    """(Sobol inputs, excluded inputs) from the Morris inputs and the per-input decisions of
    ``design.sensitivity.sobol.inputs``. Every Morris input needs a decision; a decision for an
    unknown input is an error."""
    dec = P.raw(f"{CONFIG_KEY}.inputs")
    morris = S.morris_inputs()
    names = [m.name for m in morris]
    missing, unknown = [n for n in names if n not in dec], [n for n in dec if n not in names]
    if missing or unknown:
        raise ValueError(f"{CONFIG_KEY}.inputs: no decision for Morris inputs {missing}; not Morris inputs {unknown}")
    use = [m for m in morris if bool(dec[m.name]["value"])]
    excluded = [{"name": m.name, "input": m.input, "basis": dec[m.name]["basis"]} for m in morris
                if not bool(dec[m.name]["value"])]
    if not use:
        raise ValueError(f"{CONFIG_KEY}.inputs selects no input")
    return use, excluded


def sobol_problem(inputs: list[S.MorrisInput] | None = None) -> dict:
    """SALib problem definition."""
    inputs = sobol_inputs()[0] if inputs is None else inputs
    return {"num_vars": len(inputs), "names": [m.name for m in inputs], "bounds": [list(m.bounds) for m in inputs]}


def sobol_samples(n_base: int | None = None, seed: int | None = None) -> np.ndarray:
    """Saltelli sample matrix (``N (k + 2)`` rows, one column per :func:`sobol_inputs` input),
    deterministic for a given seed (defaults from :func:`sobol_config`)."""
    from SALib.sample import sobol as SS

    cfg = sobol_config()
    n = cfg["n_base"] if n_base is None else int(n_base)
    seed = cfg["seed"] if seed is None else int(seed)
    if n < 1 or n & (n - 1):
        raise ValueError(f"n_base = {n} is not a power of 2")
    return SS.sample(sobol_problem(), n, calc_second_order=False, scramble=cfg["scramble"], seed=seed)


def sobol_point(x, inputs: list[S.MorrisInput] | None = None,
                base: S.ModelInputs | None = None) -> S.ModelInputs:
    """Model inputs of one sample row (the Morris mapping: baseline with every sampled input set)."""
    inputs = sobol_inputs()[0] if inputs is None else inputs
    return S.morris_point(x, inputs, base)


def n_runs_per_lesion() -> int:
    """Predictor analyses per lesion: N (k + 2) (the baseline analysis not counted)."""
    return sobol_config()["n_base"] * (len(sobol_inputs()[0]) + 2)


def n_runs() -> int:
    """Predictor analyses over the Morris lesions (baseline analyses not counted)."""
    return n_runs_per_lesion() * len(sobol_config()["lesions"])


def planned_cost(n_base: int | None = None) -> dict:
    """Planned one-core cost over the Morris lesions: ``N (k + 2) + 1`` analyses per lesion (the baseline
    included) at ``planning_seconds_per_analysis``, against ``budget_core_hours``; ``n_base_max`` = the
    largest power of 2 within the budget (None if not even N = 1 fits). ``n_base`` overrides the config N."""
    cfg = sobol_config()
    n = cfg["n_base"] if n_base is None else int(n_base)
    k, n_les = len(sobol_inputs()[0]), len(cfg["lesions"])
    s, budget_s = cfg["planning_seconds_per_analysis"], cfg["budget_core_hours"] * SECONDS_PER_HOUR

    def analyses(m: int) -> int:
        return n_les * (m * (k + 2) + 1)

    n_max = None
    if analyses(1) * s <= budget_s:
        n_max = 1
        while analyses(2 * n_max) * s <= budget_s:
            n_max *= 2
    return {"n_base": n, "n_lesions": n_les, "n_analyses": analyses(n), "seconds_per_analysis": s,
            "core_hours": analyses(n) * s / SECONDS_PER_HOUR, "budget_core_hours": cfg["budget_core_hours"],
            "within_budget": analyses(n) * s <= budget_s, "n_base_max": n_max}


def _baseline_record(base: S.ModelInputs) -> dict:
    return {"ms": {k: getattr(base.ms, k) for k in S._MS_FIELD_NAMES}, "k_found": base.k_found,
            "p_lumen_MPa": P.get("loads.ivl_balloon_pressure_atm").value}


def sobol_design(n_base: int | None = None, seed: int | None = None) -> dict:
    """Problem, inputs, sampling settings, baseline and the sample matrix ``X`` (JSON-ready).

    ``n_base`` and ``seed`` override the config (tests); the design hash changes with them."""
    cfg = sobol_config()
    inputs, excluded = sobol_inputs()
    n = cfg["n_base"] if n_base is None else int(n_base)
    seed = cfg["seed"] if seed is None else int(seed)
    X = sobol_samples(n, seed)
    k = len(inputs)
    return {"method": METHOD, "problem": sobol_problem(inputs),
            "inputs": [{"name": m.name, "input": m.input, "bounds": list(m.bounds), "scale": m.scale,
                        "basis": m.basis} for m in inputs],
            "excluded_inputs": excluded, "predictors": list(PREDICTOR_KEYS), "lesions": cfg["lesions"],
            "n_base": n, "calc_second_order": False, "scramble": cfg["scramble"], "seed": seed,
            "rows_per_group": k + 2, "n_runs_per_lesion": n * (k + 2),
            "n_runs": n * (k + 2) * len(cfg["lesions"]), "planned_cost": planned_cost(n),
            "baseline": _baseline_record(S.ModelInputs.baseline()), "basis": cfg["basis"], "X": X.tolist()}


def design_hash(design: dict) -> str:
    """SHA-256 of what determines the stored outputs: problem, input mapping and scales, predictor
    keys, N, sampling options, seed, baseline inputs and the sample matrix (not the basis texts)."""
    ident = {k: design[k] for k in ("problem", "predictors", "n_base", "calc_second_order", "scramble", "seed",
                                    "baseline")}
    ident["inputs"] = [(m["name"], m["input"], m["scale"]) for m in design["inputs"]]
    h = hashlib.sha256(json.dumps(_to_jsonable(ident), sort_keys=True).encode())
    h.update(np.ascontiguousarray(np.asarray(design["X"], dtype="<f8")).tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def lesion_mesh(lid: str):
    """(plm, mesh info, MeshData) of a cohort lesion, meshed once per process into a private file
    (:func:`dtivl.drivers.stage_sensitivity.lesion_mesh`)."""
    return S.lesion_mesh(lid)


def evaluate(md, plm, inputs: S.ModelInputs) -> dict:
    """The cohort predictor analysis with the sample's material set and foundation stiffness."""
    from dtivl.metrics.predictors import predictor_analysis

    return predictor_analysis(md, plm, inputs.ms, k_found=inputs.k_found)


def predictor_vector(md, plm, inputs: S.ModelInputs) -> np.ndarray:
    """:func:`evaluate` as a vector over :data:`PREDICTOR_KEYS` (missing / None -> NaN)."""
    pr = evaluate(md, plm, inputs)
    return np.array([np.nan if pr.get(k) is None else float(pr[k]) for k in PREDICTOR_KEYS])


# ---------------------------------------------------------------------------
# Run files
# ---------------------------------------------------------------------------
def _run_path(run_dir: Path, lid: str) -> Path:
    return run_dir / f"{lid}.npz"


def load_runs(path: Path) -> dict | None:
    """Contents of a lesion run file (None if absent)."""
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        rec = {k: z[k] for k in z.files}
    for k in ("lesion", "design_hash", "design", "errors", "mesh", "keys"):
        rec[k] = rec[k].tolist()
    rec["design"], rec["errors"], rec["mesh"] = (json.loads(rec[k]) for k in ("design", "errors", "mesh"))
    rec["errors"] = {int(i): m for i, m in rec["errors"].items()}
    rec["base_status"] = int(rec["base_status"])
    return rec


def _save_runs(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    arrays = {k: rec[k] for k in ("X", "Y", "status", "seconds", "y_base", "base_seconds")}
    arrays.update(lesion=np.array(rec["lesion"]), design_hash=np.array(rec["design_hash"]),
                  design=np.array(json.dumps(_to_jsonable(rec["design"]))), keys=np.array(rec["keys"]),
                  errors=np.array(json.dumps({str(i): m for i, m in rec["errors"].items()})),
                  mesh=np.array(json.dumps(_to_jsonable(rec["mesh"]))), base_status=np.array(rec["base_status"]))
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    tmp.replace(path)


def _new_runs(lid: str, design: dict, h: str, info: dict) -> dict:
    X = np.asarray(design["X"], dtype=float)
    n, p = len(X), len(design["predictors"])
    return {"lesion": lid, "design_hash": h, "design": {k: v for k, v in design.items() if k != "X"}, "X": X,
            "Y": np.full((n, p), np.nan), "status": np.full(n, STATUS_PENDING, dtype=np.int8),
            "seconds": np.full(n, np.nan), "y_base": np.full(p, np.nan), "base_seconds": np.array(np.nan),
            "base_status": STATUS_PENDING, "keys": list(design["predictors"]), "errors": {},
            "mesh": {k: info.get(k) for k in ("n_nodes", "n_cells", "n_cells_per_tag")}}


def _check_stored(path: Path, rec: dict | None, h: str) -> None:
    if rec is not None and rec["design_hash"] != h:
        raise ValueError(f"{path}: stored Sobol outputs belong to another design (config, code or n_base changed); "
                         f"move {path.parent} away to start a new design")


def _complete(rec: dict | None) -> bool:
    return bool(rec is not None and rec["base_status"] == STATUS_OK and np.all(rec["status"] == STATUS_OK))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _todo(rec: dict, rows_end: int, tried: set) -> list[int]:
    """Rows before ``rows_end`` not yet run, and failed rows not yet retried in this :func:`run` call."""
    st = rec["status"][:rows_end]
    retry = (st == STATUS_ERROR) & ~np.isin(np.arange(st.size), sorted(tried))
    return [int(i) for i in np.nonzero((st == STATUS_PENDING) | retry)[0]]


def _attempt(rec: dict, i: int, fn) -> None:
    """Run ``fn`` (one predictor analysis) for row ``i`` (-1: the baseline); a failure is recorded."""
    with Timer() as t:
        try:
            y, status = fn(), STATUS_OK
            rec["errors"].pop(i, None)
        except Exception as exc:              # recorded; the next run() retries it
            log.exception("%s row %d failed: %s", rec["lesion"], i, exc)
            y, status = np.full(len(rec["keys"]), np.nan), STATUS_ERROR
            rec["errors"][i] = f"{type(exc).__name__}: {exc}"
    if i < 0:
        rec["y_base"], rec["base_status"], rec["base_seconds"] = y, status, np.array(t.seconds)
    else:
        rec["Y"][i], rec["status"][i], rec["seconds"][i] = y, status, t.seconds


def _run_block(run_dir: Path, lid: str, design: dict, h: str, rows_end: int, inputs, base, tried: set,
               limit: int | None) -> int:
    """Claim lesion ``lid``, run its baseline and the rows before ``rows_end`` still to do (at most
    ``limit`` analyses) and write its file; returns the number of analyses run here."""
    lock = run_dir / "locks" / f"{lid}.lock"
    if not S._claim(lock):
        return 0                              # another worker holds this lesion
    try:
        path = _run_path(run_dir, lid)
        rec = load_runs(path)                 # read under the lock: another worker may have written it
        _check_stored(path, rec, h)
        need_base = rec is None or (rec["base_status"] != STATUS_OK and -1 not in tried)
        if rec is not None and not need_base and not _todo(rec, rows_end, tried):
            return 0
        plm, info, md = lesion_mesh(lid)
        if rec is None:
            rec = _new_runs(lid, design, h, info)
        elif rec["mesh"].get("n_cells") != info.get("n_cells"):
            raise ValueError(f"{path}: stored outputs are for a mesh of {rec['mesh'].get('n_cells')} cells, the "
                             f"lesion now meshes to {info.get('n_cells')}; move {run_dir} away to start again")
        n_run = 0
        try:
            if need_base:
                tried.add(-1)
                _attempt(rec, -1, lambda: predictor_vector(md, plm, base))
                n_run += 1
            for i in _todo(rec, rows_end, tried):
                if limit is not None and n_run >= limit:
                    break
                tried.add(i)
                _attempt(rec, i, lambda: predictor_vector(md, plm, sobol_point(rec["X"][i], inputs, base)))
                n_run += 1
        finally:
            _save_runs(path, rec)
        return n_run
    finally:
        S._release(lock)


def run(lesions=None, *, out_dir: str | Path | None = None, max_runs: int | None = None,
        n_base: int | None = None) -> dict:
    """Run the Sobol samples on ``lesions`` (default ``design.sensitivity.morris_lesions``) block by block
    (:data:`BLOCK_GROUPS` groups, every lesion per block), resuming the stored files; once every
    requested lesion is complete, :func:`analyze`.

    ``max_runs`` stops after that many predictor analyses (baselines included) in this process;
    ``n_base`` overrides the config N (tests; a stored file of another design is refused). Lesions held
    by another worker are skipped. Returns ``{lesion: analyses run here}``.
    """
    out_dir = Path(out_dir) if out_dir is not None else RESULTS_DIR
    run_dir = out_dir / SOBOL_RUNS
    design = sobol_design(n_base)
    h = design_hash(design)
    lesions = list(lesions or design["lesions"])
    for lid in lesions:                       # refuse a design mismatch before running anything
        path = _run_path(run_dir, lid)
        _check_stored(path, load_runs(path), h)
    inputs = sobol_inputs()[0]
    base = S.ModelInputs.baseline()
    step = design["rows_per_group"]
    n_groups = len(design["X"]) // step
    done = {lid: 0 for lid in lesions}
    tried = {lid: set() for lid in lesions}
    for g_end in range(BLOCK_GROUPS, n_groups + BLOCK_GROUPS, BLOCK_GROUPS):
        for lid in lesions:
            left = None if max_runs is None else max_runs - sum(done.values())
            if left is not None and left <= 0:
                return done
            done[lid] += _run_block(run_dir, lid, design, h, min(g_end, n_groups) * step, inputs, base,
                                    tried[lid], left)
    if all(_complete(load_runs(_run_path(run_dir, lid))) for lid in lesions):
        analyze(out_dir)
    return done


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
NOTE = ("S1 (first-order) and ST (total) Sobol indices are fractions of the variance of each predictor over the "
        "stated input ranges (inputs independent and uniform on each range; log10 scale for the calcium modulus and "
        "the foundation stiffness), for this lesion and the predictor load case; they rank the inputs for these "
        "ranges only. S1_conf and ST_conf are bootstrap half-widths at conf_level. A small negative S1 (or S1 above "
        "ST) within its confidence half-width is sampling error of the estimator, not a negative contribution; "
        "1 - S1_sum is the share of variance due to interactions (up to sampling error). mean and variance (ddof 1) "
        "are taken over the 2N base samples (A and B rows) of the groups used. Only groups with all k + 2 values "
        "present enter; a predictor constant over the samples has no indices. baseline_vs_cohort_run: the baseline "
        "analysis of this stage against the predictors of the lesion's cohort run (largest relative difference; 0 "
        "when the same mesh, inputs and load case reproduce them).")
RANKING_NOTE = ("per sample row, Kendall tau-b between the lesions' baseline predictor values and their values at "
                "that row (all lesions at the same material inputs): tau = 1 when the material uncertainty leaves the "
                "cross-lesion order of the predictor unchanged; with n lesions there are n (n - 1) / 2 pairs, so one "
                "swapped pair lowers tau by 2 / n_pairs (no ties). tau_range = the tau_range_pct percentiles over the "
                "rows, fraction_order_unchanged = share of rows with tau = 1; rows not run for every lesion are left "
                "out.")
#: error messages listed per lesion in the summary (the run files hold all of them)
N_ERROR_EXAMPLES = 5


def _groups_ok(y: np.ndarray, status: np.ndarray, step: int) -> np.ndarray:
    """Indices of the groups whose ``step`` rows are all run and finite for this output."""
    ok = (status == STATUS_OK) & np.isfinite(y)
    return np.nonzero(ok.reshape(-1, step).all(axis=1))[0]


def _indices(problem: dict, y: np.ndarray, status: np.ndarray, cfg: dict) -> dict:
    """Sobol indices of one output from its complete groups, with the moments over the base samples."""
    from SALib.analyze import sobol as SA

    step = problem["num_vars"] + 2
    g = _groups_ok(y, status, step)
    out = {"n_groups": int(g.size), "n_groups_excluded": int(len(y) // step - g.size)}
    v = y[np.concatenate([g * step, g * step + step - 1])]            # A and B rows: the 2N base samples
    out.update(n_base_samples=int(v.size), mean=float(v.mean()) if v.size else None,
               variance=float(v.var(ddof=1)) if v.size > 1 else None,
               std=float(v.std(ddof=1)) if v.size > 1 else None, indices=None, S1_sum=None)
    if g.size < MIN_GROUPS:
        out["note"] = f"fewer than {MIN_GROUPS} complete groups"
        return out
    yg = y[(g[:, None] * step + np.arange(step)[None, :]).ravel()]
    if np.ptp(yg) <= CONST_RTOL * np.max(np.abs(yg)):
        out["note"] = "predictor constant over the samples: variance 0, indices undefined"
        return out
    res = SA.analyze(problem, yg, calc_second_order=False, num_resamples=cfg["conf_resamples"],
                     conf_level=cfg["conf_level"], seed=cfg["seed"])
    out["indices"] = {name: {k: float(np.asarray(res[k], dtype=float)[j]) for k in ("S1", "S1_conf", "ST", "ST_conf")}
                      for j, name in enumerate(problem["names"])}
    out["S1_sum"] = float(np.sum(np.asarray(res["S1"], dtype=float)))
    return out


def kendall_tau_b(b: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Kendall tau-b between ``b`` (n,) and every column of ``Y`` (n, m), vectorised over the columns
    (the definition of ``scipy.stats.kendalltau``, variant b): (concordant - discordant pairs) /
    sqrt(pairs untied in ``b`` x pairs untied in the column); NaN where either has no untied pair."""
    p, q = np.triu_indices(len(b), k=1)
    sb = np.sign(b[p] - b[q])[:, None]
    sy = np.sign(Y[p] - Y[q])
    den = np.sqrt(np.count_nonzero(sb) * np.count_nonzero(sy, axis=0).astype(float))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (sb * sy).sum(axis=0) / den, np.nan)


def ranking_stability(recs: dict, j: int, pct: tuple[float, float]) -> dict:
    """Cross-lesion ranking stability of predictor column ``j``: Kendall tau-b, per sample row, between
    the lesions' baseline values and their values at that row (rows not run for every lesion with a
    baseline value are left out); ``pct``: the percentiles reported as ``tau_range``."""
    lids = [lid for lid, r in recs.items() if r["base_status"] == STATUS_OK and np.isfinite(r["y_base"][j])]
    out = {"lesions": lids, "n_lesions": len(lids), "n_pairs": len(lids) * (len(lids) - 1) // 2,
           "tau_range_pct": list(pct)}
    if len(lids) < 2:
        out["note"] = "fewer than 2 lesions with a baseline value"
        return out
    b = np.array([recs[lid]["y_base"][j] for lid in lids])
    out["baseline"] = dict(zip(lids, b.tolist()))
    out["baseline_order"] = [lids[i] for i in np.argsort(-b, kind="stable")]      # largest baseline value first
    if np.ptp(b) == 0:
        out["note"] = "baseline values equal for all lesions: no order to compare"
        return out
    Y = np.array([recs[lid]["Y"][:, j] for lid in lids])
    ok = np.all([recs[lid]["status"] == STATUS_OK for lid in lids], axis=0) & np.all(np.isfinite(Y), axis=0)
    taus = kendall_tau_b(b, Y[:, ok])
    fin = taus[np.isfinite(taus)]
    out.update(n_samples=int(ok.sum()), n_tau_undefined=int(taus.size - fin.size))
    if fin.size == 0:
        out["note"] = "no sample row with a defined tau"
        return out
    lo, hi = np.percentile(fin, list(pct))
    out.update(tau_median=float(np.median(fin)), tau_range=[float(lo), float(hi)], tau_min=float(fin.min()),
               fraction_order_unchanged=float(np.mean(np.abs(fin - 1) <= TAU_ONE_ATOL)))
    return out


def _seconds_median(sec: np.ndarray, ok: np.ndarray) -> float | None:
    s = sec[ok & np.isfinite(sec)]
    return float(np.median(s)) if s.size else None


def cohort_consistency(rec: dict, cohort_dir: Path) -> dict | None:
    """Baseline outputs of a lesion against the predictors of its cohort run (``<cohort_dir>/<lesion>.json``
    of :mod:`dtivl.drivers.stage_cohort_runs`: same mesh rule, baseline inputs and load case), or None
    without that record or without a baseline here. Relative difference ``|a - b| / |b|`` (``|a - b|``
    where the cohort value is 0) over the predictors with a value in both."""
    cr = S._read(cohort_dir / f"{rec['lesion']}.json")
    if not cr or "predictors" not in cr or rec["base_status"] != STATUS_OK:
        return None
    diff = {}
    for key, a in zip(rec["keys"], rec["y_base"].tolist()):
        b = cr["predictors"].get(key)
        if b is not None and np.isfinite(a):
            diff[key] = abs(a - b) / abs(b) if b != 0 else abs(a - b)
    worst = max(diff, key=diff.get) if diff else None
    return {"cohort_run": f"{cohort_dir.name}/{rec['lesion']}.json", "n_compared": len(diff),
            "max_rel_diff": diff[worst] if diff else None,
            "max_rel_diff_predictor": worst if diff and diff[worst] > 0 else None,
            "n_cells": {"sobol": rec["mesh"].get("n_cells"), "cohort_run": (cr.get("mesh") or {}).get("n_cells")}}


def analyze(out_dir: str | Path | None = None) -> dict:
    """Sobol indices per lesion and predictor and the cross-lesion ranking stability per predictor, from
    the run files (whatever is complete; all files must share one design); writes
    ``<out_dir>/sensitivity_sobol.json``. ``out_dir`` defaults to ``results/``."""
    out_dir = Path(out_dir) if out_dir is not None else RESULTS_DIR
    run_dir = out_dir / SOBOL_RUNS
    loaded = [load_runs(p) for p in sorted(run_dir.glob("*.npz"))]
    if not loaded:
        raise FileNotFoundError(f"no Sobol run files in {run_dir}; run run() first")
    hashes = sorted({r["design_hash"] for r in loaded})
    if len(hashes) != 1:
        raise ValueError(f"{run_dir}: run files of {len(hashes)} different designs")
    design = loaded[0]["design"]
    cfg_now = sobol_config()
    planned = cfg_now["lesions"]
    by_lid = {r["lesion"]: r for r in loaded}
    order = [lid for lid in planned if lid in by_lid] + sorted(set(by_lid) - set(planned))
    recs = {lid: by_lid[lid] for lid in order}
    keys, problem = design["predictors"], design["problem"]
    cfg = {"conf_resamples": cfg_now["conf_resamples"], "conf_level": cfg_now["conf_level"], "seed": design["seed"]}
    try:
        matches = design_hash(sobol_design()) == hashes[0]
    except Exception as exc:                  # the current config may not build a design; recorded as None
        log.warning("current Sobol design not built: %s", exc)
        matches = None
    st_all = np.concatenate([r["status"] for r in recs.values()])
    sec_all = np.concatenate([r["seconds"] for r in recs.values()])
    sec_med = _seconds_median(sec_all, st_all == STATUS_OK)
    cohort_dir = out_dir / COHORT_RUNS
    out = {"stage": "sensitivity_sobol", "method": design["method"],
           "model": "dtivl.metrics.predictors.predictor_analysis (linear-elastic predictor load case, lumen pressure "
                    "loads.ivl_balloon_pressure_atm) with the sample's MaterialSet and k_found",
           **{k: design[k] for k in ("problem", "inputs", "excluded_inputs", "predictors", "n_base",
                                     "calc_second_order", "scramble", "seed", "rows_per_group", "n_runs_per_lesion",
                                     "baseline", "basis")},
           "lesions_planned": planned, "lesions": order, "lesions_missing": [lid for lid in planned if lid not in recs],
           "n_runs_planned": design["n_runs_per_lesion"] * len(planned),
           "n_runs_complete": int((st_all == STATUS_OK).sum()), "n_runs_error": int((st_all == STATUS_ERROR).sum()),
           "planned_cost": design.get("planned_cost"), "seconds_per_run_median": sec_med,
           "core_hours_projected": (None if sec_med is None else
                                    sec_med * (design["n_runs_per_lesion"] + 1) * len(planned) / SECONDS_PER_HOUR),
           "design_hash": hashes[0], "design_matches_config": matches,
           "conf_resamples": cfg["conf_resamples"], "conf_level": cfg["conf_level"], "note": NOTE,
           "lesion_results": {}, "across_lesions": {}, "ranking_note": RANKING_NOTE, "ranking_stability": {}}
    for lid, r in recs.items():
        st = r["status"]
        lres = {"n_complete": int((st == STATUS_OK).sum()), "n_error": int((st == STATUS_ERROR).sum()),
                "n_pending": int((st == STATUS_PENDING).sum()), "baseline_complete": r["base_status"] == STATUS_OK,
                "mesh": r["mesh"], "seconds_per_run_median": _seconds_median(r["seconds"], st == STATUS_OK),
                "errors": dict(sorted(r["errors"].items())[:N_ERROR_EXAMPLES]),
                "baseline": dict(zip(keys, r["y_base"].tolist())),
                "baseline_vs_cohort_run": cohort_consistency(r, cohort_dir),
                "predictors": {key: _indices(problem, r["Y"][:, j], st, cfg) for j, key in enumerate(keys)}}
        out["lesion_results"][lid] = lres
    for j, key in enumerate(keys):
        per = {}
        for name in problem["names"]:
            vals = [(lid, res["predictors"][key]["indices"][name]) for lid, res in out["lesion_results"].items()
                    if res["predictors"][key]["indices"] is not None]
            if vals:
                per[name] = {"n_lesions": len(vals), "lesions": [lid for lid, _ in vals],
                             **{f"{s}_{f.__name__}": float(f(v[s] for _, v in vals)) for s in ("S1", "ST")
                                for f in (min, max)}}
        out["across_lesions"][key] = per
        out["ranking_stability"][key] = ranking_stability(recs, j, cfg_now["tau_range_pct"])
    S._write_atomic(out_dir / SOBOL_JSON, out)
    return out


def main(argv=None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("what", choices=("run", "analyze"))
    ap.add_argument("--lesions", nargs="*", default=None, help="default: design.sensitivity.morris_lesions")
    ap.add_argument("--max-runs", type=int, default=None, help="run: stop after this many predictor analyses")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    if a.what == "run":
        print(json.dumps(run(a.lesions, max_runs=a.max_runs)))
    else:
        res = analyze()
        print(f"{RESULTS_DIR / SOBOL_JSON}: {res['n_runs_complete']} of {res['n_runs_planned']} runs complete")


if __name__ == "__main__":
    main()

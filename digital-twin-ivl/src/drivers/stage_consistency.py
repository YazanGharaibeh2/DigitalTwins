"""Stage (Phase 5): consistency of the simulated outcomes with published observations (checks C1-C12).

Inputs: ``results/cohort_outcomes.csv`` from
:func:`dtivl.drivers.stage_cohort_runs.aggregate` (one row per lesion: cohort
descriptors incl. ``set`` and ``morphology``, the linear-analysis predictors
``pred_*`` and the outcomes ``<protocol>_*`` of the nonlinear damage runs),
``cohort/cohort.csv`` (range of the cross-sectional Fujino score),
``results/cohort_summary.json`` (C12) and ``results/calibration_stent.json``
(C8). Configuration: ``params/study_config.yaml`` section ``consistency``
(every setting has status DESIGN) and the bootstrap settings of section
``metrics`` (``bootstrap_n``, ``bootstrap_stratified``, ``ci_level``).

Published values are read only through :func:`dtivl.params.lit` from
``params/literature_parameters.csv`` (:func:`published`). A row that may not
be used (status UNVERIFIED, or SECONDARY outside
``consistency.allow_status``) or whose value cell is not a plain number (for
example two values in one cell) is not parsed by hand: the check that needs it
is reported as not computed, with the reason.

Checks (one function each, see :data:`CHECKS`; criteria and thresholds in
``consistency.checks.<id>``):

* C1 / C2 - calcium arc / thickness effect: Spearman rho of the descriptor and
  the fracture indicator under the 10 atm plain balloon and under IVL against
  the expected sign; the core Youden cut of the descriptor for plain-balloon
  fracture (:func:`dtivl.stats.roc.roc_analysis`) is shown next to the
  published cutoff, descriptively.
* C3 - nodules fracture less: odds ratio of IVL fracture, nodule vs sheet
  lesions (Haldane-Anscombe correction, Woolf interval), against 1; not
  computed when a morphology group or an outcome class is (almost) empty.
* C4 - lesion-level IVL fracture proportion (Wilson interval) against the
  range of the published rates; the row note carries the configured caveat
  (designed sets, OCT detection) and its context literature row, verbatim.
* C5 - plain balloon at 10 atm: no fracture (expected disagreement, decision D8).
* C6 - IVL balloon alone: no fracture (model-internal criterion of D8).
* C7 - stent expansion after IVL: model mean against published mean +/- 1 SD
  at both published site definitions (the model value is the stent area of
  the single modelled, calcified cross-section over one reference lumen
  area, see ``consistency.checks.C7.site``).
* C8 - acute recoil of both stent protocols (calibration target).
* C9 - calcium stress under balloon loading: predictor stresses scaled
  linearly to the 10 atm plain-balloon load against another FE study's value
  (model vs model).
* C10 - fractures per fractured lesion and full-thickness fraction (lesion-level
  bootstrap interval).
* C11 - Fujino score vs stent expansion without IVL: direction only.
* C12 - geometry plausibility: implied plaque burden against CTA, copied from
  ``results/cohort_summary.json``.

Every lesion-level check is computed per set (``consistency.sets``: core, and
hold-out in its own row; never pooled); C12 is one cohort-level row. Each row
states the number of lesions used, the runs excluded (status not allowed by
``consistency.ok_status``, missing value) and, when a check cannot be computed
(too few lesions, a constant variable, a missing column or file, an unreadable
literature row), the reason. Intervals are deterministic: every bootstrap uses
its own stream from :func:`dtivl.stats.roc.rng_for` with entropy
``design.seed + consistency.seed_offset``.

Agreement labels (:data:`AGREEMENT`): agrees, disagrees, partial, not
computed. The ``kind`` of a row says what the comparison is:
``published_observation``; ``model_internal`` (C6, a criterion of the
calibration, not a publication); ``calibration_target`` (C8, the stent
stiffness was identified on the published recoil); ``model_vs_model`` (C9, the
published value is itself an FE result); ``geometry_plausibility`` (C12).

Predictors and outcomes stay apart (study rule 4): the checks read geometry
descriptors, the Fujino score and, for C9 only, the predictor stresses as
quantities; the IVL load factor (``ivl_load_*``) is never read
(:func:`check_columns`).

Outputs (``results/``): ``consistency.csv`` (one row per check and set,
:data:`CSV_COLUMNS`) and ``consistency.json`` (the same rows with every model
quantity, the literature rows as read, input hashes and the configuration).
The wording is consistency with published observations;
:func:`check_wording` refuses output text that claims more.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as _st

from dtivl import params as P
from dtivl.drivers.common import COHORT_DIR, RESULTS_DIR, _to_jsonable, write_json
from dtivl.drivers.stage_metrics import MODULUS_TOL_MPA, _as_binary, _design_value
from dtivl.fracture.cracks import DIFFUSE_REGION_FRAC
from dtivl.geometry.fujino import fujino_max_score
from dtivl.stats import roc as R

OUTCOMES_CSV = RESULTS_DIR / "cohort_outcomes.csv"
COHORT_CSV = COHORT_DIR / "cohort.csv"
COHORT_SUMMARY_JSON = RESULTS_DIR / "cohort_summary.json"
CALIBRATION_JSON = RESULTS_DIR / "calibration_stent.json"

NOT_COMPUTED = "not computed"
AGREEMENT = ("agrees", "disagrees", "partial", NOT_COMPUTED)
KINDS = ("published_observation", "model_internal", "calibration_target", "model_vs_model",
         "geometry_plausibility")
#: Checks computed once for the whole cohort (not per set).
COHORT_LEVEL_CHECKS = ("C12",)
#: Lumen load of the balloon protocols, as applied by dtivl.drivers.stage_cohort_runs.run_lesion.
PROTOCOL_LOAD = {"balloon_only": "loads.ivl_balloon_pressure_atm", "balloon_10atm": "loads.poba_pressure_atm"}
#: NUMERICAL: relative tolerance when comparing the CTA values copied into cohort_summary.json with params.
COPY_REL_TOL = 1e-9
#: Study rule 3: no output text may contain this word stem.
FORBIDDEN_STEM = "validat"
#: Fraction -> percent.
PCT = 100.0
#: Null values of the direction checks: Spearman rho (C1, C2, C11) and odds ratio (C3).
RHO_NULL = 0.0
OR_NULL = 1.0

CSV_COLUMNS = ["id", "set", "kind", "observation", "source_rows", "source_status", "published", "model_quantity",
               "model_value", "ci_lo", "ci_hi", "ci_method", "model_values", "criterion", "agreement", "n_lesions",
               "note"]

WORDING = ("Agreement labels state whether simulated (in silico) outcomes are consistent with published "
           "observations; they are not a test of the model against patients. Rows of kind model_internal "
           "(a criterion of the calcium-modulus calibration), calibration_target (a quantity the model was "
           "calibrated to), model_vs_model (the published value is another simulation) and geometry_plausibility "
           "are not independent comparisons with observations.")


# =========================================================================== configuration
@dataclass(frozen=True)
class ConsistencyConfig:
    """Resolved ``consistency`` configuration (see params/study_config.yaml)."""

    seed: int
    n_boot: int
    stratified: bool
    level: float
    sets: tuple
    derivation_set: str
    ok_status: dict
    allow_status: tuple
    min_n: int
    min_per_group: int
    require_ci: bool
    or_correction: float
    checks: dict
    bases: dict
    forbidden_prefixes: tuple
    provenance: dict = field(default_factory=dict)

    def ok(self, protocol: str) -> list:
        """Run statuses of ``protocol`` that a check uses."""
        return list(self.ok_status.get(protocol, self.ok_status["default"]))

    @property
    def level_text(self) -> str:
        return f"{PCT * self.level:g} %"


def load_config(n_boot: int | None = None) -> ConsistencyConfig:
    """Read the ``consistency`` section (``n_boot`` overrides ``metrics.bootstrap_n``, e.g. for tests)."""
    checks, bases = {}, {}
    for cid, node in P.raw("consistency.checks").items():
        if node.get("status") == "PENDING":
            raise ValueError(f"config entry consistency.checks.{cid} is PENDING")
        checks[cid] = dict(node["value"])
        bases[cid] = node.get("basis", "")
    cfg = ConsistencyConfig(
        seed=int(P.raw("design.seed")) + int(P.get("consistency.seed_offset").value),
        n_boot=int(P.get("metrics.bootstrap_n").value) if n_boot is None else int(n_boot),
        stratified=bool(_design_value("metrics.bootstrap_stratified")),
        level=float(P.get("metrics.ci_level").value),
        sets=tuple(_design_value("consistency.sets")),
        derivation_set=str(_design_value("metrics.sets")["derivation"]),
        ok_status={k: list(v) for k, v in _design_value("consistency.ok_status").items()},
        allow_status=tuple(_design_value("consistency.allow_status")),
        min_n=int(P.get("consistency.min_n").value),
        min_per_group=int(P.get("consistency.min_per_group").value),
        require_ci=bool(_design_value("consistency.direction_rule")["require_ci"]),
        or_correction=float(_design_value("consistency.odds_ratio")["correction"]),
        checks=checks, bases=bases,
        forbidden_prefixes=tuple(_design_value("metrics.forbidden_predictor_prefixes")),
        provenance={"seed": "design.seed + consistency.seed_offset",
                    "bootstrap_n": "metrics.bootstrap_n"
                    + (" (overridden by the caller)" if n_boot is not None else ""),
                    "bootstrap_stratified": "metrics.bootstrap_stratified", "ci_level": "metrics.ci_level",
                    "derivation_set": "metrics.sets"})
    if "UNVERIFIED" in cfg.allow_status:
        raise ValueError("consistency.allow_status may not contain UNVERIFIED (study rule 2)")
    unknown = [c for c in cfg.checks if c not in CHECKS]
    if unknown:
        raise ValueError(f"consistency.checks names checks without a function: {unknown}")
    bad_kind = {c: s.get("kind") for c, s in cfg.checks.items() if s.get("kind") not in KINDS}
    if bad_kind:
        raise ValueError(f"consistency.checks: unknown kind {bad_kind}")
    check_columns(cfg)
    return cfg


def check_columns(cfg: ConsistencyConfig) -> None:
    """Refuse configured columns that are the IVL load factor (an outcome-model input, study rule 4)."""
    bad = []
    for cid, spec in cfg.checks.items():
        cols = [spec.get(k) for k in ("column", "load_column", "outcome")] + list(spec.get("columns", []))
        for c in filter(None, cols):
            if any(c.startswith(p) for p in cfg.forbidden_prefixes) or "T_equivalent" in c:
                bad.append((cid, c))
    if bad:
        raise ValueError(f"consistency.checks reads the IVL load factor (ivl_load_*), which is refused: {bad}")


# =========================================================================== inputs
def load_outcomes(path: str | Path | None = None) -> pd.DataFrame:
    """Per-lesion table (``results/cohort_outcomes.csv``, written by ``stage_cohort_runs.aggregate()``)."""
    path = Path(path or OUTCOMES_CSV)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run dtivl.drivers.stage_cohort_runs.aggregate() first")
    return pd.read_csv(path)


def _read_optional_json(path: Path) -> dict | None:
    return json.loads(Path(path).read_text()) if Path(path).exists() else None


def load_cohort_summary(path: str | Path | None = None) -> dict | None:
    """``results/cohort_summary.json`` (``None`` if absent)."""
    return _read_optional_json(Path(path or COHORT_SUMMARY_JSON))


def load_calibration(path: str | Path | None = None) -> dict | None:
    """``results/calibration_stent.json`` (``None`` if absent)."""
    return _read_optional_json(Path(path or CALIBRATION_JSON))


def load_cohort(path: str | Path | None = None) -> pd.DataFrame | None:
    """``cohort/cohort.csv`` (``None`` if absent)."""
    path = Path(path or COHORT_CSV)
    return pd.read_csv(path) if path.exists() else None


# =========================================================================== literature
def published(row: str, cfg: ConsistencyConfig) -> dict:
    """One literature row as read by :mod:`dtivl.params`.

    Returns the verbatim cells, the status and source, and (``ok`` True) the
    value in base units from :func:`dtivl.params.lit` with the SD and the
    alternative value parsed by the package; ``ok`` False with ``reason`` when
    the row is missing, its status is not allowed, or its value cell is not a
    plain number (never parsed by hand).
    """
    out = {"row": row, "status": None, "unit": "", "value_cell": None, "range_cell": None, "source": None,
           "value": None, "sd": None, "alt": None, "ok": False, "reason": None}
    lit = P.load_literature()
    if row not in lit:
        out["reason"] = f"row {row!r} not in literature_parameters.csv"
        return out
    p = lit[row]
    out.update(status=p.status, unit=p.unit, value_cell=p.value_str, range_cell=p.range_str, source=p.source)
    if p.status == "UNVERIFIED":
        out["reason"] = f"row {row!r} has status UNVERIFIED (never used)"
        return out
    try:
        out["value"] = P.lit(row, allow_unverified=p.status in cfg.allow_status).value
    except ValueError as exc:
        msg = str(exc)
        if "no numeric" in msg:
            msg += f" (value cell {p.value_str!r} is not a plain number; not parsed by hand)"
        out["reason"] = msg
        return out
    out.update(ok=True, sd=p.sd, alt=p.alt)
    return out


def _pub_text(pub: dict) -> str:
    if not pub.get("ok"):
        return f"{pub['row']}: not read ({pub['reason']})"
    s = f"{pub['row']} = {pub['value']:g}"
    if pub["sd"] is not None:
        s += f" SD {pub['sd']:g}"
    if pub["unit"]:
        s += f" {pub['unit']}"
    if pub["sd"] is None and pub["range_cell"]:
        s += f" ({pub['range_cell']})"
    if pub["status"] != "VERIFIED":
        s += f" [status {pub['status']}]"
    return s


def _context_text(rows, cfg: ConsistencyConfig) -> str | None:
    """Context literature rows of a caveat (``{row, cell}``: the verbatim ``value`` or ``range`` cell, not
    compared with the model); rows that may not be used are left out."""
    parts = []
    for item in rows:
        lit = P.load_literature().get(item["row"])
        if lit is None or lit.status == "UNVERIFIED" or (lit.status not in P.MAIN_ANALYSIS_OK
                                                          and lit.status not in cfg.allow_status):
            continue
        cell = {"value": lit.value_str, "range": lit.range_str}[item["cell"]]
        parts.append(f"{item['row']} ({item['cell']} cell, status {lit.status}): '{cell}'")
    return ("context, not compared: " + "; ".join(parts)) if parts else None


# =========================================================================== statistics
def proportion(k: int, n: int, level: float) -> dict:
    """``k`` of ``n`` in percent with the Wilson score interval (:func:`dtivl.stats.roc.wilson_ci`)."""
    lo, hi = R.wilson_ci(k, n, level)
    return {"k": int(k), "n": int(n), "value": PCT * k / n if n else math.nan, "ci_lo": PCT * lo,
            "ci_hi": PCT * hi, "ci_method": f"Wilson score, {PCT * level:g} %", "unit": "%"}


def spearman_ci(x, y, cfg: ConsistencyConfig, key: str, *, binary_y: bool) -> dict:
    """Spearman rho of ``x`` and ``y`` with a percentile bootstrap interval (stream ``key``).

    With a binary ``y`` and ``metrics.bootstrap_stratified`` the lesions are
    resampled within each outcome class; resamples with a constant variable
    give no rho and are left out (``n_boot_valid``).
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    rho = R.spearman(x, y)
    strat = bool(binary_y and cfg.stratified)
    out = {"value": rho, "n": int(x.size), "ci_lo": math.nan, "ci_hi": math.nan, "n_boot": cfg.n_boot,
           "n_boot_valid": 0, "ci_method": (f"percentile bootstrap, {cfg.n_boot} resamples"
                                            + (", stratified by outcome" if strat else "") + f", {cfg.level_text}")}
    if not math.isfinite(rho):
        return out
    idx = R.bootstrap_indices(R.rng_for(cfg.seed, key), cfg.n_boot, x.size, y if strat else None, strat)
    ra, rb = _st.rankdata(x[idx], axis=1), _st.rankdata(y[idx], axis=1)
    ra, rb = ra - ra.mean(axis=1, keepdims=True), rb - rb.mean(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        rho_b = (ra * rb).sum(axis=1) / np.sqrt((ra ** 2).sum(axis=1) * (rb ** 2).sum(axis=1))
    out["ci_lo"], out["ci_hi"] = R.percentile_ci(rho_b, cfg.level)
    out["n_boot_valid"] = int(np.isfinite(rho_b).sum())
    return out


def odds_ratio(a: int, b: int, c: int, d: int, correction: float, level: float) -> dict:
    """Odds ratio ``(a d) / (b c)`` of the table [[a, b], [c, d]] (rows: exposed, reference; columns: event,
    no event) with ``correction`` added to every cell, and its Woolf (log) interval."""
    cells = np.array([a, b, c, d], float) + correction
    out = {"table": {"exposed_event": int(a), "exposed_no_event": int(b), "reference_event": int(c),
                     "reference_no_event": int(d)}, "correction": correction,
           "ci_method": f"Woolf (log), {PCT * level:g} %, Haldane-Anscombe correction {correction:g}"}
    if (cells <= 0).any():
        return {**out, "value": math.nan, "ci_lo": math.nan, "ci_hi": math.nan}
    lor = math.log(cells[0] * cells[3] / (cells[1] * cells[2]))
    se = math.sqrt(float((1 / cells).sum()))
    z = float(_st.norm.ppf((1 + level) / 2))
    return {**out, "value": math.exp(lor), "ci_lo": math.exp(lor - z * se), "ci_hi": math.exp(lor + z * se),
            "log_or": lor, "se_log_or": se}


def describe(values) -> dict:
    """n, mean, SD (ddof 1), median, quartiles and range of the finite entries of ``values``."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0, "mean": math.nan, "sd": math.nan, "median": math.nan, "q1": math.nan, "q3": math.nan,
                "min": math.nan, "max": math.nan}
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return {"n": int(v.size), "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if v.size > 1 else math.nan,
            "median": float(med), "q1": float(q1), "q3": float(q3), "min": float(v.min()), "max": float(v.max())}


def direction_label(est: float, lo: float, hi: float, null: float, sign: int, require_ci: bool) -> str:
    """Agreement of a direction estimate with the expected ``sign`` relative to ``null``
    (``consistency.direction_rule``)."""
    if not math.isfinite(est):
        return NOT_COMPUTED
    if sign * (est - null) <= 0:
        return "disagrees"
    if not require_ci:
        return "agrees"
    if math.isfinite(lo) and math.isfinite(hi) and sign * (lo - null) > 0 and sign * (hi - null) > 0:
        return "agrees"
    return "partial"


def combine(labels) -> str:
    """One label from several: agrees / disagrees if every computed part does, partial otherwise."""
    got = [lab for lab in labels if lab != NOT_COMPUTED]
    if not got:
        return NOT_COMPUTED
    if all(lab == "agrees" for lab in got):
        return "agrees"
    if all(lab == "disagrees" for lab in got):
        return "disagrees"
    return "partial"


# =========================================================================== helpers
def _g(v, sig: int = 4) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "nan"
    return f"{float(v):.{sig}g}"


def _atm(mpa: float) -> float:
    """MPa -> atm with the conversion constant of :mod:`dtivl.params`."""
    return mpa / P._UNIT_FACTOR["atm"]


def _load_text(key: str) -> str:
    p = P.get(key).value
    return f"{_atm(p):g} atm = {p:g} MPa ({key})"


def _ids(t: pd.DataFrame) -> list:
    return t["lesion_id"].tolist() if "lesion_id" in t else [str(i) for i in t.index]


def _protocol_values(t: pd.DataFrame, protocol: str, column: str, cfg: ConsistencyConfig,
                     binary: bool = False) -> tuple[pd.DataFrame, dict]:
    """Rows of ``t`` whose ``protocol`` run has an allowed status and a finite ``<protocol>_<column>``,
    with the value in ``_v``; returns ``(rows, exclusions)``."""
    st_col = f"{protocol}_status"
    status = t[st_col] if st_col in t else pd.Series([None] * len(t), index=t.index, dtype=object)
    ok = status.isin(cfg.ok(protocol))
    excl = {f"{protocol} status {k}": int(v) for k, v in status[~ok].fillna("not run").value_counts().items()}
    r = t[ok].copy()
    col = f"{protocol}_{column}"
    if col not in r:
        if len(r):
            excl[f"{col} column missing"] = int(len(r))
        r["_v"] = np.nan
        return r.iloc[0:0], excl
    v = _as_binary(r[col]) if binary else pd.to_numeric(r[col], errors="coerce").to_numpy(float)
    keep = np.isfinite(v)
    if (~keep).any():
        excl[f"{col} missing"] = int((~keep).sum())
    r["_v"] = v
    return r[keep], excl


def _excl_text(excl: dict) -> str | None:
    items = [f"{k} {v}" for k, v in excl.items() if v]
    return ("excluded: " + ", ".join(items)) if items else None


def _rule_text(cfg: ConsistencyConfig, null: str) -> str:
    if cfg.require_ci:
        return (f"agrees if the point estimate lies on the expected side of {null} and its {cfg.level_text} "
                f"interval excludes {null}, partial if only the point estimate does, disagrees otherwise")
    return f"agrees if the point estimate lies on the expected side of {null}, disagrees otherwise"


def _quantity(name: str, stat: dict, **extra) -> dict:
    q = {"name": name, "value": stat.get("value"), "ci_lo": stat.get("ci_lo"), "ci_hi": stat.get("ci_hi"),
         "ci_method": stat.get("ci_method"), "n": stat.get("n")}
    q.update({k: v for k, v in stat.items() if k not in q})
    q.update(extra)
    return q


def _summary_quantity(name: str, values, stat: str, **extra) -> dict:
    """Distribution quantity (no interval): ``value`` is the ``stat`` (``mean`` or ``median``) of
    :func:`describe`."""
    d = describe(values)
    return {"name": name, "value": d[stat], "stat": stat, "ci_lo": None, "ci_hi": None, "ci_method": None,
            **d, **extra}


def _finite(v) -> bool:
    return v is not None and isinstance(v, (int, float, np.integer, np.floating)) and math.isfinite(v)


def _values_text(quantities) -> str | None:
    """CSV text of the model values: estimate, interval or spread, range and n of each quantity."""
    parts = []
    for q in quantities:
        s = q["name"] if q.get("value") is None else f"{q['name']} {_g(q['value'])}"
        if q.get("stat") == "median" and _finite(q.get("q1")):
            s += f" (IQR {_g(q['q1'])}-{_g(q['q3'])})"
        elif _finite(q.get("sd")):
            s += f" SD {_g(q['sd'])}"
        if q.get("ci_method") and _finite(q.get("ci_lo")):
            s += f" [{_g(q['ci_lo'])}, {_g(q['ci_hi'])}]"
        if _finite(q.get("min")) and _finite(q.get("max")):
            s += f" (range {_g(q['min'])}-{_g(q['max'])})"
        if q.get("n") is not None:
            s += f" (n {q['n']})"
        parts.append(s)
    return "; ".join(parts) or None


def _clean(obj):
    """Non-finite floats -> ``None`` (``_to_jsonable`` maps NaN only; JSON has no infinity)."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (float, np.floating)) and not math.isfinite(float(obj)):
        return None
    return obj


@dataclass
class _Ctx:
    cfg: ConsistencyConfig
    set_: str
    t: pd.DataFrame
    extra: dict


def _row(ctx: _Ctx, cid: str, spec: dict, pubs: list, *, model_quantity: str, criterion: str, agreement: str,
         quantities=(), lesions=(), notes=(), details: dict | None = None) -> dict:
    """One output row (CSV columns plus the full quantities, literature rows and basis for the JSON)."""
    if agreement not in AGREEMENT:
        raise ValueError(f"unknown agreement label {agreement!r}")
    quantities = list(quantities)
    first = quantities[0] if quantities else {}
    row = {
        "id": cid, "set": ctx.set_, "kind": spec["kind"], "observation": spec.get("observation"),
        "source_rows": "; ".join(p["row"] for p in pubs) or None,
        "source_status": "; ".join(f"{p['row']}: {p.get('status')}" for p in pubs) or None,
        "published": "; ".join(_pub_text(p) for p in pubs) or "none (model-internal criterion)",
        "model_quantity": model_quantity, "model_value": first.get("value"), "ci_lo": first.get("ci_lo"),
        "ci_hi": first.get("ci_hi"), "ci_method": first.get("ci_method"), "model_values": _values_text(quantities),
        "criterion": criterion, "agreement": agreement, "n_lesions": int(len(set(lesions))),
        "note": "; ".join(n for n in notes if n) or None,
    }
    row["quantities"] = quantities
    row["published_rows"] = pubs
    row["basis"] = ctx.cfg.bases.get(cid)
    if details:
        row["details"] = details
    return row


# =========================================================================== checks
def _fracture_direction(ctx: _Ctx, cid: str, spec: dict) -> dict:
    """C1 / C2: Spearman rho of a descriptor and the fracture indicator per protocol; core Youden cut."""
    cfg, t = ctx.cfg, ctx.t
    col, sign, protos = spec["column"], int(spec["expected_sign"]), list(spec["protocols"])
    pubs = [published(r, cfg) for r in spec["published"]]
    mq = (f"Spearman rho of {col} and the fracture indicator (<protocol>_fracture) under "
          f"{', '.join(protos)}" + (f"; core Youden cut of {col} for {spec['youden_protocol']}_fracture "
                                     "(descriptive)" if spec.get("youden_protocol") else ""))
    crit = f"rho {'>' if sign > 0 else '<'} {RHO_NULL:g} under every protocol: {_rule_text(cfg, f'{RHO_NULL:g}')}"
    if col not in t:
        return _row(ctx, cid, spec, pubs, model_quantity=mq, criterion=crit, agreement=NOT_COMPUTED,
                    notes=[f"column {col} missing from cohort_outcomes.csv"])
    quantities, labels, notes, lesions = [], [], [], []
    for proto in protos:
        r, excl = _protocol_values(t, proto, "fracture", cfg, binary=True)
        notes.append(_excl_text(excl))
        x = pd.to_numeric(r[col], errors="coerce").to_numpy(float)
        m = np.isfinite(x)
        x, y, ids = x[m], r["_v"].to_numpy(float)[m], np.asarray(_ids(r))[m]
        if x.size < cfg.min_n:
            labels.append(NOT_COMPUTED)
            notes.append(f"{proto}: fewer than {cfg.min_n} lesions ({x.size})")
            continue
        sp = spearman_ci(x, y, cfg, f"{cid}|rho_{proto}|{ctx.set_}", binary_y=True)
        if not math.isfinite(sp["value"]):
            labels.append(NOT_COMPUTED)
            notes.append(f"{proto}: rho undefined ({col} or the fracture indicator is constant; "
                         f"{int(y.sum())} of {y.size} fractured)")
            continue
        lab = direction_label(sp["value"], sp["ci_lo"], sp["ci_hi"], RHO_NULL, sign, cfg.require_ci)
        labels.append(lab)
        quantities.append(_quantity(f"rho {proto}", sp, n_fractured=int(y.sum()), agreement=lab))
        lesions.extend(ids)
        notes.append(f"{proto}: {lab}")
    yp = spec.get("youden_protocol")
    if yp and ctx.set_ == cfg.derivation_set:
        q, note = _youden_quantity(ctx, cid, col, sign, yp, pubs)
        if q:
            quantities.append(q)
        notes.append(note)
    elif yp:
        notes.append(f"Youden cut derived on the {cfg.derivation_set} set only")
    return _row(ctx, cid, spec, pubs, model_quantity=mq, criterion=crit, agreement=combine(labels),
                quantities=quantities, lesions=lesions, notes=notes)


def _youden_quantity(ctx: _Ctx, cid: str, col: str, sign: int, protocol: str, pubs: list):
    cfg = ctx.cfg
    r, _ = _protocol_values(ctx.t, protocol, "fracture", cfg, binary=True)
    x = pd.to_numeric(r[col], errors="coerce").to_numpy(float)
    m = np.isfinite(x)
    x, y = x[m], r["_v"].to_numpy(float)[m].astype(int)
    n1, n0 = int(y.sum()), int(y.size - y.sum())
    if n1 < cfg.min_per_group or n0 < cfg.min_per_group:
        return None, (f"Youden cut not computed: fewer than {cfg.min_per_group} lesions in an outcome class "
                      f"under {protocol} ({n1} fractured, {n0} not)")
    if np.ptp(x) == 0:
        return None, f"Youden cut not computed: {col} constant"
    ra = R.roc_analysis(y, x, sign, n_boot=cfg.n_boot, rng=R.rng_for(cfg.seed, f"{cid}|youden|{ctx.set_}"),
                        level=cfg.level, stratified=cfg.stratified)
    q = {"name": f"Youden cut {col} ({protocol})", "value": ra["threshold"], "ci_lo": ra["threshold_ci_lo"],
         "ci_hi": ra["threshold_ci_hi"], "ci_method": (f"percentile bootstrap, {cfg.n_boot} resamples"
                                                      + (", stratified by outcome" if cfg.stratified else "")
                                                      + f", {cfg.level_text}"),
         "n": int(y.size), "rule": f"fracture when {col} {'>=' if sign > 0 else '<='} cut", "auc": ra["auc"],
         "auc_ci_lo": ra["auc_ci_lo"], "auc_ci_hi": ra["auc_ci_hi"], "sens": ra["sens"], "spec": ra["spec"],
         "n_levels": ra["n_levels"], "descriptive": True}
    pub = "; ".join(_pub_text(p) for p in pubs)
    return q, (f"Youden cut descriptive, not part of the agreement: {_g(ra['threshold'])} "
               f"({q['rule']}; AUC {_g(ra['auc'], 3)}; {ra['n_levels']} distinct values of {col} in the "
               f"{ctx.set_} set) next to the published {pub}")


def check_arc(ctx: _Ctx, cid: str, spec: dict) -> dict:
    """C1: larger calcium arc, more fracture (published plain-balloon arc cutoff)."""
    return _fracture_direction(ctx, cid, spec)


def check_thickness(ctx: _Ctx, cid: str, spec: dict) -> dict:
    """C2: thinner calcium, more fracture (published plain-balloon thickness cutoff)."""
    return _fracture_direction(ctx, cid, spec)


def check_nodule(ctx: _Ctx, cid: str, spec: dict) -> dict:
    """C3: nodules fracture less under IVL: odds ratio nodule vs sheet lesions against 1."""
    cfg, t = ctx.cfg, ctx.t
    proto, sign = spec["protocol"], int(spec["expected_sign"])
    exp_m, ref_m = list(spec["exposed"]), list(spec["reference"])
    pubs = [published(r, cfg) for r in spec["published"]]
    mq = (f"odds ratio of {proto}_fracture, {' + '.join(exp_m)} vs {' + '.join(ref_m)} lesions "
          f"(Haldane-Anscombe correction {cfg.or_correction:g}, Woolf interval); {' + '.join(exp_m)} fracture "
          "proportion (descriptive)")
    crit = f"odds ratio {'<' if sign < 0 else '>'} {OR_NULL:g}: {_rule_text(cfg, f'{OR_NULL:g}')}"
    if "morphology" not in t:
        return _row(ctx, cid, spec, pubs, model_quantity=mq, criterion=crit, agreement=NOT_COMPUTED,
                    notes=["column morphology missing from cohort_outcomes.csv"])
    r, excl = _protocol_values(t, proto, "fracture", cfg, binary=True)
    e, f = r[r["morphology"].isin(exp_m)], r[r["morphology"].isin(ref_m)]
    a, b = int(e["_v"].sum()), int(len(e) - e["_v"].sum())
    c, d = int(f["_v"].sum()), int(len(f) - f["_v"].sum())
    notes = [_excl_text(excl)]
    if len(e) < cfg.min_per_group or len(f) < cfg.min_per_group:
        return _row(ctx, cid, spec, pubs, model_quantity=mq, criterion=crit, agreement=NOT_COMPUTED,
                    lesions=_ids(e) + _ids(f), notes=[f"fewer than {cfg.min_per_group} lesions in a morphology "
                                                       f"group ({len(e)} {'+'.join(exp_m)}, {len(f)} "
                                                       f"{'+'.join(ref_m)})"] + notes)
    if a + c < cfg.min_per_group or b + d < cfg.min_per_group:
        # one outcome class (almost) empty: the corrected odds ratio would only reflect the group sizes
        return _row(ctx, cid, spec, pubs, model_quantity=mq, criterion=crit, agreement=NOT_COMPUTED,
                    lesions=_ids(e) + _ids(f), notes=[f"fewer than {cfg.min_per_group} lesions in an outcome class "
                                                       f"({a + c} fractured, {b + d} not)"] + notes)
    orr = odds_ratio(a, b, c, d, cfg.or_correction, cfg.level)
    lab = direction_label(orr["value"], orr["ci_lo"], orr["ci_hi"], OR_NULL, sign, cfg.require_ci)
    quantities = [_quantity("odds ratio", {**orr, "n": a + b + c + d}),
                  _quantity(f"{'+'.join(exp_m)} fracture %", proportion(a, a + b, cfg.level)),
                  _quantity(f"{'+'.join(ref_m)} fracture %", proportion(c, c + d, cfg.level))]
    other = sorted(set(r["morphology"].dropna()) - set(exp_m) - set(ref_m))
    notes.insert(0, f"2x2 table: {a} of {a + b} {'+'.join(exp_m)} and {c} of {c + d} {'+'.join(ref_m)} lesions "
                    "fractured" + (f"; {', '.join(other)} lesions are in neither group" if other else ""))
    if lab == NOT_COMPUTED:
        notes.insert(1, f"odds ratio undefined (an empty cell with correction {cfg.or_correction:g})")
    return _row(ctx, cid, spec, pubs, model_quantity=mq, criterion=crit, agreement=lab, quantities=quantities,
                lesions=_ids(e) + _ids(f), notes=notes)


def check_fracture_rate(ctx: _Ctx, cid: str, spec: dict) -> dict:
    """C4: lesion-level IVL fracture proportion against the range of the published rates."""
    cfg = ctx.cfg
    proto = spec["protocol"]
    if spec.get("rule") != "band_or_ci_overlap":
        raise ValueError(f"consistency.checks.{cid}: unknown rule {spec.get('rule')!r}")
    pubs = [published(r, cfg) for r in spec["published"]]
    vals = [p["value"] for p in pubs if p["ok"]]
    mq = f"proportion of lesions with {proto}_fracture (%), Wilson interval"
    band = (min(vals), max(vals)) if vals else (math.nan, math.nan)
    crit = (f"agrees if the proportion lies within [{band[0]:g}, {band[1]:g}] % (published rates) or its "
            f"{cfg.level_text} Wilson interval overlaps that range; disagrees otherwise")
    r, excl = _protocol_values(ctx.t, proto, "fracture", cfg, binary=True)
    notes = [_excl_text(excl), spec.get("caveat"), _context_text(spec.get("context_rows", []), cfg)]
    if not vals:
        return _row(ctx, cid, spec, pubs, model_quantity=mq, criterion=crit, agreement=NOT_COMPUTED,
                    lesions=_ids(r), notes=["no published value readable"] + notes)
    if len(r) < cfg.min_n:
        return _row(ctx, cid, spec, pubs, model_quantity=mq, criterion=crit, agreement=NOT_COMPUTED,
                    lesions=_ids(r), notes=[f"fewer than {cfg.min_n} lesions ({len(r)})"] + notes)
    pr = proportion(int(r["_v"].sum()), len(r), cfg.level)
    in_band = band[0] <= pr["value"] <= band[1]
    overlap = pr["ci_lo"] <= band[1] and pr["ci_hi"] >= band[0]
    lab = "agrees" if in_band or overlap else "disagrees"
    how = ("proportion within the published range" if in_band else
           "only the Wilson interval overlaps the published range" if overlap else
           "proportion and Wilson interval outside the published range")
    return _row(ctx, cid, spec, pubs, model_quantity=mq, criterion=crit, agreement=lab,
                quantities=[_quantity(f"{proto} fracture %", pr, in_published_range=bool(in_band),
                                      ci_overlaps_published_range=bool(overlap))],
                lesions=_ids(r), notes=[how] + notes, details={"context_rows": spec.get("context_rows", [])})


def check_no_fracture(ctx: _Ctx, cid: str, spec: dict) -> dict:
    """C5 / C6: no lesion fractures under a balloon without pulses (rule ``zero``)."""
    cfg = ctx.cfg
    proto = spec["protocol"]
    if spec.get("rule") != "zero":
        raise ValueError(f"consistency.checks.{cid}: unknown rule {spec.get('rule')!r}")
    pubs = [published(r, cfg) for r in spec.get("published", [])]
    load = f" ({_load_text(PROTOCOL_LOAD[proto])})" if proto in PROTOCOL_LOAD else ""
    mq = f"proportion of lesions with {proto}_fracture (%), Wilson interval"
    crit = f"agrees only if no lesion fractures under {proto}{load}; disagrees otherwise"
    notes = []
    for p in pubs:
        if p["ok"] and p["value"] != 0:
            notes.append(f"rule 'zero' but {p['row']} is {p['value']:g}")
    if spec["kind"] == "model_internal":
        notes.append("model-internal criterion, not a published observation")
    if spec.get("decision"):
        notes.append(f"decision {spec['decision']}: {P.raw('decisions.' + spec['decision'])}")
    r, excl = _protocol_values(ctx.t, proto, "fracture", cfg, binary=True)
    notes.append(_excl_text(excl))
    if len(r) < cfg.min_n:
        return _row(ctx, cid, spec, pubs, model_quantity=mq, criterion=crit, agreement=NOT_COMPUTED,
                    lesions=_ids(r), notes=[f"fewer than {cfg.min_n} lesions ({len(r)})"] + notes)
    k = int(r["_v"].sum())
    pr = proportion(k, len(r), cfg.level)
    return _row(ctx, cid, spec, pubs, model_quantity=mq, criterion=crit,
                agreement="agrees" if k == 0 else "disagrees", quantities=[_quantity(f"{proto} fracture %", pr)],
                lesions=_ids(r), notes=[f"{k} of {len(r)} lesions fractured"] + notes)


def _mean_within_sd(ctx: _Ctx, cid: str, spec: dict, extra_notes=()) -> dict:
    """C7 / C8: model mean of ``<protocol>_<column>`` against published mean +/- 1 SD, per protocol and row."""
    cfg = ctx.cfg
    if spec.get("rule") != "mean_within_sd":
        raise ValueError(f"consistency.checks.{cid}: unknown rule {spec.get('rule')!r}")
    protos = list(spec.get("protocols") or [spec["protocol"]])
    col = spec["column"]
    pubs = [published(r, cfg) for r in spec["published"]]
    usable = [p for p in pubs if p["ok"] and p["sd"] is not None]
    mq = f"<protocol>_{col} (mean, SD, range) for {', '.join(protos)}"
    crit = ("agrees if the model mean lies within published mean +/- 1 SD for every protocol and published row "
            "(" + "; ".join(f"{p['row']}: [{p['value'] - p['sd']:g}, {p['value'] + p['sd']:g}]" for p in usable)
            + "), partial if for some, disagrees if for none")
    quantities, labels, lesions = [], [], []
    notes = [f"{p['row']}: not used ({p['reason'] or 'no SD in the range cell'})" for p in pubs if p not in usable]
    for proto in protos:
        r, excl = _protocol_values(ctx.t, proto, col, cfg)
        notes.append(_excl_text(excl))
        if len(r) < cfg.min_n:
            labels.append(NOT_COMPUTED)
            notes.append(f"{proto}: fewer than {cfg.min_n} lesions ({len(r)})")
            continue
        n_lim = int((r[f"{proto}_status"] == "limited").sum()) if f"{proto}_status" in r else 0
        q = _summary_quantity(f"mean {proto}", r["_v"], "mean", n_limited=n_lim)
        inside = {}
        for p in usable:
            ok = p["value"] - p["sd"] <= q["value"] <= p["value"] + p["sd"]
            inside[p["row"]] = bool(ok)
            labels.append("agrees" if ok else "disagrees")
        q["within_published_sd"] = inside
        quantities.append(q)
        lesions.extend(_ids(r))
        if n_lim:
            notes.append(f"{proto}: {n_lim} pressure-limited runs included")
    if not usable:
        labels = [NOT_COMPUTED]
        notes.insert(0, "no published mean with SD readable")
    return _row(ctx, cid, spec, pubs, model_quantity=mq, criterion=crit, agreement=combine(labels),
                quantities=quantities, lesions=lesions, notes=list(extra_notes) + notes)


def check_stent_expansion(ctx: _Ctx, cid: str, spec: dict) -> dict:
    """C7: stent expansion after IVL against both published site definitions."""
    d_stent = P.get("geometry.stent_target_diameter_mm").value
    d_ref = P.get("geometry.reference_diameter_mm").value
    cap = PCT * (d_stent / d_ref) ** 2
    notes = [spec.get("site"),
             f"the deployment target (geometry.stent_target_diameter_mm {d_stent:g} mm, reference "
             f"geometry.reference_diameter_mm {d_ref:g} mm) caps the model expansion at {_g(cap)} % before recoil",
             "expansion includes the calibrated stent recoil (C8)"]
    return _mean_within_sd(ctx, cid, spec, notes)


def check_recoil(ctx: _Ctx, cid: str, spec: dict) -> dict:
    """C8: acute recoil of both stent protocols (calibration target, not an independent check)."""
    cal = ctx.extra.get("calibration")
    if cal:
        note = (f"calibration target, not an independent check: K_s = {_g(cal.get('K_s_MPa_per_mm'))} MPa/mm "
                f"was identified on the calcium-free version of {cal.get('lesion_template')} for a recoil of "
                f"{_g(cal.get('recoil_target_pct'))} % (results/calibration_stent.json, reached_target "
                f"{cal.get('reached_target')}); agreement is expected by construction")
    else:
        note = "calibration target, not an independent check (results/calibration_stent.json not found)"
    return _mean_within_sd(ctx, cid, spec, [note])


def check_stress(ctx: _Ctx, cid: str, spec: dict) -> dict:
    """C9: predictor stresses scaled to the 10 atm plain-balloon load against another FE study (model vs model)."""
    cfg, t = ctx.cfg, ctx.t
    cols, load_col, key, factor = list(spec["columns"]), spec["load_column"], spec["target_load"], float(spec["factor"])
    pubs = [published(r, cfg) for r in spec["published"]]
    if len(pubs) != 1:
        raise ValueError(f"consistency.checks.{cid}: exactly one published row expected, got {len(pubs)}")
    pub = pubs[0]
    p_t = P.get(key).value
    mq = f"median over lesions of <column> x p / {load_col} with p = {_load_text(key)}, for {', '.join(cols)} (MPa)"
    pv = f"{pub['value']:g} MPa" if pub["ok"] else "the published value"
    crit = (f"agrees if the median of every scaled stress lies within a factor {factor:g} of {pv}"
            + (f" (in [{pub['value'] / factor:g}, {pub['value'] * factor:g}] MPa)" if pub["ok"] else "")
            + ", partial if some, disagrees if none")
    notes = ["model vs model: the published value is itself a finite-element result (other geometry, materials "
             "and balloon model), not a measurement",
             f"the predictor model is linear elastic, so scaling its stresses from {load_col} to the target load "
             "is exact for that model (the damage model is nonlinear)"]
    if load_col not in t:
        return _row(ctx, cid, spec, pubs, model_quantity=mq, criterion=crit, agreement=NOT_COMPUTED,
                    notes=[f"column {load_col} missing from cohort_outcomes.csv"] + notes)
    load = pd.to_numeric(t[load_col], errors="coerce").to_numpy(float)
    quantities, labels, lesions = [], [], []
    for c in cols:
        if c not in t:
            labels.append(NOT_COMPUTED)
            notes.append(f"column {c} missing from cohort_outcomes.csv")
            continue
        s = pd.to_numeric(t[c], errors="coerce").to_numpy(float)
        m = np.isfinite(s) & np.isfinite(load) & (load > 0)
        if m.sum() < cfg.min_n:
            labels.append(NOT_COMPUTED)
            notes.append(f"{c}: fewer than {cfg.min_n} lesions ({int(m.sum())})")
            continue
        q = _summary_quantity(f"median {c} at target load", s[m] * p_t / load[m], "median", unit="MPa")
        if pub["ok"]:
            ok = pub["value"] / factor <= q["value"] <= pub["value"] * factor
            q["ratio_to_published"] = q["value"] / pub["value"] if pub["value"] else math.nan
            labels.append("agrees" if ok else "disagrees")
        quantities.append(q)
        lesions.extend(np.asarray(_ids(t))[m])
    if not pub["ok"]:
        labels = [NOT_COMPUTED]
        notes.insert(0, f"published value not read: {pub['reason']}")
    return _row(ctx, cid, spec, pubs, model_quantity=mq, criterion=crit, agreement=combine(labels),
                quantities=quantities, lesions=lesions, notes=notes)


def check_fracture_count(ctx: _Ctx, cid: str, spec: dict) -> dict:
    """C10: fractures per fractured lesion and full-thickness fraction of the crack components under IVL."""
    cfg, proto = ctx.cfg, spec["protocol"]
    cs, fs = spec["count"], spec["fraction"]
    for part, rule in (("count", cs.get("rule")), ("fraction", fs.get("rule"))):
        if rule != {"count": "mean_within_sd", "fraction": "published_in_ci"}[part]:
            raise ValueError(f"consistency.checks.{cid}.{part}: unknown rule {rule!r}")
    pc, pf = published(cs["published"], cfg), published(fs["published"], cfg)
    pubs = [pc, pf]
    nf_col, np_col, dif_col = f"{proto}_n_fractures", f"{proto}_n_partial", f"{proto}_diffuse"
    mq = (f"{nf_col} over the lesions with {proto}_fracture (mean, SD, range); full-thickness fraction "
          f"sum {nf_col} / sum ({nf_col} + {np_col}) over the lesions with any crack component (%, lesion-level "
          "bootstrap)")
    count_ok = pc["ok"] and pc["sd"] is not None
    crit = ("count: agrees if the mean lies within published mean +/- 1 SD"
            + (f" ([{pc['value'] - pc['sd']:g}, {pc['value'] + pc['sd']:g}])" if count_ok else "")
            + f"; fraction: agrees if the published fraction lies inside the {cfg.level_text} interval; "
            "both: agrees, one: partial, none: disagrees (a part that cannot be computed is left out and named "
            "in the note)")
    r, excl = _protocol_values(ctx.t, proto, "fracture", cfg, binary=True)
    notes = [_excl_text(excl)]
    missing = [c for c in (nf_col, np_col) if c not in r]
    if missing:
        return _row(ctx, cid, spec, pubs, model_quantity=mq, criterion=crit, agreement=NOT_COMPUTED,
                    notes=[f"columns missing from cohort_outcomes.csv: {', '.join(missing)}"] + notes)
    nf = pd.to_numeric(r[nf_col], errors="coerce").to_numpy(float)
    npart = pd.to_numeric(r[np_col], errors="coerce").to_numpy(float)
    frac = r["_v"].to_numpy(float) == 1
    ids = np.asarray(_ids(r))
    quantities, labels, lesions = [], [], []
    # count among fractured lesions
    cnt = nf[frac & np.isfinite(nf)]
    if not count_ok:
        labels.append(NOT_COMPUTED)
        notes.append(f"count: published value not read ({pc['reason'] or 'no SD'})")
    if cnt.size < cfg.min_n:
        labels.append(NOT_COMPUTED)
        notes.append(f"count: fewer than {cfg.min_n} fractured lesions ({cnt.size})")
    else:
        q = _summary_quantity("fractures per fractured lesion", cnt, "mean")
        quantities.append(q)
        lesions.extend(ids[frac & np.isfinite(nf)])
        if count_ok:
            labels.append("agrees" if pc["value"] - pc["sd"] <= q["value"] <= pc["value"] + pc["sd"] else "disagrees")
    # full-thickness fraction over lesions with any component
    tot = nf + npart
    m = np.isfinite(tot) & (tot > 0)
    if not pf["ok"]:
        labels.append(NOT_COMPUTED)
        notes.append(f"fraction: published value not read ({pf['reason']})")
    if m.sum() < cfg.min_n:
        labels.append(NOT_COMPUTED)
        notes.append(f"fraction: fewer than {cfg.min_n} lesions with a crack component ({int(m.sum())})")
    else:
        a, b = nf[m], tot[m]
        idx = R.bootstrap_indices(R.rng_for(cfg.seed, f"{cid}|full_thickness|{ctx.set_}"), cfg.n_boot, a.size)
        lo, hi = R.percentile_ci(PCT * a[idx].sum(axis=1) / b[idx].sum(axis=1), cfg.level)
        val = PCT * a.sum() / b.sum()
        quantities.append({"name": "full-thickness fraction %", "value": val, "ci_lo": lo, "ci_hi": hi,
                           "ci_method": f"lesion-level percentile bootstrap, {cfg.n_boot} resamples, {cfg.level_text}",
                           "n": int(m.sum()), "n_components": int(b.sum()), "n_full_thickness": int(a.sum())})
        lesions.extend(ids[m])
        if pf["ok"]:
            labels.append("agrees" if lo <= pf["value"] <= hi else "disagrees")
    notes.append(f"{nf_col} counts the through-thickness crack components and {np_col} the others "
                 "(dtivl.fracture.cracks)")
    if dif_col in r:
        dif = _as_binary(r[dif_col])
        notes.append(f"{int(np.nansum(dif[frac]))} of {int(frac.sum())} fractured lesions carry diffuse damage "
                     f"({dif_col}: a crack component covering at least {DIFFUSE_REGION_FRAC:.0%} of its calcium "
                     "region); a through-thickness diffuse component counts as one fracture, so damage of a whole "
                     "deposit counts as a single fracture")
    notes.append("the 2D model counts the fractures of one cross-section; OCT counts them over the lesion length")
    return _row(ctx, cid, spec, pubs, model_quantity=mq, criterion=crit, agreement=combine(labels),
                quantities=quantities, lesions=lesions, notes=notes)


def check_fujino(ctx: _Ctx, cid: str, spec: dict) -> dict:
    """C11: higher cross-sectional Fujino score, lower stent expansion without IVL (direction only)."""
    cfg = ctx.cfg
    col, proto, outc, sign = spec["column"], spec["protocol"], spec["outcome"], int(spec["expected_sign"])
    pubs = [published(r, cfg) for r in spec["published"]]
    mq = f"Spearman rho of {col} and {proto}_{outc}"
    crit = f"rho {'<' if sign < 0 else '>'} {RHO_NULL:g}: {_rule_text(cfg, f'{RHO_NULL:g}')}"
    notes = []
    if col not in ctx.t:
        return _row(ctx, cid, spec, pubs, model_quantity=mq, criterion=crit, agreement=NOT_COMPUTED,
                    notes=[f"column {col} missing from cohort_outcomes.csv"])
    r, excl = _protocol_values(ctx.t, proto, outc, cfg)
    notes.append(_excl_text(excl))
    x = pd.to_numeric(r[col], errors="coerce").to_numpy(float)
    m = np.isfinite(x)
    x, y, ids = x[m], r["_v"].to_numpy(float)[m], np.asarray(_ids(r))[m]
    if x.size:
        notes.insert(0, f"cross-sectional scores in this set: {', '.join(f'{v:g}' for v in np.unique(x))}")
    coh = ctx.extra.get("cohort")
    fmax, src = math.nan, "cohort.csv fujino_max"
    if coh is not None and "fujino_max" in coh and len(ids):
        fmax = float(pd.to_numeric(coh.loc[coh["lesion_id"].isin(ids), "fujino_max"], errors="coerce").max())
    if not math.isfinite(fmax):
        fmax, src = float(fujino_max_score(False)), "dtivl.geometry.fujino"
    notes.insert(1, f"maximum attainable cross-sectional score {fmax:g} ({src}); the full score reaches "
                    f"{fujino_max_score(True)} only with the calcium length criterion, which a cross-section cannot "
                    "meet, so the published top-score group has no 2D counterpart: direction only")
    if x.size < cfg.min_n:
        return _row(ctx, cid, spec, pubs, model_quantity=mq, criterion=crit, agreement=NOT_COMPUTED,
                    lesions=ids, notes=[f"fewer than {cfg.min_n} lesions ({x.size})"] + notes)
    sp = spearman_ci(x, y, cfg, f"{cid}|rho|{ctx.set_}", binary_y=False)
    if not math.isfinite(sp["value"]):
        return _row(ctx, cid, spec, pubs, model_quantity=mq, criterion=crit, agreement=NOT_COMPUTED,
                    lesions=ids, notes=[f"rho undefined ({col} or {proto}_{outc} constant)"] + notes)
    lab = direction_label(sp["value"], sp["ci_lo"], sp["ci_hi"], RHO_NULL, sign, cfg.require_ci)
    by_score = {f"{s:g}": float(y[x == s].mean()) for s in np.unique(x)}
    quantities = [_quantity("rho", sp), {"name": f"mean {proto}_{outc} by score", "value": None,
                                          "by_score": by_score, "n": int(x.size)}]
    quantities[1]["name"] += " " + ", ".join(f"{k}: {_g(v)}" for k, v in by_score.items())
    return _row(ctx, cid, spec, pubs, model_quantity=mq, criterion=crit, agreement=lab, quantities=quantities,
                lesions=ids, notes=notes)


def check_plaque_burden(ctx: _Ctx, cid: str, spec: dict) -> dict:
    """C12: implied plaque burden against CTA, copied from results/cohort_summary.json (not recomputed)."""
    cfg = ctx.cfg
    pubs = [published(r, cfg) for r in spec["published"]]
    mq = ("implied plaque burden (%) of the reference geometry and range over the lesions grown outward (D11), "
          "copied from results/cohort_summary.json")
    vals = sorted(p["value"] for p in pubs if p["ok"])
    crit = ("agrees if the reference geometry lies within the CTA range"
            + (f" [{vals[0]:g}, {vals[-1]:g}] %" if vals else "")
            + " and no grown lesion exceeds its upper value, partial if the reference lies within the range but "
            "some grown lesions exceed it, disagrees if the reference lies outside")
    cs = ctx.extra.get("cohort_summary")

    def nc(reason):
        return _row(ctx, cid, spec, pubs, model_quantity=mq, criterion=crit, agreement=NOT_COMPUTED, notes=[reason])

    if not cs or "plaque_burden_check" not in cs:
        return nc("results/cohort_summary.json (plaque_burden_check) not found")
    if len(vals) != len(pubs) or not vals:
        return nc("published CTA values not readable")
    pb, pg = cs["plaque_burden_check"], cs.get("plaque_growth") or {}
    need = ("implied_plaque_burden_pct", "cta_calcified_plaque_pct", "cta_calcified_nodule_pct", "within_range")
    if any(pb.get(k) is None for k in need):
        return nc(f"plaque_burden_check in results/cohort_summary.json lacks {[k for k in need if pb.get(k) is None]}")
    copied = sorted([pb["cta_calcified_plaque_pct"], pb["cta_calcified_nodule_pct"]])
    if len(copied) != len(vals) or not all(math.isclose(a, b, rel_tol=COPY_REL_TOL) for a, b in zip(copied, vals)):
        return nc(f"CTA values in cohort_summary.json ({copied}) differ from the literature table ({vals}): "
                  "rerun dtivl.drivers.stage_cohort")
    within = bool(pb["within_range"])
    n_above = pg.get("n_grown_above_nodule_cta")
    if not within:
        lab = "disagrees"
    elif n_above is None:
        lab = "agrees"
    else:
        lab = "agrees" if int(n_above) == 0 else "partial"
    quantities = [{"name": "implied plaque burden, reference geometry %", "value": pb["implied_plaque_burden_pct"],
                   "ci_lo": None, "ci_hi": None, "ci_method": None, "n": None, "within_range": within}]
    rng_ = pg.get("plaque_burden_pct_range_grown")
    if rng_:
        quantities.append({"name": "implied plaque burden, grown lesions %", "value": None, "min": rng_[0],
                           "max": rng_[1], "n": int(sum((pg.get("n_grown") or {}).values()))})
    per_set = cs.get("n_per_set") or {}
    n_cohort = int(sum(int(per_set.get(s, 0)) for s in cfg.sets))
    notes = ["values copied from results/cohort_summary.json, not recomputed",
             f"the reference geometry is that of every lesion not grown outward; n lesions = the "
             f"{' + '.join(cfg.sets)} sets of cohort_summary.json n_per_set"]
    if pg:
        notes.append(f"grown lesions per set: {pg.get('n_grown')}; above the upper CTA value: {n_above}; maximum "
                     f"outward growth {_g(pg.get('max_grown_mm'))} mm")
    if n_above is None:
        notes.append("plaque_growth without n_grown_above_nodule_cta: grown lesions not checked" if pg else
                     "no plaque_growth block in cohort_summary.json: no lesion grown outward")
    row = _row(ctx, cid, spec, pubs, model_quantity=mq, criterion=crit, agreement=lab, quantities=quantities,
               notes=notes)
    row["n_lesions"] = n_cohort
    return row


#: check id -> function (ids and order of the rows come from consistency.checks)
CHECKS = {"C1": check_arc, "C2": check_thickness, "C3": check_nodule, "C4": check_fracture_rate,
          "C5": check_no_fracture, "C6": check_no_fracture, "C7": check_stent_expansion, "C8": check_recoil,
          "C9": check_stress, "C10": check_fracture_count, "C11": check_fujino, "C12": check_plaque_burden}


# =========================================================================== stage
def check_wording(text: str) -> None:
    """Refuse output text that uses the forbidden word stem (study rule 3)."""
    if FORBIDDEN_STEM in text.lower():
        raise ValueError(f"consistency output contains the word stem {FORBIDDEN_STEM!r} (study rule 3)")


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if Path(path).exists() else None


def _software() -> dict:
    import scipy

    return {"numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__}


def evaluate(df: pd.DataFrame, cfg: ConsistencyConfig, extra: dict) -> list[dict]:
    """Every configured check on every set of ``df`` (cohort-level checks once); returns the rows."""
    rows = []
    for cid, spec in cfg.checks.items():
        fn = CHECKS[cid]
        if cid in COHORT_LEVEL_CHECKS:
            rows.append(fn(_Ctx(cfg, "cohort", df, extra), cid, spec))
            continue
        for set_ in cfg.sets:
            t = df[df["set"] == set_] if "set" in df else df.iloc[0:0]
            row = fn(_Ctx(cfg, set_, t, extra), cid, spec)
            if len(t) == 0:              # every check needs lesions; state the set-level reason first
                row["agreement"] = NOT_COMPUTED
                row["note"] = "; ".join(filter(None, [f"no '{set_}' rows in cohort_outcomes.csv", row["note"]]))
            rows.append(row)
    return rows


def run(outcomes_csv: str | Path | None = None, out_dir: str | Path | None = None, *,
        n_boot: int | None = None) -> dict:
    """Run every check and write ``consistency.csv`` and ``consistency.json``; returns the summary."""
    cfg = load_config(n_boot)
    path = Path(outcomes_csv or OUTCOMES_CSV)
    df = load_outcomes(path)
    n_in = len(df)
    input_notes = []
    base_E = P.get("materials.calcium.E").value
    if "ca_E_MPa" in df:
        same = (pd.to_numeric(df["ca_E_MPa"], errors="coerce") - base_E).abs() <= MODULUS_TOL_MPA
        if (~same).any():
            input_notes.append(f"{int((~same).sum())} rows not at the baseline calcium modulus excluded")
        df = df[same]
    extra = {"cohort_summary": load_cohort_summary(), "calibration": load_calibration(), "cohort": load_cohort()}
    rows = evaluate(df, cfg, extra)
    table = pd.DataFrame(rows).reindex(columns=CSV_COLUMNS)
    counts: dict = {}
    by_kind: dict = {}
    for r in rows:
        counts.setdefault(r["set"], {k: 0 for k in AGREEMENT})[r["agreement"]] += 1
        by_kind.setdefault(r["set"], {}).setdefault(r["kind"], {k: 0 for k in AGREEMENT})[r["agreement"]] += 1
    lit_rows = sorted({p["row"] for r in rows for p in r["published_rows"]})
    summary = {
        "stage": "consistency with published observations (Phase 5)",
        "wording": WORDING,
        "input": {"path": path, "sha256": _sha256(path), "n_rows": n_in, "n_rows_used": int(len(df)),
                  "by_set": {str(k): int(v) for k, v in (df["set"].value_counts().items() if "set" in df else [])},
                  "baseline_ca_E_MPa": base_E, "notes": input_notes,
                  "cohort_summary": {"path": COHORT_SUMMARY_JSON, "found": extra["cohort_summary"] is not None},
                  "calibration": {"path": CALIBRATION_JSON, "found": extra["calibration"] is not None},
                  "cohort": {"path": COHORT_CSV, "found": extra["cohort"] is not None}},
        "config": {"seed": cfg.seed, "n_boot": cfg.n_boot, "bootstrap_stratified": cfg.stratified,
                   "ci_level": cfg.level, "sets": list(cfg.sets), "derivation_set": cfg.derivation_set,
                   "ok_status": cfg.ok_status, "allow_status": list(cfg.allow_status), "min_n": cfg.min_n,
                   "min_per_group": cfg.min_per_group, "require_ci": cfg.require_ci,
                   "odds_ratio_correction": cfg.or_correction, "provenance": cfg.provenance,
                   "consistency_section": P.raw("consistency")},
        "software": _software(),
        "literature": {name: published(name, cfg) for name in lit_rows},
        "agreement_counts": counts,
        "agreement_counts_by_kind": by_kind,
        "not_computed": [{"id": r["id"], "set": r["set"], "reason": r["note"]} for r in rows
                         if r["agreement"] == NOT_COMPUTED],
        "checks": rows,
    }
    summary = _clean(_to_jsonable(summary))
    blob = json.dumps(summary, indent=1)
    csv_text = table.to_csv(index=False)
    check_wording(blob)
    check_wording(csv_text)
    if out_dir is None:
        write_json("consistency", summary)
        table.to_csv(RESULTS_DIR / "consistency.csv", index=False)
    else:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "consistency.json").write_text(blob)
        table.to_csv(out_dir / "consistency.csv", index=False)
    return summary


if __name__ == "__main__":
    run()

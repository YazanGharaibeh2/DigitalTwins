"""Stage (Phase 5): predictive metrics of the pre-treatment predictors for the simulated outcomes.

Input: ``results/cohort_outcomes.csv`` from
:func:`dtivl.drivers.stage_cohort_runs.aggregate` (one row per lesion: cohort
descriptors incl. ``set``, the linear-analysis predictors ``pred_*``, the IVL
load factor ``ivl_load_*`` and the outcomes ``<protocol>_*`` of the nonlinear
damage runs). Configuration: ``params/study_config.yaml`` section ``metrics``
and ``design.composite_index_name`` (every setting has status DESIGN).

Non-circularity (study rule 4): predictors are only the configured
``metrics.candidates`` (linear-analysis ``pred_*`` columns, cohort geometry
descriptors and the Fujino score); :func:`check_predictors` refuses the IVL
load factor (``ivl_load_*``: T and T_equivalent are inputs of the outcome
model) and any outcome column. Outcomes come only from the nonlinear runs.
The right-censoring value of the onset-load outcome is the peak IVL pressure
the lesion saw, ``ivl_p_top`` as stored by its IVL run;
``ivl_load_T_equivalent`` is read only to check that value against the
configured IVL load (and to recompute it where ``ivl_p_top`` is absent).

Analyses (all choices on the core set; the hold-out set only evaluates them):

(a) Univariable ROC on the core set for every candidate and binary outcome:
    AUC with a stratified bootstrap interval, Youden-optimal threshold with a
    bootstrap interval, sensitivity / specificity at it (Wilson intervals), and
    the leave-one-out estimate of the threshold rule (threshold re-derived
    without the left-out lesion). Directions from ``metrics.direction``
    (see :mod:`dtivl.stats.roc` for the threshold conventions). Zero-width
    AUC intervals (perfect separation, or every resample equal) are flagged
    ``auc_ci_degenerate``.
(b) The core thresholds applied unchanged to the hold-out set: sensitivity,
    specificity and accuracy with Wilson intervals, and the hold-out AUC.
(c) Composite index (name ``design.composite_index_name``): L2-penalised
    logistic regression on <= 3 mechanics predictors chosen by the rule of
    ``metrics.composite`` (:mod:`dtivl.stats.models`; ranked by the AUC
    oriented by ``metrics.direction``), with the rule repeated in every LOOCV
    fold; apparent and LOOCV AUC, log-loss and Brier score, selection
    frequencies, coefficients (selected predictors whose coefficient sign
    opposes their direction are listed), a Youden threshold on the core-fit
    probability (apparent and hold-out classification) and the LOOCV
    threshold rule (Youden cut re-derived in each fold on the training rows'
    fitted probabilities, applied to the left-out lesion), and the hold-out
    evaluation of the core fit. The LOOCV AUC interval is a bootstrap of the
    fixed out-of-fold predictions (refit and reselection variability not
    included).
(d) Comparison with the Fujino score (composite LOOCV, best single mechanics
    predictor, Fujino: AUC, paired bootstrap AUC differences and threshold
    performance on core and hold-out) and the incremental value of mechanics
    over geometry (``metrics.incremental``: geometry-only vs geometry +
    mechanics models; apparent AUC difference with both models refitted in
    every resample, LOOCV AUC / log-loss / Brier differences and hold-out
    differences, each with a paired bootstrap interval).

Secondary outcomes: binary ones get (a) and (b); the censored onset load
(event = local damage onset recorded, value = the onset load, or the peak
load for an onset after the peak; otherwise censored at the lesion's own
``ivl_p_top``; onsets without through-thickness damage are counted, not
excluded) gets Harrell's C per candidate with the a priori direction (a
higher oriented predictor should mean an earlier onset; ``core``-direction
predictors are flagged) and bootstrap intervals. Stent expansion has no a
priori direction (exploratory): Harrell's C of the raw predictor (higher
predictor, higher expansion), its observed sign, the discrimination
``max(C, 1 - C)`` used for the ranking (hold-out: C oriented by the core
sign) and Spearman's rho of the raw predictor. The composite index fitted for
the primary outcome is also scored against each secondary outcome (LOOCV
values on the core set, core-fit values on the hold-out set).

Analyses that cannot be computed (missing hold-out rows, a single outcome
class, constant predictors, too few rows) are listed in
``metrics_summary.json["not_computed"]`` with the reason, and the
corresponding table rows carry a ``note``. Rows without a recorded calcium
modulus, or not at the baseline modulus, are excluded (counted in the summary).

Outputs (``results/``): ``metrics_roc.csv`` (one row per outcome x set x
predictor), ``metrics_roc_curves.csv`` (empirical core ROC points of the
binary analyses) and ``metrics_summary.json``. Hold-out agreement is
described as consistency of core-derived rules with lesions not used for any
choice; no result here is a comparison with patients.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as _st

from dtivl import params as P
from dtivl.drivers.common import RESULTS_DIR, _to_jsonable
from dtivl.stats import models as M
from dtivl.stats import roc as R

OUTCOMES_CSV = RESULTS_DIR / "cohort_outcomes.csv"
#: pred_* columns written by the cohort driver that are not predictors (load and run time).
NON_PREDICTOR_PRED_COLUMNS = ("pred_p_lumen_MPa", "pred_seconds")
#: NUMERICAL: tolerance (MPa) when checking that every row is at the baseline calcium modulus.
MODULUS_TOL_MPA = 1e-6
#: NUMERICAL: tolerance (MPa) between the stored IVL peak load ivl_p_top and the value from the configured loads.
P_TOP_TOL_MPA = 1e-9
#: event rule of the censored onset outcome: the onset load column is recorded (finite)
EVENT_ONSET_RECORDED = "onset_recorded"
#: IVL loading phases (as stage_sensitivity.IVL_LOADING_PHASES); an onset in a later phase happened after the peak
IVL_LOADING_PHASES = ("ivl_inflate", "ivl_pulse")
OOF_CI_METHOD = "bootstrap of fixed out-of-fold predictions (refit and reselection variability not included)"
DEGENERATE_NOTE = ("{what} interval degenerate (value 0 or 1, or every resample equal): zero-width percentile "
                   "interval, not a precision statement")
CENSORED_CONVENTION = (
    "Harrell's C: probability that, of two comparable lesions, the one with the higher oriented predictor "
    "(a priori direction x value; 'core' directions set on the core set, flagged) has the earlier onset. "
    "Event = local damage onset recorded (first load with max d >= fracture.d_crack; a partial crack without "
    "through-thickness damage is an event), value = the onset load (the peak load ivl_p_top for an onset after the "
    "peak, as in stage_sensitivity); no onset = right-censored at the lesion's own ivl_p_top. observed_sign = "
    "direction x sign(C - 0.5): +1 when a higher raw predictor value went with the earlier onset, -1 when a lower "
    "one did")
CONTINUOUS_CONVENTION = (
    "Exploratory (direction not pre-specified): Harrell's C of the raw predictor, the probability that, of two "
    "lesions, the one with the higher predictor has the higher outcome; observed_sign = sign(C - 0.5); "
    "discrimination = max(C, 1 - C) on the core set (ranking), C oriented by the core sign on the hold-out set; "
    "spearman = Spearman's rho of the raw predictor and the outcome (positive: higher predictor, higher outcome)")

ROC_COLUMNS = [
    "outcome", "role", "kind", "set", "predictor", "group", "n", "n_pos", "n_neg", "direction", "direction_source",
    "auc", "auc_ci_lo", "auc_ci_hi", "auc_ci_degenerate", "threshold", "threshold_ci_lo", "threshold_ci_hi", "rule",
    "youden_j", "sens", "sens_ci_lo", "sens_ci_hi", "spec", "spec_ci_lo", "spec_ci_hi", "acc", "acc_ci_lo",
    "acc_ci_hi", "loocv_auc", "loocv_auc_ci_lo", "loocv_auc_ci_hi", "loocv_auc_ci_degenerate", "loocv_log_loss",
    "loocv_brier", "loocv_sens", "loocv_sens_ci_lo", "loocv_sens_ci_hi", "loocv_spec", "loocv_spec_ci_lo",
    "loocv_spec_ci_hi", "loocv_acc", "c_index", "c_index_ci_lo", "c_index_ci_hi", "observed_sign",
    "discrimination", "n_pairs", "spearman", "spearman_ci_lo", "spearman_ci_hi", "n_boot", "note",
]


# =========================================================================== configuration
def _design_value(dotted: str):
    """Raw value of a ``{value, basis, status}`` config entry (refuses PENDING entries)."""
    node = P.raw(dotted)
    if isinstance(node, dict) and "value" in node:
        if node.get("status") == "PENDING":
            raise ValueError(f"config entry {dotted} is PENDING")
        return node["value"]
    return node


@dataclass(frozen=True)
class MetricsConfig:
    """Resolved ``metrics`` configuration (see params/study_config.yaml)."""

    seed: int
    n_boot: int
    stratified: bool
    level: float
    min_per_class: int
    min_rows_continuous: int
    C: float
    derivation_set: str
    test_set: str
    groups: dict
    forbidden_prefixes: tuple
    direction_default: object
    direction_overrides: dict
    outcomes: tuple
    composite: dict
    incremental: dict
    index_name: str
    index_expansion: str
    provenance: dict = field(default_factory=dict)

    def direction(self, pred: str):
        """Configured direction of ``pred`` (+1, -1 or ``"core"``)."""
        d = self.direction_overrides.get(pred, self.direction_default)
        return d if d == "core" else int(d)

    def expand(self, names) -> list[str]:
        """Group names (``mechanics``, ``geometry``, ``comparator``) or columns -> column list."""
        out = []
        for n in names:
            out.extend(self.groups[n] if n in self.groups else [n])
        return list(dict.fromkeys(out))

    @property
    def candidates(self) -> list[tuple[str, str]]:
        """``(column, group)`` of every candidate predictor, in configuration order."""
        return [(c, g) for g, cols in self.groups.items() for c in cols]


def load_config(n_boot: int | None = None) -> MetricsConfig:
    """Read the ``metrics`` section (``n_boot`` overrides ``metrics.bootstrap_n``, e.g. for tests)."""
    seed = int(P.raw("design.seed")) + int(P.get("metrics.seed_offset").value)
    nb = int(P.get("metrics.bootstrap_n").value) if n_boot is None else int(n_boot)
    sets = _design_value("metrics.sets")
    direction = _design_value("metrics.direction")
    name = P.raw("design.composite_index_name")
    groups = {k: list(v) for k, v in _design_value("metrics.candidates").items()}
    cfg = MetricsConfig(
        seed=seed, n_boot=nb, stratified=bool(_design_value("metrics.bootstrap_stratified")),
        level=float(P.get("metrics.ci_level").value), min_per_class=int(P.get("metrics.min_per_class").value),
        min_rows_continuous=int(P.get("metrics.min_rows_continuous").value),
        C=float(P.get("metrics.logistic_C").value), derivation_set=sets["derivation"], test_set=sets["test"],
        groups=groups, forbidden_prefixes=tuple(_design_value("metrics.forbidden_predictor_prefixes")),
        direction_default=direction["default"], direction_overrides=dict(direction.get("overrides", {})),
        outcomes=tuple(_design_value("metrics.outcomes")), composite=dict(_design_value("metrics.composite")),
        incremental=dict(_design_value("metrics.incremental")), index_name=str(name["value"]),
        index_expansion=str(name.get("expansion", "")),
        provenance={"seed": "design.seed + metrics.seed_offset", "bootstrap_n": "metrics.bootstrap_n"
                    + (" (overridden by the caller)" if n_boot is not None else ""),
                    "composite_rule": P.raw("metrics.composite")["basis"],
                    "incremental": P.raw("metrics.incremental")["basis"],
                    "direction": P.raw("metrics.direction")["basis"],
                    "outcomes": P.raw("metrics.outcomes")["basis"]})
    for d in [cfg.direction_default, *cfg.direction_overrides.values()]:
        if d not in (1, -1, "core"):
            raise ValueError(f"metrics.direction: invalid direction {d!r}")
    return cfg


# =========================================================================== data
def _as_binary(s: pd.Series) -> np.ndarray:
    """True/False (bool, 0/1 or the strings) -> 1.0/0.0; anything else -> nan."""
    def conv(v):
        if isinstance(v, (bool, np.bool_)):
            return float(v)
        if isinstance(v, (int, float, np.integer, np.floating)):
            return float(v) if v in (0, 1) else math.nan
        if isinstance(v, str) and v.strip().lower() in ("true", "false", "1", "0"):
            return 1.0 if v.strip().lower() in ("true", "1") else 0.0
        return math.nan
    return np.array([conv(v) for v in s], dtype=float)


def _protocols(df: pd.DataFrame, cfg: MetricsConfig) -> set[str]:
    found = {c[: -len("_status")] for c in df.columns if c.endswith("_status")}
    return found | {o["protocol"] for o in cfg.outcomes}


def check_predictors(cfg: MetricsConfig, df: pd.DataFrame | None = None) -> None:
    """Refuse candidate predictors that are the IVL load factor, an outcome or not a pre-treatment quantity."""
    protos = _protocols(df if df is not None else pd.DataFrame(), cfg)
    outcome_cols = {o[k] for o in cfg.outcomes for k in ("column", "fracture_column") if o.get(k)}
    bad = []
    for c, g in cfg.candidates:
        if any(c.startswith(p) for p in cfg.forbidden_prefixes):
            bad.append((c, "IVL load factor / forbidden prefix"))
        elif "T_equivalent" in c or c in outcome_cols or any(c.startswith(f"{p}_") for p in protos):
            bad.append((c, "outcome-model column"))
        elif c in NON_PREDICTOR_PRED_COLUMNS:
            bad.append((c, "load or timing record, not a predictor"))
        elif g == "mechanics" and not c.startswith("pred_"):
            bad.append((c, "mechanics predictors must be linear-analysis pred_* columns"))
    if bad:
        raise ValueError(f"metrics.candidates contains columns that may not be predictors: {bad}")


def load_outcomes(path: str | Path | None = None, *, refresh: bool = False) -> pd.DataFrame:
    """Read the per-lesion table (``refresh=True`` rebuilds it with ``stage_cohort_runs.aggregate()``)."""
    if refresh:
        from dtivl.drivers.stage_cohort_runs import aggregate  # heavy import (FE stack) only when asked

        aggregate()
    path = Path(path or OUTCOMES_CSV)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run dtivl.drivers.stage_cohort_runs.aggregate() first "
                                "(or call run(..., refresh=True))")
    return pd.read_csv(path)


# =========================================================================== bookkeeping
class _Collector:
    def __init__(self):
        self.rows: list[dict] = []
        self.curves: list[dict] = []
        self.not_computed: list[dict] = []

    def nc(self, outcome: str, analysis: str, reason: str, set_: str | None = None, predictor: str | None = None):
        self.not_computed.append({k: v for k, v in (("outcome", outcome), ("set", set_), ("predictor", predictor),
                                                    ("analysis", analysis), ("reason", reason)) if v is not None})

    def row(self, **kw):
        self.rows.append(kw)


def _counts(y) -> dict:
    y = np.asarray(y)
    return {"n": int(y.size), "n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum())}


def _rule(direction: int, var: str = "x") -> str:
    return f"{var} >= thr" if direction == 1 else f"{var} <= thr"


# =========================================================================== outcome extraction
def _outcome_table(df: pd.DataFrame, oc: dict, cfg: MetricsConfig) -> tuple[pd.DataFrame, dict]:
    """Rows of ``df`` with a usable outcome; adds ``_y`` (binary) or ``_value``/``_event``. Returns (table, exclusions).

    Censored onset (``event: onset_recorded``): event = the onset load (first
    load with max d >= fracture.d_crack, a local criterion) is recorded, value
    = that load; otherwise the row is censored at its own peak IVL load
    (:func:`_censor_value`). A partial crack (onset without through-thickness
    damage) is therefore an event; ``_through_thickness`` keeps the
    ``fracture_column`` flag for the summary counts. An onset recorded after
    the peak (``<column>_phase`` not an IVL loading phase) is an event at the
    peak load (``_after_peak``), as in ``stage_sensitivity.censored_onset``.
    Rows with through-thickness damage but no recorded onset are inconsistent
    records and are excluded. ``exclusions["missing_outcome"]`` is the total
    of rows dropped after the status filter; the other keys break it down.
    """
    st_col = f"{oc['protocol']}_status"
    excl = {}
    status = df[st_col] if st_col in df else pd.Series([None] * len(df), index=df.index)
    ok = status.isin(oc.get("ok_status", ["ok"]))
    excl["status_not_ok"] = {str(k): int(v) for k, v in status[~ok].fillna("not run").value_counts().items()}
    t = df[ok].copy()
    if oc["column"] not in t:
        excl["missing_outcome"] = int(len(t))
        return t.iloc[0:0], excl
    if oc["kind"] == "binary":
        t["_y"] = _as_binary(t[oc["column"]])
        keep = np.isfinite(t["_y"].to_numpy())
    elif oc["kind"] == "censored":
        if oc.get("event") != EVENT_ONSET_RECORDED:
            raise ValueError(f"unknown event rule {oc.get('event')!r} (expected {EVENT_ONSET_RECORDED!r})")
        onset = pd.to_numeric(t[oc["column"]], errors="coerce").to_numpy(float)
        ev = np.isfinite(onset).astype(float)
        fc = oc.get("fracture_column")
        tt = _as_binary(t[fc]) if fc and fc in t else np.full(len(t), math.nan)
        ph_col = f"{oc['column']}_phase"
        ph = t[ph_col] if ph_col in t else pd.Series([None] * len(t), index=t.index)
        # an onset whose phase is missing counts as after the peak (value = peak load), as in stage_sensitivity
        after = (ev == 1) & ~ph.isin(IVL_LOADING_PHASES).to_numpy()
        excl["onset_phase_missing"] = int(((ev == 1) & ph.isna().to_numpy()).sum())
        cens, src = _censor_value(t, oc)
        need_peak = (ev == 0) | after                  # rows whose value is the peak load
        value = np.where(need_peak, cens, onset)
        bad = (tt == 1) & (ev == 0)
        t["_event"], t["_value"], t["_through_thickness"], t["_after_peak"] = ev, value, tt, after
        t["_censor_source"] = np.where(need_peak, src, None)
        keep = np.isfinite(value) & ~bad
        excl["through_thickness_without_onset"] = int(bad.sum())
        excl["peak_load_missing"] = int((need_peak & ~bad & ~np.isfinite(cens)).sum())
    elif oc["kind"] == "continuous":
        t["_value"] = pd.to_numeric(t[oc["column"]], errors="coerce")
        t["_event"] = 1.0
        keep = np.isfinite(t["_value"].to_numpy(float))
    else:
        raise ValueError(f"unknown outcome kind {oc['kind']!r}")
    excl["missing_outcome"] = int((~keep).sum())
    return t[keep], excl


def _censor_value(t: pd.DataFrame, oc: dict) -> tuple[np.ndarray, np.ndarray]:
    """Right-censoring load of the onset outcome (outcome side only; never a predictor).

    The peak IVL load ``ivl_p_top`` stored by each lesion's IVL run, checked
    against the configured load ``p_bal + T_equivalent x ivl_peak_pressure_MPa
    x ivl_pulse_scale`` (a difference above :data:`P_TOP_TOL_MPA` raises: the
    runs used other loads than the configuration). The configured value is used
    only where ``ivl_p_top`` is absent. Returns (value, source per row).
    """
    if oc.get("censor") != "ivl_peak_pressure":
        raise ValueError(f"unknown censoring rule {oc.get('censor')!r}")
    nan = np.full(len(t), math.nan)
    T = pd.to_numeric(t["ivl_load_T_equivalent"], errors="coerce").to_numpy(float) if "ivl_load_T_equivalent" in t \
        else nan
    p_bal = P.get("loads.ivl_balloon_pressure_atm").value
    p_peak = P.get("loads.ivl_peak_pressure_MPa").value
    scale = P.get("loads.ivl_pulse_scale").value
    p_cfg = p_bal + T * p_peak * scale
    stored = pd.to_numeric(t["ivl_p_top"], errors="coerce").to_numpy(float) if "ivl_p_top" in t else nan
    off = np.isfinite(stored) & np.isfinite(p_cfg) & (np.abs(stored - p_cfg) > P_TOP_TOL_MPA)
    if off.any():
        ids = t["lesion_id"].to_numpy()[off].tolist() if "lesion_id" in t else np.flatnonzero(off).tolist()
        raise ValueError(f"stored ivl_p_top differs from the configured IVL peak load by more than {P_TOP_TOL_MPA} "
                         f"MPa for {ids}: the runs used other loads than params/study_config.yaml")
    has, cfg_ok = np.isfinite(stored), np.isfinite(p_cfg)
    src = np.where(has, np.where(cfg_ok, "ivl_p_top", "ivl_p_top (not checked: T_equivalent missing)"),
                   np.where(cfg_ok, "configured load (ivl_p_top absent)", "none"))
    return np.where(has, stored, p_cfg), src


# =========================================================================== binary outcomes
def _holdout_binary(base: dict, y, x, s: int, thr: float, cfg: MetricsConfig, key: str, var: str = "x") -> dict:
    """Evaluate a core-derived rule unchanged on hold-out rows; returns the table row (reasons in ``note``).

    Set-level gaps (no hold-out rows, a single class) are reported once per
    outcome by the caller, not here.
    """
    c = _counts(y)
    row = {**base, "set": cfg.test_set, **c, "direction": s, "threshold": thr, "rule": _rule(s, var)}
    if c["n"] == 0:
        row["note"] = "no hold-out rows with this outcome and predictor"
        return row
    notes = []
    if math.isfinite(thr):
        perf = R.threshold_performance(y, R.classify(x, thr, s), cfg.level)
        row.update({k: perf[k] for k in perf if k.split("_")[0] in ("sens", "spec", "acc")})
        if c["n_pos"] == 0:
            notes.append("no hold-out events: sensitivity undefined")
        if c["n_neg"] == 0:
            notes.append("no hold-out non-events: specificity undefined")
    else:
        notes.append("no core threshold")
    if c["n_pos"] >= cfg.min_per_class and c["n_neg"] >= cfg.min_per_class:
        idx = R.bootstrap_indices(R.rng_for(cfg.seed, key), cfg.n_boot, len(y), y, cfg.stratified)
        z = s * np.asarray(x, float)
        boot = R.auc_rows(np.asarray(y)[idx], z[idx])
        row["auc"] = R.auc_score(y, z)
        row["auc_ci_lo"], row["auc_ci_hi"] = R.percentile_ci(boot, cfg.level)
        row["auc_ci_degenerate"] = R.ci_degenerate(boot, row["auc"])
        if row["auc_ci_degenerate"]:
            notes.append(DEGENERATE_NOTE.format(what="AUC"))
        row["n_boot"] = cfg.n_boot
    else:
        notes.append(f"hold-out AUC not computed (fewer than {cfg.min_per_class} lesions in a class)")
    row["note"] = "; ".join(notes) or None
    return row


def _univariable_binary(col, name, oc, core, hold, cfg) -> dict:
    """Analyses (a) and (b) for every candidate; returns {predictor: core summary}."""
    out = {}
    for pred, group in cfg.candidates:
        base = {"outcome": name, "role": oc["role"], "kind": "binary", "predictor": pred, "group": group}
        if pred not in core.columns:
            col.nc(name, "univariable ROC", "column missing from cohort_outcomes.csv", cfg.derivation_set, pred)
            col.row(**base, set=cfg.derivation_set, note="column missing")
            continue
        x = pd.to_numeric(core[pred], errors="coerce").to_numpy(float)
        m = np.isfinite(x)
        y, x = core["_y"].to_numpy(int)[m], x[m]
        c = _counts(y)
        if c["n_pos"] < cfg.min_per_class or c["n_neg"] < cfg.min_per_class:
            reason = f"fewer than {cfg.min_per_class} lesions in an outcome class among rows with this predictor"
            col.nc(name, "univariable ROC", reason, cfg.derivation_set, pred)
            col.row(**base, set=cfg.derivation_set, **c, note=reason)
            continue
        if np.ptp(x) == 0:
            reason = "predictor constant on the core set (all scores tied, no threshold)"
            col.nc(name, "univariable ROC", reason, cfg.derivation_set, pred)
            col.row(**base, set=cfg.derivation_set, **c, auc=R.auc_score(y, x), note=reason)
            continue
        s, src = R.resolve_direction(y, x, cfg.direction(pred))
        r = R.roc_analysis(y, x, s, n_boot=cfg.n_boot, rng=R.rng_for(cfg.seed, f"{name}|{pred}|core"),
                           level=cfg.level, stratified=cfg.stratified)
        perf = R.threshold_performance(y, R.classify(x, r["threshold"], s), cfg.level)
        lo = R.loocv_threshold(y, x, cfg.direction(pred), cfg.level)
        notes = []
        if src == "core":
            notes.append("direction set on the core set (not pre-specified)")
        if not r["youden_j"] > 0:
            notes.append("no discrimination at any cut (Youden J <= 0)")
        if r["auc_ci_degenerate"]:
            notes.append(DEGENERATE_NOTE.format(what="AUC"))
        row = {**base, "set": cfg.derivation_set, **c, "direction": s, "direction_source": src,
               "auc": r["auc"], "auc_ci_lo": r["auc_ci_lo"], "auc_ci_hi": r["auc_ci_hi"],
               "auc_ci_degenerate": r["auc_ci_degenerate"], "threshold": r["threshold"],
               "threshold_ci_lo": r["threshold_ci_lo"], "threshold_ci_hi": r["threshold_ci_hi"], "rule": _rule(s),
               "youden_j": r["youden_j"],
               **{k: perf[k] for k in perf if k.split("_")[0] in ("sens", "spec", "acc")},
               "loocv_sens": lo["sens"], "loocv_sens_ci_lo": lo["sens_ci_lo"], "loocv_sens_ci_hi": lo["sens_ci_hi"],
               "loocv_spec": lo["spec"], "loocv_spec_ci_lo": lo["spec_ci_lo"], "loocv_spec_ci_hi": lo["spec_ci_hi"],
               "loocv_acc": lo["acc"], "n_boot": cfg.n_boot, "note": "; ".join(notes) or None}
        col.row(**row)
        pts = R.roc_points(y, s * x)
        for cut, fpr, tpr in zip(pts["cut"], pts["fpr"], pts["tpr"]):
            col.curves.append({"outcome": name, "set": cfg.derivation_set, "predictor": pred, "group": group,
                               "direction": s, "cut_oriented": cut, "threshold": s * cut, "fpr": fpr, "tpr": tpr})
        out[pred] = {"group": group, "auc": r["auc"], "auc_ci": [r["auc_ci_lo"], r["auc_ci_hi"]],
                     "auc_ci_degenerate": r["auc_ci_degenerate"], "threshold": r["threshold"],
                     "threshold_ci": [r["threshold_ci_lo"], r["threshold_ci_hi"]], "rule": _rule(s), "direction": s,
                     "direction_source": src, "sens": perf["sens"], "spec": perf["spec"], "loocv_sens": lo["sens"],
                     "loocv_spec": lo["spec"], "n": c["n"]}
        # (b) hold-out
        xh = pd.to_numeric(hold[pred], errors="coerce").to_numpy(float) if pred in hold else np.array([])
        mh = np.isfinite(xh)
        yh = hold["_y"].to_numpy(int)[mh] if len(xh) else np.array([], int)
        hrow = _holdout_binary(base, yh, xh[mh], s, r["threshold"], cfg, f"{name}|{pred}|holdout")
        hrow["direction_source"] = src
        col.row(**hrow)
        out[pred]["holdout"] = {k: hrow.get(k) for k in ("n", "n_pos", "n_neg", "sens", "spec", "acc", "auc",
                                                         "auc_ci_lo", "auc_ci_hi", "auc_ci_degenerate")}
    return out


def _model_matrix(t: pd.DataFrame, cols) -> pd.DataFrame:
    return t[[c for c in cols if c in t.columns]].apply(pd.to_numeric, errors="coerce")


def _complete_rows(t: pd.DataFrame, cols) -> np.ndarray:
    if not cols:
        return np.ones(len(t), bool)
    return np.isfinite(_model_matrix(t, cols).to_numpy(float)).all(axis=1)


def _oof_auc_ci(y, p, cfg, key) -> tuple[float, float, bool]:
    """Interval of the LOOCV AUC: :data:`OOF_CI_METHOD`; returns (lo, hi, degenerate)."""
    idx = R.bootstrap_indices(R.rng_for(cfg.seed, key), cfg.n_boot, len(y), y, cfg.stratified)
    boot = R.auc_rows(np.asarray(y)[idx], np.asarray(p)[idx])
    return (*R.percentile_ci(boot, cfg.level), R.ci_degenerate(boot, R.auc_score(y, p)))


def _paired_scores(y, pa, pb, idx, level) -> dict:
    """Paired bootstrap of AUC, log-loss and Brier differences (a - b) for fixed probabilities."""
    y = np.asarray(y).astype(int)
    pa = np.clip(np.asarray(pa, float), M.PROB_CLIP, 1 - M.PROB_CLIP)
    pb = np.clip(np.asarray(pb, float), M.PROB_CLIP, 1 - M.PROB_CLIP)
    out = M.paired_bootstrap_fixed(y, pa, pb, idx, level)

    def ll(p, yy):
        return -(yy * np.log(p) + (1 - yy) * np.log(1 - p)).mean(axis=-1)

    def br(p, yy):
        return ((p - yy) ** 2).mean(axis=-1)

    for nm, f in (("log_loss", ll), ("brier", br)):
        a, b = f(pa, y), f(pb, y)
        d = f(pa[idx], y[idx]) - f(pb[idx], y[idx])
        out[f"{nm}_a"], out[f"{nm}_b"], out[f"delta_{nm}"] = float(a), float(b), float(a - b)
        out[f"delta_{nm}_ci_lo"], out[f"delta_{nm}_ci_hi"] = R.percentile_ci(d, level)
    return out


def _selection_frequency(selections, n: int) -> dict:
    freq = {}
    for s in selections:
        for c in s:
            freq[c] = freq.get(c, 0) + 1
    return {k: v / n for k, v in sorted(freq.items(), key=lambda kv: -kv[1])}


def _specs(cfg: MetricsConfig, columns) -> dict:
    """The three model procedures (composite, geometry, geometry + mechanics) restricted to available columns."""
    comp = cfg.composite
    avail = set(columns)
    pool = [c for c in cfg.expand(comp["pool"]) if c in avail]
    geo = [c for c in cfg.expand(cfg.incremental["geometry_model"]) if c in avail]
    mpool = [c for c in cfg.expand(cfg.incremental["mechanics_pool"]) if c in avail]
    dirs = tuple((c, cfg.direction(c)) for c in dict.fromkeys(pool + mpool))   # oriented-AUC selection
    sel = {"max_predictors": int(comp["max_predictors"]), "max_abs_spearman": float(comp["max_abs_spearman"]),
           "min_discrimination": float(comp["min_discrimination"]), "C": cfg.C, "directions": dirs}
    return {"composite": M.ModelSpec(cfg.index_name, pool=tuple(pool), **sel),
            "geometry": M.ModelSpec("geometry", fixed=tuple(geo), C=cfg.C),
            "geometry+mechanics": M.ModelSpec("geometry+mechanics", fixed=tuple(geo), pool=tuple(mpool), **sel),
            "missing": [c for c in cfg.expand(comp["pool"]) + cfg.expand(cfg.incremental["geometry_model"])
                        + cfg.expand(cfg.incremental["mechanics_pool"]) if c not in avail]}


def _sign_conflicts(fit, core, cfg) -> list[dict]:
    """Selected predictors whose fitted coefficient sign opposes their direction (configured; for ``"core"``,
    the sign set on the core set)."""
    coef = fit.coefficients().get("coef_per_sd", {})
    y = core["_y"].to_numpy(int)
    out = []
    for p in fit.selected:
        x = pd.to_numeric(core[p], errors="coerce").to_numpy(float)
        m = np.isfinite(x)
        s, src = R.resolve_direction(y[m], x[m], cfg.direction(p))
        if p in coef and coef[p] * s < 0:
            out.append({"predictor": p, "direction": s, "direction_source": src, "coef_per_sd": coef[p]})
    return out


def _composite(col, name, oc, core, hold, cfg, spec) -> dict:
    """(c) composite index on the core set (apparent and LOOCV) and its hold-out evaluation.

    The Youden cut of the core-fit probability classifies the core set
    (apparent) and the hold-out set; the LOOCV classification uses, for each
    lesion, the cut derived inside its fold (:func:`dtivl.stats.models.loocv_predict`).
    """
    base = {"outcome": name, "role": oc["role"], "kind": "binary", "predictor": cfg.index_name, "group": "composite"}
    pool = list(spec.pool)
    y = core["_y"].to_numpy(int)
    X = _model_matrix(core, pool)
    fit = M.fit_spec(spec, X, y)
    lo = M.loocv_predict(spec, X, y)
    p_app = fit.predict_proba(X)
    oof = M.prob_scores(y, lo["prob"])
    oof_ci = _oof_auc_ci(y, lo["prob"], cfg, f"{name}|composite|loocv")
    yd = R.youden(y, p_app)
    thr = yd["cut"]                                    # core-fit cut: apparent and hold-out classification only
    fin = math.isfinite(thr)
    perf_app = R.threshold_performance(y, R.classify(p_app, thr, 1), cfg.level) if fin else {}
    okf = lo["classified"] >= 0                        # folds with an in-fold threshold
    perf_oof = R.threshold_performance(y[okf], lo["classified"][okf], cfg.level) if okf.any() else {}
    n_undef = int((~okf).sum())
    conflicts = _sign_conflicts(fit, core, cfg)
    notes = ["auc = apparent (core fit)", "threshold = core-fit Youden cut (apparent and hold-out)",
             "loocv_sens/spec/acc = Youden cut re-derived in each fold, applied to the left-out lesion",
             f"loocv_auc interval: {OOF_CI_METHOD}"]
    if not fit.selected:
        notes.append("no usable pool predictor: constant model")
    if not fin:
        notes.append("no core-fit threshold (constant core-fit probability)")
    if n_undef:
        notes.append(f"{n_undef} folds without an in-fold threshold left out of the LOOCV counts")
    if oof_ci[2]:
        notes.append(DEGENERATE_NOTE.format(what="LOOCV AUC"))
    if conflicts:
        notes.append("coefficient sign opposes the direction: " + ", ".join(c["predictor"] for c in conflicts))
    col.row(**base, set=cfg.derivation_set, **_counts(y), direction=1, direction_source="model",
            auc=R.auc_score(y, p_app), threshold=thr, rule="p >= thr", youden_j=yd["j"],
            **{k: v for k, v in perf_app.items() if k.split("_")[0] in ("sens", "spec", "acc")},
            loocv_auc=oof["auc"], loocv_auc_ci_lo=oof_ci[0], loocv_auc_ci_hi=oof_ci[1],
            loocv_auc_ci_degenerate=oof_ci[2], loocv_log_loss=oof["log_loss"], loocv_brier=oof["brier"],
            **{f"loocv_{k}": perf_oof.get(k) for k in ("sens", "sens_ci_lo", "sens_ci_hi", "spec", "spec_ci_lo",
                                                       "spec_ci_hi", "acc")},
            n_boot=cfg.n_boot, note="; ".join(notes))
    mh = _complete_rows(hold, fit.predictors)
    hh = hold[mh]
    Xh = _model_matrix(hh, pool)
    ph = fit.predict_proba(Xh) if len(hh) else np.array([])
    yh = hh["_y"].to_numpy(int)
    hrow = _holdout_binary(base, yh, ph, 1, thr, cfg, f"{name}|composite|holdout", var="p")
    hrow["direction_source"] = "model"
    if (~mh).any():
        hrow["note"] = "; ".join(filter(None, [hrow.get("note"),
                                               f"{int((~mh).sum())} hold-out rows lack a model predictor"]))
    col.row(**hrow)
    lesions = [{"lesion_id": l, "set": cfg.derivation_set, "outcome": int(v), "index_core_fit": float(i),
                "p_core_fit": float(p), "index_loocv": float(il), "p_loocv": float(pl), "loocv_threshold_p": float(tl),
                "loocv_class": int(cl) if cl >= 0 else None}
               for l, v, i, p, il, pl, tl, cl in zip(core["lesion_id"], y, fit.decision(X), p_app, lo["index"],
                                                     lo["prob"], lo["threshold"], lo["classified"])]
    if len(hh):
        lesions += [{"lesion_id": l, "set": cfg.test_set, "outcome": int(v), "index_core_fit": float(i),
                     "p_core_fit": float(p)} for l, v, i, p in zip(hh["lesion_id"], yh, fit.decision(Xh), ph)]
    fold_thr = lo["threshold"][np.isfinite(lo["threshold"])]
    summary = {
        "name": cfg.index_name, "expansion": cfg.index_expansion, "rule": cfg.provenance["composite_rule"],
        "pool": pool, "selected": fit.selected, "model": fit.coefficients(),
        "selection_directions": {c: spec.direction(c) for c in pool},
        "coefficient_sign_opposes_direction": conflicts,
        "index_definition": "log-odds = intercept + sum_k coef_per_sd[k] * (x_k - mean[k]) / sd[k]; "
                            "probability = 1 / (1 + exp(-log-odds))",
        "core": {**_counts(y), "apparent_auc": R.auc_score(y, p_app), "loocv_auc": oof["auc"],
                 "loocv_auc_ci": [oof_ci[0], oof_ci[1]], "loocv_auc_ci_method": OOF_CI_METHOD,
                 "loocv_auc_ci_degenerate": oof_ci[2], "loocv_log_loss": oof["log_loss"], "loocv_brier": oof["brier"],
                 "threshold_p": thr, "threshold_p_use": "core-fit Youden cut: apparent and hold-out classification",
                 "apparent_sens": perf_app.get("sens"), "apparent_spec": perf_app.get("spec"),
                 "loocv_threshold_rule": "Youden cut of each fold model's fitted probabilities on its training rows, "
                                         "applied to the left-out lesion",
                 "loocv_threshold_p_range": [float(fold_thr.min()), float(fold_thr.max())] if fold_thr.size else None,
                 "loocv_n_folds_without_threshold": n_undef,
                 **{f"loocv_{k}": perf_oof.get(k) for k in ("sens", "spec", "acc")},
                 "loocv_sens_ci": [perf_oof.get("sens_ci_lo"), perf_oof.get("sens_ci_hi")],
                 "loocv_spec_ci": [perf_oof.get("spec_ci_lo"), perf_oof.get("spec_ci_hi")],
                 "loocv_selection_frequency": _selection_frequency(lo["selected"], len(y))},
        "holdout": {k: hrow.get(k) for k in ("n", "n_pos", "n_neg", "auc", "auc_ci_lo", "auc_ci_hi",
                                             "auc_ci_degenerate", "sens", "spec", "acc", "note")},
        "lesions": lesions,
    }
    index = {"fit": fit, "pool": pool, "core_loocv": dict(zip(core["lesion_id"], lo["index"])),
             "core_loocv_prob": lo["prob"], "holdout_rows": hh, "holdout_prob": ph, "threshold": thr,
             "perf_app": perf_app, "perf_oof": perf_oof, "holdout_row": hrow}
    return {"composite": summary, "_index": index}


def _fujino_comparison(col, name, core, cfg, uni, index) -> dict:
    """(d) Fujino score vs the composite index (LOOCV on core, core fit on hold-out) and the best single
    mechanics predictor: AUC differences with paired bootstrap intervals and threshold performance."""
    mech = [(p, u) for p, u in uni.items() if u["group"] == "mechanics" and math.isfinite(u["auc"])]
    best = max(mech, key=lambda pu: pu[1]["auc"])[0] if mech else None   # first maximum = configuration order
    comparators = [c for c in cfg.groups.get("comparator", []) if c in uni]
    out = {"best_single_mechanics": best,
           "best_single_note": "best single mechanics predictor = largest core AUC (chosen on the core set; "
                               "optimistic on the core set, unbiased on the hold-out set)"}
    if not comparators:
        col.nc(name, "Fujino comparison", "comparator not analysed on the core set (see its metrics_roc.csv row)")
        return out
    idx_lbl = {cfg.derivation_set: f"{cfg.index_name} (LOOCV)", cfg.test_set: f"{cfg.index_name} (core fit)"}
    for comp_col in comparators:
        entry = {"threshold_performance": {}, "auc_differences": {}}
        tp = {comp_col: uni[comp_col]}
        if best:
            tp[best] = uni[best]
        for lbl, u in tp.items():
            entry["threshold_performance"][lbl] = {
                "core": {k: u.get(k) for k in ("auc", "auc_ci", "threshold", "threshold_ci", "rule", "sens", "spec",
                                               "loocv_sens", "loocv_spec")},
                "holdout": u.get("holdout")}
        if index is not None:
            entry["threshold_performance"][cfg.index_name] = {
                "core": {"apparent_auc": R.auc_score(core["_y"].to_numpy(int), index["fit"].predict_proba(
                    _model_matrix(core, index["pool"]))), "threshold_p": index["threshold"], "rule": "p >= thr",
                    "apparent_sens": index["perf_app"].get("sens"), "apparent_spec": index["perf_app"].get("spec"),
                    "loocv_sens": index["perf_oof"].get("sens"), "loocv_spec": index["perf_oof"].get("spec")},
                "holdout": {k: index["holdout_row"].get(k) for k in ("n", "n_pos", "n_neg", "auc", "sens", "spec",
                                                                     "acc")}}
        for set_name in (cfg.derivation_set, cfg.test_set):
            if index is None:
                t, sidx = (core, None) if set_name == cfg.derivation_set else (None, None)
            elif set_name == cfg.derivation_set:
                t, sidx = core, index["core_loocv_prob"]
            else:
                t, sidx = index["holdout_rows"], index["holdout_prob"]
            if t is None or len(t) == 0:
                col.nc(name, "Fujino comparison (AUC differences)", "no rows", set_name)
                entry["auc_differences"][set_name] = None
                continue
            y = t["_y"].to_numpy(int)
            scores = {}
            if sidx is not None:
                scores[idx_lbl[set_name]] = np.asarray(sidx, float)
            scores[comp_col] = uni[comp_col]["direction"] * pd.to_numeric(t[comp_col], errors="coerce").to_numpy(float)
            if best:
                scores[best] = uni[best]["direction"] * pd.to_numeric(t[best], errors="coerce").to_numpy(float)
            ok = np.all([np.isfinite(v) for v in scores.values()], axis=0)
            y, scores = y[ok], {k: v[ok] for k, v in scores.items()}
            c = _counts(y)
            if c["n_pos"] < cfg.min_per_class or c["n_neg"] < cfg.min_per_class:
                reason = f"fewer than {cfg.min_per_class} lesions in a class"
                col.nc(name, "Fujino comparison (AUC differences)", reason, set_name)
                entry["auc_differences"][set_name] = {**c, "note": reason}
                continue
            idx = R.bootstrap_indices(R.rng_for(cfg.seed, f"{name}|fujino|{comp_col}|{set_name}"), cfg.n_boot,
                                      len(y), y, cfg.stratified)
            keys = list(scores)
            pairs = [(a, b) for a, b in ((keys[0], comp_col), (best, comp_col), (keys[0], best))
                     if a is not None and b is not None and a != b]
            entry["auc_differences"][set_name] = {
                **c, "scores": "oriented (direction x value); composite: probability",
                "interval": "paired bootstrap of fixed scores" + (f" (composite: {OOF_CI_METHOD})" if sidx is not None
                                                                  and set_name == cfg.derivation_set else ""),
                **{f"{a} - {b}": M.paired_bootstrap_fixed(y, scores[a], scores[b], idx, cfg.level)
                   for a, b in dict.fromkeys(pairs)}}
        out[comp_col] = entry
    return out


def _incremental(col, name, oc, core, hold, cfg, spec_g, spec_gm) -> dict:
    """(d) geometry-only vs geometry + mechanics models: apparent, LOOCV and hold-out comparisons."""
    base = {"outcome": name, "role": oc["role"], "kind": "binary", "group": "model"}
    geo, need = list(spec_g.fixed), spec_gm.columns
    inc = {"geometry_model": geo, "mechanics_pool": list(spec_gm.pool), "rule": cfg.provenance["incremental"],
           "difference": "geometry+mechanics - geometry"}
    if not geo:
        col.nc(name, "incremental value", "no geometry-model column available")
        return inc
    mc = _complete_rows(core, geo)
    tc = core[mc]
    y = tc["_y"].to_numpy(int)
    c = _counts(y)
    if c["n_pos"] < cfg.min_per_class or c["n_neg"] < cfg.min_per_class:
        col.nc(name, "incremental value", f"fewer than {cfg.min_per_class} lesions in a class", cfg.derivation_set)
        inc["core"] = {**c, "note": "not computed"}
        return inc
    X = _model_matrix(tc, need)
    idx = R.bootstrap_indices(R.rng_for(cfg.seed, f"{name}|incremental|apparent"), cfg.n_boot, len(y), y,
                              cfg.stratified)
    app = M.paired_bootstrap_refit(spec_gm, spec_g, X, y, idx, cfg.level)
    lg, lgm = M.loocv_predict(spec_g, X, y), M.loocv_predict(spec_gm, X, y)
    idx2 = R.bootstrap_indices(R.rng_for(cfg.seed, f"{name}|incremental|loocv"), cfg.n_boot, len(y), y,
                               cfg.stratified)
    loo = {**_paired_scores(y, lgm["prob"], lg["prob"], idx2, cfg.level), "interval": OOF_CI_METHOD}
    fg, fgm = M.fit_spec(spec_g, X, y), M.fit_spec(spec_gm, X, y)
    inc["core"] = {**c, "rows_dropped_missing_geometry": int((~mc).sum()),
                   "apparent_refit_bootstrap": app, "loocv": loo,
                   "geometry_fit": fg.coefficients(), "geometry_mechanics_fit": fgm.coefficients(),
                   "selected_mechanics_full_core": fgm.selected,
                   "loocv_selection_frequency": _selection_frequency(lgm["selected"], len(y))}
    # per-lesion predictions (figures): out-of-fold on the core set, core-fit models on the hold-out set
    inc["lesions"] = [{"lesion_id": lid, "set": cfg.derivation_set, "outcome": int(yy), "p_geometry_loocv": float(a),
                       "p_geometry_mechanics_loocv": float(b)}
                      for lid, yy, a, b in zip(tc["lesion_id"], y, lg["prob"], lgm["prob"])]
    for lbl, fit, lo in (("geometry", fg, lg), ("geometry+mechanics", fgm, lgm)):
        sc = M.prob_scores(y, lo["prob"])
        col.row(**base, set=cfg.derivation_set, predictor=f"model: {lbl}", **c, direction=1,
                direction_source="model", auc=R.auc_score(y, fit.predict_proba(X)), loocv_auc=sc["auc"],
                loocv_log_loss=sc["log_loss"], loocv_brier=sc["brier"], n_boot=cfg.n_boot,
                note="auc = apparent (core fit); loocv_* = leave-one-out")
    mh = _complete_rows(hold, list(dict.fromkeys(fg.predictors + fgm.predictors)))
    th = hold[mh]
    yh = th["_y"].to_numpy(int)
    ch = _counts(yh)
    if ch["n_pos"] < cfg.min_per_class or ch["n_neg"] < cfg.min_per_class:
        reason = "no hold-out rows" if ch["n"] == 0 else f"fewer than {cfg.min_per_class} hold-out lesions in a class"
        col.nc(name, "incremental value", reason, cfg.test_set)
        inc["holdout"] = {**ch, "note": reason}
        return inc
    Xh = _model_matrix(th, need)
    pg, pgm = fg.predict_proba(Xh), fgm.predict_proba(Xh)
    idx3 = R.bootstrap_indices(R.rng_for(cfg.seed, f"{name}|incremental|holdout"), cfg.n_boot, len(yh), yh,
                               cfg.stratified)
    inc["holdout"] = {**ch, "rows_dropped_missing_predictor": int((~mh).sum()),
                      **_paired_scores(yh, pgm, pg, idx3, cfg.level)}
    inc["lesions"] += [{"lesion_id": lid, "set": cfg.test_set, "outcome": int(yy), "p_geometry_core_fit": float(a),
                        "p_geometry_mechanics_core_fit": float(b)}
                       for lid, yy, a, b in zip(th["lesion_id"], yh, pg, pgm)]
    for lbl, prob in (("geometry", pg), ("geometry+mechanics", pgm)):
        sc = M.prob_scores(yh, prob)
        col.row(**base, set=cfg.test_set, predictor=f"model: {lbl}", **ch, direction=1, direction_source="model",
                auc=sc["auc"], n_boot=cfg.n_boot,
                note=f"core-fit model evaluated unchanged; log-loss {sc['log_loss']:.6g}, Brier {sc['brier']:.6g}")
    return inc


def analyse_binary(col: _Collector, name: str, oc: dict, df: pd.DataFrame, cfg: MetricsConfig) -> dict:
    """Analyses (a)-(d) of one binary outcome (models only when the outcome has ``models: true``)."""
    t, excl = _outcome_table(df, oc, cfg)
    if "_y" not in t:                                  # outcome column missing from the table
        reason = f"outcome column {oc['column']!r} not in the table"
        col.nc(name, "all analyses", reason, cfg.derivation_set)
        return {"role": oc["role"], "kind": "binary", "column": oc["column"], "protocol": oc["protocol"],
                "excluded": excl, "note": reason}
    core, hold = t[t["set"] == cfg.derivation_set], t[t["set"] == cfg.test_set]
    out = {"role": oc["role"], "kind": "binary", "column": oc["column"], "protocol": oc["protocol"],
           "ok_status": oc.get("ok_status"), "excluded": excl,
           "core": _counts(core["_y"].to_numpy(int)), "holdout": _counts(hold["_y"].to_numpy(int))}
    c, h = out["core"], out["holdout"]
    if c["n_pos"] < cfg.min_per_class or c["n_neg"] < cfg.min_per_class:
        reason = (f"single outcome class or fewer than {cfg.min_per_class} lesions per class on the core set "
                  f"(events {c['n_pos']}, non-events {c['n_neg']})")
        col.nc(name, "all analyses", reason, cfg.derivation_set)
        col.row(outcome=name, role=oc["role"], kind="binary", set=cfg.derivation_set, predictor="(all)", **c,
                note=reason)
        out["note"] = reason
        return out
    if h["n"] == 0:
        col.nc(name, "hold-out evaluation", "no hold-out rows with this outcome", cfg.test_set)
    elif h["n_pos"] < cfg.min_per_class or h["n_neg"] < cfg.min_per_class:
        col.nc(name, "hold-out AUC", f"fewer than {cfg.min_per_class} hold-out lesions in a class (events "
               f"{h['n_pos']}, non-events {h['n_neg']}); threshold performance reported where defined", cfg.test_set)
    uni = _univariable_binary(col, name, oc, core, hold, cfg)
    ranked = sorted(((p, u["auc"]) for p, u in uni.items() if math.isfinite(u["auc"])), key=lambda pa: -pa[1])
    out["univariable_ranking"] = [{"predictor": p, "group": uni[p]["group"], "auc": a,
                                   "direction_source": uni[p]["direction_source"]} for p, a in ranked]
    out["univariable_ranking_by"] = ("core AUC of direction x value (direction_source 'core': sign set on the core "
                                     "set, AUC >= 0.5 by construction)")
    out["univariable"] = uni
    if not oc.get("models"):
        return out
    specs = _specs(cfg, t.columns)
    if specs["missing"]:
        col.nc(name, "models", f"configured model columns missing from the table: {specs['missing']}")
    res = _composite(col, name, oc, core, hold, cfg, specs["composite"])
    out.update(res)
    out["fujino_comparison"] = _fujino_comparison(col, name, core, cfg, uni, res["_index"])
    out["incremental_value"] = _incremental(col, name, oc, core, hold, cfg, specs["geometry"],
                                            specs["geometry+mechanics"])
    return out


# =========================================================================== censored / continuous outcomes
def _concordance_row(col, base, t, x, cfg, key, s: int | None, core_sign: float | None = None) -> dict:
    """Harrell's C of predictor values ``x`` for the ranked outcome of ``t``, with a bootstrap interval.

    Censored onset (``s`` = +1 or -1): C of the oriented score ``s * x``
    (higher -> earlier onset); ``observed_sign`` is ``s`` when C > 0.5 and
    ``-s`` when C < 0.5. Continuous outcome (``s`` None, no a priori
    direction): C of the raw ``x`` (higher ``x`` -> higher outcome),
    ``observed_sign`` = sign(C - 0.5), Spearman's rho of the raw ``x`` and
    ``discrimination``: max(C, 1 - C), or, for hold-out rows (``core_sign``
    given), C oriented by the sign observed on the core set.
    """
    cont = s is None
    v, e = t["_value"].to_numpy(float), t["_event"].to_numpy(float)
    m = np.isfinite(x)
    v, e, x = v[m], e[m], x[m]
    row = {**base, "n": int(m.sum()), "n_pos": int((e == 1).sum()), "n_neg": int((e == 0).sum()),
           "direction": math.nan if cont else s}
    if row["n"] < cfg.min_rows_continuous:
        row["note"] = f"fewer than {cfg.min_rows_continuous} lesions"
        col.nc(base["outcome"], "concordance", row["note"], base["set"], base["predictor"])
        return row
    if np.ptp(x) == 0:
        row["note"] = "predictor constant on this set"
        col.nc(base["outcome"], "concordance", row["note"], base["set"], base["predictor"])
        return row
    risk = -x if cont else s * x                       # harrell_c: a higher risk goes with a lower value
    cidx, npairs = R.harrell_c(v, e, risk)
    rng = R.rng_for(cfg.seed, key)
    idx = R.bootstrap_indices(rng, cfg.n_boot, len(v))
    cb = np.array([R.harrell_c(v[i], e[i], risk[i])[0] for i in idx])
    row.update(c_index=cidx, n_pairs=npairs, n_boot=cfg.n_boot)
    row["c_index_ci_lo"], row["c_index_ci_hi"] = R.percentile_ci(cb, cfg.level)
    notes = [DEGENERATE_NOTE.format(what="C-index")] if R.ci_degenerate(cb, cidx) else []
    sign = float(np.sign(cidx - 0.5)) if math.isfinite(cidx) else math.nan
    row["observed_sign"] = sign if cont else s * sign
    if cont:
        if core_sign is None:
            row["discrimination"] = max(cidx, 1.0 - cidx)
        elif core_sign in (1, -1):
            row["discrimination"] = cidx if core_sign == 1 else 1.0 - cidx
            notes.append("discrimination = C oriented by the core sign")
        else:
            notes.append("no core sign (core C = 0.5 or not computed): discrimination not defined")
        row["spearman"] = R.spearman(x, v)
        ra, rb = _st.rankdata(x[idx], axis=1), _st.rankdata(v[idx], axis=1)
        ra, rb = ra - ra.mean(axis=1, keepdims=True), rb - rb.mean(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            rho = (ra * rb).sum(axis=1) / np.sqrt((ra ** 2).sum(axis=1) * (rb ** 2).sum(axis=1))
        row["spearman_ci_lo"], row["spearman_ci_hi"] = R.percentile_ci(rho, cfg.level)
    row["note"] = "; ".join(notes) or None
    return row


def _onset_counts(t: pd.DataFrame) -> dict:
    """Informational counts of the censored onset outcome (the rows are kept: partial cracks are events)."""
    e, tt = t["_event"].to_numpy(float), t["_through_thickness"].to_numpy(float)
    after = t["_after_peak"].to_numpy(bool)
    return {"onset_without_through_thickness": int(((e == 1) & (tt == 0)).sum()),
            "onset_through_thickness_unknown": int(((e == 1) & ~np.isfinite(tt)).sum()),
            "onset_after_peak": int(((e == 1) & after).sum())}


def _censor_summary(t: pd.DataFrame) -> dict:
    """Where the peak-load values (censored rows, onsets after the peak) came from."""
    src = t["_censor_source"].dropna().astype(str)
    out = {"rule": "ivl_p_top stored by the lesion's IVL run, checked against the configured IVL loads "
                   f"(tolerance {P_TOP_TOL_MPA} MPa)",
           "source_counts": {str(k): int(v) for k, v in src.value_counts().items()},
           "source_counts_rows": "rows of both sets whose value is the peak load (censored, or onset after the peak)"}
    n_cfg = int(src.str.startswith("configured").sum())
    if n_cfg:
        out["note"] = (f"ivl_p_top absent for {n_cfg} rows: their peak load is recomputed from the configured IVL "
                       "loads (p_bal + T_equivalent x ivl_peak_pressure_MPa x ivl_pulse_scale)")
    return out


def analyse_ranked(col: _Collector, name: str, oc: dict, df: pd.DataFrame, cfg: MetricsConfig,
                   index: dict | None) -> dict:
    """Censored (onset load) or continuous (stent expansion) outcome: Harrell's C (and Spearman) per candidate."""
    t, excl = _outcome_table(df, oc, cfg)
    cont = oc["kind"] == "continuous"
    if "_value" not in t:                              # outcome column missing from the table
        reason = f"outcome column {oc['column']!r} not in the table"
        col.nc(name, "all analyses", reason, cfg.derivation_set)
        return {"role": oc["role"], "kind": oc["kind"], "column": oc["column"], "protocol": oc["protocol"],
                "excluded": excl, "note": reason}
    core, hold = t[t["set"] == cfg.derivation_set], t[t["set"] == cfg.test_set]

    def cnt(tt):
        e = tt["_event"].to_numpy(float)
        c = {"n": int(len(tt)), "n_events": int((e == 1).sum()), "n_censored": int((e == 0).sum())}
        return c if cont else {**c, **_onset_counts(tt)}

    out = {"role": oc["role"], "kind": oc["kind"], "column": oc["column"], "protocol": oc["protocol"],
           "ok_status": oc.get("ok_status"), "excluded": excl, "core": cnt(core), "holdout": cnt(hold),
           "convention": CONTINUOUS_CONVENTION if cont else CENSORED_CONVENTION}
    if cont:
        out["exploratory"] = "direction not pre-specified (metrics.outcomes): exploratory concordance analysis"
    else:
        out["censoring"] = P.raw("metrics.outcomes")["basis"]
        out["censoring_value"] = _censor_summary(t)
    hold_ok = len(hold) >= cfg.min_rows_continuous
    if len(hold) == 0:
        col.nc(name, "hold-out evaluation", "no hold-out rows with this outcome", cfg.test_set)
    elif not hold_ok:
        col.nc(name, "hold-out evaluation", f"fewer than {cfg.min_rows_continuous} hold-out lesions with this "
               f"outcome ({len(hold)})", cfg.test_set)
    if len(core) < cfg.min_rows_continuous or (oc["kind"] == "censored" and out["core"]["n_events"] == 0):
        reason = f"too few core lesions with this outcome (n {len(core)}, events {out['core']['n_events']})"
        col.nc(name, "all analyses", reason, cfg.derivation_set)
        col.row(outcome=name, role=oc["role"], kind=oc["kind"], set=cfg.derivation_set, predictor="(all)",
                n=len(core), note=reason)
        out["note"] = reason
        return out
    if cont and np.ptp(core["_value"].to_numpy(float)) == 0:
        reason = "outcome constant on the core set"
        col.nc(name, "all analyses", reason, cfg.derivation_set)
        out["note"] = reason
        return out
    res = {}
    cands = list(cfg.candidates)
    if index is not None:
        cands.append((cfg.index_name, "composite"))
    for pred, group in cands:
        base = {"outcome": name, "role": oc["role"], "kind": oc["kind"], "predictor": pred, "group": group}
        notes = ["exploratory: direction not pre-specified"] if cont else []
        if group == "composite":
            xc = np.array([index["core_loocv"].get(l, math.nan) for l in core["lesion_id"]], float)
            fit = index["fit"]
            mh = _complete_rows(hold, fit.predictors)
            xh = np.full(len(hold), math.nan)
            if mh.any():
                xh[mh] = fit.decision(_model_matrix(hold[mh], index["pool"]))
            s, src = (None, "none") if cont else (1, "model")
            notes.append("index fitted for the primary outcome; core: LOOCV log-odds, hold-out: core fit")
        else:
            if pred not in t.columns:
                col.nc(name, "concordance", "column missing from cohort_outcomes.csv", cfg.derivation_set, pred)
                continue
            xc = pd.to_numeric(core[pred], errors="coerce").to_numpy(float)
            xh = pd.to_numeric(hold[pred], errors="coerce").to_numpy(float)
            d = cfg.direction(pred)
            if cont:
                s, src = None, "none"
            elif d == "core":
                m = np.isfinite(xc)
                cc = R.harrell_c(core["_value"].to_numpy(float)[m], core["_event"].to_numpy(float)[m], xc[m])[0]
                s, src = (1 if not (cc < 0.5) else -1), "core"
                notes.append("direction set on the core set (not pre-specified; C >= 0.5 by construction)")
            else:
                s, src = int(d), "config"
        r = _concordance_row(col, {**base, "set": cfg.derivation_set}, core, xc, cfg, f"{name}|{pred}|core", s)
        r["direction_source"] = src
        r["note"] = "; ".join(filter(None, [r.get("note"), *notes])) or None
        col.row(**r)
        res[pred] = {k: r.get(k) for k in ("group", "n", "c_index", "c_index_ci_lo", "c_index_ci_hi", "n_pairs",
                                           "observed_sign", "discrimination", "spearman", "spearman_ci_lo",
                                           "spearman_ci_hi", "direction")}
        res[pred]["direction_source"] = src
        if hold_ok:
            rh = _concordance_row(col, {**base, "set": cfg.test_set}, hold, xh, cfg, f"{name}|{pred}|holdout", s,
                                  core_sign=r.get("observed_sign", math.nan) if cont else None)
            rh["direction_source"] = src
            rh["note"] = "; ".join(filter(None, [rh.get("note"), *notes])) or None
            col.row(**rh)
            res[pred]["holdout"] = {k: rh.get(k) for k in ("n", "c_index", "c_index_ci_lo", "c_index_ci_hi",
                                                           "observed_sign", "discrimination", "spearman")}
    out["predictors"] = res
    by = "discrimination" if cont else "c_index"
    fin = [(p, v[by]) for p, v in res.items() if v.get(by) is not None and math.isfinite(v[by])]
    ranked = [p for p, _ in sorted(fin, key=lambda pc: -pc[1])]          # stable: ties in configuration order
    out["ranking_by"] = ("discrimination = max(C, 1 - C) of the raw predictor on the core set (exploratory)" if cont
                         else "core C of the oriented predictor (a priori direction; direction_source 'core' flagged)")
    out["ranking"] = [{"predictor": p, "group": res[p]["group"], "c_index": res[p]["c_index"],
                       "observed_sign": res[p]["observed_sign"],
                       **({"discrimination": res[p]["discrimination"], "spearman": res[p]["spearman"]} if cont else
                          {"direction": res[p]["direction"], "direction_source": res[p]["direction_source"],
                           "direction_set_on_core": res[p]["direction_source"] == "core"})}
                      for p in ranked]
    return out


# =========================================================================== stage
def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _software() -> dict:
    import scipy
    import sklearn

    return {"numpy": np.__version__, "scipy": scipy.__version__, "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__}


def run(outcomes_csv: str | Path | None = None, out_dir: str | Path | None = None, *, n_boot: int | None = None,
        refresh: bool = False) -> dict:
    """Run every analysis on the per-lesion table and write the three result files; returns the summary."""
    cfg = load_config(n_boot)
    path = Path(outcomes_csv or OUTCOMES_CSV)
    df = load_outcomes(path, refresh=refresh)
    check_predictors(cfg, df)
    out_dir = Path(out_dir or RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    col = _Collector()
    base_E = P.get("materials.calcium.E").value
    n_in = len(df)
    E = pd.to_numeric(df["ca_E_MPa"], errors="coerce") if "ca_E_MPa" in df else pd.Series(math.nan, index=df.index)
    no_E = ~np.isfinite(E.to_numpy(float))
    off_E = ~no_E & ((E - base_E).abs() > MODULUS_TOL_MPA).to_numpy()
    excluded = {"calcium modulus not recorded": int(no_E.sum()),
                "not at the baseline calcium modulus": int(off_E.sum())}
    for reason, n in excluded.items():
        if n:
            col.nc("(all)", "input rows", f"{n} rows excluded: {reason}")
    df = df[~no_E & ~off_E]
    sets = df["set"].value_counts().to_dict() if "set" in df else {}
    summary = {
        "stage": "metrics (Phase 5)",
        "input": {"path": path, "sha256": _sha256(path), "n_rows": n_in, "n_rows_used": int(len(df)),
                  "n_rows_excluded": excluded, "by_set": {str(k): int(v) for k, v in sets.items()},
                  "baseline_ca_E_MPa": base_E},
        "config": {"seed": cfg.seed, "n_boot": cfg.n_boot, "bootstrap_stratified": cfg.stratified,
                   "ci_level": cfg.level, "min_per_class": cfg.min_per_class,
                   "min_rows_continuous": cfg.min_rows_continuous, "logistic_C": cfg.C,
                   "derivation_set": cfg.derivation_set, "test_set": cfg.test_set, "candidates": cfg.groups,
                   "forbidden_predictor_prefixes": list(cfg.forbidden_prefixes),
                   "direction_default": cfg.direction_default, "direction_overrides": cfg.direction_overrides,
                   "composite": cfg.composite, "incremental": cfg.incremental,
                   "composite_index_name": cfg.index_name, "composite_index_expansion": cfg.index_expansion,
                   "provenance": cfg.provenance},
        "software": _software(),
        "wording": ("Hold-out results evaluate core-derived thresholds and models, unchanged, on lesions not used "
                    "for any choice; agreement is reported as consistency. All outcomes are simulated (in silico); "
                    "no comparison with patients is made here."),
        "outcomes": {},
    }
    if cfg.test_set not in sets:
        col.nc("(all)", "hold-out evaluation", f"no '{cfg.test_set}' rows in the input table", cfg.test_set)
    index = None
    order = sorted(cfg.outcomes, key=lambda o: 0 if o.get("models") else 1)   # model outcomes first (index reuse)
    for oc in order:
        name = oc["name"]
        if oc["kind"] == "binary":
            res = analyse_binary(col, name, oc, df, cfg)
            idx_res = res.pop("_index", None)
            if index is None and idx_res is not None:
                index = idx_res          # index of the first outcome with models (the primary outcome)
        else:
            res = analyse_ranked(col, name, oc, df, cfg, index)
        summary["outcomes"][name] = res
    summary["outcomes"] = {o["name"]: summary["outcomes"][o["name"]] for o in cfg.outcomes}
    summary["not_computed"] = col.not_computed
    roc = pd.DataFrame(col.rows).reindex(columns=ROC_COLUMNS)
    roc.to_csv(out_dir / "metrics_roc.csv", index=False)
    pd.DataFrame(col.curves).to_csv(out_dir / "metrics_roc_curves.csv", index=False)
    with open(out_dir / "metrics_summary.json", "w") as f:
        json.dump(_to_jsonable(summary), f, indent=1)
    return summary


def main(argv=None) -> None:
    """Command line: ``python -m dtivl.drivers.stage_metrics [--outcomes CSV] [--out-dir DIR] [--refresh]``."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outcomes", default=None, help="per-lesion table (default results/cohort_outcomes.csv)")
    ap.add_argument("--out-dir", default=None, help="output directory (default results/)")
    ap.add_argument("--n-boot", type=int, default=None, help="override metrics.bootstrap_n (testing only)")
    ap.add_argument("--refresh", action="store_true", help="rebuild cohort_outcomes.csv with aggregate() first")
    a = ap.parse_args(argv)
    s = run(a.outcomes, a.out_dir, n_boot=a.n_boot, refresh=a.refresh)
    print(json.dumps({"input": _to_jsonable(s["input"]), "n_not_computed": len(s["not_computed"])}, indent=1))


if __name__ == "__main__":
    main()

"""Small penalised logistic models for the Phase 5 composite index and the incremental-value comparison.

A :class:`ModelSpec` is a fitting *procedure*, not a fitted model: a fixed
predictor list (``fixed``) plus, optionally, predictors chosen from a ``pool``
by the pre-specified rule :func:`select_predictors`. Fitting a spec
(:func:`fit_spec`) applies the rule to the training data only, so leave-one-out
(LOOCV) predictions (:func:`loocv_predict`) and bootstrap refits
(:func:`paired_bootstrap_refit`) repeat the selection inside every fold or
resample, and data outside the training rows (the hold-out set) are never used
for a choice.

Selection rule (:func:`select_predictors`): candidates with a missing value or
a constant value on the training rows are dropped; each remaining candidate
gets its oriented univariable AUC on the training rows, ``AUC`` for an a
priori direction ``+1``, ``1 - AUC`` for ``-1`` and ``max(AUC, 1 - AUC)`` for
``"core"`` (no sign stated a priori: the sign is set on the training rows, so
it is re-derived in every fold or resample). Candidates with an oriented AUC
below ``min_discrimination`` are dropped (so a predictor whose association
has the opposite of its a priori sign cannot enter); the rest are ranked by
the oriented AUC (ties: pool order) and taken in that order, skipping any
whose |Spearman rho| with an already selected predictor exceeds
``max_abs_spearman``, until ``max_predictors`` are chosen (fewer if the
candidates run out). Pool columns without a stated direction are ``"core"``.

Model: predictors z-scored with the training mean and population standard
deviation (as ``sklearn.preprocessing.StandardScaler``) and an L2-penalised
logistic regression (scikit-learn ``LogisticRegression`` with ``l1_ratio=0``,
``lbfgs``, inverse penalty ``C``). With no usable predictor, or a single
outcome class in the training rows, the model predicts the training event
fraction for every lesion. The index value is the linear predictor (log-odds)
of the fitted model.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as _st
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from .roc import auc_rows, auc_score, ci_degenerate, percentile_ci, youden

#: NUMERICAL: iteration limit of the lbfgs solver (z-scored inputs with an L2 penalty converge well within it).
LOGISTIC_MAX_ITER = 1000
#: NUMERICAL: probabilities are clipped to [eps, 1 - eps] before the log-loss.
PROB_CLIP = 1e-12

__all__ = ["ModelSpec", "IndexModel", "select_predictors", "fit_spec", "loocv_predict", "prob_scores",
           "paired_bootstrap_fixed", "paired_bootstrap_refit", "PROB_CLIP"]


@dataclass(frozen=True)
class ModelSpec:
    """A logistic-model fitting procedure (see the module docstring)."""

    name: str
    fixed: tuple[str, ...] = ()
    pool: tuple[str, ...] = ()
    max_predictors: int = 0
    max_abs_spearman: float = 1.0
    C: float = 1.0
    min_discrimination: float = 0.5
    #: ``(column, direction)`` pairs of pool columns (+1, -1 or ``"core"``); unlisted columns are ``"core"``
    directions: tuple[tuple[str, object], ...] = ()

    @property
    def columns(self) -> list[str]:
        """Every column the procedure may read."""
        return list(dict.fromkeys(self.fixed + self.pool))

    def direction(self, col: str):
        """A priori direction of pool column ``col`` (+1, -1 or ``"core"``)."""
        return dict(self.directions).get(col, "core")


@dataclass
class IndexModel:
    """A fitted model: predictors, scaler, logistic coefficients (or a constant) and training event fraction."""

    predictors: list[str]
    prevalence: float
    mean: np.ndarray | None = None
    sd: np.ndarray | None = None
    lr: LogisticRegression | None = None
    selected: list[str] = field(default_factory=list)

    def _logit(self, A: np.ndarray) -> np.ndarray:
        if self.lr is None:
            p = min(max(self.prevalence, PROB_CLIP), 1.0 - PROB_CLIP)
            return np.full(A.shape[0], math.log(p / (1.0 - p)))
        return self.lr.decision_function((A - self.mean) / self.sd)

    def decision(self, X: pd.DataFrame) -> np.ndarray:
        """Linear predictor (log-odds, the index value) for the rows of ``X``."""
        return self._logit(X[self.predictors].to_numpy(float) if self.predictors else np.empty((len(X), 0)))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predicted event probability for the rows of ``X``."""
        if self.lr is None:
            return np.full(len(X), float(self.prevalence))
        return 1.0 / (1.0 + np.exp(-self.decision(X)))

    def coefficients(self) -> dict:
        """Intercept, coefficients per z-scored predictor and the scaler (mean, sd) so the index can be recomputed."""
        if self.lr is None:
            return {"intercept": float(self._logit(np.empty((1, 0)))[0]), "coef_per_sd": {}, "mean": {}, "sd": {},
                    "note": "no usable predictor or one outcome class: constant model (training event fraction)"}
        return {"intercept": float(self.lr.intercept_[0]),
                "coef_per_sd": {k: float(v) for k, v in zip(self.predictors, self.lr.coef_[0])},
                "mean": {k: float(v) for k, v in zip(self.predictors, self.mean)},
                "sd": {k: float(v) for k, v in zip(self.predictors, self.sd)}}


# --------------------------------------------------------------------------- array core
def _select_idx(A: np.ndarray, y: np.ndarray, pool: list[int], max_predictors: int,
                max_abs_spearman: float, min_discrimination: float = 0.5, directions=None) -> list[int]:
    """Selection rule on column indices ``pool`` of the training matrix ``A`` (``directions``: one per pool column)."""
    if not pool or max_predictors <= 0:
        return []
    dirs = ["core"] * len(pool) if directions is None else list(directions)
    if len(dirs) != len(pool) or any(d not in (1, -1, "core") for d in dirs):
        raise ValueError(f"directions must give +1, -1 or 'core' for every pool column, got {dirs!r}")
    n1 = int(y.sum())
    n0 = y.size - n1
    if n1 == 0 or n0 == 0:
        return []
    sub = A[:, pool]
    filled = np.where(np.isfinite(sub), sub, 0.0)
    ok = np.isfinite(sub).all(axis=0) & (np.ptp(filled, axis=0) > 0)
    if not ok.any():
        return []
    r = _st.rankdata(filled, axis=0)
    auc = (r[y == 1].sum(axis=0) - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    oriented = np.array([np.maximum(a, 1.0 - a) if d == "core" else (a if d == 1 else 1.0 - a)
                         for a, d in zip(auc, dirs)])
    order = [j for j in np.argsort(-oriented, kind="stable") if ok[j] and oriented[j] >= min_discrimination]
    rc = r - r.mean(axis=0)
    norm = np.sqrt((rc ** 2).sum(axis=0))
    chosen: list[int] = []
    for j in order:
        if len(chosen) >= max_predictors:
            break
        if all(abs(float(rc[:, j] @ rc[:, k]) / (norm[j] * norm[k])) <= max_abs_spearman for k in chosen):
            chosen.append(j)
    return [pool[j] for j in chosen]


def _fit_idx(A: np.ndarray, y: np.ndarray, names: list[str], cols: list[int], C: float) -> IndexModel:
    prev = float(y.mean()) if y.size else math.nan
    cols = [c for c in cols if np.isfinite(A[:, c]).all() and np.ptp(A[:, c]) > 0]
    pred = [names[c] for c in cols]
    if not cols or y.min() == y.max():
        return IndexModel(pred, prev)
    S = A[:, cols]
    mean, sd = S.mean(axis=0), S.std(axis=0)
    lr = LogisticRegression(C=C, l1_ratio=0.0, solver="lbfgs", max_iter=LOGISTIC_MAX_ITER)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        lr.fit((S - mean) / sd, y)
    return IndexModel(pred, prev, mean, sd, lr)


def _fit_spec_idx(spec: ModelSpec, A: np.ndarray, y: np.ndarray, names: list[str]) -> IndexModel:
    pos = {n: i for i, n in enumerate(names)}
    pool = [pos[c] for c in spec.pool if c in pos]
    sel = _select_idx(A, y, pool, spec.max_predictors, spec.max_abs_spearman, spec.min_discrimination,
                      [spec.direction(names[c]) for c in pool])
    cols = list(dict.fromkeys([pos[c] for c in spec.fixed if c in pos] + sel))
    m = _fit_idx(A, y, names, cols, spec.C)
    m.selected = [names[c] for c in sel]
    return m


def _matrix(spec: ModelSpec, X: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    names = [c for c in spec.columns if c in X.columns]
    return X[names].to_numpy(float) if names else np.empty((len(X), 0)), names


def _prob(m: IndexModel, A: np.ndarray, names: list[str]) -> np.ndarray:
    if m.lr is None:
        return np.full(A.shape[0], float(m.prevalence))
    pos = {n: i for i, n in enumerate(names)}
    return 1.0 / (1.0 + np.exp(-m._logit(A[:, [pos[c] for c in m.predictors]])))


# --------------------------------------------------------------------------- public API
def select_predictors(X: pd.DataFrame, y, pool, *, max_predictors: int, max_abs_spearman: float,
                      min_discrimination: float = 0.5, directions: dict | None = None) -> list[str]:
    """Pre-specified selection rule (module docstring) applied to the training rows ``X``, ``y``.

    ``directions`` maps pool columns to +1, -1 or ``"core"`` (unlisted: ``"core"``).
    """
    pool = [c for c in pool if c in X.columns]
    A = X[pool].to_numpy(float) if pool else np.empty((len(X), 0))
    sel = _select_idx(A, np.asarray(y).astype(int), list(range(len(pool))), max_predictors, max_abs_spearman,
                      min_discrimination, [(directions or {}).get(c, "core") for c in pool])
    return [pool[i] for i in sel]


def fit_spec(spec: ModelSpec, X: pd.DataFrame, y) -> IndexModel:
    """Apply the selection rule of ``spec`` to (``X``, ``y``) and fit the logistic model."""
    A, names = _matrix(spec, X)
    return _fit_spec_idx(spec, A, np.asarray(y).astype(int), names)


def loocv_predict(spec: ModelSpec, X: pd.DataFrame, y) -> dict:
    """Leave-one-out predictions of ``spec`` (selection, fit and threshold repeated in each fold).

    Returns ``prob`` and ``index`` (out-of-fold probability and log-odds per
    row), ``selected`` (pool predictors chosen in each fold), ``threshold``
    (the Youden cut of the fold model's fitted probabilities on its training
    rows; ``nan`` when the fold has none, e.g. a constant model) and
    ``classified`` (1 when ``prob >= threshold``, 0 below it, -1 where the
    fold has no threshold). Nothing in a fold uses the left-out outcome.
    """
    y = np.asarray(y).astype(int)
    A, names = _matrix(spec, X)
    pos = {nm: i for i, nm in enumerate(names)}
    n = y.size
    prob, idx, sel = np.full(n, math.nan), np.full(n, math.nan), []
    thr, cls = np.full(n, math.nan), np.full(n, -1)
    for i in range(n):
        m = np.arange(n) != i
        fit = _fit_spec_idx(spec, A[m], y[m], names)
        row = A[[i]][:, [pos[c] for c in fit.predictors]] if fit.predictors else np.empty((1, 0))
        idx[i] = fit._logit(row)[0]
        prob[i] = fit.prevalence if fit.lr is None else 1.0 / (1.0 + math.exp(-idx[i]))
        sel.append(fit.selected)
        thr[i] = youden(y[m], _prob(fit, A[m], names))["cut"]
        if math.isfinite(thr[i]):
            cls[i] = int(prob[i] >= thr[i])
    return {"prob": prob, "index": idx, "selected": sel, "threshold": thr, "classified": cls}


def prob_scores(y, p) -> dict:
    """AUC, log-loss and Brier score of probabilities ``p`` (``nan`` entries when undefined)."""
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype=float), PROB_CLIP, 1.0 - PROB_CLIP)
    if y.size == 0:
        return {"auc": math.nan, "log_loss": math.nan, "brier": math.nan}
    return {"auc": auc_score(y, p) if y.min() != y.max() else math.nan,
            "log_loss": float(log_loss(y, p, labels=[0, 1])), "brier": float(brier_score_loss(y, p))}


def _boot_summary(auc_a: float, auc_b: float, d: np.ndarray, level: float) -> dict:
    lo, hi = percentile_ci(d, level)
    fin = np.isfinite(d)
    return {"auc_a": auc_a, "auc_b": auc_b, "delta_auc": auc_a - auc_b, "delta_ci_lo": lo, "delta_ci_hi": hi,
            "delta_ci_degenerate": ci_degenerate(d, bounds=()),
            "boot_frac_delta_le_0": float(np.mean(d[fin] <= 0.0)) if fin.any() else math.nan,
            "n_boot_valid": int(fin.sum())}


def paired_bootstrap_fixed(y, score_a, score_b, idx: np.ndarray, level: float) -> dict:
    """Paired bootstrap of AUC(a) - AUC(b) for fixed scores (rows resampled with the ``(B, n)`` matrix ``idx``).

    ``boot_frac_delta_le_0`` is the fraction of resamples with a difference
    <= 0 (a descriptive bootstrap fraction, not a test); ``delta_ci_degenerate``
    flags an interval whose resamples all give the same difference.
    """
    y = np.asarray(y).astype(int)
    a, b = np.asarray(score_a, float), np.asarray(score_b, float)
    d = auc_rows(y[idx], a[idx]) - auc_rows(y[idx], b[idx])
    return _boot_summary(auc_score(y, a), auc_score(y, b), d, level)


def paired_bootstrap_refit(spec_a: ModelSpec, spec_b: ModelSpec, X: pd.DataFrame, y, idx: np.ndarray,
                           level: float) -> dict:
    """Paired bootstrap of the apparent AUC difference with both procedures refitted in every resample.

    Each resample (row of ``idx``) is used to select and fit both specs and to
    evaluate them (apparent, in-sample AUC). Also returns how often each pool
    predictor was selected by each spec (``selection_frequency_a`` / ``_b``).
    """
    y = np.asarray(y).astype(int)
    cols = list(dict.fromkeys(spec_a.columns + spec_b.columns))
    names = [c for c in cols if c in X.columns]
    A = X[names].to_numpy(float) if names else np.empty((len(X), 0))
    d, counts = np.full(len(idx), math.nan), ({}, {})
    for b, rows in enumerate(idx):
        Ab, yb = A[rows], y[rows]
        fa, fb = _fit_spec_idx(spec_a, Ab, yb, names), _fit_spec_idx(spec_b, Ab, yb, names)
        d[b] = auc_score(yb, _prob(fa, Ab, names)) - auc_score(yb, _prob(fb, Ab, names))
        for cnt, f in zip(counts, (fa, fb)):
            for c in f.selected:
                cnt[c] = cnt.get(c, 0) + 1
    fa, fb = _fit_spec_idx(spec_a, A, y, names), _fit_spec_idx(spec_b, A, y, names)
    out = _boot_summary(auc_score(y, _prob(fa, A, names)), auc_score(y, _prob(fb, A, names)), d, level)
    for key, cnt in zip(("selection_frequency_a", "selection_frequency_b"), counts):
        out[key] = {k: v / len(idx) for k, v in sorted(cnt.items(), key=lambda kv: -kv[1])}
    return out

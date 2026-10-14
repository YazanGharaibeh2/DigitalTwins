"""ROC, threshold, proportion and concordance statistics for the Phase 5 metrics (no I/O, no plotting).

Conventions
-----------
* Binary outcome ``y`` (1 = event, for example through-thickness damage under
  IVL; 0 = no event), predictor ``x`` and direction ``s``: ``+1`` when a
  higher ``x`` predicts the event, ``-1`` when a lower ``x`` does. Every
  function works on the oriented score ``z = s * x``. A lesion is classified
  positive when ``z >= z_cut``, i.e. ``x >= thr`` for ``s = +1`` and
  ``x <= thr`` for ``s = -1`` (``thr = s * z_cut``).
* AUC: Mann-Whitney estimate with tied scores counted 1/2 (identical to
  ``sklearn.metrics.roc_auc_score``).
* Youden threshold: the cut maximising ``J = sensitivity + specificity - 1``
  over the non-trivial cuts ``z >= u_k`` (``u_1 < ... < u_m`` the distinct
  observed scores, ``k = 2..m``; the cut ``z >= u_1`` classifies every lesion
  positive). Ties in ``J`` are broken toward the higher, more specific cut.
  The reported threshold is the midpoint ``(u_{k-1} + u_k) / 2``: it gives
  the same classification of the derivation set and does not put a new lesion
  that falls between two observed values on the boundary. A constant
  predictor has no threshold (``nan``).
* Bootstrap: percentile intervals; binary outcomes are resampled within each
  class (stratified) so every resample keeps both classes. Random streams come
  from :func:`rng_for` (a ``numpy`` ``SeedSequence`` with the study seed as
  entropy and a CRC32 of an analysis key as spawn key), so an analysis gives
  the same numbers whichever other analyses run before it. An interval is
  flagged degenerate (:func:`ci_degenerate`) when every finite resample gives
  the same value or the point estimate is at a bound (AUC or C of 0 or 1,
  perfect separation): its zero width says nothing about precision.
* Proportions (sensitivity, specificity, accuracy): Wilson score intervals.
* Concordance (censored or continuous outcomes): Harrell's C with ties in the
  risk score counted 1/2; a pair is comparable when the lesion with the lower
  outcome value had an event (or, at equal values, it had an event and the
  other was censored).
"""
from __future__ import annotations

import math
import zlib

import numpy as np
from scipy import stats as _st

__all__ = [
    "rng_for", "auc_score", "auc_rows", "roc_points", "youden", "classify", "wilson_ci",
    "percentile_ci", "ci_degenerate", "bootstrap_indices", "resolve_direction", "roc_analysis",
    "threshold_performance",
    "loocv_threshold", "harrell_c", "spearman",
]


# --------------------------------------------------------------------------- random streams
def rng_for(seed: int, key: str) -> np.random.Generator:
    """Deterministic generator for one analysis: entropy ``seed``, spawn key ``crc32(key)``."""
    return np.random.default_rng(np.random.SeedSequence(entropy=int(seed), spawn_key=(zlib.crc32(key.encode()),)))


def bootstrap_indices(rng: np.random.Generator, n_boot: int, n: int, y=None, stratified: bool = True) -> np.ndarray:
    """``(n_boot, n)`` resample indices; with ``y`` and ``stratified`` each class is resampled within itself."""
    if y is None or not stratified:
        return rng.integers(0, n, size=(int(n_boot), n))
    y = np.asarray(y)
    parts = []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        parts.append(idx[rng.integers(0, idx.size, size=(int(n_boot), idx.size))])
    return np.concatenate(parts, axis=1)


def percentile_ci(values, level: float) -> tuple[float, float]:
    """Percentile interval of the finite entries of ``values`` (``nan`` if there are none)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return math.nan, math.nan
    a = 100.0 * (1.0 - level) / 2.0
    lo, hi = np.percentile(v, [a, 100.0 - a])
    return float(lo), float(hi)


def ci_degenerate(values, point: float = math.nan, bounds=(0.0, 1.0)) -> bool:
    """True when the percentile interval of ``values`` is degenerate.

    Degenerate: every finite resample value is equal, or ``point`` is one of
    ``bounds`` (an AUC or C of 0 or 1, i.e. perfect separation, where every
    resample that keeps both classes repeats it). ``False`` without finite
    resample values (the interval is then ``nan``, not degenerate).
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return False
    return bool(np.ptp(v) == 0 or any(point == b for b in bounds))


# --------------------------------------------------------------------------- AUC and ROC
def _check_binary(y) -> np.ndarray:
    y = np.asarray(y)
    if y.size and not np.isin(y, (0, 1)).all():
        raise ValueError("binary outcome must contain only 0 and 1")
    return y.astype(int)


def auc_score(y, score) -> float:
    """Mann-Whitney AUC of ``score`` for outcome ``y`` (ties 1/2); ``nan`` if a class is empty."""
    y = _check_binary(y)
    s = np.asarray(score, dtype=float)
    n1 = int(y.sum())
    n0 = y.size - n1
    if n1 == 0 or n0 == 0:
        return math.nan
    r = _st.rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def auc_rows(y_rows: np.ndarray, s_rows: np.ndarray) -> np.ndarray:
    """Row-wise AUC of a ``(B, n)`` matrix of outcomes and scores (vectorised bootstrap helper)."""
    y_rows = np.asarray(y_rows).astype(int)
    r = _st.rankdata(np.asarray(s_rows, dtype=float), axis=1)
    n1 = y_rows.sum(axis=1).astype(float)
    n0 = y_rows.shape[1] - n1
    with np.errstate(invalid="ignore", divide="ignore"):
        out = ((r * y_rows).sum(axis=1) - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    out[(n1 == 0) | (n0 == 0)] = math.nan
    return out


def roc_points(y, score) -> dict:
    """Empirical ROC curve of ``score`` (positive when ``score >= cut``), cuts from +inf down to the minimum."""
    y = _check_binary(y)
    s = np.asarray(score, dtype=float)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    u = np.unique(s)[::-1]
    sp, sn = np.sort(s[y == 1]), np.sort(s[y == 0])
    tp = n1 - np.searchsorted(sp, u, side="left")
    fp = n0 - np.searchsorted(sn, u, side="left")
    tpr = np.concatenate([[0.0], tp / n1]) if n1 else np.full(u.size + 1, math.nan)
    fpr = np.concatenate([[0.0], fp / n0]) if n0 else np.full(u.size + 1, math.nan)
    return {"cut": np.concatenate([[math.inf], u]), "fpr": fpr, "tpr": tpr}


def youden(y, score) -> dict:
    """Youden-optimal cut of ``score`` (positive when ``score >= cut``); see the module conventions.

    Returns ``cut`` (midpoint threshold on the score scale), ``sens``, ``spec``,
    ``j`` and ``n_levels`` (distinct score values); ``cut`` is ``nan`` for a
    constant score or an empty class.
    """
    return _youden(_check_binary(y), np.asarray(score, dtype=float))


def _youden(y: np.ndarray, s: np.ndarray) -> dict:
    """:func:`youden` without input checks (``y`` int 0/1, ``s`` float; bootstrap inner loop)."""
    n1 = int(y.sum())
    n0 = y.size - n1
    u = np.unique(s)
    out = {"cut": math.nan, "sens": math.nan, "spec": math.nan, "j": math.nan, "n_levels": int(u.size)}
    if n1 == 0 or n0 == 0 or u.size < 2:
        return out
    cand = u[1:]
    sp, sn = np.sort(s[y == 1]), np.sort(s[y == 0])
    sens = (n1 - np.searchsorted(sp, cand, side="left")) / n1
    spec = np.searchsorted(sn, cand, side="left") / n0
    j = sens + spec - 1.0
    k = j.size - 1 - int(np.argmax(j[::-1]))          # last maximum = highest cut
    out.update(cut=float(0.5 * (u[k] + u[k + 1])), sens=float(sens[k]), spec=float(spec[k]), j=float(j[k]))
    return out


def classify(x, thr: float, direction: int) -> np.ndarray:
    """Positive (1) when ``direction * x >= direction * thr``."""
    x = np.asarray(x, dtype=float)
    return (direction * x >= direction * thr).astype(int)


# --------------------------------------------------------------------------- proportions
def wilson_ci(k: int, n: int, level: float) -> tuple[float, float]:
    """Wilson score interval of the proportion ``k / n`` (``nan`` for ``n = 0``)."""
    if n <= 0:
        return math.nan, math.nan
    z = float(_st.norm.ppf(1.0 - (1.0 - level) / 2.0))
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / den
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def threshold_performance(y, pred, level: float) -> dict:
    """Sensitivity, specificity and accuracy (Wilson intervals) of 0/1 predictions ``pred``.

    Entries that need an empty class (sensitivity without events, specificity
    without non-events) are ``nan``.
    """
    y = _check_binary(y)
    pred = np.asarray(pred).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    out = {"tp": tp, "fn": fn, "tn": tn, "fp": fp}
    for name, k, n in (("sens", tp, tp + fn), ("spec", tn, tn + fp), ("acc", tp + tn, y.size)):
        out[name] = k / n if n else math.nan
        out[f"{name}_ci_lo"], out[f"{name}_ci_hi"] = wilson_ci(k, n, level)
    return out


# --------------------------------------------------------------------------- univariable ROC analysis
def resolve_direction(y, x, direction) -> tuple[int, str]:
    """``(s, source)``: a configured ``+1``/``-1`` (source ``"config"``) or ``"core"``.

    ``"core"`` sets ``s`` on the given (derivation) data as the sign giving
    AUC >= 0.5 (source ``"core"``; optimistic, reported as such).
    """
    if direction in (1, -1):
        return int(direction), "config"
    if direction != "core":
        raise ValueError(f"direction must be 1, -1 or 'core', got {direction!r}")
    a = auc_score(y, x)
    return (1 if not (a < 0.5) else -1), "core"


def roc_analysis(y, x, direction: int, *, n_boot: int, rng: np.random.Generator, level: float,
                 stratified: bool = True) -> dict:
    """Univariable ROC of ``x`` with fixed ``direction``: AUC and Youden threshold with bootstrap intervals.

    The direction is kept fixed in the resamples. Returns ``auc``,
    ``auc_ci_lo/hi``, ``threshold`` (in ``x`` units), ``threshold_ci_lo/hi``,
    ``sens``, ``spec``, ``youden_j``, ``n_levels``, ``n_boot``, the number
    of resamples with a defined AUC / threshold and ``auc_ci_degenerate``
    (:func:`ci_degenerate`).
    """
    y = _check_binary(y)
    x = np.asarray(x, dtype=float)
    z = direction * x
    yd = youden(y, z)
    out = {"auc": auc_score(y, z), "threshold": direction * yd["cut"], "sens": yd["sens"], "spec": yd["spec"],
           "youden_j": yd["j"], "n_levels": yd["n_levels"], "n_boot": int(n_boot)}
    idx = bootstrap_indices(rng, n_boot, y.size, y, stratified)
    yb, zb = y[idx], z[idx]
    aucs = auc_rows(yb, zb)
    thr = np.array([direction * _youden(yb[b], zb[b])["cut"] for b in range(idx.shape[0])])
    out["auc_ci_lo"], out["auc_ci_hi"] = percentile_ci(aucs, level)
    out["auc_ci_degenerate"] = ci_degenerate(aucs, out["auc"])
    out["threshold_ci_lo"], out["threshold_ci_hi"] = sorted(percentile_ci(thr, level))
    out["n_boot_valid_auc"] = int(np.isfinite(aucs).sum())
    out["n_boot_valid_threshold"] = int(np.isfinite(thr).sum())
    return out


def loocv_threshold(y, x, direction, level: float) -> dict:
    """Leave-one-out estimate of the Youden-threshold rule.

    For each lesion the direction (if ``"core"``) and the Youden threshold are
    derived on the other lesions and applied to the left-out one. Returns
    :func:`threshold_performance` of the left-out classifications plus
    ``n_undefined`` (folds whose training data gave no threshold; those
    lesions are left out of the counts).
    """
    y = _check_binary(y)
    x = np.asarray(x, dtype=float)
    pred = np.full(y.size, -1)
    for i in range(y.size):
        m = np.arange(y.size) != i
        s, _ = resolve_direction(y[m], x[m], direction)
        cut = youden(y[m], s * x[m])["cut"]
        if math.isfinite(cut):
            pred[i] = int(s * x[i] >= cut)
    ok = pred >= 0
    out = threshold_performance(y[ok], pred[ok], level)
    out["n_undefined"] = int((~ok).sum())
    return out


# --------------------------------------------------------------------------- concordance
def harrell_c(value, event, risk) -> tuple[float, int]:
    """Harrell's C of ``risk`` for an outcome where a lower ``value`` is worse (earlier onset, lower expansion).

    ``event`` is 1 where ``value`` was observed and 0 where it is a
    right-censoring value. Returns ``(C, n_comparable_pairs)``; ``C`` is
    ``nan`` without comparable pairs.
    """
    t = np.asarray(value, dtype=float)
    e = np.asarray(event).astype(bool)
    r = np.asarray(risk, dtype=float)
    ti, tj = t[:, None], t[None, :]
    comp = e[:, None] & ((ti < tj) | ((ti == tj) & ~e[None, :]))
    n = int(comp.sum())
    if n == 0:
        return math.nan, 0
    conc = ((r[:, None] > r[None, :]) & comp).sum()
    ties = ((r[:, None] == r[None, :]) & comp).sum()
    return float((conc + 0.5 * ties) / n), n


def spearman(a, b) -> float:
    """Spearman rank correlation (``nan`` when either variable is constant or fewer than 3 pairs)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return math.nan
    return float(_st.spearmanr(a, b).statistic)

"""Linear-elastic fracture predictors of the calcium (design.md section 4).

``predictors(linres, plm, ms)`` condenses the per-cell calcium arrays of a
:class:`dtivl.solvers.linear.LinearResult` into scalar predictors.
"Area-weighted" means weighted by the cell measure (area in 2D, volume in
3D); weighted percentiles use the inverted weighted CDF (smallest value whose
cumulative weight fraction reaches q).

Keys
----
* ``sigma1_peak`` - max cell-averaged max principal stress in calcium (MPa).
* ``sigma1_p95`` - area-weighted 95th percentile of sigma1 (MPa).
* ``sed_mean`` / ``sed_peak`` - area-weighted mean / max of the cell-averaged
  strain energy density (MPa = mJ/mm^3).
* ``frac_eps_gt_half_epsf`` / ``frac_eps_gt_epsf`` - calcium area fraction
  with eps1 > eps_f/2 / > eps_f (``ms.eps_f``).
* ``frac_sig_gt_half_sigc`` / ``frac_sig_gt_sigc`` - same for sigma1 vs
  ``ms.sigma_crit``.
* ``scf_edge`` - max sigma1 over calcium cells whose centroid lies within the
  angular window ``max(EDGE_MIN_WINDOW_DEG, 2 t_ca / r_mid)`` of any deposit
  end, divided by the area-weighted mean sigma1 of all calcium cells. 1.0
  for closed rings (``has_edges = False``); NaN if the mean is <= 0 or no
  cell lies in an end window.
* ``has_edges`` - True if any deposit has angular ends.
* ``sigma1_p99`` - area-weighted 99th percentile of sigma1.
* ``sigma1_peak_nl`` / ``sed_peak_nl`` / ``scf_edge_nl`` - *nonlocal* versions
  of the peak quantities: each cell value is replaced by the area-weighted
  mean over calcium cells whose centroids lie within ``rho_nl`` of its
  centroid (area method of critical-distance theory), then the maximum is
  taken. ``rho_nl`` = ``fracture.ell_over_h`` x ``discretisation.h_calcium_mm``,
  i.e. the nominal phase-field regularisation length, so the averaging scale
  does not change with mesh refinement. Raw cell peaks grow under refinement
  near image-derived boundary kinks (see results/verification_mesh_convergence.json).
* ``min_thickness_high_stress`` - min local calcium thickness (calcium
  sample count of the A-line through the cell centroid x dr) over the cells
  with sigma1 >= the area-weighted ``HIGH_STRESS_PERCENTILE`` percentile.

Deposits and ends (label map only)
----------------------------------
Deposits are the 8-connected components of ``labels == LABEL_CALCIUM`` with
the theta direction treated periodically (row 0 adjoins row n_theta-1).
A deposit is a closed ring (no ends) iff it contains a cycle winding once
around the polar centre (see :func:`_unroll`; touching every A-line is not
sufficient, e.g. for a spiral). Otherwise its samples are unrolled in
theta and the deposit has two end angles at the outer edges of its first
and last unrolled A-line (``theta_i -/+ dtheta/2``). ``t_ca`` is the deposit's maximum per-A-line
sample count x dr and ``r_mid`` the mean radius of its samples. The edge
window is purely angular (3D: axial deposit ends are not treated as edges;
the label map is a single cross-section).

A-line of a cell: ``i = round(phi / dtheta) mod n_theta`` with ``phi`` the
polar angle of the centroid. If that A-line has no calcium (boundary
smoothing of the mesh), the nearest A-line with calcium is used.

Also provided: :func:`predictor_analysis`, which solves the design load case
(lumen pressure ``params.get('loads.ivl_balloon_pressure_atm')``) and returns
the predictors.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from dtivl import params as P
from dtivl.geometry.labels import LABEL_CALCIUM
from dtivl.geometry.polar import PolarLabelMap
from dtivl.materials.constitutive import MaterialSet

EDGE_MIN_WINDOW_DEG = 10.0     #: minimum angular half-window around deposit ends (design §4)
HIGH_STRESS_PERCENTILE = 90.0  #: percentile defining "high-stress" cells
SIGMA1_PERCENTILE = 95.0       #: percentile reported as sigma1_p95
SIGMA1_PERCENTILE_HIGH = 99.0  #: percentile reported as sigma1_p99

PREDICTOR_KEYS = ("sigma1_peak", "sigma1_p95", "sed_mean", "sed_peak", "frac_eps_gt_half_epsf",
                  "frac_eps_gt_epsf", "frac_sig_gt_half_sigc", "frac_sig_gt_sigc", "scf_edge",
                  "has_edges", "min_thickness_high_stress", "sigma1_p99", "sigma1_peak_nl",
                  "sed_peak_nl", "scf_edge_nl")


def nonlocal_radius() -> float:
    """Averaging radius of the nonlocal peaks: nominal phase-field length (mm)."""
    return float(P.raw("fracture.ell_over_h")["value"]) * P.get("discretisation.h_calcium_mm").value


def nonlocal_average(values: np.ndarray, weights: np.ndarray, centroids: np.ndarray, radius: float) -> np.ndarray:
    """Area-weighted mean of ``values`` over cells within ``radius`` of each cell centroid."""
    tree = cKDTree(centroids)
    out = np.empty_like(values, dtype=float)
    for i, nb in enumerate(tree.query_ball_point(centroids, r=radius)):
        nb = np.asarray(nb, dtype=int)
        out[i] = float((values[nb] * weights[nb]).sum() / weights[nb].sum())
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Weighted quantile (``q`` in [0, 1]) by the inverted weighted CDF."""
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    if values.size == 0:
        return math.nan
    order = np.argsort(values)
    cw = np.cumsum(weights[order])
    k = int(np.searchsorted(cw, q * cw[-1], side="left"))
    return float(values[order][min(k, values.size - 1)])


def _unroll(mask: np.ndarray) -> tuple[bool, int, int]:
    """Winding test and unrolled A-line extent of one theta-periodic component.

    ``mask`` (n_theta, n_r) holds one 8-connected component (theta periodic).
    Every sample gets an *unrolled* A-line index U (A-line index plus a
    multiple of n_theta) by breadth-first propagation along the 8-neighbour
    graph (a step to the next row adds +1, across the seam included). The
    component encloses the lumen (closed ring) iff some edge is inconsistent
    with U, i.e. a cycle winds around the centre. Otherwise its angular
    extent is the unrolled interval [U.min(), U.max()] (this also covers a
    spiral that touches every A-line without closing). Returns
    ``(closed, U.min(), U.max())``.
    """
    from scipy import sparse
    from scipy.sparse import csgraph

    n, n_r = mask.shape
    ii, jj = np.nonzero(mask)
    N = ii.size
    idx = np.full(mask.shape, -1, dtype=np.int64)
    idx[ii, jj] = np.arange(N)
    p_l, q_l, d_l = [], [], []
    for di, dj in ((0, 1), (1, -1), (1, 0), (1, 1)):     # forward half of the 8-neighbourhood
        i2, j2 = (ii + di) % n, jj + dj
        ok = (j2 >= 0) & (j2 < n_r)
        q = np.full(N, -1, dtype=np.int64)
        q[ok] = idx[i2[ok], j2[ok]]
        sel = np.nonzero(q >= 0)[0]
        p_l.append(sel), q_l.append(q[sel]), d_l.append(np.full(sel.size, di))
    p, q, d = np.concatenate(p_l), np.concatenate(q_l), np.concatenate(d_l)
    G = sparse.coo_matrix((np.ones(p.size), (p, q)), shape=(N, N)).tocsr()
    order, pred = csgraph.breadth_first_order(G, 0, directed=False, return_predecessors=True)
    if order.size != N:
        raise ValueError("mask is not a single connected component")
    step = ((ii - ii[np.maximum(pred, 0)] + 1) % n) - 1    # row offset pred -> node in {-1, 0, 1}
    U = np.zeros(N, dtype=np.int64)
    U_l, pred_l, step_l = U.tolist(), pred.tolist(), step.tolist()
    for k in order[1:].tolist():
        U_l[k] = U_l[pred_l[k]] + step_l[k]
    U = np.asarray(U_l, dtype=np.int64)
    closed = bool(np.any(U[q] - U[p] != d))
    return closed, int(U.min()), int(U.max())


def deposits(plm: PolarLabelMap) -> list[dict]:
    """Calcium deposits of a label map (theta-periodic 8-connectivity).

    Each entry: ``alines`` (bool, n_theta), ``closed`` (ring), ``ends``
    (list of end angles, rad), ``t_ca`` (mm), ``r_mid`` (mm), ``n_samples``.
    """
    ca = plm.labels == LABEL_CALCIUM
    lab, n = ndimage.label(ca, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return []
    parent = np.arange(n + 1)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    first, last = lab[0], lab[-1]
    for j in range(plm.n_r):
        if first[j] == 0:
            continue
        for dj in (-1, 0, 1):
            jj = j + dj
            if 0 <= jj < plm.n_r and last[jj] > 0:
                ra, rb = find(first[j]), find(last[jj])
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([find(k) for k in range(n + 1)])
    lab = roots[lab]
    dth = 2.0 * np.pi / plm.n_theta
    out = []
    for k in np.unique(lab[lab > 0]):
        m = lab == k
        counts = m.sum(axis=1)
        alines = counts > 0
        closed, u0, u1 = _unroll(m)
        ends = [] if closed else [u0 * dth - 0.5 * dth, u1 * dth + 0.5 * dth]
        _, jj = np.nonzero(m)
        out.append({"alines": alines, "closed": closed, "ends": ends,
                    "t_ca": float(counts.max() * plm.dr), "r_mid": float(plm.r[jj].mean()),
                    "n_samples": int(m.sum())})
    return out


def _angdist(a, b):
    """Absolute circular angular distance (rad)."""
    return np.abs((np.asarray(a) - b + np.pi) % (2.0 * np.pi) - np.pi)


def local_thickness(plm: PolarLabelMap, phi: np.ndarray) -> np.ndarray:
    """Calcium thickness (count x dr) of the A-line nearest to each angle ``phi``.

    Falls back to the nearest A-line with calcium; NaN if the map has none.
    """
    n = plm.n_theta
    thick = (plm.labels == LABEL_CALCIUM).sum(axis=1) * plm.dr
    idx = np.mod(np.rint(np.mod(phi, 2 * np.pi) / (2 * np.pi / n)).astype(int), n)
    has = np.nonzero(thick > 0)[0]
    if has.size == 0:
        return np.full(len(phi), math.nan)
    out = thick[idx].astype(float)
    bad = out == 0
    if bad.any():
        d = np.abs(idx[bad][:, None] - has[None, :])
        d = np.minimum(d, n - d)
        out[bad] = thick[has[np.argmin(d, axis=1)]]
    return out


# ---------------------------------------------------------------------------
# Predictors
# ---------------------------------------------------------------------------
def predictors(linres, plm: PolarLabelMap, ms: MaterialSet) -> dict:
    """Scalar fracture predictors from a linear result (see module docstring)."""
    s1, e1, sed, w = linres.sigma1, linres.eps1, linres.sed, linres.measure
    deps = deposits(plm)
    has_edges = any(not d["closed"] for d in deps)
    out = {k: math.nan for k in PREDICTOR_KEYS}
    out["has_edges"] = bool(has_edges)
    if s1.size == 0:
        return out
    wt = w.sum()
    frac = lambda mask: float(w[mask].sum() / wt)  # noqa: E731
    sig_c, eps_f = ms.sigma_crit, ms.eps_f
    out.update({
        "sigma1_peak": float(s1.max()),
        "sigma1_p95": weighted_quantile(s1, w, SIGMA1_PERCENTILE / 100.0),
        "sed_mean": float((sed * w).sum() / wt),
        "sed_peak": float(sed.max()),
        "frac_eps_gt_half_epsf": frac(e1 > 0.5 * eps_f),
        "frac_eps_gt_epsf": frac(e1 > eps_f),
        "frac_sig_gt_half_sigc": frac(s1 > 0.5 * sig_c),
        "frac_sig_gt_sigc": frac(s1 > sig_c),
    })
    s1_mean = float((s1 * w).sum() / wt)
    out["sigma1_mean"] = s1_mean
    out["ca_measure_fe"] = float(wt)
    out["sigma1_p99"] = weighted_quantile(s1, w, SIGMA1_PERCENTILE_HIGH / 100.0)
    rho = nonlocal_radius()
    s1_nl = nonlocal_average(s1, w, linres.centroid, rho)
    out["sigma1_peak_nl"] = float(s1_nl.max())
    out["sed_peak_nl"] = float(nonlocal_average(sed, w, linres.centroid, rho).max())
    out["rho_nl"] = rho

    phi = np.arctan2(linres.centroid[:, 1], linres.centroid[:, 0])
    if not has_edges:
        out["scf_edge"] = 1.0
        out["scf_edge_nl"] = 1.0
        out["n_edge_cells"] = 0
    else:
        near = np.zeros(s1.size, dtype=bool)
        for d in deps:
            win = max(math.radians(EDGE_MIN_WINDOW_DEG), 2.0 * d["t_ca"] / d["r_mid"])
            for th_end in d["ends"]:
                near |= _angdist(phi, th_end) <= win
        out["n_edge_cells"] = int(near.sum())
        out["scf_edge"] = float(s1[near].max() / s1_mean) if near.any() and s1_mean > 0 else math.nan
        out["scf_edge_nl"] = float(s1_nl[near].max() / s1_mean) if near.any() and s1_mean > 0 else math.nan

    thr = weighted_quantile(s1, w, HIGH_STRESS_PERCENTILE / 100.0)
    high = s1 >= thr
    out["min_thickness_high_stress"] = float(np.min(local_thickness(plm, phi[high])))
    return out


def predictor_analysis(meshdata, plm: PolarLabelMap, ms: MaterialSet | None = None, *,
                       p_lumen: float | None = None, k_found: float | None = None,
                       degree: int = 2) -> dict:
    """Solve the predictor load case and return :func:`predictors` plus metadata.

    ``p_lumen`` defaults to ``params.get('loads.ivl_balloon_pressure_atm')``
    (MPa). Extra keys: ``p_lumen_MPa``, ``foundation_ratio``, ``ndofs``.
    """
    from dtivl.solvers.linear import solve_linear

    ms = MaterialSet.baseline() if ms is None else ms
    if p_lumen is None:
        p_lumen = P.get("loads.ivl_balloon_pressure_atm").value
    res = solve_linear(meshdata, ms, p_lumen, k_found=k_found, degree=degree)
    out = predictors(res, plm, ms)
    out.update({"p_lumen_MPa": float(p_lumen), "foundation_ratio": res.foundation_ratio,
                "ndofs": res.info["ndofs"]})
    return out

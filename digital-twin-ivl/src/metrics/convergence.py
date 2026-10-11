"""Grid convergence index (GCI) following Celik et al. (2008), J. Fluids Eng. 130:078001.

Three solutions phi1 (finest), phi2, phi3 (coarsest) on grids with
representative sizes h1 < h2 < h3.
"""
from __future__ import annotations

import math

SAFETY_FACTOR = 1.25      # Celik et al. recommendation for three-grid studies
_MAX_ITER = 200
_TOL = 1e-10


def representative_size(measure_total: float, n_cells: int, dim: int) -> float:
    """h = (total area or volume / number of cells)^(1/dim)."""
    return (measure_total / n_cells) ** (1.0 / dim)


def gci_three_grid(h: tuple[float, float, float], phi: tuple[float, float, float]) -> dict:
    """Celik three-grid procedure.

    Parameters
    ----------
    h : (h1, h2, h3) with h1 < h2 < h3.
    phi : (phi1, phi2, phi3) solutions on those grids.

    Returns a dict with apparent order ``p``, ``phi_ext`` (Richardson
    extrapolation from the two finest grids), relative errors ``e_a21``,
    ``e_ext21`` and ``gci_fine21`` (fractions), ``r21``, ``r32``,
    ``convergence`` in {"monotonic", "oscillatory", "divergent", "converged"}.
    """
    h1, h2, h3 = h
    f1, f2, f3 = phi
    if not (h1 < h2 < h3):
        raise ValueError("grid sizes must satisfy h1 < h2 < h3")
    r21, r32 = h2 / h1, h3 / h2
    e21, e32 = f2 - f1, f3 - f2
    out = {"h": list(h), "phi": list(phi), "r21": r21, "r32": r32}
    if e21 == 0.0 or e32 == 0.0:
        out.update(p=float("nan"), phi_ext=f1, e_a21=abs(e21 / f1) if f1 else 0.0,
                   e_ext21=0.0, gci_fine21=0.0, convergence="converged")
        return out
    ratio = e32 / e21
    s = 1.0 if ratio > 0 else -1.0
    p = abs(math.log(abs(ratio))) / math.log(r21)  # initial guess (q = 0)
    for _ in range(_MAX_ITER):
        try:
            q = math.log((r21 ** p - s) / (r32 ** p - s))
        except (ValueError, ZeroDivisionError):
            q = 0.0
        p_new = abs(math.log(abs(ratio)) + q) / math.log(r21)
        if abs(p_new - p) < _TOL:
            p = p_new
            break
        p = p_new
    if abs(e21 / e32) > 1.0:          # Celik R = e21/e32; |R| > 1 -> divergence
        conv = "divergent"
    elif s < 0:
        conv = "oscillatory"
    else:
        conv = "monotonic"
    denom = r21 ** p - 1.0
    phi_ext = (r21 ** p * f1 - f2) / denom if denom != 0 else float("nan")
    e_a21 = abs((f1 - f2) / f1) if f1 != 0 else float("inf")
    e_ext21 = abs((phi_ext - f1) / phi_ext) if phi_ext not in (0.0,) and math.isfinite(phi_ext) else float("nan")
    gci = SAFETY_FACTOR * e_a21 / denom if denom > 0 else float("nan")
    out.update(p=p, phi_ext=phi_ext, e_a21=e_a21, e_ext21=e_ext21, gci_fine21=gci, convergence=conv)
    return out

"""Stent outcome metrics (design.md section 5.5).

All inputs are plain numbers / arrays (no dolfinx), so the metrics can be
recomputed from stored results. Lengths in mm, areas in mm^2, pressures in
MPa.

Definitions
-----------
* ``lumen_area_pre`` / ``lumen_area_post``: lumen area before stenting
  (reference configuration of the stent run) and after recoil (stent at
  ``R_final``); ``lumen_gain = post - pre``.
* ``msa = pi R_final^2`` (minimum stent area of the idealised circular
  stent); ``stent_expansion_pct = msa / (pi (D_ref/2)^2) x 100`` with
  ``D_ref = geometry.reference_diameter_mm``.
* ``acute_recoil_pct = (R_dep - R_final) / R_dep x 100``.
* ``malapposition_area``: area of the deformed lumen polygon outside the
  circle of radius ``R_final + gap`` (``gap = loads.malapposition_gap_mm``),
  i.e. lumen farther than the gap threshold from the stent.
* ``malapposed_frac``: fraction of ``MALAPPOSITION_N_RAYS`` equally spaced
  rays from the axis along which the wall (nearest crossing of the lumen
  boundary to the stent circle) lies more than ``gap`` outside the stent.
* ``reached_target``: the expansion reached the target radius without
  exceeding the rated pressure; ``max_contact_pressure_mean``: largest
  balloon-equivalent mean contact pressure during the expansion.
"""
from __future__ import annotations

import math

import numpy as np

from dtivl import params as P

#: Number of rays used for the malapposed circumference fraction.
MALAPPOSITION_N_RAYS = 720
#: Segments per quarter circle of the stent-circle polygon.
CIRCLE_QUAD_SEGS = 256


def polygon_area(xy: np.ndarray) -> float:
    """Absolute shoelace area of a closed polyline (n, 2) (last point not repeated)."""
    xy = np.asarray(xy, dtype=float)
    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def wall_radius_along_rays(lumen_xy: np.ndarray, theta: np.ndarray, R_ref: float) -> np.ndarray:
    """Distance from the axis of the lumen-boundary crossing nearest to ``R_ref`` along each ray."""
    import shapely

    poly = shapely.Polygon(np.asarray(lumen_xy, dtype=float))
    ring = poly.exterior
    rmax = 4.0 * max(float(np.hypot(*np.asarray(lumen_xy).T).max()), R_ref)
    out = np.full(theta.size, np.nan)
    rays = shapely.linestrings(np.stack([np.zeros((theta.size, 2)),
                                         np.column_stack([rmax * np.cos(theta), rmax * np.sin(theta)])], axis=1))
    inter = shapely.intersection(rays, ring)
    for k, g in enumerate(inter):
        pts = shapely.get_coordinates(g)
        if pts.size:
            r = np.hypot(pts[:, 0], pts[:, 1])
            out[k] = r[np.argmin(np.abs(r - R_ref))]
    return out


def malapposition(lumen_xy: np.ndarray, R_stent: float, gap: float,
                  n_rays: int = MALAPPOSITION_N_RAYS) -> tuple[float, float]:
    """``(malapposition_area, malapposed_frac)`` of a lumen polygon around a circular stent."""
    import shapely

    poly = shapely.make_valid(shapely.Polygon(np.asarray(lumen_xy, dtype=float)))
    circle = shapely.Point(0.0, 0.0).buffer(R_stent + gap, quad_segs=CIRCLE_QUAD_SEGS)
    area = float(poly.difference(circle).area)
    theta = 2.0 * math.pi * (np.arange(n_rays) + 0.5) / n_rays
    rw = wall_radius_along_rays(lumen_xy, theta, R_stent)
    frac = float(np.mean(np.nan_to_num(rw, nan=R_stent) - R_stent > gap))
    return area, frac


def stent_metrics(*, lumen_area_pre: float, lumen_area_post: float, lumen_post_xy: np.ndarray | None,
                  R_dep: float, R_final: float, reached_target: bool, max_contact_pressure_mean: float,
                  gap: float | None = None, reference_diameter: float | None = None) -> dict:
    """Stent metrics of design.md section 5.5 (see module docstring).

    ``gap`` defaults to ``loads.malapposition_gap_mm`` and
    ``reference_diameter`` to ``geometry.reference_diameter_mm``.
    ``lumen_post_xy`` (ordered deformed lumen polyline after recoil) is
    needed for the malapposition metrics (NaN if ``None``).
    """
    gap = P.get("loads.malapposition_gap_mm").value if gap is None else float(gap)
    D_ref = P.get("geometry.reference_diameter_mm").value if reference_diameter is None else float(reference_diameter)
    msa = math.pi * R_final ** 2
    if lumen_post_xy is not None and len(lumen_post_xy) >= 3:
        mal_area, mal_frac = malapposition(lumen_post_xy, R_final, gap)
    else:
        mal_area = mal_frac = float("nan")
    return {
        "lumen_area_pre": float(lumen_area_pre),
        "lumen_area_post": float(lumen_area_post),
        "lumen_gain": float(lumen_area_post - lumen_area_pre),
        "msa": float(msa),
        "stent_expansion_pct": float(msa / (math.pi * (0.5 * D_ref) ** 2) * 100.0),
        "acute_recoil_pct": float((R_dep - R_final) / R_dep * 100.0) if R_dep > 0 else float("nan"),
        "malapposition_area": float(mal_area),
        "malapposed_frac": float(mal_frac),
        "reached_target": bool(reached_target),
        "max_contact_pressure_mean": float(max_contact_pressure_mean),
        "R_dep": float(R_dep),
        "R_final": float(R_final),
        "malapposition_gap_mm": gap,
        "reference_diameter_mm": D_ref,
    }

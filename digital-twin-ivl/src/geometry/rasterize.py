"""Rasterisation of an idealised lesion into an IVOCT-like polar label map.

Each polar sample centre ``(r_j, theta_i)`` (``r_j = (j+0.5)*dr``,
``theta_i = 2*pi*i/n_theta``) is labelled by vectorised point-in-polygon
tests (``shapely.contains_xy``) against the exact geometry from
:func:`dtivl.geometry.build.lesion_polygons`:
lumen -> ``LABEL_LUMEN``, calcium -> ``LABEL_CALCIUM``, lipid ->
``LABEL_LIPID``, everything else (fibrous plaque, wall, and the few samples
beyond the outer wall at the end of short A-lines) -> ``LABEL_OTHER``.

The outer wall and IEL are not labels; their radii per A-line are the exact
ray-circle intersections (the circles may be offset from the polar centre).
The map has ``n_r = ceil(max(r_out)/dr) + RADIAL_PAD_SAMPLES`` columns so that
every A-line reaches its outer wall.
"""
from __future__ import annotations

import math

import numpy as np
import shapely

from dtivl import params

from .build import iel_center, iel_radius, lesion_polygons, outer_radius
from .labels import LABEL_CALCIUM, LABEL_LIPID, LABEL_LUMEN, LABEL_OTHER
from .polar import PolarLabelMap
from .spec import LesionSpec

#: extra radial samples beyond the largest outer-wall radius
RADIAL_PAD_SAMPLES = 2


def ray_circle_radius(theta: np.ndarray, center: tuple[float, float], radius: float) -> np.ndarray:
    """Distance from the origin to a circle along rays at angles ``theta`` (rad).

    Solves ``|s e(theta) - c| = R`` for the positive root
    ``s = c.e + sqrt((c.e)^2 - |c|^2 + R^2)``; the origin must lie inside the
    circle.
    """
    theta = np.asarray(theta, dtype=float)
    cx, cy = center
    c2 = cx * cx + cy * cy
    if c2 >= radius * radius:
        raise ValueError("ray_circle_radius: the origin must lie inside the circle")
    ce = cx * np.cos(theta) + cy * np.sin(theta)
    return ce + np.sqrt(ce * ce - c2 + radius * radius)


def default_resolution() -> tuple[int, float]:
    """(n_theta, dr) from ``discretisation.polar_n_theta`` / ``polar_dr_mm``."""
    return (int(params.raw("discretisation.polar_n_theta")),
            float(params.raw("discretisation.polar_dr_mm")))


def rasterize_polar(spec: LesionSpec, n_theta: int | None = None, dr: float | None = None) -> PolarLabelMap:
    """Rasterise ``spec`` to a :class:`PolarLabelMap` (defaults from config)."""
    d_nt, d_dr = default_resolution()
    n_theta = int(n_theta or d_nt)
    dr = float(dr or d_dr)
    geo = lesion_polygons(spec)

    theta = 2.0 * np.pi * np.arange(n_theta) / n_theta
    c = iel_center(spec)
    r_out = ray_circle_radius(theta, c, outer_radius(spec))
    r_iel = ray_circle_radius(theta, c, iel_radius(spec))

    n_r = int(math.ceil(r_out.max() / dr)) + RADIAL_PAD_SAMPLES
    r = (np.arange(n_r) + 0.5) * dr
    X = r[None, :] * np.cos(theta)[:, None]
    Y = r[None, :] * np.sin(theta)[:, None]

    labels = np.full((n_theta, n_r), LABEL_OTHER, dtype=np.uint8)
    for key, lab in (("lipid", LABEL_LIPID), ("calcium", LABEL_CALCIUM), ("lumen", LABEL_LUMEN)):
        g = geo[key]
        if g.is_empty:
            continue
        shapely.prepare(g)
        labels[shapely.contains_xy(g, X, Y)] = lab

    plm = PolarLabelMap(labels=labels, dr=dr, r_out=r_out, r_iel=r_iel,
                        meta={"lesion_id": spec.lesion_id, "source": "synthetic",
                              "set_name": spec.set_name, "spec": spec.to_dict()})
    plm.validate()
    return plm

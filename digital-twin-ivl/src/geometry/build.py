"""Exact (shapely) geometry of an idealised calcified lesion cross-section.

``lesion_polygons(spec)`` turns a :class:`~dtivl.geometry.spec.LesionSpec`
into shapely regions in the Cartesian frame whose origin is the lumen /
polar centre (design.md section 1-2). All regions are *filled*:

* ``'outer'`` - disc bounded by the outer vessel wall (IEL circle grown by
  ``wall_thickness``; same, possibly offset, centre as the IEL);
* ``'iel'``   - disc bounded by the internal elastic lamina, radius
  ``r_lumen + plaque_thickness``, centre shifted by ``plaque_offset`` toward
  ``ca_center_deg``;
* ``'lumen'`` - the lumen (circle of radius ``r_lumen``; for nodules the
  circle minus the covered protrusion);
* ``'calcium'`` - MultiPolygon of calcium deposits;
* ``'lipid'`` - MultiPolygon (empty when ``lipid_arc_deg == 0``).

Tissue regions follow by set difference (fibrous = iel - lumen - calcium -
lipid, wall = outer - iel).

Geometric conventions (implementation choices, see also ``LesionSpec``)
---------------------------------------------------------------------
* Circles are sampled with ``N_CIRCLE`` (>= 1440) vertices; arcs with the
  same angular density; each semicircular calcium cap with ``N_CAP`` vertices
  (i.e. 2*N_CAP per full cap circle).
* Calcium deposit with angular extent ``[lo, hi]``, inner radius
  ``r1 = r_lumen + ca_depth`` and outer radius ``r2 = r1 + ca_thickness``:
  an annular sector whose two ends are semicircular caps of radius
  ``rho = ca_thickness/2`` centred on the mid radius ``r_m = (r1+r2)/2`` at
  angles ``lo + delta`` and ``hi - delta`` with ``delta = asin(rho/r_m)``. The
  caps are therefore tangent to the rays at ``lo`` and ``hi``: the deposit's
  angular extent seen from the catheter is exactly ``hi - lo``. Area =
  ``2*rho*r_m*(hi - lo - 2*delta) + pi*rho**2``.
* A 360 degree sheet (non-fragmented) is a closed ring (no ends).
* Fragmented: ``n = spec.n_fragments`` deposits on an equal pitch
  ``ca_arc_deg / n`` spanning the calcium arc; each deposit spans
  ``fragment_fill * pitch`` centred in its pitch interval.
* Nodule: the calcium inner surface follows
  ``r_ci(phi) = r1 - nodule_height * b(phi)`` with the C1 bump
  ``b = cos^2(pi*dphi/nodule_base_arc)`` for ``|dphi| <= base_arc/2`` (0
  elsewhere), so the calcium protrudes by ``nodule_height`` at the arc centre.
  The covering layer has uniform *normal* thickness ``ca_depth``: the lumen
  is the set of points of the ``r_lumen`` disc farther than ``ca_depth``
  from the calcium inner surface (computed per ray, exactly up to the curve
  sampling ``NODULE_CURVE_SPACING_MM``). Along the A-line through the apex
  and away from the bump, the radial cover depth is exactly ``ca_depth``;
  on the flanks the radial depth is larger (``ca_depth / cos(slope)``).
* Lipid: plain annular sector (no rounded ends) between
  ``r_lumen + lipid_cap_thickness`` and that plus ``lipid_thickness``.
"""
from __future__ import annotations

import math

import numpy as np
import shapely
from shapely.geometry import MultiPolygon, Polygon

from .spec import LesionSpec

#: vertices per full circle (design.md section 2 requires >= 1440)
N_CIRCLE = 2880
#: vertices per semicircular calcium end cap (full-circle equivalent 1440)
N_CAP = 720
#: maximum arc-length spacing of the calcium-surface samples used to build
#: the nodule covering layer (mm); refined further for very thin covers
NODULE_CURVE_SPACING_MM = 1e-4
#: covers thinner than this (mm) are treated as zero (lumen = calcium surface)
MIN_COVER_MM = 1e-6
#: area tolerance (mm^2) for overlap checks between regions
OVERLAP_AREA_TOL = 1e-8


# ----------------------------------------------------------------- helpers
def _unit(phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.cos(phi), np.sin(phi)


def _n_arc(span_rad: float, n_circle: int = N_CIRCLE) -> int:
    """Number of vertices (incl. both ends) to sample an arc of ``span_rad``."""
    return max(2, int(math.ceil(abs(span_rad) / (2 * math.pi) * n_circle)) + 1)


def circle_coords(center: tuple[float, float], radius: float, n: int = N_CIRCLE) -> np.ndarray:
    """(n, 2) CCW vertices of a circle (closing vertex not repeated)."""
    t = 2 * math.pi * np.arange(n) / n
    return np.column_stack([center[0] + radius * np.cos(t), center[1] + radius * np.sin(t)])


def iel_center(spec: LesionSpec) -> tuple[float, float]:
    """Centre of the IEL and outer-wall circles (offset toward the calcium)."""
    phi = math.radians(spec.ca_center_deg)
    return spec.plaque_offset * math.cos(phi), spec.plaque_offset * math.sin(phi)


def iel_radius(spec: LesionSpec) -> float:
    """Radius of the IEL circle."""
    return spec.r_lumen + spec.plaque_thickness


def outer_radius(spec: LesionSpec) -> float:
    """Radius of the outer vessel-wall circle."""
    return iel_radius(spec) + spec.wall_thickness


def calcium_radii(spec: LesionSpec) -> tuple[float, float]:
    """Inner and outer radius (r1, r2) of the calcium sheet (away from a nodule)."""
    r1 = spec.r_lumen + spec.ca_depth
    return r1, r1 + spec.ca_thickness


def cap_half_angle(spec: LesionSpec) -> float:
    """Angle (rad) between a deposit's angular limit and its cap centre."""
    r1, r2 = calcium_radii(spec)
    r_m, rho = 0.5 * (r1 + r2), 0.5 * (r2 - r1)
    return math.asin(rho / r_m)


def is_ring(spec: LesionSpec) -> bool:
    """True when the calcium is a closed 360 degree ring (no deposit ends)."""
    return spec.morphology != "fragmented" and spec.ca_arc_deg >= 360.0


def deposit_extents_deg(spec: LesionSpec) -> list[tuple[float, float]]:
    """Angular extents ``(lo, hi)`` (deg, lo < hi, unwrapped) of each deposit.

    A closed ring returns ``[(c - 180, c + 180)]``.
    """
    c, arc = spec.ca_center_deg, spec.ca_arc_deg
    if spec.morphology != "fragmented":
        return [(c - arc / 2, c + arc / 2)]
    n = spec.n_fragments
    pitch = arc / n
    width = spec.fragment_fill * pitch
    out = []
    for k in range(n):
        mid = c - arc / 2 + (k + 0.5) * pitch
        out.append((mid - width / 2, mid + width / 2))
    return out


def nodule_bump(spec: LesionSpec, phi: np.ndarray) -> np.ndarray:
    """Normalised protrusion profile b(phi) in [0, 1] (0 for non-nodules)."""
    phi = np.asarray(phi, dtype=float)
    if spec.morphology != "nodule" or spec.nodule_height <= 0:
        return np.zeros_like(phi)
    beta = math.radians(spec.nodule_base_arc_deg)
    d = np.mod(phi - math.radians(spec.ca_center_deg) + math.pi, 2 * math.pi) - math.pi
    return np.where(np.abs(d) <= beta / 2, np.cos(math.pi * d / beta) ** 2, 0.0)


def calcium_inner_radius(spec: LesionSpec, phi: np.ndarray) -> np.ndarray:
    """Radius of the calcium inner (luminal) surface along angle ``phi`` (rad)."""
    r1, _ = calcium_radii(spec)
    return r1 - spec.nodule_height * nodule_bump(spec, phi)


# ----------------------------------------------------------- calcium shapes
def _deposit_polygon(spec: LesionSpec, lo: float, hi: float) -> Polygon:
    """Annular-sector deposit with semicircular end caps (angles in rad)."""
    r1, r2 = calcium_radii(spec)
    r_m, rho = 0.5 * (r1 + r2), 0.5 * (r2 - r1)
    delta = cap_half_angle(spec)
    t0, t1 = lo + delta, hi - delta
    if t1 <= t0:
        raise ValueError(f"{spec.lesion_id}: deposit arc {math.degrees(hi - lo):.2f} deg is shorter "
                         f"than its rounded ends ({math.degrees(2 * delta):.2f} deg)")
    n = _n_arc(t1 - t0)
    # outer arc t0 -> t1 at r2
    phi = np.linspace(t0, t1, n)
    ox, oy = _unit(phi)
    outer = np.column_stack([r2 * ox, r2 * oy])
    # cap at t1: centre r_m*e(t1), from the outer point (psi=0) CCW to the inner point (psi=pi)
    psi = np.linspace(0.0, math.pi, N_CAP + 1)[1:-1]
    c1 = r_m * np.array([math.cos(t1), math.sin(t1)])
    cap1 = np.column_stack([c1[0] + rho * np.cos(t1 + psi), c1[1] + rho * np.sin(t1 + psi)])
    # inner arc t1 -> t0 (nodule profile when applicable)
    phi_in = np.linspace(t1, t0, n)
    ri = calcium_inner_radius(spec, phi_in)
    ix, iy = _unit(phi_in)
    inner = np.column_stack([ri * ix, ri * iy])
    # cap at t0: from the inner point (psi=pi) to the outer point (psi=2pi)
    c0 = r_m * np.array([math.cos(t0), math.sin(t0)])
    psi0 = psi + math.pi
    cap0 = np.column_stack([c0[0] + rho * np.cos(t0 + psi0), c0[1] + rho * np.sin(t0 + psi0)])
    poly = Polygon(np.vstack([outer, cap1, inner, cap0]))
    if not poly.is_valid:
        raise ValueError(f"{spec.lesion_id}: invalid calcium deposit polygon "
                         f"({shapely.is_valid_reason(poly)})")
    return poly


def _ring_polygon(spec: LesionSpec) -> Polygon:
    """Closed calcium ring (360 deg sheet or nodule)."""
    _, r2 = calcium_radii(spec)
    shell = circle_coords((0.0, 0.0), r2)
    phi = 2 * math.pi * np.arange(N_CIRCLE) / N_CIRCLE
    ri = calcium_inner_radius(spec, phi)
    hole = np.column_stack([ri * np.cos(phi), ri * np.sin(phi)])
    return Polygon(shell, [hole[::-1]])


def calcium_geometry(spec: LesionSpec) -> MultiPolygon:
    """MultiPolygon of all calcium deposits of ``spec``."""
    if is_ring(spec):
        return MultiPolygon([_ring_polygon(spec)])
    polys = [_deposit_polygon(spec, math.radians(lo), math.radians(hi))
             for lo, hi in deposit_extents_deg(spec)]
    return MultiPolygon(polys)


# ------------------------------------------------------------------ lumen
def lumen_radius_profile(spec: LesionSpec, phi: np.ndarray) -> np.ndarray:
    """Lumen boundary radius along each angle ``phi`` (rad).

    Equal to ``r_lumen`` except over a nodule, where the lumen boundary is the
    locus at normal distance ``ca_depth`` from the calcium inner surface.
    """
    phi = np.asarray(phi, dtype=float)
    r_lb = np.full(phi.shape, spec.r_lumen)
    if spec.morphology != "nodule" or spec.nodule_height <= 0:
        return r_lb
    depth = spec.ca_depth
    beta = math.radians(spec.nodule_base_arc_deg)
    pc = math.radians(spec.ca_center_deg)
    if depth < MIN_COVER_MM:
        return calcium_inner_radius(spec, phi) - depth
    # dense samples of the calcium inner surface over the bump (+ the cover width)
    r1, _ = calcium_radii(spec)
    span = beta / 2 + 2 * math.asin(min(1.0, depth / max(r1 - spec.nodule_height, depth)))
    spacing = min(NODULE_CURVE_SPACING_MM, depth / 10)
    # curve length bound: arc length at r1 + twice the height (bump sides)
    length = 2 * span * r1 + 2 * spec.nodule_height
    m = max(int(math.ceil(length / spacing)), 2)
    s = np.linspace(pc - span, pc + span, m)
    rc = calcium_inner_radius(spec, s)
    px, py = rc * np.cos(s), rc * np.sin(s)
    p2 = px ** 2 + py ** 2
    d_phi = np.mod(phi - pc + math.pi, 2 * math.pi) - math.pi
    sel = np.abs(d_phi) <= span
    idx = np.nonzero(sel)[0]
    # process in chunks to bound memory
    chunk = max(1, int(4e6 // m))
    for k0 in range(0, idx.size, chunk):
        ii = idx[k0:k0 + chunk]
        ex, ey = np.cos(phi[ii])[:, None], np.sin(phi[ii])[:, None]
        a = px[None, :] * ex + py[None, :] * ey          # projection on the ray
        q2 = p2[None, :] - a ** 2                         # squared distance to the ray
        disc = depth ** 2 - q2
        s_hit = np.where((disc >= 0) & (a > 0), a - np.sqrt(np.clip(disc, 0.0, None)), np.inf)
        r_lb[ii] = np.minimum(r_lb[ii], s_hit.min(axis=1))
    return r_lb


def lumen_geometry(spec: LesionSpec) -> Polygon:
    """Lumen polygon (star-shaped from the origin)."""
    phi = 2 * math.pi * np.arange(N_CIRCLE) / N_CIRCLE
    r = lumen_radius_profile(spec, phi)
    return Polygon(np.column_stack([r * np.cos(phi), r * np.sin(phi)]))


# ------------------------------------------------------------------ lipid
def lipid_geometry(spec: LesionSpec) -> MultiPolygon:
    """Lipid pool as a plain annular sector (empty MultiPolygon if absent)."""
    if spec.lipid_arc_deg <= 0:
        return MultiPolygon()
    r_a = spec.r_lumen + spec.lipid_cap_thickness
    r_b = r_a + spec.lipid_thickness
    if spec.lipid_arc_deg >= 360:
        return MultiPolygon([Polygon(circle_coords((0, 0), r_b), [circle_coords((0, 0), r_a)[::-1]])])
    c, half = math.radians(spec.lipid_center_deg), math.radians(spec.lipid_arc_deg) / 2
    phi = np.linspace(c - half, c + half, _n_arc(2 * half))
    outer = np.column_stack([r_b * np.cos(phi), r_b * np.sin(phi)])
    inner = np.column_stack([r_a * np.cos(phi[::-1]), r_a * np.sin(phi[::-1])])
    return MultiPolygon([Polygon(np.vstack([outer, inner]))])


def _component_containing_origin(geom) -> Polygon:
    """The polygon of ``geom`` that contains the origin (largest as fallback)."""
    parts = list(getattr(geom, "geoms", [geom]))
    origin = shapely.Point(0.0, 0.0)
    for p in parts:
        if p.contains(origin):
            return p
    return max(parts, key=lambda p: p.area)


# ------------------------------------------------------------------ public
def lesion_polygons(spec: LesionSpec) -> dict:
    """Shapely regions of ``spec``: keys outer, iel, lumen, calcium, lipid.

    Raises ``ValueError`` if the spec is invalid or the regions are not nested
    as required (calcium/lipid inside the IEL, no overlap with the lumen or
    with each other).
    """
    spec.validate()
    c = iel_center(spec)
    outer = Polygon(circle_coords(c, outer_radius(spec)))
    iel = Polygon(circle_coords(c, iel_radius(spec)))
    lumen = lumen_geometry(spec)
    calcium = calcium_geometry(spec)
    lipid = lipid_geometry(spec)
    if spec.ca_depth < MIN_COVER_MM:
        # lumen and calcium share a boundary: remove sampling mismatches exactly
        lumen = _component_containing_origin(lumen.difference(calcium))

    if not lumen.is_valid or lumen.area <= 0:
        raise ValueError(f"{spec.lesion_id}: invalid lumen polygon")
    if not calcium.within(iel):
        raise ValueError(f"{spec.lesion_id}: calcium is not inside the IEL")
    if calcium.intersection(lumen).area > OVERLAP_AREA_TOL:
        raise ValueError(f"{spec.lesion_id}: calcium overlaps the lumen")
    if not lipid.is_empty:
        if not lipid.within(iel):
            raise ValueError(f"{spec.lesion_id}: lipid is not inside the IEL")
        if lipid.intersection(lumen).area > OVERLAP_AREA_TOL:
            raise ValueError(f"{spec.lesion_id}: lipid overlaps the lumen")
        if lipid.intersection(calcium).area > OVERLAP_AREA_TOL:
            raise ValueError(f"{spec.lesion_id}: lipid overlaps calcium")
    return {"outer": outer, "iel": iel, "lumen": lumen, "calcium": calcium, "lipid": lipid}


def nominal_descriptors(spec: LesionSpec) -> dict:
    """Descriptor values implied analytically by ``spec`` (for verification).

    Keys match :func:`dtivl.geometry.descriptors.descriptors`:
    ``ca_arc_total_deg``, ``ca_arc_max_contig_deg``, ``ca_thickness_max``,
    ``ca_depth_min``, ``n_deposits``. Angles are angular extents seen from
    the polar centre; the nodule thickness includes the protrusion.
    """
    ext = deposit_extents_deg(spec)
    widths = [hi - lo for lo, hi in ext]
    ring = is_ring(spec)
    t_max = spec.ca_thickness + (spec.nodule_height if spec.morphology == "nodule" else 0.0)
    return {
        "ca_arc_total_deg": 360.0 if ring else float(sum(widths)),
        "ca_arc_max_contig_deg": 360.0 if ring else float(max(widths)),
        "ca_thickness_max": t_max,
        "ca_depth_min": spec.ca_depth,
        "n_deposits": len(ext),
    }

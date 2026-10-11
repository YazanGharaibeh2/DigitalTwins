"""Polar label map -> conforming 2D gmsh mesh -> dolfinx MeshData.

Pipeline (design.md section 3)
------------------------------
1. **Region boundaries from the label map.** For each label class (lumen,
   calcium, lipid) the binary polar indicator is resampled onto a fine
   Cartesian grid (pixel ``CART_PIXEL_FRAC * dr``) by *bilinear*
   interpolation in (theta, r) index space. Because sample ``j`` sits at
   ``r = (j + 0.5) dr`` and A-line ``i`` at ``theta_i``, the 0.5 level of the
   interpolant lies exactly on the polar-pixel edges, so contour areas agree
   with polar-pixel areas ``r dr dtheta``. The resampled field is smoothed by
   an isotropic Gaussian of physical width ``smooth`` (default
   ``SMOOTH_FRAC * dr``) to remove the 1-A-line staircase (which would
   otherwise create spurious corner stress concentrations), contoured with
   ``skimage.measure.find_contours`` at 0.5, assembled into shapely polygons
   (even-odd nesting), ``simplify``-ed (``SIMPLIFY_FRAC * dr``) and made
   valid. Components smaller than ``MIN_AREA_PX`` polar pixels are dropped.
   The outer boundary is the polygon through ``(r_out_i, theta_i)`` and the
   IEL the polygon through ``(r_iel_i, theta_i)``.
2. **Conforming partition.** Precedence: calcium > lipid > lumen > tissue.
   The lumen is the connected lumen component containing the polar centre
   (holes filled, other label-1 islands treated as tissue), minus calcium and
   lipid. Tissue components not connected to the main vessel wall (e.g.
   calcium floating in the lumen) are discarded. All boundaries are snapped
   to a ``GRID_MM`` grid, noded by ``shapely.union_all`` and ``polygonize``-d into
   faces, so neighbouring faces share *identical* vertex chains. Each face is
   classified by ``point_on_surface`` into lumen (not meshed), calcium,
   lipid, fibrous (inside the IEL) or wall (outside the IEL).
3. **gmsh geometry.** The faces are written to the gmsh *built-in* (geo)
   kernel with one point per unique vertex and one line per unique edge, so
   the partition is conforming by construction (equivalent to, but more
   robust than, ``occ.fragment`` on independently drawn loops; see the
   module report). Physical groups: cell tags ``TAG_*``; facet tags
   ``TAG_LUMEN_BND`` (lines bounding exactly one meshed face and not on the
   outer polygon) and ``TAG_OUTER_BND``. Size field: ``h_ca`` inside calcium
   and within ``REFINE_BAND_FACTOR * h_ca`` of it, growing linearly with
   gradient ``SIZE_GRADIENT`` to ``h_soft``. Linear triangles.
4. ``.msh`` (format 4.1) is written and a summary dict is returned.

Real segmentations (``plm.r_iel is None``, labels {0, 1, 4})
-------------------------------------------------------------
The IEL is then ``r_out - wall_thickness`` with ``wall_thickness`` from
``params.get('geometry.wall_thickness_mm')``, unless ``r_iel_default`` is
given explicitly (scalar or per-A-line radius array). That config entry is
currently PENDING, so callers must either pass ``r_iel_default`` or
``allow_pending=True``. Calcium crossing the IEL stays calcium.

gmsh session handling: if gmsh is already initialised by the caller, a
temporary model is created and removed again (the caller's current model is
restored); otherwise gmsh is initialised and finalised here.
"""
from __future__ import annotations

import contextlib
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import shapely
from scipy import ndimage
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import polygonize
from skimage import measure

from dtivl import params as P
from dtivl.geometry.labels import (LABEL_CALCIUM, LABEL_LIPID, LABEL_LUMEN, LABEL_OTHER,
                                   TAG_CALCIUM, TAG_FIBROUS, TAG_LIPID, TAG_LUMEN_BND,
                                   TAG_OUTER_BND, TAG_WALL)
from dtivl.geometry.polar import PolarLabelMap

# ---------------------------------------------------------------------------
# Numerical settings (not physical inputs)
# ---------------------------------------------------------------------------
CART_PIXEL_FRAC = 0.5      # Cartesian resampling pixel = 0.5 dr
SMOOTH_FRAC = 1.5          # Gaussian smoothing width (sigma) = 1.5 dr
SIMPLIFY_FRAC = 0.25       # Douglas-Peucker tolerance = 0.25 dr
MIN_AREA_PX = 4.0          # drop components smaller than 4 dr^2
GRID_MM = 1e-6             # snapping grid for the shapely overlay (mm)
REFINE_BAND_FACTOR = 2.0   # h_ca is kept within 2 h_ca of calcium
SIZE_GRADIENT = 0.3        # mesh size growth (mm of size per mm of distance)
DISTANCE_SAMPLING = 20     # gmsh Distance-field sampling points per curve
MESH_ALGORITHM_2D = 6      # gmsh Frontal-Delaunay

#: Face class used for the (removed) lumen region.
LUMEN = 0


# ---------------------------------------------------------------------------
# Partition (shapely) - shared by the 2D and the extruded 3D mesher
# ---------------------------------------------------------------------------
@dataclass
class Partition:
    """Conforming planar partition of one cross-section (all units mm).

    ``faces`` are polygons whose shared edges have identical vertex chains;
    ``face_class`` holds ``LUMEN`` (0, not meshed) or a cell tag ``TAG_*``.
    ``outer`` / ``iel`` / ``lumen`` / ``calcium`` / ``lipid`` are the region
    geometries the partition was built from.
    """

    faces: list[Polygon]
    face_class: list[int]
    outer: Polygon
    iel: Polygon
    lumen: Polygon
    calcium: MultiPolygon
    lipid: MultiPolygon
    info: dict = field(default_factory=dict)

    def tag_areas(self) -> dict[int, float]:
        """Geometric area per cell tag (lumen excluded)."""
        out: dict[int, float] = {}
        for poly, c in zip(self.faces, self.face_class):
            if c != LUMEN:
                out[c] = out.get(c, 0.0) + poly.area
        return out


def _as_multipolygon(geom) -> MultiPolygon:
    """Return the polygonal part of ``geom`` as a MultiPolygon."""
    if geom is None or geom.is_empty:
        return MultiPolygon()
    if isinstance(geom, Polygon):
        return MultiPolygon([geom])
    if isinstance(geom, MultiPolygon):
        return geom
    polys = [g for g in getattr(geom, "geoms", []) if isinstance(g, (Polygon, MultiPolygon))]
    parts: list[Polygon] = []
    for g in polys:
        parts.extend(g.geoms if isinstance(g, MultiPolygon) else [g])
    return MultiPolygon(parts)


def _drop_small(geom, min_area: float) -> MultiPolygon:
    """Remove components (and holes) smaller than ``min_area``."""
    kept = []
    for p in _as_multipolygon(geom).geoms:
        if p.area < min_area:
            continue
        holes = [h for h in p.interiors if Polygon(h).area >= min_area]
        kept.append(Polygon(p.exterior, holes))
    return MultiPolygon(kept)


def _profile_polygon(radius: np.ndarray, theta: np.ndarray) -> Polygon:
    """Star-shaped polygon through ``(radius_i, theta_i)``."""
    return shapely.make_valid(Polygon(np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])))


def _label_region(plm: PolarLabelMap, mask: np.ndarray, *, pixel: float, smooth: float,
                  simplify_tol: float, min_area: float) -> MultiPolygon:
    """Smooth polygonal region of a binary polar mask (see module docstring)."""
    if not mask.any():
        return MultiPolygon()
    n_theta = plm.n_theta
    ind = mask.astype(np.float64)
    # periodic in theta: append A-line 0 after A-line n-1
    ind = np.vstack([ind, ind[:1]])
    # Cartesian window covering the mask (+ smoothing support)
    j_max = int(np.nonzero(mask.any(axis=0))[0].max())
    rmax = (j_max + 1.0) * plm.dr + 4.0 * smooth + 3.0 * pixel
    n = int(np.ceil(2.0 * rmax / pixel)) + 1
    c = (np.arange(n) - (n - 1) / 2.0) * pixel
    X, Y = np.meshgrid(c, c, indexing="xy")
    R = np.hypot(X, Y)
    T = np.mod(np.arctan2(Y, X), 2.0 * np.pi)
    fi = T / (2.0 * np.pi) * n_theta
    fj = R / plm.dr - 0.5
    img = ndimage.map_coordinates(ind, [fi.ravel(), fj.ravel()], order=1, mode="nearest").reshape(R.shape)
    if smooth > 0:
        img = ndimage.gaussian_filter(img, sigma=smooth / pixel, mode="constant")
    img = np.pad(img, 1, mode="constant", constant_values=0.0)
    rings = []
    for cont in measure.find_contours(img, 0.5):
        if len(cont) < 4:
            continue
        xy = np.column_stack([c[0] + (cont[:, 1] - 1) * pixel, c[0] + (cont[:, 0] - 1) * pixel])
        poly = shapely.make_valid(Polygon(xy))
        if not poly.is_empty and poly.area > 0:
            rings.append(poly)
    geom = Polygon()
    for poly in sorted(rings, key=lambda p: -p.area):
        geom = geom.symmetric_difference(poly)
    geom = shapely.make_valid(geom.simplify(simplify_tol, preserve_topology=True))
    return _drop_small(geom, min_area)


def resolve_r_iel(plm: PolarLabelMap, r_iel_default=None, *, allow_pending: bool = False) -> tuple[np.ndarray, str]:
    """Per-A-line IEL radius and a provenance string.

    Uses ``plm.r_iel`` if present; else ``r_iel_default`` (scalar or
    ``(n_theta,)`` radius array, mm); else ``r_out - wall_thickness`` with
    ``geometry.wall_thickness_mm`` from the study config (PENDING entries
    require ``allow_pending=True``).
    """
    if plm.r_iel is not None:
        return np.asarray(plm.r_iel, dtype=float), "plm.r_iel"
    if r_iel_default is not None:
        r = np.broadcast_to(np.asarray(r_iel_default, dtype=float), (plm.n_theta,)).copy()
        return r, "r_iel_default"
    q = P.get("geometry.wall_thickness_mm", allow_pending=allow_pending)
    return np.asarray(plm.r_out, dtype=float) - q.value, f"r_out - wall_thickness ({q.provenance})"


def build_partition(plm: PolarLabelMap, *, r_iel_default=None, allow_pending: bool = False,
                    smooth: float | None = None, simplify_tol: float | None = None) -> Partition:
    """Build the conforming planar partition of a label map (steps 1-2)."""
    plm.validate()
    dr = plm.dr
    pixel = CART_PIXEL_FRAC * dr
    smooth = SMOOTH_FRAC * dr if smooth is None else smooth
    simplify_tol = SIMPLIFY_FRAC * dr if simplify_tol is None else simplify_tol
    min_area = MIN_AREA_PX * dr * dr
    theta = plm.theta
    r_iel, iel_src = resolve_r_iel(plm, r_iel_default, allow_pending=allow_pending)
    if np.any(r_iel >= plm.r_out):
        raise ValueError("IEL radius must be smaller than r_out on every A-line")

    def snap(g):
        return shapely.set_precision(shapely.make_valid(g), GRID_MM)

    outer = snap(_profile_polygon(np.asarray(plm.r_out, float), theta).simplify(simplify_tol))
    iel = snap(_profile_polygon(r_iel, theta).simplify(simplify_tol)).intersection(outer, grid_size=GRID_MM)
    kw = dict(pixel=pixel, smooth=smooth, simplify_tol=simplify_tol, min_area=min_area)
    ca = _label_region(plm, plm.labels == LABEL_CALCIUM, **kw)
    lip = _label_region(plm, plm.labels == LABEL_LIPID, **kw)
    lum = _label_region(plm, plm.labels == LABEL_LUMEN, **kw)

    ca = snap(ca).intersection(outer, grid_size=GRID_MM)
    lip = snap(lip).intersection(outer, grid_size=GRID_MM).difference(ca, grid_size=GRID_MM)
    origin = Point(0.0, 0.0)
    main = [p for p in _as_multipolygon(lum).geoms if p.contains(origin)]
    if not main:
        raise ValueError("no lumen component contains the polar centre")
    lumen = snap(Polygon(main[0].exterior)).intersection(outer, grid_size=GRID_MM)
    lumen = lumen.difference(ca, grid_size=GRID_MM).difference(lip, grid_size=GRID_MM)
    # keep the lumen component around the origin; keep tissue connected to the wall
    lumen = [p for p in _as_multipolygon(lumen).geoms if p.buffer(dr).contains(origin)]
    if not lumen:
        raise ValueError("lumen vanished after removing calcium/lipid")
    lumen = Polygon(lumen[0].exterior)
    tissue = _as_multipolygon(outer.difference(lumen, grid_size=GRID_MM))
    main_tissue = max(tissue.geoms, key=lambda p: p.area)
    dropped = sum(p.area for p in tissue.geoms) - main_tissue.area
    ca = _drop_small(ca.intersection(main_tissue, grid_size=GRID_MM), min_area)
    lip = _drop_small(lip.intersection(main_tissue, grid_size=GRID_MM), min_area)
    # every region boundary, noded once -> faces share identical edges
    lines = [outer.exterior, iel.exterior, lumen.exterior]
    for g in (ca, lip, iel):
        for p in _as_multipolygon(g).geoms:
            lines.append(p.exterior)
            lines.extend(p.interiors)
    noded = shapely.union_all([LineString(l.coords) for l in lines], grid_size=GRID_MM)
    faces, classes = [], []
    for face in polygonize(noded):
        if face.area <= 0:
            continue
        pt = face.point_on_surface()
        if not main_tissue.contains(pt):
            if lumen.contains(pt):
                faces.append(face)
                classes.append(LUMEN)
            continue  # outside the vessel or a discarded island
        if ca.contains(pt):
            cls = TAG_CALCIUM
        elif lip.contains(pt):
            cls = TAG_LIPID
        elif iel.contains(pt):
            cls = TAG_FIBROUS
        else:
            cls = TAG_WALL
        faces.append(face)
        classes.append(cls)
    info = {"r_iel_source": iel_src, "smooth_mm": smooth, "simplify_tol_mm": simplify_tol,
            "dropped_island_area": float(dropped),
            "n_calcium_components": len(_as_multipolygon(ca).geoms)}
    return Partition(faces, classes, outer, iel, lumen, _as_multipolygon(ca), _as_multipolygon(lip), info)


def polar_pixel_areas(plm: PolarLabelMap, *, r_iel_default=None, allow_pending: bool = False) -> dict:
    """Reference areas from the label map: sum of polar pixels ``r dr dtheta``.

    Returns ``{TAG_*: area, 'lumen': area}`` using the same tissue assignment
    as the mesher (label 0 inside/outside the IEL -> fibrous/wall), counting
    samples with ``r < r_out`` only.
    """
    r_iel, _ = resolve_r_iel(plm, r_iel_default, allow_pending=allow_pending)
    r = plm.r[None, :]
    pa = r * plm.dr * (2.0 * np.pi / plm.n_theta) * np.ones_like(plm.labels, dtype=float)
    inside = r < plm.r_out[:, None]
    lab = plm.labels
    out = {
        TAG_CALCIUM: pa[inside & (lab == LABEL_CALCIUM)].sum(),
        TAG_LIPID: pa[inside & (lab == LABEL_LIPID)].sum(),
        TAG_FIBROUS: pa[inside & (lab == LABEL_OTHER) & (r < r_iel[:, None])].sum(),
        TAG_WALL: pa[inside & (lab == LABEL_OTHER) & (r >= r_iel[:, None])].sum(),
        "lumen": pa[inside & (lab == LABEL_LUMEN)].sum(),
    }
    return {k: float(v) for k, v in out.items()}


# ---------------------------------------------------------------------------
# gmsh helpers
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def gmsh_model(name: str = "dtivl"):
    """Fresh gmsh model; initialises/finalises gmsh only if not already active."""
    import gmsh

    owns = not gmsh.isInitialized()
    if owns:
        gmsh.initialize(interruptible=False)
    prev = None if owns else gmsh.model.getCurrent()
    old_terminal = gmsh.option.getNumber("General.Terminal")
    gmsh.option.setNumber("General.Terminal", 0)
    model_name = f"{name}_{uuid.uuid4().hex[:8]}"
    gmsh.model.add(model_name)
    try:
        yield gmsh
    finally:
        gmsh.model.setCurrent(model_name)
        gmsh.model.remove()
        if owns:
            gmsh.finalize()
        else:
            gmsh.option.setNumber("General.Terminal", old_terminal)
            if prev:
                with contextlib.suppress(Exception):
                    gmsh.model.setCurrent(prev)


@dataclass
class GeoModel:
    """Tags of the 2D geo-kernel entities created from a :class:`Partition`."""

    surfaces: dict[int, list[int]]              # cell tag -> surface tags
    surface_class: list[tuple[int, int]]        # (surface tag, cell tag) in creation order
    lumen_lines: list[int]
    outer_lines: list[int]
    calcium_lines: list[int]
    line_midpoints: dict[int, tuple[float, float]]


def add_partition_geo(gmsh, part: Partition, h_default: float) -> GeoModel:
    """Write the meshed faces of ``part`` into the current model (geo kernel).

    One gmsh point per unique vertex and one line per unique edge, so faces
    sharing an edge share the gmsh line (conforming by construction).
    """
    geo = gmsh.model.geo
    pts: dict[tuple[float, float], int] = {}
    lines: dict[tuple[int, int], int] = {}
    line_faces: dict[int, list[int]] = {}
    mids: dict[int, tuple[float, float]] = {}

    def pt(xy):
        key = (round(xy[0], 9), round(xy[1], 9))
        if key not in pts:
            pts[key] = geo.addPoint(key[0], key[1], 0.0, h_default)
        return pts[key]

    def loop(ring, face_idx):
        coords = list(ring.coords)
        if coords[0] == coords[-1]:
            coords = coords[:-1]
        ids = [pt(c) for c in coords]
        signed = []
        for a, b in zip(ids, ids[1:] + ids[:1]):
            if a == b:
                continue
            if (a, b) in lines:
                t = lines[(a, b)]
                signed.append(t)
            elif (b, a) in lines:
                t = lines[(b, a)]
                signed.append(-t)
            else:
                t = geo.addLine(a, b)
                lines[(a, b)] = t
                signed.append(t)
            line_faces.setdefault(abs(t), []).append(face_idx)
        return geo.addCurveLoop(signed)

    surfaces: dict[int, list[int]] = {}
    surface_class: list[tuple[int, int]] = []
    meshed = [(p, c) for p, c in zip(part.faces, part.face_class) if c != LUMEN]
    for k, (poly, cls) in enumerate(meshed):
        poly = shapely.geometry.polygon.orient(poly, 1.0)
        loops = [loop(poly.exterior, k)] + [loop(h, k) for h in poly.interiors]
        s = geo.addPlaneSurface(loops)
        surfaces.setdefault(cls, []).append(s)
        surface_class.append((s, cls))
    inv = {v: k for k, v in pts.items()}
    for (a, b), t in lines.items():
        pa, pb = inv[a], inv[b]
        mids[t] = (0.5 * (pa[0] + pb[0]), 0.5 * (pa[1] + pb[1]))
    outer_ring = part.outer.exterior
    tol = 10 * GRID_MM
    outer_lines, lumen_lines, ca_lines = [], [], []
    ca_classes = {k for k, (_, c) in enumerate(meshed) if c == TAG_CALCIUM}
    for t, fl in line_faces.items():
        if len(fl) == 1:
            if outer_ring.distance(Point(mids[t])) < tol:
                outer_lines.append(t)
            else:
                lumen_lines.append(t)
        if any(f in ca_classes for f in fl):
            ca_lines.append(t)
    return GeoModel(surfaces, surface_class, sorted(lumen_lines), sorted(outer_lines), sorted(ca_lines), mids)


def add_size_field(gmsh, geom: GeoModel, h_ca: float, h_soft: float) -> int:
    """Background field: ``h_ca`` in/near calcium, growing to ``h_soft``."""
    f = gmsh.model.mesh.field
    fields = []
    ca_surf = geom.surfaces.get(TAG_CALCIUM, [])
    if ca_surf:
        dist = f.add("Distance")
        f.setNumbers(dist, "CurvesList", geom.calcium_lines)
        f.setNumber(dist, "Sampling", DISTANCE_SAMPLING)
        thr = f.add("Threshold")
        f.setNumber(thr, "InField", dist)
        f.setNumber(thr, "SizeMin", h_ca)
        f.setNumber(thr, "SizeMax", h_soft)
        f.setNumber(thr, "DistMin", REFINE_BAND_FACTOR * h_ca)
        f.setNumber(thr, "DistMax", REFINE_BAND_FACTOR * h_ca + (h_soft - h_ca) / SIZE_GRADIENT)
        const = f.add("Constant")
        f.setNumbers(const, "SurfacesList", ca_surf)
        f.setNumber(const, "VIn", h_ca)
        f.setNumber(const, "VOut", h_soft)
        f.setNumber(const, "IncludeBoundary", 1)
        fields += [thr, const]
    else:
        const = f.add("MathEval")
        f.setString(const, "F", repr(float(h_soft)))
        fields.append(const)
    fmin = f.add("Min")
    f.setNumbers(fmin, "FieldsList", fields)
    f.setAsBackgroundMesh(fmin)
    for opt, val in (("Mesh.MeshSizeExtendFromBoundary", 0), ("Mesh.MeshSizeFromPoints", 0),
                     ("Mesh.MeshSizeFromCurvature", 0), ("Mesh.Algorithm", MESH_ALGORITHM_2D),
                     ("Mesh.ElementOrder", 1), ("Mesh.MaxNumThreads2D", 1),
                     ("Mesh.MshFileVersion", 4.1), ("Mesh.MeshSizeMax", h_soft)):
        gmsh.option.setNumber(opt, val)
    return fmin


def node_lookup(gmsh) -> tuple[np.ndarray, np.ndarray]:
    """(tag -> row index array, (n, 3) coordinates) of all mesh nodes."""
    tags, xyz, _ = gmsh.model.mesh.getNodes()
    lookup = np.zeros(int(tags.max()) + 1, dtype=np.int64)
    lookup[tags.astype(np.int64)] = np.arange(tags.size)
    return lookup, np.asarray(xyz).reshape(-1, 3)


def element_measure(gmsh, dim: int, tag: int, nodes) -> tuple[int, float, np.ndarray]:
    """(number of elements, total area/volume, centroids) of one entity's mesh."""
    lookup, xyz = nodes
    types, _, enodes = gmsh.model.mesh.getElements(dim, tag)
    n, total, cents = 0, 0.0, [np.zeros((0, 3))]
    for et, en in zip(types, enodes):
        _, edim, _, nper, _, _ = gmsh.model.mesh.getElementProperties(et)
        if edim != dim:
            continue
        coords = xyz[lookup[np.asarray(en, dtype=np.int64).reshape(-1, nper)]]
        a, b, c = coords[:, 0], coords[:, 1], coords[:, 2]
        if dim == 2:
            total += 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum()
        else:
            d = coords[:, 3]
            total += np.abs(np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a)).sum() / 6.0
        n += coords.shape[0]
        cents.append(coords.mean(axis=1))
    return n, float(total), np.vstack(cents)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def mesh_from_labelmap(plm: PolarLabelMap, out_msh, *, h_ca: float | None = None, h_soft: float | None = None,
                       r_iel_default=None, allow_pending: bool = False, smooth: float | None = None,
                       simplify_tol: float | None = None) -> dict:
    """Conforming 2D triangle mesh of a polar label map, written to ``out_msh``.

    Parameters
    ----------
    plm : the label map.
    out_msh : output ``.msh`` path (parent directory is created).
    h_ca, h_soft : element sizes in/near calcium and elsewhere (mm); defaults
        ``discretisation.h_calcium_mm`` / ``discretisation.h_soft_mm``.
    r_iel_default : IEL radius (scalar or per-A-line array) used when
        ``plm.r_iel is None``; else ``r_out - geometry.wall_thickness_mm``.
    allow_pending : permit PENDING config entries (tests only).
    smooth, simplify_tol : boundary smoothing width and simplification
        tolerance (mm); defaults ``SMOOTH_FRAC * dr`` and ``SIMPLIFY_FRAC * dr``.

    Returns
    -------
    dict with ``msh`` (path), ``gdim`` (2), ``n_nodes``, ``n_cells``,
    ``n_cells_per_tag``, ``areas`` (mesh area per cell tag), ``geom_areas``
    (polygon area per tag), ``pixel_areas`` (polar-pixel reference, see
    :func:`polar_pixel_areas`), ``lumen_area`` (polygon), ``n_lumen_facets``,
    ``n_outer_facets``, ``h_ca``, ``h_soft`` and ``partition_info``.
    """
    h_ca = P.get("discretisation.h_calcium_mm").value if h_ca is None else float(h_ca)
    h_soft = P.get("discretisation.h_soft_mm").value if h_soft is None else float(h_soft)
    part = build_partition(plm, r_iel_default=r_iel_default, allow_pending=allow_pending,
                           smooth=smooth, simplify_tol=simplify_tol)
    out_msh = Path(out_msh)
    out_msh.parent.mkdir(parents=True, exist_ok=True)
    with gmsh_model("mesh2d") as gmsh:
        geom = add_partition_geo(gmsh, part, h_soft)
        gmsh.model.geo.synchronize()
        for tag, surfs in sorted(geom.surfaces.items()):
            gmsh.model.addPhysicalGroup(2, surfs, tag)
        gmsh.model.addPhysicalGroup(1, geom.lumen_lines, TAG_LUMEN_BND)
        gmsh.model.addPhysicalGroup(1, geom.outer_lines, TAG_OUTER_BND)
        add_size_field(gmsh, geom, h_ca, h_soft)
        gmsh.model.mesh.generate(2)
        nodes = node_lookup(gmsh)
        n_per_tag, areas = {}, {}
        for tag, surfs in sorted(geom.surfaces.items()):
            n_per_tag[tag], areas[tag] = 0, 0.0
            for s in surfs:
                n, a, _ = element_measure(gmsh, 2, s, nodes)
                n_per_tag[tag] += n
                areas[tag] += a
        n_nodes = nodes[1].shape[0]
        n_lum = sum(len(gmsh.model.mesh.getElements(1, t)[1][0]) for t in geom.lumen_lines)
        n_out = sum(len(gmsh.model.mesh.getElements(1, t)[1][0]) for t in geom.outer_lines)
        gmsh.write(str(out_msh))
    return {
        "msh": str(out_msh), "gdim": 2, "n_nodes": int(n_nodes),
        "n_cells": int(sum(n_per_tag.values())), "n_cells_per_tag": n_per_tag,
        "areas": areas, "geom_areas": part.tag_areas(),
        "pixel_areas": polar_pixel_areas(plm, r_iel_default=r_iel_default, allow_pending=allow_pending),
        "lumen_area": float(part.lumen.area), "n_lumen_facets": int(n_lum), "n_outer_facets": int(n_out),
        "h_ca": h_ca, "h_soft": h_soft, "partition_info": part.info,
    }


def load_mesh(msh_path, gdim: int, comm=None, rank: int = 0):
    """Read a ``.msh`` written by this package into a dolfinx ``MeshData``.

    Returns ``dolfinx.io.gmsh.MeshData`` (``.mesh``, ``.cell_tags``,
    ``.facet_tags``). Safe to call while the caller holds a gmsh session.
    """
    from mpi4py import MPI
    from dolfinx.io.gmsh import model_to_mesh

    comm = MPI.COMM_WORLD if comm is None else comm
    if comm.rank == rank:
        with gmsh_model("load") as gmsh:
            gmsh.merge(str(msh_path))
            return model_to_mesh(gmsh.model, comm, rank, gdim=gdim)
    import gmsh  # non-root ranks only receive data
    return model_to_mesh(gmsh.model, comm, rank, gdim=gdim)

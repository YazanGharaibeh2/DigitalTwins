"""Crack post-processing on the calcium cells (design.md section 5.5).

:func:`crack_analysis` works on cell-averaged damage (phase field) or on a
per-cell eroded indicator (erosion comparator) and on the lesion's polar
label map, so that synthetic and real segmentations are treated alike.
Serial (single-rank) use.

Definitions
-----------
* Cracked cell: calcium cell with cell-averaged ``d > fracture.d_crack``.
  Crack components: connected components of cracked cells sharing a facet.
* Calcium surfaces: facets of calcium cells whose other side is not
  calcium (or the domain boundary). With ``n`` the facet normal pointing
  out of the calcium and ``e_r`` the in-plane radial direction at the facet
  midpoint (distance from the vessel axis), a facet is *luminal* if
  ``n . e_r <= -SURFACE_NORMAL_COS`` and *abluminal* if
  ``n . e_r >= SURFACE_NORMAL_COS``; other facets (deposit ends, 3D end
  faces) are lateral.
* ``fracture``: some component is through-thickness, i.e. it touches both a
  luminal and an abluminal surface (``spans_surfaces``) AND cuts its
  calcium region (``cuts_deposit``): removing the component's cells from
  the facet-connected calcium region either splits it into more pieces of
  at least ``MIN_PIECE_CELLS`` cells, removes its closed loop around the
  vessel axis (a ring is cut open; detected by the winding of the cell
  centroids along the cell graph), or leaves nothing. The second condition
  excludes damage confined to a deposit end (rounded cap, or the axial end
  in 3D), which touches both surfaces through the end without fracturing
  the deposit. ``n_fractures`` counts the through-thickness components and
  ``n_partial`` the others.
* Depth: on the A-line of each cell centroid (``theta`` -> row of the label
  map) the calcium spans ``[r_in, r_out]`` (first sample start to last
  sample end). A cell covers the normalised depths ``s = (r - r_in) /
  (r_out - r_in)`` of its vertex radii, clipped to [0, 1]. A component's
  ``depth_frac`` is 1 if through-thickness, ``max s`` if it touches only the
  luminal surface, ``1 - min s`` if only the abluminal one, else
  ``max s - min s``. ``max_depth_frac`` is the maximum over components (0
  without cracks). A-lines without calcium in the label map use the
  nearest A-line with calcium.
* ``pattern``: orientation of the crack from the damage *gradient*
  (structure tensor ``S = sum_c |c| grad d (x) grad d`` over the calcium
  cells within ``PATTERN_NEIGHBOUR_ELL`` x ell of the component, in-plane
  components). The crack normal is the principal eigenvector of S and the
  crack line is perpendicular to it; the crack is ``"radial"`` if the crack
  line lies within ``PATTERN_ANGLE_DEG`` of the radial direction at the
  component centroid, else ``"circumferential"``. This replaces the
  centroid-PCA rule of design.md 5.5 (decision 2026-10-11): in thin sheets
  the cracked band is wider than the sheet is thick, so the centroid axis
  is tangential even for a crack that cuts the sheet through its
  thickness. For per-cell damage arrays (erosion), which have no gradient,
  the centroid-PCA rule is kept. ``"undetermined"`` for components with
  fewer than ``MIN_CELLS_PATTERN`` cells or a degenerate tensor.
* ``diffuse``: a component that covers at least ``DIFFUSE_REGION_FRAC`` of
  the measure of its facet-connected calcium region is diffuse damage, not a
  discrete crack: its ``pattern`` is ``"diffuse"`` and its
  ``location_frac`` is ``None``. It still counts in ``n_fractures`` when it
  is through-thickness (the deposit has lost its integrity); ``n_diffuse``
  counts such components and ``damaged_fraction`` is
  ``cracked_measure / ca_measure``. Added 2026-10-12 after the pilot showed
  whole-deposit damage at tissue-level calcium moduli, which the gradient
  rule would label circumferential or radial.
* ``location_frac``: deposits are the wrap-aware 4-connected calcium
  components of the label map; a component is assigned to the deposit whose
  angular interval contains (or is nearest to) its circular-mean angle, and
  ``location_frac = |angle - deposit centre| / deposit half-span`` (0 =
  centre, 1 = end), ``None`` for closed rings.
* ``onset_load``: first ``load`` in ``history`` (step records of
  :mod:`dtivl.solvers.nonlinear`) with ``max_d >= fracture.d_crack``
  (``onset_phase`` gives its protocol phase); ``None`` without history or
  if never reached.
"""
from __future__ import annotations

import math

import numpy as np

from dtivl import params as P
from dtivl.geometry.labels import LABEL_CALCIUM, TAG_CALCIUM

SURFACE_NORMAL_COS = 0.5     #: |n . e_r| threshold for luminal/abluminal facets (60 deg)
PATTERN_ANGLE_DEG = 45.0     #: radial if the principal axis is within 45 deg of e_r
MIN_CELLS_PATTERN = 2        #: smaller components have no pattern
PATTERN_NEIGHBOUR_ELL = 2.0  #: structure-tensor neighbourhood radius in units of the regularisation length
MIN_PIECE_CELLS = 3          #: intact pieces smaller than this do not count as separated fragments
DIFFUSE_REGION_FRAC = 0.5    #: a component covering this fraction of its calcium region is diffuse damage


# ---------------------------------------------------------------------------
# Cell values
# ---------------------------------------------------------------------------

def _cell_gradients(meshdata, d, cells: np.ndarray) -> np.ndarray | None:
    """In-plane gradient of a P1 damage Function per cell (None for arrays/other spaces)."""
    if isinstance(d, np.ndarray) or not hasattr(d, "function_space"):
        return None
    V = d.function_space
    if V.ufl_element().basix_element.degree != 1 or V.dofmap.index_map_bs != 1:
        return None
    mesh = V.mesh
    gdim = mesh.geometry.dim
    xg = mesh.geometry.x
    gdm = mesh.geometry.dofmaps[0]
    vals = d.x.array
    grads = np.zeros((cells.size, gdim))
    for i, c in enumerate(cells):
        dofs = V.dofmap.cell_dofs(int(c))
        geo = gdm[int(c)]
        X = xg[geo, :gdim]
        A = X[1:] - X[0]
        b = vals[dofs[1:]] - vals[dofs[0]]
        grads[i] = np.linalg.lstsq(A, b, rcond=None)[0]
    return grads[:, :2]

def cell_values(meshdata, d) -> np.ndarray:
    """Per-cell values over all local cells from a dolfinx Function or an array.

    A DG0 function is read directly; any other function is cell-averaged by
    a DG0 assembly. An array must have one value per local cell (owned
    cells first, as dolfinx numbers them).
    """
    mesh = meshdata.mesh
    tdim = mesh.topology.dim
    imap = mesh.topology.index_map(tdim)
    n_all = imap.size_local + imap.num_ghosts
    if isinstance(d, np.ndarray) or isinstance(d, (list, tuple)):
        arr = np.asarray(d, dtype=float)
        if arr.shape[0] < imap.size_local:
            raise ValueError(f"per-cell array has {arr.shape[0]} entries, mesh has {imap.size_local} cells")
        out = np.zeros(n_all)
        out[:arr.shape[0]] = arr[:n_all]
        return out
    import ufl
    from dolfinx import fem

    V = d.function_space
    el = V.ufl_element()
    if getattr(el, "embedded_superdegree", None) == 0 and V.dofmap.index_map_bs == 1:
        return d.x.array[V.dofmap.list[:, 0]].astype(float)
    Q0 = fem.functionspace(mesh, ("DG", 0))
    w = ufl.TestFunction(Q0)
    num = fem.assemble_vector(fem.form(d * w * ufl.dx)).array
    vol = fem.assemble_vector(fem.form(w * ufl.dx)).array
    blocks = Q0.dofmap.list[:, 0]
    return num[blocks] / vol[blocks]


# ---------------------------------------------------------------------------
# Label-map helpers
# ---------------------------------------------------------------------------
def calcium_deposits(plm) -> tuple[np.ndarray, list[dict]]:
    """Wrap-aware 4-connected calcium components of a polar label map.

    Returns ``(labels, deposits)``: an int array (n_theta, n_r) with deposit
    ids (0 = no calcium) and per deposit ``{id, closed, theta_center,
    half_span}`` (radians; angular interval of the A-lines it occupies).
    """
    from scipy import ndimage

    mask = plm.labels == LABEL_CALCIUM
    lab, n = ndimage.label(mask)
    if n == 0:
        return lab, []
    parent = np.arange(n + 1)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for j in np.where(mask[0] & mask[-1])[0]:
        a, b = find(lab[0, j]), find(lab[-1, j])
        if a != b:
            parent[max(a, b)] = min(a, b)
    roots = np.array([find(i) for i in range(n + 1)])
    uniq = np.unique(roots[1:])
    remap = np.zeros(n + 1, dtype=int)
    remap[uniq] = np.arange(1, uniq.size + 1)
    lab = remap[roots[lab]]
    lab[~mask] = 0
    nt = plm.n_theta
    dth = 2.0 * math.pi / nt
    deps = []
    for k in range(1, uniq.size + 1):
        rows = np.unique(np.where(lab == k)[0])
        if rows.size == nt:
            deps.append({"id": k, "closed": True, "theta_center": 0.0, "half_span": math.pi})
            continue
        gaps = (np.roll(rows, -1) - rows) % nt
        m = int(np.argmax(gaps))
        first, last = rows[(m + 1) % rows.size], rows[m]
        n_rows = (last - first) % nt + 1
        start = first * dth - 0.5 * dth
        span = n_rows * dth
        deps.append({"id": k, "closed": False, "theta_center": (start + 0.5 * span) % (2 * math.pi),
                     "half_span": 0.5 * span})
    return lab, deps


def _aline_extent(plm) -> tuple[np.ndarray, np.ndarray]:
    """Per-A-line calcium start/end radius (nearest calcium A-line where absent)."""
    mask = plm.labels == LABEL_CALCIUM
    has = mask.any(axis=1)
    if not has.any():
        raise ValueError("label map contains no calcium")
    j0 = np.argmax(mask, axis=1)
    j1 = mask.shape[1] - 1 - np.argmax(mask[:, ::-1], axis=1)
    r_in, r_out = j0 * plm.dr, (j1 + 1) * plm.dr
    if not has.all():
        nt = plm.n_theta
        idx = np.where(has)[0]
        dist = np.abs(np.arange(nt)[:, None] - idx[None, :])
        dist = np.minimum(dist, nt - dist)
        near = idx[np.argmin(dist, axis=1)]
        r_in, r_out = r_in[near], r_out[near]
    return r_in, r_out


def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


# ---------------------------------------------------------------------------
# Mesh helpers
# ---------------------------------------------------------------------------
def _cell_geometry(mesh, cells):
    """Vertex coordinates (n, nv, gdim), centroids and measures of ``cells``."""
    gdim, tdim = mesh.geometry.dim, mesh.topology.dim
    gdm = mesh.geometry.dofmaps[0][:, :tdim + 1]       # vertex nodes (straight-sided approximation)
    X = mesh.geometry.x[gdm[cells]][:, :, :gdim]
    cen = X.mean(axis=1)
    E = X[:, 1:, :] - X[:, :1, :]
    meas = np.abs(np.linalg.det(E)) / math.factorial(gdim)
    return X, cen, meas


def _surface_flags(mesh, ca_cells: np.ndarray, is_ca: np.ndarray):
    """Per calcium cell: touches a luminal / abluminal calcium surface facet."""
    from dolfinx import mesh as dmesh

    tdim = mesh.topology.dim
    fdim = tdim - 1
    mesh.topology.create_connectivity(tdim, fdim)
    mesh.topology.create_connectivity(fdim, tdim)
    c2f = mesh.topology.connectivity(tdim, fdim)
    f2c = mesh.topology.connectivity(fdim, tdim)
    nfc = tdim + 1
    cf = c2f.array.reshape(-1, nfc)[ca_cells]                  # (n, nfc)
    off, arr = f2c.offsets, f2c.array
    cnt = off[1:] - off[:-1]
    first = arr[off[:-1]]
    second = np.where(cnt == 2, arr[np.minimum(off[:-1] + 1, arr.size - 1)], -1)
    C = np.repeat(ca_cells[:, None], nfc, axis=1)
    other = np.where(first[cf] == C, second[cf], first[cf])
    bnd = (other < 0) | ~is_ca[np.maximum(other, 0)]
    lum = np.zeros(ca_cells.size, dtype=bool)
    abl = np.zeros(ca_cells.size, dtype=bool)
    if not bnd.any():
        return lum, abl
    ci, fi = np.nonzero(bnd)
    facets = cf[ci, fi].astype(np.int32)
    gdim = mesh.geometry.dim
    fn = dmesh.entities_to_geometry(mesh, fdim, facets)[:, :fdim + 1]    # vertex nodes
    P_ = mesh.geometry.x[fn][:, :, :gdim]                         # (m, nvf, gdim)
    mid = P_.mean(axis=1)
    if gdim == 2:
        t = P_[:, 1] - P_[:, 0]
        n = np.column_stack([t[:, 1], -t[:, 0]])
    else:
        n = np.cross(P_[:, 1] - P_[:, 0], P_[:, 2] - P_[:, 0])
    n /= np.linalg.norm(n, axis=1)[:, None]
    _, cen, _ = _cell_geometry(mesh, ca_cells[ci])
    flip = np.einsum("ij,ij->i", n, mid - cen) < 0
    n[flip] *= -1.0
    er = np.zeros_like(mid)
    er[:, :2] = mid[:, :2]
    er /= np.linalg.norm(er, axis=1)[:, None]
    s = np.einsum("ij,ij->i", n, er)
    np.logical_or.at(lum, ci, s <= -SURFACE_NORMAL_COS)
    np.logical_or.at(abl, ci, s >= SURFACE_NORMAL_COS)
    return lum, abl


def _calcium_graph(mesh, ca_cells: np.ndarray):
    """Facet adjacency (scipy CSR, symmetric) between the calcium cells ``ca_cells``."""
    from scipy.sparse import coo_matrix

    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim, tdim - 1)
    mesh.topology.create_connectivity(tdim - 1, tdim)
    f2c = mesh.topology.connectivity(tdim - 1, tdim)
    off, arr = f2c.offsets, f2c.array
    interior = np.where(off[1:] - off[:-1] == 2)[0]
    a, b = arr[off[interior]], arr[off[interior] + 1]
    n_all = mesh.topology.index_map(tdim).size_local + mesh.topology.index_map(tdim).num_ghosts
    loc = np.full(n_all, -1, dtype=np.int64)
    loc[ca_cells] = np.arange(ca_cells.size)
    keep = (loc[a] >= 0) & (loc[b] >= 0)
    i, j = loc[a[keep]], loc[b[keep]]
    n = ca_cells.size
    return coo_matrix((np.ones(2 * i.size), (np.r_[i, j], np.r_[j, i])), shape=(n, n)).tocsr()


def _has_axis_loop(graph, nodes: np.ndarray, theta: np.ndarray) -> bool:
    """True if the subgraph on ``nodes`` (one connected piece) winds around the vessel axis.

    The polar angle is unwrapped along a breadth-first spanning tree; an edge
    whose unwrapped angles disagree with its wrapped difference by more than
    pi closes a cycle around the axis.
    """
    from scipy.sparse.csgraph import breadth_first_order

    sub = graph[nodes][:, nodes]
    order, pred = breadth_first_order(sub, 0, directed=False, return_predecessors=True)
    th = theta[nodes]
    phi = np.full(nodes.size, np.nan)
    phi[order[0]] = th[order[0]]
    for k in order[1:]:
        p = pred[k]
        phi[k] = phi[p] + _wrap(th[k] - th[p])
    coo = sub.tocoo()
    jump = np.abs(phi[coo.col] - phi[coo.row] - _wrap(th[coo.col] - th[coo.row]))
    return bool((jump > math.pi).any())


def _cuts_region(graph, region: np.ndarray, removed: np.ndarray, theta: np.ndarray) -> bool:
    """Whether removing ``removed`` from the connected calcium ``region`` cuts it (see module docstring)."""
    from scipy.sparse.csgraph import connected_components

    rest = np.setdiff1d(region, removed)
    if rest.size == 0:
        return True
    ncomp, lab = connected_components(graph[rest][:, rest], directed=False)
    sizes = np.bincount(lab, minlength=ncomp)
    if int((sizes >= MIN_PIECE_CELLS).sum()) >= 2:
        return True
    if _has_axis_loop(graph, region, theta):
        big = np.argmax(sizes)
        return not _has_axis_loop(graph, rest[lab == big], theta)
    return False


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
def crack_analysis(meshdata, d, plm, *, history: list | None = None, d_crack: float | None = None) -> dict:
    """Crack metrics of design.md section 5.5 (see module docstring).

    Parameters
    ----------
    meshdata : MeshData with ``TAG_CALCIUM`` cells.
    d : damage Function (P1 or DG0) or per-cell array (e.g. the erosion
        indicator of :func:`dtivl.fracture.erosion.eroded_indicator`).
    plm : :class:`~dtivl.geometry.polar.PolarLabelMap` of the lesion.
    history : optional step records (for ``onset_load``).
    d_crack : threshold (default ``fracture.d_crack``).
    """
    from scipy.sparse.csgraph import connected_components

    d_crack = float(P.raw("fracture.d_crack")) if d_crack is None else float(d_crack)
    mesh = meshdata.mesh
    tdim = mesh.topology.dim
    imap = mesh.topology.index_map(tdim)
    n_all = imap.size_local + imap.num_ghosts
    ct = meshdata.cell_tags
    ca = ct.find(TAG_CALCIUM)
    ca = ca[ca < imap.size_local].astype(np.int32)
    dc = cell_values(meshdata, d)
    is_ca = np.zeros(n_all, dtype=bool)
    is_ca[ca] = True
    X, cen, meas = _cell_geometry(mesh, ca)
    cracked_mask = dc[ca] > d_crack
    out = {"d_crack": d_crack, "n_cracked_cells": int(cracked_mask.sum()),
           "cracked_measure": float(meas[cracked_mask].sum()), "ca_measure": float(meas.sum())}
    onset = None
    if history:
        onset = next((r for r in history if r.get("max_d", 0.0) >= d_crack), None)
    out["onset_load"] = None if onset is None else float(onset["load"])
    out["onset_phase"] = None if onset is None else onset.get("phase")

    comps = []
    if cracked_mask.any():
        lum, abl = _surface_flags(mesh, ca, is_ca)
        sel = np.where(cracked_mask)[0]                        # indices into ca
        g_ca = _calcium_graph(mesh, ca)
        ncomp, lab = connected_components(g_ca[sel][:, sel], directed=False)
        _, region_of = connected_components(g_ca, directed=False)   # facet-connected calcium regions

        r_in_al, r_out_al = _aline_extent(plm)
        dep_lab, deposits = calcium_deposits(plm)
        th = np.mod(np.arctan2(cen[:, 1], cen[:, 0]), 2 * math.pi)
        ia = np.mod(np.rint(th / (2 * math.pi) * plm.n_theta).astype(int), plm.n_theta)
        rv = np.hypot(X[:, :, 0], X[:, :, 1])
        span = np.maximum(r_out_al[ia] - r_in_al[ia], plm.dr)
        s_lo = np.clip((rv.min(axis=1) - r_in_al[ia]) / span, 0.0, 1.0)
        s_hi = np.clip((rv.max(axis=1) - r_in_al[ia]) / span, 0.0, 1.0)
        cos_pat = math.cos(math.radians(PATTERN_ANGLE_DEG))
        grads = _cell_gradients(meshdata, d, ca)
        if grads is not None:
            from scipy.spatial import cKDTree

            from dtivl.fracture.phasefield import regularisation_length
            tree = cKDTree(cen[:, :2])
            r_nb = PATTERN_NEIGHBOUR_ELL * regularisation_length()
        for k in range(ncomp):
            idx = sel[lab == k]
            w = meas[idx]
            spans = bool(lum[idx].any() and abl[idx].any())
            cuts = False
            region = np.where(region_of == region_of[idx[0]])[0]
            region_frac = float(w.sum() / meas[region].sum())
            diffuse = region_frac >= DIFFUSE_REGION_FRAC
            if spans:
                cuts = _cuts_region(g_ca, region, idx, th)
            through = spans and cuts
            if through:
                depth = 1.0
            elif lum[idx].any():
                depth = float(s_hi[idx].max())
            elif abl[idx].any():
                depth = float(1.0 - s_lo[idx].min())
            else:
                depth = float(s_hi[idx].max() - s_lo[idx].min())
            xy = cen[idx, :2]
            m = (w[:, None] * xy).sum(axis=0) / w.sum()
            pattern, cosang, aniso = "undetermined", float("nan"), float("nan")
            er = m / max(np.linalg.norm(m), 1e-300)
            if idx.size >= MIN_CELLS_PATTERN and grads is not None:
                nb = np.unique(np.concatenate([np.asarray(v, dtype=int) for v in tree.query_ball_point(xy, r=r_nb)]))
                g = grads[nb]
                S = (meas[nb][:, None, None] * np.einsum("ni,nj->nij", g, g)).sum(axis=0)
                ev, evec = np.linalg.eigh(S)
                if ev[-1] > 0:
                    normal = evec[:, -1]
                    line = np.array([-normal[1], normal[0]])
                    cosang = float(abs(line @ er))
                    aniso = float(ev[-1] / max(ev[0], 1e-300))
                    pattern = "radial" if cosang >= cos_pat else "circumferential"
            elif idx.size >= MIN_CELLS_PATTERN:
                dxy = xy - m
                cov = (w[:, None, None] * np.einsum("ni,nj->nij", dxy, dxy)).sum(axis=0) / w.sum()
                ev, evec = np.linalg.eigh(cov)
                axis = evec[:, -1]
                cosang = float(abs(axis @ er))
                aniso = float(ev[-1] / max(ev[0], 1e-300))
                pattern = "radial" if cosang >= cos_pat else "circumferential"
            if diffuse:
                pattern = "diffuse"
            ang = float(np.mod(math.atan2((w * np.sin(th[idx])).sum(), (w * np.cos(th[idx])).sum()), 2 * math.pi))
            locf, dep_id = None, None
            if deposits:
                dist = []
                for dep in deposits:
                    if dep["closed"]:
                        dist.append(0.0)
                    else:
                        dist.append(max(0.0, abs(_wrap(ang - dep["theta_center"])) - dep["half_span"]))
                dep = deposits[int(np.argmin(dist))]
                dep_id = dep["id"]
                if not dep["closed"] and not diffuse:
                    locf = float(min(abs(_wrap(ang - dep["theta_center"])) / dep["half_span"], 1.0))
            comps.append({"n_cells": int(idx.size), "measure": float(w.sum()), "through": through,
                          "spans_surfaces": spans, "cuts_deposit": cuts,
                          "touches_luminal": bool(lum[idx].any()), "touches_abluminal": bool(abl[idx].any()),
                          "depth_frac": depth, "pattern": pattern, "diffuse": bool(diffuse),
                          "region_frac": region_frac, "axis_cos_radial": cosang,
                          "anisotropy": aniso, "angle_deg": math.degrees(ang), "deposit": dep_id,
                          "location_frac": locf})
        comps.sort(key=lambda c: -c["measure"])
    out.update({
        "fracture": any(c["through"] for c in comps),
        "n_fractures": sum(c["through"] for c in comps),
        "n_partial": sum(not c["through"] for c in comps),
        "diffuse": any(c["diffuse"] for c in comps),
        "n_diffuse": sum(c["diffuse"] for c in comps),
        "damaged_fraction": out["cracked_measure"] / out["ca_measure"] if out["ca_measure"] > 0 else 0.0,
        "max_depth_frac": max((c["depth_frac"] for c in comps), default=0.0),
        "pattern": [c["pattern"] for c in comps],
        "location_frac": [c["location_frac"] for c in comps],
        "components": comps,
    })
    return out

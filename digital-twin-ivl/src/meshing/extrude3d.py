"""Extruded 3D segment mesh from a polar label map (design.md section 3).

The conforming 2D partition of :func:`dtivl.meshing.mask_to_mesh.build_partition`
is written to the gmsh built-in kernel at ``z = 0`` and extruded along +z in
three blocks with structured layers (tetrahedra, no recombination):

    block 0: ``[0, m]``           margin, calcium-shaped volumes -> TAG_FIBROUS
    block 1: ``[m, m + L]``       calcified segment, calcium -> TAG_CALCIUM
    block 2: ``[m + L, 2m + L]``  margin, calcium-shaped volumes -> TAG_FIBROUS

Each block extrudes the top surfaces of the previous one, so the three
blocks share their interfaces (conforming). Lipid, fibrous and wall volumes
keep their tag in every block (lipid therefore spans the whole segment).
The cross-sectional mesh (size field ``h_ca``/``h_soft`` as in 2D) is copied
to every layer; the axial layer count per block is
``ceil(block_length / h_axial)``.

Facet tags: ``TAG_LUMEN_BND`` / ``TAG_OUTER_BND`` on the lateral surfaces
extruded from the 2D lumen / outer lines (identified by the midpoint of the
generating 2D segment), ``TAG_END_BND`` on the faces at ``z = 0`` and
``z = 2m + L``.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from dtivl import params as P
from dtivl.geometry.labels import (TAG_CALCIUM, TAG_END_BND, TAG_FIBROUS, TAG_LUMEN_BND,
                                   TAG_OUTER_BND)
from dtivl.geometry.polar import PolarLabelMap

from .mask_to_mesh import (add_partition_geo, add_size_field, build_partition, element_measure,
                           gmsh_model, node_lookup, polar_pixel_areas)

#: Relative tolerance for classifying exterior surfaces (fraction of h_soft).
CLASSIFY_TOL_FRAC = 1e-3


def _extrude_block(gmsh, surfaces: list[int], dz: float, n_layers: int) -> tuple[list[int], list[int]]:
    """Extrude ``surfaces`` by ``dz``; return (top surfaces, volumes) in input order."""
    out = gmsh.model.geo.extrude([(2, s) for s in surfaces], 0.0, 0.0, dz,
                                 numElements=[n_layers], recombine=False)
    tops, vols = [], []
    for k, (dim, tag) in enumerate(out):
        if dim == 3:
            vols.append(tag)
            tops.append(out[k - 1][1])
    if len(vols) != len(surfaces):
        raise RuntimeError("unexpected gmsh extrude output")
    return tops, vols


def mesh3d_from_labelmap(plm: PolarLabelMap, ca_length: float, out_msh, *, margin: float | None = None,
                         h_axial: float | None = None, h_ca: float | None = None, h_soft: float | None = None,
                         r_iel_default=None, allow_pending: bool = False, smooth: float | None = None,
                         simplify_tol: float | None = None) -> dict:
    """Extruded tetrahedral mesh of a calcified segment, written to ``out_msh``.

    Parameters
    ----------
    plm : cross-sectional label map (constant along the calcified length).
    ca_length : axial calcium length ``L`` (mm).
    out_msh : output ``.msh`` path.
    margin : calcium-free length ``m`` at each end; default
        ``geometry.margin_3d_mm``.
    h_axial : axial element size; default ``discretisation.h_axial_3d_mm``.
    h_ca, h_soft, r_iel_default, allow_pending, smooth, simplify_tol : as in
        :func:`dtivl.meshing.mask_to_mesh.mesh_from_labelmap`.

    Returns
    -------
    dict with ``msh``, ``gdim`` (3), ``n_nodes``, ``n_cells``,
    ``n_cells_per_tag``, ``volumes`` (mesh volume per cell tag),
    ``areas_2d`` (cross-section polygon area per tag), ``pixel_areas``,
    ``z_blocks`` ``(0, m, m + L, 2m + L)``, ``n_layers`` per block,
    ``length`` (total), ``ca_length``, ``margin``, ``h_*``, ``lumen_area``
    and ``partition_info``.
    """
    if ca_length <= 0:
        raise ValueError("ca_length must be > 0")
    margin = P.get("geometry.margin_3d_mm").value if margin is None else float(margin)
    h_axial = P.get("discretisation.h_axial_3d_mm").value if h_axial is None else float(h_axial)
    h_ca = P.get("discretisation.h_calcium_mm").value if h_ca is None else float(h_ca)
    h_soft = P.get("discretisation.h_soft_mm").value if h_soft is None else float(h_soft)
    part = build_partition(plm, r_iel_default=r_iel_default, allow_pending=allow_pending,
                           smooth=smooth, simplify_tol=simplify_tol)
    blocks = [margin, float(ca_length), margin] if margin > 0 else [float(ca_length)]
    middle = 1 if margin > 0 else 0
    n_layers = [max(1, math.ceil(b / h_axial - 1e-9)) for b in blocks]
    z_total = float(sum(blocks))
    out_msh = Path(out_msh)
    out_msh.parent.mkdir(parents=True, exist_ok=True)

    with gmsh_model("mesh3d") as gmsh:
        geom = add_partition_geo(gmsh, part, h_soft)
        base = [s for s, _ in geom.surface_class]
        classes = [c for _, c in geom.surface_class]
        vol_tags: dict[int, list[int]] = {}
        current = base
        for b, (dz, nl) in enumerate(zip(blocks, n_layers)):
            current, vols = _extrude_block(gmsh, current, dz, nl)
            for v, c in zip(vols, classes):
                tag = TAG_FIBROUS if (c == TAG_CALCIUM and b != middle) else c
                vol_tags.setdefault(tag, []).append(v)
        gmsh.model.geo.synchronize()

        # exterior surfaces: bounding exactly one volume
        count: dict[int, int] = {}
        for v in gmsh.model.getEntities(3):
            for _, s in gmsh.model.getBoundary([v], oriented=False):
                count[abs(s)] = count.get(abs(s), 0) + 1
        line_ids = geom.lumen_lines + geom.outer_lines
        tree = cKDTree(np.array([geom.line_midpoints[t] for t in line_ids]))
        is_lumen = np.array([True] * len(geom.lumen_lines) + [False] * len(geom.outer_lines))
        tol = CLASSIFY_TOL_FRAC * h_soft
        groups = {TAG_LUMEN_BND: [], TAG_OUTER_BND: [], TAG_END_BND: []}
        for s, n in count.items():
            if n != 1:
                continue
            x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(2, s)
            if z1 - z0 < tol:
                groups[TAG_END_BND].append(s)
                continue
            dist, k = tree.query([0.5 * (x0 + x1), 0.5 * (y0 + y1)])
            if dist > tol:
                raise RuntimeError(f"cannot classify exterior surface {s}")
            groups[TAG_LUMEN_BND if is_lumen[k] else TAG_OUTER_BND].append(s)
        for tag, vols in sorted(vol_tags.items()):
            gmsh.model.addPhysicalGroup(3, vols, tag)
        for tag, surfs in groups.items():
            gmsh.model.addPhysicalGroup(2, sorted(surfs), tag)

        add_size_field(gmsh, geom, h_ca, h_soft)
        gmsh.model.mesh.generate(3)
        nodes = node_lookup(gmsh)
        n_per_tag, volumes = {}, {}
        for tag, vols in sorted(vol_tags.items()):
            n_per_tag[tag], volumes[tag] = 0, 0.0
            for v in vols:
                n, vol, _ = element_measure(gmsh, 3, v, nodes)
                n_per_tag[tag] += n
                volumes[tag] += vol
        gmsh.write(str(out_msh))

    z_blocks = tuple(float(z) for z in np.concatenate([[0.0], np.cumsum(blocks)]))
    return {
        "msh": str(out_msh), "gdim": 3, "n_nodes": int(nodes[1].shape[0]),
        "n_cells": int(sum(n_per_tag.values())), "n_cells_per_tag": n_per_tag,
        "volumes": volumes, "areas_2d": part.tag_areas(),
        "pixel_areas": polar_pixel_areas(plm, r_iel_default=r_iel_default, allow_pending=allow_pending),
        "z_blocks": z_blocks, "n_layers": n_layers, "length": z_total,
        "ca_length": float(ca_length), "margin": margin,
        "h_ca": h_ca, "h_soft": h_soft, "h_axial": h_axial,
        "lumen_area": float(part.lumen.area), "partition_info": part.info,
    }

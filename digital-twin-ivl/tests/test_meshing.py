"""Tests for dtivl.meshing (mask -> conforming gmsh mesh -> dolfinx).

Fixtures are hand-made PolarLabelMaps obtained by evaluating simple shapes
on the polar grid (no dependency on the geometry/cohort modules). The
fixture dimensions below are test-only shapes, not study inputs.
"""
from __future__ import annotations

import numpy as np
import pytest

from dtivl import params as P
from dtivl.geometry.labels import (LABEL_CALCIUM, LABEL_LIPID, LABEL_LUMEN, LABEL_OTHER,
                                   TAG_CALCIUM, TAG_END_BND, TAG_FIBROUS, TAG_LIPID,
                                   TAG_LUMEN_BND, TAG_OUTER_BND, TAG_WALL)
from dtivl.geometry.polar import PolarLabelMap
from dtivl.meshing.extrude3d import mesh3d_from_labelmap
from dtivl.meshing.mask_to_mesh import build_partition, load_mesh, mesh_from_labelmap

N_THETA, DR, N_R = 360, 0.01, 260
R_LUMEN, R_IEL, R_OUT = 0.84, 1.6, 2.3
CA_IN, CA_OUT = 0.89, 1.09            # 0.2 mm sheet at 0.05 mm depth
AREA_RTOL = 0.03

_TH = 2 * np.pi * np.arange(N_THETA) / N_THETA
_R = (np.arange(N_R) + 0.5) * DR
T, R = np.meshgrid(_TH, _R, indexing="ij")


def _angdist(center_deg):
    return np.abs((np.degrees(T) - center_deg + 180.0) % 360.0 - 180.0)


def _base(r_lumen=R_LUMEN):
    lab = np.full((N_THETA, N_R), LABEL_OTHER, np.uint8)
    lab[R < r_lumen] = LABEL_LUMEN
    return lab


def _sector(lab, r0, r1, center, arc, val=LABEL_CALCIUM):
    lab[(R >= r0) & (R < r1) & (_angdist(center) <= arc / 2.0)] = val
    return lab


def make_case(name: str) -> PolarLabelMap:
    """PolarLabelMap fixtures: sheet, ring, fragmented, nodule, lipid, real."""
    r_out = np.full(N_THETA, R_OUT)
    r_iel = np.full(N_THETA, R_IEL)
    if name == "sheet":
        lab = _sector(_base(), CA_IN, CA_OUT, 0.0, 90.0)
    elif name == "ring":
        lab = _sector(_base(), CA_IN, CA_OUT, 0.0, 360.0)
    elif name == "fragmented":
        lab = _base()
        for k in range(4):
            _sector(lab, CA_IN, CA_OUT, -67.5 + 45.0 * k, 27.0)
    elif name == "nodule":          # lumen with an inward calcified protrusion
        bump = 0.3 * np.clip(1.0 - _angdist(0.0) / 30.0, 0.0, 1.0)
        lab = np.full((N_THETA, N_R), LABEL_OTHER, np.uint8)
        lab[R < R_LUMEN - bump] = LABEL_LUMEN
        lab[(R >= CA_IN - bump) & (R < CA_OUT) & (_angdist(0.0) <= 45.0)] = LABEL_CALCIUM
    elif name == "lipid":           # eccentric outer wall + lipid pool
        lab = _sector(_base(), CA_IN, CA_OUT, 0.0, 90.0)
        _sector(lab, 0.92, 1.25, 180.0, 100.0, LABEL_LIPID)
        r_out = R_OUT + 0.15 * np.cos(_TH)
        r_iel = R_IEL + 0.1 * np.cos(_TH)
    elif name == "real":            # labels {0,1,4}, calcium touching lumen, speckle
        lab = _sector(_base(), R_LUMEN, R_LUMEN + 0.3, 90.0, 120.0)
        lab[200, 150] = LABEL_CALCIUM   # isolated 1-pixel speckle (dropped)
        return PolarLabelMap(lab, DR, r_out, None, {"id": name})
    else:
        raise KeyError(name)
    return PolarLabelMap(lab, DR, r_out, r_iel, {"id": name})


def pixel_areas(plm: PolarLabelMap, r_iel=None) -> dict:
    """Independent polar-pixel reference areas per FE tag."""
    r_iel = plm.r_iel if r_iel is None else np.broadcast_to(r_iel, (plm.n_theta,))
    pa = (R * DR * 2 * np.pi / N_THETA)[:, : plm.n_r]
    inside = R < plm.r_out[:, None]
    lab = plm.labels
    other = inside & (lab == LABEL_OTHER)
    return {TAG_CALCIUM: pa[inside & (lab == LABEL_CALCIUM)].sum(),
            TAG_LIPID: pa[inside & (lab == LABEL_LIPID)].sum(),
            TAG_FIBROUS: pa[other & (R < r_iel[:, None])].sum(),
            TAG_WALL: pa[other & (R >= r_iel[:, None])].sum()}


CASES_2D = ["sheet", "ring", "fragmented", "nodule", "lipid"]


@pytest.fixture(scope="module")
def meshes_2d(tmp_path_factory):
    out = tmp_path_factory.mktemp("mesh2d")
    res = {}
    for name in CASES_2D:
        plm = make_case(name)
        res[name] = (plm, mesh_from_labelmap(plm, out / f"{name}.msh"))
    return res


# ---------------------------------------------------------------------------
# helpers on dolfinx meshes
# ---------------------------------------------------------------------------
def _exterior_facets(mesh):
    from dolfinx.mesh import exterior_facet_indices
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, tdim)
    return exterior_facet_indices(mesh.topology)


def _loops(mesh, facets):
    """Number of closed loops formed by 2D facets (None if not all closed)."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    mesh.topology.create_connectivity(1, 0)
    f2v = mesh.topology.connectivity(1, 0)
    edges = np.array([f2v.links(f) for f in facets])
    verts, inv = np.unique(edges, return_inverse=True)
    inv = inv.reshape(-1, 2)
    if not np.all(np.bincount(inv.ravel()) == 2):
        return None
    g = sp.coo_matrix((np.ones(len(inv)), (inv[:, 0], inv[:, 1])), shape=(len(verts),) * 2)
    return connected_components(g, directed=False)[0]


# ---------------------------------------------------------------------------
# 2D
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", CASES_2D)
def test_tag_areas_match_polar_pixels(meshes_2d, name):
    plm, res = meshes_2d[name]
    ref = pixel_areas(plm)
    for tag, a_ref in ref.items():
        if a_ref == 0:
            assert tag not in res["areas"]
            continue
        a = res["areas"][tag]
        assert abs(a / a_ref - 1) < AREA_RTOL, (name, tag, a, a_ref)
        # mesh area equals the polygon area (the mesh resolves the partition)
        assert res["areas"][tag] == pytest.approx(res["geom_areas"][tag], rel=1e-6)
    lumen_ref = (R * DR * 2 * np.pi / N_THETA)[plm.labels == LABEL_LUMEN].sum()
    assert abs(res["lumen_area"] / lumen_ref - 1) < AREA_RTOL


@pytest.mark.parametrize("name", CASES_2D)
def test_dolfinx_load_tags_and_lumen_loop(meshes_2d, name):
    plm, res = meshes_2d[name]
    md = load_mesh(res["msh"], 2)
    mesh = md.mesh
    n_cells = mesh.topology.index_map(2).size_local
    assert mesh.geometry.x.shape[1] == 3 and np.allclose(mesh.geometry.x[:, 2], 0.0)
    # every cell tagged exactly once with a valid tag
    assert md.cell_tags.indices.size == n_cells == res["n_cells"]
    assert np.unique(md.cell_tags.indices).size == n_cells
    expected = {t for t, a in pixel_areas(plm).items() if a > 0}
    assert set(np.unique(md.cell_tags.values)) == expected
    # boundary facets: all exterior facets tagged, lumen and outer each one loop
    ext = _exterior_facets(mesh)
    ft = md.facet_tags
    assert set(np.unique(ft.values)) == {TAG_LUMEN_BND, TAG_OUTER_BND}
    assert np.array_equal(np.sort(ft.indices), np.sort(ext))       # conforming, no cracks
    assert _loops(mesh, ft.find(TAG_LUMEN_BND)) == 1
    assert _loops(mesh, ft.find(TAG_OUTER_BND)) == 1
    assert ft.find(TAG_LUMEN_BND).size == res["n_lumen_facets"]


def test_ring_makes_annular_fibrous_region(meshes_2d):
    plm, res = meshes_2d["ring"]
    part = build_partition(plm)
    ca = part.calcium
    assert len(ca.geoms) == 1 and len(ca.geoms[0].interiors) == 1   # closed ring
    fib = [f for f, c in zip(part.faces, part.face_class) if c == TAG_FIBROUS]
    # one fibrous annulus between lumen and ring, one between ring and IEL
    assert len(fib) == 2 and all(len(f.interiors) == 1 for f in fib)
    assert res["n_cells_per_tag"][TAG_CALCIUM] > 0


def test_fragmented_deposits(meshes_2d):
    _, res = meshes_2d["fragmented"]
    assert res["partition_info"]["n_calcium_components"] == 4
    md = load_mesh(res["msh"], 2)
    from dolfinx.mesh import compute_midpoints
    ca_cells = md.cell_tags.find(TAG_CALCIUM)
    mid = compute_midpoints(md.mesh, 2, ca_cells)
    ang = np.degrees(np.arctan2(mid[:, 1], mid[:, 0]))
    # calcium cells only inside the four 27-degree deposits (centres -67.5 + 45k)
    d = np.min(np.abs(ang[:, None] - (-67.5 + 45.0 * np.arange(4))[None, :]), axis=1)
    assert d.max() < 13.5 + 1.0


def test_size_field_refines_calcium(meshes_2d):
    _, res = meshes_2d["sheet"]
    md = load_mesh(res["msh"], 2)
    from dolfinx.cpp.mesh import h as cell_h
    mesh = md.mesh
    hs = {t: cell_h(mesh._cpp_object, 2, md.cell_tags.find(t)) for t in (TAG_CALCIUM, TAG_WALL)}
    assert np.median(hs[TAG_CALCIUM]) < 1.5 * res["h_ca"]
    assert np.median(hs[TAG_WALL]) > 0.5 * res["h_soft"]


def test_real_segmentation_path(tmp_path):
    plm = make_case("real")
    assert set(np.unique(plm.labels)) == {LABEL_OTHER, LABEL_LUMEN, LABEL_CALCIUM}
    try:
        P.get("geometry.wall_thickness_mm")
        config_pending = False
    except ValueError:
        config_pending = True
    if config_pending:  # default must refuse the PENDING wall thickness
        with pytest.raises(ValueError, match="PENDING"):
            mesh_from_labelmap(plm, tmp_path / "x.msh")
    res = mesh_from_labelmap(plm, tmp_path / "real.msh", allow_pending=True)
    wall_t = P.get("geometry.wall_thickness_mm", allow_pending=True).value
    assert res["partition_info"]["r_iel_source"].startswith("r_out - wall_thickness")
    assert res["partition_info"]["n_calcium_components"] == 1          # speckle dropped
    ref = pixel_areas(plm, r_iel=plm.r_out - wall_t)
    ref[TAG_CALCIUM] -= (_R[150] * DR * 2 * np.pi / N_THETA)           # minus speckle
    for tag in (TAG_CALCIUM, TAG_FIBROUS, TAG_WALL):
        assert abs(res["areas"][tag] / ref[tag] - 1) < AREA_RTOL, tag
    md = load_mesh(res["msh"], 2)
    ft = md.facet_tags
    assert _loops(md.mesh, ft.find(TAG_LUMEN_BND)) == 1               # lumen touches calcium
    assert np.array_equal(np.sort(ft.indices), np.sort(_exterior_facets(md.mesh)))
    # explicit IEL radius instead of the config default
    res2 = mesh_from_labelmap(plm, tmp_path / "real2.msh", r_iel_default=R_IEL)
    assert res2["partition_info"]["r_iel_source"] == "r_iel_default"
    assert abs(res2["areas"][TAG_WALL] / pixel_areas(plm, r_iel=R_IEL)[TAG_WALL] - 1) < AREA_RTOL


def test_gmsh_session_of_caller_preserved(tmp_path):
    import gmsh
    gmsh.initialize()
    try:
        gmsh.model.add("caller")
        opts = ("Mesh.MeshSizeMax", "Mesh.MeshSizeFromPoints", "Mesh.MeshSizeExtendFromBoundary",
                "Mesh.Algorithm", "General.Terminal")
        before = {o: gmsh.option.getNumber(o) for o in opts}
        mesh_from_labelmap(make_case("sheet"), tmp_path / "s.msh", h_ca=0.08, h_soft=0.3)
        load_mesh(tmp_path / "s.msh", 2)
        assert gmsh.isInitialized() and gmsh.model.getCurrent() == "caller"
        assert {o: gmsh.option.getNumber(o) for o in opts} == before   # global options restored
    finally:
        gmsh.finalize()


def _min_quality(msh):
    """Minimum gmsh minSICN over the 2D elements of a .msh file."""
    from dtivl.meshing.mask_to_mesh import gmsh_model
    with gmsh_model("quality") as g:
        g.merge(str(msh))
        _, tags, _ = g.model.mesh.getElements(2)
        return float(g.model.mesh.getElementQualities(tags[0], "minSICN").min())


@pytest.mark.parametrize("label", [LABEL_CALCIUM, LABEL_LIPID])
def test_deposit_touching_lumen_has_no_slivers(tmp_path, label):
    """A deposit sharing its boundary with the lumen must not leave sub-pixel
    tissue slivers between the two (shared boundaries simplified once)."""
    lab = _sector(_base(), R_LUMEN, R_LUMEN + 0.3, 90.0, 120.0, label)
    plm = PolarLabelMap(lab, DR, np.full(N_THETA, R_OUT), np.full(N_THETA, R_IEL), {})
    part = build_partition(plm)
    assert part.info["n_small_faces"] == 0
    tag = TAG_CALCIUM if label == LABEL_CALCIUM else TAG_LIPID
    # lumen, deposit, one fibrous region, one wall annulus
    assert sorted(part.face_class) == sorted([0, tag, TAG_FIBROUS, TAG_WALL])
    # the deposit and the lumen share their boundary (no tissue in between)
    dep = [f for f, c in zip(part.faces, part.face_class) if c == tag][0]
    shared = dep.boundary.intersection(part.lumen.boundary).length
    assert shared > 0.9 * np.radians(120.0) * R_LUMEN
    res = mesh_from_labelmap(plm, tmp_path / "touch.msh")
    assert _min_quality(res["msh"]) > 0.05   # was ~1e-4 with per-region simplification
    assert abs(res["areas"][tag] / pixel_areas(plm)[tag] - 1) < AREA_RTOL


def test_real_segmentation_mesh_quality(tmp_path):
    res = mesh_from_labelmap(make_case("real"), tmp_path / "q.msh", r_iel_default=R_IEL)
    assert res["partition_info"]["n_small_faces"] == 0
    assert _min_quality(res["msh"]) > 0.05   # was ~1e-4 with per-region simplification


# ---------------------------------------------------------------------------
# 3D
# ---------------------------------------------------------------------------
def test_extrusion_3d(tmp_path):
    plm = make_case("lipid")
    L, m = 1.0, 0.5
    res = mesh3d_from_labelmap(plm, L, tmp_path / "s3.msh", margin=m, h_axial=0.5,
                               h_ca=0.08, h_soft=0.3)
    Z = 2 * m + L
    assert res["z_blocks"] == pytest.approx((0.0, m, m + L, Z))
    A = res["areas_2d"]
    vol = res["volumes"]
    assert vol[TAG_CALCIUM] == pytest.approx(A[TAG_CALCIUM] * L, rel=1e-6)
    assert vol[TAG_FIBROUS] == pytest.approx(A[TAG_FIBROUS] * Z + A[TAG_CALCIUM] * 2 * m, rel=1e-6)
    assert vol[TAG_WALL] == pytest.approx(A[TAG_WALL] * Z, rel=1e-6)
    assert vol[TAG_LIPID] == pytest.approx(A[TAG_LIPID] * Z, rel=1e-6)
    ref = pixel_areas(plm)
    assert abs(vol[TAG_CALCIUM] / (ref[TAG_CALCIUM] * L) - 1) < AREA_RTOL

    md = load_mesh(res["msh"], 3)
    mesh = md.mesh
    n_cells = mesh.topology.index_map(3).size_local
    assert md.cell_tags.indices.size == n_cells == res["n_cells"]
    assert set(np.unique(md.cell_tags.values)) == {TAG_WALL, TAG_FIBROUS, TAG_LIPID, TAG_CALCIUM}
    from dolfinx.mesh import compute_midpoints
    zc = compute_midpoints(mesh, 3, md.cell_tags.find(TAG_CALCIUM))[:, 2]
    assert zc.min() > m and zc.max() < m + L                           # calcium only in middle block
    ft = md.facet_tags
    assert set(np.unique(ft.values)) == {TAG_LUMEN_BND, TAG_OUTER_BND, TAG_END_BND}
    assert np.array_equal(np.sort(ft.indices), np.sort(_exterior_facets(mesh)))  # conforming
    fz = compute_midpoints(mesh, 2, ft.find(TAG_END_BND))[:, 2]
    assert np.all(np.isclose(fz, 0.0) | np.isclose(fz, Z))
    xo = compute_midpoints(mesh, 2, ft.find(TAG_OUTER_BND))
    xl = compute_midpoints(mesh, 2, ft.find(TAG_LUMEN_BND))
    assert np.hypot(xl[:, 0], xl[:, 1]).max() < np.hypot(xo[:, 0], xo[:, 1]).min()


def test_extrusion_3d_ring_defaults(tmp_path):
    """Ring case with the configured margin and axial size (coarse cross-section)."""
    res = mesh3d_from_labelmap(make_case("ring"), 1.0, tmp_path / "r3.msh", h_ca=0.1, h_soft=0.3)
    assert res["margin"] == P.get("geometry.margin_3d_mm").value
    assert res["volumes"][TAG_CALCIUM] == pytest.approx(res["areas_2d"][TAG_CALCIUM] * 1.0, rel=1e-6)


def test_extrusion_3d_calcium_crossing_iel(tmp_path):
    """End-block parts of a deposit crossing the IEL: fibrous inside, wall outside."""
    lab = _sector(_base(), 1.5, 1.75, 0.0, 60.0)          # crosses the IEL at 1.6
    plm = PolarLabelMap(lab, DR, np.full(N_THETA, R_OUT), np.full(N_THETA, R_IEL), {})
    m = 0.5
    res = mesh3d_from_labelmap(plm, 1.0, tmp_path / "x3.msh", margin=m, h_axial=0.5, h_ca=0.08, h_soft=0.3)
    md = load_mesh(res["msh"], 3)
    from dolfinx.mesh import compute_midpoints
    for tag in (TAG_FIBROUS, TAG_WALL):
        c = compute_midpoints(md.mesh, 3, md.cell_tags.find(tag))
        rr = np.hypot(c[:, 0], c[:, 1])
        ang = np.abs(np.degrees(np.arctan2(c[:, 1], c[:, 0])))
        out_iel = (c[:, 2] < m) & (rr > 1.63) & (rr < 1.72) & (ang < 25.0)
        assert (out_iel.sum() == 0) if tag == TAG_FIBROUS else (out_iel.sum() > 0)
    # volume bookkeeping unchanged: calcium only in the middle block
    assert res["volumes"][TAG_CALCIUM] == pytest.approx(res["areas_2d"][TAG_CALCIUM] * 1.0, rel=1e-6)


def test_inconsistent_sizes_rejected(tmp_path):
    """h_ca > h_soft would be silently capped at h_soft by Mesh.MeshSizeMax."""
    with pytest.raises(ValueError, match="h_ca"):
        mesh_from_labelmap(make_case("sheet"), tmp_path / "bad.msh", h_ca=0.2, h_soft=0.15)
    with pytest.raises(ValueError, match="h_ca"):
        mesh3d_from_labelmap(make_case("sheet"), 1.0, tmp_path / "bad3.msh", h_ca=0.2, h_soft=0.15)

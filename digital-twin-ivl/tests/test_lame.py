"""Verification of the linear solver against the Lame thick-cylinder solution."""
import json

import numpy as np
import pytest

from dtivl.geometry.labels import (TAG_CALCIUM, TAG_END_BND, TAG_FIBROUS, TAG_LUMEN_BND,
                                   TAG_OUTER_BND)
from dtivl.materials.constitutive import MaterialSet
from dtivl.solvers import linear as L


def test_lame_exact_boundary_conditions():
    lam, mu = MaterialSet.baseline().lame(TAG_CALCIUM)
    _, srr, _ = L.lame_exact([L.LAME_A, L.LAME_B], L.LAME_A, L.LAME_B, L.LAME_P, lam, mu)
    assert srr[0] == pytest.approx(-L.LAME_P)
    assert srr[1] == pytest.approx(0.0, abs=1e-14)


def test_verification_lame_writes_json(tmp_path):
    out = tmp_path / "verification_lame.json"
    rep = L.verification_lame(out)
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["passed"] and rep["passed"]
    mats = {c["material"] for c in data["cases"]}
    assert mats == {"calcium", "fibrous"}
    for c in data["cases"]:
        assert c["max_rel_err_ur"] < 0.01
        assert c["max_rel_err_hoop"] < 0.01
        assert c["max_rel_err_hoop_cellavg"] < 0.01
        # weak foundation: total reaction is negligible compared with the load
        assert c["foundation_ratio"] < 1e-3


def test_plane_strain_sigma33_and_principal():
    """sigma_33 = lambda tr(eps) and sigma1 is the 3D max principal stress."""
    ms = MaterialSet.baseline()
    md = L.annulus_meshdata(1.0, 2.0, 0.2, TAG_CALCIUM)
    res = L.solve_linear(md, ms, 0.1)
    ca = L.cell_averages(res.u, res.lam, res.mu)
    lam, _ = ms.lame(TAG_CALCIUM)
    tr = ca["eps"][:, 0, 0] + ca["eps"][:, 1, 1]
    np.testing.assert_allclose(ca["sigma"][:, 2, 2], lam * tr, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(ca["eps"][:, 2, 2], 0.0, atol=1e-15)
    eig2 = np.linalg.eigvalsh(ca["sigma"][:, :2, :2])[:, -1]
    np.testing.assert_allclose(ca["sigma1"], np.maximum(eig2, ca["sigma"][:, 2, 2]), rtol=1e-12)
    # all cells are calcium here: the result arrays cover the whole mesh
    assert res.sigma1.size == md.mesh.topology.index_map(2).size_local
    assert res.measure.sum() == pytest.approx(np.pi * (4.0 - 1.0), rel=0.02)
    np.testing.assert_allclose(res.sed, 0.5 * np.einsum("nij,nij->n", res.sigma, res.eps), rtol=0.05)


def test_untagged_material_rejected():
    md = L.annulus_meshdata(1.0, 2.0, 0.3, 99)
    with pytest.raises(ValueError):
        L.solve_linear(md, MaterialSet.baseline(), 0.1)


def _annulus3d(a, b, length, h, tag):
    from mpi4py import MPI
    from dolfinx.io.gmsh import model_to_mesh

    with L._gmsh_model("annulus3d") as gmsh:
        occ = gmsh.model.occ
        occ.cut([(3, occ.addCylinder(0, 0, 0, 0, 0, length, b))],
                [(3, occ.addCylinder(0, 0, 0, 0, 0, length, a))])
        occ.synchronize()
        vols = [t for _, t in gmsh.model.getEntities(3)]
        gmsh.model.addPhysicalGroup(3, vols, tag)
        groups = {TAG_LUMEN_BND: [], TAG_OUTER_BND: [], TAG_END_BND: []}
        for _, s in gmsh.model.getBoundary([(3, v) for v in vols], oriented=False):
            x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(2, s)
            if abs(z1 - z0) < 1e-6:
                groups[TAG_END_BND].append(s)
            elif max(x1 - x0, y1 - y0) > a + b:
                groups[TAG_OUTER_BND].append(s)
            else:
                groups[TAG_LUMEN_BND].append(s)
        for k, v in groups.items():
            gmsh.model.addPhysicalGroup(2, v, k)
        gmsh.option.setNumber("Mesh.MeshSizeMin", h)
        gmsh.option.setNumber("Mesh.MeshSizeMax", h)
        gmsh.model.mesh.generate(3)
        return model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=3)


def test_3d_tube_with_end_constraints_matches_plane_strain():
    """With u_z = 0 on both ends a pressurised tube is in plane strain."""
    ms = MaterialSet.baseline()
    a, b, p = 1.0, 2.0, 0.1
    md = _annulus3d(a, b, 0.4, 0.2, TAG_CALCIUM)
    res = L.solve_linear(md, ms, p)
    lam, mu = ms.lame(TAG_CALCIUM)
    th = np.linspace(0, 2 * np.pi, 8, endpoint=False) + 0.1
    r = 1.5
    pts = np.column_stack([r * np.cos(th), r * np.sin(th), np.full_like(th, 0.2)])
    u = L.displacement_at_points(res, pts)
    ur = u[:, 0] * np.cos(th) + u[:, 1] * np.sin(th)
    ur_ex = L.lame_exact(r, a, b, p, lam, mu)[0]
    assert np.max(np.abs(ur - ur_ex)) / ur_ex < 0.02
    assert np.max(np.abs(u[:, 2])) < 1e-3 * ur_ex
    assert res.centroid.shape[1] == 3
    assert res.foundation_ratio < 1e-3


def test_fibrous_single_material_runs():
    rep = L.lame_check(tag=TAG_FIBROUS, h=0.1)
    assert rep["max_rel_err_ur"] < 0.01


def _annulus_with_island():
    """Fibrous annulus plus a calcium disk meshed WITHOUT fragmentation (floating)."""
    from mpi4py import MPI
    from dolfinx.io.gmsh import model_to_mesh

    with L._gmsh_model("island") as gmsh:
        occ = gmsh.model.occ
        ann = occ.cut([(2, occ.addDisk(0, 0, 0, 2, 2))], [(2, occ.addDisk(0, 0, 0, 1, 1))])[0]
        isl = occ.addDisk(1.5, 0, 0, 0.1, 0.1)
        occ.synchronize()
        gmsh.model.addPhysicalGroup(2, [ann[0][1]], TAG_FIBROUS)
        gmsh.model.addPhysicalGroup(2, [isl], TAG_CALCIUM)
        inner, outer = [], []
        for _, c in gmsh.model.getBoundary(ann, oriented=False):
            x0, y0, _, x1, y1, _ = gmsh.model.getBoundingBox(1, c)
            (outer if max(x1 - x0, y1 - y0) > 3 else inner).append(c)
        gmsh.model.addPhysicalGroup(1, inner, TAG_LUMEN_BND)
        gmsh.model.addPhysicalGroup(1, outer, TAG_OUTER_BND)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 0.15)
        gmsh.model.mesh.generate(2)
        return model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)


def test_singular_system_is_not_reported_as_success():
    """A floating (non-conforming) calcium body must raise, not return sigma1 = 0."""
    with pytest.raises(RuntimeError, match="null pivot"):
        L.solve_linear(_annulus_with_island(), MaterialSet.baseline(), 0.4)


def test_regular_solve_has_no_null_pivots():
    res = L.solve_linear(L.annulus_meshdata(1.0, 2.0, 0.3, TAG_CALCIUM), MaterialSet.baseline(), 0.1)
    assert res.info["null_pivots"] == 0


def test_3d_without_end_tags_rejected():
    """design.md section 4: 3D ends carry u_z = 0, so TAG_END_BND is required."""
    from dolfinx.io.gmsh import MeshData
    from dolfinx import mesh as dmesh

    md = _annulus3d(1.0, 2.0, 0.4, 0.4, TAG_CALCIUM)
    ft = md.facet_tags
    keep = ft.values != TAG_END_BND
    ft2 = dmesh.meshtags(md.mesh, 2, ft.indices[keep], ft.values[keep])
    md2 = MeshData(md.mesh, md.cell_tags, ft2, *([None] * (len(MeshData._fields) - 3)))
    with pytest.raises(ValueError, match=str(TAG_END_BND)):
        L.solve_linear(md2, MaterialSet.baseline(), 0.1)


_MPI_SCRIPT = r'''
import numpy as np
from mpi4py import MPI
from dolfinx import mesh as dm
from dolfinx.io.gmsh import MeshData
from dtivl.geometry.labels import TAG_CALCIUM, TAG_LUMEN_BND, TAG_OUTER_BND
from dtivl.materials.constitutive import MaterialSet
from dtivl.solvers import linear as L
comm = MPI.COMM_WORLD
msh = dm.create_rectangle(comm, [[0, 0], [4, 1]], [40, 10])
nc = msh.topology.index_map(2).size_local + msh.topology.index_map(2).num_ghosts
ct = dm.meshtags(msh, 2, np.arange(nc, dtype=np.int32), np.full(nc, TAG_CALCIUM, np.int32))
msh.topology.create_connectivity(1, 2)
lum = dm.locate_entities_boundary(msh, 1, lambda x: np.isclose(x[0], 0.0))
out = dm.locate_entities_boundary(msh, 1, lambda x: np.isclose(x[1], 1.0))
f = np.concatenate([lum, out]).astype(np.int32)
v = np.concatenate([np.full(len(lum), TAG_LUMEN_BND), np.full(len(out), TAG_OUTER_BND)]).astype(np.int32)
o = np.argsort(f)
ft = dm.meshtags(msh, 1, f[o], v[o])
md = MeshData(msh, ct, ft, *([None] * (len(MeshData._fields) - 3)))
counts = comm.allgather(len(lum))
L.solve_linear(md, MaterialSet.baseline(), 0.1)
if comm.rank == 0:
    print("DONE", counts)
'''


def test_facet_tag_check_is_collective(tmp_path):
    """Lumen facets on one rank only: solve_linear must not deadlock (2 ranks)."""
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    mpiexec = Path(sys.executable).parent / "mpiexec"
    if not mpiexec.exists():
        mpiexec = shutil.which("mpiexec")
    if mpiexec is None:
        pytest.skip("mpiexec not available")
    script = tmp_path / "mpi_check.py"
    script.write_text(_MPI_SCRIPT)
    proc = subprocess.run([str(mpiexec), "-n", "2", sys.executable, str(script)], capture_output=True,
                          text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "DONE" in proc.stdout

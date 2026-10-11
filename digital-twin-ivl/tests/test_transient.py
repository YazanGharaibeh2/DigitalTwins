"""Tests for the transient IVL pulse solver (design.md section 6).

Geometries are built here directly (dolfinx box/rectangle meshes and small
gmsh annuli); the vessel radii come from the study config so no literature
number is typed. Strip/column dimensions, mesh sizes and time steps are
numerical test settings.
"""
from __future__ import annotations

import math
import types

import numpy as np
import pytest

from dtivl import params as P
from dtivl.geometry.labels import (TAG_CALCIUM, TAG_END_BND, TAG_FIBROUS, TAG_LIPID, TAG_LUMEN_BND,
                                   TAG_OUTER_BND, TAG_WALL)
from dtivl.materials.constitutive import MaterialSet
from dtivl.solvers import transient as TR

# --- numerical test settings ------------------------------------------------
STRIP_L = 12.0          #: strip / column length (mm)
STRIP_W = 0.2           #: strip width (mm)
STRIP_H = 0.05          #: strip mesh size (mm)
COLUMN_H = 0.1          #: 3D column mesh size (mm)
COLUMN_H_MIXED = 0.05   #: 3D column mesh size for the Taylor-Hood check (mm)
COLUMN_W_MIXED = 0.1    #: 3D column width for the Taylor-Hood check (mm)
PROBES = (3.0, 7.0)     #: probe positions along the strip (mm)
PULSE_W = 1.0           #: test pulse width (us)
DT_STRIP = 0.01         #: strip time step (us)
DT_COLUMN = 0.02        #: 3D column time step (us)
SPEED_TOL = 0.02        #: design.md: wave speed within 2 %
ENERGY_TOL = 0.01       #: energy conservation tolerance
TAG_SIDE_Y = 40         #: extra facet tag for the y-faces of the 3D column
H_CA, H_SOFT = 0.05, 0.2  #: sector annulus mesh sizes (mm)
SECTOR_ARC = 90.0       #: calcium sector arc (deg)
T_BOUND = 1.0           #: sensible upper bound: a ~2 us pulse is far shorter than the hoop response time
LAME_3D_LENGTH = 0.3    #: 3D tube length for the static Lame check (mm)
H_CA_3D, H_SOFT_3D = 0.08, 0.3  #: coarse 3D sector mesh sizes (mm)
LENGTH_3D = 0.3         #: 3D sector extrusion length (mm)
T3D_WINDOW_PULSES = 5.0 #: 3D consistency window (pulse widths)


# ---------------------------------------------------------------------------
# Fixtures / mesh builders
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ms():
    return MaterialSet.baseline()


@pytest.fixture(scope="module")
def acoustic():
    return TR.default_densities(allow_pending=True), TR.default_sound_speeds(allow_pending=True)


def _tag_facets(msh, pieces):
    from dolfinx import mesh as dmesh

    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim - 1, tdim)
    idx, val = [], []
    for marker, tag in pieces:
        f = dmesh.locate_entities_boundary(msh, tdim - 1, marker)
        idx.append(f)
        val.append(np.full(f.size, tag, dtype=np.int32))
    idx, val = np.concatenate(idx), np.concatenate(val)
    idx, first = np.unique(idx, return_index=True)
    return dmesh.meshtags(msh, tdim - 1, idx.astype(np.int32), val[first])


def strip_meshdata(gdim: int, cell_tag: int = TAG_CALCIUM, h3d: float = COLUMN_H, width: float = STRIP_W):
    """Homogeneous strip (2D) / column (3D) along x; load face x=0, far face x=L.

    Facet tags: x=0 lumen, x=L outer, lateral faces TAG_END_BND (y faces in
    2D, z faces in 3D) and TAG_SIDE_Y (y faces in 3D).
    """
    from mpi4py import MPI
    from dolfinx import mesh as dmesh

    if gdim == 2:
        msh = dmesh.create_rectangle(MPI.COMM_WORLD, [[0, 0], [STRIP_L, width]],
                                     [round(STRIP_L / STRIP_H), round(width / STRIP_H)],
                                     dmesh.CellType.triangle)
    else:
        n = max(1, round(width / h3d))
        msh = dmesh.create_box(MPI.COMM_WORLD, [[0, 0, 0], [STRIP_L, width, width]],
                               [round(STRIP_L / h3d), n, n], dmesh.CellType.tetrahedron)
    nc = msh.topology.index_map(gdim).size_local
    ct = dmesh.meshtags(msh, gdim, np.arange(nc, dtype=np.int32), np.full(nc, cell_tag, dtype=np.int32))
    pieces = [(lambda x: np.isclose(x[0], 0.0), TAG_LUMEN_BND),
              (lambda x: np.isclose(x[0], STRIP_L), TAG_OUTER_BND)]
    if gdim == 2:
        pieces.append((lambda x: np.isclose(x[1], 0.0) | np.isclose(x[1], width), TAG_END_BND))
    else:
        pieces.append((lambda x: np.isclose(x[2], 0.0) | np.isclose(x[2], width), TAG_END_BND))
        pieces.append((lambda x: np.isclose(x[1], 0.0) | np.isclose(x[1], width), TAG_SIDE_Y))
    return types.SimpleNamespace(mesh=msh, cell_tags=ct, facet_tags=_tag_facets(msh, pieces))


def _radii():
    r_l = math.sqrt(P.get("geometry.lumen_area_mm2").value / math.pi)
    r_iel = r_l + P.get("geometry.plaque_thickness_mm").value
    r_o = r_iel + P.get("geometry.wall_thickness_mm").value
    return r_l, r_iel, r_o


def sector_annulus_meshdata(arc_deg: float | None, h_ca: float = H_CA, h_soft: float = H_SOFT,
                            length: float | None = None):
    """Vessel annulus (config radii) with fibrous intima, wall and a calcium sector.

    ``arc_deg=None`` gives a homogeneous single-tag (TAG_WALL) annulus.
    Calcium: superficial depth and the first D2 thickness level, centred on
    +x. ``length`` extrudes to 3D (ends tagged TAG_END_BND).
    """
    import gmsh
    from mpi4py import MPI
    from dolfinx.io.gmsh import model_to_mesh

    r_l, r_iel, r_o = _radii()
    owns = not gmsh.isInitialized()
    if owns:
        gmsh.initialize(interruptible=False)
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(f"sector_{np.random.default_rng().integers(1 << 30)}")
    try:
        occ = gmsh.model.occ
        tissue, _ = occ.cut([(2, occ.addDisk(0, 0, 0, r_o, r_o))], [(2, occ.addDisk(0, 0, 0, r_l, r_l))])
        tools = []
        if arc_deg is not None:
            d = P.get("geometry.depth_superficial_mm").value
            t_ca = float(P.raw("design.core_factorial.thickness_mm")[0])
            r1, r2, a = r_l + d, r_l + d + t_ca, math.radians(arc_deg)
            c = occ.addPoint(0, 0, 0)
            p = [occ.addPoint(r * math.cos(s * a / 2), r * math.sin(s * a / 2), 0)
                 for r, s in ((r1, -1), (r2, -1), (r2, 1), (r1, 1))]
            loop = occ.addCurveLoop([occ.addLine(p[0], p[1]), occ.addCircleArc(p[1], c, p[2]),
                                     occ.addLine(p[2], p[3]), occ.addCircleArc(p[3], c, p[0])])
            sec = occ.addPlaneSurface([loop])
            occ.remove([(0, c)])
            tools = [(2, occ.addDisk(0, 0, 0, r_iel, r_iel)), (2, sec)]
            _, cmap = occ.fragment(tissue, tools)
            occ.synchronize()
            kids = [set(t for _, t in m) for m in cmap]
            surf = {TAG_WALL: [], TAG_FIBROUS: [], TAG_CALCIUM: []}
            for _, s in gmsh.model.getEntities(2):
                if s in kids[2]:
                    surf[TAG_CALCIUM].append(s)
                elif s in kids[0] and s in kids[1]:
                    surf[TAG_FIBROUS].append(s)
                elif s in kids[0]:
                    surf[TAG_WALL].append(s)
                else:
                    occ.remove([(2, s)], recursive=True)
            occ.synchronize()
        else:
            occ.synchronize()
            surf = {TAG_WALL: [s for _, s in tissue]}
        dim = 2
        if length is not None:
            all_s = [(2, s) for ss in surf.values() for s in ss]
            src_tag = [tag for tag, ss in surf.items() for _ in ss]
            ext = occ.extrude(all_s, 0, 0, length)
            vols = [(3, t) for d_, t in ext if d_ == 3]          # one volume per input surface, in order
            if len(vols) > 1:
                _, vmap = occ.fragment(vols, [])                # conforming shared faces
            else:
                vmap = [[v] for v in vols]
            occ.synchronize()
            surf = {tag: [] for tag in surf}
            for tag, children in zip(src_tag, vmap):
                surf[tag].extend(t for _, t in children)
            dim = 3
        for tag, ss in surf.items():
            gmsh.model.addPhysicalGroup(dim, ss, tag)
        ents = [(dim, s) for ss in surf.values() for s in ss]
        bnd = gmsh.model.getBoundary(ents, combined=True, oriented=False)
        groups = {TAG_LUMEN_BND: [], TAG_OUTER_BND: [], TAG_END_BND: []}
        for _, e in bnd:
            e = abs(e)
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim - 1, e)
            if dim == 3 and zmax - zmin < 0.5 * length:      # planar end face (OCC boxes are padded)
                groups[TAG_END_BND].append(e)
                continue
            r = max(abs(xmin), abs(xmax), abs(ymin), abs(ymax))
            groups[TAG_LUMEN_BND if r < 0.5 * (r_l + r_o) else TAG_OUTER_BND].append(e)
        for tag, ee in groups.items():
            if ee:
                gmsh.model.addPhysicalGroup(dim - 1, ee, tag)
        f = gmsh.model.mesh.field
        if TAG_CALCIUM in surf and surf[TAG_CALCIUM]:
            dist = f.add("Distance")
            if dim == 2:
                f.setNumbers(dist, "SurfacesList", surf[TAG_CALCIUM])
            else:
                faces = gmsh.model.getBoundary([(3, v) for v in surf[TAG_CALCIUM]], oriented=False)
                f.setNumbers(dist, "SurfacesList", [abs(t) for _, t in faces])
            th = f.add("Threshold")
            f.setNumber(th, "InField", dist)
            f.setNumber(th, "SizeMin", h_ca)
            f.setNumber(th, "SizeMax", h_soft)
            f.setNumber(th, "DistMin", 2 * h_ca)
            f.setNumber(th, "DistMax", 2 * h_ca + 2 * h_soft)
            f.setAsBackgroundMesh(th)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        else:
            gmsh.option.setNumber("Mesh.MeshSizeMax", h_soft)
        gmsh.model.mesh.generate(dim)
        md = model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=dim)
        for tag in ((TAG_LUMEN_BND, TAG_OUTER_BND) + ((TAG_END_BND,) if dim == 3 else ())):
            assert md.facet_tags.find(tag).size > 0, f"facet tag {tag} missing"
        return md
    finally:
        gmsh.model.remove()
        if owns:
            gmsh.finalize()


# ---------------------------------------------------------------------------
# Inputs and pulse
# ---------------------------------------------------------------------------
def test_inputs_and_units(ms, acoustic):
    rho, c = acoustic
    assert set(rho) == set(c) == {TAG_WALL, TAG_FIBROUS, TAG_LIPID, TAG_CALCIUM}
    for tag in rho:
        # m/s -> mm/us: soft tissue and calcium wave speeds are O(1) mm/us
        assert 1.0 < c[tag] < 3.0
        assert rho[tag] > 100.0
    mats = TR.acoustic_materials(ms, rho, c, [TAG_WALL, TAG_CALCIUM])
    m = mats[TAG_CALCIUM]
    assert m.M == pytest.approx(rho[TAG_CALCIUM] * c[TAG_CALCIUM] ** 2, rel=1e-12)
    assert m.mu == pytest.approx(ms.lame(TAG_CALCIUM)[1])
    assert m.lam + 2.0 * m.mu == pytest.approx(m.M, rel=1e-12)
    assert 0.49 < m.nu < 0.5
    # design.md example: M = 2200 MPa, rho = 1000 -> 1.483 mm/us
    ex = TR.AcousticMaterial(rho=1000.0, c_p=math.sqrt(2.2), mu=0.0, K=2200.0)
    assert math.sqrt(ex.M / ex.rho) == pytest.approx(1.483, abs=1e-3)
    with pytest.raises(ValueError):
        TR.acoustic_materials(ms, {TAG_WALL: rho[TAG_WALL]}, c, [TAG_CALCIUM])
    w = TR.default_pulse_width_us()
    assert 0.0 < w < 10.0
    assert TR.default_peak_pressure() > 0.0


def test_half_sine():
    p = TR.half_sine(2.0, 4.0)
    assert p(-1.0) == 0.0 and p(5.0) == 0.0
    assert p(2.0) == pytest.approx(2.0)
    assert p(1.0) == pytest.approx(2.0 * math.sin(math.pi / 4))
    with pytest.raises(ValueError):
        TR.half_sine(1.0, 0.0)


# ---------------------------------------------------------------------------
# Verification: wave speed
# ---------------------------------------------------------------------------
def _arrival_speed(md, ms, acoustic, rollers, outer_bc="absorbing", t_end=None, dt=DT_STRIP,
                   formulation="mixed"):
    rho, c = acoustic
    model = TR.build_model(md, ms, rho, c, roller_tags=rollers, outer_bc=outer_bc, formulation=formulation)
    mat = model.materials[TAG_CALCIUM]
    c_exact = math.sqrt(mat.M / mat.rho)
    X = model.dof_coordinates()
    gdim = model.gdim
    on_axis = np.all(np.abs(X[:, 1:]) < 1e-9, axis=1)
    ids = []
    for xp in PROBES:
        cand = np.where(on_axis & (np.abs(X[:, 0] - xp) < 1e-9))[0]
        ids.append(int(cand[cand % gdim == 0][0]))   # x-component DOF
    rec = []
    if t_end is None:
        t_end = PROBES[1] / c_exact + 2.0 * PULSE_W
    hist = model.run(TR.half_sine(1.0, PULSE_W), dt, t_end, record_energy=True,
                     callback=lambda k, t, u, v, a: rec.append(np.r_[u[ids], v[ids]]))
    R = np.asarray(rec)
    U, V = R[:, :len(ids)], R[:, len(ids):]
    t = hist.t
    # arrival time = time at which the displacement reaches half of its plateau, i.e. the
    # median of the transmitted impulse (robust to dispersion and to zero-mean noise)
    t_half = [float(np.interp(0.5 * U[-1, j], np.maximum.accumulate(U[:, j]), t)) for j in range(len(ids))]
    c_meas = (PROBES[1] - PROBES[0]) / (t_half[1] - t_half[0])
    noise = float(np.max(-V.min(axis=0) / V.max(axis=0)))   # backward particle velocity / peak
    return c_meas, c_exact, hist, model, noise


@pytest.mark.parametrize("formulation", ["mixed", "displacement"])
def test_wave_speed_strip_2d(ms, acoustic, formulation):
    md = strip_meshdata(2)
    c_meas, c_exact, _, _, noise = _arrival_speed(md, ms, acoustic, {TAG_END_BND: 1}, formulation=formulation)
    print(f"2D strip ({formulation}): c_meas = {c_meas:.5f} mm/us, sqrt(M/rho) = {c_exact:.5f} mm/us, "
          f"rel err = {abs(c_meas / c_exact - 1):.2e}, spurious backward velocity {noise:.3f} of peak")
    assert abs(c_meas / c_exact - 1.0) < SPEED_TOL


def test_wave_speed_column_3d(ms, acoustic):
    md = strip_meshdata(3)
    c_meas, c_exact, _, _, noise = _arrival_speed(md, ms, acoustic, {TAG_END_BND: 2, TAG_SIDE_Y: 1},
                                                  dt=DT_COLUMN, formulation="displacement")
    print(f"3D column (displacement): c_meas = {c_meas:.5f} mm/us, sqrt(M/rho) = {c_exact:.5f} mm/us, "
          f"rel err = {abs(c_meas / c_exact - 1):.2e}, spurious backward velocity {noise:.3f} of peak")
    assert abs(c_meas / c_exact - 1.0) < SPEED_TOL


def test_wave_speed_column_3d_mixed(ms, acoustic):
    """Taylor-Hood tets need a finer mesh than P2 displacement (spurious incompressible modes)."""
    md = strip_meshdata(3, h3d=COLUMN_H_MIXED, width=COLUMN_W_MIXED)
    c_meas, c_exact, _, _, noise = _arrival_speed(md, ms, acoustic, {TAG_END_BND: 2, TAG_SIDE_Y: 1},
                                                  dt=DT_COLUMN, formulation="mixed")
    print(f"3D column (mixed, h = {COLUMN_H_MIXED}): c_meas = {c_meas:.5f} mm/us, sqrt(M/rho) = {c_exact:.5f} "
          f"mm/us, rel err = {abs(c_meas / c_exact - 1):.2e}, spurious backward velocity {noise:.3f} of peak")
    assert abs(c_meas / c_exact - 1.0) < SPEED_TOL


def test_absorbing_boundary_strip(ms, acoustic):
    """Normal-incidence plane wave leaves through the dashpot end almost entirely."""
    md = strip_meshdata(2)
    rho, c = acoustic
    t_exit = STRIP_L / c[TAG_CALCIUM] + 2.0 * PULSE_W
    _, _, hist, _, _ = _arrival_speed(md, ms, acoustic, {TAG_END_BND: 1}, t_end=t_exit)
    E = hist.energy.sum(axis=1)
    resid = E[-1] / E.max()
    print(f"absorbing end: residual energy fraction = {resid:.2e}")
    assert resid < 0.02
    # closed (fixed) far end reflects: energy stays
    _, _, hist_f, _, _ = _arrival_speed(md, ms, acoustic, {TAG_END_BND: 1}, outer_bc="fixed", t_end=t_exit)
    Ef = hist_f.energy.sum(axis=1)
    assert Ef[-1] / Ef.max() > 0.99


# ---------------------------------------------------------------------------
# Verification: energy conservation (closed boundaries, no damping)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("outer_bc,formulation", [("free", "mixed"), ("fixed", "mixed"),
                                                  ("free", "displacement")])
def test_energy_conservation_annulus(ms, acoustic, outer_bc, formulation):
    md = sector_annulus_meshdata(SECTOR_ARC)
    rho, c = acoustic
    model = TR.build_model(md, ms, rho, c, outer_bc=outer_bc, formulation=formulation)
    width = TR.default_pulse_width_us()
    dt = model.default_dt(width)
    hist = model.run(TR.half_sine(1.0, width), dt, 3.0 * width, record_energy=True)
    E = hist.energy.sum(axis=1)
    after = hist.t >= width
    E0 = E[after][0]
    drift = np.max(np.abs(E[after] - E0)) / E0
    print(f"annulus ({outer_bc}, {formulation}): energy after pulse {E0:.4e}, max relative drift {drift:.2e}")
    assert E0 > 0.0
    assert drift < ENERGY_TOL
    # generalized-alpha with rho_inf < 1 dissipates (monotone non-increasing after the pulse)
    hist_d = model.run(TR.half_sine(1.0, width), dt, 3.0 * width, record_energy=True, rho_inf=0.5)
    Ed = hist_d.energy.sum(axis=1)[after]
    assert Ed[-1] < Ed[0]


# ---------------------------------------------------------------------------
# Transfer factor
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def sector_md():
    return sector_annulus_meshdata(SECTOR_ARC)


def test_transfer_factor_sector(sector_md, ms, acoustic):
    rho, c = acoustic
    res = TR.transfer_factor(sector_md, TAG_CALCIUM, ms, rho, c)
    print(f"sector annulus: T = {res['T']:.4f}, sigma1_peak_transient = {res['sigma1_peak_transient']:.4f} MPa, "
          f"sigma1_static = {res['sigma1_static']:.4f} MPa, t_peak = {res['t_peak']:.3f} us, "
          f"T_equivalent = {res['T_equivalent']:.4f}, min history = {np.nanmin(res['sigma1_max_history']):.3f} MPa")
    assert np.isfinite(res["T"])
    assert 0.0 < res["T"] < T_BOUND
    assert res["sigma1_static"] > 0.0
    assert 0.0 < res["T_equivalent"] < T_BOUND
    assert isinstance(res["peak_at_window_end"], bool)
    hist = res["sigma1_max_history"]
    assert hist.shape == res["t"].shape == res["pressure_history"].shape
    assert np.isfinite(hist).all()
    assert res["sigma1_peak_transient"] == pytest.approx(np.max(hist))
    assert res["sigma1_cell_peak"].max() == pytest.approx(res["sigma1_peak_transient"])
    assert res["ca_cells"].size == res["sigma1_static_cells"].size == res["ca_measure"].size
    # calcium is loaded in compression while the pulse passes (peak pressure dominates)
    during = res["t"] <= res["width_us"]
    assert np.min(hist[during]) < -0.1 * res["p_peak"]
    # linearity: T is independent of the pulse amplitude
    res2 = TR.transfer_factor(sector_md, TAG_CALCIUM, ms, rho, c, p_peak=0.5 * res["p_peak"],
                              equivalent_static=False)
    assert res2["T"] == pytest.approx(res["T"], rel=1e-8)
    assert res2["sigma1_peak_transient"] == pytest.approx(0.5 * res["sigma1_peak_transient"], rel=1e-8)


def test_transfer_factor_defaults_and_errors(sector_md, ms):
    # config defaults (densities/sound speeds) resolve; tag lookup errors are explicit
    res = TR.transfer_factor(sector_md, None, ms, t_end=2.0 * TR.default_pulse_width_us(),
                             equivalent_static=False, allow_pending=True)
    assert np.isfinite(res["sigma1_static"])
    assert set(res["materials"]) == {"wall", "fibrous", "calcium"}
    # displacement formulation: same physics, static peak within discretisation differences
    res_d = TR.transfer_factor(sector_md, TAG_CALCIUM, ms, t_end=2.0 * TR.default_pulse_width_us(),
                               equivalent_static=False, formulation="displacement", allow_pending=True)
    assert res_d["sigma1_static"] == pytest.approx(res["sigma1_static"], rel=0.1)
    assert res_d["info"]["formulation"] == "displacement"
    with pytest.raises(ValueError):
        TR.transfer_factor(sector_md, TAG_CALCIUM, ms, formulation="bogus")
    with pytest.raises(ValueError):
        TR.transfer_factor(sector_md, 99, ms)


def _lame_errors(md, model, h, gdim):
    """Cell sigma1 vs Lame hoop stress (p = 1) for a homogeneous annulus/tube."""
    k_found = P.get("loads.perivascular_k_MPa_per_mm").value
    n = md.mesh.topology.index_map(gdim).size_local
    cells = np.arange(n, dtype=np.int32)
    cen = md.mesh.geometry.x[md.mesh.geometry.dofmaps[0][cells]].mean(axis=1)
    r = np.hypot(cen[:, 0], cen[:, 1])
    a, _, b = _radii()
    hoop = a * a / (b * b - a * a) * (1.0 + b * b / r ** 2)
    sel = (r > a + 2 * h) & (r < b - 2 * h)     # interior cells (avoid boundary-layer cells)
    s1 = model.cell_stress_operator(cells)(*model.static_solution(1.0, k_found))
    err = np.abs(s1[sel] - hoop[sel]) / hoop.max()
    return err, abs(s1.max() / hoop.max() - 1.0)


def test_static_matches_lame_homogeneous(ms, acoustic):
    """Static branch (T's denominator), 2D: hoop stress of a homogeneous wall annulus vs Lame."""
    md = sector_annulus_meshdata(None, h_soft=H_CA)
    rho, c = acoustic
    res = {}
    for form, vol in (("mixed", "full"), ("displacement", "full"), ("displacement", "reduced")):
        model = TR.build_model(md, ms, rho, c, outer_bc="free", formulation=form, volumetric=vol)
        res[(form, vol)] = _lame_errors(md, model, H_CA, 2)
        err, peak = res[(form, vol)]
        print(f"2D Lame (K/mu = {model.materials[TAG_WALL].K / model.materials[TAG_WALL].mu:.3g}) {form}/{vol}: "
              f"peak err {peak:.4f}, cell err p99 {np.percentile(err, 99):.4f}, max {err.max():.4f}")
    err, peak = res[("mixed", "full")]
    assert err.max() < 0.01 and peak < 0.01
    # displacement/full: accurate peak and bulk, isolated pressure-noise cells (documented)
    err, peak = res[("displacement", "full")]
    assert peak < 0.01 and np.percentile(err, 99) < 0.01 and err.max() < 0.1
    err, peak = res[("displacement", "reduced")]
    assert err.max() < 0.01


def layered_annulus_meshdata(radii, tags, h):
    """Concentric layers ``radii[i] < r < radii[i+1]`` with cell tag ``tags[i]`` (2D)."""
    import gmsh
    from mpi4py import MPI
    from dolfinx.io.gmsh import model_to_mesh

    owns = not gmsh.isInitialized()
    if owns:
        gmsh.initialize(interruptible=False)
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(f"layers_{np.random.default_rng().integers(1 << 30)}")
    try:
        occ = gmsh.model.occ
        disks = [occ.addDisk(0, 0, 0, r, r) for r in radii]
        occ.fragment([(2, disks[-1])], [(2, d) for d in disks[:-1]])
        occ.synchronize()
        groups: dict[int, list[int]] = {}
        for _, s in gmsh.model.getEntities(2):
            r_max = gmsh.model.getBoundingBox(2, s)[3]
            k = int(np.argmin([abs(r_max - r) for r in radii]))
            if k == 0:
                occ.remove([(2, s)], recursive=True)          # the lumen disk
            else:
                groups.setdefault(tags[k - 1], []).append(s)
        occ.synchronize()
        for tag, ss in groups.items():
            gmsh.model.addPhysicalGroup(2, ss, tag)
        ents = [(2, s) for ss in groups.values() for s in ss]
        lum, out = [], []
        for _, e in gmsh.model.getBoundary(ents, combined=True, oriented=False):
            r_max = gmsh.model.getBoundingBox(1, abs(e))[3]
            (lum if r_max < 0.5 * (radii[0] + radii[-1]) else out).append(abs(e))
        gmsh.model.addPhysicalGroup(1, lum, TAG_LUMEN_BND)
        gmsh.model.addPhysicalGroup(1, out, TAG_OUTER_BND)
        gmsh.option.setNumber("Mesh.MeshSizeMax", h)
        gmsh.option.setNumber("Mesh.MeshSizeMin", h)
        gmsh.model.mesh.generate(2)
        return model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
    finally:
        gmsh.model.remove()
        if owns:
            gmsh.finalize()


def _layered_lame(radii, tags, mats, p):
    """Exact plane-strain solution u_r = A_i r + B_i / r per layer; returns (n_layers, 2) [A, B]."""
    n = len(tags)
    A, b = np.zeros((2 * n, 2 * n)), np.zeros(2 * n)

    def srr(i, r):
        m = mats[tags[i]]
        return np.array([2.0 * (m.lam + m.mu), -2.0 * m.mu / r ** 2])

    A[0, 0:2], b[0] = srr(0, radii[0]), -p
    for i in range(n - 1):
        r = radii[i + 1]
        A[2 * i + 1, 2 * i:2 * i + 4] = [r, 1.0 / r, -r, -1.0 / r]               # u_r continuous
        A[2 * i + 2, 2 * i:2 * i + 2], A[2 * i + 2, 2 * i + 2:2 * i + 4] = srr(i, r), -srr(i + 1, r)
    A[-1, -2:] = srr(n - 1, radii[-1])                                           # traction-free outside
    return np.linalg.solve(A, b).reshape(n, 2)


def test_static_layered_interfaces(ms, acoustic):
    """Multi-material ring (fibrous / calcium / fibrous / wall) vs the exact layered Lame solution.

    Regression for a defect of the first implementation: a Taylor-Hood pressure that is
    continuous *across* material interfaces cannot represent the pressure jump where mu and K
    jump, which polluted the calcium stresses next to the interfaces (cell errors up to ~30 %
    of the peak, not converging). The default per-material pressure is exact to < 1 %.
    """
    r_l, r_iel, r_o = _radii()
    d = P.get("geometry.depth_superficial_mm").value
    t_ca = float(P.raw("design.core_factorial.thickness_mm")[0])
    radii = [r_l, r_l + d, r_l + d + t_ca, r_iel, r_o]
    tags = [TAG_FIBROUS, TAG_CALCIUM, TAG_FIBROUS, TAG_WALL]
    md = layered_annulus_meshdata(radii, tags, H_CA)
    rho, c = acoustic
    k_found = P.get("loads.perivascular_k_MPa_per_mm").value
    cells = np.sort(md.cell_tags.find(TAG_CALCIUM)).astype(np.int32)
    cen = md.mesh.geometry.x[md.mesh.geometry.dofmaps[0][cells]].mean(axis=1)
    r = np.hypot(cen[:, 0], cen[:, 1])
    errs = {}
    for form, ip in (("mixed", "discontinuous"), ("displacement", "discontinuous"), ("mixed", "continuous")):
        model = TR.build_model(md, ms, rho, c, outer_bc="free", formulation=form, interface_pressure=ip)
        a, b = _layered_lame(radii, tags, model.materials, 1.0)[1]
        m = model.materials[TAG_CALCIUM]
        s_tt = 2 * (m.lam + m.mu) * a + 2 * m.mu * b / r ** 2
        s_rr = 2 * (m.lam + m.mu) * a - 2 * m.mu * b / r ** 2
        exact = np.maximum(np.maximum(s_tt, s_rr), 2 * m.lam * a)
        s1 = model.cell_stress_operator(cells)(*model.static_solution(1.0, k_found))
        errs[(form, ip)] = (np.abs(s1 - exact).max() / exact.max(), abs(s1.max() / exact.max() - 1.0))
        print(f"layered ring {form}/{ip}: calcium cell err max {errs[(form, ip)][0]:.4f}, "
              f"peak err {errs[(form, ip)][1]:.4f}")
    for key in (("mixed", "discontinuous"), ("displacement", "discontinuous")):
        assert errs[key][0] < 0.01 and errs[key][1] < 0.01
    assert errs[("mixed", "continuous")][0] > 0.05       # documents the interface error of the old default


def test_static_matches_lame_3d(ms, acoustic):
    """3D tube (u_z = 0 ends = plane strain): mixed is locking-free, P2 displacement locks."""
    h = H_SOFT
    md = sector_annulus_meshdata(None, h_soft=h, length=LAME_3D_LENGTH)
    rho, c = acoustic
    out = {}
    for form in ("mixed", "displacement"):
        model = TR.build_model(md, ms, rho, c, outer_bc="free", formulation=form)
        out[form] = _lame_errors(md, model, h, 3)
        err, peak = out[form]
        print(f"3D Lame {form}: peak err {peak:.4f}, cell err p99 {np.percentile(err, 99):.4f}, max {err.max():.4f}")
    err, peak = out["mixed"]
    assert err.max() < 0.03 and peak < 0.03
    assert out["displacement"][1] > peak     # documents the locking of P2 tetrahedra at K/mu ~ 1e5


def test_transfer_factor_3d_plane_strain_consistency(ms, acoustic):
    """3D extrusion with u_z = 0 ends (plane strain) reproduces the 2D transfer factor."""
    rho, c = acoustic
    kw = dict(t_end=T3D_WINDOW_PULSES * TR.default_pulse_width_us(), equivalent_static=False)
    md3 = sector_annulus_meshdata(SECTOR_ARC, h_ca=H_CA_3D, h_soft=H_SOFT_3D, length=LENGTH_3D)
    md2 = sector_annulus_meshdata(SECTOR_ARC, h_ca=H_CA_3D, h_soft=H_SOFT_3D)
    r3 = TR.transfer_factor(md3, TAG_CALCIUM, ms, rho, c, **kw)
    r2 = TR.transfer_factor(md2, TAG_CALCIUM, ms, rho, c, **kw)
    print(f"3D sector: T = {r3['T']:.4f} (ndofs {r3['info']['ndofs']}), 2D: T = {r2['T']:.4f}; "
          f"sigma1_static 3D {r3['sigma1_static']:.3f} / 2D {r2['sigma1_static']:.3f} MPa; "
          f"min history 3D {np.nanmin(r3['sigma1_max_history']):.3f} / 2D {np.nanmin(r2['sigma1_max_history']):.3f}")
    assert np.isfinite(r3["T"]) and 0.0 < r3["T"] < T_BOUND
    assert r3["sigma1_static"] == pytest.approx(r2["sigma1_static"], rel=0.15)
    assert np.nanmin(r3["sigma1_max_history"]) == pytest.approx(np.nanmin(r2["sigma1_max_history"]), rel=0.05)
    assert r3["T"] == pytest.approx(r2["T"], rel=0.5)

"""Small-strain linear elastic predictor analysis (design.md section 4).

Problem
-------
Plane strain (2D meshes) or 3D small-strain elasticity

    find u in V:  a(u, v) = L(v)  for all v in V,
    a(u, v) = sum_tags  int sigma(u) : eps(v) dx  +  k_f int u . v ds(TAG_OUTER_BND)
    L(v)    = - p_lumen int n . v ds(TAG_LUMEN_BND)

with ``sigma = lambda tr(eps) I + 2 mu eps`` and per-tag Lame constants
``MaterialSet.lame(tag)`` stored as DG0 functions (one value per cell).
``n`` is the outward unit normal of the tissue domain, so on the lumen
boundary it points into the lumen and the traction ``-p n`` pushes the wall
outward. Rigid-body modes are removed by a weak elastic foundation of
stiffness ``k_f`` (MPa/mm) on the outer (adventitial) boundary; its default
``params.get('loads.perivascular_k_MPa_per_mm')`` is a negligible support
and the solver reports the ratio of the total foundation reaction to the
total applied lumen load (``foundation_ratio``) so this can be checked. In
3D the end faces (``TAG_END_BND``) additionally carry ``u_z = 0``.
Displacement: vector Lagrange of degree ``degree`` (default P2). Linear
system: direct LU (MUMPS).

Post-processing (calcium cells)
-------------------------------
For every cell ``K`` the *cell averages* (quadrature averages)

    <q>_K = (1/|K|) int_K q dx

of the full 3x3 small-strain Cauchy stress and strain tensors, of the strain
energy density ``SED = 1/2 sigma : eps``, and of the position ``x`` (the
centroid) are obtained in a single assembly against a blocked DG0 test
function (exact quadrature: UFL estimates the polynomial degree). In plane
strain the 3D tensors are

    eps3   = [[e11, e12, 0], [e21, e22, 0], [0, 0, 0]]
    sigma3 = [[s11, s12, 0], [s21, s22, 0], [0, 0, lambda tr(eps)]]

so the reported maximum principal stress ``sigma1`` includes the
out-of-plane stress ``sigma_33 = lambda (e11 + e22)``, and the maximum
principal strain ``eps1 = max(eig_2D, 0)`` (``e33 = 0``). Principal values
are the eigenvalues of the *cell-averaged* tensors; the SED is the cell
average of the pointwise SED (not the SED of the averaged tensors).

Verification
------------
:func:`verification_lame` builds a thick-walled annulus with gmsh directly
(single material), solves the pressurised plane-strain problem with the
default weak foundation and compares radial displacement and hoop stress at
interior sample radii with the Lame solution (see ``tests/test_lame.py``).
"""
from __future__ import annotations

import contextlib
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import ufl

from dtivl import ROOT
from dtivl import params as P
from dtivl.geometry.labels import (MATERIAL_NAMES, TAG_CALCIUM, TAG_END_BND, TAG_FIBROUS, TAG_LUMEN_BND,
                                   TAG_OUTER_BND)
from dtivl.materials.constitutive import MaterialSet

#: PETSc options of the direct solver (numerical setting).
LU_OPTIONS = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "ksp_error_if_not_converged": True,
}

# --- Lame verification settings (numerical test geometry, not literature) ---
LAME_A = 1.0                    #: inner radius (mm)
LAME_B = 2.0                    #: outer radius (mm)
LAME_P = 0.1                    #: lumen pressure (MPa)
LAME_H = 0.025                  #: mesh size (mm), linear triangles ("refined mesh";
#: at 0.05 mm the pointwise hoop stress of the nu~0.49 case is off by ~1.8 %)
LAME_DEGREE = 2                 #: displacement degree
LAME_TOL = 0.01                 #: max relative error allowed (1 %)
LAME_FOUNDATION_TOL = 1e-3      #: max foundation reaction / applied load
LAME_SAMPLE_RADII = (1.1, 1.25, 1.5, 1.75, 1.9)   #: interior radii (mm)
LAME_N_SAMPLE_ANGLES = 12
#: Materials used for the verification: calcium (compressible) and the
#: fibrous intima (nearly incompressible, K/mu from config).
LAME_TAGS = (TAG_CALCIUM, TAG_FIBROUS)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class LinearResult:
    """Output of :func:`solve_linear`.

    Per-cell arrays refer to the calcium cells ``ca_cells`` (local cell
    indices, in the order of ``meshdata.cell_tags.find(TAG_CALCIUM)``).
    """

    meshdata: object
    u: object                         # dolfinx.fem.Function (displacement, mm)
    lam: object                       # DG0 Function, Lame lambda (MPa)
    mu: object                        # DG0 Function, shear modulus (MPa)
    p_lumen: float
    k_found: float
    degree: int
    gdim: int
    ca_cells: np.ndarray              # (n,) int32
    sigma1: np.ndarray                # (n,) max principal Cauchy stress (MPa)
    eps1: np.ndarray                  # (n,) max principal strain (-)
    sed: np.ndarray                   # (n,) cell-averaged SED (MPa = mJ/mm^3)
    measure: np.ndarray               # (n,) cell area (2D, mm^2) or volume (3D, mm^3)
    centroid: np.ndarray              # (n, gdim) cell centroid (mm)
    sigma: np.ndarray                 # (n, 3, 3) cell-averaged stress tensor
    eps: np.ndarray                   # (n, 3, 3) cell-averaged strain tensor
    foundation_ratio: float           # int k_f|u| ds(outer) / (p |lumen boundary|)
    info: dict = field(default_factory=dict)

    @property
    def plane_strain(self) -> bool:
        return self.gdim == 2


# ---------------------------------------------------------------------------
# Kinematics with the 3D embedding
# ---------------------------------------------------------------------------
def _strain_stress_3d(u, lam, mu):
    """3x3 small-strain tensors (plane strain embedding for 2D ``u``)."""
    gdim = u.ufl_shape[0]
    e = ufl.sym(ufl.grad(u))
    if gdim == 3:
        eps3 = e
        sig3 = lam * ufl.tr(e) * ufl.Identity(3) + 2.0 * mu * e
        return eps3, sig3
    s = lam * ufl.tr(e) * ufl.Identity(2) + 2.0 * mu * e
    eps3 = ufl.as_matrix([[e[0, 0], e[0, 1], 0], [e[1, 0], e[1, 1], 0], [0, 0, 0]])
    sig3 = ufl.as_matrix([[s[0, 0], s[0, 1], 0], [s[1, 0], s[1, 1], 0], [0, 0, lam * ufl.tr(e)]])
    return eps3, sig3


def _sym_components(T):
    """The 6 independent components (11, 22, 33, 12, 13, 23) of a 3x3 tensor."""
    return [T[0, 0], T[1, 1], T[2, 2], T[0, 1], T[0, 2], T[1, 2]]


def _from_sym(c: np.ndarray) -> np.ndarray:
    """(n, 6) components -> (n, 3, 3) symmetric tensors."""
    n = c.shape[0]
    T = np.empty((n, 3, 3))
    T[:, 0, 0], T[:, 1, 1], T[:, 2, 2] = c[:, 0], c[:, 1], c[:, 2]
    T[:, 0, 1] = T[:, 1, 0] = c[:, 3]
    T[:, 0, 2] = T[:, 2, 0] = c[:, 4]
    T[:, 1, 2] = T[:, 2, 1] = c[:, 5]
    return T


# ---------------------------------------------------------------------------
# Material fields
# ---------------------------------------------------------------------------
def lame_fields(meshdata, ms: MaterialSet):
    """DG0 functions (lambda, mu) with the per-tag constants of ``ms``.

    Every cell must carry a known material tag; a ``ValueError`` is raised
    otherwise.
    """
    from dolfinx import fem

    mesh, ct = meshdata.mesh, meshdata.cell_tags
    Q0 = fem.functionspace(mesh, ("DG", 0))
    lam, mu = fem.Function(Q0, name="lambda"), fem.Function(Q0, name="mu")
    n_local = mesh.topology.index_map(mesh.topology.dim).size_local
    assigned = np.zeros(n_local + mesh.topology.index_map(mesh.topology.dim).num_ghosts, dtype=bool)
    dofs_of = Q0.dofmap.list
    for tag in np.unique(ct.values):
        if int(tag) not in MATERIAL_NAMES:
            raise ValueError(f"unknown cell tag {tag}")
        cells = ct.find(int(tag))
        lam_t, mu_t = ms.lame(int(tag))
        dofs = dofs_of[cells, 0]
        lam.x.array[dofs] = lam_t
        mu.x.array[dofs] = mu_t
        assigned[cells] = True
    if not assigned[:n_local].all():
        raise ValueError(f"{int((~assigned[:n_local]).sum())} cells carry no material tag")
    lam.x.scatter_forward()
    mu.x.scatter_forward()
    return lam, mu


# ---------------------------------------------------------------------------
# Cell averages
# ---------------------------------------------------------------------------
def cell_averages(u, lam, mu, cells: np.ndarray | None = None) -> dict:
    """Cell (quadrature) averages of stress, strain, SED and position.

    Returns a dict with ``sigma``/``eps`` (n, 3, 3), ``sigma1``/``eps1``
    (max principal values of the averaged tensors), ``sed`` (average of
    ``1/2 sigma:eps``), ``measure`` (cell area/volume) and ``centroid``
    (n, gdim) for the requested local ``cells`` (default: all owned cells).
    """
    from dolfinx import fem

    mesh = u.function_space.mesh
    gdim = mesh.geometry.dim
    eps3, sig3 = _strain_stress_3d(u, lam, mu)
    x = ufl.SpatialCoordinate(mesh)
    comps = (_sym_components(sig3) + _sym_components(eps3)
             + [0.5 * ufl.inner(sig3, eps3), 1.0] + [x[i] for i in range(gdim)])
    ncomp = len(comps)
    W = fem.functionspace(mesh, ("DG", 0, (ncomp,)))
    w = ufl.TestFunction(W)
    vec = fem.assemble_vector(fem.form(ufl.inner(ufl.as_vector(comps), w) * ufl.dx))
    if cells is None:
        cells = np.arange(mesh.topology.index_map(mesh.topology.dim).size_local, dtype=np.int32)
    blocks = W.dofmap.list[cells, 0]
    data = vec.array.reshape(-1, ncomp)[blocks]
    meas = data[:, 13]
    avg = data / meas[:, None]
    sigma, eps = _from_sym(avg[:, 0:6]), _from_sym(avg[:, 6:12])
    return {
        "sigma": sigma,
        "eps": eps,
        "sigma1": np.linalg.eigvalsh(sigma)[:, -1],
        "eps1": np.linalg.eigvalsh(eps)[:, -1],
        "sed": avg[:, 12],
        "measure": meas,
        "centroid": avg[:, 14:14 + gdim],
    }


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
def solve_linear(meshdata, ms: MaterialSet, p_lumen: float, *, k_found: float | None = None,
                 degree: int = 2, petsc_options: dict | None = None) -> LinearResult:
    """Solve the small-strain lumen-pressure problem (plane strain or 3D).

    Parameters
    ----------
    meshdata : ``dolfinx.io.gmsh.MeshData`` with cell tags ``TAG_*`` and
        facet tags ``TAG_LUMEN_BND``, ``TAG_OUTER_BND`` (3D: ``TAG_END_BND``).
    ms : material set (per-tag Lame constants from ``ms.lame(tag)``).
    p_lumen : lumen pressure (MPa), applied as traction ``-p n``.
    k_found : foundation stiffness on the outer boundary (MPa/mm); default
        ``params.get('loads.perivascular_k_MPa_per_mm')``. Must be > 0 (it
        is the only constraint against in-plane rigid-body motion).
    degree : displacement polynomial degree.
    petsc_options : overrides of :data:`LU_OPTIONS`.
    """
    from dolfinx import fem
    from dolfinx.fem.petsc import LinearProblem

    if k_found is None:
        k_found = P.get("loads.perivascular_k_MPa_per_mm").value
    if not k_found > 0:
        raise ValueError("k_found must be positive (rigid-body modes are otherwise free)")
    mesh, ft = meshdata.mesh, meshdata.facet_tags
    if ft is None:
        raise ValueError("meshdata has no facet tags (lumen/outer boundaries required)")
    gdim, tdim = mesh.geometry.dim, mesh.topology.dim
    if gdim != tdim:
        raise ValueError("solve_linear expects gdim == tdim (2D plane strain or 3D)")
    for tag in (TAG_LUMEN_BND, TAG_OUTER_BND):
        if ft.find(tag).size == 0 and mesh.comm.allreduce(ft.find(tag).size) == 0:
            raise ValueError(f"facet tag {tag} missing")
    mesh.topology.create_connectivity(tdim - 1, tdim)

    lam, mu = lame_fields(meshdata, ms)
    V = fem.functionspace(mesh, ("Lagrange", degree, (gdim,)))
    u_, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=ft)
    n = ufl.FacetNormal(mesh)
    eps = lambda w: ufl.sym(ufl.grad(w))  # noqa: E731
    sig = lambda w: lam * ufl.tr(eps(w)) * ufl.Identity(gdim) + 2.0 * mu * eps(w)  # noqa: E731
    kf = fem.Constant(mesh, float(k_found))
    p = fem.Constant(mesh, float(p_lumen))
    a = ufl.inner(sig(u_), eps(v)) * ufl.dx + kf * ufl.inner(u_, v) * ds(TAG_OUTER_BND)
    L = -p * ufl.inner(n, v) * ds(TAG_LUMEN_BND)

    bcs = []
    if gdim == 3:
        end_facets = ft.find(TAG_END_BND)
        Vz = V.sub(2)
        dofs = fem.locate_dofs_topological(Vz, tdim - 1, end_facets)
        bcs.append(fem.dirichletbc(0.0, dofs, Vz))

    opts = dict(LU_OPTIONS)
    if petsc_options:
        opts.update(petsc_options)
    uh = fem.Function(V, name="u")
    problem = LinearProblem(a, L, u=uh, bcs=bcs, petsc_options=opts,
                            petsc_options_prefix=f"dtivl_linear_{uuid.uuid4().hex[:8]}_")
    problem.solve()
    reason = problem.solver.getConvergedReason()
    if reason <= 0:
        raise RuntimeError(f"linear solve failed (KSP reason {reason})")

    # Foundation reaction vs applied load (both as total force magnitudes).
    reac = fem.assemble_scalar(fem.form(kf * ufl.sqrt(ufl.inner(uh, uh)) * ds(TAG_OUTER_BND)))
    lumen_meas = fem.assemble_scalar(fem.form(1.0 * ds(TAG_LUMEN_BND)))
    reac, lumen_meas = mesh.comm.allreduce(reac), mesh.comm.allreduce(lumen_meas)
    applied = abs(float(p_lumen)) * lumen_meas
    f_ratio = float(reac / applied) if applied > 0 else 0.0

    ca_cells = meshdata.cell_tags.find(TAG_CALCIUM)
    n_owned = mesh.topology.index_map(tdim).size_local
    ca_cells = ca_cells[ca_cells < n_owned].astype(np.int32)
    ca = cell_averages(uh, lam, mu, ca_cells)
    info = {"ndofs": int(V.dofmap.index_map.size_global * V.dofmap.index_map_bs),
            "ncells": int(mesh.topology.index_map(tdim).size_global),
            "ksp_reason": int(reason), "lumen_boundary_measure": float(lumen_meas)}
    return LinearResult(meshdata=meshdata, u=uh, lam=lam, mu=mu, p_lumen=float(p_lumen),
                        k_found=float(k_found), degree=degree, gdim=gdim, ca_cells=ca_cells,
                        sigma1=ca["sigma1"], eps1=ca["eps1"], sed=ca["sed"], measure=ca["measure"],
                        centroid=ca["centroid"], sigma=ca["sigma"], eps=ca["eps"],
                        foundation_ratio=f_ratio, info=info)


# ---------------------------------------------------------------------------
# Point evaluation of the stress field
# ---------------------------------------------------------------------------
def stress_at_points(linres: LinearResult, points: np.ndarray) -> np.ndarray:
    """Pointwise 3x3 Cauchy stress (n, 3, 3) at ``points`` (n, gdim).

    The stress is interpolated into a discontinuous Lagrange space of degree
    ``degree - 1`` (exact for affine cells) and evaluated in the first
    colliding cell of each point (local mesh only; serial use).
    """
    from dolfinx import fem

    mesh = linres.u.function_space.mesh
    _, sig3 = _strain_stress_3d(linres.u, linres.lam, linres.mu)
    S = fem.functionspace(mesh, ("DG", max(linres.degree - 1, 0), (3, 3)))
    s = fem.Function(S)
    s.interpolate(fem.Expression(sig3, S.element.interpolation_points))
    vals = _eval(s, points)
    return vals.reshape(-1, 3, 3)


def displacement_at_points(linres: LinearResult, points: np.ndarray) -> np.ndarray:
    """Displacement (n, gdim) at ``points`` (n, gdim) (local mesh; serial use)."""
    return _eval(linres.u, points)


def _eval(fun, points: np.ndarray) -> np.ndarray:
    from dolfinx import geometry

    mesh = fun.function_space.mesh
    pts = np.zeros((len(points), 3))
    pts[:, :points.shape[1]] = points
    tree = geometry.bb_tree(mesh, mesh.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    coll = geometry.compute_colliding_cells(mesh, cand, pts)
    cells = np.empty(len(pts), dtype=np.int32)
    for i in range(len(pts)):
        links = coll.links(i)
        if len(links) == 0:
            raise ValueError(f"point {points[i]} is outside the mesh")
        cells[i] = links[0]
    return fun.eval(pts, cells)


# ---------------------------------------------------------------------------
# Lame verification
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def _gmsh_model(name: str):
    """Fresh gmsh model; initialises/finalises gmsh only if not already active."""
    import gmsh

    owns = not gmsh.isInitialized()
    if owns:
        gmsh.initialize(interruptible=False)
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


def annulus_meshdata(a: float, b: float, h: float, cell_tag: int, comm=None):
    """Single-material annulus a <= r <= b (linear triangles, size ``h``).

    Cell tag ``cell_tag``; facet tags ``TAG_LUMEN_BND`` (r = a) and
    ``TAG_OUTER_BND`` (r = b). Returns ``dolfinx.io.gmsh.MeshData``.
    """
    from mpi4py import MPI
    from dolfinx.io.gmsh import model_to_mesh

    comm = MPI.COMM_WORLD if comm is None else comm
    with _gmsh_model("annulus") as gmsh:
        occ = gmsh.model.occ
        outer = occ.addDisk(0, 0, 0, b, b)
        inner = occ.addDisk(0, 0, 0, a, a)
        occ.cut([(2, outer)], [(2, inner)])
        occ.synchronize()
        surf = [s for _, s in gmsh.model.getEntities(2)]
        gmsh.model.addPhysicalGroup(2, surf, cell_tag)
        inner_c, outer_c = [], []
        for _, c in gmsh.model.getBoundary([(2, s) for s in surf], oriented=False):
            xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(1, c)
            (outer_c if max(xmax - xmin, ymax - ymin) > a + b else inner_c).append(c)
        gmsh.model.addPhysicalGroup(1, inner_c, TAG_LUMEN_BND)
        gmsh.model.addPhysicalGroup(1, outer_c, TAG_OUTER_BND)
        gmsh.option.setNumber("Mesh.MeshSizeMin", h)
        gmsh.option.setNumber("Mesh.MeshSizeMax", h)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.model.mesh.generate(2)
        return model_to_mesh(gmsh.model, comm, 0, gdim=2)


def lame_exact(r, a: float, b: float, p: float, lam: float, mu: float):
    """Plane-strain Lame solution: (u_r, sigma_rr, sigma_tt) at radii ``r``."""
    r = np.asarray(r, dtype=float)
    A = p * a ** 2 / (b ** 2 - a ** 2)
    B = p * a ** 2 * b ** 2 / (b ** 2 - a ** 2)
    ur = A * r / (2.0 * (lam + mu)) + B / (2.0 * mu * r)
    return ur, A - B / r ** 2, A + B / r ** 2


def lame_check(ms: MaterialSet | None = None, *, tag: int = TAG_CALCIUM, a: float = LAME_A,
               b: float = LAME_B, p: float = LAME_P, h: float = LAME_H, degree: int = LAME_DEGREE,
               k_found: float | None = None) -> dict:
    """Solve the pressurised annulus and compare with Lame (one material).

    The pipeline's default weak foundation is kept (instead of point
    constraints, which would need an ad-hoc choice of nodes and create
    singular reactions); its effect is checked through ``foundation_ratio``
    (total foundation reaction / total applied load), which bounds the
    relative perturbation of the outer traction-free condition.
    Errors: pointwise radial displacement and hoop stress at
    ``LAME_SAMPLE_RADII`` x ``LAME_N_SAMPLE_ANGLES`` interior points, and the
    cell-averaged hoop stress of every cell against the exact value at its
    centroid. Returns a dict of errors and settings.
    """
    ms = MaterialSet.baseline() if ms is None else ms
    md = annulus_meshdata(a, b, h, tag)
    res = solve_linear(md, ms, p, k_found=k_found, degree=degree)
    lam, mu = ms.lame(tag)
    th = 2.0 * np.pi * (np.arange(LAME_N_SAMPLE_ANGLES) + 0.25) / LAME_N_SAMPLE_ANGLES
    R, T = np.meshgrid(np.asarray(LAME_SAMPLE_RADII), th, indexing="ij")
    R, T = R.ravel(), T.ravel()
    pts = np.column_stack([R * np.cos(T), R * np.sin(T)])
    er = np.column_stack([np.cos(T), np.sin(T)])
    et = np.column_stack([-np.sin(T), np.cos(T)])
    u = displacement_at_points(res, pts)
    s = stress_at_points(res, pts)[:, :2, :2]
    ur_num = np.einsum("ni,ni->n", u, er)
    stt_num = np.einsum("ni,nij,nj->n", et, s, et)
    ur_ex, _, stt_ex = lame_exact(R, a, b, p, lam, mu)
    err_ur = np.abs(ur_num - ur_ex) / np.abs(ur_ex)
    err_tt = np.abs(stt_num - stt_ex) / np.abs(stt_ex)
    # Cell-averaged hoop stress in every cell (the quantity used by the
    # predictors) vs the exact value at the cell centroid.
    ca = cell_averages(res.u, res.lam, res.mu)
    xc = ca["centroid"]
    rc, tc = np.hypot(xc[:, 0], xc[:, 1]), np.arctan2(xc[:, 1], xc[:, 0])
    etc = np.column_stack([-np.sin(tc), np.cos(tc)])
    stt_cell = np.einsum("ni,nij,nj->n", etc, ca["sigma"][:, :2, :2], etc)
    err_cell = np.abs(stt_cell - lame_exact(rc, a, b, p, lam, mu)[2]) / lame_exact(rc, a, b, p, lam, mu)[2]
    out = {
        "material": MATERIAL_NAMES[tag], "lambda_MPa": lam, "mu_MPa": mu,
        "nu": lam / (2.0 * (lam + mu)),
        "a_mm": a, "b_mm": b, "p_MPa": p, "h_mm": h, "degree": degree,
        "k_found_MPa_per_mm": res.k_found, "ndofs": res.info["ndofs"], "ncells": res.info["ncells"],
        "sample_radii_mm": list(LAME_SAMPLE_RADII), "n_sample_angles": LAME_N_SAMPLE_ANGLES,
        "max_rel_err_ur": float(err_ur.max()), "max_rel_err_hoop": float(err_tt.max()),
        "max_rel_err_hoop_cellavg": float(err_cell.max()),
        "foundation_ratio": res.foundation_ratio,
        "per_radius": [
            {"r_mm": float(r), "ur_exact": float(lame_exact(r, a, b, p, lam, mu)[0]),
             "ur_num_mean": float(ur_num[R == r].mean()),
             "hoop_exact": float(lame_exact(r, a, b, p, lam, mu)[2]),
             "hoop_num_mean": float(stt_num[R == r].mean())}
            for r in LAME_SAMPLE_RADII],
    }
    out["passed"] = bool(max(out["max_rel_err_ur"], out["max_rel_err_hoop"],
                             out["max_rel_err_hoop_cellavg"]) < LAME_TOL
                         and out["foundation_ratio"] < LAME_FOUNDATION_TOL)
    return out


def verification_lame(out_json: str | Path | None = None, *, ms: MaterialSet | None = None,
                      tags: tuple[int, ...] = LAME_TAGS) -> dict:
    """Run the Lame verification for each material in ``tags`` and write JSON.

    ``out_json`` defaults to ``results/verification_lame.json``. Returns the
    written dict (``passed`` is True only if every case passes).
    """
    import dolfinx

    out_json = ROOT / "results" / "verification_lame.json" if out_json is None else Path(out_json)
    cases = [lame_check(ms, tag=t) for t in tags]
    report = {
        "check": "plane-strain thick-walled cylinder vs Lame (design.md section 4)",
        "tolerance_rel": LAME_TOL, "foundation_tolerance": LAME_FOUNDATION_TOL,
        "dolfinx_version": dolfinx.__version__,
        "cases": cases, "passed": all(c["passed"] for c in cases),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2))
    return report

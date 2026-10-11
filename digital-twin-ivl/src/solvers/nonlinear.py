"""Finite-strain quasi-static solver with AT1 phase-field fracture (design.md section 5).

Unknowns and spaces
-------------------
``u``: vector Lagrange P2 (plane strain in 2D, F33 = 1; 3D with the vessel
axis along z and ``u_z = 0`` on ``TAG_END_BND``). ``d``: scalar P1 on the
whole mesh, confined to calcium by its upper bound and made irreversible by
its lower bound (see :mod:`dtivl.fracture.phasefield`). Material constants
enter per tag through ``dx(tag)`` with the UFL energies of
:mod:`dtivl.materials.constitutive`; the numerical values of the material
set are wrapped in ``dolfinx.fem.Constant`` objects so that forms are
compiled once per tag combination (sensitivity runs reuse the JIT cache).

Energy and residual (reference configuration, N = outward normal of the
tissue domain, which points into the lumen on ``TAG_LUMEN_BND``)
-------------------------------------------------------------------------
    Pi(u, d) = sum_soft int W_soft(F) dx(tag) + int [g(d) psi+(F) + psi-(F)] dx(Ca)
             + (Gc/c_w) int (d/ell + ell |grad d|^2) dx(Ca)
             + 1/2 k_f int |u|^2 ds(outer)
             + 1/2 k_pen int <R - rho>+^2 ds(lumen)        (inner rigid cylinder, if active)
             + 1/2 k_pen int <rho - R_cap>+^2 ds(lumen)    (outer envelope cap, if active)
    R_u(u; v) = D_u Pi [v] + p int J F^-T N . v ds(lumen)  (follower lumen pressure)

with ``rho = |(x + u)_perp|`` the distance of the current position from the
vessel axis (the origin in 2D, the z-axis in 3D), ``k_f`` =
``loads.perivascular_k_MPa_per_mm`` and ``k_pen`` =
``loads.contact_penalty_MPa_per_mm``. ``fracture="erosion"`` drops ``d`` and
scales the calcium energy cell-wise by a DG0 erosion factor
(:mod:`dtivl.fracture.erosion`); ``fracture="none"`` keeps the calcium intact.

Algorithm
---------
Per load increment, alternate minimisation: (i) ``u`` with ``d`` frozen,
SNES ``newtonls`` with backtracking line search and MUMPS LU; (ii) ``d`` with
``u`` frozen, SNES ``vinewtonrsls`` with bounds ``[d_prev, ub]``; repeated
until ``||d_k - d_{k-1}||_inf < STAGGER_TOL`` (at most ``STAGGER_MAX_IT``
iterations; non-convergence is logged and flagged in the step record, and
the step is accepted). After a change of ``d`` the displacement is re-solved
once so that recorded quantities are consistent with the final ``d``.
Adaptive load stepping: a failed solve restores the last converged state and
halves the increment; after a success the increment grows by
``STEP_GROWTH`` up to the nominal one. The minimum increment is the nominal
increment / 2**``MAX_HALVINGS``; below it the protocol stops with status
``"failed"``. The same mechanism implements the pressure-limited stent
expansion (a converged step whose mean contact pressure exceeds the rated
pressure is rejected), which then stops with status ``"limited"``.

Recorded per step (``state.history``)
-------------------------------------
``phase, step, load_name, load, increment, p, R_in, R_cap, max_d,
n_crack_cells, n_eroded, lumen_area, p_contact_mean, p_contact_max,
p_cap_mean, contact_force, E_elastic, E_fracture, E_contact, E_foundation,
stagger_its, stagger_err, stagger_converged, newton_its, d_its,
n_new_eroded`` (energies in N mm, per mm of length in plane strain).

Definitions:

* ``lumen_area``: area enclosed by the deformed lumen boundary,
  ``-1/2 int (x+u)_perp . (J F^-T N)_perp dS`` (Nanson). In 3D this is the
  lumen volume between the reference planes ``z0 <= Z <= z1`` divided by
  ``z1 - z0`` (axial mean), with ``[z0, z1]`` the axial extent of the
  calcium cells (whole segment if there is no calcium).
* ``p_contact_mean`` (balloon-equivalent mean contact pressure): total
  radial contact force of the inner cylinder per unit axial length divided
  by the cylinder circumference ``2 pi R`` (a balloon at pressure ``p`` in
  full apposition gives ``p``). ``p_cap_mean`` likewise for the cap.
  ``contact_force`` is the total radial force (N; per mm in plane strain).
* ``p_contact_max``: largest nodal contact pressure ``k_pen <R - rho>+`` over
  the P2 nodes of the lumen boundary (per reference area).
* ``max_d``: largest nodal damage; ``n_crack_cells``: calcium cells with
  cell-averaged ``d > fracture.d_crack`` (eroded cells in erosion mode).

Load protocols
--------------
:func:`run_pressure`, :func:`run_ivl` and :func:`run_stent` return a
:class:`ProtocolResult` (unpackable as ``state, history``) with a ``status``
(``"ok"``, ``"limited"`` or ``"failed"``) and a protocol ``summary``. Damage
is active in every protocol. ``run_ivl``: the envelope cap is kept during
the unloading ramp and removed at ``p = 0`` (removing it at peak pressure
would apply an unphysical load jump; while the wall recoils it is inactive
anyway). ``run_stent``: ``u`` is reset to zero first (stent after IVL starts
from the post-IVL ``d``), the cylinder starts ``STENT_START_GAP_FRAC`` inside
the minimum lumen radius ``R_min`` and the recoil solves
``P_tissue(R) = K_s (R_dep - R)`` on ``[R_min, R_dep]`` by bisection; the
unloaded wall has no residual deformation (hyperelastic-damage model), so
``P_tissue(R_min) = 0`` and the bracket is valid.

Discretisation note (contact on linear meshes): the pipeline meshes are
straight-sided, so the discrete lumen is a polygon. A rigid cylinder then
touches the chord midpoints while the vertices stay outside ``R`` by up to
the chord sag ``h^2 / (8 R)`` (about 3 um for h = 0.15 mm at R = 1 mm), which
is much larger than the penalty penetration ``p_c / k_pen``. This shifts the
effective cylinder radius by O(h^2/R) and is part of the mesh-convergence
error. The contact verification therefore uses curved (second-order) cells.

Verification helpers (design.md section 5.7): :func:`annulus_meshdata`,
:func:`tube_meshdata`, :func:`hyperelastic_lame_check`, :func:`contact_check`
and :func:`verification_nonlinear` (``results/verification_nonlinear.json``);
the AT1 onset check is
:func:`dtivl.fracture.phasefield.verification_phasefield`. Serial use.
"""
from __future__ import annotations

import contextlib
import json
import logging
import math
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

import numpy as np
import ufl

from dtivl import ROOT
from dtivl import params as P
from dtivl.fracture import erosion as ER
from dtivl.fracture import phasefield as PF
from dtivl.geometry.labels import (MATERIAL_NAMES, TAG_CALCIUM, TAG_END_BND, TAG_FIBROUS, TAG_LUMEN_BND,
                                   TAG_OUTER_BND)
from dtivl.materials.constitutive import (MaterialSet, deformation_gradient, degradation, psi_calcium_split,
                                          soft_energy)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Numerical settings
# ---------------------------------------------------------------------------
#: PETSc options of the displacement problem.
U_SNES_OPTIONS = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "bt",
    "snes_rtol": 1e-8,
    "snes_atol": 1e-10,
    "snes_max_it": 60,            # damaged calcium (vol-dev split) needs many damped Newton steps
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}
STAGGER_TOL = 1e-3            #: ||Delta d||_inf tolerance of the alternate minimisation
STAGGER_MAX_IT = 100          #: max alternate-minimisation iterations per increment
MAX_HALVINGS = 6              #: min increment = nominal / 2**MAX_HALVINGS
STEP_GROWTH = 2.0             #: increment growth after a converged step (capped at nominal)
STENT_START_GAP_FRAC = 0.01   #: rigid cylinder starts 1 % inside the minimum lumen radius
RECOIL_TOL_MM = 1e-4          #: bisection tolerance on R_final (mm)
RECOIL_MAX_IT = 40            #: max bisection iterations
#: Default number of increments per protocol phase.
DEFAULT_STEPS = {"pressure": 10, "ivl_inflate": 10, "ivl_pulse": 10, "ivl_unload": 10, "stent_expand": 20}

# --- verification settings (numerical test geometry, not literature) ---
VERIF_A, VERIF_B = 1.0, 2.0   #: annulus radii (mm)
VERIF_H = 0.15                #: mesh size (mm)
VERIF_ORDER = 2               #: curved (quadratic) cells: the discrete boundaries are circles
VERIF_P_REL = 1e-4            #: Lame check pressure = 1e-4 x shear modulus (small strain)
VERIF_LAME_TOL = 0.01         #: 1 % (design.md section 5.7)
VERIF_CONTACT_EXP = (0.01, 0.02)   #: contact check: R = a (1 + e)
VERIF_CONTACT_PEN_TOL = 0.5   #: |R - rho_i| <= (1 + tol) p_c/k_pen at contact nodes
VERIF_CONTACT_LAME_TOL = 0.1  #: mean contact pressure vs small-strain Lame estimate

_KEEP = object()
_MS_FIELDS = ("wall_c10", "wall_c20", "wall_c30", "fibrous_E", "lipid_E", "ca_E", "ca_nu", "k_over_mu")


class SolveError(RuntimeError):
    """A displacement or damage solve did not converge."""


@dataclass
class SolverSettings:
    """Numerical settings of :class:`NonlinearState` (defaults: module constants).

    ``erosion_split=True`` scales only ``psi+`` of eroded cells (see
    :mod:`dtivl.fracture.erosion`); ``quadrature_degree`` defaults to
    ``2 * degree``; ``u_petsc_options``/``d_petsc_options`` override
    :data:`U_SNES_OPTIONS` / :data:`dtivl.fracture.phasefield.D_SNES_OPTIONS`.
    """

    stagger_tol: float = STAGGER_TOL
    stagger_max_it: int = STAGGER_MAX_IT
    max_halvings: int = MAX_HALVINGS
    erosion_max_rounds: int = ER.ERODE_MAX_ROUNDS
    erosion_split: bool = False               # True: eroded cells keep psi- (compression)
    quadrature_degree: int | None = None      # default 2 * degree
    u_petsc_options: dict = field(default_factory=dict)
    d_petsc_options: dict = field(default_factory=dict)


@dataclass
class ProtocolResult:
    """Result of a load protocol; iterating yields ``(state, history)``."""

    state: "NonlinearState"
    history: list
    status: str
    summary: dict = field(default_factory=dict)

    def __iter__(self):
        yield self.state
        yield self.history


def _pos(x):
    """Macaulay bracket <x>+ (UFL)."""
    return ufl.conditional(ufl.gt(x, 0.0), x, 0.0)


def _ufl_material_set(mesh, ms: MaterialSet) -> MaterialSet:
    """Copy of ``ms`` whose numeric fields are ``fem.Constant`` (no recompilation per value)."""
    from dolfinx import fem

    return replace(ms, **{f: fem.Constant(mesh, float(getattr(ms, f))) for f in _MS_FIELDS})


def _mpi_max():
    from mpi4py import MPI
    return MPI.MAX


def _mpi_min():
    from mpi4py import MPI
    return MPI.MIN


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class NonlinearState:
    """Fields, forms and solvers of one lesion model.

    Parameters
    ----------
    meshdata : ``MeshData`` with cell tags ``TAG_*`` and facet tags
        (``TAG_LUMEN_BND``, ``TAG_OUTER_BND``; 3D ``TAG_END_BND``). Missing
        facet tags simply switch the corresponding terms off.
    ms : material set (default ``MaterialSet.baseline()``).
    fracture : ``"phasefield"`` (default), ``"erosion"`` or ``"none"``.
    h_ca : calcium mesh size used for ``ell = fracture.ell_over_h * h_ca``
        (default ``discretisation.h_calcium_mm``); ``ell`` overrides it.
    k_found, k_pen : foundation and contact penalty stiffness (MPa/mm);
        defaults from ``loads.*``.
    degree : displacement degree (2).
    bc_factory : optional callable ``V -> list[DirichletBC]`` with extra
        Dirichlet conditions (verification problems).
    settings : :class:`SolverSettings`.
    """

    def __init__(self, meshdata, ms: MaterialSet | None = None, *, fracture: str = "phasefield",
                 h_ca: float | None = None, ell: float | None = None, k_found: float | None = None,
                 k_pen: float | None = None, degree: int = 2,
                 bc_factory: Callable | None = None, settings: SolverSettings | None = None,
                 name: str = "dtivl_nl"):
        from dolfinx import fem
        from dolfinx import mesh as dmesh
        from dolfinx.fem.petsc import NonlinearProblem

        if fracture not in ("phasefield", "erosion", "none"):
            raise ValueError(f"unknown fracture model {fracture!r}")
        self.meshdata = meshdata
        self.ms = MaterialSet.baseline() if ms is None else ms
        self.fracture = fracture
        self.settings = SolverSettings() if settings is None else settings
        self.degree = degree
        msh = meshdata.mesh
        self.mesh, self.comm = msh, msh.comm
        gdim, tdim = msh.geometry.dim, msh.topology.dim
        if gdim != tdim:
            raise ValueError("expected gdim == tdim (2D plane strain or 3D)")
        self.gdim = gdim
        fdim = tdim - 1
        msh.topology.create_connectivity(fdim, tdim)
        msh.topology.create_connectivity(tdim, fdim)
        ct = meshdata.cell_tags
        if ct is None:
            raise ValueError("meshdata has no cell tags")
        ft = meshdata.facet_tags
        if ft is None:
            ft = dmesh.meshtags(msh, fdim, np.zeros(0, np.int32), np.zeros(0, np.int32))
        self.cell_tags, self.facet_tags = ct, ft
        tags = sorted({int(t) for vals in self.comm.allgather(np.unique(ct.values).tolist()) for t in vals})
        unknown = [t for t in tags if t not in MATERIAL_NAMES]
        if unknown:
            raise ValueError(f"unknown cell tags {unknown}")
        self.soft_tags = [t for t in tags if t != TAG_CALCIUM]
        n_owned = msh.topology.index_map(tdim).size_local
        ca = ct.find(TAG_CALCIUM)
        self.ca_cells = ca[ca < n_owned].astype(np.int32)
        self.has_calcium = TAG_CALCIUM in tags

        # --- parameters --------------------------------------------------
        self.k_found = P.get("loads.perivascular_k_MPa_per_mm").value if k_found is None else float(k_found)
        self.k_pen = P.get("loads.contact_penalty_MPa_per_mm").value if k_pen is None else float(k_pen)
        self.ell = PF.regularisation_length(h_ca) if ell is None else float(ell)
        self.Gc_value = self.ms.gc(self.ell)
        self.d_crack = float(P.raw("fracture.d_crack"))

        # --- spaces and functions ------------------------------------------
        V = fem.functionspace(msh, ("Lagrange", degree, (gdim,)))
        self.V, self._Vs = V, fem.functionspace(msh, ("Lagrange", degree))
        self.D = fem.functionspace(msh, ("Lagrange", 1))
        self.Q0 = fem.functionspace(msh, ("DG", 0))
        self.u = fem.Function(V, name="u")
        self.d = fem.Function(self.D, name="d")
        self.d_lb = fem.Function(self.D, name="d_prev")
        self.d_ub = PF.damage_upper_bound(self.D, ct)
        self.eta = fem.Function(self.Q0, name="eta")
        self.eta.x.array[:] = 1.0

        def const(v):
            return fem.Constant(msh, float(v))
        self._p, self._R_in, self._on_in = const(0.0), const(1.0), const(0.0)
        self._R_cap, self._on_cap = const(1.0), const(0.0)
        self._kpen, self._kf = const(self.k_pen), const(self.k_found)
        self._Gc, self._ell, self._kres = const(self.Gc_value), const(self.ell), const(self.ms.k_res)
        self._z0, self._z1 = const(0.0), const(0.0)
        self.loads = {"p": 0.0, "R_in": None, "R_cap": None}

        q = self.settings.quadrature_degree or 2 * degree
        self.dx = ufl.Measure("dx", domain=msh, subdomain_data=ct, metadata={"quadrature_degree": q})
        self.ds = ufl.Measure("ds", domain=msh, subdomain_data=ft, metadata={"quadrature_degree": q})
        dx, ds = self.dx, self.ds

        # --- energy ----------------------------------------------------------
        Fv, F3 = deformation_gradient(self.u, plane_strain=(gdim == 2))
        self.F, self.F3 = Fv, F3
        self.ms_ufl = _ufl_material_set(msh, self.ms)
        W_el = None
        for t in self.soft_tags:
            term = soft_energy(t, F3, self.ms_ufl) * dx(t)
            W_el = term if W_el is None else W_el + term
        self.psi_plus = None
        if self.has_calcium:
            psi_p, psi_m = psi_calcium_split(F3, self.ms_ufl.ca_mu, self.ms_ufl.ca_K)
            self.psi_plus = psi_p
            if fracture == "phasefield":
                W_ca = degradation(self.d, self._kres) * psi_p + psi_m
            elif fracture == "erosion" and self.settings.erosion_split:
                W_ca = self.eta * psi_p + psi_m
            elif fracture == "erosion":
                W_ca = self.eta * (psi_p + psi_m)
            else:
                W_ca = psi_p + psi_m
            term = W_ca * dx(TAG_CALCIUM)
            W_el = term if W_el is None else W_el + term

        x = ufl.SpatialCoordinate(msh)
        xc = x + self.u
        rho = ufl.sqrt(xc[0] ** 2 + xc[1] ** 2)
        gap_in, gap_cap = self._R_in - rho, rho - self._R_cap
        E_in = 0.5 * self._kpen * self._on_in * _pos(gap_in) ** 2 * ds(TAG_LUMEN_BND)
        E_cap = 0.5 * self._kpen * self._on_cap * _pos(gap_cap) ** 2 * ds(TAG_LUMEN_BND)
        E_f = 0.5 * self._kf * ufl.inner(self.u, self.u) * ds(TAG_OUTER_BND)
        Pi = W_el + E_in + E_cap + E_f
        N = ufl.FacetNormal(msh)
        cof = ufl.det(Fv) * ufl.inv(Fv).T
        v = ufl.TestFunction(V)
        F_u = ufl.derivative(Pi, self.u, v) + self._p * ufl.inner(cof * N, v) * ds(TAG_LUMEN_BND)
        J_u = ufl.derivative(F_u, self.u, ufl.TrialFunction(V))

        bcs = []
        if gdim == 3:
            end = ft.find(TAG_END_BND)
            if self.comm.allreduce(end.size) > 0:
                Vz = V.sub(2)
                bcs.append(fem.dirichletbc(0.0, fem.locate_dofs_topological(Vz, fdim, end), Vz))
        if bc_factory is not None:
            bcs += list(bc_factory(V))
        self.bcs = bcs
        uopts = dict(U_SNES_OPTIONS)
        uopts.update(self.settings.u_petsc_options)
        tag = uuid.uuid4().hex[:8]
        self._uprob = NonlinearProblem(F_u, self.u, bcs=bcs, J=J_u, petsc_options=uopts,
                                       petsc_options_prefix=f"{name}_u_{tag}_")
        self._dsolver = None
        if fracture == "phasefield" and self.has_calcium:
            self._dsolver = PF.DamageSolver(self.d, self.d_lb, self.d_ub, self.psi_plus, dx, self._Gc,
                                            self._ell, self._kres, soft_tags=self.soft_tags,
                                            petsc_options=self.settings.d_petsc_options,
                                            prefix=f"{name}_d_{tag}_")
        self._eroder = ER.ErosionUpdater(self) if (fracture == "erosion" and self.has_calcium) else None

        # --- geometry of the lumen boundary and axial ranges ------------------
        self._setup_lumen()
        if gdim == 3:
            zc = msh.geometry.x[:, 2]
            ca_nodes = mesh_nodes_of_cells(msh, self.ca_cells) if self.has_calcium else np.arange(zc.size)
            z0 = self.comm.allreduce(float(zc[ca_nodes].min()) if ca_nodes.size else np.inf, op=_mpi_min())
            z1 = self.comm.allreduce(float(zc[ca_nodes].max()) if ca_nodes.size else -np.inf, op=_mpi_max())
            self._z0.value, self._z1.value = z0, z1
            self.z_range = (z0, z1)
            lz = self._lumen_X[:, 2] if self._lumen_X.size else np.zeros(0)
            lo = self.comm.allreduce(float(lz.min()) if lz.size else np.inf, op=_mpi_min())
            hi = self.comm.allreduce(float(lz.max()) if lz.size else -np.inf, op=_mpi_max())
            self.axial_length = hi - lo if hi > lo else 1.0
            chi = ufl.conditional(ufl.And(ufl.ge(x[2], self._z0), ufl.le(x[2], self._z1)), 1.0, 0.0)
            xp = ufl.as_vector([xc[0], xc[1], 0.0])
            area_integrand = -0.5 * chi * ufl.inner(xp, cof * N)
            self._area_len = z1 - z0 if z1 > z0 else 1.0
        else:
            self.z_range = None
            self.axial_length = 1.0
            area_integrand = -0.5 * ufl.inner(xc, cof * N)
            self._area_len = 1.0

        forms = {
            "E_elastic": W_el,
            "E_contact_in": E_in, "E_contact_cap": E_cap, "E_foundation": E_f,
            "F_contact_in": self._kpen * self._on_in * _pos(gap_in) * ds(TAG_LUMEN_BND),
            "F_contact_cap": self._kpen * self._on_cap * _pos(gap_cap) * ds(TAG_LUMEN_BND),
            "lumen_area": area_integrand * ds(TAG_LUMEN_BND),
        }
        if self._dsolver is not None:
            forms["E_fracture"] = PF.fracture_energy_density(self.d, self._Gc, self._ell) * dx(TAG_CALCIUM)
        self._forms = {k: fem.form(f) for k, f in forms.items()}
        w0 = ufl.TestFunction(self.Q0)
        self._q0_blocks = self.Q0.dofmap.list[self.ca_cells, 0]
        self._cell_vol = fem.assemble_vector(fem.form(w0 * dx)).array[self._q0_blocks].copy()
        self._d_cell_form = fem.form(self.d * w0 * dx) if self._dsolver is not None else None
        #: reference length (2D, mm) or area (3D, mm^2) of the lumen boundary
        self.lumen_measure_ref = float(self.comm.allreduce(
            fem.assemble_scalar(fem.form(1.0 * ds(TAG_LUMEN_BND)))))
        self.history: list[dict] = []
        self.status = "ok"

    # ------------------------------------------------------------------ setup
    def _setup_lumen(self):
        """P2 nodes of the lumen boundary and, in 2D, its edges (a, b, midside)."""
        import basix
        from dolfinx import fem

        msh, fdim, tdim = self.mesh, self.mesh.topology.dim - 1, self.mesh.topology.dim
        lf = self.facet_tags.find(TAG_LUMEN_BND)
        self._lumen_facets = lf
        Xall = self.V.tabulate_dof_coordinates()[:, :self.gdim]
        nodes = fem.locate_dofs_topological(self._Vs, fdim, lf) if lf.size else np.zeros(0, np.int32)
        self._lumen_nodes = np.asarray(nodes, dtype=np.int32)
        self._lumen_X = Xall[self._lumen_nodes]
        self._lumen_edges = None
        if self.gdim == 2 and lf.size:
            f2c = msh.topology.connectivity(fdim, tdim)
            c2f = msh.topology.connectivity(tdim, fdim)
            layout = self._Vs.dofmap.dof_layout
            ref_edges = basix.topology(basix.CellType.triangle)[1]
            edges = []
            for f in lf:
                c = f2c.links(f)[0]
                loc = int(np.where(c2f.links(c) == f)[0][0])
                cd = self._Vs.dofmap.cell_dofs(c)
                va, vb = ref_edges[loc]
                a = cd[layout.entity_dofs(0, va)[0]]
                b = cd[layout.entity_dofs(0, vb)[0]]
                mids = layout.entity_dofs(1, loc)
                m = cd[mids[0]] if len(mids) else -1
                edges.append((a, b, m))
            self._lumen_edges = np.asarray(edges, dtype=np.int64)

    # ------------------------------------------------------------- load control
    def set_loads(self, *, p=_KEEP, R_in=_KEEP, R_cap=_KEEP) -> None:
        """Set the lumen pressure (MPa) and the cylinder/cap radii (mm; ``None`` = inactive)."""
        if p is not _KEEP:
            self._p.value = float(p)
            self.loads["p"] = float(p)
        if R_in is not _KEEP:
            self._on_in.value = 0.0 if R_in is None else 1.0
            if R_in is not None:
                self._R_in.value = float(R_in)
            self.loads["R_in"] = None if R_in is None else float(R_in)
        if R_cap is not _KEEP:
            self._on_cap.value = 0.0 if R_cap is None else 1.0
            if R_cap is not None:
                self._R_cap.value = float(R_cap)
            self.loads["R_cap"] = None if R_cap is None else float(R_cap)

    def snapshot(self) -> dict:
        """Copy of the solution fields and loads (for step rejection)."""
        return {"u": self.u.x.array.copy(), "d": self.d.x.array.copy(), "d_lb": self.d_lb.x.array.copy(),
                "eta": self.eta.x.array.copy(), "loads": dict(self.loads)}

    def restore(self, snap: dict) -> None:
        """Restore a :meth:`snapshot`."""
        self.u.x.array[:] = snap["u"]
        self.d.x.array[:] = snap["d"]
        self.d_lb.x.array[:] = snap["d_lb"]
        self.eta.x.array[:] = snap["eta"]
        self.set_loads(**snap["loads"])

    def commit(self) -> None:
        """Accept the current increment: the damage becomes the new lower bound."""
        self.d_lb.x.array[:] = self.d.x.array

    # ---------------------------------------------------------------- solving
    def _solve_u(self, info: dict) -> None:
        self._uprob.solve()
        snes = self._uprob.solver
        reason = int(snes.getConvergedReason())
        info["newton_its"] += int(snes.getIterationNumber())
        if reason <= 0 or not np.all(np.isfinite(self.u.x.array)):
            raise SolveError(f"displacement SNES failed (reason {reason})")

    def _solve_d(self, info: dict) -> None:
        ok, its, reason = self._dsolver.solve()
        info["d_its"] += its
        if not ok:
            raise SolveError(f"damage SNES failed (reason {reason})")

    def solve_step(self) -> dict:
        """Solve the current load state (staggered); raises :class:`SolveError`."""
        s = self.settings
        info = {"newton_its": 0, "d_its": 0, "stagger_its": 0, "stagger_err": 0.0,
                "stagger_converged": True, "n_new_eroded": 0}
        if self._dsolver is not None:
            err = 0.0
            for it in range(1, s.stagger_max_it + 1):
                self._solve_u(info)
                d_old = self.d.x.array.copy()
                self._solve_d(info)
                err = float(self.comm.allreduce(np.abs(self.d.x.array - d_old).max(initial=0.0), op=_mpi_max()))
                info["stagger_its"], info["stagger_err"] = it, err
                if err < s.stagger_tol:
                    break
            else:
                info["stagger_converged"] = False
                log.warning("alternate minimisation not converged after %d iterations (||dd||=%.2e, loads %s)",
                            s.stagger_max_it, err, self.loads)
            if err > 0.0:
                self._solve_u(info)
        elif self._eroder is not None:
            for rnd in range(1, s.erosion_max_rounds + 1):
                self._solve_u(info)
                new = self._eroder.find_new()
                n_new = int(self.comm.allreduce(int(new.size)))
                info["stagger_its"] = rnd
                info["n_new_eroded"] += n_new
                if n_new == 0:
                    break
                self._eroder.erode_with_continuation(new, lambda: self._solve_u(info))
            else:
                info["stagger_converged"] = False
                log.warning("erosion not converged after %d rounds (loads %s)", s.erosion_max_rounds, self.loads)
                self._solve_u(info)
        else:
            self._solve_u(info)
            info["stagger_its"] = 1
        return info

    # --------------------------------------------------------- post-processing
    def _scalar(self, key: str) -> float:
        from dolfinx import fem
        return float(self.comm.allreduce(fem.assemble_scalar(self._forms[key])))

    def lumen_points(self, deformed: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Reference and (deformed) positions (n, gdim) of the lumen P2 nodes."""
        X = self._lumen_X
        if not deformed:
            return X, X.copy()
        U = self.u.x.array.reshape(-1, self.gdim)[self._lumen_nodes]
        return X, X + U

    def min_lumen_radius(self) -> float:
        """Smallest distance of the deformed lumen boundary nodes from the axis (mm)."""
        _, x = self.lumen_points()
        r = np.hypot(x[:, 0], x[:, 1]) if x.size else np.array([np.inf])
        return float(self.comm.allreduce(float(r.min()), op=_mpi_min()))

    def lumen_loop(self, z: float | None = None) -> np.ndarray:
        """Ordered (CCW) deformed lumen boundary polyline (n, 2) incl. midside nodes.

        2D: walk along the lumen edges (the longest loop if several).
        3D: section of the deformed lumen surface by the reference plane
        ``Z = z`` (default: centre of the calcium range, else of the
        segment), ordered by polar angle (star-shaped lumen assumed); the
        straight-sided (vertex) geometry of the facets is used.
        """
        if self.gdim == 3:
            if z is None:
                z = 0.5 * (self.z_range[0] + self.z_range[1])
            return self._lumen_section(float(z))
        X, x = self.lumen_points()
        pos = {int(n): i for i, n in enumerate(self._lumen_nodes)}
        loops = _walk_loops(self._lumen_edges)
        best, best_area = None, -1.0
        for loop in loops:
            pts = x[[pos[n] for n in loop]]
            a = _shoelace(pts)
            if abs(a) > best_area:
                best, best_area = (pts if a > 0 else pts[::-1]), abs(a)
        return best

    def _vertex_positions(self) -> tuple[np.ndarray, np.ndarray]:
        """Reference and deformed positions of all geometry nodes (P1 interpolation of u)."""
        from dolfinx import fem

        if getattr(self, "_V1", None) is None:
            self._V1 = fem.functionspace(self.mesh, ("Lagrange", 1, (self.gdim,)))
            self._u1 = fem.Function(self._V1)
            nv = self.mesh.topology.dim + 1
            gdm = self.mesh.geometry.dofmaps[0][:, :nv]
            n2d = np.full(self.mesh.geometry.x.shape[0], -1, dtype=np.int64)
            n2d[gdm.ravel()] = self._V1.dofmap.list.ravel()
            self._node2dof = n2d
        self._u1.interpolate(self.u)
        X = self.mesh.geometry.x[:, :self.gdim]
        U = np.zeros_like(X)
        ok = self._node2dof >= 0
        U[ok] = self._u1.x.array.reshape(-1, self.gdim)[self._node2dof[ok]]
        return X, X + U

    def _lumen_section(self, z: float) -> np.ndarray:
        """3D: deformed lumen boundary points on the reference plane Z = z, ordered by angle.

        Edges of the lumen facets that cross the plane are cut by linear
        interpolation of the deformed vertex positions (vertices on the
        plane are taken as they are); duplicates are removed.
        """
        from dolfinx import mesh as dmesh

        X, x = self._vertex_positions()
        fn = dmesh.entities_to_geometry(self.mesh, 2, self._lumen_facets)[:, :3]
        tol = 1e-9 * max(1.0, abs(z))
        pts = []
        for i, j in ((0, 1), (1, 2), (2, 0)):
            a, b = fn[:, i], fn[:, j]
            za, zb = X[a, 2] - z, X[b, 2] - z
            cross = (za * zb < 0.0)
            t = za[cross] / (za[cross] - zb[cross])
            pts.append(x[a[cross], :2] + t[:, None] * (x[b[cross], :2] - x[a[cross], :2]))
        on = np.unique(fn[np.abs(X[fn, 2] - z) <= tol])
        pts.append(x[on, :2])
        P2d = np.vstack(pts)
        if P2d.size == 0:
            return P2d.reshape(0, 2)
        key = np.round(P2d / (1e-9 * max(1.0, float(np.abs(P2d).max()))))
        _, first = np.unique(key, axis=0, return_index=True)
        P2d = P2d[np.sort(first)]
        return P2d[np.argsort(np.arctan2(P2d[:, 1], P2d[:, 0]))]

    def cell_damage(self) -> np.ndarray:
        """Cell-averaged damage of the calcium cells (eroded indicator in erosion mode)."""
        from dolfinx import fem

        if self._d_cell_form is not None:
            vals = fem.assemble_vector(self._d_cell_form).array[self._q0_blocks]
            return vals / self._cell_vol
        if self._eroder is not None:
            return (~self._eroder.intact()).astype(float)
        return np.zeros(self.ca_cells.size)

    def damage_field(self):
        """The object to pass as ``d`` to :func:`dtivl.fracture.cracks.crack_analysis`."""
        if self.fracture == "erosion":
            return ER.eroded_indicator(self)
        return self.d

    def observables(self) -> dict:
        """Load-independent observables of the current state (see module docstring)."""
        out = {}
        Fin, Fcap = self._scalar("F_contact_in"), self._scalar("F_contact_cap")
        R_in, R_cap = self.loads["R_in"], self.loads["R_cap"]
        out["contact_force"] = Fin
        out["p_contact_mean"] = Fin / (2.0 * math.pi * R_in * self.axial_length) if R_in else 0.0
        out["p_cap_mean"] = Fcap / (2.0 * math.pi * R_cap * self.axial_length) if R_cap else 0.0
        _, x = self.lumen_points()
        rho = np.hypot(x[:, 0], x[:, 1]) if x.size else np.zeros(0)
        pmax = float(self.k_pen * np.maximum(R_in - rho, 0.0).max(initial=0.0)) if R_in else 0.0
        out["p_contact_max"] = float(self.comm.allreduce(pmax, op=_mpi_max()))
        out["lumen_area"] = self._scalar("lumen_area") / self._area_len
        dc = self.cell_damage()
        out["n_crack_cells"] = int(self.comm.allreduce(int((dc > self.d_crack).sum())))
        if self.fracture == "phasefield":
            out["max_d"] = float(self.comm.allreduce(self.d.x.array.max(initial=0.0), op=_mpi_max()))
        else:
            out["max_d"] = float(self.comm.allreduce(dc.max(initial=0.0), op=_mpi_max()))
        out["n_eroded"] = int(self.comm.allreduce(int((dc > 0.5).sum()))) if self.fracture == "erosion" else 0
        out["E_elastic"] = self._scalar("E_elastic")
        out["E_fracture"] = self._scalar("E_fracture") if "E_fracture" in self._forms else 0.0
        out["E_contact"] = self._scalar("E_contact_in") + self._scalar("E_contact_cap")
        out["E_foundation"] = self._scalar("E_foundation")
        return out

    def record(self, phase: str, load_name: str, load: float, info: dict, obs: dict | None = None,
               increment: float = 0.0) -> dict:
        """Append one step record to :attr:`history` and return it."""
        obs = self.observables() if obs is None else obs
        rec = {"phase": phase, "step": len(self.history), "load_name": load_name, "load": float(load),
               "increment": float(increment), "p": self.loads["p"], "R_in": self.loads["R_in"],
               "R_cap": self.loads["R_cap"], **obs,
               **{k: info.get(k) for k in ("stagger_its", "stagger_err", "stagger_converged", "newton_its",
                                           "d_its", "n_new_eroded")}}
        self.history.append(rec)
        return rec


def mesh_nodes_of_cells(msh, cells: np.ndarray) -> np.ndarray:
    """Geometry node indices of ``cells``."""
    gdm = msh.geometry.dofmaps[0]
    return np.unique(gdm[cells].ravel()) if cells.size else np.zeros(0, np.int64)


def _shoelace(xy: np.ndarray) -> float:
    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _walk_loops(edges: np.ndarray) -> list[list[int]]:
    """Closed node loops from (a, b, midside) edges; midside nodes are inserted."""
    if edges is None or len(edges) == 0:
        return []
    adj: dict[int, list[int]] = {}
    for i, (a, b, _) in enumerate(edges):
        adj.setdefault(int(a), []).append(i)
        adj.setdefault(int(b), []).append(i)
    used = np.zeros(len(edges), dtype=bool)
    loops = []
    for start in range(len(edges)):
        if used[start]:
            continue
        a0, b, m = (int(v) for v in edges[start])
        used[start] = True
        loop = [a0] + ([m] if m >= 0 else [])
        cur = b
        while cur != a0:
            nxt = [e for e in adj[cur] if not used[e]]
            if not nxt:
                break
            e = nxt[0]
            used[e] = True
            a, bb, mm = (int(v) for v in edges[e])
            loop.append(cur)
            if mm >= 0:
                loop.append(mm)
            cur = bb if a == cur else a
        loops.append(loop)
    return loops


# ---------------------------------------------------------------------------
# Load stepping
# ---------------------------------------------------------------------------
def _ramp(state: NonlinearState, *, phase: str, load_name: str, start: float, end: float, n_steps: int,
          apply: Callable[[float], None], accept: Callable[[dict], bool] | None = None) -> str:
    """Adaptive ramp of one load parameter from ``start`` to ``end``.

    ``apply(lam)`` sets the loads for parameter ``lam``. A converged step is
    rejected if ``accept(observables)`` is False. Returns ``"ok"``,
    ``"limited"`` (``accept`` kept rejecting down to the minimum increment)
    or ``"failed"`` (solver failure at the minimum increment). The state is
    left at the last accepted step.
    """
    n_steps = max(int(n_steps), 1)
    span = float(end) - float(start)
    tol = 1e-12 * max(1.0, abs(start), abs(end))
    if abs(span) <= tol:
        snap = state.snapshot()
        apply(end)
        try:
            info = state.solve_step()
        except SolveError as exc:
            log.warning("%s: %s", phase, exc)
            state.restore(snap)
            return "failed"
        obs = state.observables()
        if accept is not None and not accept(obs):
            state.restore(snap)
            return "limited"
        state.commit()
        state.record(phase, load_name, end, info, obs, 0.0)
        return "ok"
    nominal = span / n_steps
    min_inc = abs(nominal) / 2 ** state.settings.max_halvings
    lam, inc, limited = float(start), nominal, False
    while abs(end - lam) > tol:
        step = inc if abs(inc) < abs(end - lam) else end - lam
        lam_new = lam + step
        snap = state.snapshot()
        apply(lam_new)
        ok, why = True, ""
        try:
            info = state.solve_step()
            obs = state.observables()
            if accept is not None and not accept(obs):
                ok, why = False, "rejected"
        except SolveError as exc:
            ok, why = False, str(exc)
        if ok:
            state.commit()
            state.record(phase, load_name, lam_new, info, obs, step)
            lam = lam_new
            if not limited:
                inc = math.copysign(min(abs(inc) * STEP_GROWTH, abs(nominal)), nominal)
            continue
        state.restore(snap)
        apply(lam)
        limited = limited or why == "rejected"
        inc = inc / 2.0
        log.info("%s: step to %s=%.6g %s; increment halved to %.3g", phase, load_name, lam_new, why, inc)
        if abs(inc) < min_inc:
            if why == "rejected":
                return "limited"
            log.warning("%s: minimum increment reached at %s=%.6g (%s)", phase, load_name, lam, why)
            return "failed"
    return "ok"


def _n(n_steps, key):
    return DEFAULT_STEPS[key] if n_steps is None else int(n_steps)


# ---------------------------------------------------------------------------
# Load protocols
# ---------------------------------------------------------------------------
def reset_displacement(state: NonlinearState) -> None:
    """Set ``u = 0`` and remove all loads; ``d``, its lower bound and erosion are kept."""
    state.u.x.array[:] = 0.0
    state.set_loads(p=0.0, R_in=None, R_cap=None)


def run_pressure(state: NonlinearState, p_target: float, *, cap_radius: float | None = None,
                 n_steps: int | None = None, phase: str = "pressure") -> ProtocolResult:
    """Ramp the lumen pressure from its current value to ``p_target`` (MPa).

    ``cap_radius``: optional envelope cap (mm), active during the ramp;
    ``None`` removes any cap. The physiological load is
    ``params.get('loads.physiological_pressure')``.
    """
    i0 = len(state.history)
    state.set_loads(R_cap=cap_radius)
    status = _ramp(state, phase=phase, load_name="p", start=state.loads["p"], end=float(p_target),
                   n_steps=_n(n_steps, "pressure"), apply=lambda lam: state.set_loads(p=lam))
    state.status = status
    return ProtocolResult(state, state.history[i0:], status,
                          {"p_target": float(p_target), "p_reached": state.loads["p"], "cap_radius": cap_radius})


def run_ivl(state: NonlinearState, *, T: float, cap_radius: float | None = None,
            n_steps: tuple[int, int, int] | None = None, p_bal: float | None = None,
            p_peak: float | None = None, pulse_scale: float | None = None) -> ProtocolResult:
    """IVL protocol: balloon inflation, pulse peak, unloading (design.md section 5.4).

    Pressure ``0 -> p_bal`` then ``-> p_bal + T * p_peak * pulse_scale`` with
    the envelope cap ``R_ivl = geometry.ivl_balloon_diameter_mm / 2``, then
    unloading to 0 (cap removed at ``p = 0``). ``T`` is the dynamic
    amplification factor from the transient analysis (required). Defaults:
    ``p_bal = loads.ivl_balloon_pressure_atm``, ``p_peak =
    loads.ivl_peak_pressure_MPa``, ``pulse_scale = loads.ivl_pulse_scale``.
    ``n_steps`` = increments of the three ramps.
    """
    p_bal = P.get("loads.ivl_balloon_pressure_atm").value if p_bal is None else float(p_bal)
    p_peak = P.get("loads.ivl_peak_pressure_MPa").value if p_peak is None else float(p_peak)
    scale = P.get("loads.ivl_pulse_scale").value if pulse_scale is None else float(pulse_scale)
    R_cap = 0.5 * P.get("geometry.ivl_balloon_diameter_mm").value if cap_radius is None else float(cap_radius)
    p_top = p_bal + float(T) * p_peak * scale
    if n_steps is None:
        n_steps = (DEFAULT_STEPS["ivl_inflate"], DEFAULT_STEPS["ivl_pulse"], DEFAULT_STEPS["ivl_unload"])
    i0 = len(state.history)
    summary = {"T": float(T), "p_bal": p_bal, "p_peak": p_peak, "pulse_scale": scale, "p_top": p_top,
               "cap_radius": R_cap}
    state.set_loads(R_cap=R_cap)
    status = "ok"
    for phase, target, n in (("ivl_inflate", p_bal, n_steps[0]), ("ivl_pulse", p_top, n_steps[1]),
                             ("ivl_unload", 0.0, n_steps[2])):
        status = _ramp(state, phase=phase, load_name="p", start=state.loads["p"], end=target, n_steps=n,
                       apply=lambda lam: state.set_loads(p=lam))
        if status != "ok":
            summary["failed_phase"] = phase
            break
    else:
        status = _ramp(state, phase="ivl_release", load_name="p", start=0.0, end=0.0, n_steps=1,
                       apply=lambda lam: state.set_loads(p=lam, R_cap=None))
    state.status = status
    hist = state.history[i0:]
    summary["max_d"] = max((r["max_d"] for r in hist), default=0.0)
    summary["max_p_cap_mean"] = max((r["p_cap_mean"] for r in hist), default=0.0)
    return ProtocolResult(state, hist, status, summary)


def run_stent(state: NonlinearState, *, K_s: float | None = None, R_target: float | None = None,
              p_rated: float | None = None, n_steps: int | None = None, reset_u: bool = True,
              recoil_tol: float = RECOIL_TOL_MM, gap: float | None = None) -> ProtocolResult:
    """Stent deployment by an expanding rigid cylinder, then recoil (design.md section 5.4).

    The cylinder expands from ``(1 - STENT_START_GAP_FRAC) R_min`` (``R_min``
    = minimum lumen radius) to ``R_target`` (default
    ``geometry.stent_target_diameter_mm / 2``); a step whose mean contact
    pressure exceeds ``p_rated`` (default ``loads.balloon_rated_pressure_atm``)
    is rejected, so the expansion stops at the pressure-limited ``R_dep``
    (under-expansion). Recoil: ``R_final`` in ``[R_min, R_dep]`` with
    ``P_tissue(R_final) = K_s (R_dep - R_final)`` by bisection, ``P_tissue``
    being the mean contact pressure with the cylinder fixed at ``R``.
    ``K_s`` (MPa/mm) defaults to the placeholder
    ``loads.stent_radial_stiffness_MPa_per_mm`` and is meant to be
    identified by the integrator. Stent metrics
    (:func:`dtivl.metrics.outcomes.stent_metrics`) are in ``summary``.
    """
    from dtivl.metrics.outcomes import stent_metrics

    K_s = P.get("loads.stent_radial_stiffness_MPa_per_mm").value if K_s is None else float(K_s)
    R_target = 0.5 * P.get("geometry.stent_target_diameter_mm").value if R_target is None else float(R_target)
    p_rated = P.get("loads.balloon_rated_pressure_atm").value if p_rated is None else float(p_rated)
    if reset_u:
        reset_displacement(state)
    state.set_loads(p=0.0, R_cap=None)
    i0 = len(state.history)
    lumen_pre = state.lumen_loop()
    area_pre = state.observables()["lumen_area"]
    R_min = (1.0 - STENT_START_GAP_FRAC) * state.min_lumen_radius()
    summary = {"K_s": K_s, "R_target": R_target, "p_rated": p_rated, "R_min": R_min}

    def at(R):
        state.set_loads(R_in=R)

    status = _ramp(state, phase="stent_expand", load_name="R", start=R_min, end=R_min, n_steps=1, apply=at)
    if status == "ok" and R_target > R_min:
        status = _ramp(state, phase="stent_expand", load_name="R", start=R_min, end=R_target,
                       n_steps=_n(n_steps, "stent_expand"), apply=at,
                       accept=lambda obs: obs["p_contact_mean"] <= p_rated)
    R_dep = state.loads["R_in"] if state.loads["R_in"] is not None else R_min
    expand = [r for r in state.history[i0:] if r["phase"] == "stent_expand"]
    P_dep = expand[-1]["p_contact_mean"] if expand else 0.0
    reached = status == "ok" and abs(R_dep - R_target) <= 1e-9 * max(1.0, R_target)
    summary.update(R_dep=R_dep, P_dep=P_dep, reached_target=bool(reached), expand_status=status,
                   max_contact_pressure_mean=max((r["p_contact_mean"] for r in expand), default=0.0))

    # --- recoil by bisection on f(R) = P(R) - K_s (R_dep - R) -------------------
    R_final, recoil_status, n_bis = R_dep, "ok", 0
    if status != "failed" and P_dep > 0.0 and K_s > 0.0:
        lo, hi = R_min, R_dep
        while hi - lo > recoil_tol and n_bis < RECOIL_MAX_IT:
            n_bis += 1
            mid = 0.5 * (lo + hi)
            st = _ramp(state, phase="stent_recoil", load_name="R", start=state.loads["R_in"], end=mid,
                       n_steps=1, apply=at)
            if st != "ok":
                recoil_status = st
                break
            f_mid = state.history[-1]["p_contact_mean"] - K_s * (R_dep - mid)
            if f_mid > 0.0:
                hi = mid
            else:
                lo = mid
        if recoil_status == "ok":
            R_final = 0.5 * (lo + hi)
            recoil_status = _ramp(state, phase="stent_recoil", load_name="R", start=state.loads["R_in"],
                                  end=R_final, n_steps=1, apply=at)
        else:
            R_final = state.loads["R_in"]
    final_status = "failed" if "failed" in (status, recoil_status) else ("limited" if status == "limited" else "ok")
    state.status = final_status
    hist = state.history[i0:]
    P_final = hist[-1]["p_contact_mean"] if hist else 0.0
    lumen_post = state.lumen_loop()
    area_post = state.observables()["lumen_area"]
    summary.update(R_final=R_final, P_final=P_final, recoil_status=recoil_status, n_bisection=n_bis,
                   recoil_residual=P_final - K_s * (R_dep - R_final),
                   lumen_pre_xy=lumen_pre, lumen_post_xy=lumen_post)
    summary["metrics"] = stent_metrics(lumen_area_pre=area_pre, lumen_area_post=area_post,
                                       lumen_post_xy=lumen_post, R_dep=R_dep, R_final=R_final,
                                       reached_target=reached,
                                       max_contact_pressure_mean=summary["max_contact_pressure_mean"], gap=gap)
    return ProtocolResult(state, hist, final_status, summary)


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def _gmsh_model(name: str):
    """Fresh gmsh model; initialises/finalises gmsh only if not already active."""
    import gmsh

    owns = not gmsh.isInitialized()
    if owns:
        gmsh.initialize(interruptible=False)
    old = gmsh.option.getNumber("General.Terminal")
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
            gmsh.option.setNumber("General.Terminal", old)


def annulus_meshdata(a: float, b: float, h: float, *, tag: int = TAG_FIBROUS,
                     ca_sector: tuple[float, float, float] | None = None, h_ca: float | None = None,
                     ca_center_deg: float = 0.0, order: int = 1, comm=None):
    """Annulus ``a <= r <= b`` (triangles), optionally with a calcium sector.

    ``order = 2`` gives curved (quadratic) cells whose boundary nodes lie on
    the circles; the lesion meshes of the pipeline are linear (``order = 1``),
    for which the discrete lumen is a polygon.

    ``ca_sector = (r1, r2, arc_deg)``: annular sector ``r1 <= r <= r2`` of
    angular width ``arc_deg`` centred on ``ca_center_deg`` (square ends),
    tagged ``TAG_CALCIUM`` and meshed with size ``h_ca`` (default ``h``);
    the rest carries ``tag``. Facet tags ``TAG_LUMEN_BND`` (r = a) and
    ``TAG_OUTER_BND`` (r = b). Returns ``dolfinx.io.gmsh.MeshData``.
    """
    from mpi4py import MPI
    from dolfinx.io.gmsh import model_to_mesh

    comm = MPI.COMM_WORLD if comm is None else comm
    h_ca = h if h_ca is None else h_ca
    with _gmsh_model("annulus_nl") as gmsh:
        occ = gmsh.model.occ
        outer = occ.addDisk(0, 0, 0, b, b)
        inner = occ.addDisk(0, 0, 0, a, a)
        ring, _ = occ.cut([(2, outer)], [(2, inner)])
        if ca_sector is not None:
            r1, r2, arc = ca_sector
            c0 = math.radians(ca_center_deg)
            half = math.radians(arc) / 2.0
            ann, _ = occ.cut([(2, occ.addDisk(0, 0, 0, r2, r2))], [(2, occ.addDisk(0, 0, 0, r1, r1))])
            if arc < 360.0:
                n = max(8, int(arc / 10.0))
                R = 2.0 * b
                pts = [occ.addPoint(0, 0, 0)] + [
                    occ.addPoint(R * math.cos(c0 - half + 2 * half * k / n), R * math.sin(c0 - half + 2 * half * k / n), 0)
                    for k in range(n + 1)]
                lines = [occ.addLine(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]
                wedge = occ.addPlaneSurface([occ.addCurveLoop(lines)])
                ann, _ = occ.intersect(ann, [(2, wedge)])
            sector = ann
            _, children = occ.fragment(ring, sector)
            # children[i] lists the fragments of input i (object entities first, then tools)
            ca_set = {t for dimtags in children[len(ring):] for _, t in dimtags}
        else:
            ca_set = set()
        occ.synchronize()
        surfs = [s for _, s in gmsh.model.getEntities(2)]
        ca_s = [s for s in surfs if s in ca_set]
        soft_s = [s for s in surfs if s not in ca_set]
        if soft_s:
            gmsh.model.addPhysicalGroup(2, soft_s, tag)
        if ca_s:
            gmsh.model.addPhysicalGroup(2, ca_s, TAG_CALCIUM)
        lum, out = [], []
        for _, c in gmsh.model.getBoundary([(2, s) for s in surfs], oriented=False, combined=True):
            xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(1, abs(c))
            rc = max(abs(xmin), abs(xmax), abs(ymin), abs(ymax))
            (lum if abs(rc - a) < abs(rc - b) else out).append(abs(c))
        gmsh.model.addPhysicalGroup(1, lum, TAG_LUMEN_BND)
        gmsh.model.addPhysicalGroup(1, out, TAG_OUTER_BND)
        if ca_sector is not None and h_ca < h:
            r1, r2, arc = ca_sector
            band = 2.0 * h_ca

            def size_cb(dim, tg, x, y, z, lc):
                rr = math.hypot(x, y)
                dth = abs((math.degrees(math.atan2(y, x)) - ca_center_deg + 180.0) % 360.0 - 180.0)
                ang_ok = arc >= 360.0 or math.radians(dth - arc / 2.0) * rr <= band
                return h_ca if (r1 - band <= rr <= r2 + band and ang_ok) else h
            gmsh.model.mesh.setSizeCallback(size_cb)
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        else:
            gmsh.option.setNumber("Mesh.MeshSizeMin", h)
            gmsh.option.setNumber("Mesh.MeshSizeMax", h)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.model.mesh.generate(2)
        if order > 1:
            gmsh.model.mesh.setOrder(order)
        return model_to_mesh(gmsh.model, comm, 0, gdim=2)


def tube_meshdata(a: float, b: float, length: float, h: float, *, tag: int = TAG_FIBROUS,
                  ca_sector: tuple[float, float, float] | None = None, ca_center_deg: float = 0.0, comm=None):
    """3D tube ``a <= r <= b``, ``0 <= z <= length`` (linear tetrahedra, size ``h``).

    Optional calcium sector ``(r1, r2, arc_deg)`` over the full length.
    Facet tags ``TAG_LUMEN_BND``, ``TAG_OUTER_BND`` and ``TAG_END_BND``
    (z = 0 and z = length). Verification/test geometry.
    """
    from mpi4py import MPI
    from dolfinx.io.gmsh import model_to_mesh

    comm = MPI.COMM_WORLD if comm is None else comm
    with _gmsh_model("tube_nl") as gmsh:
        occ = gmsh.model.occ
        ring, _ = occ.cut([(3, occ.addCylinder(0, 0, 0, 0, 0, length, b))],
                          [(3, occ.addCylinder(0, 0, 0, 0, 0, length, a))])
        ca_set = set()
        if ca_sector is not None:
            r1, r2, arc = ca_sector
            sec, _ = occ.cut([(3, occ.addCylinder(0, 0, 0, 0, 0, length, r2))],
                             [(3, occ.addCylinder(0, 0, 0, 0, 0, length, r1))])
            if arc < 360.0:
                c0, half, n, R = math.radians(ca_center_deg), math.radians(arc) / 2.0, max(8, int(arc / 10.0)), 2.0 * b
                pts = [occ.addPoint(0, 0, 0)] + [
                    occ.addPoint(R * math.cos(c0 - half + 2 * half * k / n), R * math.sin(c0 - half + 2 * half * k / n), 0)
                    for k in range(n + 1)]
                lines = [occ.addLine(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]
                wedge = occ.extrude([(2, occ.addPlaneSurface([occ.addCurveLoop(lines)]))], 0, 0, length)
                sec, _ = occ.intersect(sec, [dt for dt in wedge if dt[0] == 3])
            _, children = occ.fragment(ring, sec)
            ca_set = {t for dimtags in children[len(ring):] for _, t in dimtags}
        occ.synchronize()
        vols = [v for _, v in gmsh.model.getEntities(3)]
        soft = [v for v in vols if v not in ca_set]
        if soft:
            gmsh.model.addPhysicalGroup(3, soft, tag)
        if ca_set:
            gmsh.model.addPhysicalGroup(3, sorted(ca_set), TAG_CALCIUM)
        lum, out, end = [], [], []
        for _, f in gmsh.model.getBoundary([(3, v) for v in vols], oriented=False, combined=True):
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, abs(f))
            if zmax - zmin < 1e-3 * length:          # flat (OCC boxes carry a tolerance)
                end.append(abs(f))
                continue
            rc = max(abs(xmin), abs(xmax), abs(ymin), abs(ymax))
            (lum if abs(rc - a) < abs(rc - b) else out).append(abs(f))
        gmsh.model.addPhysicalGroup(2, lum, TAG_LUMEN_BND)
        gmsh.model.addPhysicalGroup(2, out, TAG_OUTER_BND)
        gmsh.model.addPhysicalGroup(2, end, TAG_END_BND)
        gmsh.option.setNumber("Mesh.MeshSizeMin", h)
        gmsh.option.setNumber("Mesh.MeshSizeMax", h)
        gmsh.model.mesh.generate(3)
        return model_to_mesh(gmsh.model, comm, 0, gdim=3)


def lame_plane_strain_ur(r, a: float, b: float, p: float, lam: float, mu: float):
    """Small-strain plane-strain Lame radial displacement of a pressurised annulus."""
    r = np.asarray(r, dtype=float)
    A = p * a ** 2 / (b ** 2 - a ** 2)
    B = p * a ** 2 * b ** 2 / (b ** 2 - a ** 2)
    return A * r / (2.0 * (lam + mu)) + B / (2.0 * mu * r)


def _radial_u(state: NonlinearState, facet_tag: int) -> tuple[np.ndarray, np.ndarray]:
    """(|X|, u_r) at the P2 nodes of the facets with ``facet_tag`` (2D)."""
    from dolfinx import fem

    fdim = state.mesh.topology.dim - 1
    nodes = fem.locate_dofs_topological(state._Vs, fdim, state.facet_tags.find(facet_tag))
    X = state.V.tabulate_dof_coordinates()[nodes, :2]
    U = state.u.x.array.reshape(-1, state.gdim)[nodes]
    r = np.hypot(X[:, 0], X[:, 1])
    return r, np.einsum("ni,ni->n", U, X / r[:, None])


def hyperelastic_lame_check(ms: MaterialSet | None = None, *, tag: int = TAG_FIBROUS, a: float = VERIF_A,
                            b: float = VERIF_B, h: float = VERIF_H, p_rel: float = VERIF_P_REL) -> dict:
    """Small-pressure response of a single-material hyperelastic annulus vs Lame.

    The pressure is ``p_rel`` times the small-strain shear modulus of
    ``tag`` (one increment, follower load); the nodal radial displacement on
    the lumen and outer boundaries is compared with the plane-strain Lame
    solution for the small-strain limits ``ms.lame(tag)``.
    """
    ms = MaterialSet.baseline() if ms is None else ms
    lam, mu = ms.lame(tag)
    p = p_rel * mu
    state = NonlinearState(annulus_meshdata(a, b, h, tag=tag, order=VERIF_ORDER), ms, fracture="none",
                           name="verif_lame_nl")
    res = run_pressure(state, p, n_steps=1)
    errs = {}
    for name, ftag in (("lumen", TAG_LUMEN_BND), ("outer", TAG_OUTER_BND)):
        r, ur = _radial_u(state, ftag)
        ex = lame_plane_strain_ur(r, a, b, p, lam, mu)
        errs[name] = float(np.max(np.abs(ur - ex) / np.abs(ex)))
    out = {"material": MATERIAL_NAMES[tag], "lambda_MPa": lam, "mu_MPa": mu, "a_mm": a, "b_mm": b, "h_mm": h,
           "p_MPa": p, "status": res.status, "max_rel_err_ur_lumen": errs["lumen"],
           "max_rel_err_ur_outer": errs["outer"]}
    out["passed"] = bool(res.status == "ok" and max(errs.values()) < VERIF_LAME_TOL)
    return out


def contact_check(ms: MaterialSet | None = None, *, tag: int = TAG_FIBROUS, a: float = VERIF_A,
                  b: float = VERIF_B, h: float = VERIF_H, expansions=VERIF_CONTACT_EXP) -> dict:
    """Homogeneous tube expanded by the rigid cylinder to ``R = a (1 + e)``.

    For each ``R``: nodal penetration ``R - rho_i`` of the lumen nodes in
    contact vs the penalty penetration ``p_ref / k_pen`` of the mean contact
    pressure per reference area ``p_ref = F / |lumen|``, mean contact
    pressure (must be positive and increase with ``R``), and ``p_ref`` vs the
    small-strain Lame pressure that produces the same lumen displacement.
    """
    ms = MaterialSet.baseline() if ms is None else ms
    lam, mu = ms.lame(tag)
    state = NonlinearState(annulus_meshdata(a, b, h, tag=tag, order=VERIF_ORDER), ms, fracture="none",
                           name="verif_contact_nl")
    R0 = (1.0 - STENT_START_GAP_FRAC) * state.min_lumen_radius()
    state.set_loads(R_in=R0)
    lumen_len = state.lumen_measure_ref
    rows, status, R_prev = [], "ok", R0
    for e in expansions:
        R = a * (1.0 + e)
        status = _ramp(state, phase="contact", load_name="R", start=R_prev, end=R, n_steps=2,
                       apply=lambda lam_: state.set_loads(R_in=lam_))
        if status != "ok":
            break
        R_prev = R
        rec = state.history[-1]
        _, x = state.lumen_points()
        rho = np.hypot(x[:, 0], x[:, 1])
        pen = R - rho
        p_ref = rec["contact_force"] / lumen_len
        pen_ref = p_ref / state.k_pen
        in_contact = pen > 0.0
        ur_unit = lame_plane_strain_ur(a, a, b, 1.0, lam, mu)
        p_lame = (R - a - pen_ref) / ur_unit
        rows.append({"R_mm": R, "p_contact_mean_MPa": rec["p_contact_mean"], "p_ref_MPa": p_ref,
                     "penalty_penetration_mm": pen_ref, "frac_nodes_in_contact": float(in_contact.mean()),
                     "max_abs_R_minus_rho_mm": float(np.abs(pen).max()),
                     "max_dev_from_penalty_pen_mm": float(np.abs(pen - pen_ref).max()),
                     "p_lame_MPa": float(p_lame), "rel_err_p_vs_lame": float(abs(p_ref / p_lame - 1.0))})
    ok = status == "ok" and len(rows) == len(expansions)
    if ok:
        pm = [r["p_contact_mean_MPa"] for r in rows]
        ok = (pm[0] > 0 and all(np.diff(pm) > 0)
              and all(r["frac_nodes_in_contact"] == 1.0 for r in rows)
              and all(r["max_abs_R_minus_rho_mm"] <= (1 + VERIF_CONTACT_PEN_TOL) * r["penalty_penetration_mm"]
                      for r in rows)
              and all(r["rel_err_p_vs_lame"] < VERIF_CONTACT_LAME_TOL for r in rows))
    return {"material": MATERIAL_NAMES[tag], "a_mm": a, "b_mm": b, "h_mm": h, "k_pen": state.k_pen,
            "status": status, "rows": rows, "passed": bool(ok)}


def verification_nonlinear(out_json: str | Path | None = None, *, ms: MaterialSet | None = None) -> dict:
    """Hyperelastic Lame and rigid-cylinder contact checks; writes JSON.

    ``out_json`` defaults to ``results/verification_nonlinear.json``.
    """
    import dolfinx

    out_json = ROOT / "results" / "verification_nonlinear.json" if out_json is None else Path(out_json)
    t0 = time.time()
    lame = hyperelastic_lame_check(ms)
    contact = contact_check(ms)
    report = {"check": "hyperelastic thick cylinder vs Lame; rigid-cylinder penalty contact (design.md 5.7)",
              "dolfinx_version": dolfinx.__version__, "lame": lame, "contact": contact,
              "passed": bool(lame["passed"] and contact["passed"]), "wall_time_s": time.time() - t0}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2))
    return report

"""AT1 phase-field fracture confined to calcium (design.md sections 5.1-5.3, 5.7).

Model
-----
The damage field ``d`` is a scalar P1 function on the whole mesh, but it can
only grow inside calcium: the upper bound is 1 on every DOF of a calcium
cell and 0 elsewhere, and the lower bound is the damage of the last
converged load step (irreversibility). With ``u`` frozen, ``d`` minimises

    Pi_d(d) = int_Ca g(d) psi+(F) dx  +  (Gc / c_w) int_Ca (d / ell + ell |grad d|^2) dx
              + 1/2 eps_reg int_soft d^2 dx

with ``g(d) = (1 - d)^2 + k_res`` (:func:`dtivl.materials.constitutive.degradation`),
``c_w = 8/3`` (AT1) and ``psi+`` the tensile part of the volumetric-deviatoric
split (:func:`dtivl.materials.constitutive.psi_calcium_split`). The last term
is purely numerical: soft-tissue DOFs carry no energy, so their rows of the
reduced (inactive-set) Jacobian of ``vinewtonrsls`` would be empty and the LU
factorisation singular. ``eps_reg = D_REG_REL * Gc / (c_w ell)`` is so small
that its residual contribution at calcium-interface DOFs is negligible
(relative weight ``D_REG_REL``), and it vanishes identically on DOFs that are
bounded to zero. The problem is quadratic in ``d`` and is solved by PETSc
SNES ``vinewtonrsls`` (reduced-space active-set Newton) with MUMPS LU.

Calibration (AT1, homogeneous state): at ``d = 0`` the residual density is
``-2 psi+ + Gc / (c_w ell)``, so damage starts at
``psi+ = Gc / (2 c_w ell) = 3 Gc / (16 ell)``. With
``Gc = MaterialSet.gc(ell) = 16 ell psi_crit / 3`` the onset is at
``psi+ = psi_crit`` (``E eps_f^2 / 2`` in the primary strain-controlled mode),
independent of ``ell``. The regularisation length is
``ell = fracture.ell_over_h * h_ca`` (:func:`regularisation_length`).

Verification
------------
:func:`verification_phasefield` loads a homogeneous calcium plate (plane
strain) and a homogeneous calcium bar (3D) in uniaxial tension by
displacing one end (symmetry rollers on the other faces) and reports the
tensile energy and the applied strain at damage onset against
``3 Gc / (16 ell)`` and against the small-strain onset strain (``eps_f`` for
the 3D uniaxial-stress bar; ``eps_f sqrt(1 - nu^2)`` for the plane-strain
plate, whose uniaxial modulus is ``E / (1 - nu^2)``). The onset energy is
estimated from the first damaged step with the homogeneous AT1 relation
``1 - d = psi_c / psi+`` (exact for a homogeneous ``d`` at a converged
alternate-minimisation state) and bracketed by the last undamaged and the
first damaged step.
"""
from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path

import numpy as np
import ufl

from dtivl import ROOT
from dtivl import params as P
from dtivl.geometry.labels import TAG_CALCIUM
from dtivl.materials.constitutive import MaterialSet, degradation

log = logging.getLogger(__name__)

#: AT1 normalisation constant c_w = 4 int_0^1 sqrt(w(s)) ds with w(d) = d.
AT1_CW = 8.0 / 3.0
#: Relative weight of the numerical d-regularisation outside calcium.
D_REG_REL = 1e-6
#: PETSc options of the bound-constrained damage problem (numerical settings).
D_SNES_OPTIONS = {
    "snes_type": "vinewtonrsls",
    "snes_linesearch_type": "basic",
    "snes_rtol": 1e-9,
    "snes_atol": 1e-12,
    "snes_stol": 0.0,
    "snes_max_it": 50,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}

# --- verification settings (numerical test geometry, not literature) ---
VERIF_PLATE = (1.0, 0.25)            #: plate length x width (mm), plane strain
VERIF_BAR = (1.0, 0.25, 0.25)        #: bar length x width x height (mm), 3D
VERIF_CELLS_2D = (8, 2)              #: structured cells (homogeneous field: mesh-independent)
VERIF_CELLS_3D = (4, 1, 1)
VERIF_PRE_FRAC = 0.9                 #: coarse ramp up to 0.9 x predicted onset strain
VERIF_PRE_STEPS = 3
VERIF_FINE_STEP = 0.005              #: then steps of 0.5 % of the predicted onset strain
VERIF_MAX_FRAC = 1.3                 #: give up beyond 1.3 x predicted onset strain
VERIF_D_TOL = 1e-8                   #: "damaged" = max d above this value
VERIF_TOL = 0.05                     #: 5 % tolerance (design.md section 5.7)


# ---------------------------------------------------------------------------
# Parameters and energies
# ---------------------------------------------------------------------------
def regularisation_length(h_ca: float | None = None, ell_over_h: float | None = None) -> float:
    """``ell = fracture.ell_over_h * h_ca`` (mm).

    ``h_ca`` defaults to ``discretisation.h_calcium_mm`` and ``ell_over_h``
    to ``params.get('fracture.ell_over_h')``.
    """
    if h_ca is None:
        h_ca = float(P.raw("discretisation.h_calcium_mm"))
    if ell_over_h is None:
        ell_over_h = P.get("fracture.ell_over_h").value
    return float(ell_over_h) * float(h_ca)


def onset_energy(Gc: float, ell: float) -> float:
    """AT1 damage-onset tensile energy density ``3 Gc / (16 ell)`` (MPa)."""
    return 3.0 * Gc / (16.0 * ell)


def fracture_energy_density(d, Gc, ell):
    """AT1 crack-surface energy density ``Gc/c_w (d/ell + ell |grad d|^2)`` (UFL)."""
    return Gc / AT1_CW * (d / ell + ell * ufl.inner(ufl.grad(d), ufl.grad(d)))


def damage_upper_bound(D, cell_tags, ca_tag: int = TAG_CALCIUM):
    """P1 function equal to 1 on every DOF of a calcium cell and 0 elsewhere."""
    from dolfinx import fem

    ub = fem.Function(D, name="d_ub")
    cells = cell_tags.find(ca_tag)
    if cells.size:
        ub.x.array[np.unique(D.dofmap.list[cells].ravel())] = 1.0
    ub.x.scatter_forward()
    return ub


# ---------------------------------------------------------------------------
# Damage sub-problem
# ---------------------------------------------------------------------------
class DamageSolver:
    """Bound-constrained AT1 damage problem with ``u`` frozen.

    Parameters
    ----------
    d : P1 damage function (solution, updated in place).
    d_lb, d_ub : P1 functions with the lower (irreversibility) and upper
        (calcium confinement) bounds; they are passed by reference to SNES,
        so updating their arrays updates the bounds.
    psi_plus : UFL expression of the tensile energy density (depends on the
        frozen displacement function).
    dx : cell measure with the cell tags as subdomain data.
    Gc, ell, k_res : fracture toughness (N/mm), regularisation length (mm)
        and residual stiffness (UFL constants or floats).
    soft_tags : cell tags outside calcium (receive the numerical
        regularisation).
    """

    def __init__(self, d, d_lb, d_ub, psi_plus, dx, Gc, ell, k_res, *, soft_tags=(),
                 ca_tag: int = TAG_CALCIUM, petsc_options: dict | None = None, prefix: str | None = None):
        from dolfinx.fem.petsc import NonlinearProblem

        self.d, self.d_lb, self.d_ub = d, d_lb, d_ub
        energy = (degradation(d, k_res) * psi_plus + fracture_energy_density(d, Gc, ell)) * dx(ca_tag)
        eps_reg = D_REG_REL * Gc / (AT1_CW * ell)
        for t in soft_tags:
            energy = energy + 0.5 * eps_reg * d * d * dx(int(t))
        w = ufl.TestFunction(d.function_space)
        F = ufl.derivative(energy, d, w)
        opts = dict(D_SNES_OPTIONS)
        if petsc_options:
            opts.update(petsc_options)
        prefix = prefix or f"dtivl_d_{uuid.uuid4().hex[:8]}_"
        self.problem = NonlinearProblem(F, d, petsc_options_prefix=prefix, petsc_options=opts)
        self.problem.solver.setVariableBounds(d_lb.x.petsc_vec, d_ub.x.petsc_vec)

    def solve(self) -> tuple[bool, int, int]:
        """Solve in place; returns ``(converged, iterations, SNES reason)``."""
        # Keep the initial guess admissible (bounds may have moved).
        a = self.d.x.array
        np.clip(a, self.d_lb.x.array, self.d_ub.x.array, out=a)
        self.problem.solve()
        snes = self.problem.solver
        reason = int(snes.getConvergedReason())
        ok = reason > 0 and bool(np.all(np.isfinite(self.d.x.array)))
        return ok, int(snes.getIterationNumber()), reason


# ---------------------------------------------------------------------------
# Verification: homogeneous AT1 onset
# ---------------------------------------------------------------------------
def _box_meshdata(gdim: int):
    """All-calcium plate (2D) or bar (3D) as a ``MeshData`` (no facet tags)."""
    from mpi4py import MPI
    from dolfinx import mesh as dmesh
    from dolfinx.io.gmsh import MeshData

    if gdim == 2:
        msh = dmesh.create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], list(VERIF_PLATE)], VERIF_CELLS_2D)
    else:
        msh = dmesh.create_box(MPI.COMM_WORLD, [[0.0, 0.0, 0.0], list(VERIF_BAR)], VERIF_CELLS_3D)
    tdim = msh.topology.dim
    n = msh.topology.index_map(tdim).size_local
    ct = dmesh.meshtags(msh, tdim, np.arange(n, dtype=np.int32), np.full(n, TAG_CALCIUM, dtype=np.int32))
    msh.topology.create_connectivity(tdim - 1, tdim)
    ft = dmesh.meshtags(msh, tdim - 1, np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32))
    return MeshData(msh, ct, ft, None, None, {})


def _uniaxial_bcs(msh, length: float, disp_const):
    """Factory of the uniaxial-tension BCs: u_x = 0 / disp at x = 0 / L, rollers elsewhere."""
    from dolfinx import fem
    from dolfinx import mesh as dmesh

    gdim = msh.geometry.dim
    fdim = msh.topology.dim - 1

    def factory(V):
        def facets(axis, value):
            return dmesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[axis], value))
        bcs = [fem.dirichletbc(0.0, fem.locate_dofs_topological(V.sub(0), fdim, facets(0, 0.0)), V.sub(0)),
               fem.dirichletbc(disp_const, fem.locate_dofs_topological(V.sub(0), fdim, facets(0, length)), V.sub(0))]
        for axis in range(1, gdim):
            bcs.append(fem.dirichletbc(0.0, fem.locate_dofs_topological(V.sub(axis), fdim, facets(axis, 0.0)),
                                       V.sub(axis)))
        return bcs
    return factory


def onset_check(gdim: int = 2, ms: MaterialSet | None = None, *, h_ca: float | None = None) -> dict:
    """Homogeneous AT1 onset test in uniaxial tension (plane strain or 3D).

    Returns a dict with the predicted and measured onset tensile energy and
    strain, relative errors, the step bracket and ``passed`` (energy and
    strain both within :data:`VERIF_TOL`).
    """
    from dolfinx import fem

    from dtivl.solvers.nonlinear import NonlinearState

    ms = MaterialSet.baseline() if ms is None else ms
    md = _box_meshdata(gdim)
    msh = md.mesh
    length = (VERIF_PLATE if gdim == 2 else VERIF_BAR)[0]
    disp = fem.Constant(msh, 0.0)
    state = NonlinearState(md, ms, fracture="phasefield", h_ca=h_ca,
                           bc_factory=_uniaxial_bcs(msh, length, disp), name=f"verif_pf{gdim}d")
    psi_c = onset_energy(state.Gc_value, state.ell)
    if gdim == 2:
        eps_pred = ms.eps_f * np.sqrt(1.0 - ms.ca_nu ** 2) if ms.failure_mode == "strain" else \
            np.sqrt(2.0 * psi_c * (1.0 - ms.ca_nu ** 2) / ms.ca_E)
        state_desc = "plane strain, uniaxial in-plane stress (sigma_yy = 0, eps_zz = 0)"
    else:
        eps_pred = np.sqrt(2.0 * psi_c / ms.ca_E)
        state_desc = "3D uniaxial stress"
    psi_form = fem.form(state.psi_plus * state.dx(TAG_CALCIUM))
    vol = state.comm.allreduce(fem.assemble_scalar(fem.form(1.0 * state.dx(TAG_CALCIUM))))

    def step(eps):
        disp.value = eps * length
        state.solve_step()
        state.commit()
        psi = state.comm.allreduce(fem.assemble_scalar(psi_form)) / vol
        dmax = float(state.comm.allreduce(state.d.x.array.max(), op=_mpi_max()))
        dmean = float(state.comm.allreduce(state.d.x.array.sum()) / state.comm.allreduce(state.d.x.array.size))
        return {"strain": float(eps), "psi_plus": float(psi), "d_max": dmax, "d_mean": dmean}

    rows = []
    for k in range(1, VERIF_PRE_STEPS + 1):
        rows.append(step(VERIF_PRE_FRAC * eps_pred * k / VERIF_PRE_STEPS))
    k = 0
    while rows[-1]["d_max"] <= VERIF_D_TOL:
        k += 1
        eps = eps_pred * (VERIF_PRE_FRAC + VERIF_FINE_STEP * k)
        if eps > VERIF_MAX_FRAC * eps_pred:
            break
        rows.append(step(eps))
    first = next((i for i, r in enumerate(rows) if r["d_max"] > VERIF_D_TOL), None)
    out = {"case": f"{gdim}D", "state": state_desc, "failure_mode": ms.failure_mode,
           "E_MPa": ms.ca_E, "nu": ms.ca_nu, "eps_f": ms.eps_f, "ell_mm": state.ell,
           "Gc_N_per_mm": state.Gc_value, "psi_onset_theory_MPa": psi_c,
           "psi_crit_MPa": ms.psi_crit, "strain_onset_theory": float(eps_pred), "steps": rows}
    if first is None or first == 0:
        out.update(passed=False, error="no damage onset bracketed")
        return out
    lo, hi = rows[first - 1], rows[first]
    psi_est = hi["psi_plus"] * (1.0 - hi["d_mean"])            # homogeneous AT1: 1 - d = psi_c / psi+
    eps_est = hi["strain"] * np.sqrt(1.0 - hi["d_mean"])       # psi+ ~ strain^2 near onset
    out.update({
        "psi_bracket_MPa": [lo["psi_plus"], hi["psi_plus"]],
        "strain_bracket": [lo["strain"], hi["strain"]],
        "psi_onset_measured_MPa": float(psi_est),
        "strain_onset_measured": float(eps_est),
        "rel_err_psi": float(abs(psi_est / psi_c - 1.0)),
        "rel_err_strain": float(abs(eps_est / eps_pred - 1.0)),
        "bracket_contains_theory": bool(lo["psi_plus"] <= psi_c <= hi["psi_plus"]),
        "d_uniformity": float(hi["d_max"] - hi["d_mean"]),
    })
    out["passed"] = bool(out["rel_err_psi"] < VERIF_TOL and out["rel_err_strain"] < VERIF_TOL
                         and out["bracket_contains_theory"])
    return out


def _mpi_max():
    from mpi4py import MPI
    return MPI.MAX


def verification_phasefield(out_json: str | Path | None = None, *, ms: MaterialSet | None = None,
                            dims: tuple[int, ...] = (2, 3)) -> dict:
    """Run :func:`onset_check` for each dimension in ``dims`` and write JSON.

    ``out_json`` defaults to ``results/verification_phasefield.json``.
    Returns the written dict (``passed`` only if every case passes).
    """
    import dolfinx

    out_json = ROOT / "results" / "verification_phasefield.json" if out_json is None else Path(out_json)
    cases = [onset_check(g, ms) for g in dims]
    report = {
        "check": "AT1 damage onset in homogeneous uniaxial tension (design.md section 5.7)",
        "theory": "psi+ = 3 Gc / (16 ell) = psi_crit; onset strain from the small-strain uniaxial energy",
        "tolerance_rel": VERIF_TOL, "dolfinx_version": dolfinx.__version__,
        "cases": cases, "passed": all(c["passed"] for c in cases),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2))
    return report

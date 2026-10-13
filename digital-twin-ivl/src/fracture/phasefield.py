"""AT1 phase-field fracture confined to calcium (design.md sections 5.1-5.3, 5.7).

Model
-----
The damage field ``d`` is a scalar P1 function on the whole mesh, but it can
only grow inside calcium: the upper bound is 1 on every DOF of a calcium
cell and 0 elsewhere, and the lower bound is the damage of the last
converged load step (irreversibility). With ``u`` frozen, ``d`` minimises

    Pi_d(d) = int_Ca g(d) psi_d(F) dx  +  (Gc / c_w) int_Ca (d / ell + ell |grad d|^2) dx
              + 1/2 eps_reg int_soft d^2 dx

with ``g(d) = (1 - k)(1 - d)^2 + k``, ``k = MaterialSet.k_floor`` (:func:`dtivl.materials.constitutive.degradation`),
``c_w = 8/3`` (AT1) and ``psi_d`` the damage driving force (next section). The last term
is purely numerical: soft-tissue DOFs carry no energy, so their rows of the
reduced (inactive-set) Jacobian of ``vinewtonrsls`` would be empty and the LU
factorisation singular. ``eps_reg = D_REG_REL * Gc / (c_w ell)`` is so small
that its residual contribution at calcium-interface DOFs is negligible
(relative weight ``D_REG_REL``), and it vanishes identically on DOFs that are
bounded to zero. The problem is quadratic in ``d`` and is solved by PETSc
SNES ``vinewtonrsls`` (reduced-space active-set Newton) with MUMPS LU.

Damage driving force (``SolverSettings.driving_force``, default from the
config entry ``fracture.driving_force``; it enters the damage sub-problem
only, the displacement problem always minimises ``g(d) psi+ + psi-``):

* ``"energy"`` (default, the formulation used up to Checkpoint 2):
  ``psi_d = psi+``, the tensile part of the volumetric-deviatoric split
  (:func:`dtivl.materials.constitutive.psi_calcium_split`). The forms are
  exactly those of the code before the option existed.
* ``"rankine"`` (Checkpoint 2 decision D10, time-boxed test): a hybrid
  formulation, isotropic split for the elastic problem and a principal-strain
  driving force for damage,
  ``psi_d = psi_R = (E / 2) <e1>+^2``, with ``E`` the calcium Young modulus
  and ``e1`` the maximum principal Green-Lagrange strain of ``C = F^T F``
  (:func:`dtivl.materials.constitutive.max_principal_green_lagrange`: closed-form
  eigenvalue of the in-plane 2x2 block; the out-of-plane principal strain is
  zero in plane strain, so ``e1 = max(e1_in-plane, 0)``). ``psi_R`` does not
  depend on the volumetric or deviatoric character of the strain, only on
  the largest principal stretch, so damage is driven where the material is
  stretched most. The degradation stays isotropic (the volumetric-deviatoric
  ``psi+`` is degraded in the displacement problem), so there is no
  directional mechanism: a band perpendicular to the maximum principal strain
  can only arise if the damage localises. In the pilot it did not
  (``results/diag_rankine.json``): end states are diffuse with either force. Because
  ``psi_R`` is not the elastic energy that ``g(d)`` degrades, the coupled
  problem is no longer the minimisation of a single energy: the alternate
  minimisation then seeks a fixed point of the two sub-problems (each of
  which is still a minimisation). 2D plane strain only: 3D raises
  ``NotImplementedError``.

Calibration (AT1, homogeneous state): at ``d = 0`` the residual density is
``-2 (1 - k) psi_d + Gc / (c_w ell)``, so damage starts at
``psi_d = Gc / (2 c_w ell (1 - k)) = 3 Gc / (16 ell (1 - k))``. With
``Gc = MaterialSet.gc(ell) = 16 ell (1 - k) psi_crit / 3`` the onset is at
``psi_d = psi_crit`` (``E eps_f^2 / 2`` in the primary strain-controlled mode),
independent of ``ell``. For the energy force this is ``psi+ = psi_crit``;
for the Rankine force it is ``e1 = sqrt(2 psi_crit / E)``, i.e. ``e1 = eps_f``
in strain mode (``sigma_c / E`` in stress mode) in any stress state. ``e1`` is
a Green-Lagrange strain whereas ``eps_f`` is a failure stretch minus one;
at ``eps_f = 0.02`` the two measures differ by ``eps_f^2 / 2``, about 1 %
of ``eps_f`` (the erosion comparator uses the stretch, ``sqrt(lambda_max(C)) - 1``).
The regularisation length is ``ell = fracture.ell_over_h * h_ca`` (:func:`regularisation_length`).

Verification
------------
:func:`verification_phasefield` loads a homogeneous calcium plate (plane
strain) and a homogeneous calcium bar (3D) in uniaxial tension by
displacing one end (symmetry rollers on the other faces) and reports the
tensile energy and the applied strain at damage onset against
``3 Gc / (16 ell)`` and against the small-strain onset strain (``eps_f`` for
the 3D uniaxial-stress bar; ``eps_f sqrt(1 - nu^2)`` for the plane-strain
plate, whose uniaxial modulus is ``E / (1 - nu^2)``). The onset is bracketed
by the last undamaged and the first damaged step; ``rel_err_psi`` is the
larger relative distance of the two bracket ends from ``3 Gc / (16 ell)``
(an upper bound of the onset error, so it shrinks only with the load step).
The homogeneous AT1 relation ``psi+ (1 - d) = 3 Gc / (16 ell)`` at the first
damaged step is reported separately (``psi_equilibrium_MPa``): it holds for
any damaged homogeneous equilibrium and so checks the damage solver, not the
onset.

``onset_check(2, driving_force="rankine")`` repeats the plane-strain plate
test with the Rankine force: the energies are then ``psi_R`` and the onset is
also reported as the axial Green-Lagrange strain ``E_xx`` (the maximum
principal strain of the plate; theory ``sqrt(2 psi_crit / E)``, i.e.
``eps_f`` in strain mode) and as the cell-averaged ``e1``.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

import numpy as np
import ufl

from dtivl import ROOT
from dtivl import params as P
from dtivl.geometry.labels import TAG_CALCIUM
from dtivl.materials.constitutive import MaterialSet, degradation, max_principal_green_lagrange


#: AT1 normalisation constant c_w = 4 int_0^1 sqrt(w(s)) ds with w(d) = d.
AT1_CW = 8.0 / 3.0
#: Damage driving forces (``SolverSettings.driving_force``, config ``fracture.driving_force``).
DRIVING_FORCES = ("energy", "rankine")
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
VERIF_E1_REL_TOL = 1e-10             #: closed-form e1 vs E_xx in the homogeneous plate (rounding only)
VERIF_BRACKET_ROUND = 1e-12          #: relative rounding allowance of the bracket test (a step can land on the onset)


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


def onset_energy(Gc: float, ell: float, k: float = 0.0) -> float:
    """AT1 damage-onset tensile energy density ``3 Gc / (16 ell (1-k))`` (MPa)
    for the degradation ``g(d) = (1-k)(1-d)^2 + k``."""
    return 3.0 * Gc / (16.0 * ell * (1.0 - k))


def config_driving_force() -> str:
    """Damage driving force of the study configuration (``fracture.driving_force``)."""
    return str(P.raw("fracture.driving_force")["value"])


def check_driving_force(name: str, gdim: int | None = None) -> str:
    """Validate a driving-force name (:data:`DRIVING_FORCES`) and return it.

    ``ValueError`` for an unknown name; ``NotImplementedError`` for
    ``"rankine"`` when ``gdim == 3`` (the closed-form principal strain is
    implemented for 2D plane strain only).
    """
    if name not in DRIVING_FORCES:
        raise ValueError(f"unknown damage driving force {name!r}; expected one of {DRIVING_FORCES}")
    if name == "rankine" and gdim is not None and gdim != 2:
        raise NotImplementedError(
            f"driving_force='rankine' is implemented for 2D plane strain only (gdim={gdim}); the 3D maximum "
            "principal strain (3x3 eigenvalue) is not implemented. Use driving_force='energy' in 3D.")
    return name


def rankine_driving_force(F3, E, gdim: int):
    """Rankine-type driving force ``psi_R = (E / 2) <e1>+^2`` (UFL, 2D plane strain).

    ``e1`` is the maximum principal Green-Lagrange strain
    (:func:`dtivl.materials.constitutive.max_principal_green_lagrange`) and
    ``E`` the calcium Young modulus (float or ``fem.Constant``). In uniaxial
    tension ``psi_R`` equals the small-strain uniaxial-stress energy
    ``E e1^2 / 2``, so the AT1 calibration ``psi_R = psi_crit`` puts the onset
    at ``e1 = eps_f`` (strain mode).
    """
    check_driving_force("rankine", gdim)
    e1 = max_principal_green_lagrange(F3, gdim)
    e1p = ufl.conditional(ufl.gt(e1, 0.0), e1, 0.0)
    return 0.5 * E * e1p ** 2


def driving_force_density(name: str, *, psi_plus, F3=None, E=None, gdim: int | None = None):
    """UFL driving force ``psi_d`` of the damage sub-problem.

    ``"energy"``: returns ``psi_plus`` itself (the same UFL object, so the
    compiled damage forms are exactly those of the energy formulation);
    ``"rankine"``: :func:`rankine_driving_force` ``(F3, E, gdim)``.
    """
    check_driving_force(name, gdim)
    if name == "energy":
        return psi_plus
    return rankine_driving_force(F3, E, gdim)


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
    psi_plus : UFL expression of the damage driving force ``psi_d`` (the
        tensile energy density ``psi+`` or the Rankine force ``psi_R``, see
        :func:`driving_force_density`; it depends on the frozen displacement
        function).
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


def onset_check(gdim: int = 2, ms: MaterialSet | None = None, *, h_ca: float | None = None,
                driving_force: str | None = None) -> dict:
    """Homogeneous AT1 onset test in uniaxial tension (plane strain or 3D).

    Returns a dict with the predicted and measured onset driving-force
    energy and strain, relative errors, the step bracket and ``passed``
    (energy and strain both within :data:`VERIF_TOL`).

    ``driving_force`` (default: the config entry ``fracture.driving_force``):
    ``"energy"`` checks ``psi+ = psi_crit`` (keys ``psi_*`` refer to ``psi+``);
    ``"rankine"`` (2D only) checks ``psi_R = psi_crit`` (keys ``psi_*`` refer
    to ``psi_R``) and, in addition, the onset in terms of the axial
    Green-Lagrange strain ``E_xx`` (``gl_axial_*``; ``E_xx`` is the maximum
    principal strain of the plate, theory ``sqrt(2 psi_crit / E)``, i.e.
    ``eps_f`` in strain mode); ``e1_matches_gl_axial`` compares the
    closed-form ``e1`` with ``E_xx``.
    """
    from dolfinx import fem

    from dtivl.solvers.nonlinear import NonlinearState, SolverSettings

    ms = MaterialSet.baseline() if ms is None else ms
    settings = SolverSettings() if driving_force is None else SolverSettings(driving_force=driving_force)
    rankine = check_driving_force(settings.driving_force, gdim) == "rankine"
    md = _box_meshdata(gdim)
    msh = md.mesh
    length = (VERIF_PLATE if gdim == 2 else VERIF_BAR)[0]
    disp = fem.Constant(msh, 0.0)
    state = NonlinearState(md, ms, fracture="phasefield", h_ca=h_ca, settings=settings,
                           bc_factory=_uniaxial_bcs(msh, length, disp), name=f"verif_pf{gdim}d")
    psi_c = onset_energy(state.Gc_value, state.ell, state.ms.k_floor)
    if rankine:
        # psi_R = E e1^2 / 2 with e1 = E_xx = eps + eps^2 / 2 (eps = applied engineering strain)
        gl_pred = np.sqrt(2.0 * psi_c / ms.ca_E)
        eps_pred = np.sqrt(1.0 + 2.0 * gl_pred) - 1.0
    else:
        # small-strain uniaxial energy psi = E_u eps^2 / 2 (E_u = E / (1 - nu^2) in plane strain)
        E_u = ms.ca_E / (1.0 - ms.ca_nu ** 2) if gdim == 2 else ms.ca_E
        eps_pred = np.sqrt(2.0 * psi_c / E_u)
    state_desc = ("plane strain, uniaxial in-plane stress (sigma_yy = 0, eps_zz = 0)" if gdim == 2
                  else "3D uniaxial stress")
    key = "psi_R" if rankine else "psi_plus"
    dxc = state.dx(TAG_CALCIUM)
    forms = {key: fem.form(state.psi_drive * dxc)}
    if rankine:
        from dtivl.materials.constitutive import max_principal_green_lagrange

        forms["psi_plus"] = fem.form(state.psi_plus * dxc)
        forms["e1"] = fem.form(max_principal_green_lagrange(state.F3, gdim) * dxc)
        forms["gl_axial"] = fem.form(0.5 * ((state.F3.T * state.F3)[0, 0] - 1.0) * dxc)
    vol = state.comm.allreduce(fem.assemble_scalar(fem.form(1.0 * dxc)))

    def step(eps):
        disp.value = eps * length
        state.solve_step()
        state.commit()
        dmax = float(state.comm.allreduce(state.d.x.array.max(), op=_mpi_max()))
        dmean = float(state.comm.allreduce(state.d.x.array.sum()) / state.comm.allreduce(state.d.x.array.size))
        row = {"strain": float(eps)}
        row.update({k: float(state.comm.allreduce(fem.assemble_scalar(f)) / vol) for k, f in forms.items()})
        row.update(d_max=dmax, d_mean=dmean)
        return row

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
           "Gc_N_per_mm": state.Gc_value, "k_floor": ms.k_floor, "crack_filler": ms.crack_filler,
           "psi_onset_theory_MPa": psi_c,
           "psi_crit_MPa": ms.psi_crit, "strain_onset_theory": float(eps_pred), "steps": rows}
    if rankine:
        out = {"driving_force": "rankine", **out, "gl_axial_onset_theory": float(gl_pred)}
    if first is None or first == 0:
        out.update(passed=False, error="no damage onset bracketed")
        return out
    lo, hi = rows[first - 1], rows[first]
    # The observed onset lies in [lo, hi]; the error bound is the farther bracket end from theory.
    # (psi_d (1 - d) at the first damaged step equals 3 Gc / (16 ell (1 - k)) for ANY homogeneous damaged
    # equilibrium, wherever the onset happened, so it only checks the damage solver, not the onset.)
    psi_equil = hi[key] * (1.0 - hi["d_mean"])
    psi_est = 0.5 * (lo[key] + hi[key])
    eps_est = hi["strain"] * np.sqrt(1.0 - hi["d_mean"])       # psi_d ~ strain^2 near onset
    out.update({
        "psi_bracket_MPa": [lo[key], hi[key]],
        "strain_bracket": [lo["strain"], hi["strain"]],
        "psi_onset_measured_MPa": float(psi_est),
        "psi_equilibrium_MPa": float(psi_equil),
        "rel_err_psi_equilibrium": float(abs(psi_equil / psi_c - 1.0)),
        "strain_onset_measured": float(eps_est),
        "rel_err_psi": float(max(abs(lo[key] / psi_c - 1.0), abs(hi[key] / psi_c - 1.0))),
        "rel_err_strain": float(abs(eps_est / eps_pred - 1.0)),
        "bracket_contains_theory": _in_bracket(lo[key], psi_c, hi[key]),
        "d_uniformity": float(hi["d_max"] - hi["d_mean"]),
    })
    passed = out["rel_err_psi"] < VERIF_TOL and out["rel_err_strain"] < VERIF_TOL and out["bracket_contains_theory"]
    if rankine:
        # onset in terms of the axial Green-Lagrange strain: the bracket [lo, hi] contains the observed onset,
        # the error bound is the farther bracket end from theory (no use of the damaged-equilibrium relation)
        e1_dev = max(abs(r["e1"] / r["gl_axial"] - 1.0) for r in rows if r["gl_axial"] > 0.0)
        out.update({
            "gl_axial_bracket": [lo["gl_axial"], hi["gl_axial"]],
            "gl_axial_onset_measured": float(0.5 * (lo["gl_axial"] + hi["gl_axial"])),
            "rel_err_gl_axial": float(max(abs(lo["gl_axial"] / gl_pred - 1.0), abs(hi["gl_axial"] / gl_pred - 1.0))),
            "gl_axial_bracket_contains_theory": _in_bracket(lo["gl_axial"], gl_pred, hi["gl_axial"]),
            "e1_matches_gl_axial": float(e1_dev),
        })
        passed = (passed and out["rel_err_gl_axial"] < VERIF_TOL and out["gl_axial_bracket_contains_theory"]
                  and e1_dev < VERIF_E1_REL_TOL)
    out["passed"] = bool(passed)
    return out


def _in_bracket(lo: float, x: float, hi: float) -> bool:
    """``lo <= x <= hi`` up to the relative rounding allowance :data:`VERIF_BRACKET_ROUND`
    (a load step that lands on the onset leaves ``d = 0`` with the driving force equal to the
    onset value up to rounding)."""
    tol = VERIF_BRACKET_ROUND * abs(x)
    return bool(lo - tol <= x <= hi + tol)


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
    # this report verifies the default formulation (energy driving force) explicitly
    cases = [onset_check(g, ms, driving_force="energy") for g in dims]
    report = {
        "check": "AT1 damage onset in homogeneous uniaxial tension (design.md section 5.7)",
        "theory": "psi+ = 3 Gc / (16 ell) = psi_crit; onset strain from the small-strain uniaxial energy",
        "tolerance_rel": VERIF_TOL, "dolfinx_version": dolfinx.__version__,
        "cases": cases, "passed": all(c["passed"] for c in cases),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2))
    return report

"""Transient IVL pulse: linear elastodynamics in mm-MPa-us (design.md section 6).

Units and scaling
-----------------
Length mm, stress MPa (N/mm^2), time microseconds (us). In this system a
density in kg/m^3 has the same numerical value in N us^2/mm^4, and wave
speeds are in mm/us (= km/s); e.g. M = 2200 MPa, rho = 1000 gives
c = sqrt(M / rho) = 1.483 mm/us = 1483 m/s. Sound speeds read from the
config (m/s) are converted with :data:`M_PER_S_TO_MM_PER_US`; pulse widths
read from the literature table (seconds, base unit of :mod:`dtivl.params`)
with :data:`S_TO_US`.

Material model (per cell tag, piecewise constant)
-------------------------------------------------
* shear modulus mu from ``MaterialSet.lame(tag)`` (small-strain limit of the
  quasi-static constitutive model);
* longitudinal (P-wave) modulus from the measured sound speed and the
  density: ``M = rho c^2``, bulk modulus ``K = M - 4 mu / 3``,
  ``lambda = K - 2 mu / 3``; shear-wave speed ``c_s = sqrt(mu / rho)``.

Discretisation (near incompressibility)
---------------------------------------
The acoustic bulk moduli are 3-5 orders of magnitude above the shear moduli
(effective Poisson ratios > 0.499, also for calcium: K/mu ~ 2.4e3 calcium,
~8e4 wall). Two formulations are provided:

* ``formulation="mixed"`` (default): Taylor-Hood u-p (P2 displacement,
  P1 pressure ``p = -K div u``, positive in compression). The pressure is
  continuous inside each material and discontinuous across material
  interfaces (one P1 field per cell tag; ``interface_pressure``), because
  the exact pressure jumps where mu and K jump: a globally continuous
  pressure gave calcium cell errors up to ~30 % of the peak next to the
  interfaces of a layered ring. Locking-free: on the pressurised wall
  annulus (K/mu = 8.2e4) the cell-averaged sigma1 matches Lame within
  0.2 % (2D, h = 0.05 mm) and 0.6 % (3D tube with u_z = 0 ends,
  h = 0.2 mm); on a fibrous / calcium / fibrous / wall ring the calcium
  cell sigma1 matches the exact layered solution within 0.1 %.
* ``formulation="displacement"``: pure P2 displacement; ``volumetric="full"``
  integrates exactly (cleanest plane-wave propagation, but P2 elements lock
  at these ratios: 2D peak sigma1 within 0.4 % with isolated noisy cells up
  to ~5 %; 3D tetrahedra lock badly, peak error ~170 % at h = 0.2 mm), or
  ``volumetric="reduced"`` (one-point rule, i.e. P2-P0) which is
  locking-free in *static* solves but must not be used for transients: the
  element bubble (zero mean divergence) is then resisted by mu only and is
  strongly excited by the pulse.

Locking-free elements contain many discrete near-incompressible modes that
carry only shear stiffness; with shear-wave speeds of 0.006-0.04 mm/us the
physical shear wavelengths during a ~2 us pulse (0.01-0.08 mm) are not
resolved by practical meshes, and mode conversion at the calcium edges
excites these modes. The transient tensile sigma1 in calcium (small: about
0.2-0.3 MPa against ~4 MPa compression) is therefore mesh sensitive: in the
2D 90 deg sector test (mixed, per-material pressure, 20 us window)
T = 0.0142, 0.0166, 0.0153 for h_ca = 0.05, 0.025, 0.0125 mm, i.e. T is
resolved to about +-0.002 at the default calcium mesh size. A 3D extrusion
of the same section with u_z = 0 ends (plane strain) reproduces the 2D
value with the mixed formulation (h_ca = 0.08 mm, 10 us window:
T = 0.0139 in 3D vs 0.0150 in 2D); the 3D displacement formulation does
not (locking).

Weak form (plane strain in 2D, 3D otherwise; mixed)
---------------------------------------------------
    int rho a.v dx + int 2 mu dev(eps3(u)) : dev(eps3(v)) dx - int p div v dx
        + int_outer (rho c_p (n.v)(n.u_t) + rho c_s (v_t . u_t,t)) ds
        = - p_lumen(t) int_lumen n.v ds,
    int (div u + p / K) q dx = 0   for all q (piecewise P1, continuous per material)

(the displacement formulation replaces the pressure by
``+ int K div u div v dx``). ``eps3`` is the 3x3 strain (plane strain
embedding), ``n`` the outward normal of the tissue domain (on the lumen it
points into the lumen, so ``-p_lumen n`` pushes the wall outward) and the
outer boundary carries a Lysmer-Kuhlemeyer dashpot (normal ``rho c_p``,
tangential ``rho c_s``), which is exactly non-reflecting for plane waves at
normal incidence (``outer_bc="absorbing"``). ``outer_bc="free"``
(traction-free) and ``"fixed"`` (u = 0) give closed, energy-conserving
boundaries for verification. Optional roller constraints ``u_i = 0`` on
tagged facets (default in 3D: ``u_z = 0`` on ``TAG_END_BND``, as in the
linear module).

Load: lumen pressure half-sine ``p(t) = p_peak sin(pi t / tau)`` for
``0 <= t <= tau`` and 0 afterwards; default ``tau`` from
``lit('ivl_pulse_duration')`` (upper bound of "<2 us", see
:func:`default_pulse_width_us`) and ``p_peak`` from
``loads.ivl_peak_pressure_MPa`` (times ``loads.ivl_pulse_scale``).

Time integration
----------------
Generalized-alpha (Chung & Hulbert 1993) with spectral radius at infinity
``rho_inf``; the default ``rho_inf = 1`` is the non-dissipative average
acceleration (trapezoidal) member, which conserves the discrete energy
``1/2 v^T M v + 1/2 u^T Kd u + 1/2 p^T Mp p`` exactly without load and
damping. In the mixed formulation the algebraic pressure is solved for
together with the acceleration at the generalized-alpha level
``t_{n+1-alpha_f}``. The effective (saddle-point) matrix is factorised once
(MUMPS LU via PETSc).

Stress output
-------------
Cell averages of the 3x3 strain (and of the mixed pressure of the cell's own
material) are obtained by precomputed sparse operators (exact quadrature);
the cell stress is
``sigma3 = 2 mu dev(eps3) - p I`` with ``p`` the cell-averaged pressure
(mixed) or ``-K tr(eps)`` (displacement), i.e. the cell average of the
pointwise stress since the coefficients are cell-wise constant. In plane
strain ``sigma_33 = -2 mu tr(eps)/3 - p`` is included in the maximum
principal stress ``sigma1``.

Transfer factor
---------------
:func:`transfer_factor` runs the transient for one pulse, records the
history of max calcium sigma1, and divides its peak by the peak calcium
sigma1 of a *static* linear solve on the same mesh, with the same moduli,
discretisation and the same peak pressure:
``T = sigma1_peak,transient / sigma1_static``. The static solve has no
inertia and therefore no dashpot; the rigid-body modes are removed by the
weak foundation ``loads.perivascular_k_MPa_per_mm`` on the outer boundary
(as in design.md section 4). Additionally ``T_equivalent`` divides by the
static sigma1 of the quasi-static model (``MaterialSet.lame`` moduli,
displacement formulation with full integration, as in section 4), i.e. the factor by which the peak pulse
pressure must be scaled to reproduce the transient peak in the quasi-static
model.

Parallelism: serial only (scipy operators built from local data).
"""
from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from typing import Callable, Mapping

import numpy as np
import ufl

from dtivl import params as P
from dtivl.geometry.labels import (MATERIAL_NAMES, TAG_CALCIUM, TAG_END_BND, TAG_FIBROUS, TAG_LIPID,
                                   TAG_LUMEN_BND, TAG_OUTER_BND, TAG_WALL)
from dtivl.materials.constitutive import MaterialSet

# ---------------------------------------------------------------------------
# Unit conversions and numerical settings (not literature values)
# ---------------------------------------------------------------------------
#: m/s -> mm/us
M_PER_S_TO_MM_PER_US = 1.0e-3
#: kg/m^3 -> N us^2 / mm^4 (identical numbers)
KG_PER_M3_TO_N_US2_PER_MM4 = 1.0
#: seconds -> microseconds
S_TO_US = 1.0e6

#: default time steps per pulse width
STEPS_PER_PULSE = 40
#: default Courant number limit dt <= CFL * h_min / c_max (accuracy, not stability)
CFL = 1.0
#: default simulated window: WINDOW_PULSES pulse widths after the pulse start
WINDOW_PULSES = 10.0
#: default spectral radius at infinity (1 = average acceleration, energy conserving)
RHO_INF = 1.0
#: default displacement degree
DEGREE = 2

#: PETSc options of the direct solver (numerical setting).
LU_OPTIONS = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}

#: config keys of the per-tag acoustic inputs
_CFG_NAME = {TAG_WALL: "wall", TAG_FIBROUS: "fibrous", TAG_LIPID: "lipid", TAG_CALCIUM: "calcium"}


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def default_densities(*, allow_pending: bool = False) -> dict[int, float]:
    """Per-tag densities (N us^2/mm^4 = kg/m^3) from ``materials.density.*``.

    Raises ``ValueError`` if an entry is PENDING and ``allow_pending`` is
    False (use ``allow_pending=True`` only in tests / explicit opt-in).
    """
    return {tag: P.get(f"materials.density.{name}", allow_pending=allow_pending).value
            * KG_PER_M3_TO_N_US2_PER_MM4 for tag, name in _CFG_NAME.items()}


def default_sound_speeds(*, allow_pending: bool = False) -> dict[int, float]:
    """Per-tag longitudinal sound speeds (mm/us) from ``materials.sound_speed_m_per_s.*``."""
    return {tag: P.get(f"materials.sound_speed_m_per_s.{name}", allow_pending=allow_pending).value
            * M_PER_S_TO_MM_PER_US for tag, name in _CFG_NAME.items()}


def default_pulse_width_us() -> float:
    """Half-sine pulse width (us): upper bound of ``lit('ivl_pulse_duration')``.

    The literature row reads "<2 us" (VERIFIED, Brinton 2019) with a
    secondary "2-5 us" range in the notes; the status parser therefore
    labels the whole row SECONDARY. Only the verified "<2" bound is used
    here (the ``upper`` transform of the "<2" value); the row is accepted
    with status SECONDARY but no lower status.
    """
    row = P.load_literature()["ivl_pulse_duration"]
    if row.status not in P.MAIN_ANALYSIS_OK | {"SECONDARY"}:
        raise ValueError(f"ivl_pulse_duration has status {row.status}")
    q = P.lit("ivl_pulse_duration", "upper", allow_unverified=row.status == "SECONDARY")
    return q.value * S_TO_US


def default_peak_pressure(*, scaled: bool = True) -> float:
    """Peak pulse pressure (MPa): ``loads.ivl_peak_pressure_MPa`` (x ``loads.ivl_pulse_scale``)."""
    p = P.get("loads.ivl_peak_pressure_MPa").value
    if scaled:
        p *= P.get("loads.ivl_pulse_scale").value
    return p


def half_sine(p_peak: float, width: float) -> Callable[[float], float]:
    """Half-sine pulse ``p(t) = p_peak sin(pi t/width)`` on ``[0, width]``, 0 elsewhere."""
    if not width > 0:
        raise ValueError("pulse width must be positive")

    def p(t: float) -> float:
        return p_peak * math.sin(math.pi * t / width) if 0.0 <= t <= width else 0.0

    return p


@dataclass(frozen=True)
class AcousticMaterial:
    """Small-strain elastodynamic constants of one cell tag (mm-MPa-us)."""

    rho: float      # density (N us^2/mm^4)
    c_p: float      # longitudinal wave speed (mm/us)
    mu: float       # shear modulus (MPa)
    K: float        # bulk modulus (MPa)

    @property
    def M(self) -> float:
        """P-wave modulus (MPa)."""
        return self.K + 4.0 * self.mu / 3.0

    @property
    def lam(self) -> float:
        return self.K - 2.0 * self.mu / 3.0

    @property
    def c_s(self) -> float:
        """Shear-wave speed (mm/us)."""
        return math.sqrt(self.mu / self.rho)

    @property
    def nu(self) -> float:
        """Effective Poisson ratio."""
        return self.lam / (2.0 * (self.lam + self.mu))


def acoustic_materials(ms: MaterialSet, densities: Mapping[int, float], sound_speeds: Mapping[int, float],
                       tags) -> dict[int, AcousticMaterial]:
    """Per-tag :class:`AcousticMaterial` (mu from ``ms.lame``, ``K = rho c^2 - 4 mu/3``).

    ``densities`` in N us^2/mm^4 (= kg/m^3), ``sound_speeds`` in mm/us.
    Raises if a tag lacks an input or if ``rho c^2 <= 4 mu / 3`` (K <= 0).
    """
    out = {}
    for tag in tags:
        tag = int(tag)
        if tag not in densities or tag not in sound_speeds:
            raise ValueError(f"no density / sound speed for cell tag {tag} ({MATERIAL_NAMES.get(tag)})")
        rho, c = float(densities[tag]), float(sound_speeds[tag])
        if not (rho > 0 and c > 0):
            raise ValueError(f"density and sound speed must be positive (tag {tag})")
        _, mu = ms.lame(tag)
        K = rho * c * c - 4.0 * mu / 3.0
        if not K > 0:
            raise ValueError(f"rho c^2 <= 4 mu/3 for tag {tag}: P-wave speed below the shear limit")
        out[tag] = AcousticMaterial(rho=rho, c_p=c, mu=mu, K=K)
    return out


# ---------------------------------------------------------------------------
# Model (assembled operators)
# ---------------------------------------------------------------------------
def _dg0_from_tags(mesh, ct, values: Mapping[int, float], name: str):
    from dolfinx import fem

    Q0 = fem.functionspace(mesh, ("DG", 0))
    f = fem.Function(Q0, name=name)
    n_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    assigned = np.zeros(n_cells, dtype=bool)
    for tag, val in values.items():
        cells = ct.find(int(tag))
        f.x.array[Q0.dofmap.list[cells, 0]] = val
        assigned[cells[cells < n_cells]] = True
    if not assigned.all():
        raise ValueError(f"{int((~assigned).sum())} cells carry no known material tag")
    return f


def _eps3(u):
    e = ufl.sym(ufl.grad(u))
    if u.ufl_shape[0] == 3:
        return e
    return ufl.as_matrix([[e[0, 0], e[0, 1], 0], [e[1, 0], e[1, 1], 0], [0, 0, 0]])


def _dev(T):
    return T - ufl.tr(T) / 3.0 * ufl.Identity(3)


def _to_scipy(form):
    """Assemble a bilinear form into a scipy CSR matrix (expanded, serial)."""
    import scipy.sparse as sp
    from dolfinx.fem.petsc import assemble_matrix

    A = assemble_matrix(form)
    A.assemble()
    indptr, indices, data = A.getValuesCSR()
    out = sp.csr_matrix((data.copy(), indices.copy(), indptr.copy()), shape=A.getSize())
    A.destroy()
    return out


class ElastodynamicModel:
    """Assembled linear elastodynamic operators on a tagged mesh.

    Parameters
    ----------
    meshdata : object with ``mesh``, ``cell_tags``, ``facet_tags``
        (``dolfinx.io.gmsh.MeshData`` or equivalent). Lumen pressure acts on
        ``lumen_tag`` facets, the outer boundary condition on ``outer_tag``.
    materials : ``{cell_tag: AcousticMaterial}`` covering every cell tag.
    degree : displacement degree (vector Lagrange); the mixed pressure has
        degree ``degree - 1`` (continuous Lagrange).
    formulation : ``"mixed"`` (Taylor-Hood u-p, default) or
        ``"displacement"`` (pure displacement).
    volumetric : displacement formulation only: ``"full"`` (exact
        integration) or ``"reduced"`` (one-point rule for the volumetric
        term, P2-P0 equivalent; static use only, see the module docstring).
    outer_bc : ``"absorbing"`` (dashpot), ``"free"`` or ``"fixed"``.
    roller_tags : ``{facet_tag: component}`` constraints ``u_component = 0``;
        default ``{TAG_END_BND: 2}`` in 3D and none in 2D.
    interface_pressure : mixed formulation only. ``"discontinuous"``
        (default): one continuous pressure field per cell tag, free to jump
        across material interfaces (as the exact pressure does where mu and
        K jump). ``"continuous"``: a single globally continuous pressure;
        kept for verification only, because it smears the interface jump
        and pollutes the stresses of the cells next to every interface
        (calcium cell errors up to ~30 % of the peak on a layered ring,
        not decreasing with refinement).

    Attributes
    ----------
    M, Kd : mass and deviatoric stiffness (scipy CSR, expanded u DOFs).
    Kv : displacement formulation: volumetric stiffness (else ``None``).
    G, Mp : mixed formulation: ``G[i, j] = int q_j div v_i``,
        ``Mp[i, j] = int q_i q_j / K`` (else ``None``), in the extended
        pressure numbering: the per-tag blocks of P_{k-1} DOFs (in the order
        of the sorted cell tags) are concatenated, so a DOF on a material
        interface appears once per adjacent material. ``pressure`` arrays
        (``static_solution``, ``TransientHistory.p_final``) use this
        numbering.
    C : dashpot matrix (``None`` unless absorbing).
    f_lumen : load vector per unit lumen pressure.
    n_u, n_p : number of displacement / pressure unknowns (``n_p = 0`` for
        the displacement formulation).
    """

    def __init__(self, meshdata, materials: Mapping[int, AcousticMaterial], *, degree: int = DEGREE,
                 formulation: str = "mixed", volumetric: str = "full", outer_bc: str = "absorbing",
                 roller_tags: Mapping[int, int] | None = None,
                 lumen_tag: int = TAG_LUMEN_BND, outer_tag: int = TAG_OUTER_BND,
                 interface_pressure: str = "discontinuous"):
        from dolfinx import fem, la

        if formulation not in ("mixed", "displacement"):
            raise ValueError(f"formulation must be 'mixed' or 'displacement', got {formulation!r}")
        if interface_pressure not in ("discontinuous", "continuous"):
            raise ValueError(f"interface_pressure must be 'discontinuous' or 'continuous', got {interface_pressure!r}")
        self.interface_pressure = interface_pressure
        if volumetric not in ("reduced", "full"):
            raise ValueError(f"volumetric must be 'reduced' or 'full', got {volumetric!r}")
        if outer_bc not in ("absorbing", "free", "fixed"):
            raise ValueError(f"unknown outer_bc {outer_bc!r}")
        if formulation == "mixed" and degree < 2:
            raise ValueError("the Taylor-Hood formulation needs degree >= 2")
        self.meshdata = meshdata
        mesh, ct, ft = meshdata.mesh, meshdata.cell_tags, meshdata.facet_tags
        if mesh.comm.size != 1:
            raise NotImplementedError("transient solver is serial only")
        if ft is None:
            raise ValueError("meshdata has no facet tags")
        gdim, tdim = mesh.geometry.dim, mesh.topology.dim
        if gdim != tdim:
            raise ValueError("expects gdim == tdim (2D plane strain or 3D)")
        self.mesh, self.gdim, self.degree = mesh, gdim, degree
        self.formulation, self.volumetric, self.outer_bc = formulation, volumetric, outer_bc
        tags_present = [int(t) for t in np.unique(ct.values)]
        missing = [t for t in tags_present if t not in materials]
        if missing:
            raise ValueError(f"no material for cell tags {missing}")
        self.materials = {t: materials[t] for t in tags_present}
        mesh.topology.create_connectivity(tdim - 1, tdim)

        def dg0(attr, name):
            return _dg0_from_tags(mesh, ct, {t: attr(m) for t, m in self.materials.items()}, name)

        rho, mu, K = dg0(lambda m: m.rho, "rho"), dg0(lambda m: m.mu, "mu"), dg0(lambda m: m.K, "K")
        self._coef = {"rho": rho, "mu": mu, "K": K}

        V = fem.functionspace(mesh, ("Lagrange", degree, (gdim,)))
        self.V = V
        u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
        ds = ufl.Measure("ds", domain=mesh, subdomain_data=ft)
        n = ufl.FacetNormal(mesh)
        self._ds, self._outer_tag = ds, outer_tag

        self.M = _to_scipy(fem.form(rho * ufl.inner(u, v) * ufl.dx))
        self.Kd = _to_scipy(fem.form(2.0 * mu * ufl.inner(_dev(_eps3(u)), _dev(_eps3(v))) * ufl.dx))
        self.Kv = self.G = self.Mp = None
        self.Q = None
        self._p_blocks = None
        if formulation == "displacement":
            dx_vol = ufl.dx(metadata={"quadrature_degree": 0}) if volumetric == "reduced" else ufl.dx
            self.Kv = _to_scipy(fem.form(K * ufl.div(u) * ufl.div(v) * dx_vol))
        else:
            import scipy.sparse as sp

            Q = fem.functionspace(mesh, ("Lagrange", degree - 1))
            self.Q = Q
            p, q = ufl.TrialFunction(Q), ufl.TestFunction(Q)
            if interface_pressure == "continuous":
                self.G = _to_scipy(fem.form(ufl.div(v) * p * ufl.dx))
                self.Mp = _to_scipy(fem.form(p * q / K * ufl.dx))
                self._p_blocks = [(None, np.arange(Q.dofmap.index_map.size_local), 0)]
            else:
                # per-material pressure: the global hat functions restricted to the cells of one tag
                # are exactly the P_{k-1} basis of that subdomain, so the extended pressure vector
                # (one block per tag) is continuous inside each material and free to jump across
                # material interfaces, where -p = sigma_nn - 2 mu dev(eps)_nn jumps with mu and K.
                dxt = ufl.Measure("dx", domain=mesh, subdomain_data=ct)
                G_blocks, Mp_blocks, self._p_blocks, off = [], [], [], 0
                for t in tags_present:
                    dofs = np.unique(Q.dofmap.list[ct.find(t)].ravel())
                    G_blocks.append(_to_scipy(fem.form(ufl.div(v) * p * dxt(t)))[:, dofs])
                    Mp_blocks.append(_to_scipy(fem.form(p * q / K * dxt(t)))[dofs][:, dofs])
                    self._p_blocks.append((t, dofs, off))
                    off += dofs.size
                self.G = sp.hstack(G_blocks, format="csr")
                self.Mp = sp.block_diag(Mp_blocks, format="csr")
        if outer_bc == "absorbing":
            zc_p, zc_s = dg0(lambda m: m.rho * m.c_p, "rho_cp"), dg0(lambda m: m.rho * m.c_s, "rho_cs")
            un, vn = ufl.dot(u, n), ufl.dot(v, n)
            self.C = _to_scipy(fem.form((zc_p * un * vn + zc_s * ufl.inner(u - un * n, v - vn * n))
                                        * ds(outer_tag)))
        else:
            self.C = None
        f = fem.assemble_vector(fem.form(-ufl.inner(n, v) * ds(lumen_tag)))
        f.scatter_reverse(la.InsertMode.add)
        self.f_lumen = f.array.copy()          # load vector per unit lumen pressure
        if not np.any(self.f_lumen):
            raise ValueError(f"facet tag {lumen_tag} (lumen) is empty")
        self.n_u = self.M.shape[0]
        self.n_p = 0 if self.G is None else self.G.shape[1]

        # constrained displacement DOFs (expanded numbering)
        if roller_tags is None:
            roller_tags = {TAG_END_BND: 2} if gdim == 3 else {}
        bs = V.dofmap.index_map_bs
        fixed = []
        for ftag, comp in roller_tags.items():
            facets = ft.find(int(ftag))
            if facets.size:
                fixed.append(fem.locate_dofs_topological(V, tdim - 1, facets) * bs + int(comp))
        if outer_bc == "fixed":
            blocks = fem.locate_dofs_topological(V, tdim - 1, ft.find(outer_tag))
            fixed.append((blocks[:, None] * bs + np.arange(bs)[None, :]).ravel())
        self.fixed_dofs = np.unique(np.concatenate(fixed)) if fixed else np.zeros(0, dtype=np.int64)

        cells = np.arange(mesh.topology.index_map(tdim).size_local, dtype=np.int32)
        self.h_min = float(np.min(mesh.h(tdim, cells)))
        self.c_max = max(m.c_p for m in self.materials.values())
        self._cell_ops: dict = {}

    @property
    def ndofs(self) -> int:
        """Total number of unknowns (displacement + pressure)."""
        return self.n_u + self.n_p

    # --- energies -------------------------------------------------------
    def kinetic_energy(self, vel: np.ndarray) -> float:
        """``1/2 v^T M v`` (energy per unit thickness in 2D; N mm = mJ)."""
        return 0.5 * float(vel @ (self.M @ vel))

    def strain_energy(self, disp: np.ndarray, pres: np.ndarray | None = None) -> float:
        """``1/2 u^T Kd u + 1/2 p^T Mp p`` (mixed) or ``1/2 u^T (Kd + Kv) u``."""
        e = 0.5 * float(disp @ (self.Kd @ disp))
        if self.Kv is not None:
            e += 0.5 * float(disp @ (self.Kv @ disp))
        elif pres is not None:
            e += 0.5 * float(pres @ (self.Mp @ pres))
        return e

    def dof_coordinates(self) -> np.ndarray:
        """(n_u, gdim) coordinates of the expanded displacement DOFs (component i at bs*node+i)."""
        x = self.V.tabulate_dof_coordinates()[:, :self.gdim]
        return np.repeat(x, self.V.dofmap.index_map_bs, axis=0)

    # --- linear algebra -------------------------------------------------
    def _system(self, A_uu, c_p: float):
        """Full system matrix: ``A_uu`` (displacement) or ``[[A_uu, -G], [-G^T, -Mp/c_p]]``."""
        import scipy.sparse as sp

        if self.n_p == 0:
            return A_uu.tocsr()
        return sp.bmat([[A_uu, -self.G], [-self.G.T, -self.Mp / c_p]], format="csr")

    def _factorise(self, A):
        """Return a ``solve(b) -> x`` closure (u constraints applied, MUMPS LU)."""
        from petsc4py import PETSc

        import scipy.sparse as sp

        fixed = self.fixed_dofs
        if fixed.size:   # A <- D A D + I_fixed with D = diag(1 - fixed indicator)
            ind = np.zeros(A.shape[0])
            ind[fixed] = 1.0
            D = sp.diags(1.0 - ind)
            A = D @ A @ D + sp.diags(ind)
        A = A.tocsr()
        A.eliminate_zeros()
        A.sort_indices()
        Ap = PETSc.Mat().createAIJ(size=A.shape, csr=(A.indptr.astype(PETSc.IntType),
                                                      A.indices.astype(PETSc.IntType), A.data),
                                   comm=PETSc.COMM_SELF)
        Ap.assemble()
        ksp = PETSc.KSP().create(PETSc.COMM_SELF)
        ksp.setOperators(Ap)
        prefix = f"dtivl_transient_{uuid.uuid4().hex[:8]}_"
        ksp.setOptionsPrefix(prefix)
        opts = PETSc.Options()
        for k, val in LU_OPTIONS.items():
            opts[prefix + k] = val
        ksp.setFromOptions()
        for k in LU_OPTIONS:
            del opts[prefix + k]
        b_np, x_np = np.zeros(A.shape[0]), np.zeros(A.shape[0])
        b, x = (PETSc.Vec().createWithArray(arr, comm=PETSc.COMM_SELF) for arr in (b_np, x_np))

        def solve(rhs: np.ndarray) -> np.ndarray:
            b_np[:] = rhs
            b_np[fixed] = 0.0
            ksp.solve(b, x)
            if ksp.getConvergedReason() <= 0:
                raise RuntimeError(f"LU solve failed (KSP reason {ksp.getConvergedReason()})")
            return x_np.copy()

        solve._keep = (Ap, ksp, b, x)  # keep PETSc objects alive with the closure
        return solve

    def static_solution(self, p: float, k_found: float) -> tuple[np.ndarray, np.ndarray | None]:
        """Static lumen-pressure solution ``(u, pressure)`` with a weak outer foundation.

        ``(Kd + Kv + K_f) u = p f_lumen`` (displacement) or the corresponding
        saddle-point system (mixed; ``pressure`` is the P1 pressure field,
        positive in compression, else ``None``). No inertia, no dashpot.
        """
        from dolfinx import fem

        if not k_found > 0 and self.outer_bc != "fixed":
            raise ValueError("k_found must be positive for the static solve")
        u, v = ufl.TrialFunction(self.V), ufl.TestFunction(self.V)
        kf = fem.Constant(self.mesh, float(k_found))
        A = self.Kd + _to_scipy(fem.form(kf * ufl.inner(u, v) * self._ds(self._outer_tag)))
        if self.Kv is not None:
            A = A + self.Kv
        rhs = np.zeros(self.ndofs)
        rhs[:self.n_u] = p * self.f_lumen
        z = self._factorise(self._system(A, 1.0))(rhs)
        return z[:self.n_u], (z[self.n_u:] if self.n_p else None)

    # --- cell stresses --------------------------------------------------
    def cell_stress_operator(self, cells: np.ndarray):
        """Return ``sigma1(u, pressure=None) -> (n,)`` for the given local cells.

        Uses sparse operators mapping the displacement (and pressure) DOFs
        to cell averages of the strain components (2D: e11, e22, e12; 3D:
        six) and of the pressure. The returned function carries the cell
        measures as attribute ``measure``.
        """
        from dolfinx import fem

        cells = np.asarray(cells, dtype=np.int32)
        key = cells.tobytes()
        if key in self._cell_ops:
            return self._cell_ops[key]
        mesh, gdim = self.mesh, self.gdim
        e = ufl.sym(ufl.grad(ufl.TrialFunction(self.V)))
        comps = [e[0, 0], e[1, 1], e[0, 1]] if gdim == 2 else \
            [e[0, 0], e[1, 1], e[2, 2], e[0, 1], e[0, 2], e[1, 2]]
        nc = len(comps)
        W = fem.functionspace(mesh, ("DG", 0, (nc,)))
        B = _to_scipy(fem.form(ufl.inner(ufl.as_vector(comps), ufl.TestFunction(W)) * ufl.dx))
        Q0 = fem.functionspace(mesh, ("DG", 0))
        w0 = ufl.TestFunction(Q0)
        vol = fem.assemble_vector(fem.form(w0 * ufl.dx)).array
        dofs0 = Q0.dofmap.list[cells, 0]
        meas = vol[dofs0]
        rows = (W.dofmap.list[cells, 0][:, None] * nc + np.arange(nc)[None, :]).ravel()
        Bc = B[rows, :].multiply(np.repeat(1.0 / meas, nc)[:, None]).tocsr()
        Bp = None
        if self.Q is not None:
            import scipy.sparse as sp

            # cell integrals of each material's own pressure block (extended numbering)
            ct = self.meshdata.cell_tags
            dxt = ufl.Measure("dx", domain=mesh, subdomain_data=ct)
            pt = ufl.TrialFunction(self.Q)
            blocks = []
            for t, pdofs, _ in self._p_blocks:
                form = pt * w0 * (ufl.dx if t is None else dxt(t))
                blocks.append(_to_scipy(fem.form(form))[dofs0][:, pdofs])
            Bp = sp.hstack(blocks, format="csr").multiply((1.0 / meas)[:, None]).tocsr()
        mu = self._coef["mu"].x.array[dofs0]
        K = self._coef["K"].x.array[dofs0]

        def sigma1(disp: np.ndarray, pres: np.ndarray | None = None) -> np.ndarray:
            c = (Bc @ disp).reshape(-1, nc)
            tr = c[:, 0] + c[:, 1] + (c[:, 2] if gdim == 3 else 0.0)
            if Bp is not None:
                if pres is None:
                    raise ValueError("mixed formulation: the pressure field is required")
                pm = Bp @ pres
            else:
                pm = -K * tr
            iso = -2.0 * mu * tr / 3.0 - pm     # isotropic part of sigma: 2 mu (-tr/3) - p
            if gdim == 2:
                s11 = iso + 2 * mu * c[:, 0]
                s22 = iso + 2 * mu * c[:, 1]
                s12 = 2 * mu * c[:, 2]
                m, r = 0.5 * (s11 + s22), np.sqrt((0.5 * (s11 - s22)) ** 2 + s12 ** 2)
                return np.maximum(m + r, iso)          # sigma_33 = iso (eps_33 = 0)
            S = np.empty((c.shape[0], 3, 3))
            for i in range(3):
                S[:, i, i] = iso + 2 * mu * c[:, i]
            for (i, j), k in zip(((0, 1), (0, 2), (1, 2)), (3, 4, 5)):
                S[:, i, j] = S[:, j, i] = 2 * mu * c[:, k]
            return np.linalg.eigvalsh(S)[:, -1]

        sigma1.measure = meas
        self._cell_ops[key] = sigma1
        return sigma1

    # --- time integration -----------------------------------------------
    def default_dt(self, width: float) -> float:
        """``min(width / STEPS_PER_PULSE, CFL h_min / c_max)``."""
        return min(width / STEPS_PER_PULSE, CFL * self.h_min / self.c_max)

    def run(self, pulse: Callable[[float], float], dt: float, t_end: float, *, rho_inf: float = RHO_INF,
            cells: np.ndarray | None = None,
            callback: Callable[[int, float, np.ndarray, np.ndarray, np.ndarray], None] | None = None,
            record_energy: bool = False) -> "TransientHistory":
        """Integrate ``M a + C v + K u = p(t) f_lumen`` from rest on ``[0, t_end]``.

        Generalized-alpha with spectral radius ``rho_inf`` in [0, 1]
        (``1`` = average acceleration). In the mixed formulation the
        pressure is algebraic and is solved for together with the
        acceleration at the generalized-alpha time level. If ``cells`` is
        given, the max sigma1 over those cells is recorded at every step,
        along with the per-cell peak over time. ``callback(step, t, u, v,
        a)`` is called after every step (and at t = 0) with the expanded
        displacement-DOF arrays.
        """
        if not 0.0 <= rho_inf <= 1.0:
            raise ValueError("rho_inf must be in [0, 1]")
        if not (dt > 0 and t_end > 0):
            raise ValueError("dt and t_end must be positive")
        am = (2.0 * rho_inf - 1.0) / (rho_inf + 1.0)
        af = rho_inf / (rho_inf + 1.0)
        gamma = 0.5 - am + af
        beta = 0.25 * (1.0 - am + af) ** 2
        n_steps = int(math.ceil(t_end / dt - 1e-9))
        M, Kd, C = self.M, self.Kd, self.C
        K_u = Kd if self.Kv is None else (Kd + self.Kv).tocsr()   # u-block stiffness
        c_k = (1.0 - af) * beta * dt * dt
        A = (1.0 - am) * M + c_k * K_u
        if C is not None:
            A = A + (1.0 - af) * gamma * dt * C
        solve = self._factorise(self._system(A, c_k))

        nu_, np_ = self.n_u, self.n_p
        u, vel, acc = np.zeros(nu_), np.zeros(nu_), np.zeros(nu_)
        pres = np.zeros(np_) if np_ else None
        f0 = pulse(0.0) * self.f_lumen
        if np.any(f0):   # u = 0 -> pressure 0; M a0 = f0
            acc = self._factorise(M.tocsr())(f0)
        s1 = self.cell_stress_operator(cells) if cells is not None else None
        times = np.arange(n_steps + 1) * dt
        hist_max = np.full(n_steps + 1, np.nan)
        peak_cell = None
        if s1 is not None:
            sig = s1(u, pres)
            if sig.size:
                hist_max[0] = sig.max()
            peak_cell = sig.copy()
        energy = np.zeros((n_steps + 1, 2)) if record_energy else None
        load = np.array([pulse(t) for t in times])
        if callback is not None:
            callback(0, 0.0, u, vel, acc)
        rhs = np.zeros(self.ndofs)
        for k in range(n_steps):
            t_a = times[k + 1] - af * dt
            u_pred = u + (1.0 - af) * (dt * vel + (0.5 - beta) * dt * dt * acc)
            r_u = pulse(t_a) * self.f_lumen - am * (M @ acc) - K_u @ u_pred
            if C is not None:
                r_u -= C @ (vel + (1.0 - af) * (1.0 - gamma) * dt * acc)
            rhs[:nu_] = r_u
            if np_:
                rhs[nu_:] = (self.G.T @ u_pred) / c_k
            z = solve(rhs)
            acc_new = z[:nu_]
            u = u + dt * vel + dt * dt * ((0.5 - beta) * acc + beta * acc_new)
            vel = vel + dt * ((1.0 - gamma) * acc + gamma * acc_new)
            acc = acc_new
            if np_:   # z holds the pressure at t_{n+1-af}
                pres = (z[nu_:] - af * pres) / (1.0 - af) if af < 1.0 else z[nu_:]
            if s1 is not None and peak_cell.size:
                sig = s1(u, pres)
                hist_max[k + 1] = sig.max()
                np.maximum(peak_cell, sig, out=peak_cell)
            if energy is not None:
                energy[k + 1] = (self.kinetic_energy(vel), self.strain_energy(u, pres))
            if callback is not None:
                callback(k + 1, float(times[k + 1]), u, vel, acc)
        return TransientHistory(t=times, pressure=load, sigma1_max=hist_max, sigma1_cell_peak=peak_cell,
                                energy=energy, u_final=u, v_final=vel, p_final=pres,
                                info={"dt": dt, "t_end": float(times[-1]), "n_steps": n_steps,
                                      "rho_inf": rho_inf, "alpha_m": am, "alpha_f": af,
                                      "beta": beta, "gamma": gamma, "ndofs": self.ndofs,
                                      "formulation": self.formulation})


@dataclass
class TransientHistory:
    """Time history returned by :meth:`ElastodynamicModel.run` (times in us)."""

    t: np.ndarray                         # (n+1,) time levels (us)
    pressure: np.ndarray                  # (n+1,) lumen pressure p(t) (MPa)
    sigma1_max: np.ndarray                # (n+1,) max sigma1 over recorded cells (MPa), NaN if none
    sigma1_cell_peak: np.ndarray | None   # (n_cells,) per-cell peak sigma1 over time (MPa)
    energy: np.ndarray | None             # (n+1, 2) kinetic and strain energy, if recorded
    u_final: np.ndarray
    v_final: np.ndarray
    p_final: np.ndarray | None = None     # mixed formulation: final pressure field
    info: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Convenience drivers
# ---------------------------------------------------------------------------
def _resolve_inputs(densities, sound_speeds, allow_pending):
    if densities is None:
        densities = default_densities(allow_pending=allow_pending)
    if sound_speeds is None:
        sound_speeds = default_sound_speeds(allow_pending=allow_pending)
    return densities, sound_speeds


def build_model(meshdata, ms: MaterialSet, densities: Mapping[int, float] | None = None,
                sound_speeds: Mapping[int, float] | None = None, *, allow_pending: bool = False,
                **kwargs) -> ElastodynamicModel:
    """Build an :class:`ElastodynamicModel` with acoustic materials.

    ``densities`` (N us^2/mm^4 = kg/m^3) and ``sound_speeds`` (mm/us) are
    ``{cell_tag: value}``; ``None`` reads the config defaults
    (:func:`default_densities`, :func:`default_sound_speeds`), refusing
    PENDING entries unless ``allow_pending=True``. ``kwargs`` go to the
    model constructor.
    """
    densities, sound_speeds = _resolve_inputs(densities, sound_speeds, allow_pending)
    mats = acoustic_materials(ms, densities, sound_speeds, np.unique(meshdata.cell_tags.values))
    return ElastodynamicModel(meshdata, mats, **kwargs)


def quasi_static_materials(ms: MaterialSet, tags, densities: Mapping[int, float] | None = None
                           ) -> dict[int, AcousticMaterial]:
    """Per-tag constants of the quasi-static model (``K`` and ``mu`` from ``ms.lame``).

    The density (default 1) only enters inertia and is irrelevant for
    static solves.
    """
    out = {}
    for tag in tags:
        tag = int(tag)
        lam, mu = ms.lame(tag)
        rho = 1.0 if densities is None else float(densities[tag])
        out[tag] = AcousticMaterial(rho=rho, c_p=math.sqrt((lam + 2.0 * mu) / rho), mu=mu,
                                    K=lam + 2.0 * mu / 3.0)
    return out


def _calcium_cells(meshdata, plm_or_calcium_tag) -> tuple[np.ndarray, int]:
    if plm_or_calcium_tag is None or not isinstance(plm_or_calcium_tag, (int, np.integer)):
        tag = TAG_CALCIUM          # PolarLabelMap (or None): calcium cells carry TAG_CALCIUM
    else:
        tag = int(plm_or_calcium_tag)
    cells = meshdata.cell_tags.find(tag)
    n_owned = meshdata.mesh.topology.index_map(meshdata.mesh.topology.dim).size_local
    cells = np.sort(cells[cells < n_owned]).astype(np.int32)
    if cells.size == 0:
        raise ValueError(f"no cells with tag {tag} (calcium) in the mesh")
    return cells, tag


def transfer_factor(meshdata, plm_or_calcium_tag, ms: MaterialSet,
                    densities: Mapping[int, float] | None = None,
                    sound_speeds: Mapping[int, float] | None = None, *,
                    p_peak: float | None = None, width_us: float | None = None,
                    dt: float | None = None, t_end: float | None = None, rho_inf: float = RHO_INF,
                    degree: int = DEGREE, formulation: str = "mixed", volumetric: str = "full",
                    outer_bc: str = "absorbing", interface_pressure: str = "discontinuous",
                    k_found: float | None = None, equivalent_static: bool = True,
                    allow_pending: bool = False) -> dict:
    """Dynamic transfer factor ``T = sigma1_peak,transient / sigma1_static`` of one lesion.

    Parameters
    ----------
    meshdata : tagged mesh (``TAG_*`` cell tags, lumen/outer facet tags;
        3D: ``TAG_END_BND`` ends with ``u_z = 0``).
    plm_or_calcium_tag : the lesion's ``PolarLabelMap`` (calcium cells are
        then the ``TAG_CALCIUM`` cells of the conforming mesh built from it)
        or an integer cell tag identifying the calcium region.
    ms : material set (shear moduli; quasi-static reference moduli).
    densities, sound_speeds : ``{cell_tag: value}`` (N us^2/mm^4, mm/us);
        ``None`` reads config defaults (PENDING refused unless
        ``allow_pending``).
    p_peak : peak pulse pressure (MPa), default :func:`default_peak_pressure`.
        T is independent of it (linear problem).
    width_us : half-sine width (us), default :func:`default_pulse_width_us`.
    dt : time step (us), default :meth:`ElastodynamicModel.default_dt`.
    t_end : simulated window (us), default ``WINDOW_PULSES * width``.
    rho_inf : generalized-alpha spectral radius (1 = average acceleration).
    formulation, volumetric, degree, outer_bc, interface_pressure : see
        :class:`ElastodynamicModel` (the static denominator uses the same
        discretisation).
    k_found : static foundation stiffness, default
        ``loads.perivascular_k_MPa_per_mm``.
    equivalent_static : also compute ``T_equivalent`` with the quasi-static
        model of design.md section 4 (``MaterialSet.lame`` moduli,
        displacement formulation, full integration).

    Returns
    -------
    dict with ``T``, ``sigma1_peak_transient``, ``sigma1_static``,
    ``t_peak`` (us), ``peak_at_window_end`` (bool: the peak occurred in the
    last 5 % of the window; extend ``t_end``), ``t``, ``sigma1_max_history``,
    ``pressure_history``, ``sigma1_cell_peak``, ``sigma1_static_cells``,
    ``ca_cells``, ``ca_tag``, ``ca_measure``, ``T_equivalent`` and
    ``sigma1_static_equivalent`` (if requested), ``p_peak``, ``width_us``,
    ``dt``, ``t_end``, ``materials`` (per-tag rho, c_p, c_s, mu, K, nu) and
    ``info``.
    """
    densities, sound_speeds = _resolve_inputs(densities, sound_speeds, allow_pending)
    if p_peak is None:
        p_peak = default_peak_pressure()
    if width_us is None:
        width_us = default_pulse_width_us()
    if k_found is None:
        k_found = P.get("loads.perivascular_k_MPa_per_mm").value
    cells, ca_tag = _calcium_cells(meshdata, plm_or_calcium_tag)

    model = build_model(meshdata, ms, densities, sound_speeds, degree=degree, formulation=formulation,
                        volumetric=volumetric, outer_bc=outer_bc, interface_pressure=interface_pressure)
    if dt is None:
        dt = model.default_dt(width_us)
    if t_end is None:
        t_end = WINDOW_PULSES * width_us
    hist = model.run(half_sine(p_peak, width_us), dt, t_end, rho_inf=rho_inf, cells=cells)
    s1 = model.cell_stress_operator(cells)
    i_peak = int(np.nanargmax(hist.sigma1_max))
    sig_tr = float(hist.sigma1_max[i_peak])
    s_static = s1(*model.static_solution(p_peak, k_found))
    sig_st = float(s_static.max())

    out = {
        "T": sig_tr / sig_st if sig_st > 0 else float("nan"),
        "sigma1_peak_transient": sig_tr,
        "sigma1_static": sig_st,
        "t_peak": float(hist.t[i_peak]),
        "peak_at_window_end": bool(hist.t[i_peak] >= 0.95 * hist.t[-1]),
        "t": hist.t,
        "sigma1_max_history": hist.sigma1_max,
        "pressure_history": hist.pressure,
        "sigma1_cell_peak": hist.sigma1_cell_peak,
        "sigma1_static_cells": s_static,
        "ca_cells": cells,
        "ca_tag": ca_tag,
        "ca_measure": s1.measure,
        "p_peak": float(p_peak),
        "width_us": float(width_us),
        "dt": float(dt),
        "t_end": float(hist.t[-1]),
        "materials": {MATERIAL_NAMES.get(t, str(t)): {"rho": m.rho, "c_p": m.c_p, "c_s": m.c_s, "mu": m.mu,
                                                       "K": m.K, "nu": m.nu}
                      for t, m in model.materials.items()},
        "info": dict(hist.info, volumetric=volumetric, outer_bc=outer_bc, degree=degree,
                     interface_pressure=interface_pressure,
                     k_found=float(k_found), h_min=model.h_min),
    }
    if equivalent_static:
        qmodel = ElastodynamicModel(meshdata, quasi_static_materials(ms, model.materials), degree=degree,
                                    formulation="displacement", volumetric="full", outer_bc="free")
        s_eq = qmodel.cell_stress_operator(cells)(*qmodel.static_solution(p_peak, k_found))
        sig_eq = float(s_eq.max())
        out["sigma1_static_equivalent"] = sig_eq
        out["T_equivalent"] = sig_tr / sig_eq if sig_eq > 0 else float("nan")
    return out

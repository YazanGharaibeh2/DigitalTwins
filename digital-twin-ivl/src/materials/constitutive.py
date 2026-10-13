"""Constitutive models (UFL) and the material parameter set.

All parameter values come from :mod:`dtivl.params` (literature table + study
config). Units: mm, MPa.

Finite-strain energies (reference configuration):
  wall      Yeoh:      W = c10(I1b-3) + c20(I1b-3)^2 + c30(I1b-3)^3 + U(J)
  fibrous   neo-Hooke: W = mu/2 (I1b-3) + U(J),  mu = E/3, K = k_over_mu * mu
  lipid     neo-Hooke: same form
  calcium   compressible neo-Hooke with volumetric-deviatoric split:
            psi+ = mu/2 (I1b-3) + U(J)[J>=1],  psi- = U(J)[J<1]
            W = g(d) psi+ + psi-,  g(d) = (1-k)(1-d)^2 + k
            k = k_floor: the crack-filler floor mu_fibrous/mu_calcium (a fully
            damaged band is as stiff in shear as the fibrous matrix), never
            below the numerical k_res; crack_filler="none" gives k = k_res.
  U(J) = K/4 (J^2 - 1 - 2 ln J)   (U''(1) = K)
I1b = J^(-2/3) tr(C). Plane strain embeds the 2x2 deformation gradient in
3x3 with F33 = 1.

Kinematic measure for the optional Rankine-type damage driving force
(:mod:`dtivl.fracture.phasefield`): :func:`max_principal_green_lagrange`
returns the maximum principal Green-Lagrange strain e1 of C = F^T F in
closed form (2D plane strain only).

Small-strain limits used by the linear predictor analysis:
  mu_wall = 2 c10, mu_soft = E/3, K_soft = k_over_mu * mu, calcium (E, nu).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import ufl

from dtivl import params as P
from dtivl.geometry.labels import TAG_CALCIUM, TAG_FIBROUS, TAG_LIPID, TAG_WALL


@dataclass(frozen=True)
class MaterialSet:
    wall_c10: float
    wall_c20: float
    wall_c30: float
    fibrous_E: float
    lipid_E: float
    ca_E: float
    ca_nu: float
    k_over_mu: float
    eps_f: float
    sigma_c: float
    failure_mode: str = "strain"          # "strain" (primary, D5) or "stress"
    k_res: float = 1e-6
    crack_filler: str = "matrix"          # "matrix" (baseline) or "none"
    provenance: dict = field(default_factory=dict, compare=False)

    @classmethod
    def baseline(cls, **overrides) -> "MaterialSet":
        q = {
            "wall_c10": P.get("materials.wall.c10"),
            "wall_c20": P.get("materials.wall.c20"),
            "wall_c30": P.get("materials.wall.c30"),
            "fibrous_E": P.get("materials.fibrous.E"),
            "lipid_E": P.get("materials.lipid.E"),
            "ca_E": P.get("materials.calcium.E"),
            "ca_nu": P.get("materials.calcium.nu"),
            "k_over_mu": P.get("materials.bulk_to_shear_soft"),
            "eps_f": P.get("fracture.eps_f"),
            "sigma_c": P.get("fracture.sigma_c_stress_controlled"),
        }
        vals = {k: v.value for k, v in q.items()}
        prov = {k: v.provenance for k, v in q.items()}
        vals["k_res"] = float(P.raw("fracture.k_residual"))
        vals["crack_filler"] = str(P.raw("fracture.crack_filler")["value"])
        ms = cls(**vals, provenance=prov)
        if overrides:
            ms = replace(ms, **overrides)
            ms.provenance.update({k: "override" for k in overrides})
        return ms

    # --- derived small-strain constants -------------------------------
    @property
    def mu_wall0(self) -> float:
        return 2.0 * self.wall_c10

    @property
    def mu_fibrous(self) -> float:
        return self.fibrous_E / 3.0

    @property
    def mu_lipid(self) -> float:
        return self.lipid_E / 3.0

    @property
    def ca_mu(self) -> float:
        return self.ca_E / (2.0 * (1.0 + self.ca_nu))

    @property
    def ca_K(self) -> float:
        return self.ca_E / (3.0 * (1.0 - 2.0 * self.ca_nu))

    def soft_shear_bulk(self, tag: int) -> tuple[float, float]:
        mu = {TAG_WALL: self.mu_wall0, TAG_FIBROUS: self.mu_fibrous, TAG_LIPID: self.mu_lipid}[tag]
        return mu, self.k_over_mu * mu

    def lame(self, tag: int) -> tuple[float, float]:
        """Small-strain (lambda, mu) for a cell tag."""
        if tag == TAG_CALCIUM:
            mu, K = self.ca_mu, self.ca_K
        else:
            mu, K = self.soft_shear_bulk(tag)
        return K - 2.0 * mu / 3.0, mu

    # --- fracture ------------------------------------------------------
    @property
    def psi_crit(self) -> float:
        """Damage-onset energy density (uniaxial-tension equivalence).

        strain-controlled: E eps_f^2 / 2 ; stress-controlled: sigma_c^2 / (2E).
        """
        if self.failure_mode == "strain":
            return 0.5 * self.ca_E * self.eps_f ** 2
        if self.failure_mode == "stress":
            return 0.5 * self.sigma_c ** 2 / self.ca_E
        raise ValueError(self.failure_mode)

    @property
    def sigma_crit(self) -> float:
        return (2.0 * self.ca_E * self.psi_crit) ** 0.5

    @property
    def k_floor(self) -> float:
        """Stiffness floor of fully damaged calcium, g(1) = k_floor.

        ``crack_filler="matrix"``: ``max(k_res, mu_fibrous / mu_calcium)``;
        ``"none"``: ``k_res``. The filler is undefined when calcium is not
        stiffer in shear than the fibrous matrix (ValueError).
        """
        if self.crack_filler == "none":
            return self.k_res
        if self.crack_filler != "matrix":
            raise ValueError(self.crack_filler)
        ratio = self.mu_fibrous / self.ca_mu
        if ratio >= 1.0:
            raise ValueError(f"crack filler undefined: calcium shear modulus {self.ca_mu:.4g} MPa is not above "
                             f"the fibrous matrix ({self.mu_fibrous:.4g} MPa)")
        return max(self.k_res, ratio)

    def gc(self, ell: float) -> float:
        """AT1 with g(d) = (1-k)(1-d)^2 + k: onset when psi+ = 3 Gc / (16 ell (1-k)),
        so Gc = 16 ell (1-k) psi_crit / 3 keeps the onset at psi_crit."""
        return 16.0 * ell * (1.0 - self.k_floor) * self.psi_crit / 3.0


# ----------------------------------------------------------------------
# Kinematics and energies (UFL)
# ----------------------------------------------------------------------
def deformation_gradient(u, plane_strain: bool):
    """Return (F_var, F3): the ufl.variable to differentiate and its 3x3 embedding."""
    gdim = u.ufl_shape[0]
    Fv = ufl.variable(ufl.Identity(gdim) + ufl.grad(u))
    if gdim == 2:
        if not plane_strain:
            raise NotImplementedError("2D requires plane strain")
        F3 = ufl.as_matrix([[Fv[0, 0], Fv[0, 1], 0], [Fv[1, 0], Fv[1, 1], 0], [0, 0, 1]])
    else:
        F3 = Fv
    return Fv, F3


def _invariants(F3):
    C = F3.T * F3
    J = ufl.det(F3)
    I1b = J ** (-2.0 / 3.0) * ufl.tr(C)
    return J, I1b


def U_vol(J, K):
    return K / 4.0 * (J ** 2 - 1.0 - 2.0 * ufl.ln(J))


def psi_yeoh(F3, c10, c20, c30, K):
    J, I1b = _invariants(F3)
    x = I1b - 3.0
    return c10 * x + c20 * x ** 2 + c30 * x ** 3 + U_vol(J, K)


def psi_neo_hookean(F3, mu, K):
    J, I1b = _invariants(F3)
    return mu / 2.0 * (I1b - 3.0) + U_vol(J, K)


def psi_calcium_split(F3, mu, K):
    """Return (psi_plus, psi_minus) with the volumetric-deviatoric split."""
    J, I1b = _invariants(F3)
    dev = mu / 2.0 * (I1b - 3.0)
    vol = U_vol(J, K)
    psi_p = dev + ufl.conditional(ufl.ge(J, 1.0), vol, 0.0)
    psi_m = ufl.conditional(ufl.lt(J, 1.0), vol, 0.0)
    return psi_p, psi_m


def max_principal_green_lagrange(F3, gdim: int):
    """Maximum principal Green-Lagrange strain ``e1`` (UFL), 2D plane strain only.

    ``F3`` is the 3x3 plane-strain embedding of :func:`deformation_gradient`
    (``F33 = 1``, no out-of-plane shear). With ``E = (C - I) / 2`` and
    ``C = F3^T F3``, the in-plane 2x2 block has the closed-form eigenvalues
    ``(E11 + E22)/2 +/- sqrt(((E11 - E22)/2)^2 + E12^2)``; the third principal
    strain is the out-of-plane ``E33 = (C33 - 1)/2``, which is zero in plane
    strain, so ``e1 = max(e1_in-plane, E33) = max(e1_in-plane, 0)``.

    The square root is not differentiable where the in-plane principal
    strains coincide; the expression is meant for the damage sub-problem,
    where it is evaluated with ``u`` frozen and never differentiated with
    respect to ``u``. ``gdim == 3`` raises ``NotImplementedError``.
    """
    if gdim != 2:
        raise NotImplementedError(
            "max_principal_green_lagrange: closed-form principal strain implemented for 2D plane strain only "
            f"(gdim={gdim}); the Rankine-type damage driving force is not available in 3D")
    C = F3.T * F3
    E = 0.5 * (C - ufl.Identity(3))
    mean = 0.5 * (E[0, 0] + E[1, 1])
    radius = ufl.sqrt((0.5 * (E[0, 0] - E[1, 1])) ** 2 + E[0, 1] ** 2)
    return ufl.max_value(mean + radius, E[2, 2])


def degradation(d, k):
    """g(d) = (1-k)(1-d)^2 + k: g(0) = 1, g(1) = k."""
    return (1.0 - k) * (1.0 - d) ** 2 + k


def soft_energy(tag: int, F3, ms: MaterialSet):
    if tag == TAG_WALL:
        mu0, K = ms.soft_shear_bulk(TAG_WALL)
        return psi_yeoh(F3, ms.wall_c10, ms.wall_c20, ms.wall_c30, K)
    mu, K = ms.soft_shear_bulk(tag)
    return psi_neo_hookean(F3, mu, K)


def linear_stress(u, lam, mu, plane_strain: bool = True):
    """Small-strain Cauchy stress (in-plane block for 2D) and strain."""
    eps = ufl.sym(ufl.grad(u))
    gdim = u.ufl_shape[0]
    sig = lam * ufl.tr(eps) * ufl.Identity(gdim) + 2.0 * mu * eps
    return sig, eps

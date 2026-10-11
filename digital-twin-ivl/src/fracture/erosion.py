"""Element-erosion comparator (design.md section 5.6).

The erosion model uses the same finite-strain displacement solver as the
phase-field model (:class:`dtivl.solvers.nonlinear.NonlinearState` built with
``fracture="erosion"``) but no damage field. Each calcium cell carries a DG0
factor ``eta`` (1 = intact, ``k_res`` = eroded) that scales its energy,
``W_Ca = eta (psi+ + psi-)`` (design.md: "energy scaled by k_res"). The
option ``SolverSettings(erosion_split=True)`` scales only the tensile part,
``W_Ca = eta psi+ + psi-`` (eroded cells keep their compressive volumetric
stiffness, like the phase-field degradation); it is numerically fragile:
the Hessian of an eroded cell then jumps by ``1/k_res`` at ``J = 1``, and
Newton stagnates when many eroded cells sit near ``J = 1`` (erosion
cascades). After every converged displacement solve, intact calcium cells
whose cell-averaged failure measure reaches the threshold are eroded and the
displacement is re-solved at the same load, until no new cell erodes
(:data:`ERODE_MAX_ROUNDS` at most). Newly eroded cells reach ``k_res``
through an adaptive continuation in ``log10(eta)``
(:meth:`ErosionUpdater.erode_with_continuation`), which only helps Newton
converge when many cells erode at once.

Failure measure (``MaterialSet.failure_mode``):

* ``"strain"`` (primary, D5): ``eps1 = sqrt(lambda_max(<C>)) - 1 >= eps_f``,
  the maximum principal stretch minus one of the cell-averaged right
  Cauchy-Green tensor (``eps_f`` is a failure stretch minus one);
* ``"stress"``: ``sigma1 = lambda_max(<sigma>) >= ms.sigma_crit`` with the
  cell-averaged 3x3 Cauchy stress ``sigma = J^-1 dW/dF F^T`` of the intact
  calcium energy (plane strain includes ``sigma_33``).

Cell averages are quadrature averages ``(1/|K|) int_K q dx``.

Crack post-processing treats eroded cells as cracked:
:func:`eroded_indicator` returns the per-cell array accepted by
:func:`dtivl.fracture.cracks.crack_analysis`.
"""
from __future__ import annotations

import math

import numpy as np
import ufl

from dtivl.geometry.labels import TAG_CALCIUM
from dtivl.materials.constitutive import psi_calcium_split

#: Maximum erosion/re-solve rounds per load increment (numerical setting).
ERODE_MAX_ROUNDS = 200
#: Continuation of newly eroded cells: eta goes from 1 to k_res in steps of
#: log10(eta) (initially 2 decades), one displacement solve per stage; a failed
#: stage is retried with half the log-step (at most EROSION_MAX_HALVINGS times),
#: a converged one doubles it again (up to the initial step). Newton robustness
#: only: the converged state at eta = k_res is unchanged.
EROSION_CONTINUATION_DECADES = 2.0
EROSION_MAX_HALVINGS = 6


def failure_threshold(ms) -> tuple[str, float]:
    """``("strain", eps_f)`` or ``("stress", sigma_crit)`` from ``ms.failure_mode``."""
    if ms.failure_mode == "strain":
        return "strain", float(ms.eps_f)
    if ms.failure_mode == "stress":
        return "stress", float(ms.sigma_crit)
    raise ValueError(ms.failure_mode)


def _sym6(T):
    return [T[0, 0], T[1, 1], T[2, 2], T[0, 1], T[0, 2], T[1, 2]]


def _from6(c: np.ndarray) -> np.ndarray:
    T = np.empty((c.shape[0], 3, 3))
    T[:, 0, 0], T[:, 1, 1], T[:, 2, 2] = c[:, 0], c[:, 1], c[:, 2]
    T[:, 0, 1] = T[:, 1, 0] = c[:, 3]
    T[:, 0, 2] = T[:, 2, 0] = c[:, 4]
    T[:, 1, 2] = T[:, 2, 1] = c[:, 5]
    return T


class CalciumCellFields:
    """Cell-averaged Cauchy stress and right Cauchy-Green tensor in calcium.

    Parameters
    ----------
    u : displacement function.
    F3 : 3x3 deformation gradient (UFL; plane-strain embedding in 2D).
    ms_ufl : material set whose calcium constants may be UFL constants.
    dx : cell measure with the cell tags as subdomain data.
    ca_cells : local indices of the (owned) calcium cells.
    """

    def __init__(self, u, F3, ms_ufl, dx, ca_cells: np.ndarray):
        from dolfinx import fem

        mesh = u.function_space.mesh
        F3v = ufl.variable(F3)
        psi_p, psi_m = psi_calcium_split(F3v, ms_ufl.ca_mu, ms_ufl.ca_K)
        Pk = ufl.diff(psi_p + psi_m, F3v)
        J = ufl.det(F3v)
        sigma = Pk * F3v.T / J
        C = F3v.T * F3v
        comps = _sym6(sigma) + _sym6(C) + [ufl.as_ufl(1.0)]
        self._n = len(comps)
        W = fem.functionspace(mesh, ("DG", 0, (self._n,)))
        self._form = fem.form(ufl.inner(ufl.as_vector(comps), ufl.TestFunction(W)) * dx(TAG_CALCIUM))
        self._blocks = W.dofmap.list[ca_cells, 0]
        self.ca_cells = ca_cells

    def compute(self) -> dict:
        """Return ``sigma``/``C`` (n, 3, 3), ``sigma1``, ``eps1`` and ``measure`` for ``ca_cells``."""
        from dolfinx import fem

        vec = fem.assemble_vector(self._form)
        data = vec.array.reshape(-1, self._n)[self._blocks]
        meas = data[:, -1]
        avg = data / np.where(meas > 0, meas, 1.0)[:, None]
        sigma, C = _from6(avg[:, 0:6]), _from6(avg[:, 6:12])
        lam_max = np.linalg.eigvalsh(C)[:, -1]
        return {"sigma": sigma, "C": C, "sigma1": np.linalg.eigvalsh(sigma)[:, -1],
                "eps1": np.sqrt(np.maximum(lam_max, 0.0)) - 1.0, "measure": meas}


class ErosionUpdater:
    """Erosion step of a :class:`~dtivl.solvers.nonlinear.NonlinearState`.

    Uses the state's ``u``, ``F3``, ``ms``/``ms_ufl``, ``dx``, ``ca_cells``
    and the DG0 erosion factor ``eta``.
    """

    def __init__(self, state):
        self.state = state
        self.mode, self.threshold = failure_threshold(state.ms)
        self.fields = CalciumCellFields(state.u, state.F3, state.ms_ufl, state.dx, state.ca_cells)
        self._eta_dofs = state.eta.function_space.dofmap.list[state.ca_cells, 0]

    def intact(self) -> np.ndarray:
        """Boolean mask over ``ca_cells``: True where the cell is not eroded."""
        return self.state.eta.x.array[self._eta_dofs] >= 1.0

    def measure(self) -> np.ndarray:
        """Failure measure (eps1 or sigma1) of every calcium cell."""
        f = self.fields.compute()
        return f["eps1"] if self.mode == "strain" else f["sigma1"]

    def find_new(self) -> np.ndarray:
        """``eta`` DOFs of intact calcium cells whose measure reached the threshold."""
        crit = self.measure()
        return self._eta_dofs[self.intact() & (crit >= self.threshold)]

    def erode_with_continuation(self, dofs: np.ndarray, solve) -> int:
        """Bring ``eta`` of ``dofs`` from 1 to ``k_res`` by adaptive log-continuation.

        ``solve()`` re-solves the displacement and raises on failure (the
        displacement array is then restored and the log-step halved).
        Returns the number of displacement solves. Raises the last solver
        error if the step falls below the minimum.
        """
        u = self.state.u.x.array
        target = math.log10(self.state.ms.k_res)
        step0 = -EROSION_CONTINUATION_DECADES
        cur, step, n = 0.0, step0, 0
        while cur > target + 1e-12:
            nxt = max(cur + step, target)
            u_old = u.copy()
            self.erode(dofs, 10.0 ** nxt if nxt > target else None)
            try:
                solve()
                n += 1
                cur, step = nxt, max(2.0 * step, step0)
            except Exception:
                u[:] = u_old
                self.erode(dofs, 10.0 ** cur)
                step /= 2.0
                if abs(step) < abs(step0) / 2 ** EROSION_MAX_HALVINGS:
                    raise
        return n

    def erode(self, dofs: np.ndarray, factor: float | None = None) -> None:
        """Set ``eta = factor`` (default ``k_res``) on the given ``eta`` DOFs."""
        self.state.eta.x.array[dofs] = self.state.ms.k_res if factor is None else factor
        self.state.eta.x.scatter_forward()

    def update(self) -> int:
        """Erode (directly to ``k_res``) intact cells at or above the threshold; return the count."""
        new = self.find_new()
        if new.size:
            self.erode(new)
        return int(self.state.comm.allreduce(int(new.size)))


def eroded_indicator(state) -> np.ndarray:
    """Per-cell array over all local cells: 1.0 for eroded calcium cells, else 0.0.

    Suitable as the ``d`` argument of
    :func:`dtivl.fracture.cracks.crack_analysis`.
    """
    tdim = state.mesh.topology.dim
    imap = state.mesh.topology.index_map(tdim)
    out = np.zeros(imap.size_local + imap.num_ghosts)
    dofs = state.eta.function_space.dofmap.list[state.ca_cells, 0]
    out[state.ca_cells] = (state.eta.x.array[dofs] < 1.0).astype(float)
    return out

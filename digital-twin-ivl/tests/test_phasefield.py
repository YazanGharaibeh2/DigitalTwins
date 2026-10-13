"""Tests for the AT1 phase-field model (design.md sections 5.2, 5.7)."""
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pytest

from dtivl import params as P
from dtivl.fracture import phasefield as PF
from dtivl.geometry.labels import TAG_CALCIUM, TAG_FIBROUS
from dtivl.materials.constitutive import MaterialSet


def test_regularisation_length_and_onset_energy():
    h = float(P.raw("discretisation.h_calcium_mm"))
    assert PF.regularisation_length() == pytest.approx(P.get("fracture.ell_over_h").value * h)
    assert PF.regularisation_length(0.05, 3.0) == pytest.approx(0.15)
    ms = MaterialSet.baseline()
    ell = PF.regularisation_length()
    # calibration: Gc = 16 ell (1-k) psi_crit / 3  =>  AT1 onset energy == psi_crit
    assert PF.onset_energy(ms.gc(ell), ell, ms.k_floor) == pytest.approx(ms.psi_crit, rel=1e-12)
    stress = MaterialSet.baseline(failure_mode="stress")
    assert PF.onset_energy(stress.gc(ell), ell, stress.k_floor) == pytest.approx(stress.psi_crit, rel=1e-12)
    assert PF.onset_energy(3.0, 1.0) == pytest.approx(3.0 * 3.0 / 16.0)


def test_crack_filler_floor():
    """g(0) = 1, g(1) = k_floor; the matrix filler gives a cracked band the fibrous shear modulus."""
    from dtivl.materials.constitutive import degradation

    ms = MaterialSet.baseline()
    assert ms.crack_filler == "matrix"
    k = ms.k_floor
    assert k == pytest.approx(ms.mu_fibrous / ms.ca_mu)
    assert degradation(0.0, k) == pytest.approx(1.0) and degradation(1.0, k) == pytest.approx(k)
    assert k * ms.ca_mu == pytest.approx(ms.mu_fibrous)
    stiff = MaterialSet.baseline(ca_E=1.0e5)
    assert stiff.k_floor == pytest.approx(max(stiff.k_res, stiff.mu_fibrous / stiff.ca_mu))
    assert MaterialSet.baseline(crack_filler="none").k_floor == ms.k_res
    with pytest.raises(ValueError):
        MaterialSet.baseline(ca_E=0.5 * ms.fibrous_E).k_floor


@pytest.fixture(scope="module")
def phasefield_report(tmp_path_factory):
    out = tmp_path_factory.mktemp("verif") / "verification_phasefield.json"
    rep = PF.verification_phasefield(out)
    return out, rep


def test_at1_onset_homogeneous_plate(phasefield_report):
    """Damage onset at psi+ = 3 Gc / (16 ell) within 5 % (plane strain plate and 3D bar)."""
    out, rep = phasefield_report
    assert out.exists()
    on_disk = json.loads(out.read_text())
    assert on_disk["passed"] is True
    assert {c["case"] for c in on_disk["cases"]} == {"2D", "3D"}
    for c in rep["cases"]:
        assert c["passed"], c
        assert c["psi_onset_theory_MPa"] == pytest.approx(
            3.0 * c["Gc_N_per_mm"] / (16.0 * c["ell_mm"] * (1.0 - c["k_floor"])))
        assert c["rel_err_psi"] < 0.05
        assert c["rel_err_strain"] < 0.05
        lo, hi = c["psi_bracket_MPa"]
        assert lo <= c["psi_onset_theory_MPa"] <= hi
        # the damage field stays homogeneous at onset
        assert c["d_uniformity"] < 1e-6
    case3d = next(c for c in rep["cases"] if c["case"] == "3D")
    # strain mode: the 3D uniaxial onset strain is the calibrated eps_f
    assert case3d["strain_onset_measured"] == pytest.approx(MaterialSet.baseline().eps_f, rel=0.05)


def test_onset_error_reflects_step_resolution(monkeypatch):
    """A coarse load step widens the bracket; the reported error must grow with it and fail
    (the AT1 equilibrium value psi+ (1 - d) equals the theory at any damaged step)."""
    monkeypatch.setattr(PF, "VERIF_PRE_FRAC", 0.6)
    monkeypatch.setattr(PF, "VERIF_PRE_STEPS", 1)
    monkeypatch.setattr(PF, "VERIF_FINE_STEP", 0.15)
    c = PF.onset_check(2)
    lo, hi = c["psi_bracket_MPa"]
    assert lo <= c["psi_onset_theory_MPa"] <= hi
    assert c["rel_err_psi_equilibrium"] < 1e-8           # damage solver consistency only
    assert c["rel_err_psi"] > PF.VERIF_TOL               # bracket is about 20 % wide
    assert not c["passed"]


def _bimaterial_plate():
    """2 x 0.5 mm plate, left half calcium, right half fibrous (no facet tags)."""
    from mpi4py import MPI
    from dolfinx import mesh as dmesh
    from dolfinx.io.gmsh import MeshData

    msh = dmesh.create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [2.0, 0.5]], (8, 2))
    tdim = msh.topology.dim
    n = msh.topology.index_map(tdim).size_local
    cen = msh.geometry.x[msh.geometry.dofmaps[0]].mean(axis=1)
    tags = np.where(cen[:n, 0] < 1.0, TAG_CALCIUM, TAG_FIBROUS).astype(np.int32)
    ct = dmesh.meshtags(msh, tdim, np.arange(n, dtype=np.int32), tags)
    msh.topology.create_connectivity(tdim - 1, tdim)
    ft = dmesh.meshtags(msh, tdim - 1, np.zeros(0, np.int32), np.zeros(0, np.int32))
    return MeshData(msh, ct, ft, None, None, {})


def test_damage_confined_to_calcium_and_irreversible():
    from dolfinx import fem

    from dtivl.solvers.nonlinear import NonlinearState, _ramp

    md = _bimaterial_plate()
    msh = md.mesh
    disp = fem.Constant(msh, 0.0)
    # tissue-level calcium modulus: the test needs damage within the soft-dominated strain ramp
    ms_t = MaterialSet.baseline(ca_E=P.get("materials.calcium.E_tissue_level").value)
    state = NonlinearState(md, ms_t, fracture="phasefield", bc_factory=PF._uniaxial_bcs(msh, 2.0, disp))
    ca_dofs = np.unique(state.D.dofmap.list[md.cell_tags.find(TAG_CALCIUM)].ravel())
    soft_only = np.setdiff1d(np.arange(state.d.x.array.size), ca_dofs)
    eps_f = state.ms.eps_f
    # the soft half carries most of the strain; ramp until calcium damages
    status = _ramp(state, phase="pull", load_name="strain", start=0.0, end=8.0 * eps_f, n_steps=8,
                   apply=lambda e: setattr(disp, "value", e * 2.0))
    assert status == "ok"
    d_loaded = state.d.x.array.copy()
    assert d_loaded.max() > 0.5
    assert np.all(d_loaded <= 1.0 + 1e-10) and np.all(d_loaded >= -1e-12)
    assert np.abs(d_loaded[soft_only]).max() < 1e-12
    hist_max = [r["max_d"] for r in state.history]
    assert all(b >= a - 1e-12 for a, b in zip(hist_max, hist_max[1:]))
    # unloading: damage does not heal
    status = _ramp(state, phase="release", load_name="strain", start=8.0 * eps_f, end=0.0, n_steps=2,
                   apply=lambda e: setattr(disp, "value", e * 2.0))
    assert status == "ok"
    assert np.all(state.d.x.array >= d_loaded - 1e-12)
    assert state.history[-1]["E_fracture"] > 0.0


# ---------------------------------------------------------------------------
# Damage driving force (D10 option b): "energy" (default) vs "rankine"
# ---------------------------------------------------------------------------
def test_driving_force_option_and_config():
    """SolverSettings takes its default from fracture.driving_force (energy); unknown names are refused."""
    from dtivl.solvers.nonlinear import SolverSettings

    entry = P.raw("fracture.driving_force")
    assert entry["value"] == "energy" and entry["sens"] == ["rankine"] and entry["status"] == "ASSUMPTION"
    assert PF.config_driving_force() == "energy"
    assert SolverSettings().driving_force == "energy"
    assert SolverSettings(driving_force="rankine").driving_force == "rankine"
    assert set(PF.DRIVING_FORCES) == {"energy", "rankine"}
    with pytest.raises(ValueError):
        PF.check_driving_force("tresca")
    assert PF.check_driving_force("rankine", 2) == "rankine"


def test_energy_driving_force_is_psi_plus():
    """Default: the damage problem is driven by the very same UFL psi+ (unchanged forms)."""
    from dtivl.solvers.nonlinear import NonlinearState, SolverSettings

    md = PF._box_meshdata(2)
    st = NonlinearState(md, MaterialSet.baseline(), fracture="phasefield")
    assert st.driving_force == "energy"
    assert st.psi_drive is st.psi_plus
    st_r = NonlinearState(md, MaterialSet.baseline(), settings=SolverSettings(driving_force="rankine"))
    assert st_r.driving_force == "rankine"
    assert st_r.psi_drive is not st_r.psi_plus


@pytest.mark.parametrize("grad", [
    [[0.03, 0.01], [-0.02, -0.01]],      # general in-plane state with shear (principal axes rotated)
    [[0.0, 0.04], [0.04, 0.0]],          # pure shear: e1 > 0 although E11 = E22 ~ 0
    [[-0.02, 0.0], [0.0, -0.01]],        # biaxial compression: in-plane e1 < 0, so e1 = E33 = 0
    [[0.02, 0.0], [0.0, 0.02]],          # equibiaxial: coincident eigenvalues (sqrt of zero)
    [[0.0, 0.0], [0.0, 0.0]],            # undeformed
])
def test_max_principal_green_lagrange_closed_form(grad):
    """Closed-form e1 (UFL) equals the largest eigenvalue of the 3x3 plane-strain Green-Lagrange tensor."""
    import ufl
    from dolfinx import fem

    from dtivl.materials.constitutive import deformation_gradient, max_principal_green_lagrange

    md = PF._box_meshdata(2)
    V = fem.functionspace(md.mesh, ("Lagrange", 1, (2,)))
    u = fem.Function(V)
    G = np.asarray(grad)
    u.interpolate(lambda x: G @ x[:2])
    _, F3 = deformation_gradient(u, plane_strain=True)
    dx = ufl.dx(domain=md.mesh)
    area = PF.VERIF_PLATE[0] * PF.VERIF_PLATE[1]
    F = np.eye(3)
    F[:2, :2] += G
    e1_ref = np.linalg.eigvalsh(0.5 * (F.T @ F - np.eye(3)))[-1]
    e1 = fem.assemble_scalar(fem.form(max_principal_green_lagrange(F3, 2) * dx)) / area
    assert e1 == pytest.approx(e1_ref, abs=1e-14)
    psi_R = fem.assemble_scalar(fem.form(PF.rankine_driving_force(F3, 2.0, 2) * dx)) / area
    assert psi_R == pytest.approx(max(e1_ref, 0.0) ** 2, abs=1e-14)
    with pytest.raises(NotImplementedError):
        max_principal_green_lagrange(F3, 3)


@pytest.mark.parametrize("modulus", ["baseline", "tissue_level"])
def test_rankine_onset_homogeneous_plate(modulus):
    """Plane-strain plate in uniaxial tension with the Rankine force: damage starts when the axial
    Green-Lagrange strain (the maximum principal strain) reaches eps_f, within 5 %, at the D8 baseline
    calcium modulus and at the tissue-level modulus (the onset strain does not depend on E)."""
    ms = (MaterialSet.baseline() if modulus == "baseline"
          else MaterialSet.baseline(ca_E=P.get("materials.calcium.E_tissue_level").value))
    assert ms.failure_mode == "strain"
    c = PF.onset_check(2, ms, driving_force="rankine")
    assert c["passed"], {k: v for k, v in c.items() if k != "steps"}
    assert c["driving_force"] == "rankine"
    assert c["gl_axial_onset_theory"] == pytest.approx(ms.eps_f, rel=1e-12)
    lo, hi = c["gl_axial_bracket"]
    assert lo <= ms.eps_f * (1 + 1e-12) and hi >= ms.eps_f
    assert c["rel_err_gl_axial"] < 0.05
    assert c["gl_axial_onset_measured"] == pytest.approx(ms.eps_f, rel=0.05)
    # AT1 calibration kept: onset at psi_R = psi_crit
    assert c["psi_onset_theory_MPa"] == pytest.approx(ms.psi_crit, rel=1e-12)
    assert c["rel_err_psi"] < 0.05
    # e1 from the closed-form eigenvalue is the axial strain of the plate; damage stays homogeneous
    assert c["e1_matches_gl_axial"] < 1e-10
    assert c["d_uniformity"] < 1e-6
    # no damage while the axial strain is below eps_f
    for r in c["steps"]:
        if r["gl_axial"] < ms.eps_f * (1 - 1e-12):
            assert r["d_max"] <= PF.VERIF_D_TOL


def test_rankine_3d_not_implemented():
    from dtivl.solvers.nonlinear import NonlinearState, SolverSettings

    md = PF._box_meshdata(3)
    with pytest.raises(NotImplementedError, match="2D plane strain only"):
        NonlinearState(md, MaterialSet.baseline(), settings=SolverSettings(driving_force="rankine"))
    with pytest.raises(NotImplementedError):
        PF.onset_check(3, driving_force="rankine")
    # the option only acts on the phase-field damage problem: accepted without a phase field
    NonlinearState(md, MaterialSet.baseline(), fracture="none", settings=SolverSettings(driving_force="rankine"))

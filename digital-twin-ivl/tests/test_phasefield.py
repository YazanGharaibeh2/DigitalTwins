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
    # calibration: Gc = 16 ell psi_crit / 3  =>  AT1 onset energy == psi_crit
    assert PF.onset_energy(ms.gc(ell), ell) == pytest.approx(ms.psi_crit, rel=1e-12)
    stress = MaterialSet.baseline(failure_mode="stress")
    assert PF.onset_energy(stress.gc(ell), ell) == pytest.approx(stress.psi_crit, rel=1e-12)


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
        assert c["psi_onset_theory_MPa"] == pytest.approx(3.0 * c["Gc_N_per_mm"] / (16.0 * c["ell_mm"]))
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
    state = NonlinearState(md, fracture="phasefield", bc_factory=PF._uniaxial_bcs(msh, 2.0, disp))
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

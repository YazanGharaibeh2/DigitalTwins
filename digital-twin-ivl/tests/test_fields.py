"""Tests for the field-export stage (src/drivers/stage_fields.py): layout of the exported fields."""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pytest

from dtivl.drivers import stage_fields as F
from dtivl.geometry.labels import TAG_CALCIUM, TAG_WALL

A, B, R1, R2, ARC = 0.8, 1.5, 0.9, 1.1, 180.0


@pytest.fixture(scope="module")
def state():
    from dtivl.solvers.nonlinear import NonlinearState, annulus_meshdata

    md = annulus_meshdata(A, B, 0.2, ca_sector=(R1, R2, ARC), h_ca=0.08, tag=TAG_WALL)
    return NonlinearState(md)


def test_extract_fields_layout(state):
    from dolfinx import fem

    # known fields: u = (x, 2y) (exact in P1), d = 1 on calcium nodes
    state.u.interpolate(lambda x: np.vstack([x[0], 2.0 * x[1]]))
    state.d.x.array[:] = 0.0
    f = F.extract_fields(state)
    n, m = f["points"].shape[0], f["triangles"].shape[0]
    assert f["points"].shape == (n, 2) and f["u"].shape == (n, 2) and f["d"].shape == (n,)
    assert f["triangles"].shape == (m, 3) and f["cell_tag"].shape == (m,) and f["d_cell"].shape == (m,)
    assert f["triangles"].min() == 0 and f["triangles"].max() == n - 1
    # displacement belongs to the points it is stored with (vector and scalar P1 layouts agree)
    np.testing.assert_allclose(f["u"][:, 0], f["points"][:, 0], atol=1e-12)
    np.testing.assert_allclose(f["u"][:, 1], 2.0 * f["points"][:, 1], atol=1e-12)
    S1 = fem.functionspace(state.mesh, ("Lagrange", 1))
    V1 = fem.functionspace(state.mesh, ("Lagrange", 1, (2,)))
    np.testing.assert_array_equal(np.asarray(S1.dofmap.list), np.asarray(V1.dofmap.list))
    # tags: calcium triangles lie in the calcium annular sector
    ca = f["cell_tag"] == TAG_CALCIUM
    assert ca.sum() == state.ca_cells.size > 0
    cen = f["points"][f["triangles"][ca]].mean(axis=1)
    r = np.hypot(cen[:, 0], cen[:, 1])
    assert (r > R1 - 1e-9).all() and (r < R2 + 1e-9).all()
    np.testing.assert_array_equal(f["d_cell"], 0.0)


def test_cell_damage_maps_to_calcium_cells(state):
    state.d.x.array[:] = 1.0
    f = F.extract_fields(state)
    ca = f["cell_tag"] == TAG_CALCIUM
    np.testing.assert_allclose(f["d_cell"][ca], 1.0, rtol=1e-10)
    np.testing.assert_array_equal(f["d_cell"][~ca], 0.0)
    state.d.x.array[:] = 0.0


def test_field_lesions_are_core_lesions():
    from dtivl.drivers.stage_cohort_runs import cohort_ids

    ids = F.field_lesions()
    assert ids and set(ids) <= set(cohort_ids(("core",)))

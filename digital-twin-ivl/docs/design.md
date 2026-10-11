# Software design and interface specification (Phase 2–4)

This file is the contract between modules. Implementers must follow the
signatures and conventions here; anything not specified is the implementer's
choice but must be documented in the module docstring.

## 0. Ground rules

* Environment: `/opt/mamba/envs/dtivl/bin/python` (DOLFINx **0.11.0**, UFL
  2026.1, basix 0.11, PETSc 3.25.6 real, gmsh 4.15.2, numpy 2.5, scipy 1.18,
  shapely, scikit-image, SALib, scikit-learn). Package import name `dtivl`
  maps to `src/` (editable install done). Run tests with
  `/opt/mamba/envs/dtivl/bin/python -m pytest tests -q`.
* DOLFINx 0.11 API differs from older tutorials. Inspect the installed API
  (`inspect.signature`, `help`) instead of relying on memory. Known facts:
  `dolfinx.io.gmsh.model_to_mesh(model, comm, rank, gdim=...)` returns a
  `MeshData` with `.mesh`, `.cell_tags`, `.facet_tags`;
  `dolfinx.fem.petsc.NonlinearProblem(F, u, *, petsc_options_prefix=..., bcs=..., J=..., petsc_options=...)`
  wraps a PETSc **SNES** (use its `.solve()` and the underlying SNES for
  bounds/VI); `dolfinx.fem.petsc.LinearProblem(a, L, *, petsc_options_prefix=..., ...)`.
  PETSc SNES type `vinewtonrsls` is available for bound-constrained solves.
* **No literature number may be typed in code.** Read every physical input via
  `dtivl.params` (`get`, `lit`, `raw`) or `MaterialSet.baseline()`. Numerical
  settings (tolerances, step counts) may be typed but must be named constants.
* Units: mm, MPa (N/mm²), N, degrees; transient module uses μs (see §6).
* Serial (MPI.COMM_WORLD with 1 rank) is sufficient; code must not break if
  run in serial only.
* Determinism: any randomness uses `numpy.random.default_rng(seed)` with the
  seed from `params.raw('design.seed')` (plus a documented offset).
* Every public function has a docstring; every module has a test file in
  `tests/` named `test_<module>.py`. Mark tests taking > 60 s with
  `@pytest.mark.slow`.

## 1. Conventions (already implemented)

* `dtivl.geometry.labels`: polar label values `LABEL_OTHER=0, LABEL_LUMEN=1,
  LABEL_CALCIUM=4, LABEL_LIPID=5`; FE cell tags `TAG_WALL=1, TAG_FIBROUS=2,
  TAG_LIPID=3, TAG_CALCIUM=4`; facet tags `TAG_LUMEN_BND=10,
  TAG_OUTER_BND=20, TAG_END_BND=30`.
* `dtivl.geometry.polar.PolarLabelMap(labels[n_theta,n_r] uint8, dr, r_out[n_theta], r_iel[n_theta]|None, meta)`.
  Row i ↔ θ_i = 2πi/n_theta (CCW from +x), column j ↔ r_j = (j+0.5)·dr.
  Polar centre = lumen centre = Cartesian origin.
* `dtivl.geometry.spec.LesionSpec` (see docstring for the geometric
  definitions of each morphology).
* `dtivl.params`, `dtivl.materials.constitutive.MaterialSet` and the UFL
  energies `psi_yeoh`, `psi_neo_hookean`, `psi_calcium_split`,
  `soft_energy`, `degradation`, `deformation_gradient`, `linear_stress`.

## 2. Geometry and cohort (`src/geometry/`)

* `build.py`: `lesion_polygons(spec) -> dict` with shapely geometries
  `'outer'` (Polygon), `'iel'` (Polygon), `'lumen'` (Polygon; for nodules the
  lumen is the circle minus the covered protrusion), `'calcium'`
  (MultiPolygon), `'lipid'` (MultiPolygon, possibly empty). Circles sampled
  with ≥ 1440 points. Rounded calcium ends as specified in `LesionSpec`.
* `rasterize.py`: `rasterize_polar(spec, n_theta=None, dr=None) -> PolarLabelMap`
  (defaults from `params.raw('discretisation.*')`). Labels by point-in-polygon
  of each polar sample centre (vectorised `shapely.contains_xy`). `r_out` and
  `r_iel` per A-line by exact ray–circle intersection.
* `descriptors.py`: `descriptors(plm, length_mm=None) -> dict` computed **only
  from the label map** (so real segmentations get identical treatment):
  `ca_arc_total_deg` (A-lines containing calcium × Δθ), `ca_arc_max_contig_deg`
  (longest contiguous run, wrap-around aware), `ca_thickness_max/min/mean`
  (per-A-line calcium sample count × dr, over A-lines with calcium),
  `ca_depth_min/mean` (lumen boundary to first calcium sample along A-line),
  `ca_area`, `lumen_area`, `n_deposits` (connected components, wrap-aware),
  `lipid_arc_deg`, `cap_thickness_min` (if lipid), `plaque_ecc_index`
  ((t_max−t_min)/t_max of intima thickness lumen→r_iel), and the Fujino score
  (below). Areas use polar pixel areas r·dr·Δθ.
* `fujino.py`: `fujino_score(max_angle_deg, max_thickness_mm, length_mm) -> int`
  with thresholds read from `params.lit('fujino_angle_threshold')`,
  `...thickness...`, `...length...`; 2/1/1 points for strictly greater than.
  `length_mm=None` → length criterion not scored (2D; document as
  "cross-sectional score, max 3"). The Fujino angle uses `ca_arc_total_deg`.
* `cohort.py`:
  * `core_specs() -> list[LesionSpec]` — full factorial
    `design.core_factorial` (arc × thickness × morphology = 32), fixed geometry
    from `geometry.*` config; lumen radius from lumen area
    (`geometry.lumen_area_mm2`, circle). Depth: superficial morphologies and
    nodules use `depth_superficial_mm`, deep sheets `depth_deep_mm`.
    Fragmented deposits use the superficial depth. IDs like
    `C-A270-T0.4-NOD`.
  * `pilot_specs()` — from `design.pilot_2d`; `example3d_spec()` — from
    `design.fujino4_example_3d` (with `ca_length`).
  * `holdout_specs(n=None, seed=None)` — scipy.stats.qmc LatinHypercube over
    ranges in `design.holdout_lhs.ranges` (to be added to the YAML by the
    integrator; implement reading whatever keys exist, with a clear error).
  * `build_cohort(specs, out_dir) -> pandas.DataFrame`: writes
    `out_dir/masks/<id>.npz` (PolarLabelMap.save), `out_dir/previews/<id>.png`
    (Cartesian label image), and returns/writes `out_dir/cohort.csv` with one
    row per lesion: all spec fields + all descriptors + Fujino score.
  * Use `allow_pending=True` only behind an explicit argument; default must
    refuse PENDING config values.

## 3. Meshing (`src/meshing/`)

* `mask_to_mesh.py`:
  `mesh_from_labelmap(plm, out_msh, *, h_ca=None, h_soft=None, r_iel_default=None) -> dict`
  builds a **conforming** 2D partition from the label map:
  1. Region boundaries from the label map (e.g. Cartesian resampling with
     `PolarLabelMap.to_cartesian`, `skimage.measure.find_contours` at level
     0.5, shapely polygons, `simplify(~0.5·dr)`, `make_valid`). The outer
     boundary comes from `r_out`, the IEL from `r_iel` (if `None`, use
     `r_out − wall_thickness` from config and document it).
  2. gmsh OCC geometry (polyline or spline loops), `occ.fragment` to get a
     conforming partition, classify each surface by an interior point
     (shapely `point_on_surface`) into lumen (removed), calcium, lipid,
     fibrous (inside IEL), wall (outside IEL).
  3. Physical groups: cell tags `TAG_*`, facet tags `TAG_LUMEN_BND`,
     `TAG_OUTER_BND`. Size field: `h_ca` in calcium and within ~2·h_ca of
     it, growing to `h_soft` elsewhere (defaults from
     `discretisation.h_calcium_mm`, `h_soft_mm`). Linear triangles.
  4. Write `.msh`; return a dict with paths, element counts and per-tag
     areas.
* `load_mesh(msh_path, gdim) -> MeshData` (mesh, cell_tags, facet_tags).
* `extrude3d.py`: `mesh3d_from_labelmap(plm, ca_length, out_msh, *, margin=None, h_axial=None, ...)`:
  the 2D partition is extruded along z in three blocks `[0,m]`, `[m,m+L]`,
  `[m+L,2m+L]` (tetrahedra). Calcium-shaped volumes in the end blocks get
  `TAG_FIBROUS`; ends get `TAG_END_BND`.
* Tests: per-tag areas match polar pixel areas within 2–3 %; every cell
  tagged; lumen boundary is one closed loop; 3D volume ≈ area × length; the
  real-segmentation path works on a hand-made mask with only labels {0,1,4}
  and `r_iel=None`.

## 4. Linear predictor analysis (`src/solvers/linear.py`, `src/metrics/predictors.py`)

* `solve_linear(meshdata, ms, p_lumen, *, k_found=None, degree=2) -> LinearResult`
  small-strain plane strain (2D) or 3D elasticity; per-tag Lamé constants from
  `MaterialSet.lame(tag)` (piecewise constants via DG0 functions); lumen
  pressure `p_lumen` (MPa) on `TAG_LUMEN_BND` (traction −p n); rigid-body
  modes removed by a weak elastic foundation `k_found` on `TAG_OUTER_BND`
  (default `params.get('loads.perivascular_k_MPa_per_mm')`); 3D ends: u_z=0.
  Returns displacement, and per-cell (DG0, quadrature-averaged) arrays for
  calcium cells: max principal Cauchy stress σ1 (3D tensor incl. σ33 for
  plane strain), max principal strain ε1, SED ½σ:ε, cell area/volume,
  cell centroid.
* `predictors(linres, plm, ms) -> dict` with keys:
  `sigma1_peak, sigma1_p95, sed_mean, sed_peak, frac_eps_gt_half_epsf,
  frac_eps_gt_epsf, frac_sig_gt_half_sigc, frac_sig_gt_sigc, scf_edge,
  has_edges, min_thickness_high_stress` (area-weighted where applicable).
  Load case: lumen pressure = `params.get('loads.ivl_balloon_pressure_atm')`
  (4 atm in MPa). σ_c = `ms.sigma_crit`, ε_f = `ms.eps_f`.
  `scf_edge` = max σ1 in calcium cells within angular distance
  max(10°, 2·t_ca/r_mid) of any deposit end / area-weighted mean σ1 in
  calcium (1.0 and `has_edges=False` for closed rings).
  `min_thickness_high_stress` = min local calcium thickness (from the label
  map A-line of each cell centroid) over cells with σ1 ≥ 90th percentile.
* Verification test (`tests/test_lame.py`): thick-walled cylinder (single
  material, inner radius a, outer b, pressure p, plane strain, weak or no
  foundation) vs Lamé: radial displacement and hoop stress, **max relative
  error < 1 %** on a refined mesh. Write the numbers to
  `results/verification_lame.json` when run via the verification driver.

## 5. Nonlinear quasi-static solver with phase-field fracture (`src/solvers/nonlinear.py`, `src/fracture/`)

### 5.1 Unknowns and spaces
u: vector Lagrange P2 (2D) / P2 (3D); d: scalar P1, restricted to calcium by
bounds: upper bound 1 on DOFs of calcium cells, 0 elsewhere; lower bound
d_prev (irreversibility). Material constants as DG0 per tag or per-tag
integrals `dx(tag)`.

### 5.2 Energy and residuals
Π_int = Σ_soft ∫ W_soft(F) dx(tag) + ∫ [g(d) ψ⁺(F) + ψ⁻(F)] dx(Ca)
Π_frac = (Gc/c_w) ∫ (d/ℓ + ℓ|∇d|²) dx(Ca), c_w = 8/3 (AT1), Gc = `ms.gc(ℓ)`,
ℓ = `fracture.ell_over_h` × h_ca.
Lumen follower pressure: residual term `+∫ p J F⁻ᵀN·v dS(lumen)` (in the
convention R = ∫P:∇v dX + ∫pJF⁻ᵀN·v dS = 0 with N the outward normal of
the tissue domain).
Weak foundation on outer boundary: `½ k_f |u|²` (same k as linear).
Rigid cylinder contact (frictionless penalty, k_pen =
`loads.contact_penalty_MPa_per_mm`), with ρ = |x+u−c| (c = origin, 3D:
distance from z-axis):
* inner expanding cylinder (balloon/stent) radius R: `½ k_pen ⟨R−ρ⟩₊²` on lumen facets
* outer envelope cap (balloon nominal size) R_cap: `½ k_pen ⟨ρ−R_cap⟩₊²` on lumen facets
Contact pressure (per reference area) p_c = k_pen ⟨·⟩₊.

### 5.3 Algorithm
Alternate minimisation per load increment: (i) u-problem: SNES Newton
(`newtonls`, line search `bt` or `basic`), direct LU (MUMPS), d frozen;
(ii) d-problem with u frozen: quadratic bound-constrained problem, SNES
`vinewtonrsls` with bounds [d_prev, ub]; iterate until ‖Δd‖∞ < 1e-3 (max
~100 iterations; log non-convergence). Adaptive load stepping: halve the
increment on SNES failure (min increment documented). Record per step:
load parameter, max d, crack-cell count, lumen area, mean contact pressure,
energies.

### 5.4 Load protocols (functions returning a state + history)
* `run_pressure(state, p_target, *, cap_radius=None, n_steps)` — lumen
  pressure ramp, optional envelope cap. Physiological load:
  `loads.physiological_pressure` (MPa).
* IVL: `run_ivl(state, *, T)` = pressure to p_bal (=
  `loads.ivl_balloon_pressure_atm`) with cap R_target, then to
  p_bal + T × `loads.ivl_peak_pressure_MPa` × `loads.ivl_pulse_scale` with the
  same cap, then unload to 0 without cap. Cap radius R_ivl = ½·`geometry.ivl_balloon_diameter_mm`.
* Stent: `run_stent(state, *, K_s=None)` = inner rigid cylinder expanding from just inside
  the minimum lumen radius to R_target = ½·`geometry.stent_target_diameter_mm`; stop early if the mean contact
  pressure exceeds `loads.balloon_rated_pressure_atm` (→ under-expansion);
  then recoil: find R_final in [R_min, R_dep] with bisection on
  P_tissue(R) = K_s (R_dep − R) where K_s defaults to
  `loads.stent_radial_stiffness_MPa_per_mm` (a placeholder: the integrator
  identifies K_s so that a calcium-free lesion shows the clinical acute recoil
  `loads.stent_recoil_target_pct`; therefore K_s must be an argument) and
  P_tissue(R) is the mean contact pressure with the cylinder fixed at R.
* Damage is active in every protocol (balloon/stent can fracture calcium);
  `stent after IVL` starts from u = 0 with the post-IVL d field.

### 5.5 Outputs (`src/fracture/cracks.py`, `src/metrics/outcomes.py`)
`crack_analysis(meshdata, d, plm) -> dict` on cell-averaged d (> `fracture.d_crack`):
`fracture` (bool: any crack component connecting the luminal and abluminal
calcium surfaces), `n_fractures` (number of such through-thickness
components), `n_partial`, `max_depth_frac` (radial penetration / local
thickness), `pattern` per component (`radial` if the principal axis of the
component's cell centroids is within 45° of the radial direction, else
`circumferential`), `location_frac` (angular position of each component
relative to its deposit: 0 = centre, 1 = end), `onset_load` (first load
parameter with max d ≥ 0.9). Stent metrics: `lumen_area_pre`,
`lumen_area_post`, `lumen_gain`, `msa` (π R_final²), `stent_expansion_pct`
(msa / reference lumen area × 100, reference area = π (½·`geometry.reference_diameter_mm`)²), `acute_recoil_pct` ((R_dep − R_final)/R_dep ×
100), `malapposition_area` (lumen area outside the stent circle beyond
`loads.malapposition_gap_mm`), `malapposed_frac` (fraction of the stent
circumference with gap > threshold), `reached_target` (bool),
`max_contact_pressure_mean`.

### 5.6 Element erosion comparator (`src/fracture/erosion.py`)
Same u-solver without phase field; after each converged increment, cells of
calcium with cell-averaged max principal Cauchy stress ≥ `ms.sigma_crit`
(strain mode: equivalently ε1 ≥ ε_f) are eroded (energy scaled by k_res);
re-solve at the same load until no new erosion. Same crack post-processing
with eroded cells as "cracked".

### 5.7 Verification tests
* AT1 homogeneous bar/plate (calcium only, uniaxial tension, ends displaced):
  damage onset at ψ⁺ = 3Gc/(16ℓ) within 5 %, i.e. at the calibrated
  ε_f (strain mode) — reported in `results/verification_phasefield.json`.
* Hyperelastic thick cylinder (single soft material) under pressure: compare
  small-load response to Lamé (< 1 %).
* Contact: homogeneous tube expanded by a rigid cylinder to R: lumen nodes
  in contact lie at ρ = R within penalty penetration p_c/k_pen.

## 6. Transient IVL pulse (`src/solvers/transient.py`)
Linear elastodynamics (small strain; μs time scale) in units mm, MPa, μs.
In this system a density in kg/m³ has the same numerical value in
N·μs²/mm⁴, and wave speeds come out in mm/μs (= km/s); e.g. M = 2200 MPa,
ρ = 1000 → c = 1.483 mm/μs = 1483 m/s. Time integration: Newmark (average
acceleration) or generalized-α. Per-tag shear modulus from `MaterialSet.lame`, bulk modulus
from the measured sound speed c (`lit('calcification_sound_speed_20MHz')`
etc., M = ρc², K = M − 4μ/3) and densities from config (`materials.density.*`,
PENDING until verified). Load: lumen pressure pulse p(t) (half-sine, width
from `lit('ivl_pulse_duration')`, peak `loads.ivl_peak_pressure_MPa`), outer
boundary non-reflecting (dashpot ρc). Output: peak tensile σ1 in calcium over
the simulated window vs the static linear σ1 for the same peak pressure:
`T = σ1_peak,transient / σ1_static` per lesion. Test: 1D/2D wave speed in a
homogeneous medium matches √(M/ρ) within 2 %.

## 7. Result files
All drivers write JSON/CSV under `results/` with lesion id keys. Field names
above are the column names. No plotting in core modules.

"""Section 2, Materials and Methods: 2.1-2.9 (2.10 statistics: s2_statistics.py; 2.11 software: s2_software.py).

Every number goes through ``ctx``: configuration and literature values through
the params layer, derived material values through ``MaterialSet.baseline()``
(recorded with the config/literature files as source), numerical settings of
the solvers through the named module constants (recorded with their module),
and counts and verification values through the results files.
"""
from __future__ import annotations

from build_manuscript import Ctx, Heading, Para

CFG = "params/study_config.yaml"
LIT = "params/literature_parameters.csv"
MS_SRC = "params/study_config.yaml + params/literature_parameters.csv (MaterialSet.baseline)"


def _ms():
    from dtivl.materials.constitutive import MaterialSet
    return MaterialSet.baseline()


def _const(ctx: Ctx, module: str, name: str, fmt: str = "g", scale: float = 1.0) -> str:
    import importlib
    mod = importlib.import_module(module)
    return ctx.num(getattr(mod, name), src=f"src/{module.removeprefix('dtivl.').replace('.', '/')}.py",
                   key=name, fmt=fmt, scale=scale)


def _levels(ctx: Ctx, factor: str, fmt: str = "g") -> str:
    from dtivl import params as P
    vals = P.raw(f"design.core_factorial.{factor}")
    txt = [ctx.num(v, src=CFG, key=f"design.core_factorial.{factor}[{i}]", fmt=fmt) for i, v in enumerate(vals)]
    return ", ".join(txt[:-1]) + " and " + txt[-1]


def _nlev(ctx: Ctx, factor: str) -> str:
    from dtivl import params as P
    return ctx.count(len(P.raw(f"design.core_factorial.{factor}")), src=CFG, key=f"len(design.core_factorial.{factor})")


def _fpts(ctx: Ctx, criterion: str) -> str:
    """Points of a Fujino criterion, parsed by dtivl.geometry.fujino from the literature row notes."""
    from dtivl.geometry.fujino import fujino_criteria
    row = {"angle": "fujino_angle_threshold", "thickness": "fujino_thickness_threshold"}[criterion]
    return ctx.num(fujino_criteria()[criterion].points, src=LIT, key=f"{row}.notes[points]", fmt="d")


def _fmax() -> int:
    from dtivl.geometry.fujino import fujino_max_score
    return fujino_max_score(False)


def _atm(ctx: Ctx, key: str) -> str:
    """Pressure in atm as tabulated in the literature row behind a config entry."""
    from dtivl import params as P
    row = P.raw(key)["lit"]
    p = P.load_literature()[row]
    return ctx.num(float(p.value_str), src=LIT, key=f"{row}.value [{p.unit}, as tabulated]", fmt="g")


def s21(ctx: Ctx) -> list:
    return [
        Heading("2. Materials and Methods", 1),
        Heading("2.1. Study Design", 2),
        Para(
            "This is an in silico study; no patient data were used. A structured cohort of calcified coronary "
            "cross-sections was generated from polar label maps, and every lesion was analysed by two separate "
            f"finite element models ({ctx.fig('fig1_design')}). The predictor model is a cheap pre-treatment "
            "analysis: the lesion geometry and one linear-elastic solve of the undamaged lesion under the IVL "
            "balloon pressure. The outcome model is a separate nonlinear simulation of the treatment with "
            "phase-field damage in the calcium: the IVL balloon alone, a plain balloon, IVL, and stent "
            "deployment with and without prior IVL. All predictors were computed from the predictor model and "
            "the geometry only, and all outcomes from the outcome model only. The two models share the lesion "
            "geometry and the material constants, including the calcium modulus; no quantity computed by the "
            "outcome model is used as a predictor. In particular, the IVL load factor of the transient pulse "
            "analysis (Section 2.5) is an input of the outcome model and is excluded from the candidate "
            "predictors by a check in the code. Every model input is listed with its value, source and status "
            f"in {ctx.tab('table1_parameters')}; inputs that could not be verified in a primary source were not "
            "used in the main analysis."),
    ]


def s22(ctx: Ctx) -> list:
    from dtivl import params as P

    c = ctx.cite
    cs = "results/cohort_summary.json"
    h = P.raw("design.holdout_lhs")
    rng = h["ranges"]
    return [
        Heading("2.2. Lesion Cohort", 2),
        Para(
            f"The core set is a full factorial of calcium arc ({_levels(ctx, 'arc_deg')}°), calcium thickness "
            f"({_levels(ctx, 'thickness_mm')} mm) and morphology (superficial sheet, deep sheet, calcified "
            f"nodule and fragmented calcium) {c('sugiyama2019calcified', 'mintz2015imaging')}, giving "
            f"{ctx.v(cs, 'n_per_set.core', fmt='d')} lesions ({ctx.fig('fig2_cohort_gallery')}). All core "
            "lesions share one reference geometry: a circular lumen of area "
            f"{ctx.param('geometry.lumen_area_mm2', fmt='.3g', scale=1e6)} mm² {c('hill2020disruptcad3')}, an "
            "external elastic membrane at the reference vessel radius "
            f"(diameter {ctx.param('geometry.reference_diameter_mm', fmt='.3g', scale=1e3)} mm) "
            f"{c('yamada2025distribution')}, and a wall (media and adventitia) of "
            f"{ctx.param('geometry.wall_thickness_mm', fmt='.3g', scale=1e3)} mm "
            f"{c('konishi2023media', 'rioufol2006adventitia')}, which gives a plaque burden of "
            f"{ctx.v(cs, 'plaque_burden_check.implied_plaque_burden_pct', fmt='.1f')} %. Superficial sheets, "
            "nodules and fragments start at a depth of "
            f"{ctx.param('geometry.depth_superficial_mm', fmt='.2g', scale=1e3)} mm below the lumen and deep "
            f"sheets at {ctx.param('geometry.depth_deep_mm', fmt='.2g', scale=1e3)} mm (the mean minus and plus "
            f"one standard deviation of an ex vivo OCT series) {c('fdoct_calcium_volumetric')}. A nodule adds a "
            f"protrusion of height {ctx.param('geometry.nodule_height_mm', fmt='.2g', scale=1e3)} mm over a base "
            f"arc of {ctx.param('geometry.nodule_base_arc_deg', fmt='g')}° into the lumen; fragmented calcium is "
            f"split into one deposit per {ctx.param('geometry.fragment_pitch_deg', fmt='g')}° of arc, each "
            f"filling {ctx.param('geometry.fragment_fill', fmt='.0f', scale=100.0)} % of its pitch. Where "
            "calcium, depth and a minimum fibrous layer of "
            f"{ctx.param('geometry.calcium_iel_clearance_mm', fmt='.2g', scale=1e3)} mm above the internal "
            "elastic lamina did not fit within the reference plaque, the plaque was grown outward with the lumen "
            f"unchanged ({ctx.v(cs, 'plaque_growth.n_grown.core', fmt='d')} core lesions; decision D11)."),
        Para(
            f"The hold-out set of {ctx.v(cs, 'n_per_set.holdout', fmt='d')} lesions was drawn by Latin hypercube "
            f"sampling (seed {ctx.raw('design.seed', fmt='d')}) over continuous ranges of calcium arc "
            f"({ctx.num(rng['arc_deg'][0], src=CFG, key='design.holdout_lhs.ranges.arc_deg[0]', fmt='g')}–"
            f"{ctx.num(rng['arc_deg'][1], src=CFG, key='design.holdout_lhs.ranges.arc_deg[1]', fmt='g')}°), "
            f"thickness ({ctx.num(rng['thickness_mm'][0], src=CFG, key='design.holdout_lhs.ranges.thickness_mm[0]', fmt='g')}–"
            f"{ctx.num(rng['thickness_mm'][1], src=CFG, key='design.holdout_lhs.ranges.thickness_mm[1]', fmt='g')} mm), "
            "depth, lumen area and plaque eccentricity, with the four morphologies in a balanced cycle and a lipid "
            "pool with a fibrous cap in half of the lesions; the lumen-area and depth ranges are the published "
            f"mean ± one standard deviation {c('hill2020disruptcad3', 'fdoct_calcium_volumetric')}. Thresholds and "
            "models derived on the core set were applied to the hold-out set without change."),
        Para(
            "Each lesion was stored as a polar label map in the intravascular OCT convention (A-lines × radial "
            f"samples; {ctx.raw('discretisation.polar_n_theta', fmt='d')} A-lines, "
            f"{ctx.num(P.raw('discretisation.polar_dr_mm') * 1e3, src=CFG, key='discretisation.polar_dr_mm x 1000', fmt='g')} µm "
            f"radial sampling) {c('gharaibeh2019spie', 'lee2020twostep')}, and all geometric descriptors were "
            "computed from the label map only, as they would be from a segmented image: total and largest "
            "contiguous calcium arc, maximum calcium thickness, depth, calcium area, number of deposits, lumen "
            "area, lipid arc and plaque eccentricity. The cross-sectional Fujino score awards "
            f"{_fpts(ctx, 'angle')} points for an arc above {ctx.lit('fujino_angle_threshold', fmt='g')}° and "
            f"{_fpts(ctx, 'thickness')} point for a thickness above "
            f"{ctx.lit('fujino_thickness_threshold', fmt='g')} mm {c('fujino2018score')}; in 2D the length "
            "criterion cannot be scored, so the score is at most "
            f"{ctx.num(_fmax(), src=LIT, key='sum of the cross-sectional fujino_*_threshold points', fmt='d')}."),
    ]


def s23(ctx: Ctx) -> list:
    from dtivl import params as P

    c = ctx.cite
    ex = P.raw("discretisation.example3d")
    l3 = P.raw("design.length_series_3d")
    return [
        Heading("2.3. Geometry and Meshing", 2),
        Para(
            "The label map was converted into a conforming partition of lumen, calcium, lipid, fibrous intima "
            "(inside the internal elastic lamina) and wall, and meshed with linear triangles in Gmsh "
            f"{c('geuzaine2009gmsh')}. The element size was "
            f"{ctx.num(P.raw('discretisation.h_calcium_mm') * 1e3, src=CFG, key='discretisation.h_calcium_mm x 1000', fmt='g')} µm "
            "in and next to the calcium and grew to "
            f"{ctx.num(P.raw('discretisation.h_soft_mm'), src=CFG, key='discretisation.h_soft_mm', fmt='g')} mm in "
            "the soft tissue. Displacements were interpolated with quadratic (P2) and the damage field with linear "
            "(P1) Lagrange elements, under plane strain. For the three-dimensional length series (Section 2.9), "
            "the cross-section was extruded over the calcified length with calcium-free end segments of "
            f"{ctx.param('geometry.margin_3d_mm', fmt='g', scale=1e3)} mm, using in-plane sizes of "
            f"{ctx.num(ex['h_calcium_mm'], src=CFG, key='discretisation.example3d.h_calcium_mm', fmt='g')} mm "
            f"(calcium) and {ctx.num(ex['h_soft_mm'], src=CFG, key='discretisation.example3d.h_soft_mm', fmt='g')} mm "
            "(soft tissue) and axial sizes of "
            f"{ctx.num(l3['h_axial_ca_mm'], src=CFG, key='design.length_series_3d.h_axial_ca_mm', fmt='g')} mm "
            f"(calcified block) and {ctx.num(l3['h_axial_margin_mm'], src=CFG, key='design.length_series_3d.h_axial_margin_mm', fmt='g')} mm "
            "(end blocks)."),
    ]


def s24(ctx: Ctx) -> list:
    from dtivl import params as P

    c = ctx.cite
    ms = _ms()
    k = ms.k_floor
    return [
        Heading("2.4. Constitutive Models", 2),
        Para(
            "The wall (media and adventitia, lumped) was modelled as an isotropic Yeoh solid with constants from "
            f"human coronary layers {c('escuer2019restenosis')}. The fibrous intima and the lipid pool were "
            "nearly incompressible neo-Hookean solids with small-strain Young moduli of "
            f"{ctx.num(ms.fibrous_E, src=MS_SRC, key='fibrous_E', fmt='.3g')} MPa {c('loree1994modulus')} and "
            f"{ctx.num(ms.lipid_E * 1e3, src=MS_SRC, key='lipid_E x 1000', fmt='.2g')} kPa {c('jsb2011afm')}; "
            "the soft tissues used a bulk-to-shear modulus ratio of "
            f"{ctx.num(ms.k_over_mu, src=MS_SRC, key='k_over_mu', fmt='g')} (an assumption, varied in the "
            "sensitivity analysis). Calcium was linear elastic and brittle, with a Poisson ratio of "
            f"{ctx.num(ms.ca_nu, src=MS_SRC, key='ca_nu', fmt='g')} {c('cvet2014calcmodulus')}."),
        Para(
            "The calcium modulus is the most uncertain input: published values span more than three orders of "
            f"magnitude, from {ctx.lit('calcified_tissue_tangential_modulus', fmt='.2g')} MPa for tissue-level "
            f"tension {c('loree1994modulus')} to the gigapascal range from nanoindentation "
            f"{c('cvet2014calcmodulus')}. The baseline was set to "
            f"{ctx.num(ms.ca_E, src=MS_SRC, key='ca_E', fmt='.0f')} MPa (decision D8), the point of the pilot "
            "modulus sweep at which the IVL balloon inflated alone did not damage the three pilot lesions while "
            "IVL did. This is a calibration to a criterion internal to the model, not to a measured property. "
            "With the failure strain below, it implies a calcium failure stress of "
            f"{ctx.num(ms.sigma_crit, src=MS_SRC, key='sigma_crit = ca_E x eps_f', fmt='.2g')} MPa, above the "
            f"tissue-level strength of {ctx.num(ms.sigma_c, src=MS_SRC, key='sigma_c', fmt='.2g')} MPa measured "
            f"in tension {c('holzapfel2004plaque')}, and at this modulus a plain balloon at "
            f"{_atm(ctx, 'loads.poba_pressure_atm')} atm still fractured part of the pilot lesions, unlike in "
            f"cadaver lesions {c('kawai2023cadaver')}. The modulus was therefore varied in the "
            "sensitivity analysis (Section 2.9)."),
    ]


def s25(ctx: Ctx) -> list:
    from dtivl import params as P

    c = ctx.cite
    cal = "results/calibration_stent.json"
    return [
        Heading("2.5. Boundary Conditions and Load Protocols", 2),
        Para(
            "The lumen pressure acted as a follower load on the luminal surface. Rigid-body motion was removed by "
            "a weak elastic foundation on the outer boundary "
            f"({ctx.param('loads.perivascular_k_MPa_per_mm', fmt='g')} MPa/mm; its effect on the predictors is "
            f"reported in {ctx.tab('table2_verification_consistency')}). Balloons and the stent were rigid "
            "cylinders with frictionless penalty contact (penalty "
            f"{ctx.param('loads.contact_penalty_MPa_per_mm', fmt='g')} MPa/mm). Five protocols were run on every "
            "lesion. (i) Physiological pressure "
            f"({ctx.param('loads.physiological_pressure', fmt='.3g', scale=1e3)} kPa) {c('whelton2017bp')}. "
            f"(ii) The IVL balloon alone: lumen pressure to {_atm(ctx, 'loads.ivl_balloon_pressure_atm')} atm "
            f"{c('tovarforero2019system')} inside a cap at the radius of the IVL balloon "
            f"(diameter {ctx.param('geometry.ivl_balloon_diameter_mm', fmt='.3g', scale=1e3)} mm) "
            f"{c('yamada2025distribution')}. (iii) A plain balloon at {_atm(ctx, 'loads.poba_pressure_atm')} atm "
            f"with the same cap, the balloon arm of a cadaver comparison {c('kawai2023cadaver')}. (iv) IVL "
            "(below). (v) Stent deployment without and (vi) after IVL: a rigid cylinder expanded from just inside "
            "the smallest lumen radius to the target stent diameter "
            f"({ctx.param('geometry.stent_target_diameter_mm', fmt='.3g', scale=1e3)} mm) "
            f"{c('yamada2025distribution')}, stopping early if the mean contact pressure reached the rated "
            f"balloon pressure ({_atm(ctx, 'loads.balloon_rated_pressure_atm')} atm) {c('stent_specs')}; the "
            "stent then recoiled to the radius at which the tissue pressure balanced a linear stent stiffness. "
            "The stiffness ("
            f"{ctx.v(cal, 'K_s_MPa_per_mm', fmt='.3g')} MPa/mm) was identified so that a calcium-free version of "
            "a pilot lesion showed the clinical acute recoil of "
            f"{ctx.param('loads.stent_recoil_target_pct', fmt='g')} % {c('recoil_ees_bvs')}. Damage was active "
            "in every protocol; stenting after IVL started from the IVL damage field. Loads were applied in "
            "increments that were halved on non-convergence, up to "
            f"{_const(ctx, 'dtivl.solvers.nonlinear', 'MAX_HALVINGS', fmt='d')} times."),
        Para(
            "IVL was represented by its quasi-static equivalent (decision D14). The balloon was inflated to "
            f"{_atm(ctx, 'loads.ivl_balloon_pressure_atm')} atm inside the cap, the pressure was raised by one "
            "pulse-equivalent increment to the peak $p_{top} = p_{bal} + T_{eq}\\,p_{peak}\\,s$, and the lesion "
            f"was unloaded. Here $p_{{peak}}$ = {ctx.param('loads.ivl_peak_pressure_MPa', fmt='g')} MPa is the "
            f"peak positive pressure of the pulse {c('kereiakes2021principles')}, $s$ = "
            f"{ctx.param('loads.ivl_pulse_scale', fmt='g')} a pulse-amplitude scale (varied in the sensitivity "
            "analysis), and $T_{eq}$ a lesion-specific transfer factor from a linear elastodynamic analysis of "
            "one pulse. That analysis applied a half-sine lumen pressure pulse "
            f"{c('brinton2019feasibility')} to the lesion with a non-reflecting outer boundary, used the "
            "measured sound speeds of the tissues to set their bulk moduli "
            f"{c('rooney_soundspeed', 'bjmbr2002coronarysound', 'umb1998acoustic')} with densities from modelling "
            f"studies {c('density_drilling2019', 'density_stentfe2014', 'density_thermal2007')} (secondary "
            "sources, decision D15), and integrated in time with the non-dissipative generalised-α scheme "
            "and a mixed displacement-pressure formulation. $T_{eq}$ is the peak tensile principal stress in the "
            "calcium during the pulse divided by the peak of the same quantity in the quasi-static model under the "
            "same static pressure, so that the pulse-equivalent increment reproduces the transient peak. Repeated "
            "pulses and fatigue were not modelled."),
    ]


def s26(ctx: Ctx) -> list:
    from dtivl import params as P

    c = ctx.cite
    ms = _ms()
    ell = P.get("fracture.ell_over_h").value * P.raw("discretisation.h_calcium_mm")
    return [
        Heading("2.6. Fracture Model", 2),
        Para(
            "Calcium damage was modelled with the AT1 variational phase-field model of brittle fracture "
            f"{c('bourdin2000numerical', 'tanne2018nucleation')}. The damage $d$ ($d = 0$ intact, $d = 1$ broken) "
            "lives in the calcium only and is irreversible. The calcium strain energy was split into volumetric-deviatoric "
            "tensile and compressive parts, of which only the tensile part $\\psi^+$ is degraded by "
            "$g(d) = (1-k)(1-d)^2 + k$, and the fracture energy is $\\frac{3 G_c}{8}\\int (d/\\ell + "
            "\\ell |\\nabla d|^2)\\,dx$. The residual stiffness $k$ was set so that fully damaged calcium keeps "
            "the shear stiffness of the surrounding fibrous matrix (decision D9; "
            f"$k$ = {ctx.num(ms.k_floor, src=MS_SRC, key='k_floor = mu_fibrous / mu_Ca', fmt='.2g')}), and "
            "$G_c = 16\\,\\ell\\,(1-k)\\,\\psi_{crit}/3$ places the AT1 damage onset at the critical energy "
            "$\\psi_{crit}$ of a uniaxial failure strain $\\varepsilon_f$ (decision D5). The failure strain "
            f"$\\varepsilon_f$ = {ctx.num(ms.eps_f, src=MS_SRC, key='eps_f = calcification_fracture_stretch - 1', fmt='.3g')} "
            f"is the measured failure stretch of calcified tissue minus one {c('holzapfel2004plaque')}. The "
            "regularisation length was "
            f"$\\ell$ = {ctx.num(ell * 1e3, src=CFG, key='fracture.ell_over_h x discretisation.h_calcium_mm x 1000', fmt='g')} µm, "
            f"{ctx.param('fracture.ell_over_h', fmt='g')} times the calcium element size. Each load increment was "
            f"solved by alternate minimisation {c('miehe2010phasefield')}: a Newton solve for the displacement "
            "with damage fixed, then a bound-constrained solve for the damage with displacement fixed, repeated "
            "until the largest damage change fell below "
            f"{_const(ctx, 'dtivl.solvers.nonlinear', 'STAGGER_TOL', fmt='g')} or "
            f"{_const(ctx, 'dtivl.solvers.nonlinear', 'STAGGER_MAX_IT', fmt='d')} iterations were reached; "
            "increments that reached the iteration limit were counted and are reported."),
        Para(
            "As a comparator with a different fracture model (decision D13), IVL was also simulated with element "
            "erosion: after each converged increment, calcium cells whose cell-averaged maximum principal strain "
            "reached $\\varepsilon_f$ lost their tensile stiffness, and the increment was re-solved until no "
            "further cells eroded. The comparator was run on "
            f"{ctx.count(_n_erosion(), src=CFG, key='size of design.erosion_subset', label='erosion subset size')} "
            "core lesions (two arcs, the thinnest and thickest levels, a sheet and a nodule)."),
    ]


def _n_erosion() -> int:
    from dtivl import params as P
    e = P.raw("design.erosion_subset")
    return len(e["arc_deg"]) * len(e["thickness_mm"]) * len(e["morphology"])


def s27(ctx: Ctx) -> list:
    from dtivl import params as P

    return [
        Heading("2.7. Outcome Definitions", 2),
        Para(
            "Calcium was considered damaged where the cell-averaged damage exceeded "
            f"{ctx.raw('fracture.d_crack', fmt='g')}. The damage-onset load of a protocol is the lumen pressure "
            "of the first load step at which the maximum damage reached this value. A damaged component that "
            "connects the luminal and abluminal calcium surfaces is a through-thickness fracture; other damaged "
            "components are partial. A component covering at least half of its calcium deposit was labelled "
            "diffuse; it is still a through-thickness fracture when it connects the two surfaces. The primary "
            "outcome (decision D10) is through-thickness damage under IVL; secondary outcomes are the "
            "damage-onset load under IVL, through-thickness damage under the IVL balloon alone and under the "
            "plain balloon, and stent expansion with and without prior IVL. Stent expansion is the stent area "
            "after recoil divided by the reference lumen area (the area of a circle of the reference vessel "
            "diameter), in per cent; acute recoil is the relative loss of stent radius between deployment and "
            "recoil."),
    ]


def s28(ctx: Ctx) -> list:
    from dtivl import params as P

    nl = P.get("fracture.ell_over_h").value * P.raw("discretisation.h_calcium_mm")
    return [
        Heading("2.8. Pre-Treatment Mechanics Predictors", 2),
        Para(
            "The predictor model is a small-strain, plane-strain linear-elastic solve of the undamaged lesion "
            f"under a lumen pressure of {_atm(ctx, 'loads.ivl_balloon_pressure_atm')} atm (the IVL balloon "
            "pressure), with the material constants of Section 2.4 linearised at zero strain. From the calcium "
            "cells it gives the maximum principal stress $\\sigma_1$ (including the out-of-plane stress), the "
            "maximum principal strain $\\varepsilon_1$ and the strain energy density (SED). The predictors were "
            f"the area-weighted {_const(ctx, 'dtivl.metrics.predictors', 'SIGMA1_PERCENTILE', fmt='g')}th and "
            f"{_const(ctx, 'dtivl.metrics.predictors', 'SIGMA1_PERCENTILE_HIGH', fmt='g')}th percentiles and mean of "
            "$\\sigma_1$; the mean SED; non-local peaks "
            "of $\\sigma_1$ and SED, in which each cell value is replaced by the area-weighted mean over the "
            "calcium within a radius equal to the phase-field length "
            f"({ctx.num(nl * 1e3, src=CFG, key='fracture.ell_over_h x discretisation.h_calcium_mm x 1000', fmt='g')} µm) "
            "before the maximum is taken; the calcium area fractions with $\\varepsilon_1$ above $\\varepsilon_f$ "
            "and $\\varepsilon_f/2$ and with $\\sigma_1$ above the failure stress and half of it; an edge stress "
            "concentration factor (the non-local peak $\\sigma_1$ near deposit ends over the mean $\\sigma_1$ of "
            "the calcium, within an angular window of at least "
            f"{_const(ctx, 'dtivl.metrics.predictors', 'EDGE_MIN_WINDOW_DEG', fmt='g')}° around each end); and the "
            "smallest local calcium thickness among the cells above the "
            f"{_const(ctx, 'dtivl.metrics.predictors', 'HIGH_STRESS_PERCENTILE', fmt='g')}th percentile of "
            "$\\sigma_1$. Raw cell peaks were not used as predictors because they did not converge under mesh "
            f"refinement ({ctx.tab('table2_verification_consistency')})."),
    ]


def s29(ctx: Ctx) -> list:
    from dtivl import params as P

    c = ctx.cite
    sens = P.raw("design.sensitivity")
    n_les = len(sens["morris_lesions"])
    n_inp = None
    try:
        from dtivl.drivers import stage_sensitivity as S
        n_inp = len(S.morris_inputs())
        n_runs = S.n_morris_runs()
        n_oat = S.n_oat_runs()
    except Exception:   # pragma: no cover - the stage module is part of the package
        raise
    sob = sens["sobol"]
    n_sob_inp = sum(1 for v in sob["inputs"].values() if v["value"])
    l3 = P.raw("design.length_series_3d")
    return [
        Heading("2.9. Verification, Consistency and Sensitivity Analyses", 2),
        Para(
            "Verification compared the solvers with analytical solutions: linear elasticity of a thick cylinder "
            "under pressure (Lamé solution), the small-load response of the hyperelastic solver, penalty contact "
            "with a rigid cylinder, and the AT1 damage onset under uniform strain. Discretisation error of the "
            "predictors was estimated on two lesions at three mesh levels with the grid convergence index "
            f"{c('celik2008gci')}. Criteria and results are listed in {ctx.tab('table2_verification_consistency')}."),
        Para(
            "Consistency with published observations was assessed by comparing model outcomes with reported "
            "clinical and experimental observations made in other lesions: the direction of the calcium arc and "
            "thickness effects on fracture, the lower fracture rate of nodules, lesion-level fracture rates after "
            "IVL, the absence of fracture under a plain balloon in cadaver lesions, stent expansion and recoil, the "
            "number of fractures per lesion and the association of the Fujino score with expansion. Each check has "
            "a criterion fixed in the configuration and a label (agrees, partial, disagrees, not computed). These "
            "comparisons test plausibility, not the accuracy of the predictions for any lesion."),
        Para(
            "Sensitivity analysis had four parts. One-at-a-time scenarios on the three pilot lesions "
            f"({ctx.count(n_oat, src='src/drivers/stage_sensitivity.py', key='n_oat_runs()')} runs) removed the "
            "crack filler (D9), changed the phase-field length, the pulse scale, the foundation stiffness and "
            "the soft-tissue bulk modulus, replaced the damage driving force by a Rankine-type principal-strain "
            "force (D10), lowered the calcium density (D15) and switched to a stress-based failure criterion (D5). "
            f"Morris elementary effects {c('herman2017salib')} screened "
            f"{ctx.count(n_inp, src='src/drivers/stage_sensitivity.py', key='len(morris_inputs())')} inputs of the "
            f"outcome model on {ctx.count(n_les, src=CFG, key='len(design.sensitivity.morris_lesions)')} core "
            f"lesions ({ctx.count(sens['morris_trajectories'], src=CFG, key='design.sensitivity.morris_trajectories')} "
            f"trajectories, {ctx.count(n_runs, src='src/drivers/stage_sensitivity.py', key='n_morris_runs()')} runs). "
            f"First-order and total Sobol indices {c('saltelli2010variance', 'herman2017salib')} of the "
            f"{ctx.count(n_sob_inp, src=CFG, key='number of design.sensitivity.sobol.inputs set true')} inputs that "
            "enter the predictor model were computed for every predictor on the same lesions. Finally, a "
            "three-dimensional length series (decision D12) extruded "
            f"{ctx.count(len(l3['cross_sections']), src=CFG, key='len(design.length_series_3d.cross_sections)')} "
            "core cross-section over calcified lengths of "
            + ", ".join(ctx.num(v, src=CFG, key=f'design.lengths_3d_mm[{i}]', fmt='g')
                        for i, v in enumerate(P.raw('design.lengths_3d_mm'))) +
            " mm and loaded it with the IVL ramp up to damage onset, to compare the onset load with the 2D model "
            "at the same in-plane resolution."),
    ]


def blocks(ctx: Ctx) -> list:
    return s21(ctx) + s22(ctx) + s23(ctx) + s24(ctx) + s25(ctx) + s26(ctx) + s27(ctx) + s28(ctx) + s29(ctx)

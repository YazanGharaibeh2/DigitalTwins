"""Section 2, Materials and Methods: 2.1-2.9 (2.10 statistics: s2_statistics.py; 2.11 software: s2_software.py).

Every number goes through ``ctx``: configuration and literature values through
the params layer (base units: mm, MPa, s), derived material values through
``MaterialSet.baseline()`` (recorded with the config/literature files as
source), numerical settings of the solvers through the named module constants
(recorded with their module), and counts through the files they are counted in.
Revised 2026-10-15 after an adversarial fact check against the code and config.
"""
from __future__ import annotations

from build_manuscript import Ctx, Heading, Para

import tables_ms

CFG = "params/study_config.yaml"
LIT = "params/literature_parameters.csv"
CS = "results/cohort_summary.json"
COH = "cohort/cohort.csv"
MS_SRC = "params/study_config.yaml + params/literature_parameters.csv (MaterialSet.baseline)"


# ------------------------------------------------------------------ helpers
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


def _atm_mpa(ctx: Ctx, key: str) -> str:
    """'<x> atm (<y> MPa)' for a pressure config entry."""
    return f"{_atm(ctx, key)} atm ({ctx.param(key, fmt='.3g')} MPa)"


def _n(ctx: Ctx, n: int, src: str, key: str) -> str:
    return ctx.count(int(n), src=src, key=key)


def _range(ctx: Ctx, lo: float, hi: float, src: str, key: str, fmt: str = "g") -> str:
    return f"{ctx.num(lo, src=src, key=key + '[lower]', fmt=fmt)}–{ctx.num(hi, src=src, key=key + '[upper]', fmt=fmt)}"


def _secondary_inputs() -> str:
    """Names of the Table 1 inputs with status SECONDARY (results/tables/table1_parameters.csv)."""
    import pandas as pd
    from build_manuscript import ROOT
    t = pd.read_csv(ROOT / "results" / "tables" / "table1_parameters.csv")
    sec = t[t["status"] == "SECONDARY"]
    dens = sec["parameter"].str.contains("density")
    names = (["the tissue densities of the transient model"] if dens.any() else []) + \
        [p.split(" (")[0] for p in sec.loc[~dens, "parameter"]]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


# ------------------------------------------------------------------ 2.1
def s21(ctx: Ctx) -> list:
    return [
        Heading("2. Materials and Methods", 1),
        Heading("2.1. Study Design", 2),
        Para(
            "This is an *in silico* study; no patient data were used. A structured cohort of calcified coronary "
            "cross-sections was generated and stored as polar label maps, and every lesion was analysed by two "
            f"separate finite element models ({ctx.fig('fig1_design')}). The predictor model is a cheap "
            "pre-treatment analysis: the lesion geometry and one linear-elastic solve of the undamaged lesion under "
            "the IVL balloon pressure. The outcome model is a separate nonlinear simulation of the treatment with "
            "phase-field damage in the calcium. All predictors were computed from the predictor model and the "
            "geometry only, and all outcomes from the outcome model only. The two models share the lesion "
            "geometry and the material constants, including the calcium modulus; no quantity computed by the "
            "outcome model is used as a predictor. In particular, the IVL load factor of the transient pulse "
            "analysis (Section 2.5) is an input of the outcome model and is excluded from the candidate predictors "
            "by a check in the code."),
        Para(
            "Every input of the two-dimensional (2D) main analysis is listed with its value, source and status in "
            f"{ctx.tab('table1_parameters')}, except the sampling ranges of the hold-out set, which are given in "
            "Section 2.2. Inputs with status UNVERIFIED were not used. Most literature values were checked against "
            "text attributed to their source in search results rather than against the full primary article. "
            f"Secondary-source inputs ({_secondary_inputs()}) and one provisional input (the lipid modulus) were "
            "used; decision D15 approved the densities and the physiological pressure as exceptions, and all are "
            "flagged in "
            f"{ctx.tab('table1_parameters')}. Decisions taken during the study are numbered D1–D15 and recorded "
            "in the configuration file."),
        tables_ms.table1(ctx),
    ]


# ------------------------------------------------------------------ 2.2
def s22(ctx: Ctx) -> list:
    import pandas as pd
    from dtivl import params as P
    from dtivl.geometry import cohort as G

    c = ctx.cite
    rng = P.raw("design.holdout_lhs")["ranges"]
    lum_lo, lum_hi = P.lit("pre_procedure_MLA_IVL_lesions", "value-sd").value, P.lit("pre_procedure_MLA_IVL_lesions", "value+sd").value
    dep_lo, dep_hi = P.lit("calcium_depth_ex_vivo_FDOCT", "value-sd").value, P.lit("calcium_depth_ex_vivo_FDOCT", "value+sd").value
    ecc_hi = 1.0 - 1.0 / P.lit("eccentric_plaque_threshold_max_over_min").value
    cap_lo, cap_hi = P.lit("min_cap_thickness_OCT_culprit_AMI").value, P.lit("min_cap_thickness_OCT_culprit_SAP").value
    coh = pd.read_csv(G.COHORT_DIR / "cohort.csv") if hasattr(G, "COHORT_DIR") else pd.read_csv("cohort/cohort.csv")
    hold = coh[coh["set"] == "holdout"]
    n_lipid = int((hold["lipid_arc_deg"] > 0).sum())
    pilots = P.raw("design.pilot_2d")
    return [
        Heading("2.2. Lesion Cohort", 2),
        Para(
            f"The core set is a full factorial of calcium arc ({_levels(ctx, 'arc_deg')}°), calcium thickness "
            f"({_levels(ctx, 'thickness_mm')} mm) and morphology (superficial sheet, deep sheet, calcified nodule "
            "and fragmented calcium), a design choice based on the superficial and deep distribution of calcium "
            f"{c('mintz2015imaging')} and the OCT classes of calcified plaque {c('sugiyama2019calcified')}, giving "
            f"{ctx.v(CS, 'n_per_set.core', fmt='d')} lesions ({ctx.fig('fig2_cohort_gallery')}). All core lesions "
            "share one reference geometry: a circular lumen of area "
            f"{ctx.param('geometry.lumen_area_mm2', fmt='.3g')} mm² {c('hill2020disruptcad3')} and an internal "
            "elastic lamina at the reference lumen radius (reference vessel diameter "
            f"{ctx.param('geometry.reference_diameter_mm', fmt='.3g')} mm {c('yamada2025distribution')}; no "
            "remodelling assumed), which gives a plaque "
            f"{ctx.v(CS, 'core_geometry.plaque_thickness_mm', fmt='.3g')} mm thick, surrounded by a wall (media "
            f"and adventitia, lumped) of {ctx.param('geometry.wall_thickness_mm', fmt='.3g')} mm "
            f"{c('konishi2023media', 'rioufol2006adventitia')}. With the external elastic membrane one media "
            f"thickness ({ctx.lit('media_thickness_main_branches', fmt='.3g')} mm) outside the internal elastic "
            f"lamina, the plaque burden is {ctx.v(CS, 'plaque_burden_check.implied_plaque_burden_pct', fmt='.1f')}%. "
            "Superficial sheets, nodules and fragments start at a depth of "
            f"{ctx.param('geometry.depth_superficial_mm', fmt='.2g')} mm below the lumen and deep sheets at "
            f"{ctx.param('geometry.depth_deep_mm', fmt='.2g')} mm, the mean minus and plus one standard deviation "
            f"of an *ex vivo* OCT series {c('fdoct_calcium_volumetric')}. A nodule adds a protrusion of height "
            f"{ctx.param('geometry.nodule_height_mm', fmt='.2g')} mm over a base arc of "
            f"{ctx.param('geometry.nodule_base_arc_deg', fmt='g')}° into the lumen; fragmented calcium is split into "
            f"one deposit per {ctx.param('geometry.fragment_pitch_deg', fmt='g')}° of arc, each filling "
            f"{ctx.param('geometry.fragment_fill', fmt='.0f', scale=100.0)}% of its pitch (assumptions). Where the "
            "calcium, its depth and a minimum fibrous layer of "
            f"{ctx.param('geometry.calcium_iel_clearance_mm', fmt='.2g')} mm above the internal elastic lamina did "
            "not fit within the reference plaque, the plaque was grown outward with the lumen unchanged "
            f"({ctx.v(CS, 'plaque_growth.n_grown.core', fmt='d')} core lesions, all at the thickest level; "
            "decision D11); at that level the fragment fill was raised and the base arc of the narrowest nodule "
            "reduced so that the deposits fit. The three pilot lesions of the earlier study phase "
            f"({_n(ctx, len(pilots), CFG, 'len(design.pilot_2d)')}; Section 2.4) have the geometry of core lesions "
            "C-A90-T0.2-SUP, C-A270-T0.4-DEEP and C-A360-T0.2-NOD."),
        Para(
            f"The hold-out set of {ctx.v(CS, 'n_per_set.holdout', fmt='d')} lesions was drawn by Latin hypercube "
            "sampling (seed "
            f"{ctx.num(P.raw('design.seed') + G.HOLDOUT_SEED_OFFSET, src=CFG + ' + src/geometry/cohort.py', key='design.seed + HOLDOUT_SEED_OFFSET', fmt='d')}) "
            "over continuous ranges of calcium arc "
            f"({_range(ctx, rng['arc_deg'][0], rng['arc_deg'][1], CFG, 'design.holdout_lhs.ranges.arc_deg')}°), "
            f"thickness ({_range(ctx, rng['thickness_mm'][0], rng['thickness_mm'][1], CFG, 'design.holdout_lhs.ranges.thickness_mm')} mm), "
            f"depth ({_range(ctx, dep_lo, dep_hi, LIT, 'calcium_depth_ex_vivo_FDOCT value -/+ sd', fmt='.2g')} mm) "
            f"{c('fdoct_calcium_volumetric')}, lumen area "
            f"({_range(ctx, lum_lo, lum_hi, LIT, 'pre_procedure_MLA_IVL_lesions value -/+ sd', fmt='.2g')} mm²) "
            f"{c('hill2020disruptcad3')} and plaque eccentricity index "
            f"({ctx.num(rng['ecc_index'][0], src=CFG, key='design.holdout_lhs.ranges.ecc_index[lower]', fmt='g')} to "
            f"{ctx.num(ecc_hi, src=LIT, key='1 - 1 / eccentric_plaque_threshold_max_over_min', fmt='.3f')}, the "
            f"intravascular ultrasound limit of a maximum-to-minimum plaque thickness ratio of "
            f"{ctx.lit('eccentric_plaque_threshold_max_over_min', fmt='g')} {c('cvu2016eccentricity')}). "
            "The morphologies followed a balanced cycle. A lipid pool under a fibrous cap was assigned to "
            f"{ctx.num(P.raw('design.holdout_lhs')['ranges']['lipid_fraction'] * 100, src=CFG, key='design.holdout_lhs.ranges.lipid_fraction x 100', fmt='.0f')}% "
            "of the samples and kept where it fitted beside the calcium "
            f"({_n(ctx, n_lipid, COH, 'count(set == holdout and lipid_arc_deg > 0)')} lesions), with a lipid arc of "
            f"{ctx.num(rng['lipid_arc_deg'][0], src=CFG, key='design.holdout_lhs.ranges.lipid_arc_deg[0]', fmt='g')}–"
            f"{ctx.lit('lipid_arc_high_risk_threshold', fmt='g')}° (upper bound: the high-risk lipid arc "
            f"{c('clima_lipid_arc')}; lower bound an assumption), a cap thickness of "
            f"{_range(ctx, cap_lo, cap_hi, LIT, 'min_cap_thickness_OCT_culprit_AMI / _SAP', fmt='.2g')} mm (median "
            f"minimum cap thickness of culprit plaques {c('mgh_oct_cap')}) and a pool thickness of "
            f"{ctx.num(rng['lipid_thickness_mm']['value'], src=CFG, key='design.holdout_lhs.ranges.lipid_thickness_mm', fmt='g')} mm "
            "(assumption). Where the calcium did not fit, its depth was reduced, and "
            f"{ctx.v(CS, 'plaque_growth.n_grown.holdout', fmt='d')} hold-out plaques were grown outward. Thresholds "
            "and models derived on the core set were applied to the hold-out set without change."),
        Para(
            "Each lesion was stored as a polar label map in the OCT convention of "
            f"{c('gharaibeh2019spie', 'lee2020twostep')} (A-lines × radial samples; "
            f"{ctx.raw('discretisation.polar_n_theta', fmt='d')} A-lines, "
            f"{ctx.num(P.raw('discretisation.polar_dr_mm') * 1e3, src=CFG, key='discretisation.polar_dr_mm x 1000', fmt='g')} µm "
            "radial sampling), and the geometric descriptors were computed from the label map only, as they "
            "would be from a segmented image: total calcium arc (A-lines containing calcium), maximum calcium "
            "thickness (including a nodule protrusion), minimum calcium depth, calcium area, number of deposits, "
            "lumen area, lipid arc and plaque eccentricity. The cross-sectional Fujino score awards "
            f"{_fpts(ctx, 'angle')} points for a total calcium arc above "
            f"{ctx.lit('fujino_angle_threshold', fmt='g')}° and {_fpts(ctx, 'thickness')} point for a maximum "
            f"calcium thickness above {ctx.lit('fujino_thickness_threshold', fmt='g')} mm {c('fujino2018score')}; "
            "in 2D the length criterion cannot be scored, so the score is at most "
            f"{ctx.num(_fmax(), src=LIT, key='sum of the cross-sectional fujino_*_threshold points', fmt='d')}."),
    ]


# ------------------------------------------------------------------ 2.3
def s23(ctx: Ctx) -> list:
    from dtivl import params as P
    from dtivl.fracture.phasefield import regularisation_length

    c = ctx.cite
    ex = P.raw("discretisation.example3d")
    l3 = P.raw("design.length_series_3d")
    ell3 = regularisation_length(h_ca=float(ex["h_calcium_mm"]))
    return [
        Heading("2.3. Geometry and Meshing", 2),
        Para(
            "The label map was converted into a conforming partition of lumen, calcium, lipid, fibrous intima "
            "(inside the internal elastic lamina) and wall, and meshed with linear triangles in Gmsh "
            f"{c('geuzaine2009gmsh')}. The element size was "
            f"{ctx.num(P.raw('discretisation.h_calcium_mm') * 1e3, src=CFG, key='discretisation.h_calcium_mm x 1000', fmt='g')} µm "
            "in and next to the calcium and grew to "
            f"{ctx.num(P.raw('discretisation.h_soft_mm'), src=CFG, key='discretisation.h_soft_mm', fmt='g')} mm in the "
            "soft tissue. Displacements were interpolated with quadratic (P2) and the damage field with linear (P1) "
            "Lagrange elements, under plane strain. For the three-dimensional (3D) length series (Section 2.9), "
            "the cross-section was extruded into linear tetrahedra over the calcified length, with calcium-free "
            f"end segments of {ctx.param('geometry.margin_3d_mm', fmt='g')} mm and the axial displacement fixed on "
            "the end faces, using in-plane sizes of "
            f"{ctx.num(ex['h_calcium_mm'], src=CFG, key='discretisation.example3d.h_calcium_mm', fmt='g')} mm "
            f"(calcium) and {ctx.num(ex['h_soft_mm'], src=CFG, key='discretisation.example3d.h_soft_mm', fmt='g')} mm "
            "(soft tissue) and axial sizes of "
            f"{ctx.num(l3['h_axial_ca_mm'], src=CFG, key='design.length_series_3d.h_axial_ca_mm', fmt='g')} mm "
            "(calcified segment) and "
            f"{ctx.num(l3['h_axial_margin_mm'], src=CFG, key='design.length_series_3d.h_axial_margin_mm', fmt='g')} mm "
            "(end segments); the phase-field length of these models was "
            f"{ctx.num(ell3, src=CFG, key='fracture.ell_over_h x discretisation.example3d.h_calcium_mm', fmt='g')} mm."),
    ]


# ------------------------------------------------------------------ 2.4
def s24(ctx: Ctx) -> list:
    from dtivl.drivers.stage_pilot import N_E_SWEEP, e_sweep

    c = ctx.cite
    ms = _ms()
    sweep = e_sweep()
    return [
        Heading("2.4. Constitutive Models", 2),
        Para(
            "The wall (media and adventitia, lumped) was an isotropic Yeoh solid with the constants of a Yeoh fit "
            f"{c('escuer2019restenosis')} to circumferential data of human coronary media "
            f"{c('holzapfel2005coronary')}, used for the whole lumped wall (assumption). The fibrous intima and "
            "the lipid pool were nearly incompressible neo-Hookean solids with small-strain Young moduli of "
            f"{ctx.num(ms.fibrous_E, src=MS_SRC, key='fibrous_E', fmt='.3g')} MPa (hypocellular human aortic plaque) "
            f"{c('loree1994modulus')} and {ctx.num(ms.lipid_E * 1e3, src=MS_SRC, key='lipid_E x 1000', fmt='.2g')} kPa "
            f"(atomic force microscopy; provisional, species not confirmed) {c('jsb2011afm')}; the soft tissues "
            f"used a bulk-to-shear modulus ratio of {ctx.num(ms.k_over_mu, src=MS_SRC, key='k_over_mu', fmt='g')} "
            "(an assumption, varied in the sensitivity analysis). Calcium was a compressible neo-Hookean solid "
            "defined by its small-strain modulus and a Poisson ratio of "
            f"{ctx.num(ms.ca_nu, src=MS_SRC, key='ca_nu', fmt='g')} {c('cvet2014calcmodulus')}, brittle through "
            "the damage model of Section 2.6; up to the failure strain its response is close to linear. In the "
            "predictor model all materials were linear elastic with the small-strain constants above."),
        Para(
            "The calcium modulus is the most uncertain input: published values range from "
            f"{ctx.lit('calcified_tissue_tangential_modulus', fmt='.2g')} MPa for tissue-level tension "
            f"{c('loree1994modulus')} to the gigapascal range at the particle scale, reported from nanoindentation "
            f"and summarised in a parametric FE study {c('cvet2014calcmodulus')}. In the pilot phase, "
            f"{_n(ctx, N_E_SWEEP, 'src/drivers/stage_pilot.py', 'N_E_SWEEP')} log-spaced moduli between "
            f"{ctx.num(sweep[0], src='src/drivers/stage_pilot.py e_sweep()', key='min', fmt='.3g')} MPa and "
            f"{ctx.num(sweep[-1] / 1e3, src='src/drivers/stage_pilot.py e_sweep()', key='max / 1000', fmt='g')} GPa "
            f"were set for the three pilot lesions (whose geometries are core lesions, Section 2.2){_sweep_run(ctx)}. "
            "The baseline was "
            f"set to {ctx.num(ms.ca_E, src=MS_SRC, key='ca_E', fmt='.0f')} MPa (decision D8), the stiffest swept "
            "modulus at which the IVL balloon inflated alone did not damage the pilot lesions while IVL did. This "
            "is a calibration to a criterion internal to the model, not to a measured property; because calcium "
            "failure is strain-controlled (below), it also set the calcium strength. The outcome "
            "model's response on three core geometries thereby set the modulus that both models use. With the "
            "failure strain below, it implies a calcium failure stress of "
            f"{ctx.num(ms.sigma_crit, src=MS_SRC, key='sigma_crit = ca_E x eps_f', fmt='.2g')} MPa, about "
            f"{ctx.num(ms.sigma_crit / ms.sigma_c, src=MS_SRC, key='sigma_crit / sigma_c', fmt='.0f')} times the "
            f"tissue-level strength of {ctx.num(ms.sigma_c, src=MS_SRC, key='sigma_c', fmt='.2g')} MPa measured in "
            f"tension {c('holzapfel2004plaque')}, and at this modulus a plain balloon at "
            f"{_atm(ctx, 'loads.poba_pressure_atm')} atm still fractured part of the pilot lesions, unlike in cadaver "
            f"lesions {c('kawai2023cadaver')}. The modulus was therefore varied in the sensitivity analysis "
            "(Section 2.9)."),
    ]


def _sweep_run(ctx: Ctx) -> str:
    """Which swept moduli were not run (results/pilot_nonlinear.json 'skipped') and the extra tissue-level one."""
    from build_manuscript import ROOT
    import json
    les = json.loads((ROOT / "results" / "pilot_nonlinear.json").read_text())["lesions"]
    rows = next(iter(les.values()))
    skipped = sorted({r["ca_E_MPa"] for rs in les.values() for r in rs if "skipped" in r})
    import numpy as np
    from dtivl import params as P
    from dtivl.drivers.stage_pilot import N_E_SWEEP
    lo, hi = (q.value for q in P.get_list("materials.calcium.sens_E"))
    grid = set(np.round(np.logspace(np.log10(lo), np.log10(hi), N_E_SWEEP), 6).tolist())
    extra = sorted({round(r["ca_E_MPa"], 6) for r in rows} - grid - set(skipped))
    src = "results/pilot_nonlinear.json"
    txt = ""
    if extra:
        txt += (", together with the tissue-level secant modulus ("
                + ", ".join(ctx.num(e, src=src, key="ca_E_MPa not in e_sweep()", fmt=".3g") for e in extra) + " MPa)")
    if skipped:
        txt += ("; the lowest modulus ("
                + ", ".join(ctx.num(e, src=src, key="ca_E_MPa skipped", fmt=".3g") for e in skipped)
                + " MPa) was not run, because calcium softer in shear than the fibrous matrix leaves the crack filler "
                  "undefined")
    return txt


# ------------------------------------------------------------------ 2.5
def s25(ctx: Ctx) -> list:
    from dtivl import params as P
    from dtivl.drivers.stage_cohort_runs import COHORT_PROTOCOLS
    from dtivl.solvers.transient import default_pulse_width_us

    c = ctx.cite
    cal = "results/calibration_stent.json"
    return [
        Heading("2.5. Boundary Conditions and Load Protocols", 2),
        Para(
            "The lumen pressure acted as a follower load on the luminal surface. Rigid-body motion was removed by "
            f"a weak elastic foundation on the outer boundary ({ctx.param('loads.perivascular_k_MPa_per_mm', fmt='g')} "
            "MPa/mm; its effect on the peak calcium stress is reported in "
            f"{ctx.tab('table2_verification_consistency')}). Balloons were represented by the lumen pressure inside "
            "a rigid cylindrical cap at the balloon radius and the stent by an expanding rigid cylinder, both with "
            f"frictionless penalty contact (penalty {ctx.param('loads.contact_penalty_MPa_per_mm', fmt='g')} MPa/mm). "
            f"{_n(ctx, len(COHORT_PROTOCOLS), 'src/drivers/stage_cohort_runs.py', 'len(COHORT_PROTOCOLS)')} "
            "protocols were run on every lesion: (i) physiological pressure "
            f"({ctx.param('loads.physiological_pressure', fmt='.3g', scale=1e3)} kPa) {c('whelton2017bp')}; "
            f"(ii) the IVL balloon alone, a lumen pressure of {_atm_mpa(ctx, 'loads.ivl_balloon_pressure_atm')} "
            f"{c('tovarforero2019system')} inside a cap at the radius of the IVL balloon (diameter "
            f"{ctx.param('geometry.ivl_balloon_diameter_mm', fmt='.3g')} mm) {c('yamada2025distribution')}; "
            f"(iii) a plain balloon at {_atm_mpa(ctx, 'loads.poba_pressure_atm')} with the same cap, the balloon "
            f"arm of a cadaver comparison {c('kawai2023cadaver')}; (iv) IVL (below); and stent deployment (v) "
            "without and (vi) after IVL: the rigid cylinder expanded from just inside the smallest lumen radius to "
            f"the target stent diameter ({ctx.param('geometry.stent_target_diameter_mm', fmt='.3g')} mm) "
            f"{c('yamada2025distribution')}, stopping early if the mean contact pressure reached the rated burst "
            f"pressure of the stent delivery balloon ({_atm_mpa(ctx, 'loads.balloon_rated_pressure_atm')}) "
            f"{c('stent_specs')}; the stent then recoiled to the radius at which the tissue pressure balanced a "
            f"linear stent stiffness. The stiffness ({ctx.v(cal, 'K_s_MPa_per_mm', fmt='.3g')} MPa/mm) was identified "
            "so that a calcium-free version of a pilot lesion showed the clinical acute recoil of "
            f"{ctx.param('loads.stent_recoil_target_pct', fmt='g')}% {c('recoil_ees_bvs')}. Damage was active in "
            "every protocol; stenting after IVL started from the IVL damage field. Loads were applied in increments "
            "that were halved on non-convergence, up to "
            f"{_const(ctx, 'dtivl.solvers.nonlinear', 'MAX_HALVINGS', fmt='d')} times."),
        Para(
            "IVL was represented by its quasi-static equivalent (decision D14): the balloon was inflated inside the "
            "cap, the pressure was then raised in a quasi-static ramp to the peak "
            "$p_{top} = p_{bal} + T_{eq}\\,p_{peak}\\,s$, and the lesion was unloaded. Here $p_{bal}$ is the balloon "
            f"pressure, $p_{{peak}}$ = {ctx.param('loads.ivl_peak_pressure_MPa', fmt='g')} MPa the reported peak "
            "positive pressure of the pulse (measurement location not stated) "
            f"{c('kereiakes2021principles')}, $s$ = {ctx.param('loads.ivl_pulse_scale', fmt='g')} a pulse-amplitude "
            "scale (varied in the sensitivity analysis) and $T_{eq}$ a lesion-specific transfer factor from a linear "
            "elastodynamic analysis of one pulse. That analysis applied a half-sine lumen pressure pulse of "
            f"{ctx.num(default_pulse_width_us(), src='src/solvers/transient.py default_pulse_width_us()', key='ivl_pulse_duration upper bound (us)', fmt='g')} µs "
            f"(the upper bound of the reported pulse duration {c('brinton2019feasibility')}; pulse shape assumed) with an absorbing "
            "(Lysmer–Kuhlemeyer dashpot) outer boundary. The bulk moduli came from measured longitudinal sound "
            f"speeds of arterial tissue {c('rooney1979arteries')}, highly calcified lesions "
            f"{c('rooney1982lesions')}, fibrous plaque {c('qmul_carotid_acoustic')} and fatty material "
            f"{c('umb1998acoustic')} ($K = \\rho c^2 - 4\\mu/3$), with densities from earlier modelling studies "
            f"{c('density_drilling2019', 'density_stentfe2014', 'density_thermal2007')} (secondary sources, "
            "decision D15); time integration used the non-dissipative generalised-α scheme and a mixed "
            "displacement–pressure formulation, with a time step of the smaller of the pulse width over "
            f"{_const(ctx, 'dtivl.solvers.transient', 'STEPS_PER_PULSE', fmt='g')} and the stability limit of the mesh. "
            "$T_{eq}$ is the largest cell value of the maximum principal stress in the calcium over a window of "
            f"{_const(ctx, 'dtivl.solvers.transient', 'WINDOW_PULSES', fmt='g')} pulse widths from the start of the "
            f"pulse{_t_peak(ctx)}, divided by the same quantity in a static linear-elastic solve with "
            "the constants of the predictor model under a lumen pressure equal to the pulse peak; in that linear "
            "model, the pulse-equivalent increment reproduces the transient peak. The equivalence represents the "
            "peak tensile response to one pulse; the compressive wavefront, repeated pulses and fatigue were not "
            "modelled."),
    ]


def _t_peak(ctx: Ctx) -> str:
    """When the transient peak occurred (results/cohort_outcomes.csv ivl_load_t_peak, us)."""
    import pandas as pd
    from build_manuscript import ROOT
    from dtivl.solvers.transient import default_pulse_width_us
    d = pd.read_csv(ROOT / "results" / "cohort_outcomes.csv")
    tp = pd.to_numeric(d["ivl_load_t_peak"], errors="coerce").dropna()
    if tp.empty:
        return ""
    after = int((tp > default_pulse_width_us()).sum())
    src = "results/cohort_outcomes.csv"
    return (f" (in the cohort the peak came {ctx.num(float(tp.min()), src=src, key='min(ivl_load_t_peak)', fmt='.2g')}–"
            f"{ctx.num(float(tp.max()), src=src, key='max(ivl_load_t_peak)', fmt='.2g')} µs after the pulse start, "
            f"after the end of the pulse in {ctx.count(after, src=src, key='count(ivl_load_t_peak > pulse width)')} of "
            f"{ctx.count(len(tp), src=src, key='count(ivl_load_t_peak)')} lesions)")


# ------------------------------------------------------------------ 2.6
def s26(ctx: Ctx) -> list:
    from dtivl import params as P

    c = ctx.cite
    ms = _ms()
    ell = P.get("fracture.ell_over_h").value * P.raw("discretisation.h_calcium_mm")
    e = P.raw("design.erosion_subset")
    return [
        Heading("2.6. Fracture Model", 2),
        Para(
            "Calcium damage was modelled with the AT1 variational phase-field model of brittle fracture "
            f"{c('bourdin2000numerical', 'tanne2018nucleation')}. The damage $d$ ($d = 0$ intact, $d = 1$ broken) "
            "lives in the calcium only and is irreversible. The calcium strain energy was split into a tensile part "
            "$\\psi^+$ (the deviatoric energy plus the volumetric energy under expansion) and a compressive part "
            "(the volumetric energy under compression); only $\\psi^+$ is degraded, by "
            "$g(d) = (1-k)(1-d)^2 + k$, and the fracture energy is $\\frac{3 G_c}{8}\\int (d/\\ell + "
            "\\ell |\\nabla d|^2)\\,dx$. The residual stiffness $k$ was set so that fully damaged calcium keeps the "
            "shear stiffness of the surrounding fibrous matrix (decision D9; "
            f"$k$ = {ctx.num(ms.k_floor, src=MS_SRC, key='k_floor = mu_fibrous / mu_Ca', fmt='.2g')}), and "
            "$G_c = 16\\,\\ell\\,(1-k)\\,\\psi_{crit}/3$ places the AT1 damage onset at the critical energy "
            "$\\psi_{crit}$ of a uniaxial failure strain $\\varepsilon_f$ (decision D5). The failure strain "
            f"$\\varepsilon_f$ = {ctx.num(ms.eps_f, src=MS_SRC, key='eps_f = calcification_fracture_stretch - 1', fmt='.3g')} "
            "is the failure stretch of calcified tissue minus one, measured in uniaxial tension of human iliac "
            f"plaques {c('holzapfel2004plaque')}. Because $\\ell$ is tied to the element size (below), $G_c$ is a "
            "numerical parameter rather than a measured toughness, and the model is in effect a strength-based "
            "damage-nucleation model. The regularisation length was "
            f"$\\ell$ = {ctx.num(ell * 1e3, src=CFG, key='fracture.ell_over_h x discretisation.h_calcium_mm x 1000', fmt='g')} µm, "
            f"{ctx.param('fracture.ell_over_h', fmt='g')} times the calcium element size. Each load increment was "
            f"solved by alternate minimisation {c('miehe2010phasefield')}: a Newton solve for the displacement with "
            "damage fixed, then a bound-constrained solve for the damage with displacement fixed, repeated until "
            f"the largest damage change fell below {_const(ctx, 'dtivl.solvers.nonlinear', 'STAGGER_TOL', fmt='g')} "
            f"or {_const(ctx, 'dtivl.solvers.nonlinear', 'STAGGER_MAX_IT', fmt='d')} iterations were reached; "
            "increments that reached the iteration limit were counted and are reported."),
        Para(
            "As a comparator with a different fracture model (decision D13), IVL was also simulated with element "
            "erosion: after each converged increment, calcium cells whose cell-averaged maximum principal stretch "
            "minus one reached $\\varepsilon_f$ were eroded (their whole strain energy, in tension and compression, "
            "was scaled by the residual stiffness $k$), and the increment was re-solved until no further cells "
            f"eroded. The comparator was run on "
            f"{_n(ctx, len(e['arc_deg']) * len(e['thickness_mm']) * len(e['morphology']), CFG, 'size of design.erosion_subset')} "
            "core lesions: the factorial of two arcs "
            f"({ctx.num(e['arc_deg'][0], src=CFG, key='design.erosion_subset.arc_deg[0]', fmt='g')}° and "
            f"{ctx.num(e['arc_deg'][1], src=CFG, key='design.erosion_subset.arc_deg[1]', fmt='g')}°), the thinnest "
            "and thickest levels, a superficial sheet and a nodule."),
    ]


# ------------------------------------------------------------------ 2.7
def s27(ctx: Ctx) -> list:
    from dtivl import params as P

    D = P.raw("design.core_factorial")   # noqa: F841 (kept for symmetry with the other sections)
    exp_cap = 100.0 * (P.get("geometry.stent_target_diameter_mm").value / P.get("geometry.reference_diameter_mm").value) ** 2
    return [
        Heading("2.7. Outcome Definitions", 2),
        Para(
            "Calcium was considered damaged where the cell-averaged damage exceeded "
            f"{ctx.raw('fracture.d_crack', fmt='g')}. The damage-onset load of a protocol is the lumen pressure of "
            "the first load step at which the largest nodal damage in the calcium reached this value (a local "
            "criterion that can precede any damaged cell); for the stent protocols the load is the stent radius. "
            "A damaged component that connects "
            "the luminal and abluminal calcium surfaces and cuts its deposit (splits it into pieces of at least "
            f"{_const(ctx, 'dtivl.fracture.cracks', 'MIN_PIECE_CELLS', fmt='d')} cells, opens a closed ring or "
            "removes it entirely) is a through-thickness fracture; other damaged components, including damage "
            "confined to a rounded deposit end, are partial. A component covering at least "
            f"{_const(ctx, 'dtivl.fracture.cracks', 'DIFFUSE_REGION_FRAC', fmt='.0f', scale=100.0)}% of its "
            "deposit was labelled diffuse; it is still a through-thickness fracture when it meets that definition. "
            "Decision D10 named through-thickness damage and the damage-onset pressure as the primary fracture "
            "outcomes and the damaged fraction as secondary. In the statistical analysis (Section 2.10), "
            "through-thickness damage under IVL is the primary outcome; the secondary outcomes are the "
            "damage-onset load under IVL, through-thickness damage under the IVL balloon alone, under the plain "
            "balloon and under IVL with the erosion comparator, and stent expansion with and without prior IVL. "
            "Stent expansion is the stent area after recoil divided by the reference lumen area (the area of a "
            "circle of the reference vessel diameter), in per cent; because the stent was expanded at most to its "
            "target diameter, expansion before recoil cannot exceed "
            f"{ctx.num(exp_cap, src=CFG, key='100 x (stent_target_diameter_mm / reference_diameter_mm)^2', fmt='.1f')}%. "
            "Acute recoil is the relative loss of stent radius between deployment and recoil."),
    ]


# ------------------------------------------------------------------ 2.8
def s28(ctx: Ctx) -> list:
    from dtivl import params as P

    nl = P.get("fracture.ell_over_h").value * P.raw("discretisation.h_calcium_mm")
    return [
        Heading("2.8. Pre-Treatment Mechanics Predictors", 2),
        Para(
            "The predictor model is a small-strain, plane-strain linear-elastic solve of the undamaged lesion under "
            f"a lumen pressure of {_atm_mpa(ctx, 'loads.ivl_balloon_pressure_atm')} (the IVL balloon pressure), with "
            "the material constants of Section 2.4 linearised at zero strain. From the calcium cells it gives the "
            "maximum principal stress $\\sigma_1$ (including the out-of-plane stress), the maximum principal "
            "strain $\\varepsilon_1$ and the strain energy density (SED). The predictors were the area-weighted "
            f"{_const(ctx, 'dtivl.metrics.predictors', 'SIGMA1_PERCENTILE', fmt='g')}th and "
            f"{_const(ctx, 'dtivl.metrics.predictors', 'SIGMA1_PERCENTILE_HIGH', fmt='g')}th percentiles and mean of "
            "$\\sigma_1$; the mean SED; non-local peaks of $\\sigma_1$ and SED, in which each cell value is replaced "
            "by the area-weighted mean over the calcium within a radius equal to the phase-field length "
            f"({ctx.num(nl * 1e3, src=CFG, key='fracture.ell_over_h x discretisation.h_calcium_mm x 1000', fmt='g')} µm) "
            "before the maximum is taken; the calcium area fractions with $\\varepsilon_1$ above $\\varepsilon_f$ "
            "and $\\varepsilon_f/2$ and with $\\sigma_1$ above the implied failure stress "
            "$\\sigma_{crit} = E_{Ca}\\varepsilon_f$ of the outcome model and half of it; an edge stress "
            "concentration factor (the non-local peak $\\sigma_1$ near deposit ends over the mean $\\sigma_1$ of "
            "the calcium, within the larger of "
            f"{_const(ctx, 'dtivl.metrics.predictors', 'EDGE_MIN_WINDOW_DEG', fmt='g')}° and twice the deposit "
            "thickness over its mean radius on either side of each end, and set to one for a closed ring); and the "
            "smallest local calcium thickness among the cells above the "
            f"{_const(ctx, 'dtivl.metrics.predictors', 'HIGH_STRESS_PERCENTILE', fmt='g')}th percentile of "
            "$\\sigma_1$. Raw cell peaks were not used as predictors because they changed by more than the "
            "acceptance limit under mesh refinement (Section 2.9)."),
    ]


# ------------------------------------------------------------------ 2.9
def s29(ctx: Ctx) -> list:
    from dtivl import params as P
    from dtivl.drivers import stage_sensitivity as S
    from dtivl.drivers import stage_verification as V
    from dtivl.drivers.stage_pilot import pilot_ids

    c = ctx.cite
    sens = P.raw("design.sensitivity")
    sob = sens["sobol"]
    n_sob_inp = sum(1 for v in sob["inputs"].values() if v["value"])
    l3 = P.raw("design.length_series_3d")
    SS = "src/drivers/stage_sensitivity.py morris_inputs()"
    mi = {m.name: m for m in S.morris_inputs()}
    rng = lambda name, f=lambda x: x, fmt="g": (  # noqa: E731
        f"{ctx.num(f(mi[name].bounds[0]), src=SS, key=name + '[lower]', fmt=fmt)}–"
        f"{ctx.num(f(mi[name].bounds[1]), src=SS, key=name + '[upper]', fmt=fmt)}")
    p10 = lambda x: 10.0 ** x  # noqa: E731
    lengths = P.raw("design.lengths_3d_mm")
    len_txt = [ctx.num(v, src=CFG, key=f"design.lengths_3d_mm[{i}]", fmt="g") for i, v in enumerate(lengths)]
    return [
        Heading("2.9. Verification, Consistency and Sensitivity Analyses", 2),
        Para(
            "Verification compared the solvers with analytical solutions: linear elasticity of a thick cylinder "
            "under pressure (Lamé solution), the small-load response of the hyperelastic solver, penalty contact "
            "with a rigid cylinder, and the AT1 damage onset under uniform strain. Discretisation error of the "
            "stress, strain-energy and edge predictors was estimated on "
            f"{_n(ctx, len(V.GCI_LESIONS), 'src/drivers/stage_verification.py', 'len(GCI_LESIONS)')} lesions at "
            f"{_n(ctx, len(V.GCI_FACTORS), 'src/drivers/stage_verification.py', 'len(GCI_FACTORS)')} mesh levels "
            f"with the grid convergence index {c('celik2008gci')}. Criteria and results are listed in "
            f"{ctx.tab('table2_verification_consistency')}."),
        Para(
            "Consistency with published observations was assessed by comparing model outcomes with clinical and "
            "experimental observations made in other lesions: the direction of the calcium arc and thickness "
            f"effects on fracture {c('fujino2018fracture')}, the lower fracture rate of nodules "
            f"{c('eij2024calcifiednodules', 'emori2023ivlfracture')}, lesion-level fracture rates after IVL "
            f"{c('ali2019disruptcad2', 'hill2020disruptcad3')}, the absence of fracture under a plain balloon in "
            f"cadaver lesions {c('kawai2023cadaver')}, stent expansion {c('hill2020disruptcad3')}, the number of "
            f"fractures per lesion and the fraction of full-thickness fractures {c('tct120pooledoct2021')} and the "
            f"association of the Fujino score with expansion {c('fujino2018score')}. Acute recoil was also compared "
            f"with its clinical value {c('recoil_ees_bvs')}, but it is the calibration target of the stent "
            "stiffness, not an independent check. The balloon-alone criterion of D8 is internal to the model, the "
            f"calcium stress magnitude was compared with an earlier FE study {c('dong2023loadsharing')}, and the "
            f"implied plaque burden with a computed tomography series {c('bmc2022nodulecta')}. Each check has a "
            "criterion fixed in the configuration and a label (agrees, partial, disagrees, not computed). These "
            "comparisons test plausibility, not the accuracy of the predictions for any lesion."),
        Para(
            "Sensitivity analysis had four parts. One-at-a-time scenarios on the "
            f"{_n(ctx, len(pilot_ids()), 'src/drivers/stage_pilot.py', 'len(pilot_ids())')} pilot lesions "
            f"({_n(ctx, S.n_oat_runs(), 'src/drivers/stage_sensitivity.py', 'n_oat_runs()')} runs) removed the "
            "crack filler (D9), changed the phase-field length, the pulse scale, the foundation stiffness and the "
            "soft-tissue bulk modulus, replaced the damage driving force by a principal-strain (Rankine-type) "
            "force (D10), lowered the calcium density (D15) and switched to a stress-based failure criterion (D5). "
            f"Morris elementary effects {c('herman2017salib')} screened "
            f"{_n(ctx, len(mi), 'src/drivers/stage_sensitivity.py', 'len(morris_inputs())')} inputs of the outcome "
            f"model over these ranges: calcium modulus {rng('log10_ca_E', p10, '.4g')} MPa (log scale; the sweep "
            f"moduli next to the baseline), failure strain {rng('eps_f', fmt='.3g')}, pulse scale "
            f"{rng('ivl_pulse_scale')}, phase-field length {rng('ell_over_h')} times the element size, foundation "
            f"stiffness {rng('log10_k_perivascular', p10)} MPa/mm (log scale), calcium density "
            f"{rng('rho_calcium')} kg/m³, soft-tissue bulk-to-shear ratio {rng('k_over_mu_soft')} and fibrous "
            f"modulus {rng('fibrous_E', fmt='.3g')} MPa, on "
            f"{_n(ctx, len(sens['morris_lesions']), CFG, 'len(design.sensitivity.morris_lesions)')} core lesions "
            f"({_n(ctx, sens['morris_trajectories'], CFG, 'design.sensitivity.morris_trajectories')} trajectories, "
            f"{_n(ctx, S.n_morris_runs(), 'src/drivers/stage_sensitivity.py', 'n_morris_runs()')} runs). "
            f"First-order and total Sobol indices {c('saltelli2010variance', 'herman2017salib')} of the "
            f"{_n(ctx, n_sob_inp, CFG, 'number of design.sensitivity.sobol.inputs set true')} inputs that enter the "
            "predictor model were computed for every predictor on the same lesions. Finally, the 3D length series "
            "(decision D12) extruded the core cross-section "
            + ", ".join(l3["cross_sections"]) +
            ", chosen because its 2D run shows damage onset under IVL and its thickness is resolved by the 3D mesh, "
            f"over calcified lengths of {', '.join(len_txt[:-1])} and {len_txt[-1]} mm, and loaded it with the IVL "
            "ramp up to damage onset, to compare the onset load with the 2D model at the same in-plane resolution."),
    ]


def blocks(ctx: Ctx) -> list:
    return s21(ctx) + s22(ctx) + s23(ctx) + s24(ctx) + s25(ctx) + s26(ctx) + s27(ctx) + s28(ctx) + s29(ctx)

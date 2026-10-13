"""Checkpoint 2 report: ``CHECKPOINT_2.md`` and its figures, built only from files.

Every result number in the report is read from ``results/*.json``,
``cohort/cohort.csv``, ``params/literature_parameters.csv`` or
``params/study_config.yaml`` (rule 1 of the study brief); protocol pressures in
labels are read from the parameter files too. Interpretive sentences that depend
on results (which moduli fracture under which protocol, whether a published
observation is met) are computed here. The options and recommendations put to
the PI are fixed text; the only numbers in fixed text are the decision and
section labels.

Run: ``python -m dtivl.drivers.report_checkpoint2`` (after the pilot stages).
"""
from __future__ import annotations

import datetime as _dt
import math

import numpy as np
import pandas as pd

from dtivl import ROOT
from dtivl import params as P
from dtivl.drivers.common import COHORT_DIR, RESULTS_DIR, read_json
from dtivl.materials.constitutive import MaterialSet

OUT = ROOT / "CHECKPOINT_2.md"
PROTOCOLS = ("physiological", "balloon_only", "balloon_10atm", "ivl", "stent_no_ivl", "stent_post_ivl", "ivl_erosion")
ATM_MPA = 0.101325          # unit conversion (1 atm in MPa)
WHOLE = 0.995               # damaged fraction reported as "whole deposit"


# ---------------------------------------------------------------- formatting
def f(x, nd: int = 3) -> str:
    """Number with ``nd`` significant digits; '–' for None/NaN."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    x = float(x)
    if x == 0.0:
        return "0"
    mag = math.floor(math.log10(abs(x)))
    if mag >= 5 or mag <= -4:
        return f"{x:.{nd - 1}e}"
    return f"{x:.{max(nd - 1 - mag, 0)}f}"


def pct(x, nd: int = 2) -> str:
    if x is None:
        return "–"
    x = float(x)
    if WHOLE <= x < 1.0:
        return f">{f(100.0 * WHOLE, 3)} %"
    return f(100.0 * x, nd) + " %"


def table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _load(name: str):
    path = RESULTS_DIR / f"{name}.json"
    return read_json(name) if path.exists() else None


def _elist(es) -> str:
    return ", ".join(f(e) for e in es) if es else "none"


def _emp(es) -> str:
    """Moduli list with unit, or 'none'."""
    return f"{_elist(es)} MPa" if es else "none"


def _lit(name: str) -> str:
    """'value (range) unit [STATUS]' of a literature row."""
    q = P.load_literature()[name]
    rng = f" ({q.range_str})" if q.range_str and q.range_str != "nan" else ""
    return f"{q.value_str} {q.unit}{rng} [{q.status}]"


def _litn(name: str) -> str:
    """Literature row with its name (self-describing label) and status."""
    return f"`{name}` = {_lit(name)}"


def _q(dotted: str) -> str:
    """Config quantity with status and sensitivity values."""
    q = P.get(dotted)
    sens = f"; sensitivity {', '.join(f(v) for v in q.sens)}" if q.sens else ""
    return f"{f(q.value)} {q.unit} [{q.status}{sens}]".replace("  ", " ")


def _atm(p_mpa: float) -> str:
    return f"{f(p_mpa / ATM_MPA)} atm"


def protocol_label(k: str) -> str:
    p_bal = P.get("loads.ivl_balloon_pressure_atm").value
    p_poba = P.get("loads.poba_pressure_atm").value
    return {"physiological": "Physiological pressure", "balloon_only": f"IVL balloon alone ({_atm(p_bal)})",
            "balloon_10atm": f"plain balloon ({_atm(p_poba)})", "ivl": "IVL (balloon + pulse)",
            "stent_no_ivl": "Stent, no IVL", "stent_post_ivl": "Stent after IVL",
            "ivl_erosion": "IVL, erosion comparator"}[k]


def _ok(rec) -> bool:
    return rec is not None and rec.get("status") == "ok"


def _pilot_rows() -> dict:
    """Per-lesion nonlinear rows: the aggregate file, else the per-lesion files plus part files."""
    agg = _load("pilot_nonlinear")
    if agg is not None:
        return agg["lesions"]
    from dtivl.drivers.stage_pilot import pilot_ids

    out = {}
    for lid in pilot_ids():
        rows = {}
        names = [f"pilot_nonlinear_{lid}"] + sorted(p.stem for p in RESULTS_DIR.glob(f"pilot_nonlinear_{lid}_*.json"))
        for name in names:                       # main file first, then part files (union of protocols)
            part = _load(name)
            for r in (part or {}).get("rows", []):
                e = round(r["ca_E_MPa"], 6)
                rows[e] = {**rows.get(e, {}), **r}
        if rows:
            out[lid] = sorted(rows.values(), key=lambda x: x["ca_E_MPa"])
    return out


def _sets(nl: dict, proto: str) -> dict:
    """lesion -> (moduli with a completed run, moduli with through-thickness damage, failed moduli)."""
    out = {}
    for lid, lr in nl.items():
        ok = [r["ca_E_MPa"] for r in lr if _ok(r.get(proto))]
        hit = [r["ca_E_MPa"] for r in lr if _ok(r.get(proto)) and r[proto]["cracks"]["fracture"]]
        bad = [r["ca_E_MPa"] for r in lr if proto in r and not _ok(r[proto])]
        out[lid] = (ok, hit, bad)
    return out


def _row(lr, E):
    return next((x for x in lr if abs(x["ca_E_MPa"] - E) < 1e-9), None)


def _step(proto: str) -> float:
    """Load increment of a protocol's main ramp (MPa)."""
    from dtivl.solvers.nonlinear import DEFAULT_STEPS

    p_bal = P.get("loads.ivl_balloon_pressure_atm").value
    return {"balloon_only": p_bal / DEFAULT_STEPS["pressure"],
            "balloon_10atm": P.get("loads.poba_pressure_atm").value / DEFAULT_STEPS["pressure"],
            "ivl": p_bal / DEFAULT_STEPS["ivl_inflate"]}.get(proto, float("nan"))


def _onset_str(rec: dict) -> str:
    """Damage onset (first step with a calcium cell at d >= d_crack) and whether it became through-thickness."""
    c = rec["cracks"]
    if c.get("onset_load") is None:
        return "no damage"
    unit = "mm" if (c.get("onset_phase") or "").startswith("stent") else "MPa"
    tail = "" if c["fracture"] else " (partial only)"
    return f"{f(c['onset_load'])} {unit}{tail}"


def _unconv(rec: dict) -> str:
    v = rec.get("n_steps_unconverged_stagger")
    return "not recorded" if v is None else str(v)


def _patterns(c: dict) -> str:
    if not c["pattern"]:
        return "–"
    counts = {}
    for p in c["pattern"]:
        counts[p] = counts.get(p, 0) + 1
    return ", ".join(f"{k} x{v}" if v > 1 else k for k, v in counts.items())


def _exceptions() -> list[tuple[str, str, list]]:
    """Config entries that use literature rows below the main-analysis status (explicit allow_status)."""
    out = []

    def walk(n, pre=""):
        if isinstance(n, dict):
            if "allow_status" in n and "lit" in n:
                out.append((pre, n["lit"], list(n["allow_status"])))
            for k, v in n.items():
                walk(v, f"{pre}.{k}" if pre else k)

    walk(P.load_config())
    return out


# ---------------------------------------------------------------- figures
def fig_onset(nl: dict) -> str:
    """Damage-onset pressure under the balloon-alone and IVL protocols vs calcium modulus."""
    import matplotlib.pyplot as plt

    from dtivl.drivers.figstyle import SERIES, TEXT_2, apply_style, save

    apply_style()
    p_bal = P.get("loads.ivl_balloon_pressure_atm").value
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.6), sharey=True)
    for ax, proto in zip(axs, ("balloon_only", "ivl")):
        for i, (lid, rows) in enumerate(nl.items()):
            E, on, none_E, none_p = [], [], [], []
            for r in rows:
                rec = r.get(proto)
                if not _ok(rec):
                    continue
                c = rec["cracks"]
                if c["fracture"] and c["onset_load"] is not None:
                    E.append(r["ca_E_MPa"])
                    on.append(c["onset_load"])
                else:
                    none_E.append(r["ca_E_MPa"])
                    none_p.append(rec["summary"].get("p_top", rec["summary"].get("p_reached")))
            col = SERIES[i]
            if E:
                ax.plot(E, on, "-o", color=col, label=lid, ms=4)
            if none_E:
                ax.plot(none_E, none_p, "^", mfc="white", mec=col, ms=5,
                        label=f"{lid}: no through-thickness damage (marker = peak load)")
        ax.axhline(p_bal, color=TEXT_2, lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.text(0.02, p_bal, " IVL balloon pressure", color=TEXT_2, fontsize=6, va="bottom",
                transform=ax.get_yaxis_transform())
        ax.set_xlabel("Calcium modulus E (MPa)")
        ax.set_title(protocol_label(proto))
    axs[0].set_ylabel("Damage onset pressure (MPa, step-resolved)")
    axs[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    return str(save(fig, "checkpoint2_pilot_onset").relative_to(ROOT))


def fig_transfer(lt: dict) -> str:
    """IVL transfer factor T_equivalent vs calcium modulus (transient elastodynamics)."""
    import matplotlib.pyplot as plt

    from dtivl.drivers.figstyle import SERIES, apply_style, save

    apply_style()
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    for i, (lid, rec) in enumerate(lt["lesions"].items()):
        E = [r["ca_E_MPa"] for r in rec["by_E"]]
        T = [r["transient"]["T_equivalent"] for r in rec["by_E"]]
        ax.plot(E, T, "-o", color=SERIES[i], label=lid, ms=4)
    ax.set_xscale("log")
    ax.set_xlabel("Calcium modulus E (MPa)")
    ax.set_ylabel("T_equivalent (-)")
    ax.legend(frameon=False)
    return str(save(fig, "checkpoint2_transfer_factor").relative_to(ROOT))


# ---------------------------------------------------------------- facts
def collect_facts(nl: dict) -> dict:
    from dtivl.fracture.cracks import DIFFUSE_REGION_FRAC

    facts = {"failed": [], "skipped": {}, "unconverged": {"recorded": 0, "not_recorded": 0, "steps": 0, "runs": []},
             "diffuse": 0, "diffuse_whole": 0, "discrete": 0, "diffuse_frac": DIFFUSE_REGION_FRAC}
    for lid, lr in nl.items():
        for r in lr:
            E = r["ca_E_MPa"]
            if "skipped" in r:
                facts["skipped"].setdefault(lid, []).append(E)
                continue
            for proto in PROTOCOLS:
                rec = r.get(proto)
                if rec is None:
                    continue
                if not _ok(rec):
                    facts["failed"].append((lid, E, proto, rec))
                nu = rec.get("n_steps_unconverged_stagger")
                if nu is None:
                    facts["unconverged"]["not_recorded"] += 1
                else:
                    facts["unconverged"]["recorded"] += 1
                    facts["unconverged"]["steps"] += nu
                    if nu:
                        facts["unconverged"]["runs"].append((lid, E, proto, nu))
                c = rec["cracks"]
                if c["fracture"]:
                    if c["diffuse"]:
                        facts["diffuse"] += 1
                        facts["diffuse_whole"] += c["damaged_fraction"] >= WHOLE
                    else:
                        facts["discrete"] += 1
    return facts


# ---------------------------------------------------------------- sections
def sec_cohort() -> str:
    cs = read_json("cohort_summary")
    df = pd.read_csv(COHORT_DIR / "cohort.csv")
    g = cs["core_geometry"]
    pb = cs["plaque_burden_check"]
    rows = [[k, v, ", ".join(f"{s}: {n}" for s, n in sorted(cs["fujino_score_counts"].get(k, {}).items()))]
            for k, v in cs["n_per_set"].items()]
    hold = df[df["set"] == "holdout"]
    depth = hold.groupby("morphology")["ca_depth"].agg(["min", "max", "count"])
    n_lipid = int((hold["lipid_arc_deg"].fillna(0) > 0).sum())
    lip_frac = P.raw("design.holdout_lhs.ranges")["lipid_fraction"]
    s = [f"## 1. In silico cohort ({cs['n_total']} lesions)", "",
         table(["Set", "Lesions", "Fujino score: count"], rows), "",
         f"Core reference geometry: lumen radius {f(g['r_lumen_mm'])} mm, plaque thickness "
         f"{f(g['plaque_thickness_mm'])} mm, media + adventitia {f(g['wall_thickness_mm'])} mm, calcium depth "
         f"{f(g['depth_superficial_mm'])} mm (superficial) and {f(g['depth_deep_mm'])} mm (deep).",
         f"Implied plaque burden {f(pb['implied_plaque_burden_pct'])} %, between the mean CT-angiography plaque "
         f"burden of calcified plaques ({_lit('plaque_burden_calcified_plaque_CTA')}) and of calcified nodules "
         f"({_lit('plaque_burden_calcified_nodule_CTA')}): {'yes' if pb['within_range'] else 'NO'}.", "",
         "Hold-out depths now follow the morphology label (deep sheets sample the deeper half of the "
         "literature depth range, other morphologies the shallower half):", "",
         table(["Morphology", "min depth (mm)", "max depth (mm)", "n"],
               [[m, f(r["min"]), f(r["max"]), int(r["count"])] for m, r in depth.iterrows()]), "",
         f"Hold-out lesions with a lipid pool: {n_lipid} of {len(hold)}. The LHS lipid indicator selects a "
         f"fraction {f(lip_frac)} of them; a pool is dropped when the calcium arc leaves no angular room opposite it.", "",
         "Figure: `figures/fig2_cohort_gallery.png` (core factorial set).", ""]
    return "\n".join(s)


def sec_verification() -> str:
    from dtivl.fracture.phasefield import VERIF_TOL

    la = read_json("verification_lame")
    nl = read_json("verification_nonlinear")
    pf = read_json("verification_phasefield")
    gc = read_json("verification_mesh_convergence")
    pf_rows = []
    for c in pf["cases"]:
        interp = abs(c["psi_onset_measured_MPa"] - c["psi_onset_theory_MPa"]) / c["psi_onset_theory_MPa"]
        pf_rows.append([c["case"], f(c["E_MPa"]), f(c["k_floor"]), f(c["psi_onset_theory_MPa"]),
                        f(c["psi_onset_measured_MPa"], 4), pct(interp), pct(c["rel_err_psi"]), f(c["passed"])])
    s = ["## 2. Verification", "", "### 2.1 Linear elasticity: thick-walled cylinder vs Lamé (plane strain)", "",
         table(["Material", "h (mm)", "max rel. error u_r", "max rel. error hoop stress", "passed"],
               [[c["material"], f(c["h_mm"]), pct(c["max_rel_err_ur"]), pct(c["max_rel_err_hoop"]),
                 f(c["passed"])] for c in la["cases"]]),
         f"\nTolerance {pct(la['tolerance_rel'], 2)}.", "",
         "### 2.2 Finite strain: hyperelastic Lamé limit and rigid-cylinder contact", "",
         f"Hyperelastic ({nl['lame']['material']}) small-load limit: max rel. error of u_r "
         f"{pct(nl['lame']['max_rel_err_ur_lumen'])} at the lumen, {pct(nl['lame']['max_rel_err_ur_outer'])} "
         f"at the outer surface; passed: {f(nl['lame']['passed'])}. Contact check passed: "
         f"{f(nl['contact']['passed'])}.", "",
         "### 2.3 Phase-field (AT1) damage onset, homogeneous bar", "",
         table(["Case", "E (MPa)", "k_floor", "psi_c theory (MPa)", "psi onset measured, interpolated (MPa)",
                "rel. error (interpolated)", "rel. error (first damaged step)", "passed"], pf_rows),
         f"\nTolerance {pct(VERIF_TOL, 2)} on the first-damaged-step error (energy and strain).", "",
         "### 2.4 Mesh convergence (three grids; Celik GCI)", ""]
    ratios = "; ".join(f"{lid}: r21 = {f(L['gci'][next(iter(L['gci']))]['r21'])}, r32 = "
                       f"{f(L['gci'][next(iter(L['gci']))]['r32'])}" for lid, L in gc["lesions"].items())
    rows, foot = [], []
    for q, v in gc["max_rel_change_default_vs_fine"].items():
        conv = {lid: L["gci"][q]["convergence"] for lid, L in gc["lesions"].items()}
        mono = [L["gci"][q]["gci_fine21"] for L in gc["lesions"].values() if L["gci"][q]["convergence"] == "monotonic"]
        gci = pct(max(mono)) if mono else "n/a (no monotonic case)"
        rows.append([q, pct(v), gci, ", ".join(f"{k}: {c}" for k, c in conv.items()),
                     "mesh-dependent (not a predictor)" if q in gc["mesh_dependent_quantities"] else
                     ("accepted" if v < gc["rel_change_accept"] else "NOT accepted")])
        if q not in gc["mesh_dependent_quantities"]:
            for lid, L in gc["lesions"].items():
                g = L["gci"][q]
                if g["convergence"] != "monotonic" and g.get("gci_fine21") is not None and g["gci_fine21"] > gc["rel_change_accept"]:
                    foot.append(f"{q} ({lid}, {g['convergence']}): formal GCI {pct(g['gci_fine21'])}, error vs "
                                f"extrapolated value {pct(g.get('rel_err_default_vs_extrapolated'))}")
    s += [f"Nominal refinement factors {', '.join(f(x) for x in gc['factors'])}; measured grid ratios {ratios}.", "",
          table(["Quantity", "max rel. change default vs fine", "max GCI (fine), monotonic cases only",
                 "convergence class", "status"], rows),
          "", f"Acceptance: relative change below {pct(gc['rel_change_accept'], 2)} between the default and "
          "fine grids. The GCI assumes monotonic convergence (Celik et al.); for oscillatory or divergent "
          "triplets the relative change is the criterion. The raw peaks are singular at deposit corners and are "
          "not used as predictors; the nonlocal (critical-distance) peaks replace them."]
    if foot:
        s.append("Non-monotonic accepted quantities whose formal Richardson numbers exceed the criterion (shown "
                 "for transparency; Richardson extrapolation is not meaningful for these triplets): "
                 + "; ".join(foot) + ".")
    s.append("")
    return "\n".join(s)


def sec_transient(lt: dict) -> str:
    dens = [e for e in _exceptions() if e[0].startswith("materials.density")]
    p_lin = next(iter(lt["lesions"].values()))["by_E"][0]["predictors"]["p_lumen_MPa"]
    s = ["## 3. Pilot (3 lesions): linear predictors and the IVL load factor", "",
         f"### 3.1 Linear predictors (pre-treatment analysis) at lumen pressure {f(p_lin)} MPa", "",
         "The threshold-area predictor uses the same failure strain eps_f as the damage model (a shared material "
         "input, as are the geometry and the moduli); no quantity from the damage simulation enters a predictor.", ""]
    rows = []
    for lid, rec in lt["lesions"].items():
        for r in rec["by_E"]:
            p = r["predictors"]
            rows.append([lid, f(r["ca_E_MPa"]), f(p["sigma1_p95"]), f(p["sigma1_peak_nl"]), pct(p["frac_eps_gt_epsf"])])
    s += [table(["Lesion", "E (MPa)", "sigma1 P95 (MPa)", "nonlocal peak sigma1 (MPa)",
                 "calcium area with eps1 > eps_f"], rows), "",
          "### 3.2 IVL load factor (input of the outcome model, not a predictor)", "",
          "T = peak transient maximum principal stress in calcium / static stress of the same (transient-model) "
          "geometry under the peak pulse pressure. T_equivalent = the same transient peak / the static stress "
          "of the quasi-static outcome model, so that the quasi-static IVL step reproduces the transient peak. "
          "T_equivalent sets the pulse load of every IVL outcome run and is excluded from the Phase 5 predictor "
          "set. It depends on tissue densities used under a SECONDARY-status exception ("
          + ", ".join(dict.fromkeys(f"`{lit}`" for _, lit, _ in dens)) + "), so every IVL outcome is conditional on them (D15). "
          "T is mesh-sensitive at the default calcium mesh (notes in `src/solvers/transient.py`) and was not part "
          "of the Section 2.4 convergence study.", ""]
    rows = [[lid, f(r["ca_E_MPa"]), f(r["transient"]["T"]), f(r["transient"]["T_equivalent"]),
             f(r["transient"]["peak_at_window_end"])] for lid, rec in lt["lesions"].items() for r in rec["by_E"]]
    s += [table(["Lesion", "E (MPa)", "T", "T_equivalent", "peak at window end"], rows), "",
          f"Figure: `{fig_transfer(lt)}`.", ""]
    return "\n".join(s)


def _steps_text() -> str:
    from dtivl.solvers.nonlinear import DEFAULT_STEPS

    return (f"pressure ramps use {DEFAULT_STEPS['pressure']} increments ({protocol_label('balloon_only')}: "
            f"{f(_step('balloon_only'))} MPa; {protocol_label('balloon_10atm')}: {f(_step('balloon_10atm'))} MPa), "
            f"IVL inflation {DEFAULT_STEPS['ivl_inflate']} increments to the balloon pressure and "
            f"{DEFAULT_STEPS['ivl_pulse']} to the pulse peak, stent expansion {DEFAULT_STEPS['stent_expand']} "
            "radius increments; an increment is halved only when a step fails")


def sec_nonlinear(nl: dict, facts: dict) -> str:
    cal = read_json("calibration_stent")
    R_ivl = 0.5 * P.get("geometry.ivl_balloon_diameter_mm").value
    d_crack = float(P.raw("fracture.d_crack"))
    s = ["## 4. Pilot (3 lesions): nonlinear phase-field outcomes", "",
         "Assumptions of these protocols: a balloon is a uniform follower pressure on the lumen with a rigid "
         f"envelope cap at the IVL balloon radius ({f(R_ivl)} mm, from `geometry.ivl_balloon_diameter_mm`); the "
         f"plain balloon has the same size as the IVL balloon (as in the cadaver study's balloon arm); "
         f"'damage onset' is the first load step at which a calcium cell reaches d >= {f(d_crack)}; 'fracture' "
         "is a damaged component that spans the calcium thickness and cuts the deposit; a component covering at "
         f"least {pct(facts['diffuse_frac'], 2)} of its calcium region is labelled diffuse (an analysis choice, "
         "decision D10) and still counts as a fracture. Model fracture is not the same quantity as fracture "
         "seen on histology or OCT.",
         f"Stent radial stiffness K_s = {f(cal['K_s_MPa_per_mm'])} MPa/mm, calibrated so that the calcium-free "
         f"template ({cal['lesion_template']}) recoils by {_lit('acute_recoil_EES_CoCr_clinical')}; recoil and "
         "stent expansion are therefore not consistency checks.",
         f"Onsets are resolved only to the load step: {_steps_text()}.", ""]
    s.append(outcome_matrix(nl))
    rows = []
    for lid, lrows in nl.items():
        for r in lrows:
            E = r["ca_E_MPa"]
            if "skipped" in r:
                rows.append([lid, f(E), "all", "skipped", "", r["skipped"], "", "", "", ""])
                continue
            for proto in PROTOCOLS:
                rec = r.get(proto)
                if rec is None:
                    continue
                c, m = rec["cracks"], rec["summary"].get("metrics", {})
                if m and _ok(rec):
                    stent = (f"{f(m.get('stent_expansion_pct'))} % / {f(m.get('acute_recoil_pct'))} % / "
                             f"{f(rec['summary'].get('P_dep'))} MPa")
                elif m:
                    sm = rec["summary"]
                    stent = (f"not valid: solver failed in stent expansion at R = {f(sm.get('R_dep'))} mm "
                             f"(mean contact pressure {f(sm.get('P_dep'))} MPa, {pct(sm['P_dep'] / sm['p_rated'], 2)} "
                             "of rated)")
                else:
                    stent = ""
                rows.append([lid, f(E), protocol_label(proto), rec["status"], _unconv(rec), _onset_str(rec),
                             f"{c['n_fractures']} / {c['n_partial']}", _patterns(c), pct(c["damaged_fraction"], 3),
                             f(rec.get("max_lumen_area")), stent])
    s += ["### 4.2 All protocol runs", "",
          table(["Lesion", "E (MPa)", "Protocol", "Status", "unconverged (u, d) steps", "Damage onset",
                 "through-thickness / partial components", "patterns", "Damaged calcium area",
                 "Max lumen area (mm²)", "Stent expansion / recoil / mean contact pressure at deployment"], rows), "",
          f"Figure: `{fig_onset(nl)}`.", ""]
    return "\n".join(s)


def outcome_matrix(nl: dict) -> str:
    """Modulus x lesion matrix of fracture outcomes."""
    Es = sorted({r["ca_E_MPa"] for lr in nl.values() for r in lr})
    tags = (("balloon_only", "B"), ("balloon_10atm", "P"), ("ivl", "I"), ("stent_no_ivl", "S"))
    n_stent = len({r["ca_E_MPa"] for lr in nl.values() for r in lr if "stent_no_ivl" in r})
    rows = []
    for E in Es:
        cells = []
        for lid, lr in nl.items():
            r = _row(lr, E)
            if r is None:
                cells.append("not run")
                continue
            if "skipped" in r:
                cells.append("skipped")
                continue
            parts = []
            for key, tag in tags:
                rec = r.get(key)
                if rec is None:
                    continue
                if not _ok(rec):
                    parts.append(f"{tag}: {rec['status']}")
                    continue
                c = rec["cracks"]
                mark = "–" if not c["fracture"] else ("diffuse" if c["diffuse"] else f"{c['n_fractures']} crack(s)")
                parts.append(f"{tag}: {mark}")
            cells.append("; ".join(parts))
        rows.append([f(E)] + cells)
    return "\n".join([
        "### 4.1 Fracture outcome by calcium modulus", "",
        f"B = {protocol_label('balloon_only')}, P = {protocol_label('balloon_10atm')}, I = IVL (balloon + pulse), "
        f"S = stent without IVL (run at {n_stent} moduli); – = no through-thickness damage; 'failed' = the solver "
        "stopped before the end of the protocol (no outcome).", "",
        table(["E (MPa)"] + list(nl), rows), ""])


def sec_example3d() -> str:
    ex = _load("pilot_example3d")
    if ex is None:
        return "## 5. 3D Fujino-4 example\n\nNot run.\n"
    s = [f"## 5. 3D Fujino-4 example ({ex['lesion']}) vs its 2D mid-section", "",
         f"Regularisation length in this example: {f(ex['ell_mm'])} mm (coarser 3D mesh). 'not run' = the analysis "
         "was not completed for that model and modulus (see the note below the table).", ""]
    rows = []
    for key in ("3d", "2d"):
        m = ex["models"].get(key)
        if m is None:
            continue
        for r in m["by_E"]:
            p = r["predictors"]
            t = r.get("transient")
            ivl = r.get("ivl")
            rows.append([key.upper(), f(r["ca_E_MPa"]), m["mesh"]["n_cells"], f(p["sigma1_p95"]),
                         f(p["sigma1_peak_nl"]), f(p["sed_mean"]),
                         "not run" if t is None else f(t.get("T_equivalent")),
                         "not run" if ivl is None else f"{ivl['status']}, {_onset_str(ivl)}",
                         "not run" if ivl is None else pct(ivl["cracks"]["damaged_fraction"], 3),
                         "not run" if ivl is None else f(ivl["seconds"], 2)])
    s += [table(["Model", "E (MPa)", "cells", "sigma1 P95 (MPa)", "nonlocal peak sigma1 (MPa)",
                 "mean SED (MPa)", "T_equivalent", "IVL: status, damage onset", "IVL damaged area",
                 "IVL run time (s)"], rows), "",
          "The full 3D IVL protocol at the tissue-level baseline modulus was stopped before completion because "
          "its damage-propagation steps were too slow; it was replaced by the onset run below."
          + ("" if any("ivl" in r for r in ex["models"].get("3d", {}).get("by_E", [])) else
             " The 3D IVL run at the high modulus had not finished when this report was built (it runs alone "
             "because two 3D runs together exceed the memory limit); the report is regenerated when it finishes."),
          ""]
    on = _load("pilot_example3d_onset")
    if on is not None:
        rows = []
        for key in ("3d", "2d"):
            m = on["models"].get(key)
            if m is None:
                continue
            for r in m["by_E"]:
                o = r["onset"]
                nu = sum(1 for st in r["steps"] if st.get("stagger_converged") is False)
                c = None if o is None else o["cracks"]
                rows.append([key.upper(), f(r["ca_E_MPa"]), r["status"], "–" if o is None else f(o["p"]),
                             "–" if c is None else pct(c["damaged_fraction"], 3),
                             "–" if c is None else f(c["fracture"]),
                             "–" if c is None else pct(c["max_depth_frac"], 3),
                             "–" if c is None else ", ".join(c["pattern"]) or "–", nu, f(r["seconds"], 2)])
        s += [f"Balloon-pressure ramp to damage onset ({on['n_steps']} increments up to {f(on['p_end_MPa'])} MPa, "
              "IVL balloon cap), 3D vs the 2D mid-section (`results/pilot_example3d_onset.json`):", "",
              table(["Model", "E (MPa)", "status", "damage onset p (MPa)", "damaged area at onset",
                     "through-thickness at onset", "max depth at onset", "pattern at onset",
                     "unconverged (u, d) steps", "run time (s)"], rows), ""]
    return "\n".join(s)


def sec_findings(facts: dict, nl: dict) -> str:
    ms = MaterialSet.baseline()
    u = facts["unconverged"]
    lines = ["## 6. What the pilot shows", ""]
    for lid in nl:
        for proto in ("balloon_only", "balloon_10atm", "ivl"):
            ok, hit, bad = _sets(nl, proto)[lid]
            if not ok and not bad:
                continue
            lines.append(f"* {lid}, {protocol_label(proto)}: through-thickness damage at E = {_emp(hit)}; none at "
                         f"E = {_emp([e for e in ok if e not in hit])}"
                         + (f"; failed at E = {_emp(bad)}" if bad else "") + ".")
    lines += ["",
              f"* Baseline calcium modulus E = {f(ms.ca_E)} MPa (secant modulus at failure of tissue-level "
              f"calcification; failure stress {_lit('calcification_fracture_stress')}). Fractured outcomes that are "
              f"diffuse: {facts['diffuse']} (of which {facts['diffuse_whole']} cover the whole deposit); with "
              f"discrete cracks: {facts['discrete']}.",
              "* Moduli skipped because the calcium is not stiffer than the fibrous matrix (fibrous modulus "
              f"{_lit('intima_hypocellular_tangential_modulus')}): "
              + ("; ".join(f"{k}: {_emp(v)}" for k, v in facts["skipped"].items()) or "none") + ".",
              "* Protocol runs that did not finish: " + ("; ".join(
                  f"{lid} E = {f(E)} MPa {p} ({rec['status']}"
                  + (f", in stent expansion at R = {f(rec['summary'].get('R_dep'))} mm" if p.startswith("stent") else
                     f", phase {rec['summary'].get('failed_phase')}") + ")"
                  for lid, E, p, rec in facts["failed"]) or "none") + ".",
              f"* Staggered (u, d) loop at its iteration limit (step accepted with a warning): {u['steps']} steps in "
              f"{len(u['runs'])} runs (" + ("; ".join(f"{lid} E = {f(E)} MPa {p}: {n}" for lid, E, p, n in u["runs"])
                                            or "none") + f"). The flag was added during the pilot and is not recorded "
              f"for {u['not_recorded']} of {u['recorded'] + u['not_recorded']} runs; it will be recorded for every "
              "run of the full cohort.", ""]
    return "\n".join(lines)


def sec_consistency(nl: dict) -> str:
    """Comparison with published observations (qualitative; no patient data)."""
    df = pd.read_csv(COHORT_DIR / "cohort.csv").set_index("lesion_id")
    L = P.load_literature()
    a_cut = L["fracture_angle_cutoff_balloon"].value
    t_cut = L["fracture_thickness_cutoff_balloon"].value
    rows = []
    phys = [r["physiological"]["cracks"]["fracture"] for lr in nl.values() for r in lr if _ok(r.get("physiological"))]
    rows.append(["No fracture at physiological pressure", f"expected; pressure {_lit('normal_systolic_blood_pressure_upper')}",
                 f"{sum(phys)} of {len(phys)} runs fracture", "agrees" if not any(phys) else "disagrees"])
    stent = _sets(nl, "stent_no_ivl")
    for lid in nl:
        arc, thk = df.loc[lid, "ca_arc_total_deg"], df.loc[lid, "ca_thickness_max"]
        expect = bool(arc > a_cut and thk < t_cut)
        by_thk = arc > a_cut and not thk < t_cut
        ok, hit, bad = stent[lid]
        agree = [e for e in ok if (e in hit) == expect]
        dis = [e for e in ok if (e in hit) != expect]
        rows.append([f"Fracture after high-pressure dilation without IVL predicted by arc > {f(a_cut)} deg and "
                     f"thickness < {f(t_cut)} mm ({lid}: arc {f(arc)} deg, max thickness {f(thk)} mm, so "
                     f"{'fracture' if expect else 'no fracture'} expected"
                     + ("; decided by the SECONDARY thickness cut-off" if by_thk else "") + ")",
                     f"{_lit('fracture_angle_cutoff_balloon')}; {_lit('fracture_thickness_cutoff_balloon')}",
                     f"stent without IVL fractures at E = {_emp(hit)} of {_emp(ok)}"
                     + (f"; failed (excluded) at {_emp(bad)}" if bad else ""),
                     f"agrees at {_emp(agree)}; disagrees at {_emp(dis)}"])
    poba, ivl = _sets(nl, "balloon_10atm"), _sets(nl, "ivl")
    for lid in nl:
        both = [e for e in poba[lid][0] if e in ivl[lid][0]]
        agree = [e for e in both if e not in poba[lid][1] and e in ivl[lid][1]]
        rows.append([f"Plain balloon at {_atm(P.get('loads.poba_pressure_atm').value)} does not fracture calcium; "
                     f"IVL does (cadaver histology) — {lid}",
                     _lit("calcium_fracture_rate_POBA_10atm_cadaver_histology"),
                     f"plain balloon fractures at E = {_emp(poba[lid][1])}; IVL at E = {_emp(ivl[lid][1])}",
                     f"met at {_emp(agree)}; not met at {_emp([e for e in both if e not in agree])}"])
    nod = [l for l in nl if df.loc[l, "morphology"] == "nodule"]
    sheets = [l for l in nl if l not in nod]
    if nod and sheets:
        same = all(set(ivl[n][1]) == set(ivl[sh][1]) or set(ivl[n][1]) >= set(ivl[sh][1]) for n in nod for sh in sheets)
        rows.append(["Nodular calcium fractures less often after IVL than sheets",
                     _lit("nodular_calcium_odds_ratio_for_fracture"),
                     "; ".join(f"{l}: IVL fracture at E = {_emp(ivl[l][1])}" for l in nl),
                     "disagrees (no nodule effect in the pilot)" if same else "partly (see moduli)"])
    comps = [r["ivl"]["cracks"] for lr in nl.values() for r in lr if _ok(r.get("ivl"))
             and (r["ivl"]["cracks"]["fracture"] or r["ivl"]["cracks"]["n_partial"])]
    n_thr = sum(c["n_fractures"] for c in comps)
    n_all = n_thr + sum(c["n_partial"] for c in comps)
    rows.append(["Most IVL fractures are full thickness", _lit("full_thickness_fracture_fraction_pooled_IVL_OCT"),
                 f"IVL protocol: {n_thr} of {n_all} damaged components through-thickness, "
                 f"{sum(c['n_diffuse'] for c in comps)} of them diffuse",
                 "not comparable while damage is diffuse (D10)"])
    rows.append(["Several fractures per lesion", _lit("fractures_per_lesion_pooled_IVL_OCT"),
                 "discrete crack count not defined when damage is diffuse", "not comparable (D10)"])
    exp = {round(r[k]["summary"]["metrics"]["stent_expansion_pct"], 6) for lr in nl.values() for r in lr
           for k in ("stent_no_ivl", "stent_post_ivl") if _ok(r.get(k)) and r[k]["summary"].get("metrics")}
    rows.append(["Stent expansion at the minimum stent area", _lit("stent_expansion_MSA_site_DisruptCAD_III"),
                 f"{f(min(exp))}-{f(max(exp))} % ({len(exp)} distinct values)" if exp else "–",
                 "not a check: set by the target and reference diameters and the recoil calibration"])
    gain = [(lid, r["ca_E_MPa"], r["ivl"]["final"]["lumen_area"], r["stent_no_ivl"]["summary"]["metrics"]["lumen_area_pre"])
            for lid, lr in nl.items() for r in lr if _ok(r.get("stent_no_ivl")) and _ok(r.get("ivl"))]
    rows.append(["Lumen gain after IVL alone (before stenting)", _lit("COCALP_MLA_pre_post_IVL_post_stent"),
                 "; ".join(f"{lid} E = {f(E)}: {f(a)} vs {f(b)} mm² (after IVL vs no IVL)" for lid, E, a, b in gain),
                 "disagrees: no residual gain (hyperelastic model with damage, no permanent set)"])
    return "\n".join(["## 7. Consistency with published observations (preliminary, qualitative)", "",
                      "These are comparisons with published cohorts, not validation: no patient-specific outcome "
                      "data exist for these idealised lesions.", "",
                      table(["Observation", "Published value [status]", "Pilot", "Agreement"], rows), ""])


def sec_diagnostics() -> str:
    """Damage localisation (decision D10), from the diagnostics result files."""
    d = _load("diag_localisation")
    e = _load("diag_ell")
    s = ["## 8. Why the damage is diffuse (diagnostics for D10)", ""]
    if d is not None:
        rows = []
        for r in d["by_E"]:
            for k, lab in (("fine", "fine steps"), ("seeded", "seeded radial crack"),
                           ("heterogeneous", "G_c heterogeneity")):
                x = r[k]
                first = next((st for st in x["steps"] if st.get("damaged_fraction", 0.0) > 0.0), None)
                nu = sum(1 for st in x["steps"] if st.get("stagger_converged") is False)
                rows.append([f(r["ca_E_MPa"]), lab, f(x["first_damaged_p"]),
                             "–" if first is None else pct(first["damaged_fraction"], 3),
                             "–" if first is None else _patterns(first),
                             pct(x["final_damaged_fraction"], 3), f(x["final_diffuse"]), nu])
        g = d["by_E"][0]["heterogeneous"].get("gc_factor_calcium", {})
        s += [f"Lesion {d['lesion']}, IVL balloon cap, pressure steps of {f(d['dp_MPa'])} MPa up to "
              f"{f(d['p_end_MPa'])} MPa (`results/diag_localisation.json`). G_c heterogeneity: nominal "
              f"coefficient of variation {pct(d['cov'], 2)}, realised {pct(g.get('sd', float('nan')) / g.get('mean', 1.0), 3)} "
              f"(factor clipped to {f(g.get('min'))}-{f(g.get('max'))}, correlation length {f(g.get('corr_len_mm'))} mm).", "",
              table(["E (MPa)", "Experiment", "first cracked step p (MPa)", "damaged area at that step",
                     "pattern at that step", "damaged area at the end", "diffuse at the end",
                     "steps with unconverged (u, d) loop"], rows), ""]
    if e is not None:
        rows = []
        for h in e["by_h"]:
            for r in h["by_E"]:
                last = r["steps"][-1] if r["steps"] else {}
                nu = sum(1 for st in r["steps"] if st.get("stagger_converged") is False)
                rows.append([f(h["h_ca_mm"]), f(h["ell_mm"]), h["n_cells_calcium"], f(r["ca_E_MPa"]),
                             f(r["first_damaged_p"]), pct(r["final_damaged_fraction"], 3), f(r["final_diffuse"]),
                             f(last.get("n_fractures")), _patterns(last) if last else "–", nu, f(r["seconds"], 2)])
        s += ["Regularisation length refined with the calcium mesh (ell / h fixed; `results/diag_ell.json`):", "",
              table(["h_Ca (mm)", "ell (mm)", "calcium cells", "E (MPa)", "first cracked p (MPa)",
                     "damaged area at the end", "diffuse", "through-thickness components", "patterns",
                     "unconverged (u, d) steps", "run time (s)"], rows), ""]
    return "\n".join(s)


def decisions(facts: dict, nl: dict) -> str:
    from dtivl.geometry.fujino import fujino_criteria

    ms = MaterialSet.baseline()
    p_bal = P.get("loads.ivl_balloon_pressure_atm").value
    p_poba = P.get("loads.poba_pressure_atm").value
    bal, poba, ivl = _sets(nl, "balloon_only"), _sets(nl, "balloon_10atm"), _sets(nl, "ivl")
    common = sorted(set.intersection(*[set(bal[l][0]) & set(ivl[l][0]) for l in nl])) if nl else []
    selective = [E for E in common if not any(E in bal[l][1] for l in nl) and any(E in ivl[l][1] for l in nl)]
    common10 = sorted(set.intersection(*[set(poba[l][0]) & set(ivl[l][0]) for l in nl])) if nl else []
    selective10 = [E for E in common10 if not any(E in poba[l][1] for l in nl) and any(E in ivl[l][1] for l in nl)]
    L = P.load_literature()
    nano = L["calcification_modulus_nanoindentation_carotid"]
    maxE = max((r["ca_E_MPa"] for lr in nl.values() for r in lr), default=float("nan"))
    lines = ["## 0. Decisions needed before the full cohort run", ""]

    # ---- D8
    rows = [[l, _elist(bal[l][1]), _elist(poba[l][1]), _elist(ivl[l][1]),
             _elist([e for e in ivl[l][0] if e not in ivl[l][1]])] for l in nl]
    base = {}
    d = _load("diag_localisation")
    for l, lr in nl.items():
        r = _row(lr, ms.ca_E)
        if r is not None and _ok(r.get("balloon_only")) and r["balloon_only"]["cracks"]["onset_load"] is not None:
            base[l] = r["balloon_only"]["cracks"]["onset_load"]
    fine = None
    if d is not None:
        fine = next((r["fine"]["first_damaged_p"] for r in d["by_E"] if abs(r["ca_E_MPa"] - ms.ca_E) < 1e-9), None)
    lines += ["**D8: calcium modulus regime (the main decision).** The pilot outcome depends on the calcium modulus "
              "more than on anything else (Section 4.1). Moduli (MPa) at which each protocol produces "
              "through-thickness damage (completed runs only):", "",
              table(["Lesion", protocol_label("balloon_only"), protocol_label("balloon_10atm"),
                     "IVL (balloon + pulse)", "no through-thickness damage even with IVL"], rows), "",
              f"* Baseline E = {f(ms.ca_E)} MPa (secant modulus at failure of tissue-level calcification, failure "
              f"stress {_lit('calcification_fracture_stress')}): the {protocol_label('balloon_only')} damages every "
              "pilot lesion through its thickness, with damage onset between "
              + "; ".join(f"{l}: {f(v - _step('balloon_only'))} and {f(v)} MPa" for l, v in base.items())
              + (f" (fine steps on {d['lesion']}: {f(fine)} MPa = {_atm(fine)})" if fine is not None else "")
              + ". IVL then adds nothing. Every pilot lesion has the same outcome; if that holds for the cohort, "
              "the Phase 5 ROC analysis would have no non-fracture cases.",
              f"* Published contrast: {_litn('calcium_fracture_rate_POBA_10atm_cadaver_histology')}. Measured moduli: "
              f"{_litn('calcified_tissue_tangential_modulus')}; {_litn('calcification_modulus_nanoindentation_carotid')}.",
              f"* Moduli at which no pilot lesion has through-thickness damage under the "
              f"{protocol_label('balloon_only')} but at least one has it with IVL: {_emp(selective)}. With the "
              f"{protocol_label('balloon_10atm')} of the cadaver study instead: {_emp(selective10)}.", "",
              "Options:",
              "  (a) keep the tissue-level baseline and report that the model then cannot separate lesions;",
              f"  (b) use a GPa-level baseline: up to the largest tested modulus ({f(maxE)} MPa, "
              f"{'below' if maxE < nano.value else 'within'} the carotid nanoindentation value, which was not "
              "tested) no pilot lesion fractures even with IVL, which leaves nothing to predict and contradicts the "
              "clinical IVL fracture rates;",
              "  (c) calibrate the modulus to a criterion and report the whole sweep as the sensitivity analysis, "
              "keeping the strain criterion of D5 (failure stress = E x eps_f).", ""]
    cand = selective10 or selective
    if cand:
        Ec = cand[-1]
        sig = MaterialSet.baseline(ca_E=Ec).sigma_crit
        on = {l: (_row(lr, Ec) or {}).get("ivl") for l, lr in nl.items()}
        p10 = [l for l in nl if Ec in poba[l][1]]
        uc = [(l, p, n) for l, E, p, n in facts["unconverged"]["runs"] if abs(E - Ec) < 1e-9]
        lines += [f"  For (c) the only criterion the pilot can meet is a model-internal surrogate: inflating the IVL "
                  f"balloon ({_atm(p_bal)}) without pulses should not fracture the calcium, IVL should. It is a modelling assumption, not a published observation. "
                  f"Candidate: {f(Ec)} MPa. There the {protocol_label('balloon_10atm')} fractures "
                  f"{len(p10)} of {len(nl)} pilot lesions ({', '.join(p10) or 'none'}), so the published cadaver "
                  f"contrast is {'not ' if p10 else ''}met (D14 explains why). The implied failure stress is "
                  f"{f(sig)} MPa, against the measured tissue-level strength {_lit('calcification_fracture_stress')}. "
                  "IVL damage onset at the candidate: "
                  + "; ".join(f"{l}: {f(v['cracks']['onset_load'])} MPa" if _ok(v) and v["cracks"]["fracture"]
                              else f"{l}: none" for l, v in on.items())
                  + (". Runs at the candidate with unconverged (u, d) steps: "
                     + "; ".join(f"{l} {p}: {n}" for l, p, n in uc) if uc else "") + ".",
                  "  The comparisons that stay independent of this choice are the nodule effect, the arc/thickness "
                  "cut-offs and the fracture rates; stent expansion and recoil are not checks (Section 4).", "",
                  "  Recommendation: none of the three is satisfactory. I lean to (c) with the candidate above, "
                  "labelled as calibration to a model-internal surrogate, with (a) and the full sweep reported as "
                  "sensitivity, because only an intermediate modulus gives outcomes that can differ between "
                  "lesions. The alternative is to report the regime dependence itself as the main finding. Your call.", ""]
    else:
        lines += ["  No tested modulus meets even the model-internal surrogate criterion.", ""]

    # ---- D9
    loree = L["calcified_tissue_tangential_modulus"]
    sk = sorted({e for v in facts["skipped"].values() for e in v})
    k_c = MaterialSet.baseline(ca_E=cand[-1]).k_floor if cand else None
    lines += [f"**D9: crack filler (new assumption).** A fully damaged calcium band keeps the shear stiffness of the "
              f"fibrous matrix (degradation floor k = mu_fibrous / mu_calcium = {f(ms.k_floor)} at the tissue-level "
              f"baseline" + (f", {f(k_c)} at {f(cand[-1])} MPa" if k_c is not None else "") + ") instead of becoming "
              "a void. Without it the displacement solver failed in stiff-calcium runs once a deposit cracked (seen "
              "during development; those runs were not kept). G_c is rescaled so that the verified onset criterion "
              f"is unchanged (Section 2.3). The filler is undefined when calcium is not stiffer than the matrix, so "
              f"the moduli {_emp(sk)} were skipped"
              + (f"; this includes the measured tissue-level modulus `calcified_tissue_tangential_modulus` "
                 f"({loree.value_str} {loree.unit}), the lower bound of the sensitivity range"
                 if any(abs(e - loree.value) < 1e-6 for e in sk) else "")
              + ". Proposed sensitivity: k = k_residual (no filler) where the solver allows. Approve, or ask for an "
              "alternative.", ""]

    # ---- D10
    e = _load("diag_ell")
    if d is not None and e is not None:
        onset = {(h["ell_mm"], round(r["ca_E_MPa"], 6)): r for h in e["by_h"] for r in h["by_E"]}
        Es = sorted({k[1] for k in onset})
        ells = sorted({k[0] for k in onset}, reverse=True)
        gcc = _load("verification_mesh_convergence")["rel_change_accept"]
        sens = []
        for E in Es:
            ps = [(l, onset[(l, E)]) for l in ells if (l, E) in onset]
            p0, p1 = ps[0][1]["first_damaged_p"], ps[-1][1]["first_damaged_p"]
            nu = sum(1 for _, r in ps for st in r["steps"] if st.get("stagger_converged") is False)
            sens.append(f"E = {f(E)} MPa: {', '.join(f(r['first_damaged_p']) for _, r in ps)} MPa for ell = "
                        f"{', '.join(f(l) for l, _ in ps)} mm (change {pct((p1 - p0) / p0, 2)}; unconverged (u, d) "
                        f"steps in these runs: {nu})")
        firsts = []
        for r in d["by_E"]:
            st = next((x for x in r["fine"]["steps"] if x.get("damaged_fraction", 0.0) > 0.0), None)
            if st is not None:
                firsts.append(f"at E = {f(r['ca_E_MPa'])} MPa the first cracked step has {pct(st['damaged_fraction'], 3)} "
                              f"of the calcium damaged ({_patterns(st)})")
        g = d["by_E"][0]["heterogeneous"].get("gc_factor_calcium", {})
        diffuse_all = all(r[k]["final_diffuse"] for r in d["by_E"] for k in ("fine", "seeded", "heterogeneous")) and \
            all(r["final_diffuse"] for h in e["by_h"] for r in h["by_E"])
        lines += ["**D10: fracture outcomes and the damage model.** Every fractured pilot run is diffuse "
                  f"({facts['diffuse']} runs; {facts['diffuse_whole']} cover the whole deposit; {facts['discrete']} "
                  "have discrete cracks). Diagnostics (Section 8): finer load steps, a seeded radial crack, a "
                  f"toughness heterogeneity (nominal coefficient of variation {pct(d['cov'], 2)}, realised "
                  f"{pct(g.get('sd', float('nan')) / g.get('mean', 1.0), 3)}) and a regularisation length reduced from "
                  f"{f(max(ells))} to {f(min(ells))} mm. End state diffuse in all of them: "
                  f"{'yes' if diffuse_all else 'no'}. " + "; ".join(x[0].upper() + x[1:] for x in firsts[:1])
                  + ("; " + "; ".join(firsts[1:]) if firsts[1:] else "") + ". My explanation, not yet tested: the volumetric-deviatoric energy split "
                  "lets damage degrade a whole high-energy layer instead of opening a crack perpendicular to the "
                  "hoop tension; option (b) would test it. Sensitivity of the damage onset to the regularisation "
                  f"length: {'; '.join(sens)}; compare the {pct(gcc, 2)} mesh criterion of Section 2.4.",
                  "Options:",
                  "  (a) keep the model; primary outcomes = through-thickness damage yes/no and damage-onset pressure "
                  "(resolved to the load step, and sensitive to the regularisation length as shown); damaged fraction "
                  "is almost always the whole deposit and carries little information; crack number and "
                  "radial/circumferential pattern not reported;",
                  "  (b) replace the energy split by a principal-strain (spectral) driving force, which opens cracks "
                  "perpendicular to the maximum principal strain, and re-test localisation on the pilot before the "
                  "full run;",
                  "  Recommendation: (a) for the main analysis plus a time-boxed (b) on the pilot lesions; if (b) "
                  "gives discrete cracks, crack number and pattern become secondary outcomes.", ""]

    # ---- D11
    cs = read_json("cohort_summary")
    core = cs["fujino_score_counts"]["core"]
    thr = fujino_criteria()["thickness"].threshold
    df = pd.read_csv(COHORT_DIR / "cohort.csv")
    c = df[df["set"] == "core"]
    over = c[c["ca_thickness_max"] > thr]
    lines += ["**D11: Fujino range.** Core-set Fujino scores: " + ", ".join(f"{k}: {v}" for k, v in sorted(core.items()))
              + f". Only {len(over)} core lesions exceed the Fujino thickness threshold ({f(thr)} mm): "
              + ", ".join(sorted({f"{m} with thickness {f(t)} mm" for m, t in zip(over['morphology'], over['ca_thickness'])}))
              + " (the protrusion counts); 2D has no length criterion. The Phase 5 comparison with Fujino is then "
              "mostly a comparison with the arc criterion. IVL-treated lesions in Disrupt CAD III had a maximum "
              f"thickness of {_lit('max_calcium_thickness_DisruptCAD_III_OCT')}. Options: keep the D2 thickness "
              "levels and state the limitation, or add a thicker level (value for you to choose; +"
              f"{len(P.raw('design.core_factorial.arc_deg')) * len(P.raw('design.core_factorial.morphology'))} "
              "core lesions).", ""]

    # ---- D12
    ex = _load("pilot_example3d")
    t3 = [(r["ca_E_MPa"], r["ivl"]["seconds"]) for r in (ex or {}).get("models", {}).get("3d", {}).get("by_E", []) if "ivl" in r]
    t2 = [(r["ca_E_MPa"], r["ivl"]["seconds"]) for r in (ex or {}).get("models", {}).get("2d", {}).get("by_E", []) if "ivl" in r]
    on3 = _load("pilot_example3d_onset")
    lines += ["**D12: 3D length series (D3).** I propose running the "
              f"{'/'.join(f(x) for x in P.raw('design.lengths_3d_mm'))} mm series after this checkpoint on a subset "
              "you choose, and in 3D only up to damage onset. Run times on the Fujino-4 example: full IVL "
              + ("; ".join(f"3D at {f(E)} MPa {f(t, 2)} s" for E, t in t3) or "3D: not completed") + "; "
              + "; ".join(f"2D section at {f(E)} MPa {f(t, 2)} s" for E, t in t2)
              + ((". Onset-only run: " + "; ".join(f"{k.upper()} at {f(r['ca_E_MPa'])} MPa {f(r['seconds'], 2)} s"
                                                   for k, m in on3["models"].items() for r in m["by_E"])) if on3 else "")
              + ". The 3D baseline-modulus IVL run was stopped (Section 5).", ""]

    # ---- D13
    ero = []
    for lid, lr in nl.items():
        for r in lr:
            if _ok(r.get("ivl_erosion")) and _ok(r.get("ivl")):
                a, b = r["ivl"]["cracks"], r["ivl_erosion"]["cracks"]
                ero.append(f"{lid} E = {f(r['ca_E_MPa'])} MPa: phase field {_onset_str(r['ivl'])}, "
                           f"{pct(a['damaged_fraction'], 3)} damaged; erosion {_onset_str(r['ivl_erosion'])}, "
                           f"{pct(b['damaged_fraction'], 3)} damaged, {b['n_partial']} partial components")
    lines += ["**D13: erosion comparator.** It now completes. Under IVL: " + "; ".join(ero) +
              f". Where both damage the calcium, the onsets differ by about one load increment "
              f"({f(_step('ivl'))} MPa). Erosion uses the cell-averaged maximum principal strain, the phase field "
              "an energy criterion; they disagree at the stiff modulus for the deep sheet, where only erosion "
              "damages the calcium. Keep erosion as the subset comparator?", ""]

    # ---- D14
    lt = read_json("pilot_linear_transient")
    peak, scale = P.get("loads.ivl_peak_pressure_MPa"), P.get("loads.ivl_pulse_scale")
    T_star = (p_poba - p_bal) / (peak.value * scale.value)
    above = {l: [r["ca_E_MPa"] for r in rec["by_E"] if r["transient"]["T_equivalent"] > T_star]
             for l, rec in lt["lesions"].items()}
    tmax = {l: max(r["transient"]["T_equivalent"] for r in rec["by_E"]) for l, rec in lt["lesions"].items()}
    lines += ["**D14: how IVL is represented (assumption to accept or change).** The quasi-static IVL step adds "
              f"T_equivalent x peak pressure {_lit('ivl_peak_positive_pressure')} x pulse scale {_q('loads.ivl_pulse_scale')} "
              f"to the balloon pressure {_lit('ivl_balloon_inflation_pressure')}, once (single-pulse peak "
              f"equivalence, an assumption). It therefore exceeds the {protocol_label('balloon_10atm')} only where "
              f"T_equivalent > {f(T_star)}. Moduli where that holds: "
              + "; ".join(f"{l}: {_emp(v)} (max T_equivalent {f(tmax[l])})" for l, v in above.items())
              + ". Where it does not hold, the model cannot reproduce 'IVL fractures what a plain balloon at that "
              "pressure does not', whatever the modulus. The model has no cumulative effect of repeated pulses "
              "(no fatigue data for calcified plaque were found; `calcium_fatigue_SN_data`). Options: (a) keep the "
              "single-pulse equivalence and state the limitation; (b) add a pulse-count damage accumulation law, an "
              "unverified assumption; (c) treat the pulse scale as a calibration parameter. Recommendation: (a).", ""]

    # ---- D15
    exc = _exceptions()
    lines += ["**D15: inputs used under a SECONDARY-status exception.** Rule 2 keeps SECONDARY values out of the "
              "main analysis unless you approve an exception. Current exceptions: "
              + "; ".join(f"`{path}` <- {_litn(lit)}" for path, lit, _ in exc)
              + ". The densities enter T_equivalent and therefore every IVL outcome; the physiological pressure "
              "enters the physiological protocol. Options: approve them with a density sensitivity run (for "
              "example the alternative calcium density in that row's range column), or find verified sources "
              "first.", ""]
    return "\n".join(lines)


def sec_plan(nl: dict) -> str:
    """Run-time based cost estimate of the full 2D cohort, and the plan after approval."""
    import os

    cs = read_json("cohort_summary")
    n_run = cs["n_per_set"]["core"] + cs["n_per_set"]["holdout"]
    secs = {}
    for lr in nl.values():
        for r in lr:
            for k in PROTOCOLS:
                if k in r and "seconds" in r[k]:
                    secs.setdefault(k, []).append(r[k]["seconds"])
    rows = [[protocol_label(k), len(v), f(float(np.median(v)), 2), f(float(np.max(v)), 2)] for k, v in secs.items()]
    med = sum(float(np.median(v)) for k, v in secs.items() if k != "ivl_erosion")
    mx = sum(float(np.max(v)) for k, v in secs.items() if k != "ivl_erosion")
    ncpu = os.cpu_count() or 1
    return "\n".join([
        "## 9. Cost of the full cohort and plan after approval", "",
        "Wall-clock time per pilot protocol run (2D, one process per run; several pilot processes shared the "
        "machine, so these are upper-side estimates; all moduli pooled):", "",
        table(["Protocol", "runs", "median (s)", "max (s)"], rows), "",
        f"One lesion at one modulus with every protocol except erosion: {f(med / 3600, 2)} h (medians) to "
        f"{f(mx / 3600, 2)} h (maxima) of wall-clock time. The {n_run} core and hold-out lesions at one baseline "
        f"modulus then need about {f(n_run * med / 3600, 2)}-{f(n_run * mx / 3600, 2)} process-hours, i.e. "
        f"{f(n_run * med / 3600 / ncpu, 2)}-{f(n_run * mx / 3600 / ncpu, 2)} h on the {ncpu} cores of this "
        "container, before the sensitivity analysis. Stiff-calcium stent runs dominate the upper bound.", "",
        "Plan after your decisions: (1) fix the nodule stent solver failure at stiff calcium (Section 4) and "
        "re-run that case; (2) run the core and hold-out sets at the chosen baseline (2D, all protocols, with the "
        "convergence flag recorded for every run); (3) sensitivity analysis (Morris screening, then Sobol on the "
        "influential inputs) over calcium modulus, failure strain, pulse scale, regularisation length, crack "
        "filler, densities and perivascular support; (4) Phase 5 metrics (ROC/Youden with bootstrap CIs, LOOCV, "
        "composite index, comparison with Fujino); (5) 3D length series on the subset you choose. No full-cohort "
        "run starts before your approval.", ""])


def build() -> str:
    from dtivl.drivers.stage_pilot import merge_example3d

    if any((RESULTS_DIR / f"pilot_example3d_part{i}.json").exists() for i in (2, 3, 4)):
        merge_example3d()
    lt = read_json("pilot_linear_transient")
    nl = _pilot_rows()
    facts = collect_facts(nl)
    head = ["# Checkpoint 2: cohort, verification, 3-lesion pilot, 3D example", "",
            f"Generated {_dt.date.today().isoformat()} by `python -m dtivl.drivers.report_checkpoint2` from "
            "`results/*.json`, `cohort/cohort.csv` and the parameter files. Every result number is read from those "
            "files; the fixed text holds only the options and recommendations.",
            "Nothing here is validated against patients; this checkpoint covers verification, a pilot and "
            "qualitative comparisons with published observations.", ""]
    text = "\n".join(head + [decisions(facts, nl), sec_cohort(), sec_verification(), sec_transient(lt),
                             sec_nonlinear(nl, facts), sec_example3d(), sec_findings(facts, nl), sec_consistency(nl),
                             sec_diagnostics(), sec_plan(nl)])
    OUT.write_text(text)
    return str(OUT)


if __name__ == "__main__":
    print(build())

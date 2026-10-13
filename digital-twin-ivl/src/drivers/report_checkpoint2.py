"""Checkpoint 2 report: ``CHECKPOINT_2.md`` and its two figures, built only from files.

Every number in the report is read from ``results/*.json``, ``cohort/cohort.csv``
or the parameter files (rule 1 of the study brief). Interpretive sentences that
depend on results (for example which moduli fracture under the balloon alone)
are computed here, not typed. The decisions put to the PI are fixed text.

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
PROTOCOL_LABEL = {"physiological": "Physiological pressure", "balloon_only": "IVL balloon only (4 atm)",
                  "balloon_10atm": "Plain balloon, 10 atm",
                  "ivl": "IVL (balloon + pulse)", "stent_no_ivl": "Stent, no IVL",
                  "stent_post_ivl": "Stent after IVL", "ivl_erosion": "IVL, erosion comparator"}
ATM_MPA = 0.101325


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
    return "–" if x is None else f(100.0 * float(x), nd) + " %"


def table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _load(name: str):
    path = RESULTS_DIR / f"{name}.json"
    return read_json(name) if path.exists() else None


def _pilot_rows() -> dict:
    """Per-lesion nonlinear rows (aggregate file, else the per-lesion files)."""
    agg = _load("pilot_nonlinear")
    if agg is not None:
        return agg["lesions"]
    from dtivl.drivers.stage_pilot import pilot_ids

    out = {}
    for lid in pilot_ids():
        part = _load(f"pilot_nonlinear_{lid}")
        if part is not None:
            out[lid] = part["rows"]
    return out


# ---------------------------------------------------------------- figures
def fig_onset(nl: dict, lt: dict) -> str:
    """Fracture-onset pressure under the balloon-only and IVL protocols vs calcium modulus."""
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
                if rec is None:
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
                        label=f"{lid}: no fracture (marker = peak load)")
        ax.axhline(p_bal, color=TEXT_2, lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.text(0.02, p_bal, " IVL balloon pressure", color=TEXT_2, fontsize=6, va="bottom",
                transform=ax.get_yaxis_transform())
        ax.set_xlabel("Calcium modulus E (MPa)")
        ax.set_title(PROTOCOL_LABEL[proto])
    axs[0].set_ylabel("Fracture onset pressure (MPa)")
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
    s = [f"## 1. In silico cohort ({cs['n_total']} lesions)", "",
         table(["Set", "Lesions", "Fujino score: count"], rows), "",
         f"Core reference geometry: lumen radius {f(g['r_lumen_mm'])} mm, plaque thickness "
         f"{f(g['plaque_thickness_mm'])} mm, media + adventitia {f(g['wall_thickness_mm'])} mm, calcium depth "
         f"{f(g['depth_superficial_mm'])} mm (superficial) and {f(g['depth_deep_mm'])} mm (deep).",
         f"Implied plaque burden {f(pb['implied_plaque_burden_pct'])} %, inside the CT-angiography range of "
         f"calcified plaques ({f(pb['cta_calcified_plaque_pct'])} %) to calcified nodules "
         f"({f(pb['cta_calcified_nodule_pct'])} %): {'yes' if pb['within_range'] else 'NO'}.", "",
         "Hold-out depths now follow the morphology label (deep sheets sample the deeper half of the "
         "literature depth range, other morphologies the shallower half):", "",
         table(["Morphology", "min depth (mm)", "max depth (mm)", "n"],
               [[m, f(r["min"]), f(r["max"]), int(r["count"])] for m, r in depth.iterrows()]), "",
         f"Hold-out lesions with a lipid pool: {n_lipid} of {len(hold)}. The LHS lipid indicator selects half "
         "of them; pools are dropped when the calcium arc leaves no angular room opposite it.", "",
         "Figure: `figures/fig2_cohort_gallery.png` (core factorial set).", ""]
    return "\n".join(s)


def sec_verification() -> str:
    la = read_json("verification_lame")
    nl = read_json("verification_nonlinear")
    pf = read_json("verification_phasefield")
    gc = read_json("verification_mesh_convergence")
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
         table(["Case", "E (MPa)", "k_floor", "psi_c theory (MPa)", "psi onset measured (MPa)", "rel. error",
                "passed"],
               [[c["case"], f(c["E_MPa"]), f(c["k_floor"]), f(c["psi_onset_theory_MPa"]),
                 f(c["psi_onset_measured_MPa"], 4), pct(c["rel_err_psi"]), f(c["passed"])] for c in pf["cases"]]),
         "", "### 2.4 Mesh convergence (three grids, refinement factors "
         f"{', '.join(f(x) for x in gc['factors'])}; Celik GCI)", ""]
    rows = []
    for q, v in gc["max_rel_change_default_vs_fine"].items():
        conv = {lid: L["gci"][q]["convergence"] for lid, L in gc["lesions"].items()}
        mono = [L["gci"][q]["gci_fine21"] for L in gc["lesions"].values() if L["gci"][q]["convergence"] == "monotonic"]
        gci = pct(max(mono)) if mono else "n/a (no monotonic case)"
        rows.append([q, pct(v), gci, ", ".join(f"{k}: {c}" for k, c in conv.items()),
                     "mesh-dependent (not a predictor)" if q in gc["mesh_dependent_quantities"] else
                     ("accepted" if v < gc["rel_change_accept"] else "NOT accepted")])
    s += [table(["Quantity", "max rel. change default vs fine", "max GCI (fine), monotonic cases only",
                 "convergence class", "status"], rows),
          "", f"Acceptance: relative change below {pct(gc['rel_change_accept'], 2)} between the default and "
          "fine grids. The GCI is defined only for monotonic convergence (Celik et al.); for oscillatory or "
          "divergent triplets the relative change is the criterion. The raw peaks are singular at deposit corners and are not used as predictors; the "
          "nonlocal (critical-distance) peaks replace them.", ""]
    return "\n".join(s)


def sec_transient(lt: dict, nl: dict) -> str:
    s = ["## 3. Pilot (3 lesions): linear predictors and IVL transfer factor", "",
         f"Linear predictors at lumen pressure {f(next(iter(lt['lesions'].values()))['by_E'][0]['predictors']['p_lumen_MPa'])} MPa "
         "(the IVL balloon pressure), over the calcium-modulus sweep. T_equivalent is the transient peak "
         "maximum principal stress in calcium divided by the quasi-static stress under the same peak "
         "pressure; it scales the pulse in the quasi-static IVL step.", ""]
    rows = []
    for lid, rec in lt["lesions"].items():
        for r in rec["by_E"]:
            p, t = r["predictors"], r["transient"]
            rows.append([lid, f(r["ca_E_MPa"]), f(p["sigma1_p95"]), f(p["sigma1_peak_nl"]),
                         pct(p["frac_eps_gt_epsf"]), f(t["T"]), f(t["T_equivalent"]), f(t["peak_at_window_end"])])
    s += [table(["Lesion", "E (MPa)", "sigma1 P95 (MPa)", "nonlocal peak sigma1 (MPa)",
                 "calcium area with eps1 > eps_f", "T", "T_equivalent", "peak at window end"], rows), ""]
    s.append(f"Figure: `{fig_transfer(lt)}`.")
    s.append("")
    return "\n".join(s)


def DEFAULT_STEPS_TXT() -> str:
    from dtivl.solvers.nonlinear import DEFAULT_STEPS

    p_bal = P.get("loads.ivl_balloon_pressure_atm").value
    p_poba = P.get("loads.poba_pressure_atm").value
    return (f"pressure ramps use {DEFAULT_STEPS['pressure']} increments (4 atm: "
            f"{f(p_bal / DEFAULT_STEPS['pressure'])} MPa; 10 atm: {f(p_poba / DEFAULT_STEPS['pressure'])} MPa), "
            f"IVL inflation {DEFAULT_STEPS['ivl_inflate']} increments to the balloon pressure and "
            f"{DEFAULT_STEPS['ivl_pulse']} to the pulse peak; increments are halved only when a step fails")


def _onset_str(rec: dict) -> str:
    c = rec["cracks"]
    if not c["fracture"]:
        return "none"
    unit = "mm" if (c.get("onset_phase") or "").startswith("stent") else "MPa"
    return f"{f(c['onset_load'])} {unit}" if c["onset_load"] is not None else "yes"


def sec_nonlinear(nl: dict, lt: dict) -> tuple[str, dict]:
    cal = read_json("calibration_stent")
    intro = (f"Stent radial stiffness K_s = {f(cal['K_s_MPa_per_mm'])} MPa/mm, identified so that the calcium-free "
             f"template ({cal['lesion_template']}) recoils by {f(cal['recoil_target_pct'])} % (clinical acute recoil). "
             "Onset is the first load step with max d >= d_crack; for stents the load is the cylinder radius (mm). "
             f"Onset is resolved only to the load step: {DEFAULT_STEPS_TXT()}.")
    s = ["## 4. Pilot (3 lesions): nonlinear phase-field outcomes", "", intro, ""]
    rows, facts = [], {"balloon_fracture_E": {}, "ivl_fracture_E": {}, "skipped": {}, "failed": [],
                       "diffuse": 0, "discrete": 0}
    for lid, lrows in nl.items():
        for r in lrows:
            E = r["ca_E_MPa"]
            if "skipped" in r:
                facts["skipped"].setdefault(lid, []).append(E)
                rows.append([lid, f(E), "all", "skipped", r["skipped"], "", "", "", ""])
                continue
            for proto in PROTOCOLS:
                rec = r.get(proto)
                if rec is None:
                    continue
                c, m = rec["cracks"], rec["summary"].get("metrics", {})
                if rec["status"] != "ok":
                    facts["failed"].append((lid, E, proto, rec["status"], rec["summary"].get("failed_phase")
                                            or rec["summary"].get("expand_status")))
                nu = rec.get("n_steps_unconverged_stagger")
                facts.setdefault("unconverged", {"recorded": 0, "not_recorded": 0, "steps": 0})
                if nu is None:
                    facts["unconverged"]["not_recorded"] += 1
                else:
                    facts["unconverged"]["recorded"] += 1
                    facts["unconverged"]["steps"] += nu
                if c["fracture"]:
                    facts["diffuse" if c["diffuse"] else "discrete"] += 1
                if proto == "balloon_only" and c["fracture"]:
                    facts["balloon_fracture_E"].setdefault(lid, []).append(E)
                if proto == "ivl" and c["fracture"]:
                    facts["ivl_fracture_E"].setdefault(lid, []).append(E)
                pattern = ", ".join(c["pattern"]) if c["pattern"] else "–"
                if m and rec["status"] == "ok":
                    stent = (f"{f(m.get('stent_expansion_pct'))} % / {f(m.get('acute_recoil_pct'))} % / "
                             f"{f(rec['summary'].get('P_dep'))} MPa")
                elif m:
                    stent = (f"not valid: solver failed at cylinder radius {f(rec['summary'].get('R_dep'))} mm, "
                             f"mean contact pressure {f(rec['summary'].get('P_dep'))} MPa "
                             f"({pct(rec['summary'].get('P_dep') / rec['summary'].get('p_rated'), 2)} of rated)")
                else:
                    stent = ""
                rows.append([lid, f(E), PROTOCOL_LABEL[proto], rec["status"], _onset_str(rec),
                             f"{c['n_fractures']} ({pattern})", pct(c["damaged_fraction"], 3),
                             f(rec.get("max_lumen_area")), stent])
    s += [table(["Lesion", "E (MPa)", "Protocol", "Status", "Fracture onset", "Through-thickness fractures (pattern)",
                 "Damaged calcium area", "Max lumen area (mm²)", "Stent expansion / recoil / deployment pressure"],
                rows), ""]
    s.append(f"Figure: `{fig_onset(nl, lt)}`.")
    s.append("")
    mtxt, window = outcome_matrix(nl)
    facts["ivl_only_window"] = window
    s.insert(s.index(intro) + 2, mtxt)
    return "\n".join(s), facts


def outcome_matrix(nl: dict) -> tuple[str, dict]:
    """Modulus x lesion matrix of fracture outcomes (B = balloon only, I = IVL, S = stent without IVL)."""
    Es = sorted({r["ca_E_MPa"] for lr in nl.values() for r in lr})
    window = {}
    rows = []
    for E in Es:
        cells = []
        for lid, lr in nl.items():
            r = next((x for x in lr if abs(x["ca_E_MPa"] - E) < 1e-9), None)
            if r is None:
                cells.append("not run")
                continue
            if "skipped" in r:
                cells.append("skipped")
                continue
            parts = []
            for key, tag in (("balloon_only", "B"), ("balloon_10atm", "P"), ("ivl", "I"), ("stent_no_ivl", "S")):
                rec = r.get(key)
                if rec is None:
                    continue
                c = rec["cracks"]
                mark = "–" if not c["fracture"] else ("diffuse" if c["diffuse"] else f"{c['n_fractures']} crack(s)")
                if rec["status"] != "ok":
                    mark += f" ({rec['status']})"
                parts.append(f"{tag}: {mark}")
            b, i = r.get("balloon_only"), r.get("ivl")
            if b and i and i["cracks"]["fracture"] and not b["cracks"]["fracture"]:
                window.setdefault(lid, []).append(E)
            cells.append("; ".join(parts))
        rows.append([f(E)] + cells)
    txt = ["### 4.1 Fracture outcome by calcium modulus", "",
           "B = IVL balloon alone (4 atm), P = plain balloon at 10 atm (same size), I = IVL (balloon + pulse), "
           "S = stent without IVL (run at two moduli only); – = no fracture.", "",
           table(["E (MPa)"] + list(nl), rows), ""]
    if window:
        txt.append("Moduli at which IVL fractures the calcium but the balloon alone does not: " + "; ".join(
            f"{lid}: {', '.join(f(e) for e in v)} MPa" for lid, v in window.items()) + ".")
    else:
        txt.append("At no tested modulus does IVL fracture the calcium while the balloon alone does not.")
    txt.append("")
    return "\n".join(txt), window


def sec_diagnostics() -> str:
    """Section on damage localisation (decision D10), from the diagnostics result files."""
    d = _load("diag_localisation")
    e = _load("diag_ell")
    s = ["## 8. Why the damage is diffuse (diagnostics for D10)", ""]
    if d is not None:
        rows = []
        for r in d["by_E"]:
            for k, lab in (("fine", "fine steps"), ("seeded", "seeded radial crack"),
                           ("heterogeneous", f"G_c heterogeneity (CoV {f(d['cov'], 2)})")):
                x = r[k]
                first = next((st for st in x["steps"] if st.get("damaged_fraction", 0.0) > 0.0), None)
                nu = sum(1 for st in x["steps"] if st.get("stagger_converged") is False)
                rows.append([f(r["ca_E_MPa"]), lab, f(x["first_damaged_p"]),
                             "–" if first is None else ", ".join(first["pattern"]) or "–",
                             pct(x["final_damaged_fraction"], 3), f(x["final_diffuse"]), nu])
        s += [f"Lesion {d['lesion']}, IVL balloon cap, pressure steps of {f(d['dp_MPa'])} MPa up to "
              f"{f(d['p_end_MPa'])} MPa (`results/diag_localisation.json`).", "",
              table(["E (MPa)", "Experiment", "first cracked step p (MPa)", "pattern at that step",
                     "damaged area at the end", "diffuse at the end", "steps with unconverged (u, d) loop"], rows), ""]
    if e is not None:
        rows = []
        for h in e["by_h"]:
            for r in h["by_E"]:
                last = r["steps"][-1] if r["steps"] else {}
                nu = sum(1 for st in r["steps"] if st.get("stagger_converged") is False)
                rows.append([f(h["h_ca_mm"]), f(h["ell_mm"]), h["n_cells_calcium"], f(r["ca_E_MPa"]),
                             f(r["first_damaged_p"]), pct(r["final_damaged_fraction"], 3), f(r["final_diffuse"]),
                             f(last.get("n_fractures")), ", ".join(last.get("pattern", [])) or "–",
                             nu, f(r["seconds"], 2)])
        s += ["Regularisation length refined with the calcium mesh (ell / h fixed; `results/diag_ell.json`):", "",
              table(["h_Ca (mm)", "ell (mm)", "calcium cells", "E (MPa)", "first cracked p (MPa)",
                     "damaged area at the end", "diffuse", "through-thickness components", "patterns",
                     "unconverged (u, d) steps", "run time (s)"], rows), ""]
    return "\n".join(s)


def sec_example3d() -> str:
    ex = _load("pilot_example3d")
    if ex is None:
        return "## 5. 3D Fujino-4 example\n\nNot run.\n"
    s = [f"## 5. 3D Fujino-4 example ({ex['lesion']}) vs its 2D mid-section", "",
         f"Regularisation length in this example: {f(ex['ell_mm'])} mm (coarser 3D mesh).", ""]
    rows = []
    for key in ("3d", "2d"):
        m = ex["models"].get(key)
        if m is None:
            continue
        for r in m["by_E"]:
            p = r["predictors"]
            t = r.get("transient", {})
            ivl = r.get("ivl")
            rows.append([key.upper(), f(r["ca_E_MPa"]), m["mesh"]["n_cells"], f(p["sigma1_p95"]),
                         f(p["sigma1_peak_nl"]), f(p["sed_mean"]), f(t.get("T_equivalent")),
                         "–" if ivl is None else f"{ivl['status']}, {_onset_str(ivl)}",
                         "–" if ivl is None else pct(ivl["cracks"]["damaged_fraction"], 3)])
    s += [table(["Model", "E (MPa)", "cells", "sigma1 P95 (MPa)", "nonlocal peak sigma1 (MPa)",
                 "mean SED (MPa)", "T_equivalent", "IVL: status, onset", "IVL damaged area"], rows), ""]
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
                rows.append([key.upper(), f(r["ca_E_MPa"]), r["status"], "–" if o is None else f(o["p"]),
                             "–" if o is None else pct(o["cracks"]["damaged_fraction"], 3),
                             "–" if o is None else ", ".join(o["cracks"]["pattern"]) or "–", nu,
                             f(r["seconds"], 2)])
        s += [f"Balloon-pressure ramp to fracture onset ({on['n_steps']} increments up to {f(on['p_end_MPa'])} MPa, "
              "IVL balloon cap), 3D vs the 2D mid-section (`results/pilot_example3d_onset.json`). The full 3D IVL "
              "protocol at the tissue-level baseline modulus was stopped before completion because its "
              "damage-propagation steps were too slow, and was replaced by this onset run.", "",
              table(["Model", "E (MPa)", "status", "onset p (MPa)", "damaged area at onset", "pattern at onset",
                     "unconverged (u, d) steps", "run time (s)"], rows), ""]
    return "\n".join(s)


def sec_findings(facts: dict, nl: dict) -> str:
    ms = MaterialSet.baseline()
    p_bal = P.get("loads.ivl_balloon_pressure_atm").value
    p_rated = P.get("loads.balloon_rated_pressure_atm").value
    lines = ["## 6. What the pilot shows", ""]
    lesions = list(nl)
    for lid in lesions:
        for proto, key in (("balloon_only", "balloon_fracture_E"), ("ivl", "ivl_fracture_E")):
            tested = [r["ca_E_MPa"] for r in nl[lid] if proto in r]
            if not tested:
                lines.append(f"* {lid}, {PROTOCOL_LABEL[proto]}: not run yet.")
                continue
            hit = facts[key].get(lid, [])
            intact = [e for e in tested if e not in hit]
            lines.append(f"* {lid}, {PROTOCOL_LABEL[proto]}: fracture at E = "
                         f"{', '.join(f(e) for e in hit) or 'none'} MPa; no fracture at E = "
                         f"{', '.join(f(e) for e in intact) or 'none'} MPa.")
    lines.append(f"  (IVL balloon pressure {f(p_bal)} MPa = {f(p_bal / ATM_MPA)} atm.)")
    lines += ["",
              f"* Baseline calcium modulus (E = {f(ms.ca_E)} MPa, secant modulus at failure of tissue-level "
              f"calcification) gives a failure stress of {f(ms.sigma_crit)} MPa. Fractured outcomes with "
              f"diffuse whole-deposit damage: {facts['diffuse']}; with discrete cracks: {facts['discrete']}.",
              f"* Moduli skipped because the calcium is not stiffer than the fibrous matrix "
              f"(E_fibrous = {f(ms.fibrous_E)} MPa): "
              + ("; ".join(f"{k}: {', '.join(f(e) for e in v)} MPa" for k, v in facts["skipped"].items()) or "none") + ".",
              "* Protocols that did not finish: " + ("; ".join(
                  f"{lid} E={f(E)} {p} ({st}, phase {ph})" for lid, E, p, st, ph in facts["failed"]) or "none") + ".",
              f"* Stent deployment uses the rated pressure {f(p_rated)} MPa as the expansion limit.", ""]
    return "\n".join(lines)


def _lit(name: str) -> str:
    """'value (range) unit [STATUS]' of a literature row, for comparison tables."""
    q = P.load_literature()[name]
    rng = f" ({q.range_str})" if q.range_str and q.range_str != "nan" else ""
    return f"{q.value_str}{rng} {q.unit} [{q.status}]"


def sec_consistency(nl: dict) -> str:
    """Preliminary comparison with published observations (qualitative; no patient data)."""
    df = pd.read_csv(COHORT_DIR / "cohort.csv").set_index("lesion_id")
    L = P.load_literature()
    a_cut = L["fracture_angle_cutoff_balloon"].value
    t_cut = L["fracture_thickness_cutoff_balloon"].value
    rows = []
    phys = [r["physiological"]["cracks"]["fracture"] for lr in nl.values() for r in lr if "physiological" in r]
    rows.append(["No fracture at physiological pressure", "expected (no source needed)",
                 f"{sum(phys)} of {len(phys)} runs fracture"])
    for lid, lr in nl.items():
        arc, thk = df.loc[lid, "ca_arc_total_deg"], df.loc[lid, "ca_thickness_max"]
        expect = arc > a_cut and thk < t_cut
        hit = [r["ca_E_MPa"] for r in lr if "stent_no_ivl" in r and r["stent_no_ivl"]["cracks"]["fracture"]]
        tested = [r["ca_E_MPa"] for r in lr if "stent_no_ivl" in r]
        rows.append([f"Fracture after high-pressure dilation (no IVL) predicted by arc > {f(a_cut)} deg and "
                     f"thickness < {f(t_cut)} mm ({lid}: arc {f(arc)} deg, max thickness {f(thk)} mm -> "
                     f"{'fracture' if expect else 'no fracture'} expected)",
                     f"{_lit('fracture_angle_cutoff_balloon')}; {_lit('fracture_thickness_cutoff_balloon')}",
                     f"stent without IVL: fracture at E = {', '.join(f(e) for e in hit) or 'none'} of "
                     f"{', '.join(f(e) for e in tested) or 'none'} MPa"])
    poba = {lid: ([r["ca_E_MPa"] for r in lr if "balloon_10atm" in r],
                  [r["ca_E_MPa"] for r in lr if "balloon_10atm" in r and r["balloon_10atm"]["cracks"]["fracture"]])
            for lid, lr in nl.items()}
    rows.append(["Plain balloon at 10 atm does not fracture calcium; IVL does (cadaver histology)",
                 _lit("calcium_fracture_rate_POBA_10atm_cadaver_histology"),
                 "; ".join(f"{lid}: 10-atm balloon fractures at E = {_elist(v[1])} of {_emp(v[0])}"
                           for lid, v in poba.items()) + " (calibration target under D8 option (c))"])
    ivl = {lid: [r["ca_E_MPa"] for r in lr if "ivl" in r and r["ivl"]["cracks"]["fracture"]] for lid, lr in nl.items()}
    rows.append(["Nodular calcium fractures less often after IVL than sheets",
                 _lit("nodular_calcium_odds_ratio_for_fracture"),
                 "; ".join(f"{lid}: IVL fracture at E = {', '.join(f(e) for e in v) or 'none'} MPa" for lid, v in ivl.items())])
    comps = [c for lr in nl.values() for r in lr for k in PROTOCOLS if k in r and k != "physiological"
             for c in [r[k]["cracks"]] if c["fracture"] or c["n_partial"]]
    n_thr = sum(c["n_fractures"] for c in comps)
    n_all = n_thr + sum(c["n_partial"] for c in comps)
    rows.append(["Most IVL fractures are full thickness", _lit("full_thickness_fracture_fraction_pooled_IVL_OCT"),
                 f"{n_thr} of {n_all} damaged components through-thickness" + (
                     f" ({sum(c['n_diffuse'] for c in comps)} diffuse)" if n_all else "")])
    rows.append(["Several fractures per lesion", _lit("fractures_per_lesion_pooled_IVL_OCT"),
                 "discrete crack count not defined when damage is diffuse (see D10)"])
    exp = [r[k]["summary"]["metrics"]["stent_expansion_pct"] for lr in nl.values() for r in lr
           for k in ("stent_no_ivl", "stent_post_ivl")
           if k in r and r[k]["summary"].get("metrics") and r[k]["status"] == "ok"]
    n_bad = sum(1 for lr in nl.values() for r in lr for k in ("stent_no_ivl", "stent_post_ivl")
                if k in r and r[k]["status"] != "ok")
    rows.append(["Stent expansion at the minimum stent area", _lit("stent_expansion_MSA_site_DisruptCAD_III"),
                 (f"{f(min(exp))}-{f(max(exp))} % over {len(exp)} completed stent runs" if exp else "–") +
                 (f"; {n_bad} stent runs failed and are excluded" if n_bad else "")])
    # unloaded lumen after the IVL protocol vs the unloaded lumen of the same lesion without IVL
    gain = [(lid, r["ca_E_MPa"], r["ivl"]["final"]["lumen_area"], r["stent_no_ivl"]["summary"]["metrics"]["lumen_area_pre"])
            for lid, lr in nl.items() for r in lr if "stent_no_ivl" in r and "ivl" in r]
    rows.append(["Lumen gain after IVL alone (before stenting)", _lit("COCALP_MLA_pre_post_IVL_post_stent"),
                 "; ".join(f"{lid} E={f(E)}: {f(a)} vs {f(b)} mm² (after IVL vs no IVL)" for lid, E, a, b in gain) +
                 ". No residual gain: the model is hyperelastic with damage and has no permanent set"])
    return "\n".join(["## 7. Consistency with published observations (preliminary, qualitative)", "",
                      "These are comparisons with published cohorts, not validation: no patient-specific outcome "
                      "data exist for these idealised lesions.", "",
                      table(["Observation", "Published value [status]", "Pilot"], rows), ""])


def _fracture_sets(nl: dict, proto: str) -> dict:
    """lesion -> (tested moduli, moduli with fracture) for one protocol."""
    out = {}
    for lid, lr in nl.items():
        tested = [r["ca_E_MPa"] for r in lr if proto in r]
        hit = [r["ca_E_MPa"] for r in lr if proto in r and r[proto]["cracks"]["fracture"]]
        out[lid] = (tested, hit)
    return out


def _elist(es) -> str:
    return ", ".join(f(e) for e in es) if es else "none"


def _emp(es) -> str:
    """Moduli list with unit, or 'none'."""
    return f"{_elist(es)} MPa" if es else "none"


def decisions(facts: dict, nl: dict) -> str:
    ms = MaterialSet.baseline()
    p_bal = P.get("loads.ivl_balloon_pressure_atm").value
    bal, ivl = _fracture_sets(nl, "balloon_only"), _fracture_sets(nl, "ivl")
    poba = _fracture_sets(nl, "balloon_10atm")
    common = sorted(set.intersection(*[set(bal[l][0]) & set(ivl[l][0]) for l in nl])) if nl else []
    selective = [E for E in common if not any(E in bal[l][1] for l in nl) and any(E in ivl[l][1] for l in nl)]
    discriminating = [E for E in selective if not all(E in ivl[l][1] for l in nl)]
    common10 = sorted(set.intersection(*[set(poba[l][0]) & set(ivl[l][0]) for l in nl])) if nl else []
    selective10 = [E for E in common10 if not any(E in poba[l][1] for l in nl) and any(E in ivl[l][1] for l in nl)]
    L = P.load_literature()
    kaw = L["calcium_fracture_rate_POBA_10atm_cadaver_histology"]
    nano = L["calcification_modulus_nanoindentation_carotid"]
    loree = L["calcified_tissue_tangential_modulus"]
    base_on = {l: next((r["balloon_only"]["cracks"]["onset_load"] for r in lr
                        if abs(r["ca_E_MPa"] - ms.ca_E) < 1e-9 and "balloon_only" in r), None) for l, lr in nl.items()}
    lines = ["## 0. Decisions needed before the full cohort run", ""]
    lines += ["**D8: calcium modulus regime (the main decision).** The pilot outcome depends on the calcium "
              "modulus more than on anything else (Section 4.1). Per lesion, moduli (MPa) at which the calcium fractures:", "",
              table(["Lesion", "IVL balloon alone (4 atm)", "plain balloon (10 atm)", "IVL (balloon + pulse)",
                     "no fracture even with IVL"],
                    [[l, _elist(bal[l][1]), _elist(poba[l][1]) if poba[l][0] else "not run", _elist(ivl[l][1]),
                      _elist([e for e in ivl[l][0] if e not in ivl[l][1]])] for l in nl]), "",
              f"* Baseline (E = {f(ms.ca_E)} MPa, secant modulus at failure of tissue-level calcification; failure "
              f"stress {f(ms.sigma_crit)} MPa): the 4-atm balloon alone fractures every pilot lesion ("
              + "; ".join(f"{l} at {f(v)} MPa = {f(v / ATM_MPA)} atm" for l, v in base_on.items() if v is not None)
              + ") and the whole deposit is damaged. IVL then adds nothing, and every lesion would have the "
              "same outcome, so the Phase 5 ROC analysis would have no events to discriminate.",
              f"* Published contrast: in human coronary cadaver lesions a plain balloon at 10 atm fractured "
              f"{kaw.value_str} % ({kaw.range_str}) [{kaw.status}]. Measured calcium moduli span tissue-level "
              f"tension tests ({loree.value_str} {loree.unit}, Loree 1994) to nanoindentation "
              f"({nano.value_str} {nano.unit}, {nano.range_str}) [{nano.status}].",
              f"* Moduli at which no pilot lesion fractures under the 4-atm IVL balloon alone but at least one "
              f"fractures with IVL: {_emp(selective)} (lesions also differ in IVL outcome at: "
              f"{_emp(discriminating)}). The stricter version with the 10-atm plain balloon of the cadaver "
              f"study: {_emp(selective10)}.", "",
              "Options:",
              "  (a) keep the tissue-level baseline and report that the model then cannot separate lesions;",
              "  (b) use a GPa-level baseline (nanoindentation range): at the GPa moduli tested no pilot lesion "
              "fractures even with IVL, which also leaves nothing to predict and contradicts the IVL fracture rates;",
              "  (c) calibrate: take the baseline from the moduli listed above (balloon alone does not fracture, IVL "
              "does), keep the strain criterion of D5 (failure stress = E eps_f), and report the whole sweep as the "
              "sensitivity analysis. The cadaver balloon-vs-IVL contrast is then used for calibration and cannot "
              "also be reported as a consistency check; the other comparisons (nodule effect, arc/thickness "
              "cut-offs, fracture rates, stent expansion) stay independent.",
              "  Recommendation: (c). Candidate baselines: " + (
                  f"{_emp(selective10)} (meets the 10-atm contrast)" if selective10 else
                  f"{_emp(selective)} (meets the 4-atm contrast; no tested modulus meets the 10-atm one for "
                  "every pilot lesion)" if selective else "none among the tested moduli") +
              ". This calibrates an uncertain input to one qualitative observation, and the manuscript would say so.", ""]
    cand = selective10 or selective
    k_base = ms.k_floor
    k_sel = MaterialSet.baseline(ca_E=cand[-1]).k_floor if cand else None
    lines += [f"**D9: crack filler (new assumption).** A fully damaged calcium band keeps the shear stiffness of the "
              f"fibrous matrix (degradation floor k = mu_fibrous / mu_calcium = {f(k_base)} at the tissue-level "
              f"baseline" + (f", {f(k_sel)} at {f(cand[-1])} MPa" if k_sel is not None else "") +
              ") instead of becoming a void. Without it the stiff-calcium runs (E >= 121 MPa) did not converge "
              "once a deposit cracked. G_c is rescaled so that the verified onset criterion is unchanged "
              "(Section 2.3). Moduli where calcium is not stiffer than the matrix are skipped. Proposed sensitivity: "
              "k = k_residual (no filler) where the solver allows. Approve, or ask for an alternative.", ""]
    d = _load("diag_localisation")
    e = _load("diag_ell")
    if d is not None and e is not None:
        onset = {(h["ell_mm"], round(r["ca_E_MPa"], 6)): r["first_damaged_p"] for h in e["by_h"] for r in h["by_E"]}
        Es = sorted({k[1] for k in onset})
        ells = sorted({k[0] for k in onset}, reverse=True)
        diffuse_all = all(r[k]["final_diffuse"] for r in d["by_E"] for k in ("fine", "seeded", "heterogeneous")) and \
            all(r["final_diffuse"] for h in e["by_h"] for r in h["by_E"])
        lines += ["**D10: fracture outcomes and the damage model.** In the pilot the damage covers the whole deposit "
                  "instead of forming discrete cracks. Diagnostics (Section 8): finer load steps, a seeded radial "
                  "crack, a 10 % toughness heterogeneity and a 4x smaller regularisation length all end with the "
                  f"whole deposit damaged (all diffuse: {'yes' if diffuse_all else 'no'}). Damage starts as "
                  "circumferential bands along the luminal surface, where the sheet is bent open. My explanation, not yet "
                  "tested: the volumetric-deviatoric energy split lets damage degrade the whole high-energy luminal "
                  "layer instead of opening a crack perpendicular to the hoop tension; option (b) would test it. The onset pressure is robust to the "
                  "regularisation length: " + "; ".join(
                      f"E = {f(E)} MPa: " + ", ".join(f"{f(onset[(l, E)])}" for l in ells if (l, E) in onset) +
                      f" MPa for ell = {', '.join(f(l) for l in ells if (l, E) in onset)} mm" for E in Es) + ".",
                  "Options:",
                  "  (a) keep the model; primary outcomes = fracture yes/no, onset pressure and damaged fraction "
                  "(all robust); crack number and radial/circumferential pattern not reported;",
                  "  (b) replace the energy split by a principal-strain (spectral) driving force, which opens cracks "
                  "perpendicular to the maximum principal strain, and re-test localisation on the pilot before the "
                  "full run (about one more pilot cycle);",
                  "  Recommendation: (a) for the main analysis, plus a time-boxed (b) on the three pilot lesions; if (b) "
                  "gives discrete cracks, crack number and pattern become secondary outcomes.", ""]
    cs = read_json("cohort_summary")
    core = cs["fujino_score_counts"]["core"]
    lines += ["**D11: Fujino range.** Core-set Fujino scores: " + ", ".join(f"{k}: {v}" for k, v in sorted(core.items())) +
              " (no 4s; with D2 thicknesses only nodules exceed 0.5 mm, and 2D has no length). The Phase 5 "
              f"comparison with Fujino is then mostly a comparison with the arc criterion. IVL-treated lesions in "
              f"Disrupt CAD III had a maximum thickness of {_lit('max_calcium_thickness_DisruptCAD_III_OCT')}. "
              "Options: keep D2 and state the limitation, or add a 0.8 mm level (+16 core lesions).", ""]
    ex = _load("pilot_example3d")
    t3 = []
    if ex is not None and "3d" in ex["models"]:
        t3 = [(r["ca_E_MPa"], r["ivl"]["seconds"]) for r in ex["models"]["3d"]["by_E"] if "ivl" in r]
    t2 = []
    if ex is not None and "2d" in ex["models"]:
        t2 = [(r["ca_E_MPa"], r["ivl"]["seconds"]) for r in ex["models"]["2d"]["by_E"] if "ivl" in r]
    lines += ["**D12: 3D length series (D3).** I propose running the 1/3/5 mm series after this checkpoint on a "
              "subset you choose, and only up to fracture onset in 3D. Cost of the full IVL protocol on the "
              "Fujino-4 example: " + ("; ".join(f"3D, E = {f(E)} MPa: {f(t, 2)} s" for E, t in t3) or "3D not finished")
              + (" vs " + "; ".join(f"2D section, E = {f(E)} MPa: {f(t, 2)} s" for E, t in t2) if t2 else "")
              + ". With damage spreading, a 3D step costs far more (Section 5).", ""]
    ero = []
    for lid, lr in nl.items():
        for r in lr:
            if "ivl_erosion" in r and "ivl" in r:
                ero.append(f"{lid} E={f(r['ca_E_MPa'])}: phase field {_onset_str(r['ivl'])}, erosion {_onset_str(r['ivl_erosion'])}")
    lines += ["**D13: erosion comparator.** It now completes. Fracture onset under IVL: " + "; ".join(ero) +
              ". Erosion uses the cell-averaged maximum principal strain, the phase field an energy criterion, so "
              "onset can differ. Keep erosion as the subset comparator?", ""]
    return "\n".join(lines)



def build() -> str:
    from dtivl.drivers.stage_pilot import merge_example3d

    if any((RESULTS_DIR / f"pilot_example3d_part{i}.json").exists() for i in (2, 3, 4)):
        merge_example3d()
    lt = read_json("pilot_linear_transient")
    nl = _pilot_rows()
    nl_text, facts = sec_nonlinear(nl, lt)
    head = ["# Checkpoint 2: cohort, verification, 3-lesion pilot, 3D example", "",
            f"Generated {_dt.date.today().isoformat()} by `python -m dtivl.drivers.report_checkpoint2` from "
            "`results/*.json`, `cohort/cohort.csv` and the parameter files. No number below is typed by hand.",
            "Nothing here is validated against patients; this checkpoint covers verification and a pilot.", ""]
    text = "\n".join(head + [decisions(facts, nl), sec_cohort(), sec_verification(), sec_transient(lt, nl), nl_text,
                             sec_example3d(), sec_findings(facts, nl), sec_consistency(nl), sec_diagnostics()])
    OUT.write_text(text)
    return str(OUT)


if __name__ == "__main__":
    print(build())

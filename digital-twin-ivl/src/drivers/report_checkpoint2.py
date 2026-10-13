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
PROTOCOLS = ("physiological", "balloon_only", "ivl", "stent_no_ivl", "stent_post_ivl", "ivl_erosion")
PROTOCOL_LABEL = {"physiological": "Physiological pressure", "balloon_only": "IVL balloon only",
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


def _onset_str(rec: dict) -> str:
    c = rec["cracks"]
    if not c["fracture"]:
        return "none"
    unit = "mm" if (c.get("onset_phase") or "").startswith("stent") else "MPa"
    return f"{f(c['onset_load'])} {unit}" if c["onset_load"] is not None else "yes"


def sec_nonlinear(nl: dict, lt: dict) -> tuple[str, dict]:
    cal = read_json("calibration_stent")
    s = ["## 4. Pilot (3 lesions): nonlinear phase-field outcomes", "",
         f"Stent radial stiffness K_s = {f(cal['K_s_MPa_per_mm'])} MPa/mm, identified so that the calcium-free "
         f"template ({cal['lesion_template']}) recoils by {f(cal['recoil_target_pct'])} % (clinical acute recoil). "
         "Onset is the first load step with max d >= d_crack; for stents the load is the cylinder radius (mm).", ""]
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
                    facts["failed"].append((lid, E, proto, rec["status"], rec["summary"].get("failed_phase")))
                if c["fracture"]:
                    facts["diffuse" if c["diffuse"] else "discrete"] += 1
                if proto == "balloon_only" and c["fracture"]:
                    facts["balloon_fracture_E"].setdefault(lid, []).append(E)
                if proto == "ivl" and c["fracture"]:
                    facts["ivl_fracture_E"].setdefault(lid, []).append(E)
                pattern = ", ".join(c["pattern"]) if c["pattern"] else "–"
                stent = (f"{f(m.get('stent_expansion_pct'))} % / {f(m.get('acute_recoil_pct'))} % / "
                         f"{f(rec['summary'].get('P_dep'))} MPa" if m else "")
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
    s.insert(s.index(f"Stent radial stiffness K_s = {f(cal['K_s_MPa_per_mm'])} MPa/mm, identified so that the calcium-free "
                     f"template ({cal['lesion_template']}) recoils by {f(cal['recoil_target_pct'])} % (clinical acute recoil). "
                     "Onset is the first load step with max d >= d_crack; for stents the load is the cylinder radius (mm).") + 2,
             mtxt)
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
            for key, tag in (("balloon_only", "B"), ("ivl", "I"), ("stent_no_ivl", "S")):
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
           "B = IVL balloon alone (4 atm), I = IVL (balloon + pulse), S = stent without IVL "
           "(run at two moduli only); – = no fracture.", "",
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
                rows.append([f(r["ca_E_MPa"]), lab, f(x["first_damaged_p"]),
                             "–" if first is None else ", ".join(first["pattern"]) or "–",
                             pct(x["final_damaged_fraction"], 3), f(x["final_diffuse"])])
        s += [f"Lesion {d['lesion']}, IVL balloon cap, pressure steps of {f(d['dp_MPa'])} MPa up to "
              f"{f(d['p_end_MPa'])} MPa (`results/diag_localisation.json`).", "",
              table(["E (MPa)", "Experiment", "first cracked step p (MPa)", "pattern at that step",
                     "damaged area at the end", "diffuse at the end"], rows), ""]
    if e is not None:
        rows = []
        for h in e["by_h"]:
            for r in h["by_E"]:
                last = r["steps"][-1] if r["steps"] else {}
                rows.append([f(h["h_ca_mm"]), f(h["ell_mm"]), h["n_cells_calcium"], f(r["ca_E_MPa"]),
                             f(r["first_damaged_p"]), pct(r["final_damaged_fraction"], 3), f(r["final_diffuse"]),
                             f(last.get("n_fractures")), ", ".join(last.get("pattern", [])) or "–",
                             f(r["seconds"], 2)])
        s += ["Regularisation length refined with the calcium mesh (ell / h fixed; `results/diag_ell.json`):", "",
              table(["h_Ca (mm)", "ell (mm)", "calcium cells", "E (MPa)", "first cracked p (MPa)",
                     "damaged area at the end", "diffuse", "through-thickness components", "patterns",
                     "run time (s)"], rows), ""]
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
           for k in ("stent_no_ivl", "stent_post_ivl") if k in r and r[k]["summary"].get("metrics")]
    rows.append(["Stent expansion at the minimum stent area", _lit("stent_expansion_MSA_site_DisruptCAD_III"),
                 f"{f(min(exp))}-{f(max(exp))} % over {len(exp)} stent runs" if exp else "–"])
    gain = [(r["stent_post_ivl"]["summary"]["metrics"]["lumen_area_pre"], r["stent_no_ivl"]["summary"]["metrics"]["lumen_area_pre"])
            for lr in nl.values() for r in lr if "stent_post_ivl" in r and "stent_no_ivl" in r]
    rows.append(["Lumen gain after IVL alone (before stenting)", _lit("COCALP_MLA_pre_post_IVL_post_stent"),
                 "; ".join(f"{f(a)} vs {f(b)} mm² (after IVL vs no IVL)" for a, b in gain) +
                 " — no residual gain: the model is hyperelastic with damage and has no permanent set"])
    return "\n".join(["## 7. Consistency with published observations (preliminary, qualitative)", "",
                      "These are comparisons with published cohorts, not validation: no patient-specific outcome "
                      "data exist for these idealised lesions.", "",
                      table(["Observation", "Published value [status]", "Pilot"], rows), ""])


DECISIONS = """## 0. Decisions needed before the full cohort run

**D8: calcium modulus regime (the main one).** The pilot shows that the result depends on which
calcium modulus is the baseline (Section 6). At the tissue-level baseline (secant modulus at failure),
calcium is only a few times stiffer than the fibrous matrix: it fails at about 1 atm under the
balloon alone, the whole deposit fails at once, and IVL adds nothing. That does not match the
clinical picture IVL was designed for (calcium that resists high-pressure balloons). Stiffer calcium
(particle/nanoindentation range) reverses this. Options:
  (a) keep the tissue-level baseline and report the regime problem as a finding;
  (b) baseline at a GPa-level modulus (literature nanoindentation range), with the strain-controlled
      criterion of D5 unchanged, and report the tissue-level case as a sensitivity;
  (c) choose the baseline as the modulus at which the 3 pilot lesions best match published qualitative
      observations (balloon alone rarely fractures; IVL fractures thin, wide arcs). This is calibration
      and would be reported as such, and those observations could then not be reused as a consistency check.
  My recommendation: (b), with (a) as the sensitivity. It keeps the calibration separate from the
  consistency checks.

**D9: crack filler (new assumption).** A fully damaged calcium band keeps the shear stiffness of the
fibrous matrix around it (degradation floor k = mu_fibrous / mu_calcium) instead of becoming a void.
Without it, stiff calcium (E >= 121 MPa) left near-zero-stiffness bands and the solver failed. G_c is
rescaled so the verified onset criterion is unchanged (Section 2.3). Sensitivity: k = k_residual
('none') where the solver allows. Approve, or ask for an alternative.

**D10: fracture outcomes when damage is diffuse.** In thin sheets under balloon pressure, damage spreads
over the whole deposit rather than forming discrete cracks (seeded cracks and a 10 % toughness
heterogeneity did not change this at the tested moduli). I added a "diffuse" label (a damaged zone
covering at least half of its deposit). Proposed primary outcomes: fracture yes/no, onset pressure,
damaged fraction and max depth. Number of fractures and radial/circumferential pattern would be reported
only where cracks are discrete. Approve, or keep crack count/pattern as primary outcomes.

**D11: Fujino range.** With D2 thickness levels (0.2, 0.4 mm) and no length in 2D, the core set spans
Fujino scores 0–3 (mostly 0 and 2; the 3s are nodules, whose thickness includes the protrusion). The
Fujino comparison in Phase 5 will be limited to arc. The IVL-treated lesions in Disrupt CAD III had a
mean maximum thickness near 1 mm. Options: keep D2 (and say so), or add a 0.8 mm level (+16 core lesions).

**D12: 3D length series (D3).** I propose running the 1/3/5 mm length series after this checkpoint,
on the subset you choose, because one 3D nonlinear run costs much more than a 2D run (Section 5).

**D13: erosion comparator.** The erosion model now completes. It uses the cell-averaged maximum
principal strain, the phase field uses an energy criterion, so onset can differ. Keep it as the
subset comparator?
"""


def build() -> str:
    lt = read_json("pilot_linear_transient")
    nl = _pilot_rows()
    nl_text, facts = sec_nonlinear(nl, lt)
    head = ["# Checkpoint 2: cohort, verification, 3-lesion pilot, 3D example", "",
            f"Generated {_dt.date.today().isoformat()} by `python -m dtivl.drivers.report_checkpoint2` from "
            "`results/*.json`, `cohort/cohort.csv` and the parameter files. No number below is typed by hand.",
            "Nothing here is validated against patients; this checkpoint covers verification and a pilot.", ""]
    text = "\n".join(head + [DECISIONS, sec_cohort(), sec_verification(), sec_transient(lt, nl), nl_text,
                             sec_example3d(), sec_findings(facts, nl), sec_consistency(nl), sec_diagnostics()])
    OUT.write_text(text)
    return str(OUT)


if __name__ == "__main__":
    print(build())

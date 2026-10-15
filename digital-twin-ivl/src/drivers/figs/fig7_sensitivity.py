"""Figure 7: sensitivity and uncertainty analyses, read from the sensitivity and 3D results files.

(a) One-at-a-time scenarios (``results/sensitivity_oat.json``): change of the IVL damage-onset load
    relative to each pilot lesion's baseline run, per scenario (open marker at zero: onset censored
    or after the peak in either run, so no difference is defined); a black ring marks a scenario
    that changed the IVL through-thickness damage outcome.
(b) Morris screening (``results/sensitivity_morris.json``, pooled over the screening lesions):
    mu* of the IVL damage-onset load with its bootstrap interval, per input, over the stated ranges.
(c) Same for IVL through-thickness damage (0/1 outcome; mu* = mean absolute change per
    normalised input range).
(d) Sobol indices (``results/sensitivity_sobol.json``) of the best single mechanics predictor of
    ``results/metrics_summary.json``: total-order (filled) and first-order (open) index per input
    and lesion.
(e) Three-dimensional length series (``results/length3d.json``): damage-onset load of each 3D
    model relative to the 2D model of the same cross-section, against the model length.

A panel whose input file does not exist yet states so; the figure raises FileNotFoundError only
when none of the inputs exists. Labels of scenarios are built from the scenario names stored in the
results file; every number printed on the figure is read from a results file.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from dtivl.drivers.common import RESULTS_DIR
from dtivl.drivers.figstyle import GRID, SERIES, TEXT, TEXT_2, apply_style, save

OAT_JSON = "sensitivity_oat.json"
MORRIS_JSON = "sensitivity_morris.json"
SOBOL_JSON = "sensitivity_sobol.json"
L3D_JSON = "length3d.json"
METRICS_JSON = "metrics_summary.json"

INPUT_LABELS = {
    "log10_ca_E": "calcium modulus (log)", "eps_f": "calcium failure strain", "ivl_pulse_scale": "IVL pulse scale",
    "ell_over_h": "phase-field length", "log10_k_perivascular": "perivascular support (log)",
    "rho_calcium": "calcium density", "k_over_mu_soft": "soft-tissue K/μ", "fibrous_E": "fibrous modulus",
}
SCENARIO_LABELS = {
    "fracture.crack_filler": "crack filler", "fracture.ell_over_h": "phase-field length ℓ/h",
    "loads.ivl_pulse_scale": "IVL pulse scale", "loads.perivascular_k_MPa_per_mm": "perivascular k (MPa/mm)",
    "materials.bulk_to_shear_soft": "soft-tissue K/μ", "fracture.driving_force": "damage driving force",
    "materials.density.calcium": "calcium density (kg/m³)", "failure_mode": "failure criterion",
}


def _read(d: Path, name: str):
    p = d / name
    return json.loads(p.read_text()) if p.exists() else None


def _scenario_label(s: str) -> str:
    key, _, val = s.partition("=")
    return f"{SCENARIO_LABELS.get(key, key)} = {val}" if val else SCENARIO_LABELS.get(key, key)


def _empty(ax, title: str, msg: str) -> None:
    ax.set_title(title, loc="left")
    ax.text(0.5, 0.5, msg, ha="center", va="center", color=TEXT_2, fontsize=6, transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


def _panel_oat(ax, oat) -> None:
    title = "a  One-at-a-time scenarios (pilot lesions)"
    if not oat or not oat.get("runs"):
        return _empty(ax, title, "one-at-a-time runs not yet computed")
    runs = oat["runs"]
    lesions = sorted(runs)
    scen = []
    for lid in lesions:
        for s in runs[lid]:
            if s != "baseline" and s not in scen:
                scen.append(s)
    for j, lid in enumerate(lesions):
        color = SERIES[j % len(SERIES)]
        off = (j - (len(lesions) - 1) / 2) * 0.22
        for k, s in enumerate(scen):
            r = runs[lid].get(s)
            if not r or not r.get("complete"):
                continue
            d = r.get("ivl_onset_load_minus_baseline")
            y = k + off
            if d is None:
                if r.get("ivl_onset_load_censored_vs_baseline"):
                    ax.plot(0.0, y, marker="o", mfc="white", mec=color, ms=3.5, mew=0.8, ls="none")
                continue
            ax.plot(d, y, marker="o", color=color, ms=3.5, mec="white", mew=0.4, ls="none")
            if r.get("ivl_fracture_changed_vs_baseline"):
                ax.plot(d, y, marker="o", mfc="none", mec=TEXT, ms=6.5, mew=0.8, ls="none")
    ax.axvline(0.0, color=GRID, lw=1.0, zorder=0)
    ax.set_yticks(range(len(scen)))
    ax.set_yticklabels([_scenario_label(s) for s in scen], fontsize=5.5)
    ax.invert_yaxis()
    ax.set_xlabel("IVL damage-onset load minus baseline (MPa)")
    ax.set_title(title, loc="left")
    ax.grid(axis="y", visible=False)
    from matplotlib.lines import Line2D
    hand = [Line2D([], [], color=SERIES[j % len(SERIES)], marker="o", ls="none", ms=3.5, label=lid)
            for j, lid in enumerate(lesions)]
    hand += [Line2D([], [], color=TEXT_2, marker="o", mfc="white", ls="none", ms=3.5, label="onset censored"),
             Line2D([], [], color=TEXT, marker="o", mfc="none", ls="none", ms=6.5, label="damage outcome changed")]
    ax.legend(handles=hand, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=5.5)


def _panel_morris(ax, morris, output: str, title: str, xlabel: str, color: str) -> None:
    pooled = (morris or {}).get("pooled", {}).get(output)
    if not pooled or not pooled.get("indices"):
        return _empty(ax, title, "Morris screening not yet computed")
    idx = pooled["indices"]
    names = sorted(idx, key=lambda n: idx[n].get("mu_star") or 0.0)
    mu = np.array([idx[n].get("mu_star") or 0.0 for n in names])
    conf = np.array([idx[n].get("mu_star_conf") or 0.0 for n in names])
    y = np.arange(len(names))
    ax.barh(y, mu, height=0.6, color=color, edgecolor="white", linewidth=0.5)
    ax.errorbar(mu, y, xerr=conf, fmt="none", ecolor=TEXT_2, elinewidth=0.7, capsize=1.5)
    ax.set_yticks(y)
    ax.set_yticklabels([INPUT_LABELS.get(n, n) for n in names], fontsize=5.5)
    ax.set_xlim(left=0)
    ax.set_xlabel(xlabel)
    ax.set_title(f"{title} (trajectories: {pooled.get('n_trajectories')})", loc="left")
    ax.grid(axis="y", visible=False)


def _panel_sobol(ax, sobol, best: str | None) -> None:
    title = "d  Sobol indices of the best predictor"
    key = best[len("pred_"):] if best and best.startswith("pred_") else best
    lr = (sobol or {}).get("lesion_results") or {}
    have = {lid: r["predictors"][key] for lid, r in lr.items()
            if key and key in (r.get("predictors") or {}) and (r["predictors"][key] or {}).get("indices")}
    if not have:
        return _empty(ax, title, "Sobol analysis not yet computed")
    names = list(next(iter(have.values()))["indices"])
    y = np.arange(len(names))
    lids = sorted(have)
    for j, lid in enumerate(lids):
        off = (j - (len(lids) - 1) / 2) * 0.18
        ind = have[lid]["indices"]
        st = [ind[n].get("ST") for n in names]
        s1 = [ind[n].get("S1") for n in names]
        c = SERIES[j % len(SERIES)]
        ax.plot(st, y + off, marker="o", color=c, ms=3.5, mec="white", mew=0.4, ls="none", label=lid)
        ax.plot(s1, y + off, marker="o", mfc="white", mec=c, ms=3.0, mew=0.7, ls="none")
    ax.set_yticks(y)
    ax.set_yticklabels([INPUT_LABELS.get(n, n) for n in names], fontsize=5.5)
    ax.invert_yaxis()
    ax.set_xlim(-0.05, 1.05)
    ax.set_xlabel("Sobol index (total order filled, first order open)")
    ax.set_title(title, loc="left")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", frameon=False, fontsize=5)


def _panel_length(ax, l3d) -> None:
    title = "e  3D length series: onset load, 3D / 2D"
    runs = (l3d or {}).get("runs") or {}
    rows = {}
    for r in runs.values():
        if r.get("model") != "3d":
            continue
        ref = runs.get(f"{r['lesion']}|2d") or {}
        p3, p2 = (r.get("onset") or {}).get("p"), (ref.get("onset") or {}).get("p")
        if p3 and p2:
            rows.setdefault(r["lesion"], []).append((r["length_mm"], p3 / p2))
    if not rows:
        return _empty(ax, title, "3D length series not yet computed")
    for j, (lid, pts) in enumerate(sorted(rows.items())):
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", color=SERIES[j % len(SERIES)], ms=3.5,
                label=lid)
    ax.axhline(1.0, color=GRID, lw=1.0, ls="--", zorder=0)
    ax.set_xlabel("model length (mm)")
    ax.set_ylabel("onset load, 3D / 2D")
    ax.set_title(title, loc="left")
    ax.legend(loc="best", frameon=False, fontsize=5)


def make(name: str = "fig7_sensitivity", results_dir: str | Path | None = None) -> str:
    import matplotlib.pyplot as plt

    d = Path(results_dir) if results_dir is not None else RESULTS_DIR
    oat, morris, sobol, l3d = (_read(d, n) for n in (OAT_JSON, MORRIS_JSON, SOBOL_JSON, L3D_JSON))
    if not any((oat, morris, sobol, l3d)):
        raise FileNotFoundError(f"no sensitivity results in {d}")
    metrics = _read(d, METRICS_JSON) or {}
    best = (((metrics.get("outcomes") or {}).get("ivl_fracture") or {}).get("fujino_comparison") or {}).get(
        "best_single_mechanics")
    apply_style()
    fig = plt.figure(figsize=(7.1, 7.0))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1, 1], wspace=0.75, hspace=0.75)
    _panel_oat(fig.add_subplot(gs[0, :]), oat)
    _panel_length(fig.add_subplot(gs[2, 1]), l3d)
    _panel_morris(fig.add_subplot(gs[1, 0]), morris, "ivl_onset_load", "b  Morris: IVL onset load",
                  "μ* (MPa per input range)", SERIES[0])
    _panel_morris(fig.add_subplot(gs[1, 1]), morris, "ivl_fracture", "c  Morris: IVL through-thickness damage",
                  "μ* (change in outcome per input range)", SERIES[1])
    _panel_sobol(fig.add_subplot(gs[2, 0]), sobol, best)
    return str(save(fig, name))


def _labels_have_no_typed_numbers() -> bool:
    """Static check used by the tests: label tables carry no digits (numbers come from results files)."""
    return not any(re.search(r"\d", v) for v in {**INPUT_LABELS, **SCENARIO_LABELS}.values())

"""Figure 1: study design.

(a) Pipeline: structured lesion cohort (core factorial and hold-out set; counts and factor levels
    read from ``cohort/cohort.csv``) -> two separate branches: the pre-treatment linear-elastic
    analysis giving the mechanics predictors, and the nonlinear phase-field outcome model (its IVL
    load factor from the transient analysis is an outcome-model input, never a predictor) -> the
    predictive metrics. Text labels only; the only numbers are counts read from the cohort table.
(b) Example core lesion (``design.fig1_example_lesion``) meshed with
    :func:`dtivl.meshing.mask_to_mesh.mesh_from_labelmap` into a temporary file, cells coloured by
    tissue tag (``TISSUE_RGB``); scale bar from the mesh coordinates.
(c) Load protocols on the nominal increments of
    :data:`dtivl.solvers.nonlinear.DEFAULT_STEPS` (pseudo-time): lumen pressure of the physiological,
    IVL-balloon-alone, plain-balloon and IVL protocols (IVL peak ``p_top`` read from the cohort run
    record of the example lesion, or of the first core lesion that has one), and, in a separate
    panel, the radius of the expanding stent cylinder of the stent-stiffness identification
    (``results/calibration_stent.json``: deployment to ``R_dep`` = the target radius, recoil to
    ``R_f``; start ``R_min`` = the template lesion's lumen radius from ``cohort.csv`` less the
    solver's start gap).
"""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from dtivl import params as P
from dtivl.drivers.common import COHORT_DIR, RESULTS_DIR, load_mask, read_json
from dtivl.drivers.figstyle import GRID, SERIES, SURFACE, TEXT, TEXT_2, TISSUE_RGB, apply_style, save
from dtivl.geometry.labels import MATERIAL_NAMES

NAME = "fig1_design"
FIGSIZE = (7.1, 6.4)               # inches (max width 7.1 in)
RUNS_DIR = RESULTS_DIR / "cohort_runs"


# ---------------------------------------------------------------- data
def example_lesion() -> str:
    return str(P.raw("design.fig1_example_lesion")["value"])


def cohort_counts() -> dict:
    """Counts of the cohort table: core set size and factor levels, hold-out set size."""
    df = pd.read_csv(COHORT_DIR / "cohort.csv")
    core, hold = df[df["set"] == "core"], df[df["set"] == "holdout"]
    return {"n_core": len(core), "n_arc": core["ca_arc_deg"].nunique(), "n_thick": core["ca_thickness"].nunique(),
            "n_morph": core["morphology"].nunique(), "n_holdout": len(hold),
            "core_ids": core["lesion_id"].tolist()}


def ivl_peak_load(preferred: str | None = None) -> tuple[str, float]:
    """(lesion, p_top) of the IVL protocol: ``protocols.ivl.summary.p_top`` of the cohort run record
    of ``preferred`` (default :func:`example_lesion`), else of the first core lesion in
    ``cohort.csv`` order whose record has it."""
    preferred = preferred or example_lesion()
    for lid in [preferred] + cohort_counts()["core_ids"]:
        path = RUNS_DIR / f"{lid}.json"
        if not path.exists():
            continue
        p_top = ((json.loads(path.read_text()).get("protocols") or {}).get("ivl") or {}).get("summary", {}).get("p_top")
        if p_top is not None:
            return lid, float(p_top)
    raise FileNotFoundError(f"no cohort run record in {RUNS_DIR} has protocols.ivl.summary.p_top")


def stent_radius_path() -> dict:
    """Cylinder radius of the stent-stiffness identification on nominal increments: from ``R_min``
    to ``R_dep`` in ``DEFAULT_STEPS['stent_expand']`` increments, then the recoil to ``R_f`` in the
    identification's ``RECOIL_RAMP_STEPS`` increments."""
    from dtivl.drivers.stage_pilot import RECOIL_RAMP_STEPS
    from dtivl.solvers.nonlinear import DEFAULT_STEPS, STENT_START_GAP_FRAC

    cal = read_json("calibration_stent")
    df = pd.read_csv(COHORT_DIR / "cohort.csv").set_index("lesion_id")
    r_lumen = float(df.loc[cal["lesion_template"], "r_lumen"])
    R_min = (1.0 - STENT_START_GAP_FRAC) * r_lumen
    R_target = 0.5 * P.get("geometry.stent_target_diameter_mm").value
    n_exp, n_rec = DEFAULT_STEPS["stent_expand"], RECOIL_RAMP_STEPS
    R = np.concatenate([[R_min], np.linspace(R_min, cal["R_dep"], n_exp + 1),
                        np.linspace(cal["R_dep"], cal["R_f"], n_rec + 1)[1:]])
    t = np.arange(len(R))
    return {"t": t, "R": R, "R_min": R_min, "R_target": R_target, "R_dep": cal["R_dep"], "R_f": cal["R_f"],
            "n_contact": 1, "n_exp": n_exp, "lesion": cal["lesion_template"]}


def pressure_protocols(p_top: float) -> list[dict]:
    """Lumen pressure (MPa) on nominal increments for the pressure protocols of the cohort runs."""
    from dtivl.solvers.nonlinear import DEFAULT_STEPS

    n = DEFAULT_STEPS["pressure"]
    ni, npu, nu = DEFAULT_STEPS["ivl_inflate"], DEFAULT_STEPS["ivl_pulse"], DEFAULT_STEPS["ivl_unload"]
    p_phys = P.get("loads.physiological_pressure").value
    p_bal = P.get("loads.ivl_balloon_pressure_atm").value
    p_poba = P.get("loads.poba_pressure_atm").value
    ramp = lambda a, b, k: np.linspace(a, b, k + 1)  # noqa: E731
    ivl = np.concatenate([ramp(0.0, p_bal, ni), ramp(p_bal, p_top, npu)[1:], ramp(p_top, 0.0, nu)[1:]])
    return [
        {"key": "physiological", "label": f"Physiological ({_lit_value('loads.physiological_pressure')})",
         "p": ramp(0.0, p_phys, n), "marker": "s"},
        {"key": "balloon_only", "label": f"IVL balloon alone ({_lit_value('loads.ivl_balloon_pressure_atm')})",
         "p": ramp(0.0, p_bal, n), "marker": "o"},
        {"key": "balloon_10atm", "label": f"Plain balloon ({_lit_value('loads.poba_pressure_atm')})",
         "p": ramp(0.0, p_poba, n), "marker": "^"},
        {"key": "ivl", "label": "IVL (pulse-equivalent peak)", "p": ivl, "marker": "D",
         "phases": (ni, ni + npu, ni + npu + nu)},
    ]


def _lit_value(key: str) -> str:
    """Stated value and unit of the literature row behind a config entry (e.g. ``4 atm``)."""
    row = P.load_literature()[P.raw(key)["lit"]]
    return f"{row.value_str} {row.unit}"


def lesion_mesh(lesion_id: str) -> dict:
    """Triangles, vertex coordinates and cell tags of the 2D mesh of ``lesion_id`` (temporary file)."""
    from dtivl.meshing.mask_to_mesh import load_mesh, mesh_from_labelmap

    plm = load_mask(lesion_id)
    with tempfile.TemporaryDirectory() as td:
        info = mesh_from_labelmap(plm, Path(td) / f"{lesion_id}.msh")
        md = load_mesh(info["msh"], gdim=2)
    geo = md.mesh.geometry
    tri = np.asarray(geo.dofmaps(0) if callable(getattr(geo, "dofmaps", None)) else geo.dofmap)
    tags = np.zeros(tri.shape[0], dtype=int)
    tags[md.cell_tags.indices] = md.cell_tags.values
    return {"xy": np.asarray(geo.x[:, :2]), "tri": tri, "tags": tags, "n_cells": int(info["n_cells"])}


# ---------------------------------------------------------------- panel (a)
def _box(ax, x, y, w, h, title, body="", *, fc="white", ec=TEXT_2, lw=0.7, ls="-", fs=6.0, title_fs=6.5):
    from matplotlib.patches import FancyBboxPatch

    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.012", fc=fc, ec=ec, lw=lw,
                                ls=ls, mutation_aspect=0.5, zorder=2))
    if body:
        ax.text(x + w / 2, y + h - 0.012, title, ha="center", va="top", fontsize=title_fs, fontweight="bold",
                color=TEXT, zorder=3)
        ax.text(x + w / 2, y + h / 2 - 0.022, body, ha="center", va="center", fontsize=fs, color=TEXT,
                linespacing=1.25, zorder=3)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=title_fs, fontweight="bold",
                color=TEXT, zorder=3)


def _arrow(ax, xy0, xy1, *, color=TEXT_2, ls="-", rad=0.0):
    from matplotlib.patches import FancyArrowPatch

    ax.add_patch(FancyArrowPatch(xy0, xy1, arrowstyle="-|>", mutation_scale=7, lw=0.8, color=color, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0, zorder=4))


def panel_pipeline(ax) -> None:
    c = cohort_counts()
    index_name = P.raw("design.composite_index_name")["value"]
    sets = P.raw("metrics.sets")["value"]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    # cohort
    _box(ax, 0.17, 0.835, 0.66, 0.155, "Structured lesion cohort (2D cross-sections from polar label maps)",
         f"Core factorial: {c['n_arc']} calcium arcs × {c['n_thick']} thicknesses × {c['n_morph']} morphologies "
         f"({c['n_core']} lesions)\nHold-out set: space-filling Latin hypercube over continuous ranges "
         f"({c['n_holdout']} lesions)")
    # lanes (predictor / outcome separation)
    lane = dict(fc=SURFACE, ec=GRID, lw=0.8)
    _box(ax, 0.01, 0.205, 0.43, 0.575, "", **lane)
    _box(ax, 0.47, 0.205, 0.52, 0.575, "", **lane)
    ax.text(0.225, 0.762, "PREDICTORS: pre-treatment, linear", ha="center", va="top", fontsize=6.5,
            fontweight="bold", color=SERIES[0])
    ax.text(0.73, 0.762, "OUTCOMES: treatment simulation, nonlinear", ha="center", va="top", fontsize=6.5,
            fontweight="bold", color=SERIES[1])
    ax.plot([0.455, 0.455], [0.215, 0.77], color=TEXT_2, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.text(0.455, 0.49, "no outcome information\nenters the predictors", rotation=90, ha="center", va="center",
            fontsize=5.5, color=TEXT_2, bbox=dict(fc="white", ec="none", pad=0.6), zorder=3)
    # predictor branch
    _box(ax, 0.04, 0.56, 0.37, 0.14, "Linear-elastic analysis",
         "small strain, intact calcium,\nlumen pressure of the IVL balloon", ec=SERIES[0])
    _box(ax, 0.04, 0.235, 0.37, 0.25, "Mechanics predictors",
         "principal stress (percentiles, mean, peak),\nstrain energy density, over-threshold\narea fractions, "
         "edge stress concentration;\ngeometry descriptors; Fujino score (comparator)", ec=SERIES[0])
    _arrow(ax, (0.225, 0.56), (0.225, 0.485), color=SERIES[0])
    # outcome branch
    _box(ax, 0.50, 0.615, 0.46, 0.105, "Transient single-pulse analysis",
         "IVL load factor: outcome-model input, never a predictor", ec=SERIES[1], ls=(0, (3, 1.5)))
    _box(ax, 0.50, 0.395, 0.46, 0.18, "Finite-strain phase-field fracture model",
         "IVL balloon alone  ·  plain balloon  ·  IVL (single-pulse\nequivalent load)  ·  stent without IVL  ·  "
         "stent after IVL", ec=SERIES[1])
    _box(ax, 0.50, 0.235, 0.46, 0.10, "Outcomes",
         "through-thickness fracture, damage-onset pressure; stent expansion", ec=SERIES[1])
    _arrow(ax, (0.73, 0.615), (0.73, 0.575), color=SERIES[1])
    _arrow(ax, (0.73, 0.395), (0.73, 0.335), color=SERIES[1])
    # cohort -> branches
    _arrow(ax, (0.225, 0.835), (0.225, 0.78))
    _arrow(ax, (0.73, 0.835), (0.73, 0.78))
    # metrics
    _box(ax, 0.06, 0.005, 0.88, 0.15, f"Predictive metrics (derivation: {sets['derivation']} set; "
                                      f"test: {sets['test'].replace('holdout', 'hold-out')} set)",
         f"ROC analysis with Youden threshold  ·  leave-one-out cross-validation  ·  composite mechanics index "
         f"({index_name})\ncomparison with the Fujino score  ·  incremental value over geometry")
    _arrow(ax, (0.225, 0.235), (0.30, 0.155), color=SERIES[0])
    _arrow(ax, (0.73, 0.235), (0.70, 0.155), color=SERIES[1])


# ---------------------------------------------------------------- panel (b)
def _scale_length(span: float) -> float:
    """Largest 1-2-5 x 10^k length not above a quarter of ``span``."""
    target = span / 4.0
    k = math.floor(math.log10(target))
    return max(m * 10.0 ** k for m in (1, 2, 5) if m * 10.0 ** k <= target)


def panel_mesh(ax, lesion_id: str) -> None:
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Circle

    m = lesion_mesh(lesion_id)
    xy, tri, tags = m["xy"], m["tri"], m["tags"]
    present = sorted(set(tags.tolist()))
    names = [MATERIAL_NAMES[t] for t in present]
    cmap = ListedColormap([TISSUE_RGB[n] for n in names])
    norm = BoundaryNorm(np.arange(len(present) + 1) - 0.5, len(present))
    idx = np.searchsorted(present, tags)
    ax.tripcolor(xy[:, 0], xy[:, 1], tri, facecolors=idx, cmap=cmap, norm=norm, edgecolors="none", zorder=1)
    ax.triplot(xy[:, 0], xy[:, 1], tri, color=TEXT_2, lw=0.06, alpha=0.6, zorder=2)
    r = np.hypot(xy[:, 0], xy[:, 1])
    R_out = float(r.max())
    ax.add_patch(Circle((0, 0), float(r.min()), fc=TISSUE_RGB["lumen"], ec="none", zorder=0))
    ax.set_aspect("equal")
    lim = 1.42 * R_out
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.axis("off")
    # label anchors: cell centroids per tag
    cen = xy[tri].mean(axis=1)
    ang = np.arctan2(cen[:, 1], cen[:, 0])
    rad = np.hypot(cen[:, 0], cen[:, 1])
    tag_of = {v: k for k, v in MATERIAL_NAMES.items()}
    ca = tags == tag_of["calcium"]
    th_ca = float(np.arctan2(np.sin(ang[ca]).mean(), np.cos(ang[ca]).mean())) if ca.any() else 0.0

    def anchor(name, theta):
        sel = tags == tag_of[name]
        d = np.abs(np.angle(np.exp(1j * (ang[sel] - theta))))
        near = np.flatnonzero(sel)[d <= np.quantile(d, 0.02)]
        i = near[np.argsort(rad[near])[len(near) // 2]]
        return cen[i]

    labels = [("Calcium", "calcium", th_ca, (0.98, 0.97), "right"),
              ("Fibrous intima", "fibrous", th_ca + math.pi, (0.02, 0.97), "left"),
              ("Wall (media + adventitia)", "wall", th_ca - 0.75 * math.pi, (0.02, 0.03), "left")]
    for text, name, theta, pos, ha in labels:
        if tag_of[name] not in present:
            continue
        ax.annotate(text, xy=anchor(name, theta), xycoords="data", xytext=pos, textcoords="axes fraction",
                    ha=ha, va="top" if pos[1] > 0.5 else "bottom", fontsize=6.5, color=TEXT,
                    arrowprops=dict(arrowstyle="-", color=TEXT, lw=0.6, shrinkA=1, shrinkB=0), zorder=5)
    ax.text(0, 0, "Lumen", ha="center", va="center", fontsize=6.5, color=TEXT, zorder=5)
    L = _scale_length(2 * R_out)
    x1, y0 = lim * 0.95, -lim * 0.90
    ax.plot([x1 - L, x1], [y0, y0], color=TEXT, lw=1.5, solid_capstyle="butt", zorder=5)
    ax.text(x1 - L / 2, y0 + 0.04 * lim, f"{L:g} mm", ha="center", va="bottom", fontsize=6, color=TEXT, zorder=5)
    ax.set_title(f"{lesion_id} ({m['n_cells']} cells)", fontsize=6.5, color=TEXT_2, pad=2)


# ---------------------------------------------------------------- panel (c)
def panel_protocols(ax, lesion_id: str, p_top: float) -> None:
    protos = pressure_protocols(p_top)
    for i, pr in enumerate(protos):
        t = np.arange(len(pr["p"]))
        # the IVL balloon-alone ramp coincides with the IVL inflation: open markers drawn on top
        style = (dict(ms=4.2, mfc="white", mec=SERIES[i], mew=0.8, zorder=4) if pr["key"] == "balloon_only"
                 else dict(ms=2.6, mew=0, zorder=3))
        ax.plot(t, pr["p"], color=SERIES[i], marker=pr["marker"], lw=1.2, label=pr["label"], **style)
    ivl = protos[-1]
    ni, npk, nend = ivl["phases"]
    ax.annotate(f"p_top ({lesion_id})", xy=(npk, p_top), xytext=(npk + 1.0, p_top + 0.12), fontsize=6,
                color=TEXT, ha="left", va="bottom",
                arrowprops=dict(arrowstyle="-", color=TEXT_2, lw=0.6, shrinkA=0, shrinkB=2))
    for x0, x1, name in ((0, ni, "inflation"), (ni, npk, "pulse"), (npk, nend, "unloading")):
        ax.text((x0 + x1) / 2, -0.1, name, transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=5.5, color=TEXT_2)
    for x in (ni, npk):
        ax.axvline(x, color=GRID, lw=0.8, ls=":", zorder=1)
    ax.set_xlim(0, nend)
    ax.set_ylim(0, 1.3 * max(pr["p"].max() for pr in protos))
    ax.set_xlabel("Pseudo-time (nominal load increments)", labelpad=13)
    ax.set_ylabel("Lumen pressure (MPa)")
    ax.legend(loc="upper right", frameon=False, handlelength=2.2, borderaxespad=0.2)


def panel_stent(ax) -> None:
    s = stent_radius_path()
    ax.plot(s["t"], s["R"], color=SERIES[0], marker="o", ms=2.2, lw=1.2, mew=0, zorder=3)
    ax.axhline(s["R_target"], color=TEXT_2, lw=0.7, ls="--", zorder=1)
    ax.text(0.5, s["R_target"] - 0.01, "target radius", va="top", ha="left", fontsize=5.5, color=TEXT_2)
    ax.annotate("recoil", xy=(s["t"][-1], s["R_f"]), xytext=(s["t"][-1] - 4, s["R_f"] - 0.18), fontsize=5.5,
                color=TEXT, ha="right", arrowprops=dict(arrowstyle="-", color=TEXT_2, lw=0.6, shrinkB=2))
    ax.text(2.5, s["R_min"], "start: lumen radius", va="center", ha="left", fontsize=5.5, color=TEXT_2)
    ax.set_xlim(0, s["t"][-1])
    ax.set_ylim(0.95 * s["R_min"], None)
    ax.set_xlabel("Pseudo-time (increments)", labelpad=13)
    ax.set_ylabel("Stent cylinder radius (mm)")
    ax.set_title(f"Stent, calcium-free\n{s['lesion']}", fontsize=6.5, color=TEXT_2, pad=2)


# ---------------------------------------------------------------- figure
def make() -> str:
    import matplotlib.pyplot as plt

    apply_style()
    lesion = example_lesion()
    p_lesion, p_top = ivl_peak_load(lesion)
    fig = plt.figure(figsize=FIGSIZE)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1.0], width_ratios=[1.0, 1.25, 0.62], hspace=0.12,
                          wspace=0.42, left=0.02, right=0.985, top=0.985, bottom=0.075)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[1, 2])
    panel_pipeline(ax_a)
    panel_mesh(ax_b, lesion)
    panel_protocols(ax_c, p_lesion, p_top)
    panel_stent(ax_d)
    for ax, letter, dx in ((ax_a, "a", 0.0), (ax_b, "b", 0.0), (ax_c, "c", -0.12)):   # (c): both load panels
        ax.text(dx, 1.0, f"({letter})", transform=ax.transAxes, fontsize=8, fontweight="bold", color=TEXT,
                ha="left", va="bottom" if ax is not ax_a else "top")
    return str(save(fig, NAME))

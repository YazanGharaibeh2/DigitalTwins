"""Figure 5: outcomes of the core factorial (nonlinear damage and stent runs).

(a) Outcome maps: for each damage protocol of ``figures.fig5_outcomes.map_protocols``
    (rows) and each morphology (columns), a grid of calcium arc (x) by
    thickness (y) whose cells give the damage class of the lesion
    (``figures.fig5_outcomes.onset_evidence``): through-thickness damage
    (dark), partial damage only (hatched mid grey), no damage onset (light),
    not run or failed (white with a cross). IVL cells print the damage-onset
    load (MPa). The classes are greys so that hue keeps one meaning in the
    figure (morphology, panels b and c).
(b) Stent expansion without IVL and after IVL for each lesion (slope chart
    grouped by morphology; colour and marker by morphology; open markers:
    pressure-limited expansion, status 'limited'), with the expansion of a
    stent at its target diameter (``figures.fig5_outcomes.expansion_reference``).
(c) Damage-onset load under IVL against the calcium arc, one panel per
    morphology, marker by thickness. Lesions without damage onset are drawn
    open at their peak IVL load (right-censored, arrow up): ``ivl_p_top`` of
    the outcome table, else the ``p_top`` of the lesion's IVL record. As in the
    metrics stage, an onset recorded after the loading phases is drawn at the
    peak load.

Reads results/cohort_outcomes.csv (columns of stage_cohort_runs._flat),
results/cohort_runs/<lesion>.json (peak load only, where the table lacks it)
and params/ only. Lesions or protocols without a record are drawn as not run,
so the figure builds with a partial cohort.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
from matplotlib.colors import to_hex, to_rgb

from dtivl import params as P
from dtivl.drivers.common import RESULTS_DIR
from dtivl.drivers.figstyle import GRID, SERIES, TEXT, TEXT_2, apply_style, save
from dtivl.drivers.stage_cohort_runs import OUT_DIR as COHORT_RUNS_DIR
from dtivl.drivers.stage_figures import MORPH_LABEL
from dtivl.drivers.stage_metrics import IVL_LOADING_PHASES

OUTCOMES_CSV = RESULTS_DIR / "cohort_outcomes.csv"
CFG = "figures.fig5_outcomes"
#: damage classes of panel (a), in legend order
CLASSES = ("through", "partial", "none", "na")
CLASS_LABEL = {"through": "through-thickness damage", "partial": "partial damage only (onset, not through)",
               "none": "no damage onset", "na": "not run / failed"}
MORPH_MARKER = ("o", "s", "^", "D")       # by morphology, in design.core_factorial order
THICK_MARKER = ("o", "s", "^", "D", "v")  # by thickness level, in design.core_factorial order


def read_outcomes() -> pd.DataFrame:
    """The per-lesion outcome table written by stage_cohort_runs.aggregate()."""
    return pd.read_csv(OUTCOMES_CSV)


def _class_colors() -> dict:
    """Neutral grey steps of the figure tokens: light (no onset), mid (partial), dark (through-thickness)."""
    lo, hi = np.array(to_rgb(GRID)), np.array(to_rgb(TEXT_2))
    return {"none": GRID, "partial": to_hex(0.5 * (lo + hi)), "through": TEXT_2, "na": "white"}


def _ok_status(protocol: str) -> list:
    """Statuses accepted for a protocol: its first entry in metrics.outcomes, else ['ok']."""
    for oc in P.raw("metrics.outcomes")["value"]:
        if oc.get("protocol") == protocol:
            return list(oc.get("ok_status", ["ok"]))
    return ["ok"]


def _flag(s: pd.Series | None, index) -> pd.Series:
    """Boolean column as 1.0 / 0.0 / NaN (csv round trips give bools, strings or NaN)."""
    if s is None:
        return pd.Series(np.nan, index=index)
    return s.map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0, 1: 1.0, 0: 0.0}).astype(float)


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce") if col in df else pd.Series(np.nan, index=df.index)


def damage_class(df: pd.DataFrame, protocol: str) -> pd.Series:
    """Damage class of every row under ``protocol`` (see figures.fig5_outcomes.onset_evidence)."""
    st = df.get(f"{protocol}_status")
    ok = st.isin(_ok_status(protocol)) if st is not None else pd.Series(False, index=df.index)
    fr = _flag(df.get(f"{protocol}_fracture"), df.index)
    onset = pd.Series(False, index=df.index)
    for key in P.raw(f"{CFG}.onset_evidence")["value"]:
        col = f"{protocol}_{key}"
        if col not in df:
            continue
        v = _num(df, col)
        onset |= v.notna() if key == "onset" else (v > 0)    # onset load recorded; counts and fractions > 0
    out = pd.Series("none", index=df.index)
    out[onset] = "partial"
    out[fr == 1.0] = "through"
    out[~ok | fr.isna()] = "na"
    return out


def peak_load(df: pd.DataFrame) -> pd.Series:
    """Peak IVL load of every row (censoring value): ivl_p_top, else p_top of the lesion's IVL record."""
    p = _num(df, "ivl_p_top")
    for i in df.index[p.isna()]:
        path = COHORT_RUNS_DIR / f"{df.at[i, 'lesion_id']}.json"
        if path.exists():
            v = ((json.loads(path.read_text()).get("protocols", {}).get("ivl") or {}).get("summary") or {}).get("p_top")
            p.at[i] = np.nan if v is None else float(v)
    return p


def onset_table(df: pd.DataFrame) -> pd.DataFrame:
    """IVL onset rows with ok status: ``value`` (onset load, or the peak load), ``event`` and ``after_peak``."""
    t = df[df["ivl_status"].isin(_ok_status("ivl"))].copy() if "ivl_status" in df else df.iloc[0:0].copy()
    on = _num(t, "ivl_onset")
    ph = t["ivl_onset_phase"] if "ivl_onset_phase" in t else pd.Series(None, index=t.index, dtype=object)
    t["event"] = on.notna()
    t["after_peak"] = t["event"] & ~ph.isin(IVL_LOADING_PHASES)
    top = peak_load(t)
    t["value"] = np.where(t["event"] & ~t["after_peak"], on, top)
    return t[np.isfinite(t["value"].to_numpy(float))]


def _lit_label(dotted: str) -> str:
    """'4 atm': value and unit of the literature row behind a config entry, as stated in the table."""
    row = P.load_literature()[P.raw(dotted)["lit"]]
    return f"{float(row.value_str):g} {row.unit}"


def protocol_label(protocol: str) -> str:
    return {"balloon_only": f"IVL balloon alone,\n{_lit_label('loads.ivl_balloon_pressure_atm')}",
            "balloon_10atm": f"Plain balloon,\n{_lit_label('loads.poba_pressure_atm')}",
            "ivl": "IVL\n(balloon + pulse)"}.get(protocol, protocol)


def expansion_reference() -> float | None:
    """Expansion (%) of a stent at its target diameter: 100 (D_stent / D_ref)^2; None if not configured."""
    if P.raw(f"{CFG}.expansion_reference")["value"] != "stent_target":
        return None
    d_s, d_r = P.get("geometry.stent_target_diameter_mm").value, P.get("geometry.reference_diameter_mm").value
    return 100.0 * (d_s / d_r) ** 2


def _bare(ax) -> None:
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


def _not_available(ax) -> None:
    ax.text(0.5, 0.5, "not available", ha="center", va="center", color=TEXT_2, transform=ax.transAxes)


def _map_panel(fig, gs, df, morphs, arcs, ths, protocols) -> list:
    """Panel (a): one arc x thickness grid per protocol (row) and morphology (column)."""
    from matplotlib.patches import Rectangle

    col = _class_colors()
    axes = []
    for i, proto in enumerate(protocols):
        cls = damage_class(df, proto)
        on = _num(df, f"{proto}_onset")
        run = cls != "na"
        for j, m in enumerate(morphs):
            ax = fig.add_subplot(gs[i, j])
            axes.append(ax)
            for ia, a in enumerate(arcs):
                for it, t in enumerate(ths):
                    sel = df.index[(df["morphology"] == m) & np.isclose(df["ca_arc_deg"], a)
                                   & np.isclose(df["ca_thickness"], t)]
                    c = cls[sel[0]] if len(sel) else "na"
                    ax.add_patch(Rectangle((ia - 0.46, it - 0.46), 0.92, 0.92, fc=col[c],
                                           ec=GRID if c == "na" else "none", lw=0.4,
                                           hatch="//////" if c == "partial" else None))
                    if c == "na":
                        ax.plot([ia - 0.25, ia + 0.25], [it - 0.25, it + 0.25], color=TEXT_2, lw=0.6)
                        ax.plot([ia - 0.25, ia + 0.25], [it + 0.25, it - 0.25], color=TEXT_2, lw=0.6)
                    elif proto == "ivl" and np.isfinite(on[sel[0]]):
                        ax.text(ia, it, f"{on[sel[0]]:#.3g}", ha="center", va="center", fontsize=5,
                                color="white" if c == "through" else TEXT)
            ax.set_xlim(-0.5, len(arcs) - 0.5)
            ax.set_ylim(-0.5, len(ths) - 0.5)
            ax.set_aspect("equal")
            _bare(ax)
            ax.tick_params(length=0, pad=1.5)
            ax.set_xticks(range(len(arcs)))
            ax.set_yticks(range(len(ths)))
            ax.set_xticklabels([f"{a:g}" for a in arcs] if i == len(protocols) - 1 else [])
            ax.set_yticklabels([f"{t:g}" for t in ths] if j == 0 else [])
            if i == 0:
                ax.set_title(MORPH_LABEL.get(m, m), fontsize=6.5, color=TEXT)
            if i == len(protocols) - 1:
                ax.set_xlabel("Calcium arc (°)", fontsize=6)
            if j == 0:
                ax.set_ylabel("Thickness (mm)", fontsize=6)
                n_t, n_r = int((cls[run] == "through").sum()), int(run.sum())
                ax.text(-0.52, 0.5, f"{protocol_label(proto)}\n{n_t} of {n_r} through", transform=ax.transAxes,
                        rotation=90, ha="center", va="center", fontsize=6, color=TEXT, linespacing=1.2)
    return axes


def _expansion_panel(ax, df, morphs, colors) -> str:
    """Panel (b): slope chart of stent expansion without and after IVL; returns the panel title."""
    cols = {k: f"{k}_stent_expansion_pct" for k in ("stent_no_ivl", "stent_post_ivl")}
    if not all(c in df for c in cols.values()):
        _not_available(ax)
        ax.set_xticks([])
        return "b  Stent expansion"
    vals, lim = {}, {}
    for k, c in cols.items():
        st = df.get(f"{k}_status", pd.Series(None, index=df.index))
        vals[k] = _num(df, c).where(st.isin(_ok_status(k)))
        lim[k] = st == "limited"
    n_pairs = 0
    for g, m in enumerate(morphs):
        x0, color, mk = 3 * g, colors[m], MORPH_MARKER[g % len(MORPH_MARKER)]
        for i in df.index[df["morphology"] == m]:
            a, b = vals["stent_no_ivl"][i], vals["stent_post_ivl"][i]
            if np.isfinite(a) and np.isfinite(b):
                ax.plot([x0, x0 + 1], [a, b], color=color, lw=0.6, alpha=0.6, zorder=2)
                n_pairs += 1
            for dx, k, v in ((0, "stent_no_ivl", a), (1, "stent_post_ivl", b)):
                if np.isfinite(v):
                    ax.plot(x0 + dx, v, marker=mk, ms=3.5, ls="none", mec=color, mew=0.8, zorder=3,
                            mfc="white" if lim[k][i] else color)
        ax.text(x0 + 0.5, -0.1, MORPH_LABEL.get(m, m).replace(" ", "\n"), transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=5.5, color=TEXT, linespacing=1.1)
    ref = expansion_reference()
    if ref is not None:
        ax.axhline(ref, color=TEXT_2, lw=0.8, ls="--", zorder=1)
        ax.annotate(f"stent at target diameter (before recoil): {ref:.1f} %", xy=(3 * len(morphs) - 1.6, ref),
                    xytext=(0, -2), textcoords="offset points", ha="right", va="top", fontsize=5.5, color=TEXT_2)
    ax.set_xticks([3 * g + dx for g in range(len(morphs)) for dx in (0, 1)])
    ax.set_xticklabels(["no", "yes"] * len(morphs), fontsize=5.5)
    ax.set_xlim(-0.6, 3 * len(morphs) - 1.4)
    ax.set_xlabel("IVL before stenting, by morphology", labelpad=19)
    ax.set_ylabel("Stent expansion (%)")
    ax.grid(axis="x", visible=False)
    if not n_pairs and not any(vals[k].notna().any() for k in vals):
        _not_available(ax)
    return f"b  Stent expansion without and after IVL (n = {n_pairs} pairs)"


def _onset_panel(fig, gs, df, morphs, arcs, ths, colors) -> tuple[list, str]:
    """Panel (c): onset load vs arc, one axes per morphology; returns the axes and the panel title."""
    from matplotlib.transforms import offset_copy

    t = onset_table(df)
    spacing = float(np.min(np.diff(arcs))) if len(arcs) > 1 else 1.0
    dodge = P.get(f"{CFG}.dodge_frac").value * spacing
    p_bal = P.get("loads.ivl_balloon_pressure_atm").value
    axes = []
    for g, m in enumerate(morphs):
        ax = fig.add_subplot(gs[0, g], sharey=axes[0] if axes else None)
        axes.append(ax)
        color = colors[m]
        head = offset_copy(ax.transData, fig=fig, x=0, y=9, units="points")
        for it, th in enumerate(ths):
            r = t[(t["morphology"] == m) & np.isclose(t["ca_thickness"], th)]
            x = r["ca_arc_deg"].to_numpy(float) + (it - (len(ths) - 1) / 2) * dodge
            y, ev = r["value"].to_numpy(float), r["event"].to_numpy(bool)
            mk = THICK_MARKER[it % len(THICK_MARKER)]
            ax.plot(x[ev], y[ev], marker=mk, ms=3.5, ls="none", color=color, mec=color, zorder=3)
            ax.plot(x[~ev], y[~ev], marker=mk, ms=3.5, ls="none", mfc="white", mec=color, mew=0.8, zorder=3)
            for xc, yc in zip(x[~ev], y[~ev]):
                ax.annotate("", xy=(xc, yc), xycoords=head, xytext=(xc, yc), textcoords="data",
                            arrowprops={"arrowstyle": "-|>", "color": color, "lw": 0.6, "shrinkA": 2.5,
                                        "shrinkB": 0, "mutation_scale": 4})
        ax.axhline(p_bal, color=TEXT_2, lw=0.7, ls="--", zorder=1)
        ax.set_xticks(arcs)
        ax.set_xticklabels([f"{a:g}" for a in arcs], fontsize=5.5)
        ax.set_xlim(arcs[0] - spacing / 2, arcs[-1] + spacing / 2)
        ax.set_title(MORPH_LABEL.get(m, m), fontsize=6.5, color=TEXT)
        ax.grid(axis="x", visible=False)
        if g:
            ax.tick_params(labelleft=False)
        if t.empty:
            _not_available(ax)
    axes[0].set_ylabel("Lumen pressure at damage onset (MPa)")
    axes[0].text(0.03, p_bal, "IVL balloon", transform=axes[0].get_yaxis_transform(), ha="left", va="bottom",
                 fontsize=5, color=TEXT_2)
    if not t.empty:
        lo, hi = min(t["value"].min(), p_bal), max(t["value"].max(), p_bal)
        pad = 0.08 * (hi - lo) if hi > lo else 0.05 * hi
        axes[0].set_ylim(lo - pad, hi + 2.2 * pad)
    n_c, n_ap = int((~t["event"]).sum()), int(t["after_peak"].sum())
    title = f"c  Damage-onset load under IVL (n = {len(t)}; {n_c} without onset)"
    if n_ap:
        title += f"; {n_ap} onset(s) after the peak drawn at the peak load"
    return axes, title


def make(name: str = "fig5_outcomes") -> str:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    apply_style()
    plt.rcParams["hatch.linewidth"] = 0.5
    fac = P.raw("design.core_factorial")
    morphs, arcs, ths = list(fac["morphology"]), [float(a) for a in fac["arc_deg"]], \
        [float(t) for t in fac["thickness_mm"]]
    protocols = list(P.raw(f"{CFG}.map_protocols")["value"])
    df = read_outcomes()
    set_name = P.raw(f"{CFG}.set")["value"]
    if "set" in df:
        df = df[df["set"] == set_name]
    df = df.reset_index(drop=True)
    colors = {m: SERIES[g] for g, m in enumerate(morphs)}

    fig = plt.figure(figsize=(7.1, 7.6))
    outer = fig.add_gridspec(2, 1, height_ratios=[3.35, 2.2], left=0.125, right=0.99, top=0.955, bottom=0.115,
                             hspace=0.42)
    gA = outer[0].subgridspec(len(protocols), len(morphs), wspace=0.12, hspace=0.12)
    gBC = outer[1].subgridspec(1, 2, width_ratios=[1.0, 1.55], wspace=0.22)
    gC = gBC[1].subgridspec(1, len(morphs), wspace=0.1)

    axA = _map_panel(fig, gA, df, morphs, arcs, ths, protocols)
    axB = fig.add_subplot(gBC[0])
    title_b = _expansion_panel(axB, df, morphs, colors)
    axC, title_c = _onset_panel(fig, gC, df, morphs, arcs, ths, colors)

    # panel titles (figure text so that they align above the blocks)
    ya = axA[0].get_position().y1 + 0.03
    fig.text(0.01, ya, f"a  Outcome maps of the core factorial (n = {len(df)} lesions with a record)",
             fontsize=7, color=TEXT, va="bottom")
    yb = axB.get_position().y1 + 0.035
    fig.text(0.01, yb, title_b, fontsize=7, color=TEXT, va="bottom")
    fig.text(axC[0].get_position().x0 - 0.035, yb, title_c, fontsize=7, color=TEXT, va="bottom")

    # legends: damage classes (a), stent status (b), onset encoding (c)
    col = _class_colors()
    hand_a = [Patch(fc=col[c], ec=GRID if c == "na" else "none", lw=0.4, hatch="//////" if c == "partial" else None,
                    label=CLASS_LABEL[c]) for c in CLASSES]
    hand_a.append(Line2D([], [], ls="none", label="numbers: IVL damage-onset load (MPa)"))
    y_leg = axA[-1].get_position().y0 - 0.052
    fig.legend(handles=hand_a, loc="upper center", bbox_to_anchor=(0.56, y_leg), ncol=5, frameon=False,
               fontsize=5.5, handlelength=1.4, columnspacing=1.0, borderaxespad=0.0)
    hand_b = [Line2D([], [], color=TEXT_2, marker="o", ms=3.5, ls="none", label="expansion run ok"),
              Line2D([], [], color=TEXT_2, marker="o", ms=3.5, mfc="white", ls="none",
                     label="pressure-limited ('limited')")]
    axB.legend(handles=hand_b, loc="upper left", bbox_to_anchor=(0.0, -0.33), ncol=2, frameon=False, fontsize=5.5,
               borderaxespad=0.0, handletextpad=0.3, columnspacing=0.8)
    hand_c = [Line2D([], [], color=TEXT_2, marker=THICK_MARKER[i % len(THICK_MARKER)], ms=3.5, ls="none",
                     label=f"{t:g} mm") for i, t in enumerate(ths)]
    hand_c += [Line2D([], [], color=TEXT_2, marker="o", ms=3.5, mfc="white", ls="none",
                      label="no onset up to the peak IVL load (↑)")]
    x_c = axC[0].get_position().x0
    fig.text(0.5 * (x_c + axC[-1].get_position().x1), axC[0].get_position().y0 - 0.03, "Calcium arc (°)",
             ha="center", va="top", fontsize=7, color=TEXT)
    fig.legend(handles=hand_c, loc="upper left", bbox_to_anchor=(x_c - 0.01, axC[0].get_position().y0 - 0.06),
               ncol=len(hand_c), frameon=False, fontsize=5.5, title="Calcium thickness", title_fontsize=5.5,
               alignment="left", borderaxespad=0.0, handletextpad=0.3, columnspacing=0.8)
    return str(save(fig, name))

"""Figure 4: calcium damage fields of one lesion per morphology (same arc and thickness).

Rows: the lesions of ``design.field_lesions``, labelled by their morphology in
cohort/cohort.csv. Columns (final state of each run, drawn in the deformed
configuration points + u; the IVL runs end unloaded):

(a) IVL with the phase field: nodal damage d on the calcium triangles
    (Gouraud shading);
(b) IVL with the erosion comparator: eroded calcium cells (d_cell = 1);
(c) stent deployed after IVL (continuation of run a): phase-field damage, the
    lumen outline after stenting and the lumen of the reference configuration.

Damage uses one sequential single-hue map (0 to 1) shared by all panels; the
other tissues are drawn in the TISSUE_RGB greys, lumen and outside white, and
the calcium outline in a thin dark line so that calcium is delimited without
relying on colour. Each panel is rotated so that the calcium centroid lies at
``figures.fig4_damage_fields.orientation_deg`` and every panel shows the same
window (calcium nodes of all panels plus ``zoom_margin_frac``), so one scale
holds for the whole figure. The annotation of each panel (through-thickness
fracture, damage-onset load) is read from results/fields/fields_summary.json.
A lesion or protocol whose fields are not on disk yet is drawn as an empty
panel marked 'not available'.

Reads results/fields/<lesion>_<protocol>.npz, results/fields/fields_summary.json
(src/drivers/stage_fields.py) and cohort/cohort.csv only.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb

from dtivl import params as P
from dtivl.drivers.common import COHORT_DIR
from dtivl.drivers.figstyle import SERIES, TEXT, TEXT_2, TISSUE_RGB, apply_style, save
from dtivl.drivers.stage_fields import FIELDS_DIR, field_lesions
from dtivl.drivers.stage_figures import MORPH_LABEL
from dtivl.geometry.labels import MATERIAL_NAMES, TAG_CALCIUM

#: columns: (protocol, title, damage source); 'nodal' = d (Gouraud), 'cell' = d_cell (flat)
COLUMNS = (("ivl", "IVL, phase field", "nodal"),
           ("ivl_erosion", "IVL, erosion comparator", "cell"),
           ("stent_post_ivl", "Stent after IVL", "nodal"))
CFG = "figures.fig4_damage_fields"


def damage_cmap() -> LinearSegmentedColormap:
    """Sequential single-hue map of the calcium series colour: light tint (d = 0) to dark shade (d = 1)."""
    c = np.array(to_rgb(SERIES[0]))
    return LinearSegmentedColormap.from_list("damage", [0.85 + 0.15 * c, c, 0.5 * c])


def _rot(theta: float) -> np.ndarray:
    return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])


def load_panel(lid: str, protocol: str) -> dict | None:
    """Fields of one run (stage_fields npz layout) or None when the file is not on disk yet."""
    path = FIELDS_DIR / f"{lid}_{protocol}.npz"
    if not path.exists():
        return None
    with np.load(path) as z:
        return {k: z[k] for k in ("points", "triangles", "cell_tag", "d", "d_cell", "u")}


def _calcium_rotation(f: dict) -> np.ndarray:
    """Rotation taking the area-weighted calcium centroid (reference configuration) to orientation_deg."""
    pts, tri = f["points"], f["triangles"][f["cell_tag"] == TAG_CALCIUM]
    if len(tri) == 0:
        return np.eye(2)
    a, b, c = pts[tri[:, 0]], pts[tri[:, 1]], pts[tri[:, 2]]
    area = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    cen = (area[:, None] * (a + b + c) / 3.0).sum(axis=0) / area.sum()
    target = math.radians(P.get(f"{CFG}.orientation_deg").value)
    return _rot(target - math.atan2(cen[1], cen[0]))


def _boundary_edges(tri: np.ndarray) -> np.ndarray:
    """Edges (k, 2) that belong to exactly one triangle of ``tri``."""
    e = np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1)
    u, cnt = np.unique(e, axis=0, return_counts=True)
    return u[cnt == 1]


def _lumen_edges(pts: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Boundary edges of the connected boundary loop nearest to the lumen centre (the lumen outline)."""
    be = _boundary_edges(tri)
    nodes = np.unique(be)
    idx = {n: i for i, n in enumerate(nodes)}
    parent = list(range(len(nodes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in be:
        parent[find(idx[a])] = find(idx[b])
    root = np.array([find(idx[n]) for n in nodes])
    inner = root[np.argmin(np.hypot(pts[nodes, 0], pts[nodes, 1]))]
    keep = set(nodes[root == inner].tolist())
    return be[[a in keep for a in be[:, 0]]]


def prepare(f: dict) -> dict:
    """Rotated reference and final (points + u) coordinates and the calcium / lumen edges of one run."""
    R = _calcium_rotation(f)
    ca = f["cell_tag"] == TAG_CALCIUM
    return {**f, "ref": f["points"] @ R.T, "xy": (f["points"] + f["u"]) @ R.T, "ca": ca,
            "ca_edges": _boundary_edges(f["triangles"][ca]), "lumen_edges": _lumen_edges(f["points"], f["triangles"])}


def window(panels: list[dict]) -> tuple[float, float, float, float]:
    """Common (x0, x1, y0, y1): bounding box of the calcium nodes of every panel plus the configured margin."""
    xy = [p["xy"][np.unique(p["triangles"][p["ca"]])] for p in panels if p is not None and p["ca"].any()]
    if not xy:
        return (-1.0, 1.0, -1.0, 1.0)
    xy = np.concatenate(xy)
    (x0, y0), (x1, y1) = xy.min(axis=0), xy.max(axis=0)
    m = P.get(f"{CFG}.zoom_margin_frac").value * max(x1 - x0, y1 - y0)
    return (x0 - m, x1 + m, y0 - m, y1 + m)


def nice_length(width: float) -> float:
    """Largest 1, 2 or 5 x 10^k not exceeding scale_bar_frac x ``width``."""
    target = P.get(f"{CFG}.scale_bar_frac").value * width
    k = math.floor(math.log10(target))
    return max(m * 10.0 ** k for m in (1, 2, 5) if m * 10.0 ** k <= target)


def outcome_text(rec: dict | None) -> str:
    """Panel annotation from a fields_summary.json entry (values formatted from the file)."""
    if not rec:
        return "outcome not recorded"
    lines = []
    if rec.get("status") != "ok":
        lines.append(f"run status: {rec.get('status')}")
    fr = rec.get("fracture")
    lines.append("through-thickness: " + ("yes" if fr is True else "no" if fr is False else "–"))
    on = rec.get("onset_load")
    lines.append(f"onset {on:#.3g} MPa" if on is not None else "no damage onset")
    if rec.get("cohort_record") and not all(rec.get(k, True) for k in ("fracture_matches", "onset_matches",
                                                                         "status_matches")):
        lines.append("differs from cohort record")
    return "\n".join(lines)


def _draw(ax, p: dict, source: str, cmap, norm, *, lumen_lines: bool) -> None:
    from matplotlib.collections import LineCollection, PolyCollection

    xy, tri, tag = p["xy"], p["triangles"], p["cell_tag"]
    soft = ~p["ca"]
    face = [TISSUE_RGB.get(MATERIAL_NAMES.get(int(t), ""), TISSUE_RGB["fibrous"]) for t in tag[soft]]
    ax.add_collection(PolyCollection(xy[tri[soft]], facecolors=face, edgecolors="face", linewidths=0.05,
                                     rasterized=True, zorder=1))
    if p["ca"].any():
        if source == "nodal":
            ax.tripcolor(xy[:, 0], xy[:, 1], tri[p["ca"]], p["d"], shading="gouraud", cmap=cmap, norm=norm,
                         rasterized=True, zorder=2)
        else:
            ax.add_collection(PolyCollection(xy[tri[p["ca"]]], facecolors=cmap(norm(p["d_cell"][p["ca"]])),
                                             edgecolors="face", linewidths=0.05, rasterized=True, zorder=2))
        ax.add_collection(LineCollection(xy[p["ca_edges"]], colors=TEXT_2, linewidths=0.3, zorder=3))
    if lumen_lines:
        ax.add_collection(LineCollection(p["ref"][p["lumen_edges"]], colors=TEXT_2, linewidths=0.6,
                                         linestyles=(0, (2.0, 1.5)), zorder=4))
        ax.add_collection(LineCollection(xy[p["lumen_edges"]], colors=TEXT, linewidths=0.8, zorder=4))


def _scale_bar(ax, win) -> None:
    x0, x1, y0, y1 = win
    L = nice_length(x1 - x0)
    xc, yb = 0.5 * (x0 + x1), y0 + 0.06 * (y1 - y0)
    ax.plot([xc - L / 2, xc + L / 2], [yb, yb], color=TEXT, lw=1.2, solid_capstyle="butt", zorder=6)
    ax.text(xc, yb + 0.025 * (y1 - y0), f"{L:g} mm", ha="center", va="bottom", fontsize=5.5, color=TEXT, zorder=6)


def make(name: str = "fig4_damage_fields") -> str:
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    apply_style()
    lids = field_lesions()
    spath = FIELDS_DIR / "fields_summary.json"
    summary = json.loads(spath.read_text()) if spath.exists() else {}
    coh = pd.read_csv(COHORT_DIR / "cohort.csv").set_index("lesion_id")
    panels = {}
    for lid in lids:
        for proto, _, _ in COLUMNS:
            f = load_panel(lid, proto)
            panels[lid, proto] = prepare(f) if f is not None else None
    win = window(list(panels.values()))
    x0, x1, y0, y1 = win

    n_rows, n_cols = len(lids), len(COLUMNS)
    left, right, top, bottom, gap = 0.62, 0.05, 0.25, 0.80, 0.06     # inches
    pw = (7.1 - left - right - gap * (n_cols - 1)) / n_cols
    ph = pw * (y1 - y0) / (x1 - x0)
    H = top + bottom + n_rows * ph + gap * (n_rows - 1)
    fig = plt.figure(figsize=(7.1, H))
    gs = fig.add_gridspec(n_rows, n_cols, left=left / 7.1, right=1 - right / 7.1, top=1 - top / H,
                          bottom=bottom / H, wspace=gap / pw, hspace=gap / ph)
    cmap, norm = damage_cmap(), Normalize(0.0, 1.0)
    for i, lid in enumerate(lids):
        for j, (proto, title, source) in enumerate(COLUMNS):
            ax = fig.add_subplot(gs[i, j])
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            ax.set_facecolor(TISSUE_RGB["outside"])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if i == 0:
                ax.set_title(f"{'abc'[j]}  {title}", loc="left", color=TEXT)
            if j == 0:
                morph = coh.loc[lid, "morphology"] if lid in coh.index else None
                ax.set_ylabel(f"{MORPH_LABEL.get(morph, morph or lid)}\n{lid}", fontsize=6, color=TEXT)
            p = panels[lid, proto]
            if p is None:
                ax.text(0.5, 0.5, "not available", ha="center", va="center", color=TEXT_2,
                        transform=ax.transAxes)
                continue
            _draw(ax, p, source, cmap, norm, lumen_lines=proto.startswith("stent"))
            _scale_bar(ax, win)
            ax.text(0.015, 0.97, outcome_text((summary.get(lid) or {}).get(proto)), transform=ax.transAxes,
                    ha="left", va="top", fontsize=5.5, color=TEXT, linespacing=1.15, zorder=7,
                    bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "none", "alpha": 0.85})

    # shared colour bar and legend
    cax = fig.add_axes([left / 7.1, 0.52 / H, 0.30, 0.08 / H])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    cax.grid(False)
    cb.set_ticks([0.0, 0.5, 1.0])
    cb.set_label("Calcium damage d (erosion: eroded cell = 1)", fontsize=6, color=TEXT, labelpad=2)
    cb.outline.set_linewidth(0.4)
    hand = [Patch(fc=TISSUE_RGB["fibrous"], ec="none", label="Fibrous intima"),
            Patch(fc=TISSUE_RGB["wall"], ec="none", label="Media + adventitia"),
            Line2D([], [], color=TEXT_2, lw=0.6, label="Calcium outline"),
            Line2D([], [], color=TEXT, lw=0.8, label="Lumen after stenting"),
            Line2D([], [], color=TEXT_2, lw=0.6, ls=(0, (2.0, 1.5)), label="Lumen, reference configuration")]
    fig.legend(handles=hand, loc="center left", bbox_to_anchor=(left / 7.1 + 0.34, 0.50 / H), ncol=3,
               frameon=False, fontsize=5.5, handlelength=2.0, columnspacing=1.2, borderaxespad=0.0)
    arcs = {coh.loc[lid, "ca_arc_deg"] for lid in lids if lid in coh.index}
    ths = {coh.loc[lid, "ca_thickness"] for lid in lids if lid in coh.index}
    note = "Final state of each run (IVL runs: after unloading)."
    if len(arcs) == 1 and len(ths) == 1:
        note = f"All rows: calcium arc {arcs.pop():g}°, thickness {ths.pop():g} mm. " + note
    fig.text(left / 7.1, 0.05 / H, note, fontsize=5.5, color=TEXT_2, va="bottom")
    return str(save(fig, name))

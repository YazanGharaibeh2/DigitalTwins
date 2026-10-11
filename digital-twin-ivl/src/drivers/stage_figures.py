"""Figures (each reads only cohort/ and results/ files)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch

from dtivl.drivers.common import COHORT_DIR, load_mask
from dtivl.drivers.figstyle import TEXT_2, TISSUE_RGB, apply_style, save
from dtivl.geometry.labels import LABEL_CALCIUM, LABEL_LIPID, LABEL_LUMEN

MORPH_ORDER = ("superficial_sheet", "deep_sheet", "nodule", "fragmented")
MORPH_LABEL = {"superficial_sheet": "Superficial sheet", "deep_sheet": "Deep sheet", "nodule": "Nodule",
               "fragmented": "Fragmented"}


def tissue_image(lesion_id: str) -> tuple[np.ndarray, tuple]:
    """RGB image of a lesion: lumen, fibrous intima, wall, calcium, lipid."""
    plm = load_mask(lesion_id)
    img, ext = plm.to_cartesian()
    n = img.shape[0]
    c = np.linspace(ext[0], ext[1], n)
    X, Y = np.meshgrid(c, c, indexing="xy")
    R, T = np.hypot(X, Y), np.mod(np.arctan2(Y, X), 2 * np.pi)
    it = np.mod(np.rint(T / (2 * np.pi) * plm.n_theta).astype(int), plm.n_theta)
    rgb = np.empty(img.shape + (3,))
    rgb[:] = to_rgb(TISSUE_RGB["outside"])
    tissue = img == 0
    inner = R < (plm.r_iel[it] if plm.r_iel is not None else plm.r_out[it])
    rgb[tissue & inner] = to_rgb(TISSUE_RGB["fibrous"])
    rgb[tissue & ~inner] = to_rgb(TISSUE_RGB["wall"])
    rgb[img == LABEL_LUMEN] = to_rgb(TISSUE_RGB["lumen"])
    rgb[img == LABEL_CALCIUM] = to_rgb(TISSUE_RGB["calcium"])
    rgb[img == LABEL_LIPID] = to_rgb(TISSUE_RGB["lipid"])
    return np.flipud(rgb), ext


def fig_cohort_gallery(set_name: str = "core", name: str = "fig2_cohort_gallery") -> str:
    """Core factorial gallery: rows = morphology, columns = arc x thickness."""
    import matplotlib.pyplot as plt

    apply_style()
    df = pd.read_csv(COHORT_DIR / "cohort.csv")
    df = df[df["set"] == set_name]
    arcs = sorted(df["ca_arc_deg"].unique())
    ths = sorted(df["ca_thickness"].unique())
    cols = [(a, t) for t in ths for a in arcs]
    fig, axes = plt.subplots(len(MORPH_ORDER), len(cols), figsize=(7.1, 3.9))
    for i, m in enumerate(MORPH_ORDER):
        for j, (a, t) in enumerate(cols):
            ax = axes[i, j]
            row = df[(df["morphology"] == m) & (df["ca_arc_deg"] == a) & (df["ca_thickness"] == t)]
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
            if row.empty:
                continue
            rgb, ext = tissue_image(row["lesion_id"].iloc[0])
            ax.imshow(rgb, extent=ext, interpolation="nearest")
            fs = int(row["fujino_score"].iloc[0])
            ax.text(0.02, 0.02, f"F{fs}", transform=ax.transAxes, fontsize=5.5, color=TEXT_2, va="bottom")
            if i == 0:
                ax.set_title(f"{a:g}°\n{t:g} mm", fontsize=6)
            if j == 0:
                ax.set_ylabel(MORPH_LABEL[m], fontsize=6.5)
    handles = [Patch(color=TISSUE_RGB[k], label=k.capitalize()) for k in ("calcium", "fibrous", "wall", "lumen")]
    handles[-1].set_edgecolor(TEXT_2)
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(wspace=0.04, hspace=0.06, bottom=0.07)
    out = save(fig, name)
    return str(out)

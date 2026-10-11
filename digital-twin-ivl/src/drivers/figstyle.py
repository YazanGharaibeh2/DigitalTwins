"""Shared figure style: colour-blind-safe palette (validated), fonts, export.

Categorical slots follow the reference data-viz palette (fixed order); the
calcium/lipid pair was validated with the palette checker (CVD dE 24.7,
normal-vision dE 33.6, contrast >= 3:1). Tissue context is drawn in neutral
greys so that colour carries only calcium and lipid.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dtivl import ROOT  # noqa: E402

FIG_DIR = ROOT / "figures"
DPI = 600                          # >= 300 dpi requirement
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#d9d8d4"
SURFACE = "#fcfcfb"
TISSUE_RGB = {                     # label-image colours
    "outside": "#ffffff",
    "lumen": "#fcfcfb",
    "fibrous": "#dedcd6",
    "wall": "#a9a7a0",
    "calcium": SERIES[0],
    "lipid": SERIES[1],
}


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
        "axes.edgecolor": TEXT_2, "axes.labelcolor": TEXT, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.5, "lines.linewidth": 1.5, "lines.markersize": 4, "savefig.dpi": DPI,
        "savefig.bbox": "tight", "figure.dpi": 150,
    })


def save(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / f"{name}.png"
    fig.savefig(out, dpi=DPI, facecolor="white")
    fig.savefig(FIG_DIR / f"{name}.pdf", facecolor="white")
    plt.close(fig)
    return out

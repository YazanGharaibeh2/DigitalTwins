"""Manuscript figures: one module per figure, each reading only cohort/, params/ and results/ files.

Every figure function takes no required argument, writes ``figures/<name>.png`` (600 dpi) and
``.pdf`` through :func:`dtivl.drivers.figstyle.save` and returns the PNG path. Numbers printed on a
figure (annotations, labels) are read from results files, never typed.
"""
from __future__ import annotations

import importlib

#: figure name -> "module:function" (module relative to this package, or absolute)
REGISTRY = {
    "fig1_design": "fig1_design:make",
    "fig2_cohort": "dtivl.drivers.stage_figures:fig_cohort_gallery",
    "fig3_verification": "fig3_verification:make",
    "fig4_damage_fields": "fig4_damage_fields:make",
    "fig5_outcomes": "fig5_outcomes:make",
    "fig6_prediction": "fig6_prediction:make",
    "fig7_sensitivity": "fig7_sensitivity:make",
}


def _resolve(target: str):
    mod, fn = target.split(":")
    mod = mod if mod.startswith("dtivl.") else f"{__name__}.{mod}"
    return getattr(importlib.import_module(mod), fn)


def make(name: str) -> str:
    return _resolve(REGISTRY[name])()


def make_all(names=None, *, skip_missing: bool = False) -> dict:
    """Build the figures of ``names`` (default all). ``skip_missing``: a figure whose module or input
    file does not exist yet is reported instead of raising."""
    out = {}
    for name in names or REGISTRY:
        try:
            out[name] = make(name)
        except (ModuleNotFoundError, FileNotFoundError) as exc:
            if not skip_missing:
                raise
            out[name] = f"skipped: {type(exc).__name__}: {exc}"
    return out

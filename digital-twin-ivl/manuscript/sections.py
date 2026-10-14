"""Manuscript text, one function per section; every number through ``ctx`` (see build_manuscript.py).

Draft status: skeleton. The sections are filled once the cohort, sensitivity
and metrics stages are complete.
"""
from __future__ import annotations

from build_manuscript import Ctx, Figure, Heading, Para, Table

TITLE = "In Silico Digital Twins of Calcified Coronary Lesions under Intravascular Lithotripsy and Stenting"


def front(ctx: Ctx) -> list:
    return [Heading(TITLE, 0)]


def methods_cohort(ctx: Ctx) -> list:
    core = ctx.data("results/cohort_summary.json")["n_per_set"]
    return [
        Heading("2. Materials and Methods", 1),
        Heading("2.1. Lesion Cohort", 2),
        Para(f"The core set contains {ctx.v('results/cohort_summary.json', 'n_per_set.core', fmt='d')} "
             f"two-dimensional lesions and the hold-out set "
             f"{ctx.v('results/cohort_summary.json', 'n_per_set.holdout', fmt='d')} lesions "
             f"({ctx.fig('fig2_cohort_gallery')}). Calcium fracture under IVL is reported in "
             f"{ctx.cite('hill2020disruptcad3', 'ali2019disruptcad2')}."),
        Figure("fig2_cohort_gallery", "Core factorial cohort."),
    ]


def document(ctx: Ctx) -> list:
    return front(ctx) + methods_cohort(ctx)

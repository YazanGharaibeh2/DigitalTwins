"""Manuscript assembly: one module per part, each a function ``blocks(ctx) -> list`` of blocks.

Every number goes through ``ctx`` (see build_manuscript.py). Parts whose module does not exist yet
are left out, so the manuscript builds at every stage of drafting.
"""
from __future__ import annotations

import importlib

from build_manuscript import Ctx, Heading

TITLE = "In Silico Digital Twins of Calcified Coronary Lesions under Intravascular Lithotripsy and Stenting"

#: manuscript order (MDPI research article)
PARTS = ("s0_front",          # title, authors, abstract, keywords
         "s1_introduction",
         "s2_methods",
         "s2_statistics",     # 2.10 Statistical Analysis
         "s2_software",       # 2.11 Software and Reproducibility
         "s3_results",
         "s4_discussion",
         "s5_conclusions",
         "s6_back")           # supplementary, contributions, funding, statements, abbreviations


def document(ctx: Ctx) -> list:
    out = []
    for name in PARTS:
        try:
            mod = importlib.import_module(name)
        except ModuleNotFoundError as exc:
            if exc.name != name:
                raise
            continue
        out += mod.blocks(ctx)
    if not out or not isinstance(out[0], Heading) or out[0].level != 0:
        out = [Heading(TITLE, 0)] + out
    return out

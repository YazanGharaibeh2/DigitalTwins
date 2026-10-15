"""Back matter in MDPI order. Items the authors must supply are marked as such (no invented content)."""
from __future__ import annotations

from build_manuscript import Ctx, Heading, Para

ABBREVIATIONS = (
    ("AT1", "phase-field fracture model with linear local dissipation (Ambrosio–Tortorelli-type functional)"),
    ("AUC", "area under the receiver operating characteristic curve"),
    ("CI", "confidence interval"),
    ("EEM", "external elastic membrane"),
    ("FE", "finite element"),
    ("GCI", "grid convergence index"),
    ("IEL", "internal elastic lamina"),
    ("IVL", "intravascular lithotripsy"),
    ("LOOCV", "leave-one-out cross-validation"),
    ("OCT", "optical coherence tomography"),
    ("ROC", "receiver operating characteristic"),
    ("SED", "strain energy density"),
)


def blocks(ctx: Ctx) -> list:
    from dtivl import params as P
    name = P.raw("design.composite_index_name")
    abbr = list(ABBREVIATIONS) + [(name["value"], name["expansion"])]
    abbr.sort(key=lambda a: a[0])
    return [
        Heading("Supplementary Materials", 2),
        Para("The source code, configuration, literature parameter table, numbers ledger of this manuscript and all "
             "results files are available in the study repository (see Data Availability Statement)."),
        Heading("Author Contributions", 2),
        Para("[To be completed by the authors (CRediT taxonomy).]"),
        Heading("Funding", 2),
        Para("[To be completed by the authors.]"),
        Heading("Institutional Review Board Statement", 2),
        Para("Not applicable. This *in silico* study used no human or animal data."),
        Heading("Informed Consent Statement", 2),
        Para("Not applicable."),
        Heading("Data Availability Statement", 2),
        Para("All code, inputs and results are available in the study repository [URL and archived release to be "
             "added]. A single script (run_all.py) regenerates every result, figure, table and this manuscript."),
        Heading("Acknowledgments", 2),
        Para("During the preparation of this work, the authors used Claude (Anthropic), an AI assistant, to write "
             "the simulation and analysis code, run the simulations and draft the manuscript text. The authors "
             "reviewed and edited the output and take full responsibility for the content of this publication."),
        Heading("Conflicts of Interest", 2),
        Para("[To be completed by the authors.]"),
        Heading("Abbreviations", 2),
        Para("The following abbreviations are used in this manuscript: "
             + "; ".join(f"{a}, {b}" for a, b in abbr) + "."),
    ]

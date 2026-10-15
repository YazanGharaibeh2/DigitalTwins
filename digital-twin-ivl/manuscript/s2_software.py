"""Section 2.11, Software and Reproducibility (versions read from environment.lock.txt and requirements.txt)."""
from __future__ import annotations

import re

from build_manuscript import ROOT, Ctx, Heading, Para

LOCK = "environment.lock.txt"
REQ = "requirements.txt"


def _lock_version(ctx: Ctx, package: str) -> str:
    """Version of a conda package from the explicit lock file (URL .../<package>-<version>-<build>.conda)."""
    for line in (ROOT / LOCK).read_text().splitlines():
        m = re.search(rf"/{re.escape(package)}-(\d[^-]*)-[^/]*\.(?:conda|tar\.bz2)", line)
        if m:
            return ctx.num(m.group(1), src=LOCK, key=package, fmt="")
    raise KeyError(f"{package} not in {LOCK}")


def _req_version(ctx: Ctx, package: str) -> str:
    for line in (ROOT / REQ).read_text().splitlines():
        m = re.match(rf"{re.escape(package)}==(\S+)", line.strip())
        if m:
            return ctx.num(m.group(1), src=REQ, key=package, fmt="")
    raise KeyError(f"{package} not in {REQ}")


def blocks(ctx: Ctx) -> list:
    c = ctx.cite
    return [
        Heading("2.11. Software and Reproducibility", 2),
        Para(
            f"The models were implemented in Python {_lock_version(ctx, 'python')} with DOLFINx "
            f"{_lock_version(ctx, 'fenics-dolfinx')} {c('dolfinx2023')} and PETSc {_lock_version(ctx, 'petsc')} "
            f"with the direct solver MUMPS {_lock_version(ctx, 'mumps-mpi')}. Meshes were generated with Gmsh "
            f"{_req_version(ctx, 'gmsh')} {c('geuzaine2009gmsh')}. The statistics used scikit-learn "
            f"{_req_version(ctx, 'scikit-learn')} and SciPy {_req_version(ctx, 'scipy')}, and the global "
            f"sensitivity analyses used SALib {_req_version(ctx, 'SALib')} {c('herman2017salib')}. "
            "A single script regenerates every result, figure, table and this manuscript from the configuration "
            "and the literature parameter table, and all random draws are seeded. Every number in the text is read, "
            "when the manuscript is built, from a results file, the parameter files, the source code (solver and "
            "design constants) or the environment files (software versions). The build lists each number with its "
            "source and fails if the text contains a number that was not read from a file."),
    ]

"""OCT calcium score of Fujino et al. (EuroIntervention 2018).

Each criterion awards its points when the measurement is *strictly greater*
than its threshold. Thresholds come from the literature table rows
``fujino_angle_threshold``, ``fujino_thickness_threshold`` and
``fujino_length_threshold`` (via :func:`dtivl.params.lit`); the points of each
criterion are parsed from the same rows' ``notes`` column ("Score: N
point(s) if ..."), so no literature number is typed here.

With ``length_mm=None`` (2D cross-sections) the length criterion is not
scored: the result is the *cross-sectional score, max 3*.
"""
from __future__ import annotations

import functools
import math
import re
from dataclasses import dataclass

from dtivl import params

_ROWS = {"angle": "fujino_angle_threshold",
         "thickness": "fujino_thickness_threshold",
         "length": "fujino_length_threshold"}
_POINTS_RE = re.compile(r"Score:\s*(\d+)\s+points?\b", re.IGNORECASE)


@dataclass(frozen=True)
class FujinoCriterion:
    """One criterion: award ``points`` if the measurement > ``threshold``."""

    name: str
    threshold: float
    points: int
    provenance: str


@functools.lru_cache(maxsize=None)
def fujino_criteria() -> dict[str, FujinoCriterion]:
    """The three criteria (angle in deg, thickness and length in mm)."""
    table = params.load_literature()
    out = {}
    for key, row in _ROWS.items():
        q = params.lit(row)
        m = _POINTS_RE.search(table[row].notes)
        if m is None:
            raise ValueError(f"cannot parse the Fujino points from the notes of {row!r}")
        out[key] = FujinoCriterion(key, q.value, int(m.group(1)), q.provenance)
    return out


def fujino_max_score(three_d: bool) -> int:
    """Maximum attainable score (3 cross-sectional, 4 with length)."""
    c = fujino_criteria()
    return sum(v.points for k, v in c.items() if three_d or k != "length")


def fujino_score(max_angle_deg: float, max_thickness_mm: float, length_mm: float | None = None) -> int:
    """Fujino OCT calcium score; ``length_mm=None`` -> cross-sectional score (max 3).

    NaN measurements (e.g. thickness of a calcium-free section) score 0.
    """
    c = fujino_criteria()

    def above(x, crit):
        return x is not None and not math.isnan(float(x)) and float(x) > crit.threshold

    score = 0
    if above(max_angle_deg, c["angle"]):
        score += c["angle"].points
    if above(max_thickness_mm, c["thickness"]):
        score += c["thickness"].points
    if length_mm is not None and above(length_mm, c["length"]):
        score += c["length"].points
    return score

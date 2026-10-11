"""Lesion descriptors computed only from a polar label map.

Real IVOCT segmentations and synthetic lesions get identical treatment: the
only inputs are ``plm.labels``, ``plm.dr``, ``plm.r_iel`` (optional) and the
angular sampling. Conventions (``dtheta = 360/n_theta`` deg):

* **Arc**: an A-line "contains calcium" if any of its samples is calcium;
  ``ca_arc_total_deg`` = number of such A-lines x dtheta;
  ``ca_arc_max_contig_deg`` = longest circular run of such A-lines x dtheta
  (wrap-around aware; 360 when every A-line has calcium).
* **Thickness** per A-line = number of calcium samples x dr, over A-lines
  with calcium (max / min / mean).
* **Lumen boundary** per A-line: between the last sample of the initial lumen
  run and the next sample, i.e. at ``r = j_L*dr`` where ``j_L`` is the first
  non-lumen sample. **Depth** = ``(j_C - j_L) * dr`` with ``j_C`` the first
  calcium sample (distance from the lumen boundary to the leading edge of
  the first calcium sample cell); min and mean over A-lines with calcium.
  ``cap_thickness_min`` is the same quantity for lipid (NaN without lipid).
* **Areas** use polar cell areas ``r_j * dr * dtheta`` (exact annular cells).
* **n_deposits**: connected components of calcium samples with
  8-connectivity in the (theta, r) grid, merged across the theta = 0/360 seam.
* ``plaque_ecc_index`` = (t_max - t_min)/t_max of the intima thickness
  ``r_iel - r_lumen_boundary`` per A-line (NaN if ``r_iel`` is None).
* Fujino score with ``ca_arc_total_deg``, ``ca_thickness_max`` and
  ``length_mm`` (None -> cross-sectional score, max 3).
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from .fujino import fujino_max_score, fujino_score
from .labels import LABEL_CALCIUM, LABEL_LIPID, LABEL_LUMEN
from .polar import PolarLabelMap

#: 8-connectivity structuring element for connected components
_CONN8 = np.ones((3, 3), dtype=bool)


def _max_circular_run(mask: np.ndarray) -> int:
    """Length of the longest run of True in a circular 1D boolean array."""
    n = mask.size
    if n == 0 or not mask.any():
        return 0
    if mask.all():
        return n
    # rotate so that the array starts right after a False entry
    k = int(np.argmin(mask))  # index of a False
    m = np.roll(mask, -k - 1)
    best = run = 0
    for v in m:
        run = run + 1 if v else 0
        best = max(best, run)
    return best


def count_components_wrapped(mask: np.ndarray) -> int:
    """Connected components (8-connectivity) of a polar mask, periodic in rows."""
    lab, n = ndimage.label(mask, structure=_CONN8)
    if n == 0:
        return 0
    parent = list(range(n + 1))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    first, last = lab[0], lab[-1]
    n_r = mask.shape[1]
    for j in range(n_r):
        a = first[j]
        if a == 0:
            continue
        for jj in range(max(0, j - 1), min(n_r, j + 2)):
            b = last[jj]
            if b:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb
    return len({find(i) for i in range(1, n + 1)})


def lumen_boundary_index(labels: np.ndarray) -> np.ndarray:
    """Per A-line index of the first non-lumen sample (``n_r`` if none)."""
    not_lumen = labels != LABEL_LUMEN
    has = not_lumen.any(axis=1)
    return np.where(has, np.argmax(not_lumen, axis=1), labels.shape[1])


def _first_index(mask: np.ndarray) -> np.ndarray:
    """Per row index of the first True (-1 when none)."""
    has = mask.any(axis=1)
    return np.where(has, np.argmax(mask, axis=1), -1)


def descriptors(plm: PolarLabelMap, length_mm: float | None = None) -> dict:
    """Morphological descriptors and Fujino score of a polar label map."""
    lab = plm.labels
    dr = float(plm.dr)
    n_theta = plm.n_theta
    dtheta_deg = 360.0 / n_theta
    dtheta = 2.0 * math.pi / n_theta
    r = plm.r
    cell_area = r * dr * dtheta  # per column

    ca = lab == LABEL_CALCIUM
    li = lab == LABEL_LIPID
    lu = lab == LABEL_LUMEN
    j_lb = lumen_boundary_index(lab)
    r_lb = j_lb * dr

    ca_line = ca.any(axis=1)
    n_ca_lines = int(ca_line.sum())
    out: dict = {}
    out["ca_arc_total_deg"] = n_ca_lines * dtheta_deg
    out["ca_arc_max_contig_deg"] = _max_circular_run(ca_line) * dtheta_deg

    if n_ca_lines:
        th = ca.sum(axis=1)[ca_line] * dr
        j_c = _first_index(ca)[ca_line]
        depth = (j_c - j_lb[ca_line]) * dr
        out.update(ca_thickness_max=float(th.max()), ca_thickness_min=float(th.min()),
                   ca_thickness_mean=float(th.mean()),
                   ca_depth_min=float(depth.min()), ca_depth_mean=float(depth.mean()))
    else:
        out.update(ca_thickness_max=float("nan"), ca_thickness_min=float("nan"),
                   ca_thickness_mean=float("nan"), ca_depth_min=float("nan"),
                   ca_depth_mean=float("nan"))

    out["ca_area"] = float((ca * cell_area[None, :]).sum())
    out["lumen_area"] = float((lu * cell_area[None, :]).sum())
    out["n_deposits"] = count_components_wrapped(ca)

    li_line = li.any(axis=1)
    out["lipid_arc_deg"] = int(li_line.sum()) * dtheta_deg
    if li_line.any():
        j_l = _first_index(li)[li_line]
        out["cap_thickness_min"] = float(((j_l - j_lb[li_line]) * dr).min())
    else:
        out["cap_thickness_min"] = float("nan")

    if plm.r_iel is not None:
        t_int = np.asarray(plm.r_iel) - r_lb
        t_max = float(t_int.max())
        out["plaque_ecc_index"] = (t_max - float(t_int.min())) / t_max if t_max > 0 else float("nan")
    else:
        out["plaque_ecc_index"] = float("nan")

    out["fujino_score"] = fujino_score(out["ca_arc_total_deg"], out["ca_thickness_max"], length_mm)
    out["fujino_max"] = fujino_max_score(length_mm is not None)
    return out

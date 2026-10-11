"""Stage 1: build the in silico cohort (masks, previews, cohort.csv)."""
from __future__ import annotations

import math

from dtivl import params as P
from dtivl.drivers.common import COHORT_DIR, _to_jsonable, write_json
from dtivl.geometry import cohort


def all_specs() -> dict[str, list]:
    """Every lesion set used in the study, keyed by set name."""
    return {
        "core": cohort.core_specs(),
        "holdout": cohort.holdout_specs(),
        "pilot": cohort.pilot_specs(),
        "example3d": [cohort.example3d_spec()],
    }


def run() -> dict:
    sets = all_specs()
    specs = [s for v in sets.values() for s in v]
    for s in specs:  # numpy scalars in notes are not JSON-serialisable in cohort.csv
        s.notes = _to_jsonable(s.notes)
    df = cohort.build_cohort(specs, COHORT_DIR)
    df["set"] = [s.set_name for s in specs]
    df.to_csv(COHORT_DIR / "cohort.csv", index=False)

    # Plausibility check of the constructed reference geometry (not an input):
    # implied plaque burden at the lesion vs CTA plaque burden of calcified plaques.
    r_l = math.sqrt(P.get("geometry.lumen_area_mm2").value / math.pi)
    r_ref = P.get("geometry.reference_diameter_mm").value / 2
    t_media = P.lit("media_thickness_main_branches").value
    r_eem = r_ref + t_media
    pb = 100.0 * (1.0 - (r_l / r_eem) ** 2)
    pb_lo = P.lit("plaque_burden_calcified_plaque_CTA").value
    pb_hi = P.lit("plaque_burden_calcified_nodule_CTA").value
    summary = {
        "n_per_set": {k: len(v) for k, v in sets.items()},
        "n_total": len(specs),
        "core_geometry": {
            "r_lumen_mm": r_l, "plaque_thickness_mm": P.get("geometry.plaque_thickness_mm").value,
            "wall_thickness_mm": P.get("geometry.wall_thickness_mm").value,
            "depth_superficial_mm": P.get("geometry.depth_superficial_mm").value,
            "depth_deep_mm": P.get("geometry.depth_deep_mm").value,
            "r_eem_mm": r_eem,
        },
        "plaque_burden_check": {
            "implied_plaque_burden_pct": pb,
            "cta_calcified_plaque_pct": pb_lo,
            "cta_calcified_nodule_pct": pb_hi,
            "within_range": bool(pb_lo <= pb <= pb_hi),
        },
        "fujino_score_counts": {set_name: df[df["set"] == set_name]["fujino_score"].value_counts().sort_index().to_dict()
                                for set_name in sets},
    }
    write_json("cohort_summary", summary)
    return summary

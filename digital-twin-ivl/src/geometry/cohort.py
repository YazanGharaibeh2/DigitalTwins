"""Synthetic lesion cohorts: core factorial, pilot, 3D example and hold-out.

All geometry is read from ``params/study_config.yaml`` (``design.*`` and
``geometry.*``) through :mod:`dtivl.params`; the lumen is a circle whose area
is ``geometry.lumen_area_mm2``. Config entries still marked PENDING are
refused unless the caller passes ``allow_pending=True``; the keys that were
PENDING are then recorded in ``spec.notes['pending_inputs']``.

Fixed choices (documented here, not in the config):

* The calcium arc is centred on 0 deg (+x) and the plaque is concentric
  (``plaque_offset = 0``) in the core, pilot and 3D-example sets.
* Depth: ``geometry.depth_deep_mm`` for deep sheets, otherwise
  ``geometry.depth_superficial_mm`` (superficial sheets, nodules and
  fragmented deposits). ``nodule_height`` is non-zero only for nodules.
* Lesion ids: ``<P>-A<arc>-T<thickness>[-L<length>]-<MORPH>`` with prefix
  ``C`` (core), ``P`` (pilot), ``X3D`` (3D example), and ``H-<nnn>`` for the
  hold-out set; MORPH in {SUP, DEEP, NOD, FRAG}.
* Hold-out: Latin hypercube (``scipy.stats.qmc.LatinHypercube``) seeded with
  ``design.seed + HOLDOUT_SEED_OFFSET``; see :func:`holdout_specs` for the
  accepted range formats.
* ``cohort.csv``: one row per lesion with every spec field, every descriptor
  (descriptor names win; a spec field with the same name - only
  ``lipid_arc_deg`` - is written as ``spec_<name>``), the Fujino score and
  the relative paths of the mask and preview files.
"""
from __future__ import annotations

import dataclasses
import itertools
import json
import math
from pathlib import Path

import numpy as np

from dtivl import params

from .descriptors import descriptors
from .labels import LABEL_CALCIUM, LABEL_LIPID, LABEL_LUMEN, LABEL_OTHER
from .polar import PolarLabelMap
from .rasterize import rasterize_polar
from .spec import MORPHOLOGIES, LesionSpec

#: seed offset of the hold-out Latin hypercube (added to design.seed)
HOLDOUT_SEED_OFFSET = 1

MORPH_CODE = {"superficial_sheet": "SUP", "deep_sheet": "DEEP", "nodule": "NOD", "fragmented": "FRAG"}

#: config keys of the fixed reference geometry -> internal names
_GEOMETRY_KEYS = {
    "lumen_area": "geometry.lumen_area_mm2",
    "plaque_thickness": "geometry.plaque_thickness_mm",
    "wall_thickness": "geometry.wall_thickness_mm",
    "depth_superficial": "geometry.depth_superficial_mm",
    "depth_deep": "geometry.depth_deep_mm",
    "nodule_height": "geometry.nodule_height_mm",
    "nodule_base_arc_deg": "geometry.nodule_base_arc_deg",
    "fragment_pitch_deg": "geometry.fragment_pitch_deg",
    "fragment_fill": "geometry.fragment_fill",
    "margin_3d": "geometry.margin_3d_mm",
}

#: preview colours (RGB) per label; wall = tissue outside the IEL
PREVIEW_COLOURS = {
    "outside": (255, 255, 255),
    LABEL_LUMEN: (20, 20, 20),
    LABEL_OTHER: (225, 170, 160),   # fibrous plaque (inside IEL)
    "wall": (180, 90, 90),
    LABEL_CALCIUM: (245, 245, 200),
    LABEL_LIPID: (240, 200, 60),
}


# ------------------------------------------------------------ fixed geometry
def fixed_geometry(*, allow_pending: bool = False) -> dict:
    """Reference geometry of the core set (mm / deg) plus pending bookkeeping.

    Returns a dict with ``r_lumen`` (from the lumen area, circle), the values
    of ``_GEOMETRY_KEYS`` and ``pending_inputs`` (config keys that are PENDING).
    """
    out = {name: params.get(key, allow_pending=allow_pending).value
           for name, key in _GEOMETRY_KEYS.items()}
    out["r_lumen"] = math.sqrt(out["lumen_area"] / math.pi)
    out["pending_inputs"] = (params.check_ready(list(_GEOMETRY_KEYS.values()))
                             if allow_pending else [])
    return out


def _depth_for(morphology: str, geo: dict) -> float:
    return geo["depth_deep"] if morphology == "deep_sheet" else geo["depth_superficial"]


def _fmt(x: float) -> str:
    return f"{float(x):g}"


def make_spec(lesion_id: str, morphology: str, arc_deg: float, thickness_mm: float, geo: dict, *,
              set_name: str, length_mm: float | None = None, **overrides) -> LesionSpec:
    """LesionSpec on the reference geometry ``geo`` (see :func:`fixed_geometry`).

    ``overrides`` replace any LesionSpec field (e.g. ``ca_depth``).
    """
    if morphology not in MORPHOLOGIES:
        raise ValueError(f"unknown morphology {morphology!r}")
    fields = dict(
        lesion_id=lesion_id, morphology=morphology, ca_arc_deg=float(arc_deg),
        ca_thickness=float(thickness_mm), ca_depth=_depth_for(morphology, geo),
        r_lumen=geo["r_lumen"], plaque_thickness=geo["plaque_thickness"],
        wall_thickness=geo["wall_thickness"],
        ca_length=None if length_mm is None else float(length_mm),
        margin_3d=geo["margin_3d"],
        nodule_height=geo["nodule_height"] if morphology == "nodule" else 0.0,
        nodule_base_arc_deg=geo["nodule_base_arc_deg"],
        fragment_pitch_deg=geo["fragment_pitch_deg"], fragment_fill=geo["fragment_fill"],
        set_name=set_name,
        notes={"pending_inputs": list(geo.get("pending_inputs", []))},
    )
    fields.update(overrides)
    spec = LesionSpec(**fields)
    spec.validate()
    return spec


def _lesion_id(prefix: str, arc, t, morph, length=None) -> str:
    parts = [prefix, f"A{_fmt(arc)}", f"T{_fmt(t)}"]
    if length is not None:
        parts.append(f"L{_fmt(length)}")
    parts.append(MORPH_CODE[morph])
    return "-".join(parts)


# ------------------------------------------------------------------- sets
def core_specs(*, allow_pending: bool = False) -> list[LesionSpec]:
    """Full factorial ``design.core_factorial`` (arc x thickness x morphology)."""
    geo = fixed_geometry(allow_pending=allow_pending)
    f = params.raw("design.core_factorial")
    return [make_spec(_lesion_id("C", a, t, m), m, a, t, geo, set_name="core")
            for a, t, m in itertools.product(f["arc_deg"], f["thickness_mm"], f["morphology"])]


def pilot_specs(*, allow_pending: bool = False) -> list[LesionSpec]:
    """The 2D pilot lesions of ``design.pilot_2d``."""
    geo = fixed_geometry(allow_pending=allow_pending)
    return [make_spec(_lesion_id("P", e["arc_deg"], e["thickness_mm"], e["morphology"]),
                      e["morphology"], e["arc_deg"], e["thickness_mm"], geo, set_name="pilot")
            for e in params.raw("design.pilot_2d")]


def example3d_spec(*, allow_pending: bool = False) -> LesionSpec:
    """The 3D Fujino-score-4 example of ``design.fujino4_example_3d``."""
    geo = fixed_geometry(allow_pending=allow_pending)
    e = params.raw("design.fujino4_example_3d")
    return make_spec(_lesion_id("X3D", e["arc_deg"], e["thickness_mm"], e["morphology"], e["length_mm"]),
                     e["morphology"], e["arc_deg"], e["thickness_mm"], geo,
                     set_name="example3d", length_mm=e["length_mm"])


# ----------------------------------------------------------------- hold-out
_SPEC_FIELDS = {f.name for f in dataclasses.fields(LesionSpec)}
_HOLDOUT_ALIASES = {"arc_deg": "ca_arc_deg", "thickness_mm": "ca_thickness", "depth_mm": "ca_depth",
                    "length_mm": "ca_length", "lumen_area_mm2": "lumen_area"}


def _holdout_field(key: str) -> str:
    """Map a ``design.holdout_lhs.ranges`` key to a LesionSpec field name."""
    if key in _HOLDOUT_ALIASES:
        return _HOLDOUT_ALIASES[key]
    for cand in (key, key.removesuffix("_mm"), "ca_" + key.removesuffix("_mm")):
        if cand in _SPEC_FIELDS and cand not in ("lesion_id", "set_name", "notes"):
            return cand
    raise ValueError(f"design.holdout_lhs.ranges: unknown key {key!r} (expected a LesionSpec field, "
                     f"optionally with an '_mm' suffix, or one of {sorted(_HOLDOUT_ALIASES)})")


def _parse_range(key: str, node):
    """Return ('cont', lo, hi) or ('cat', choices) for one ranges entry."""
    if isinstance(node, dict):
        if "choices" in node:
            return ("cat", list(node["choices"]))
        lo = node.get("min", node.get("lower"))
        hi = node.get("max", node.get("upper"))
        if lo is None or hi is None:
            raise ValueError(f"design.holdout_lhs.ranges.{key}: need min/max (or lower/upper) or choices")
        node = [lo, hi]
    if isinstance(node, (list, tuple)):
        if all(isinstance(v, str) for v in node):
            return ("cat", list(node))
        if len(node) == 2 and all(isinstance(v, (int, float)) for v in node):
            lo, hi = float(node[0]), float(node[1])
            if hi < lo:
                raise ValueError(f"design.holdout_lhs.ranges.{key}: max < min")
            return ("cont", lo, hi)
    raise ValueError(f"design.holdout_lhs.ranges.{key}: unsupported range {node!r}")


def holdout_specs(n: int | None = None, seed: int | None = None, *, allow_pending: bool = False,
                  on_invalid: str = "raise") -> list[LesionSpec]:
    """Latin-hypercube hold-out lesions over ``design.holdout_lhs.ranges``.

    Accepted range formats per key: ``[lo, hi]``, ``{min: lo, max: hi}`` (or
    ``lower``/``upper``) for continuous fields, a list of strings or
    ``{choices: [...]}`` for categorical ones (e.g. ``morphology``). Keys are
    LesionSpec fields, optionally with an ``_mm`` suffix, or the aliases
    ``arc_deg``, ``thickness_mm``, ``depth_mm``, ``length_mm`` and
    ``lumen_area_mm2`` (sets ``r_lumen``). ``arc_deg`` and ``thickness_mm``
    are required. Fields not listed keep the core
    reference geometry; morphology defaults to the core-factorial list;
    depth defaults per morphology. ``n`` defaults to ``design.holdout_lhs.n``
    and ``seed`` to ``design.seed + HOLDOUT_SEED_OFFSET``.

    ``on_invalid='raise'`` (default) raises on a geometrically invalid sample;
    ``'skip'`` drops it (the result then has fewer than ``n`` lesions).
    """
    from scipy.stats import qmc

    from .build import lesion_polygons

    cfg = params.raw("design.holdout_lhs") or {}
    ranges = cfg.get("ranges")
    if not ranges:
        raise KeyError("design.holdout_lhs.ranges is missing from params/study_config.yaml: add the "
                       "hold-out parameter ranges (e.g. arc_deg: [lo, hi], thickness_mm: [lo, hi]) "
                       "before generating the hold-out set")
    n = int(n if n is not None else cfg["n"])
    seed = int(seed if seed is not None else params.raw("design.seed") + HOLDOUT_SEED_OFFSET)
    if on_invalid not in ("raise", "skip"):
        raise ValueError("on_invalid must be 'raise' or 'skip'")

    dims = {_holdout_field(k): _parse_range(k, v) for k, v in ranges.items()}
    missing = [f for f in ("ca_arc_deg", "ca_thickness") if f not in dims]
    if missing:
        raise KeyError(f"design.holdout_lhs.ranges must define {missing} (keys arc_deg / thickness_mm)")
    if "morphology" not in dims:
        dims["morphology"] = ("cat", list(params.raw("design.core_factorial.morphology")))
    names = list(dims)
    u = qmc.LatinHypercube(d=len(names), rng=np.random.default_rng(seed)).random(n)

    geo = fixed_geometry(allow_pending=allow_pending)
    specs = []
    for i in range(n):
        vals = {}
        for k, name in enumerate(names):
            d = dims[name]
            if d[0] == "cat":
                vals[name] = d[1][min(int(u[i, k] * len(d[1])), len(d[1]) - 1)]
            else:
                vals[name] = d[1] + u[i, k] * (d[2] - d[1])
        morph = vals.pop("morphology")
        arc = vals.pop("ca_arc_deg")
        t = vals.pop("ca_thickness")
        if "lumen_area" in vals:
            vals["r_lumen"] = math.sqrt(vals.pop("lumen_area") / math.pi)
        if morph != "nodule":
            vals["nodule_height"] = 0.0
        lid = f"H-{i + 1:03d}"
        try:
            spec = make_spec(lid, morph, arc, t, geo, set_name="holdout",
                             length_mm=vals.pop("ca_length", None), **vals)
            lesion_polygons(spec)
        except ValueError as exc:
            if on_invalid == "raise":
                raise ValueError(f"hold-out sample {lid} is invalid ({exc}); narrow "
                                 f"design.holdout_lhs.ranges or pass on_invalid='skip'") from exc
            continue
        spec.notes["lhs_seed"] = seed
        specs.append(spec)
    return specs


# ------------------------------------------------------------------ output
def label_preview_rgb(plm: PolarLabelMap, pixel: float | None = None) -> np.ndarray:
    """Cartesian RGB preview (rows top = +y) of a label map; wall shaded via r_iel."""
    img, (xmin, xmax, ymin, ymax) = plm.to_cartesian(pixel)
    n = img.shape[0]
    pix = (xmax - xmin) / n
    c = (np.arange(n) + 0.5) * pix + xmin
    X, Y = np.meshgrid(c, c, indexing="xy")
    rgb = np.zeros(img.shape + (3,), dtype=np.uint8)
    for lab in (LABEL_OTHER, LABEL_LUMEN, LABEL_CALCIUM, LABEL_LIPID):
        rgb[img == lab] = PREVIEW_COLOURS[lab]
    if plm.r_iel is not None:
        T = np.mod(np.arctan2(Y, X), 2 * np.pi)
        it = np.mod(np.rint(T / (2 * np.pi) * plm.n_theta).astype(int), plm.n_theta)
        wall = (img == LABEL_OTHER) & (np.hypot(X, Y) >= plm.r_iel[it])
        rgb[wall] = PREVIEW_COLOURS["wall"]
    rgb[img == 255] = PREVIEW_COLOURS["outside"]
    return rgb[::-1]


def _spec_row(spec: LesionSpec, desc_keys) -> dict:
    row = {}
    for k, v in spec.to_dict().items():
        if isinstance(v, (dict, list)):
            v = json.dumps(v, sort_keys=True)
        row[f"spec_{k}" if k in desc_keys else k] = v
    return row


def build_cohort(specs: list[LesionSpec], out_dir: str | Path, *, n_theta: int | None = None,
                 dr: float | None = None, previews: bool = True):
    """Rasterise ``specs`` and write masks, previews and ``cohort.csv``.

    Writes ``out_dir/masks/<id>.npz`` (:meth:`PolarLabelMap.save`),
    ``out_dir/previews/<id>.png`` and ``out_dir/cohort.csv``; returns the
    cohort table as a pandas DataFrame (one row per lesion).
    """
    import pandas as pd
    from PIL import Image

    ids = [s.lesion_id for s in specs]
    if len(set(ids)) != len(ids):
        raise ValueError("lesion ids must be unique")
    out = Path(out_dir)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    if previews:
        (out / "previews").mkdir(parents=True, exist_ok=True)

    rows = []
    for spec in specs:
        plm = rasterize_polar(spec, n_theta=n_theta, dr=dr)
        mask_rel = Path("masks") / f"{spec.lesion_id}.npz"
        plm.save(out / mask_rel)
        d = descriptors(plm, length_mm=spec.ca_length)
        row = _spec_row(spec, d.keys())
        row.update(d)
        row["mask_path"] = str(mask_rel)
        if previews:
            prev_rel = Path("previews") / f"{spec.lesion_id}.png"
            Image.fromarray(label_preview_rgb(plm)).save(out / prev_rel)
            row["preview_path"] = str(prev_rel)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / "cohort.csv", index=False)
    return df

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
  ``design.seed + HOLDOUT_SEED_OFFSET`` over ``design.holdout_lhs.ranges``;
  see :func:`holdout_specs` for the accepted keys, range formats and the
  rules (balanced morphology cycle, lipid fraction, depth rule, eccentricity
  via IEL offset, lumen area at fixed IEL radius).
* ``cohort.csv``: one row per lesion with every spec field, every descriptor
  (descriptor names win; a spec field with the same name - only
  ``lipid_arc_deg`` - is written as ``spec_<name>``), the Fujino score and
  the relative paths of the mask and preview files.
"""
from __future__ import annotations

import dataclasses
import itertools
import json
import logging
import math
import re
from pathlib import Path

import numpy as np

from dtivl import params

from .descriptors import descriptors
from .labels import LABEL_CALCIUM, LABEL_LIPID, LABEL_LUMEN, LABEL_OTHER
from .polar import PolarLabelMap
from .rasterize import rasterize_polar, ray_circle_radius
from .spec import MORPHOLOGIES, LesionSpec

_log = logging.getLogger(__name__)

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
#: ranges keys -> LesionSpec fields (or internal names for special keys)
_HOLDOUT_ALIASES = {"arc_deg": "ca_arc_deg", "thickness_mm": "ca_thickness", "depth_mm": "ca_depth",
                    "length_mm": "ca_length", "lumen_area_mm2": "lumen_area",
                    "cap_thickness_mm": "lipid_cap_thickness", "ecc_index": "ecc_index"}
#: ranges keys handled as rules rather than sampled fields
_HOLDOUT_RULE_KEYS = ("morphology", "depth_rule", "lipid_fraction")
_NUM_RE = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"


def _holdout_field(key: str) -> str:
    """Map a ``design.holdout_lhs.ranges`` key to a LesionSpec field name."""
    if key in _HOLDOUT_ALIASES:
        return _HOLDOUT_ALIASES[key]
    for cand in (key, key.removesuffix("_mm"), "ca_" + key.removesuffix("_mm")):
        if cand in _SPEC_FIELDS and cand not in ("lesion_id", "set_name", "notes", "morphology"):
            return cand
    raise ValueError(f"design.holdout_lhs.ranges: unknown key {key!r} (expected a LesionSpec field, "
                     f"optionally with an '_mm' suffix, one of {sorted(_HOLDOUT_ALIASES)} or "
                     f"{list(_HOLDOUT_RULE_KEYS)})")


def _number(x, key: str, allow_pending: bool) -> float:
    """A range bound: plain number or a resolvable config entry (value/lit/expr)."""
    if isinstance(x, bool):
        raise ValueError(f"design.holdout_lhs.ranges.{key}: boolean is not a number")
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, dict):
        return params.resolve(x, allow_pending=allow_pending).value
    raise ValueError(f"design.holdout_lhs.ranges.{key}: cannot read a number from {x!r}")


def _parse_range(key: str, node, allow_pending: bool = False):
    """Return ('cont', lo, hi), ('const', v) or ('cat', choices) for one ranges entry.

    Accepted: ``[lo, hi]`` (each a number or a ``value``/``lit``/``expr``
    entry), ``{min|lower: lo, max|upper: hi}``, a single number or resolvable
    entry (constant), a list of strings or ``{choices: [...]}`` (categorical).
    """
    if isinstance(node, dict):
        if "choices" in node:
            return ("cat", list(node["choices"]))
        if any(k in node for k in ("value", "lit", "expr")):
            return ("const", _number(node, key, allow_pending))
        lo = node.get("min", node.get("lower"))
        hi = node.get("max", node.get("upper"))
        if lo is None or hi is None:
            raise ValueError(f"design.holdout_lhs.ranges.{key}: need min/max (or lower/upper), "
                             f"choices, or a value/lit/expr entry")
        node = [lo, hi]
    if isinstance(node, (int, float)) and not isinstance(node, bool):
        return ("const", float(node))
    if isinstance(node, (list, tuple)):
        if node and all(isinstance(v, str) for v in node):
            return ("cat", list(node))
        if len(node) == 2:
            lo, hi = (_number(v, key, allow_pending) for v in node)
            if hi < lo:
                raise ValueError(f"design.holdout_lhs.ranges.{key}: max < min")
            return ("cont", lo, hi)
    raise ValueError(f"design.holdout_lhs.ranges.{key}: unsupported range {node!r}")


def _depth_clearance(rule) -> float | None:
    """Clearance (mm) from the ``depth_rule`` text ('... plaque thickness minus X mm ...')."""
    if rule is None:
        return None
    m = re.search(rf"minus\s+({_NUM_RE})\s*mm", str(rule))
    if m is None:
        raise ValueError(f"design.holdout_lhs.ranges.depth_rule: cannot parse the clearance from {rule!r}")
    return float(m.group(1))


def _min_local_plaque(r_lumen: float, plaque: float, offset: float, arc_deg: float, center_deg: float) -> float:
    """Minimum plaque thickness r_iel(phi) - r_lumen over the calcium arc."""
    half = math.radians(min(arc_deg, 360.0)) / 2
    phi = math.radians(center_deg) + np.linspace(-half, half, _DEPTH_RULE_SAMPLES)
    c = (offset * math.cos(math.radians(center_deg)), offset * math.sin(math.radians(center_deg)))
    return float((ray_circle_radius(phi, c, r_lumen + plaque) - r_lumen).min())


#: angular samples of the calcium arc used by the depth rule
_DEPTH_RULE_SAMPLES = 721
#: bisection steps used when the plaque offset must be reduced to fit the calcium
_OFFSET_BISECTION_STEPS = 60
#: tolerance (mm) when checking that the depth rule is met after the offset bisection
_DEPTH_RULE_TOL = 1e-9
#: minimum angular gap (deg) between a hold-out lipid pool and the calcium arc
LIPID_CALCIUM_GAP_DEG = 10.0


def holdout_specs(n: int | None = None, seed: int | None = None, *, allow_pending: bool = False,
                  on_invalid: str = "raise") -> list[LesionSpec]:
    """Latin-hypercube hold-out lesions over ``design.holdout_lhs.ranges``.

    Every key of ``ranges`` is read; unknown keys raise ``ValueError`` and a
    missing ``ranges`` raises ``KeyError``. Sampled/constant keys are
    LesionSpec fields (optionally with an ``_mm`` suffix) or the aliases
    ``arc_deg``, ``thickness_mm``, ``depth_mm``, ``length_mm``,
    ``cap_thickness_mm`` (-> ``lipid_cap_thickness``) and

    * ``lumen_area_mm2``: circular lumen of that area; the IEL radius stays at
      the core reference value, so ``plaque_thickness = R_iel,core - r_lumen``
      (the derivation of ``geometry.plaque_thickness_mm``);
    * ``ecc_index``: target plaque eccentricity (t_max - t_min)/t_max of the
      intima thickness, realised by the IEL offset
      ``o = ecc * P / (2 - ecc)`` toward the calcium (t = P +/- o).

    Rule keys: ``morphology`` (``balanced_cycle`` -> lesion i gets
    ``design.core_factorial.morphology[i % 4]``; a list -> categorical LHS
    dimension; default balanced_cycle), ``lipid_fraction`` (an LHS indicator
    dimension: lesion carries lipid if u < fraction; default 0) and
    ``depth_rule`` (if depth + thickness exceeds the minimum local plaque
    thickness over the calcium arc minus the clearance parsed from the rule
    text, depth is reduced to fit; if that would push depth below the lower
    bound of the depth range, the depth is set to that bound and the IEL
    offset is reduced by bisection until the calcium fits; a sample that
    still violates the rule at zero offset is invalid, see ``on_invalid``).
    ``lipid_fraction > 0`` requires a ``lipid_arc_deg`` entry. Lipid pools are
    centred opposite the calcium; their arc is clipped to the angular room
    left (``360 - arc - 2*LIPID_CALCIUM_GAP_DEG``) and dropped when the room is
    below the lower bound of the lipid-arc range. Every adjustment is logged
    and listed in ``spec.notes['adjustments']``.

    ``arc_deg`` and ``thickness_mm`` are required. ``n`` defaults to
    ``design.holdout_lhs.n`` and ``seed`` to ``design.seed +
    HOLDOUT_SEED_OFFSET``. ``on_invalid='raise'`` (default) raises on a
    geometrically invalid sample; ``'skip'`` drops it.
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

    dims = {_holdout_field(k): _parse_range(k, v, allow_pending)
            for k, v in ranges.items() if k not in _HOLDOUT_RULE_KEYS}
    missing = [f for f in ("ca_arc_deg", "ca_thickness") if f not in dims]
    if missing:
        raise KeyError(f"design.holdout_lhs.ranges must define {missing} (keys arc_deg / thickness_mm)")
    core_morphs = list(params.raw("design.core_factorial.morphology"))
    morph_rule = ranges.get("morphology", "balanced_cycle")
    if isinstance(morph_rule, str):
        if morph_rule != "balanced_cycle":
            raise ValueError(f"design.holdout_lhs.ranges.morphology: unknown rule {morph_rule!r}")
    else:
        dims["morphology"] = _parse_range("morphology", morph_rule)
    lipid_fraction = (_number(ranges["lipid_fraction"], "lipid_fraction", allow_pending)
                      if "lipid_fraction" in ranges else 0.0)
    if lipid_fraction > 0:
        if "lipid_arc_deg" not in dims:
            raise KeyError("design.holdout_lhs.ranges: lipid_fraction > 0 requires a lipid_arc_deg entry")
        dims["_lipid_u"] = ("cont", 0.0, 1.0)
    clearance = _depth_clearance(ranges.get("depth_rule"))

    sampled = [k for k, d in dims.items() if d[0] != "const"]
    u = qmc.LatinHypercube(d=max(1, len(sampled)), rng=np.random.default_rng(seed)).random(n)

    geo = fixed_geometry(allow_pending=allow_pending)
    r_iel_core = geo["r_lumen"] + geo["plaque_thickness"]
    specs = []
    for i in range(n):
        vals = {}
        for name, d in dims.items():
            if d[0] == "const":
                vals[name] = d[1]
                continue
            x = u[i, sampled.index(name)]
            vals[name] = (d[1][min(int(x * len(d[1])), len(d[1]) - 1)] if d[0] == "cat"
                          else d[1] + x * (d[2] - d[1]))
        lid = f"H-{i + 1:03d}"
        notes = {"lhs_seed": seed, "adjustments": [], "pending_inputs": list(geo["pending_inputs"])}
        morph = vals.pop("morphology") if "morphology" in vals else core_morphs[i % len(core_morphs)]
        arc, t = vals.pop("ca_arc_deg"), vals.pop("ca_thickness")
        center = vals.get("ca_center_deg", 0.0)
        r_lumen = (math.sqrt(vals.pop("lumen_area") / math.pi) if "lumen_area" in vals
                   else geo["r_lumen"])
        plaque = vals.get("plaque_thickness", r_iel_core - r_lumen)
        ecc = vals.pop("ecc_index", None)
        offset = vals.get("plaque_offset", ecc * plaque / (2.0 - ecc) if ecc is not None else 0.0)
        depth = vals.get("ca_depth", _depth_for(morph, geo))
        if ecc is not None:
            notes["ecc_index_target"] = ecc

        invalid_reason = None
        if clearance is not None:
            need = t + clearance
            t_min = _min_local_plaque(r_lumen, plaque, offset, arc, center)
            if depth + need > t_min:
                d_lo = dims["ca_depth"][1] if dims.get("ca_depth", ("const",))[0] == "cont" else 0.0
                new_depth = t_min - need
                if new_depth >= d_lo:
                    notes["adjustments"].append(f"depth {depth:.4f} -> {new_depth:.4f} mm (depth_rule)")
                    depth = new_depth
                else:
                    new_depth = min(depth, d_lo)
                    lo_o, hi_o = 0.0, offset
                    for _ in range(_OFFSET_BISECTION_STEPS):
                        mid = 0.5 * (lo_o + hi_o)
                        if _min_local_plaque(r_lumen, plaque, mid, arc, center) >= new_depth + need:
                            lo_o = mid
                        else:
                            hi_o = mid
                    notes["adjustments"].append(
                        f"depth {depth:.4f} -> {new_depth:.4f} mm and plaque offset {offset:.4f} -> "
                        f"{lo_o:.4f} mm (depth_rule: calcium does not fit otherwise)")
                    depth, offset = new_depth, lo_o
                    t_fit = _min_local_plaque(r_lumen, plaque, offset, arc, center)
                    if depth + need > t_fit + _DEPTH_RULE_TOL:
                        invalid_reason = (f"depth_rule cannot be met: depth {depth:.4f} + thickness {t:.4f} + "
                                          f"clearance {clearance:.4f} mm exceeds the minimum local plaque "
                                          f"thickness {t_fit:.4f} mm even with plaque offset {offset:.4f} mm")

        has_lipid = lipid_fraction > 0 and vals.pop("_lipid_u") < lipid_fraction
        vals.pop("_lipid_u", None)
        if has_lipid:
            lip_lo = dims["lipid_arc_deg"][1] if dims.get("lipid_arc_deg", ("const",))[0] == "cont" else 0.0
            room = 360.0 - arc - 2 * LIPID_CALCIUM_GAP_DEG
            lip_arc = vals.get("lipid_arc_deg", 0.0)
            if room < max(lip_lo, 1e-9):
                notes["adjustments"].append(f"lipid dropped: only {room:.1f} deg free beside the calcium")
                has_lipid = False
            elif lip_arc > room:
                notes["adjustments"].append(f"lipid arc {lip_arc:.1f} -> {room:.1f} deg (fit beside calcium)")
                vals["lipid_arc_deg"] = room
        if has_lipid:
            vals.setdefault("lipid_center_deg", center + 180.0)
        else:
            for k in ("lipid_arc_deg", "lipid_thickness", "lipid_cap_thickness"):
                vals[k] = 0.0
        if morph != "nodule":
            vals["nodule_height"] = 0.0
        notes["has_lipid"] = has_lipid
        for msg in notes["adjustments"]:
            _log.info("%s: %s", lid, msg)
        vals.update(r_lumen=r_lumen, plaque_thickness=plaque, plaque_offset=offset, ca_depth=depth,
                    notes=notes)
        try:
            if invalid_reason is not None:
                raise ValueError(invalid_reason)
            spec = make_spec(lid, morph, arc, t, geo, set_name="holdout",
                             length_mm=vals.pop("ca_length", None), **vals)
            lesion_polygons(spec)
        except ValueError as exc:
            if on_invalid == "raise":
                raise ValueError(f"hold-out sample {lid} is invalid ({exc}); narrow "
                                 f"design.holdout_lhs.ranges or pass on_invalid='skip'") from exc
            _log.warning("%s skipped: %s", lid, exc)
            continue
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

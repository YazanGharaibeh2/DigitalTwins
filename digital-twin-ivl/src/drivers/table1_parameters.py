"""Table 1: model inputs of the main analysis -> ``results/tables/table1_parameters.csv`` (+ ``.md``).

One row per input as the 2D cohort runs use it. Scope (``design.table1_parameters``): every entry
of the ``geometry`` and ``loads`` sections, the material constants of
:meth:`MaterialSet.baseline` (checked against its provenance strings), the transient-model
densities and sound speeds (SECONDARY, D15) and the pulse width, the fracture settings, the 2D
discretisation, and the quantities the solver derives from them. Config entries that :data:`ROWS`
does not name are still listed (the config is walked like
:func:`dtivl.params.export_assumptions_csv`), with their dotted key as the name.

Columns
-------
``group, parameter, symbol, value, unit, sensitivity, status, source, config_key, lit_key,
literature_value, value_from, basis``

* ``value``: ``params.get(config_key).value`` in base units (mm, MPa, s, deg; densities kg/m^3,
  sound speeds m/s), a derived property of ``MaterialSet.baseline()``, the identified stent
  stiffness of ``results/calibration_stent.json`` or the transient pulse width; ``value_from``
  says which.
* ``status``: as the params layer reports it (``Quantity.status`` or the literature row status).
  Plain numbers of the config resolve as DESIGN (:func:`dtivl.params.resolve`); when the YAML
  comment of such an entry starts with a status word, that word is used (``fracture.k_residual``:
  NUMERICAL).
* ``source``: 'First author year' of every literature row the entry uses (``lit:`` rows and the
  rows named in an ``expr:``), else the first clause of the config basis, else the YAML comment.
  ``basis`` holds the full config basis (or comment).
* ``sensitivity``: the ``sens`` values of the entry (``sens_E`` of a modulus), same units.

Literature rows reach the table only through :mod:`dtivl.params`, which refuses UNVERIFIED rows.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pandas as pd

from dtivl import params as P
from dtivl.drivers.common import RESULTS_DIR, read_json
from dtivl.drivers.tables import TABLE_DIR
from dtivl.materials.constitutive import MaterialSet

NAME = "table1_parameters"
GROUPS = ("geometry", "calcium", "fibrous tissue", "lipid", "wall", "loads", "stent", "fracture/phase field",
          "numerics")
#: config sections walked for completeness
SECTIONS = ("geometry", "materials", "fracture", "loads", "discretisation")
COLUMNS = ["group", "parameter", "symbol", "value", "unit", "sensitivity", "status", "source", "config_key",
           "lit_key", "literature_value", "value_from", "basis"]
STATUS_WORDS = ("VERIFIED", "CALIBRATED", "DERIVED", "ASSUMPTION", "SECONDARY", "DESIGN", "NUMERICAL",
                "PROVISIONAL")

#: config key -> (group, parameter, symbol). Model-form strings (``materials.*.model``,
#: ``fracture.model``/``split``) are read into the names, not listed as rows.
ROWS = {
    # geometry
    "geometry.lumen_area_mm2": ("geometry", "lumen area (circular lumen, core set)", "A_L"),
    "geometry.reference_diameter_mm": ("geometry", "reference vessel diameter (stent-expansion reference)", "D_ref"),
    "geometry.plaque_thickness_mm": ("geometry", "plaque thickness, lumen to IEL", "t_p"),
    "geometry.wall_thickness_mm": ("geometry", "wall thickness, media + adventitia", "t_w"),
    "geometry.depth_superficial_mm": ("geometry", "calcium depth: superficial sheet, nodule, fragmented", "z_s"),
    "geometry.depth_deep_mm": ("geometry", "calcium depth: deep sheet", "z_d"),
    "geometry.calcium_iel_clearance_mm": ("geometry", "minimum fibrous tissue between calcium and IEL", "c_IEL"),
    "geometry.nodule_height_mm": ("geometry", "nodule protrusion height", "h_nod"),
    "geometry.nodule_base_arc_deg": ("geometry", "nodule base arc", "β_nod"),
    "geometry.fragment_pitch_deg": ("geometry", "fragment pitch (one deposit per pitch)", "Δφ_frag"),
    "geometry.fragment_fill": ("geometry", "fraction of the pitch filled by a fragment", "f_frag"),
    "geometry.margin_3d_mm": ("geometry", "calcium-free end segment (3D models)", "L_m"),
    # calcium
    "materials.calcium.E": ("calcium", "Young's modulus (baseline, D8)", "E_Ca"),
    "materials.calcium.nu": ("calcium", "Poisson's ratio", "ν_Ca"),
    "materials.density.calcium": ("calcium", "density (transient model)", "ρ_Ca"),
    "materials.sound_speed_m_per_s.calcium": ("calcium", "longitudinal sound speed (transient model)", "c_Ca"),
    # fibrous tissue
    "materials.fibrous.E": ("fibrous tissue", "Young's modulus, small strain ({model}, μ = E/3)", "E_f"),
    "materials.bulk_to_shear_soft": ("fibrous tissue", "bulk-to-shear modulus ratio (wall, fibrous tissue, lipid)",
                                     "K/μ"),
    "materials.density.fibrous": ("fibrous tissue", "density (transient model)", "ρ_f"),
    "materials.sound_speed_m_per_s.fibrous": ("fibrous tissue", "longitudinal sound speed (transient model)", "c_f"),
    # lipid
    "materials.lipid.E": ("lipid", "Young's modulus, small strain ({model}, μ = E/3)", "E_lip"),
    "materials.density.lipid": ("lipid", "density (transient model)", "ρ_lip"),
    "materials.sound_speed_m_per_s.lipid": ("lipid", "longitudinal sound speed (transient model)", "c_lip"),
    "design.holdout_lhs.ranges.lipid_thickness_mm": ("lipid", "lipid pool thickness (hold-out lesions)", "t_lip"),
    # wall
    "materials.wall.c10": ("wall", "{model} coefficient c10", "c10"),
    "materials.wall.c20": ("wall", "{model} coefficient c20", "c20"),
    "materials.wall.c30": ("wall", "{model} coefficient c30", "c30"),
    "materials.density.wall": ("wall", "density (transient model)", "ρ_w"),
    "materials.sound_speed_m_per_s.wall": ("wall", "longitudinal sound speed (transient model)", "c_w"),
    # loads
    "loads.physiological_pressure": ("loads", "physiological lumen pressure", "p_phys"),
    "loads.ivl_balloon_pressure_atm": ("loads", "IVL balloon pressure (balloon alone, IVL inflation, predictor load)",
                                       "p_bal"),
    "loads.poba_pressure_atm": ("loads", "plain balloon pressure (no pulses)", "p_POBA"),
    "geometry.ivl_balloon_diameter_mm": ("loads", "IVL balloon diameter (radial envelope cap)", "D_IVL"),
    "loads.ivl_peak_pressure_MPa": ("loads", "IVL peak positive pulse pressure", "p_peak"),
    "loads.ivl_pulse_scale": ("loads", "pulse attenuation factor, emitter to tissue", "s_pulse"),
    "loads.perivascular_k_MPa_per_mm": ("loads", "perivascular support stiffness", "k_f"),
    # stent
    "geometry.stent_target_diameter_mm": ("stent", "stent target diameter (expanding rigid cylinder)", "D_stent"),
    "loads.balloon_rated_pressure_atm": ("stent", "maximum mean balloon pressure (rated burst pressure)", "p_rated"),
    "loads.stent_recoil_target_pct": ("stent", "acute recoil target of the stiffness identification", "r_rec"),
    "loads.malapposition_gap_mm": ("stent", "malapposition distance threshold", "g_mal"),
    # fracture / phase field
    "fracture.eps_f": ("fracture/phase field", "calcium failure strain (strain-controlled, primary, D5)", "ε_f"),
    "fracture.sigma_c_stress_controlled": ("fracture/phase field",
                                           "calcium failure stress (stress-controlled scenario, D5)", "σ_c"),
    "fracture.ell_over_h": ("fracture/phase field", "regularisation length / calcium element size", "ℓ/h"),
    "fracture.crack_filler": ("fracture/phase field", "crack filler of fully damaged calcium (D9)", "–"),
    "fracture.driving_force": ("fracture/phase field", "damage driving force (D10)", "–"),
    "fracture.k_residual": ("fracture/phase field", "residual stiffness of fully damaged calcium", "k_res"),
    "fracture.d_crack": ("fracture/phase field", "cell-average damage of a cracked cell (post-processing)", "d_crack"),
    # numerics
    "discretisation.polar_n_theta": ("numerics", "A-lines of the polar label map", "n_θ"),
    "discretisation.polar_dr_mm": ("numerics", "radial sampling of the polar label map", "Δr"),
    "discretisation.h_calcium_mm": ("numerics", "element size in and near calcium", "h_Ca"),
    "discretisation.h_soft_mm": ("numerics", "element size elsewhere", "h_soft"),
    "loads.contact_penalty_MPa_per_mm": ("numerics", "contact penalty stiffness", "k_pen"),
}
#: config entry whose value the main analysis replaces by an identified one (see design.table1_parameters)
STENT_STIFFNESS_KEY = "loads.stent_radial_stiffness_MPa_per_mm"
#: default mesh sizes that the mesh convergence study refines and coarsens (results key)
CONVERGENCE_KEYS = {"discretisation.h_calcium_mm": "h_ca_default", "discretisation.h_soft_mm": "h_soft_default"}
#: unit by key suffix, for config values typed without a unit
_SUFFIX_UNIT = (("_MPa_per_mm", "MPa/mm"), ("_mm2", "mm^2"), ("_mm", "mm"), ("_deg", "deg"), ("_pct", "%"),
                ("_MPa", "MPa"), ("_atm", "MPa"))

_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_DOI = re.compile(r"doi:\s*\S+|\bDOI\b[^.;]*|\b10\.\d{4,}/\S+", re.IGNORECASE)
_AUTHOR = re.compile(r"([A-Z][\w'\-]+(?:\s+[A-Z][\w'\-]+)*?)\s+(?:[A-Z]{1,3}\b|et al\b)")


# ---------------------------------------------------------------- citations and comments
def short_citation(source: str) -> str:
    """'First author year' of each reference of a literature ``source`` cell (references are
    separated by `` ; ``). Unattributed references (``[Authors to be confirmed] ...``) give
    ``[authors tbc]`` with the journal (or the start of the title) and the year; organisations
    (manufacturer documents) give the organisation name."""
    out = []
    for ref in (r.strip() for r in re.split(r"\s;\s", source)):
        if not ref:
            continue
        clean = _DOI.sub(" ", ref)
        m = _YEAR.search(clean)
        year = m.group(0) if m else None
        if ref.startswith("["):
            parts = [p.strip() for p in re.split(r"\.\s+", re.sub(r"^\[[^\]]*\]\.?\s*", "", clean)) if p.strip()]
            with_year = [p for p in parts if year and year in p]
            if with_year:
                hint = with_year[0][:with_year[0].index(year)].strip(" ,;.") or parts[0]
            else:
                words = parts[0].split() if parts else []
                hint = " ".join(words[:4]) + (" …" if len(words) > 4 else "")
            out.append(" ".join(x for x in ("[authors tbc]", hint, year or "n.d.") if x))
            continue
        a = _AUTHOR.match(ref)
        if a:
            out.append(f"{a.group(1)} {year or 'n.d.'}")
            continue
        org = re.split(r"\.\s", ref)[0].strip()
        out.append(f"{org} {year}" if year else f"{org} ({'manufacturer' if 'anufacturer' in ref else 'n.d.'})")
    return "; ".join(dict.fromkeys(out))


def yaml_comments(path: str | Path = P.CONFIG_YAML) -> dict[str, str]:
    """Trailing comment of every block-style ``key: scalar  # comment`` line of the config, by dotted
    key (the documentation of plain-number entries, which carry no ``basis`` field)."""
    out, stack = {}, []
    for line in Path(path).read_text().splitlines():
        m = re.match(r"^(\s*)([A-Za-z_]\w*):(.*)$", line)
        if not m:
            continue
        indent, key, rest = len(m.group(1)), m.group(2), m.group(3)
        while stack and stack[-1][0] >= indent:
            stack.pop()
        stack.append((indent, key))
        c = re.match(r"\s*[^#{\[\"']*?\s+#\s*(.*)$", rest)
        if c:
            out[".".join(k for _, k in stack)] = c.group(1).strip()
    return out


def _first_clause(text: str, limit: int = 110) -> str:
    """First clause of a basis/comment, without a leading ``STATUS:`` or ``STATUS (...):`` tag (the
    parenthetical of the tag, e.g. a decision label, is kept)."""
    t = text.strip()
    m = re.match(r"^([A-Z]{4,})(?:\s*\(([^)]*)\))?\s*:\s*", t)
    tag = ""
    if m:
        tag = m.group(2) or ""
        t = t[m.end():]
    t = re.split(r";\s|\.\s", t, maxsplit=1)[0].strip().rstrip(".")
    if len(t) > limit:
        t = t[:t.rfind(" ", 0, limit)] + " …"
    return f"{tag}: {t}" if tag and t else (tag or t)


def _expr_rows(expr: str) -> list[str]:
    """Literature rows named in an ``expr:`` entry (same grammar as :func:`dtivl.params._eval_expr`)."""
    names = [n.id for n in ast.walk(ast.parse(expr, mode="eval")) if isinstance(n, ast.Name)]
    return list(dict.fromkeys(n for n in names if n not in ("pi", "sqrt")))


# ---------------------------------------------------------------- config walk
def _is_option(node) -> bool:
    """A typed entry whose value is a word (``fracture.crack_filler``), not a number; YAML reads
    unsigned exponents such as ``1.0e3`` as strings, which :func:`dtivl.params.resolve` converts."""
    if not (isinstance(node, dict) and isinstance(node.get("value"), str)):
        return False
    try:
        float(node["value"])
    except ValueError:
        return True
    return False


def _model_name(model: str) -> str:
    return model.replace("neo_hookean", "neo-Hookean").replace("yeoh", "Yeoh").replace("_", " ")


def _excluded(key: str) -> bool:
    excl = P.raw("design.table1_parameters")["value"]["exclude"]
    return any(key == e or key.startswith(e + ".") for e in excl)


def config_leaves() -> list[str]:
    """Dotted keys of the input entries of :data:`SECTIONS` (typed, ``lit:`` and ``expr:`` entries and
    plain numbers; model-form strings and ``sens_*`` lists are not inputs of their own), minus the
    exclusions of ``design.table1_parameters``."""
    out = []

    def walk(node, prefix):
        if isinstance(node, dict) and any(k in node for k in ("lit", "expr", "value")):
            out.append(prefix)
            return
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{prefix}.{k}")
        elif isinstance(node, (int, float)) and not isinstance(node, bool):
            out.append(prefix)

    for sec in SECTIONS:
        walk(P.raw(sec), sec)
    return [k for k in out if not _excluded(k)]


def _unit(key: str, q_unit: str) -> str:
    if q_unit:
        return q_unit
    leaf = key.split(".")[-1]
    return next((u for suf, u in _SUFFIX_UNIT if leaf.endswith(suf)), "-")


def _fmt_list(vals) -> str:
    return "; ".join(f"{v:.6g}" if isinstance(v, (int, float)) else str(v) for v in vals)


def _lit_info(entry: dict) -> tuple[str, str, str]:
    """(lit_key, literature_value, source) of a ``lit:`` or ``expr:`` entry."""
    lit = P.load_literature()
    if "lit" in entry:
        names, tr = [entry["lit"]], entry.get("transform", "value")
    else:
        names, tr = _expr_rows(entry["expr"]), "value"
    rows = [lit[n] for n in names]
    keys = "; ".join(n if tr == "value" else f"{n} [{tr}]" for n in names)
    stated = "; ".join(f"{r.value_str} {r.unit}" + (f" ({r.range_str})" if tr != "value" and r.range_str else "")
                       for r in rows)
    return keys, stated, "; ".join(dict.fromkeys(short_citation(r.source) for r in rows))


def config_row(key: str, comments: dict | None = None) -> dict:
    """Table row of one config entry (value, status and provenance through :mod:`dtivl.params`)."""
    comments = yaml_comments() if comments is None else comments
    node = P.raw(key)
    entry = node if isinstance(node, dict) else {}
    parent = key.rsplit(".", 1)[0]
    group, name, symbol = ROWS.get(key, (key.split(".")[0], key, ""))
    if "{model}" in name:
        name = name.format(model=_model_name(str(P.raw(f"{parent}.model"))))
    row = {"group": group, "parameter": name, "symbol": symbol, "config_key": key, "lit_key": "",
           "literature_value": "", "value_from": "params", "basis": entry.get("basis", ""), "sensitivity": ""}
    if _is_option(entry):                                    # option entries (crack filler, driving force)
        row.update(value=entry["value"], unit="", status=entry.get("status", ""),
                   sensitivity=_fmt_list(entry.get("sens", [])))
    else:
        q = P.get(key)
        sens = list(q.sens)
        if not sens and key.endswith(".E") and isinstance(P.raw(parent), dict) and "sens_E" in P.raw(parent):
            sens = [s.value for s in P.get_list(f"{parent}.sens_E")]
        row.update(value=q.value, unit=_unit(key, q.unit), status=q.status, sensitivity=_fmt_list(sens))
    if "lit" in entry or "expr" in entry:
        row["lit_key"], row["literature_value"], row["source"] = _lit_info(entry)
    elif row["basis"]:
        row["source"] = _first_clause(row["basis"])
    elif key in comments:                                    # plain number: its YAML comment
        row["basis"] = comments[key]
        row["source"] = _first_clause(comments[key])
        word = re.match(r"^([A-Z]+)\b", comments[key])
        if word and word.group(1) in STATUS_WORDS:
            row["status"] = word.group(1)
    if key in CONVERGENCE_KEYS and (RESULTS_DIR / "verification_mesh_convergence.json").exists():
        mc = read_json("verification_mesh_convergence")
        if abs(float(mc.get(CONVERGENCE_KEYS[key], float("nan"))) - float(row["value"])) < 1e-12:
            row["source"] = "; ".join(x for x in (row.get("source"), "mesh convergence study "
                                                  "(results/verification_mesh_convergence.json)") if x)
    return row


# ---------------------------------------------------------------- special rows
def stent_stiffness_row() -> dict:
    """Identified stent radial stiffness (CALIBRATED), as the cohort runs read it."""
    rel = P.raw("design.table1_parameters")["value"]["stent_stiffness"]
    cal = read_json(Path(rel).stem)
    target_key = "loads.stent_recoil_target_pct"
    lit_row = P.raw(target_key)["lit"]
    cite = short_citation(P.load_literature()[lit_row].source)
    return {"group": "stent", "parameter": "stent radial stiffness (identified)", "symbol": "K_s",
            "value": cal["K_s_MPa_per_mm"], "unit": "MPa/mm", "sensitivity": "", "status": "CALIBRATED",
            "source": f"identified on the calcium-free {cal['lesion_template']} for the recoil target "
                      f"({target_key}; {cite})",
            "config_key": STENT_STIFFNESS_KEY, "lit_key": lit_row, "literature_value": "", "value_from": rel,
            "basis": P.raw(STENT_STIFFNESS_KEY).get("basis", "")}


def pulse_width_row() -> dict:
    """Half-sine pulse width of the transient IVL model (literature row read by the solver)."""
    from dtivl.solvers.transient import default_pulse_width_us

    name = "ivl_pulse_duration"
    r = P.load_literature()[name]
    return {"group": "loads", "parameter": "IVL pulse width (half sine, transient model)", "symbol": "t_pulse",
            "value": default_pulse_width_us(), "unit": "us", "sensitivity": "", "status": r.status,
            "source": short_citation(r.source), "config_key": "", "lit_key": f"{name} [upper]",
            "literature_value": f"{r.value_str} {r.unit}", "value_from": "dtivl.solvers.transient.default_pulse_width_us",
            "basis": r.notes}


def derived_rows(ms: MaterialSet) -> list[dict]:
    """Quantities the solver derives from the inputs (strain-controlled baseline, D5)."""
    from dtivl.fracture.phasefield import regularisation_length

    ell = regularisation_length()
    ms_src, pf_src = "MaterialSet.baseline()", "dtivl.fracture.phasefield.regularisation_length"
    spec = [
        ("crack-filler stiffness floor, max(k_res, μ_f/μ_Ca)", "k", ms.k_floor, "-", ms_src,
         "fracture.crack_filler; fracture.k_residual; materials.fibrous.E; materials.calcium.E; materials.calcium.nu"),
        ("regularisation length, (ℓ/h) h_Ca", "ℓ", ell, "mm", pf_src,
         "fracture.ell_over_h; discretisation.h_calcium_mm"),
        ("damage-onset energy density, E_Ca ε_f²/2", "ψ_crit", ms.psi_crit, "MPa", ms_src,
         "materials.calcium.E; fracture.eps_f"),
        ("implied calcium failure stress, E_Ca ε_f", "σ_crit", ms.sigma_crit, "MPa", ms_src,
         "materials.calcium.E; fracture.eps_f"),
        ("fracture toughness, 16 ℓ (1 − k) ψ_crit / 3 (AT1 onset at ψ_crit)", "G_c", ms.gc(ell), "N/mm",
         f"{ms_src}.gc", "fracture.ell_over_h; discretisation.h_calcium_mm; fracture.crack_filler"),
    ]
    return [{"group": "fracture/phase field", "parameter": n, "symbol": s, "value": v, "unit": u, "sensitivity": "",
             "status": "DERIVED", "source": f"derived from {k}", "config_key": k, "lit_key": "",
             "literature_value": "", "value_from": vf, "basis": n}
            for n, s, v, u, vf, k in spec]


# ---------------------------------------------------------------- table
def check_material_set(df: pd.DataFrame, ms: MaterialSet | None = None) -> None:
    """Every input of ``MaterialSet.baseline()`` is a row with the same value and provenance."""
    ms = MaterialSet.baseline() if ms is None else ms
    by_prov = {}
    for key in df.loc[df["value_from"] == "params", "config_key"]:
        if not _is_option(P.raw(key)):
            by_prov.setdefault(P.get(key).provenance, []).append(key)
    for fld, prov in ms.provenance.items():
        keys = by_prov.get(prov)
        if not keys:
            raise ValueError(f"Table 1 has no row for MaterialSet input {fld} ({prov})")
        if not any(abs(P.get(k).value - getattr(ms, fld)) <= 1e-12 * max(1.0, abs(getattr(ms, fld))) for k in keys):
            raise ValueError(f"Table 1 row {keys} does not hold the MaterialSet value of {fld}")
    for fld, key in (("k_res", "fracture.k_residual"), ("crack_filler", "fracture.crack_filler")):
        if key not in set(df["config_key"]):
            raise ValueError(f"Table 1 has no row for MaterialSet input {fld} ({key})")


def build() -> pd.DataFrame:
    """Table 1 rows (see the module docstring), ordered by :data:`GROUPS` and :data:`ROWS`."""
    comments = yaml_comments()
    keys = list(ROWS) + [k for k in config_leaves() if k not in ROWS and k != STENT_STIFFNESS_KEY]
    rows = [config_row(k, comments) for k in keys if k != STENT_STIFFNESS_KEY]
    ms = MaterialSet.baseline()
    rows += [stent_stiffness_row(), pulse_width_row()] + derived_rows(ms)
    df = pd.DataFrame(rows)[COLUMNS]
    bad = df[~df["status"].isin(P.MAIN_ANALYSIS_OK | {"SECONDARY"})]
    if not bad.empty:
        raise ValueError(f"Table 1 rows with a status not allowed in the main analysis: {bad['config_key'].tolist()}")
    rank = {g: i for i, g in enumerate(GROUPS)}
    df["_g"] = df["group"].map(lambda g: rank.get(g, len(GROUPS)))
    df = df.sort_values("_g", kind="stable").drop(columns="_g").reset_index(drop=True)
    check_material_set(df, ms)
    return df


def _md_cell(x) -> str:
    if isinstance(x, float):
        return f"{x:.4g}"
    return str(x).replace("|", "\\|")


def to_markdown(df: pd.DataFrame) -> str:
    cols = ["group", "parameter", "symbol", "value", "unit", "sensitivity", "status", "source", "config_key",
            "lit_key"]
    head = ["Group", "Parameter", "Symbol", "Value", "Unit", "Sensitivity", "Status", "Source", "Config key",
            "Literature row"]
    lines = ["# Table 1. Model inputs of the main analysis", "",
             "Generated by `dtivl.drivers.table1_parameters` from `params/` and `results/` (values in base "
             "units: mm, MPa, s; densities kg/m^3, sound speeds m/s). Review copy; the manuscript reads the CSV.",
             "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(_md_cell(r[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def make() -> str:
    """Write ``results/tables/table1_parameters.csv`` and ``.md``; returns the CSV path."""
    df = build()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLE_DIR / f"{NAME}.csv"
    df.to_csv(out, index=False)
    (TABLE_DIR / f"{NAME}.md").write_text(to_markdown(df))
    return str(out)


if __name__ == "__main__":
    print(make())

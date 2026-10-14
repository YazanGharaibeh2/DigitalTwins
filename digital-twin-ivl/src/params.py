"""Single access point for every numeric input.

Two sources are read; nothing numeric is hard-coded in analysis modules:

* ``params/literature_parameters.csv`` - cited literature values (rule 2).
  Each row has a verification status parsed from its ``notes`` column.
* ``params/study_config.yaml`` - PI design decisions and modelling
  assumptions. Entries either carry a typed ``value`` with a ``basis`` and
  ``status``, or reference literature rows with ``lit:`` / ``expr:``.

Base units after resolution: mm, MPa (N/mm^2), N, s, degrees. Literature
values are converted from their stated unit.
"""
from __future__ import annotations

import ast
import csv
import functools
import math
import operator
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from dtivl import ROOT

LIT_CSV = ROOT / "params" / "literature_parameters.csv"
CONFIG_YAML = ROOT / "params" / "study_config.yaml"
ASSUMPTIONS_CSV = ROOT / "params" / "model_assumptions.csv"

# Conversion factors to base units (mm, MPa, s, deg).
_UNIT_FACTOR = {
    "kPa": 1e-3, "MPa": 1.0, "GPa": 1e3, "Pa": 1e-6,
    "atm": 0.101325, "mmHg": 0.000133322,
    "mm": 1.0, "um": 1e-3, "m/s": 1.0,  # sound speed kept in m/s (= mm/ms)
    "mm^2": 1.0, "deg": 1.0, "-": 1.0, "": 1.0, "%": 1.0, "Hz": 1.0, "us": 1e-6,
    "kg/m^3": 1.0,
}

_NUM = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"

# Status precedence parsed from the notes column of the literature table.
_STATUS_PATTERNS = [
    ("UNVERIFIED", re.compile(r"\bUNVERIFIED\b")),
    ("PROVISIONAL", re.compile(r"\bPROVISIONAL\b")),
    ("SECONDARY", re.compile(r"\bSECONDARY\b")),
    ("VERIFIED", re.compile(r"\bVERIFIED\b")),
]

#: Statuses allowed as inputs to the main analysis.
MAIN_ANALYSIS_OK = {"VERIFIED", "PROVISIONAL", "DERIVED", "ASSUMPTION", "NUMERICAL", "DESIGN", "CALIBRATED"}


@dataclass(frozen=True)
class LitParam:
    """One row of the literature table, with parsed numbers in base units."""

    name: str
    value_str: str
    range_str: str
    unit: str
    source: str
    notes: str
    status: str
    value: float | None = None   # central value (base units)
    sd: float | None = None
    lower: float | None = None   # bounds if the value or range is "a-b"
    upper: float | None = None
    alt: float | None = None     # alternative value if the range column is a single number, e.g. "1450 (other FE input)"


@dataclass(frozen=True)
class Quantity:
    """A resolved numeric input with provenance."""

    value: float
    unit: str
    provenance: str
    status: str
    sens: tuple[float, ...] = field(default_factory=tuple)


def _status_from_notes(notes: str) -> str:
    for status, pat in _STATUS_PATTERNS:
        if pat.search(notes):
            return status
    return "UNKNOWN"


def _parse_numbers(value_str: str, range_str: str, factor: float):
    value = sd = lower = upper = None
    vs = value_str.strip()
    m_rng = re.fullmatch(rf"({_NUM})\s*-\s*({_NUM})", vs)
    if m_rng:
        lower, upper = (float(m_rng.group(1)) * factor, float(m_rng.group(2)) * factor)
    elif re.fullmatch(rf"<\s*({_NUM})", vs):
        upper = float(re.fullmatch(rf"<\s*({_NUM})", vs).group(1)) * factor
    elif re.fullmatch(_NUM, vs):
        value = float(vs) * factor
    m_sd = re.search(rf"SD\s*({_NUM})", range_str)
    if m_sd:
        sd = float(m_sd.group(1)) * factor
    m_b = re.search(rf"\(\s*({_NUM})\s*-\s*({_NUM})\s*\)", range_str)
    if m_b and lower is None:
        lower, upper = float(m_b.group(1)) * factor, float(m_b.group(2)) * factor
    return value, sd, lower, upper


def _parse_alt(range_str: str, factor: float) -> float | None:
    """Alternative value: a range column holding one number with an optional parenthesised remark
    (e.g. ``"1450 (other FE input)"``); ``None`` otherwise (SD, bounds, counts, free text)."""
    m = re.fullmatch(rf"\s*({_NUM})\s*(?:\([^()]*\))?\s*", range_str)
    return float(m.group(1)) * factor if m else None


@functools.lru_cache(maxsize=None)
def load_literature(path: str | Path = LIT_CSV) -> dict[str, LitParam]:
    out: dict[str, LitParam] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            unit = row["unit"].strip()
            factor = _UNIT_FACTOR.get(unit, 1.0)
            value, sd, lower, upper = _parse_numbers(row["value"], row["range"], factor)
            out[row["parameter"]] = LitParam(
                name=row["parameter"], value_str=row["value"], range_str=row["range"],
                unit=unit, source=row["source"], notes=row["notes"],
                status=_status_from_notes(row["notes"]),
                value=value, sd=sd, lower=lower, upper=upper, alt=_parse_alt(row["range"], factor),
            )
    return out


@functools.lru_cache(maxsize=None)
def load_config(path: str | Path = CONFIG_YAML) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def lit(name: str, transform: str = "value", allow_unverified: bool = False) -> Quantity:
    """Return a literature value in base units.

    ``transform`` is one of ``value``, ``value+sd``, ``value-sd``, ``lower``,
    ``upper``, ``alt`` (the alternative value of a single-number range column,
    :func:`_parse_alt`).
    """
    p = load_literature()[name]
    if p.status not in MAIN_ANALYSIS_OK and not allow_unverified:
        raise ValueError(f"literature parameter {name!r} has status {p.status}; not allowed in main analysis")
    if transform == "value":
        v = p.value
    elif transform == "value+sd":
        v = None if p.value is None or p.sd is None else p.value + p.sd
    elif transform == "value-sd":
        v = None if p.value is None or p.sd is None else p.value - p.sd
    elif transform == "lower":
        v = p.lower
    elif transform == "upper":
        v = p.upper
    elif transform == "alt":
        v = p.alt
    else:
        raise ValueError(f"unknown transform {transform!r}")
    if v is None:
        raise ValueError(f"literature parameter {name!r} has no numeric {transform}")
    unit = "MPa" if p.unit in ("kPa", "MPa", "GPa", "atm", "Pa", "mmHg") else p.unit
    return Quantity(v, unit, f"lit:{name}[{transform}] ({p.status})", p.status)


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.USub: operator.neg}


def _eval_expr(expr: str) -> float:
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id == "pi":
                return math.pi
            return lit(node.id).value
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "sqrt" and len(node.args) == 1:
            return math.sqrt(ev(node.args[0]))
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError(f"unsupported expression element in {expr!r}")
    return ev(ast.parse(expr, mode="eval"))


def resolve(entry: Any, *, allow_pending: bool = False) -> Quantity:
    """Resolve one config entry (typed value, ``lit:`` or ``expr:``).

    ``sens`` items are numbers or entries resolved the same way (e.g. a ``lit:`` row with a
    transform and its own ``allow_status``).
    """
    if isinstance(entry, (int, float)):
        return Quantity(float(entry), "", "config:design", "DESIGN")
    if not isinstance(entry, dict):
        raise TypeError(f"cannot resolve {entry!r}")
    sens = tuple()
    if "sens" in entry:
        sens = tuple(resolve(x, allow_pending=allow_pending).value if isinstance(x, dict) else float(x)
                     for x in entry["sens"])
    if "lit" in entry:
        allowed = set(entry.get("allow_status", []))
        p = load_literature()[entry["lit"]]
        q = lit(entry["lit"], entry.get("transform", "value"), allow_unverified=p.status in allowed)
        return Quantity(q.value, q.unit, q.provenance, q.status, sens)
    if "expr" in entry:
        v = _eval_expr(entry["expr"])  # names are literature rows (main-analysis statuses only), pi, sqrt()
        return Quantity(v, entry.get("unit", ""), f"expr:{entry['expr']}", entry.get("status", "DERIVED"), sens)
    if "value" in entry:
        status = entry.get("status", "ASSUMPTION")
        if status == "PENDING" and not allow_pending:
            raise ValueError(f"config entry is PENDING: {entry.get('basis', entry)}")
        return Quantity(float(entry["value"]), entry.get("unit", ""), f"config:{entry.get('basis', '')}", status, sens)
    raise ValueError(f"cannot resolve {entry!r}")


def get(dotted: str, *, allow_pending: bool = False) -> Quantity:
    """Resolve ``a.b.c`` from the YAML config, e.g. ``get('materials.wall.c10')``."""
    node: Any = load_config()
    for key in dotted.split("."):
        node = node[key]
    return resolve(node, allow_pending=allow_pending)


def get_list(dotted: str) -> list[Quantity]:
    node: Any = load_config()
    for key in dotted.split("."):
        node = node[key]
    return [resolve(e) for e in node]


def raw(dotted: str) -> Any:
    """Raw (unresolved) config node, for design levels and options."""
    node: Any = load_config()
    for key in dotted.split("."):
        node = node[key]
    return node


def check_ready(keys: list[str]) -> list[str]:
    """Return the dotted keys that are still PENDING (empty list = ready)."""
    pending = []
    for k in keys:
        try:
            get(k)
        except ValueError as exc:
            if "PENDING" in str(exc):
                pending.append(k)
            else:
                raise
    return pending


def export_assumptions_csv(path: str | Path = ASSUMPTIONS_CSV) -> int:
    """Regenerate ``model_assumptions.csv`` from the YAML (single source of truth)."""
    rows = []

    def walk(node, prefix):
        if isinstance(node, dict) and ("basis" in node or "status" in node) and ("value" in node or "expr" in node):
            val = node.get("value", node.get("expr", ""))
            rows.append([prefix, val, " | ".join(str(s) for s in node.get("sens", [])),
                         node.get("unit", ""), node.get("basis", ""), node.get("status", "")])
            return
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{prefix}.{k}" if prefix else k)

    walk(load_config(), "")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["assumption", "value", "sensitivity_values", "unit", "basis", "status"])
        w.writerows(rows)
    return len(rows)

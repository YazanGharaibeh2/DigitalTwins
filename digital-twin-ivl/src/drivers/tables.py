"""Manuscript tables: each builder reads only params/ and results/ files and writes
``results/tables/<name>.csv`` (machine-readable, read by the manuscript builder). Values are copied
from their source files, never typed.
"""
from __future__ import annotations

import importlib

from dtivl.drivers.common import RESULTS_DIR

TABLE_DIR = RESULTS_DIR / "tables"

#: table name -> "module:function"
REGISTRY = {
    "table1_parameters": "dtivl.drivers.table1_parameters:make",
    "table2_verification_consistency": "dtivl.drivers.table2_verification_consistency:make",
    "table3_predictors": "dtivl.drivers.table3_predictors:make",
    "table4_models": "dtivl.drivers.table4_models:make",
}


def make(name: str) -> str:
    mod, fn = REGISTRY[name].split(":")
    return getattr(importlib.import_module(mod), fn)()


def make_all(names=None, *, skip_missing: bool = False) -> dict:
    out = {}
    for name in names or REGISTRY:
        try:
            out[name] = make(name)
        except (ModuleNotFoundError, FileNotFoundError) as exc:
            if not skip_missing:
                raise
            out[name] = f"skipped: {type(exc).__name__}: {exc}"
    return out

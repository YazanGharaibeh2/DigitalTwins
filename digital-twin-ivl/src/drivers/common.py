"""Shared helpers for pipeline stages: paths, JSON I/O, cohort and mesh access."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from dtivl import ROOT
from dtivl.geometry.polar import PolarLabelMap

COHORT_DIR = ROOT / "cohort"
RESULTS_DIR = ROOT / "results"
MESH_DIR = COHORT_DIR / "meshes"


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if math.isnan(v) else v
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, np.ndarray):
        return _to_jsonable(obj.tolist())
    if isinstance(obj, Path):
        return str(obj.relative_to(ROOT)) if obj.is_relative_to(ROOT) else str(obj)
    return obj


def write_json(name: str, data: dict) -> Path:
    """Write ``results/<name>.json`` (NaN -> null, numpy -> python)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(_to_jsonable(data), f, indent=1, sort_keys=False)
    return path


def read_json(name: str) -> dict:
    with open(RESULTS_DIR / f"{name}.json") as f:
        return json.load(f)


def load_mask(lesion_id: str) -> PolarLabelMap:
    return PolarLabelMap.load(COHORT_DIR / "masks" / f"{lesion_id}.npz")


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.t0

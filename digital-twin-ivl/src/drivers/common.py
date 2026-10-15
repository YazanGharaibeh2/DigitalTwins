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


# ---------------------------------------------------------------------------
# Lock files for concurrent workers
# ---------------------------------------------------------------------------
def _process_start_epoch(pid: int) -> float | None:
    """Start time of ``pid`` (seconds since the epoch) from /proc, or None if unavailable."""
    import os

    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        start_ticks = int(stat.rsplit(")", 1)[1].split()[19])
        btime = next(int(line.split()[1]) for line in Path("/proc/stat").read_text().splitlines()
                     if line.startswith("btime "))
        return btime + start_ticks / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, StopIteration, IndexError):
        return None


def _holder_alive(text: str, written: float | None = None) -> bool:
    """True if the lock text (a pid) names a live Python process that existed when the lock was written.

    After a machine restart a pid in an old lock may belong to an unrelated, newer process; it is
    detected by the process start time (later than the lock's modification time ``written``) and by
    its command line (not a Python process).
    """
    import os

    pid = int(text) if text.isdigit() else 0
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True                                  # live process of another user
    start = _process_start_epoch(pid)
    if written is not None and start is not None and start > written + 1.0:
        return False                                 # pid reused by a process started after the lock
    try:
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return True                                  # no /proc: trust the pid check
    return b"python" in cmd


def claim_lock(lock: Path, grace_s: float = 60.0) -> bool:
    """Atomically claim ``lock`` for this process (the file holds the pid); stale locks (dead holder,
    pid reused after a restart) are taken over. An empty lock younger than ``grace_s`` counts as held
    (pid not yet written)."""
    import os

    lock = Path(lock)
    lock.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return True
        except FileExistsError:
            try:
                text, written = lock.read_text().strip(), lock.stat().st_mtime
                if not text and time.time() - written < grace_s:
                    return False
                if _holder_alive(text, written):
                    return False
            except FileNotFoundError:
                pass
            lock.unlink(missing_ok=True)             # stale lock: retry once
    return False

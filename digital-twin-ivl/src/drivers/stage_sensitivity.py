"""Stage: sensitivity analysis of the outcome model (Checkpoint 2: D5, D9, D10, D15; study plan Phase 4).

Outcome model (:func:`run_model`)
---------------------------------
The IVL protocol of :mod:`dtivl.drivers.stage_cohort_runs`
(:func:`dtivl.solvers.nonlinear.run_ivl` with ``T = T_equivalent`` from
:func:`dtivl.solvers.transient.transfer_factor` for the same inputs) plus the
balloon-alone protocol (:func:`dtivl.solvers.nonlinear.run_pressure` to
``loads.ivl_balloon_pressure_atm`` with the IVL balloon cap), each on a fresh
:class:`~dtivl.solvers.nonlinear.NonlinearState`. With the baseline inputs
(:meth:`ModelInputs.baseline`) the run is the ``ivl`` and ``balloon_only``
protocols of the cohort runs. Recorded per run: through-thickness damage
yes/no under IVL and under the balloon alone, damage-onset load
(``cracks.onset_load``) with the load of the step before onset and the step
size (:func:`onset_bracket`), damaged fraction, ``T_equivalent``,
``n_steps_unconverged_stagger``, protocol status and wall time
(:func:`outputs`). ``T_equivalent`` is an input of the outcome model and is
recorded as such; no predictor is computed here. An outcome is decided only
by a protocol that ran to its end (status ``ok``): if the IVL protocol fails
after its peak load without fracture (or without onset), the IVL fracture
flag (or the onset load) is unknown (``None``, flag
``ivl_failed_after_peak``), since it could first appear in the phases that
never ran. A run is complete only if no protocol raised (status
``error``); otherwise the record keeps the error and resume retries it.

Inputs (:class:`ModelInputs`)
-----------------------------
Every input a scenario or a Morris sample can change, starting from the
baseline: the :class:`~dtivl.materials.constitutive.MaterialSet` (moduli,
failure criterion, crack filler), ``ell_over_h`` (``ell = ell_over_h x
discretisation.h_calcium_mm``), the IVL ``pulse_scale`` (enters ``run_ivl``
only; ``T`` does not depend on the pulse amplitude because the transient
problem is linear), the perivascular foundation ``k_found`` (enters the
nonlinear state and the static denominators of ``transfer_factor``), the
damage ``driving_force`` and the calcium density ``rho_calcium`` (enters the
transient ``T_equivalent`` only, through ``transfer_factor``'s
``densities``, D15). An input changed by a scenario enters every place where
the baseline model reads it. ``T_equivalent`` is recomputed whenever an input
of the transient analysis changes (:func:`transient_key`); otherwise the
value of an earlier run of the same lesion with the same transient inputs is
reused (``transient.source``).

One-at-a-time (:func:`one_at_a_time`)
-------------------------------------
Scenarios (:func:`oat_scenarios`) are built from the ``sens`` entries of
``params/study_config.yaml`` listed in :data:`OAT_ENTRIES` (crack filler
``none``, D9; ``ell_over_h``; ``ivl_pulse_scale``; perivascular foundation;
soft-tissue ``K/mu``; damage driving force, D10; calcium density, D15, the
alternative model-input value of ``density_calcified_plaque_model_input``)
plus the D5 stress-controlled failure criterion (:data:`FAILURE_MODE_SENS`),
one input at a time; ``sens`` values equal to the baseline are skipped. The
baseline itself is the first scenario. Lesions: the pilot lesions
(:func:`dtivl.drivers.stage_pilot.pilot_ids`). One file per (lesion,
scenario) in ``results/sensitivity_oat_runs/``; ``results/sensitivity_oat.json``
collects them (:func:`collect_oat`; the IVL onset-load change relative to the
baseline only where both onsets are observed, flag
``ivl_onset_load_censored_vs_baseline``). With the configuration at the time of
writing: 13 scenarios x 3 pilot lesions = 39 runs (:func:`n_oat_runs`).

Morris screening (:func:`morris`, :func:`analyze`)
--------------------------------------------------
Elementary effects (SALib 1.6, ``SALib.sample.morris`` /
``SALib.analyze.morris``) on the core lesions of
``design.sensitivity.morris_lesions``, with ``r`` trajectories on a
``morris_levels`` grid, selected from a pool of candidates (Campolongo,
local optimisation), seed ``design.seed + design.sensitivity.seed_offset``.
The same trajectories are used for every lesion. Inputs and ranges
(:func:`morris_inputs`) come from the config ``sens`` entries or literature
ranges, each with a ``basis`` string: log10 calcium modulus between the two
pilot-sweep moduli adjacent to the D8 baseline; failure strain over the
literature range of ``calcification_fracture_stretch`` minus 1; IVL pulse
scale; ``ell_over_h``; log10 perivascular foundation; calcium density (D15);
soft-tissue ``K/mu``; fibrous modulus. Outputs (:data:`MORRIS_OUTPUTS`): IVL
damage-onset load, censored at the IVL peak load ``p_top`` without onset
(the censoring flag is recorded, :func:`censored_onset`), the same load with
censored and after-peak runs as missing values (``ivl_onset_load_uncensored``:
only trajectories without such a point), IVL through-thickness damage (0/1)
and balloon-alone through-thickness damage (0/1). A censored value ``p_top =
p_bal + T_equivalent x p_peak x pulse_scale`` moves with the pulse scale and
every input of ``T_equivalent`` although no onset occurs, so
:func:`analyze` counts per input the elementary effects with a censored (or
after-peak) endpoint and the trajectories with every point censored. Onset
loads are quantised to the load increments of the IVL ramps; the median step
(``ivl_onset_step``) is reported next to the indices. Number of model runs =
``r (k + 1) x lesions``; with the
configuration at the time of writing (r = 6, k = 8 inputs, 4 lesions):
6 x 9 x 4 = 216 runs (:func:`n_morris_runs`), each one transient (unless
reused) + one IVL + one balloon-alone protocol. One file per sample in
``results/sensitivity_morris_runs/<lesion>_<i>.json`` (``i`` = row of the
sample matrix, zero-padded); the sample matrix is stored in
``results/sensitivity_morris_runs/design.json`` and checked on every resume.
:func:`analyze` writes ``results/sensitivity_morris.json``: ``mu``, ``mu_star``,
``sigma`` and ``mu_star_conf`` per input, output and lesion (and pooled over
lesions). SALib computes the elementary effects with the inputs scaled to
[0, 1], so ``mu_star`` is in output units per normalised input range and
ranks the inputs over their stated ranges. A trajectory with a missing value
(protocol error, or a protocol that stopped before the outcome was decided)
is excluded for that output; the exclusions are listed. The pooled indices
concatenate the same ``r`` trajectories once per lesion and SALib's bootstrap
treats the pooled elementary effects as independent: the pooled
``mu_star_conf`` is conditional on the ``r`` trajectories, and with the small
``r`` of the design ``mu_star_conf`` is unstable (stated in the output note).

Workers and resume
------------------
Every (lesion, scenario) or (lesion, sample) is claimed with a lock file
holding the worker's pid (stale locks of dead processes are taken over), so
any number of worker processes can run :func:`one_at_a_time` or
:func:`morris` concurrently; complete runs are skipped (``resume=True``).
Each process meshes a lesion once into its own file (no collision with
other processes meshing the same lesion). Run from the command line:
``python -m dtivl.drivers.stage_sensitivity {oat,morris,analyze,collect}``.
"""
from __future__ import annotations

import gc
import json
import logging
import math
import os
import time
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

import numpy as np

from dtivl import params as P
from dtivl.drivers.common import MESH_DIR, RESULTS_DIR, Timer, _to_jsonable, load_mask
from dtivl.geometry.labels import TAG_CALCIUM
from dtivl.materials.constitutive import MaterialSet

log = logging.getLogger(__name__)

OAT_RUNS = "sensitivity_oat_runs"
OAT_JSON = "sensitivity_oat.json"
MORRIS_RUNS = "sensitivity_morris_runs"
MORRIS_JSON = "sensitivity_morris.json"
MORRIS_DESIGN = "design.json"
BASELINE = "baseline"

#: one-at-a-time scenarios: (config entry whose ``sens`` list gives the values, model input it sets,
#: Checkpoint decision quoted in the basis or None). Entries missing from the config are skipped.
OAT_ENTRIES = (
    ("fracture.crack_filler", "crack_filler", "D9"),
    ("fracture.ell_over_h", "ell_over_h", None),
    ("loads.ivl_pulse_scale", "pulse_scale", None),
    ("loads.perivascular_k_MPa_per_mm", "k_found", None),
    ("materials.bulk_to_shear_soft", "k_over_mu", None),
    ("fracture.driving_force", "driving_force", "D10"),
    ("materials.density.calcium", "rho_calcium", "D15"),
)
#: D5 sensitivity: stress-controlled calcium failure (sigma_c = fracture.sigma_c_stress_controlled);
#: a MaterialSet.failure_mode name, not a number.
FAILURE_MODE_SENS = ("stress",)
FAILURE_MODE_DECISION = "D5"

#: Morris outputs (keys of :func:`outputs`) -> description
MORRIS_OUTPUTS = {
    "ivl_onset_load": "IVL damage-onset load (MPa, first load with max d >= fracture.d_crack); without onset "
                      "censored at the IVL peak load p_top (flag ivl_onset_censored)",
    "ivl_onset_load_uncensored": "IVL damage-onset load (MPa) of the runs with an onset during inflation or pulse; "
                                 "a censored or after-peak run (value p_top) is a missing value, so only "
                                 "trajectories without such a point enter",
    "ivl_fracture": "IVL through-thickness damage (1 yes, 0 no)",
    "balloon_fracture": "balloon-alone (loads.ivl_balloon_pressure_atm, IVL balloon cap) through-thickness damage "
                        "(1 yes, 0 no)",
}
#: Morris output -> per-run flags (keys of :func:`outputs`) whose elementary effects :func:`analyze` counts
#: per input (the flagged value sits at p_top, which moves with pulse_scale and T_equivalent); the counts
#: of either flag are reported as :data:`AT_P_TOP`
CENSORING_FLAGS = {"ivl_onset_load": ("ivl_onset_censored", "ivl_onset_after_peak")}
AT_P_TOP = "at_p_top"
#: Morris output -> per-run load-step size at onset (key of :func:`outputs`); median reported by :func:`analyze`
ONSET_STEPS = {"ivl_onset_load": "ivl_onset_step", "ivl_onset_load_uncensored": "ivl_onset_step"}
#: protocols of one outcome-model run (keys of the run record)
PROTOCOLS = ("ivl", "balloon_only")
#: transient outputs kept in the run records (as in stage_cohort_runs)
TRANSIENT_KEYS = ("T", "T_equivalent", "sigma1_peak_transient", "sigma1_static", "sigma1_static_equivalent",
                  "t_peak", "peak_at_window_end")
#: IVL phases with rising load: a damage onset there is an onset at its load
IVL_LOADING_PHASES = ("ivl_inflate", "ivl_pulse")
#: significant digits of the inputs that identify a transient computation (:func:`transient_key`)
KEY_DIGITS = 12
#: relative tolerance to find the baseline modulus among the pilot-sweep moduli (e_sweep rounds to 1e-6 MPa)
SWEEP_MATCH_RTOL = 1e-6
#: an empty lock file (pid not yet written) younger than this (s) is treated as held
EMPTY_LOCK_GRACE_S = 60

#: MaterialSet fields entering the transient analysis (small-strain moduli of ``MaterialSet.lame``)
_TRANSIENT_MS_FIELDS = ("wall_c10", "fibrous_E", "lipid_E", "ca_E", "ca_nu", "k_over_mu")
_MS_FIELD_NAMES = tuple(f.name for f in fields(MaterialSet) if f.name != "provenance")


# ---------------------------------------------------------------------------
# Model inputs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelInputs:
    """Inputs of the outcome model that a sensitivity scenario or Morris sample can change.

    ``ms``: material set; ``ell_over_h``: phase-field length over calcium mesh size;
    ``pulse_scale``: IVL pulse amplitude factor (``run_ivl`` only); ``k_found``: perivascular
    foundation stiffness (MPa/mm; nonlinear state and transient static denominators);
    ``driving_force``: damage driving force; ``rho_calcium``: calcium density (kg/m^3; transient
    only); ``changed``: names of the inputs set away from the baseline (record only).
    """

    ms: MaterialSet
    ell_over_h: float
    pulse_scale: float
    k_found: float
    driving_force: str
    rho_calcium: float
    changed: tuple = ()

    @classmethod
    def baseline(cls) -> "ModelInputs":
        """Baseline inputs: exactly what the cohort runs use (config defaults)."""
        from dtivl.fracture import phasefield as PF

        return cls(ms=MaterialSet.baseline(), ell_over_h=P.get("fracture.ell_over_h").value,
                   pulse_scale=P.get("loads.ivl_pulse_scale").value,
                   k_found=P.get("loads.perivascular_k_MPa_per_mm").value,
                   driving_force=PF.config_driving_force(),
                   rho_calcium=P.get("materials.density.calcium").value)

    def set(self, **values) -> "ModelInputs":
        """Copy with inputs changed: MaterialSet field names (``ca_E``, ``eps_f``, ``fibrous_E``,
        ``k_over_mu``, ``crack_filler``, ``failure_mode``, ...) or ModelInputs fields."""
        own = {f.name for f in fields(self)} - {"ms", "changed"}
        ms_kw = {k: v for k, v in values.items() if k in _MS_FIELD_NAMES}
        own_kw = {k: v for k, v in values.items() if k in own}
        unknown = set(values) - set(ms_kw) - set(own_kw)
        if unknown:
            raise KeyError(f"unknown model inputs {sorted(unknown)}")
        ms = self.ms
        if ms_kw:
            prov = {**ms.provenance, **{k: "sensitivity" for k in ms_kw}}
            ms = replace(ms, **ms_kw, provenance=prov)
        changed = tuple(dict.fromkeys(self.changed + tuple(values)))
        return replace(self, ms=ms, changed=changed, **own_kw)

    def ell(self, h_ca: float | None = None) -> float:
        """Regularisation length (mm): ``ell_over_h x h_ca`` (default ``discretisation.h_calcium_mm``)."""
        from dtivl.fracture import phasefield as PF

        return PF.regularisation_length(h_ca, self.ell_over_h)

    def densities(self) -> dict[int, float]:
        """Per-tag densities of the transient analysis: config defaults, calcium replaced by ``rho_calcium``."""
        from dtivl.solvers import transient as TR

        rho = TR.default_densities()
        rho[TAG_CALCIUM] = self.rho_calcium * TR.KG_PER_M3_TO_N_US2_PER_MM4
        return rho

    def state_kwargs(self, h_ca: float | None = None) -> dict:
        """Keyword arguments of :class:`~dtivl.solvers.nonlinear.NonlinearState` (besides mesh and ``ms``)."""
        from dtivl.solvers.nonlinear import SolverSettings

        return {"ell": self.ell(h_ca), "k_found": self.k_found,
                "settings": SolverSettings(driving_force=self.driving_force)}

    def describe(self, h_ca: float | None = None) -> dict:
        """JSON record of every input (and the derived fracture constants; ``ell_mm`` for ``h_ca``)."""
        ms = self.ms
        out = {"ms": {k: getattr(ms, k) for k in _MS_FIELD_NAMES}, "ell_over_h": self.ell_over_h,
               "ell_mm": self.ell(h_ca), "pulse_scale": self.pulse_scale, "k_found": self.k_found,
               "driving_force": self.driving_force, "rho_calcium": self.rho_calcium, "changed": list(self.changed),
               "psi_crit": ms.psi_crit, "sigma_crit": ms.sigma_crit}
        try:
            out["k_floor"] = ms.k_floor
        except ValueError as exc:
            out["k_floor"], out["k_floor_error"] = None, str(exc)
        return out


def transient_key(inputs: ModelInputs, md) -> str:
    """Identity of a transient computation: the inputs of ``transfer_factor`` (small-strain moduli,
    densities, sound speeds, foundation) and the mesh size, as a JSON string."""
    from dtivl.solvers import transient as TR

    key = {f: getattr(inputs.ms, f) for f in _TRANSIENT_MS_FIELDS}
    key.update({f"rho_{t}": v for t, v in sorted(inputs.densities().items())})
    key.update({f"c_{t}": v for t, v in sorted(TR.default_sound_speeds().items())})
    key["k_found"] = inputs.k_found
    key["n_cells"] = _n_cells(md)
    return json.dumps({k: f"{float(v):.{KEY_DIGITS}g}" for k, v in key.items()}, sort_keys=True)


def _n_cells(md) -> int:
    tdim = md.mesh.topology.dim
    return int(md.mesh.topology.index_map(tdim).size_local)


# ---------------------------------------------------------------------------
# One-at-a-time scenarios
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Scenario:
    """One-at-a-time scenario: ``inputs`` = baseline with ``input`` set to ``value``."""

    name: str
    key: str
    input: str | None
    value: Any
    basis: str
    inputs: ModelInputs


def _raw_or_none(key: str):
    try:
        return P.raw(key)
    except (KeyError, TypeError):
        return None


def _sens_values(key: str) -> tuple[Any, list] | None:
    """(baseline value, sensitivity values) of a config entry with ``sens``; ``None`` without."""
    node = _raw_or_none(key)
    if not isinstance(node, dict) or "sens" not in node:
        return None
    if isinstance(node.get("value"), str):              # option entries (crack filler, driving force)
        return node["value"], [str(v) for v in node["sens"]]
    q = P.get(key)
    return q.value, list(q.sens)


def _same(a, b) -> bool:
    if isinstance(a, str) or isinstance(b, str):
        return str(a) == str(b)
    return math.isclose(float(a), float(b), rel_tol=10 ** -KEY_DIGITS)


def _fmt(v) -> str:
    return v if isinstance(v, str) else f"{float(v):g}"


def _decision(did: str | None) -> str:
    if did is None:
        return ""
    return f" {did}: {P.raw(f'decisions.{did}')}"


def oat_scenarios() -> list[Scenario]:
    """Baseline plus one scenario per ``sens`` value of the :data:`OAT_ENTRIES` config entries
    (values equal to the baseline skipped) and the D5 stress-controlled failure criterion."""
    base = ModelInputs.baseline()
    out = [Scenario(BASELINE, BASELINE, None, None, "baseline inputs (config defaults, as the cohort runs)", base)]
    for key, inp, did in OAT_ENTRIES:
        sv = _sens_values(key)
        if sv is None:
            log.info("config entry %s has no 'sens': no scenario", key)
            continue
        ref, values = sv
        node = P.raw(key)
        basis = f"config {key} sens (baseline {_fmt(ref)}): {node.get('basis', node.get('lit', ''))}" + _decision(did)
        for v in values:
            if _same(v, ref):
                continue
            out.append(Scenario(f"{key}={_fmt(v)}", key, inp, v, basis, base.set(**{inp: v})))
    for mode in FAILURE_MODE_SENS:
        if mode == base.ms.failure_mode:
            continue
        basis = (f"stress-controlled calcium failure, sigma_c = fracture.sigma_c_stress_controlled "
                 f"({base.ms.sigma_c:g} MPa)" + _decision(FAILURE_MODE_DECISION))
        out.append(Scenario(f"failure_mode={mode}", "failure_mode", "failure_mode", mode, basis,
                            base.set(failure_mode=mode)))
    names = [s.name for s in out]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate scenario names: {names}")
    return out


def n_oat_runs(lesions=None) -> int:
    """Number of one-at-a-time runs (lesions x scenarios)."""
    from dtivl.drivers.stage_pilot import pilot_ids

    return len(list(lesions or pilot_ids())) * len(oat_scenarios())


# ---------------------------------------------------------------------------
# Morris inputs and design
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MorrisInput:
    """One Morris factor: sampled variable ``name`` over ``bounds`` (``scale`` linear or log10),
    mapped to the model input ``input``."""

    name: str
    input: str
    bounds: tuple[float, float]
    scale: str
    basis: str

    def value(self, x: float) -> float:
        return float(10 ** x) if self.scale == "log10" else float(x)


def _span(key: str) -> tuple[float, float]:
    """(min, max) over the baseline value and the ``sens`` values of a config entry."""
    q = P.get(key)
    vals = (q.value,) + tuple(q.sens)
    if len(vals) < 2:
        raise ValueError(f"config entry {key} has no sens values")
    return min(vals), max(vals)


def sweep_neighbours() -> tuple[float, float]:
    """The two pilot-sweep calcium moduli adjacent to the D8 baseline (``stage_pilot.e_sweep()``)."""
    from dtivl.drivers.stage_pilot import e_sweep

    pts = e_sweep()
    base = MaterialSet.baseline().ca_E
    i = min(range(len(pts)), key=lambda j: abs(math.log10(pts[j]) - math.log10(base)))
    if not math.isclose(pts[i], base, rel_tol=SWEEP_MATCH_RTOL):
        raise ValueError(f"baseline modulus {base} not in the pilot sweep {pts}")
    if i == 0 or i == len(pts) - 1:
        raise ValueError(f"baseline modulus {base} is an end point of the pilot sweep {pts}")
    return pts[i - 1], pts[i + 1]


def morris_inputs() -> list[MorrisInput]:
    """The Morris factors with ranges from the config ``sens`` entries or literature ranges."""
    e_lo, e_hi = sweep_neighbours()
    st_lo = P.lit("calcification_fracture_stretch", "lower").value
    st_hi = P.lit("calcification_fracture_stretch", "upper").value
    fib = [P.get("materials.fibrous.E").value] + [q.value for q in P.get_list("materials.fibrous.sens_E")]
    k_lo, k_hi = _span("loads.perivascular_k_MPa_per_mm")
    return [
        MorrisInput("log10_ca_E", "ca_E", (math.log10(e_lo), math.log10(e_hi)), "log10",
                    f"log10 of the calcium modulus (MPa) between the two pilot-sweep moduli adjacent to the D8 "
                    f"baseline {MaterialSet.baseline().ca_E:g} MPa (dtivl.drivers.stage_pilot.e_sweep(): "
                    f"{e_lo:g} and {e_hi:g} MPa)"),
        MorrisInput("eps_f", "eps_f", (st_lo - 1, st_hi - 1), "linear",
                    "failure strain = literature range of calcification_fracture_stretch minus 1 "
                    f"({P.load_literature()['calcification_fracture_stretch'].range_str})"),
        MorrisInput("ivl_pulse_scale", "pulse_scale", _span("loads.ivl_pulse_scale"), "linear",
                    "loads.ivl_pulse_scale: baseline and sens values"),
        MorrisInput("ell_over_h", "ell_over_h", _span("fracture.ell_over_h"), "linear",
                    "fracture.ell_over_h: baseline and sens values"),
        MorrisInput("log10_k_perivascular", "k_found", (math.log10(k_lo), math.log10(k_hi)), "log10",
                    "log10 of loads.perivascular_k_MPa_per_mm (MPa/mm): baseline and sens values"),
        MorrisInput("rho_calcium", "rho_calcium", _span("materials.density.calcium"), "linear",
                    "materials.density.calcium (kg/m^3, D15): baseline and sens values (the two model-input values "
                    "of density_calcified_plaque_model_input); enters T_equivalent only"),
        MorrisInput("k_over_mu_soft", "k_over_mu", _span("materials.bulk_to_shear_soft"), "linear",
                    "materials.bulk_to_shear_soft (K/mu of the soft tissues): baseline and sens values"),
        MorrisInput("fibrous_E", "fibrous_E", (min(fib), max(fib)), "linear",
                    "materials.fibrous.E (MPa): baseline and sens_E values"),
    ]


def morris_config() -> dict:
    """``design.sensitivity`` with the resolved seed; the lesions must be core lesions of the cohort."""
    from dtivl.drivers.stage_cohort_runs import cohort_ids

    c = P.raw("design.sensitivity")
    lesions = [str(x) for x in c["morris_lesions"]]
    core = set(cohort_ids(("core",)))
    missing = [x for x in lesions if x not in core]
    if missing:
        raise ValueError(f"design.sensitivity.morris_lesions not in the core set: {missing}")
    pool = c.get("morris_trajectory_pool")
    r = int(c["morris_trajectories"])
    if pool is not None and int(pool) <= r:
        raise ValueError("design.sensitivity.morris_trajectory_pool must exceed morris_trajectories")
    return {"lesions": lesions, "r": r, "pool": None if pool is None else int(pool),
            "levels": int(c["morris_levels"]), "conf_resamples": int(c["morris_conf_resamples"]),
            "conf_level": float(c["morris_conf_level"]),
            "seed": int(P.raw("design.seed")) + int(c["seed_offset"])}


def morris_problem(inputs: list[MorrisInput] | None = None) -> dict:
    """SALib problem definition."""
    inputs = morris_inputs() if inputs is None else inputs
    return {"num_vars": len(inputs), "names": [m.name for m in inputs], "bounds": [list(m.bounds) for m in inputs]}


def morris_samples(seed: int | None = None) -> np.ndarray:
    """Morris sample matrix (``r (k + 1)`` rows, one column per :func:`morris_inputs` factor),
    deterministic for a given seed (default from :func:`morris_config`)."""
    from SALib.sample import morris as SM

    cfg = morris_config()
    seed = cfg["seed"] if seed is None else int(seed)
    problem = morris_problem()
    if cfg["pool"] is None:
        return SM.sample(problem, cfg["r"], num_levels=cfg["levels"], optimal_trajectories=None,
                         local_optimization=False, seed=seed)
    return SM.sample(problem, cfg["pool"], num_levels=cfg["levels"], optimal_trajectories=cfg["r"],
                     local_optimization=True, seed=seed)


def morris_point(x, inputs: list[MorrisInput] | None = None, base: ModelInputs | None = None) -> ModelInputs:
    """Model inputs of one Morris sample row (baseline with every factor set)."""
    inputs = morris_inputs() if inputs is None else inputs
    base = ModelInputs.baseline() if base is None else base
    if len(x) != len(inputs):
        raise ValueError(f"sample has {len(x)} values, {len(inputs)} factors")
    return base.set(**{m.input: m.value(v) for m, v in zip(inputs, x)})


def n_morris_runs() -> int:
    """Number of Morris model runs: r (k + 1) x lesions."""
    cfg = morris_config()
    return cfg["r"] * (len(morris_inputs()) + 1) * len(cfg["lesions"])


def morris_design() -> dict:
    """Problem, factor definitions, sampling settings and the sample matrix (JSON-ready)."""
    cfg = morris_config()
    inputs = morris_inputs()
    X = morris_samples()
    return {"method": "Morris elementary effects (SALib 1.6: SALib.sample.morris, SALib.analyze.morris)",
            "problem": morris_problem(inputs),
            "inputs": [{"name": m.name, "input": m.input, "bounds": list(m.bounds), "scale": m.scale,
                        "basis": m.basis} for m in inputs],
            "outputs": MORRIS_OUTPUTS, "lesions": cfg["lesions"], "r": cfg["r"], "pool": cfg["pool"],
            "levels": cfg["levels"], "seed": cfg["seed"], "n_runs": cfg["r"] * (len(inputs) + 1) * len(cfg["lesions"]),
            "basis": P.raw("design.sensitivity")["basis"], "X": X.tolist()}


# ---------------------------------------------------------------------------
# Running the outcome model
# ---------------------------------------------------------------------------
_MESH_CACHE: dict = {}


def lesion_mesh(lid: str):
    """(plm, mesh info, MeshData) of a cohort lesion, meshed once per process into a per-process file
    (parallel workers and the cohort runs never write the same file)."""
    from dtivl.meshing.mask_to_mesh import load_mesh, mesh_from_labelmap

    if lid not in _MESH_CACHE:
        plm = load_mask(lid)
        path = MESH_DIR / "sensitivity" / f"{lid}.{os.getpid()}.msh"
        info = mesh_from_labelmap(plm, path)
        md = load_mesh(info["msh"], gdim=2)
        path.unlink(missing_ok=True)
        _MESH_CACHE[lid] = (plm, info, md)
    return _MESH_CACHE[lid]


def _run_protocol(fn, md, plm) -> dict:
    """Run one protocol (``fn`` returns a ProtocolResult); an exception is recorded as status 'error'.
    The record adds the load steps around the damage onset (``onset_bracket``, :func:`onset_bracket`)."""
    from dtivl.drivers.stage_pilot import _protocol_record

    with Timer() as t:
        try:
            res = fn()
            rec = _protocol_record(res, md, plm)
            rec["onset_bracket"] = onset_bracket(res.state.history)    # same history as cracks.onset_load
        except Exception as exc:              # recorded; the other protocol still runs
            log.exception("protocol failed: %s", exc)
            rec = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    rec["seconds"] = t.seconds
    gc.collect()
    return rec


def run_model(md, plm, inputs: ModelInputs, *, T: float | None = None, T_source: str | None = None,
              n_steps: dict | None = None, h_ca: float | None = None, transient_kwargs: dict | None = None) -> dict:
    """One outcome-model run: ``T_equivalent`` (unless ``T`` is given), IVL and balloon alone.

    ``plm`` is the lesion's PolarLabelMap (crack analysis; ``transfer_factor`` takes the calcium cells
    from the mesh tags). ``n_steps``: optional ``{"ivl": (n1, n2, n3), "pressure": n}`` increments
    (default: the solver defaults, as in the cohort runs). ``h_ca``: calcium mesh size for ``ell``
    (default ``discretisation.h_calcium_mm``, the size of the cohort meshes). ``transient_kwargs``
    go to ``transfer_factor`` (tests only). Returns the run record with :func:`outputs`.
    """
    from dtivl.solvers.nonlinear import NonlinearState, run_ivl, run_pressure
    from dtivl.solvers.transient import transfer_factor

    n_steps = n_steps or {}
    rec = {"inputs": inputs.describe(h_ca)}
    key = transient_key(inputs, md)
    with Timer() as t_all:
        if T is None:
            with Timer() as t:
                tf = transfer_factor(md, plm, inputs.ms, densities=inputs.densities(), k_found=inputs.k_found,
                                     **(transient_kwargs or {}))
            rec["transient"] = {**{k: tf[k] for k in TRANSIENT_KEYS}, "source": "computed", "key": key,
                                "seconds": t.seconds}
        else:
            rec["transient"] = {"T_equivalent": float(T), "source": T_source or "given", "key": key}
        T = rec["transient"]["T_equivalent"]
        p_bal = P.get("loads.ivl_balloon_pressure_atm").value
        R_ivl = P.get("geometry.ivl_balloon_diameter_mm").value / 2     # radius of the IVL balloon cap
        kw = inputs.state_kwargs(h_ca)
        rec["ivl"] = _run_protocol(lambda: run_ivl(NonlinearState(md, inputs.ms, name="sens_ivl", **kw), T=T,
                                                   pulse_scale=inputs.pulse_scale, n_steps=n_steps.get("ivl")),
                                   md, plm)
        rec["balloon_only"] = _run_protocol(
            lambda: run_pressure(NonlinearState(md, inputs.ms, name="sens_bal", **kw), p_bal, cap_radius=R_ivl,
                                 phase="balloon_only", n_steps=n_steps.get("pressure")), md, plm)
    rec["seconds"] = t_all.seconds
    rec["outputs"] = outputs(rec)
    return rec


# ---------------------------------------------------------------------------
# Outputs and censoring
# ---------------------------------------------------------------------------
def ivl_reached_peak(prec: dict) -> bool:
    """True if the IVL protocol reached its peak load p_top (status ok, or a failure after the pulse)."""
    status = prec.get("status")
    if status == "ok":
        return True
    return status == "failed" and (prec.get("summary") or {}).get("failed_phase") not in IVL_LOADING_PHASES


def ivl_failed_after_peak(prec: dict) -> bool:
    """True if the IVL protocol failed after reaching p_top (while unloading or at the release step):
    a fracture or onset not found by then could still appear in the phases that never ran."""
    return prec.get("status") == "failed" and ivl_reached_peak(prec)


def fracture_flag(prec: dict, decided: bool) -> bool | None:
    """Through-thickness damage of a protocol: True once found (damage is irreversible), False only if
    the protocol ``decided`` it (ran to its end, status ok), else None (unknown)."""
    f = (prec.get("cracks") or {}).get("fracture")
    if f:
        return True
    if f is not None and decided:
        return False
    return None


def onset_bracket(history: list, d_crack: float | None = None) -> dict:
    """Load steps around the damage onset of a protocol history (rule of ``cracks.onset_load``: first
    step record with ``max_d >= fracture.d_crack``).

    ``load`` and ``phase`` of the onset step, ``lower`` = load of the last step before onset (for the
    first step of the history: its load minus its increment, the start of the ramp) and ``step`` =
    ``load - lower`` (the onset load is known to this resolution); all None without onset.
    """
    d_crack = float(P.raw("fracture.d_crack")) if d_crack is None else float(d_crack)
    i = next((j for j, h in enumerate(history) if h.get("max_d", 0) >= d_crack), None)
    if i is None:
        return {"load": None, "phase": None, "lower": None, "step": None}
    on = float(history[i]["load"])
    lower = float(history[i - 1]["load"]) if i > 0 else on - float(history[i].get("increment") or 0)
    return {"load": on, "phase": history[i].get("phase"), "lower": lower, "step": on - lower}


def censored_onset(prec: dict) -> dict:
    """IVL damage-onset load censored at the IVL peak load ``p_top``.

    * onset (``cracks.onset_load``) during inflation or pulse: ``value`` = onset load, ``censored`` False;
    * onset reached only while unloading: ``value`` = ``p_top``, ``censored`` False, ``after_peak`` True;
    * no onset and the protocol ran to its end (status ok): ``value`` = ``p_top``, ``censored`` True;
    * otherwise (error, or a failure before or after the peak without onset: the onset could still come
      in the phases that never ran): ``value`` None (unknown).
    """
    c = prec.get("cracks") or {}
    p_top = (prec.get("summary") or {}).get("p_top")
    onset, phase = c.get("onset_load"), c.get("onset_phase")
    if onset is not None and phase in IVL_LOADING_PHASES:
        return {"value": float(onset), "censored": False, "after_peak": False}
    if onset is not None and p_top is not None:
        return {"value": float(p_top), "censored": False, "after_peak": True}
    if p_top is not None and prec.get("status") == "ok":
        return {"value": float(p_top), "censored": True, "after_peak": False}
    return {"value": None, "censored": None, "after_peak": None}


def outputs(rec: dict) -> dict:
    """Per-run outputs of the outcome model (flat dict).

    ``ivl_onset_load_uncensored``: ``ivl_onset_load`` where the onset is observed during inflation or
    pulse (neither censored nor after the peak), else None. ``ivl_onset_load_lower`` /
    ``ivl_onset_step``: load of the step before that onset and the step size (:func:`onset_bracket`),
    None otherwise. ``ivl_failed_after_peak``: the IVL protocol failed after its peak load
    (:func:`ivl_failed_after_peak`); a fracture or onset not found by then is None.
    """
    ivl, bal = rec.get("ivl") or {}, rec.get("balloon_only") or {}
    ci, cb = ivl.get("cracks") or {}, bal.get("cracks") or {}
    on = censored_onset(ivl)
    observed = on["censored"] is False and on["after_peak"] is False
    br = ivl.get("onset_bracket") or {}
    if br.get("load") is None or br.get("load") != ci.get("onset_load"):
        br = {}
    return {
        "ivl_fracture": fracture_flag(ivl, ivl.get("status") == "ok"),
        "balloon_fracture": fracture_flag(bal, bal.get("status") == "ok"),
        "ivl_onset_load": on["value"], "ivl_onset_censored": on["censored"], "ivl_onset_after_peak": on["after_peak"],
        "ivl_onset_load_uncensored": on["value"] if observed else None,
        "ivl_onset_load_lower": br.get("lower") if observed else None,
        "ivl_onset_step": br.get("step") if observed else None,
        "ivl_failed_after_peak": ivl_failed_after_peak(ivl),
        "ivl_onset_load_raw": ci.get("onset_load"), "ivl_onset_phase": ci.get("onset_phase"),
        "balloon_onset_load": cb.get("onset_load"),
        "ivl_damaged_fraction": ci.get("damaged_fraction"), "balloon_damaged_fraction": cb.get("damaged_fraction"),
        "T_equivalent": (rec.get("transient") or {}).get("T_equivalent"),
        "ivl_p_top": (ivl.get("summary") or {}).get("p_top"),
        "ivl_status": ivl.get("status"), "balloon_status": bal.get("status"),
        "ivl_unconverged_steps": ivl.get("n_steps_unconverged_stagger"),
        "balloon_unconverged_steps": bal.get("n_steps_unconverged_stagger"),
        "ivl_seconds": ivl.get("seconds"), "balloon_seconds": bal.get("seconds"), "seconds": rec.get("seconds"),
    }


# ---------------------------------------------------------------------------
# Files, claims and transient reuse
# ---------------------------------------------------------------------------
def _write_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(_to_jsonable(data), indent=1))
    tmp.replace(path)


def _read(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _complete(path: Path) -> bool:
    rec = _read(path)
    return bool(rec and rec.get("complete"))


def _claim(lock: Path) -> bool:
    """Atomically claim a run for this process (lock file with the pid); locks of dead processes are
    taken over (as ``stage_cohort_runs._claim``, for any lock path)."""
    lock.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return True
        except FileExistsError:
            try:
                pid = int(lock.read_text().strip() or 0)
                if pid <= 0:
                    if time.time() - lock.stat().st_mtime < EMPTY_LOCK_GRACE_S:
                        return False          # being claimed right now (pid not yet written)
                    raise ValueError("empty lock")
                os.kill(pid, 0)
                return False                  # held by a live process
            except PermissionError:
                return False                  # live process of another user
            except (ValueError, ProcessLookupError, FileNotFoundError):
                lock.unlink(missing_ok=True)  # stale lock: retry once
    return False


def _release(lock: Path) -> None:
    lock.unlink(missing_ok=True)


def _cached_T(run_dir: Path, lid: str, key: str) -> tuple[float | None, str | None]:
    """T_equivalent of an earlier run of the same lesion with the same transient inputs (or None)."""
    for path in sorted(run_dir.glob("*.json")):
        rec = _read(path)
        if not rec or rec.get("lesion") != lid:
            continue
        tr = rec.get("transient") or {}
        if tr.get("source") == "computed" and tr.get("key") == key and tr.get("T_equivalent") is not None:
            return float(tr["T_equivalent"]), f"reused from {path.stem}"
    return None, None


def _do_run(path: Path, lock: Path, head: dict, inputs: ModelInputs, *, resume: bool, n_steps=None,
            h_ca: float | None = None) -> bool:
    """Claim, run, save one (lesion, scenario/sample); returns True if run here. The record is complete
    only if no protocol raised (status 'error'); otherwise it keeps the error and resume retries it."""
    if resume and _complete(path):
        return False
    if not _claim(lock):
        return False
    try:
        if resume and _complete(path):       # finished by another worker meanwhile
            return False
        lid = head["lesion"]
        rec = dict(head)
        try:
            plm, info, md = lesion_mesh(lid)
            rec["mesh"] = {k: info[k] for k in ("n_nodes", "n_cells", "n_cells_per_tag")}
            T, src = _cached_T(path.parent, lid, transient_key(inputs, md))
            rec.update(run_model(md, plm, inputs, T=T, T_source=src, n_steps=n_steps, h_ca=h_ca))
            errors = [f"{p}: {(rec.get(p) or {}).get('error')}" for p in PROTOCOLS
                      if (rec.get(p) or {}).get("status") == "error"]
            if errors:                        # recorded; resume retries the run
                rec["error"] = "protocol error: " + "; ".join(errors)
            else:
                rec["complete"] = True
        except Exception as exc:              # recorded; resume retries the run
            log.exception("%s failed: %s", path.stem, exc)
            rec["error"] = f"{type(exc).__name__}: {exc}"
        _write_atomic(path, rec)
        return True
    finally:
        _release(lock)


# ---------------------------------------------------------------------------
# One-at-a-time driver
# ---------------------------------------------------------------------------
def _oat_stem(lid: str, scenario: str) -> str:
    return f"{lid}__{scenario}"


def one_at_a_time(lesions=None, scenarios=None, *, resume: bool = True, out_dir: str | Path | None = None,
                  n_steps: dict | None = None, h_ca: float | None = None) -> dict:
    """Run the one-at-a-time scenarios on ``lesions`` (default: the pilot lesions).

    ``scenarios``: optional list of scenario names (default: all of :func:`oat_scenarios`). Runs
    already complete are skipped (``resume``); runs claimed by another live worker are left to it.
    Writes ``<out_dir>/sensitivity_oat_runs/<lesion>__<scenario>.json`` per run and
    ``<out_dir>/sensitivity_oat.json`` (:func:`collect_oat`) after every run. ``out_dir`` defaults
    to ``results/``. ``n_steps`` and ``h_ca``: see :func:`run_model` (defaults as the cohort runs).
    """
    from dtivl.drivers.stage_pilot import pilot_ids

    out_dir = Path(out_dir) if out_dir is not None else RESULTS_DIR
    run_dir = out_dir / OAT_RUNS
    lesions = list(lesions or pilot_ids())
    all_sc = oat_scenarios()
    if scenarios is not None:
        unknown = set(scenarios) - {s.name for s in all_sc}
        if unknown:
            raise KeyError(f"unknown scenarios {sorted(unknown)}; known: {[s.name for s in all_sc]}")
    chosen = [s for s in all_sc if scenarios is None or s.name in scenarios]
    for lid in lesions:
        for sc in chosen:
            stem = _oat_stem(lid, sc.name)
            head = {"lesion": lid, "scenario": sc.name, "key": sc.key, "input": sc.input, "value": sc.value,
                    "basis": sc.basis}
            if _do_run(run_dir / f"{stem}.json", run_dir / "locks" / f"{stem}.lock", head, sc.inputs,
                       resume=resume, n_steps=n_steps, h_ca=h_ca):
                collect_oat(out_dir)
    return collect_oat(out_dir)


def _oat_row(rec: dict) -> dict:
    o = rec.get("outputs") or {}
    tr = rec.get("transient") or {}
    return {"lesion": rec.get("lesion"), "scenario": rec.get("scenario"), "key": rec.get("key"),
            "value": rec.get("value"), "complete": bool(rec.get("complete")), "error": rec.get("error"),
            "T_source": tr.get("source"), **o}


def _onset_observed(row: dict) -> bool:
    """IVL onset observed during inflation or pulse (neither censored nor after the peak)."""
    return row.get("ivl_onset_censored") is False and row.get("ivl_onset_after_peak") is False


def collect_oat(out_dir: str | Path | None = None) -> dict:
    """Collect the one-at-a-time run files into ``<out_dir>/sensitivity_oat.json``.

    Per lesion and scenario the outputs of :func:`outputs`, and the change relative to the
    lesion's baseline run (fracture flags changed, onset-load difference) where both exist. The IVL
    onset-load difference needs an observed onset in both runs: if either is censored or after the
    peak (value p_top, which moves with the pulse scale and ``T_equivalent``), no difference is
    computed and ``ivl_onset_load_censored_vs_baseline`` is True.
    """
    from dtivl.drivers.stage_pilot import pilot_ids

    out_dir = Path(out_dir) if out_dir is not None else RESULTS_DIR
    run_dir = out_dir / OAT_RUNS
    scen = oat_scenarios()
    rows = [_oat_row(r) for r in (_read(p) for p in sorted(run_dir.glob("*.json"))) if r and "scenario" in r]
    base = {r["lesion"]: r for r in rows if r["scenario"] == BASELINE and r["complete"]}
    for r in rows:
        b = base.get(r["lesion"])
        if b is None or not r["complete"] or r["scenario"] == BASELINE:
            continue
        for k in ("ivl_fracture", "balloon_fracture"):
            if r.get(k) is not None and b.get(k) is not None:
                r[f"{k}_changed_vs_baseline"] = bool(r[k]) != bool(b[k])
        observed = _onset_observed(r) and _onset_observed(b)
        if r.get("ivl_onset_load") is not None and b.get("ivl_onset_load") is not None:
            r["ivl_onset_load_censored_vs_baseline"] = not observed
        for k in ("ivl_onset_load", "balloon_onset_load", "ivl_damaged_fraction", "balloon_damaged_fraction",
                  "T_equivalent"):
            if k == "ivl_onset_load" and not observed:
                continue
            if r.get(k) is not None and b.get(k) is not None:
                r[f"{k}_minus_baseline"] = r[k] - b[k]
    order = {s.name: i for i, s in enumerate(scen)}
    rows.sort(key=lambda r: (r["lesion"], order.get(r["scenario"], len(order)), r["scenario"]))
    lesions = sorted({r["lesion"] for r in rows} | set(pilot_ids()))
    out = {"stage": "sensitivity_oat",
           "protocol": "per (lesion, scenario): T_equivalent (transfer_factor, same inputs; reused when the transient "
                       "inputs are unchanged), run_ivl with T = T_equivalent, and the balloon alone "
                       "(run_pressure to loads.ivl_balloon_pressure_atm with the IVL balloon cap), fresh states",
           "outputs": "fracture = through-thickness damage (cracks.fracture); onset = cracks.onset_load (MPa); "
                      "ivl_onset_load censored at p_top without onset (ivl_onset_censored); "
                      "ivl_onset_load_minus_baseline only where both onsets are observed (otherwise "
                      "ivl_onset_load_censored_vs_baseline True); IVL fracture and onset unknown (null) if the "
                      "IVL protocol failed after the peak without them (ivl_failed_after_peak)",
           "lesions": lesions,
           "scenarios": [{"name": s.name, "key": s.key, "input": s.input, "value": s.value, "basis": s.basis}
                         for s in scen],
           "n_runs_planned": len(pilot_ids()) * len(scen),
           "n_runs_complete": sum(r["complete"] for r in rows),
           "n_runs_error": sum(1 for r in rows if r.get("error")),
           "runs": {}, "table": rows}
    for r in rows:
        out["runs"].setdefault(r["lesion"], {})[r["scenario"]] = r
    _write_atomic(out_dir / OAT_JSON, out)
    return out


# ---------------------------------------------------------------------------
# Morris driver and analysis
# ---------------------------------------------------------------------------
def _check_or_write_design(run_dir: Path, design: dict) -> None:
    path = run_dir / MORRIS_DESIGN
    old = _read(path)
    if old is None:
        _write_atomic(path, design)
        return
    same = (old.get("X") == json.loads(json.dumps(design["X"])) and old.get("problem") == design["problem"]
            and old.get("seed") == design["seed"])
    if not same:
        raise ValueError(f"the Morris design changed since {path} was written (config or code); move "
                         f"{run_dir} away to start a new design")
    if old.get("lesions") != design["lesions"]:
        _write_atomic(path, {**old, "lesions": design["lesions"], "n_runs": design["n_runs"]})


def _morris_stem(lid: str, i: int) -> str:
    return f"{lid}_{i:03d}"


def morris(lesions=None, *, resume: bool = True, out_dir: str | Path | None = None, n_steps: dict | None = None,
           h_ca: float | None = None, max_runs: int | None = None) -> list[str]:
    """Run the Morris samples (trajectory by trajectory, every lesion per trajectory, so that partial
    results hold complete trajectories) and, once every sample is complete, :func:`analyze`.

    ``lesions`` default ``design.sensitivity.morris_lesions``; ``max_runs`` stops after that many
    runs in this process; ``n_steps`` and ``h_ca``: see :func:`run_model`. Returns the file stems run here.
    """
    out_dir = Path(out_dir) if out_dir is not None else RESULTS_DIR
    run_dir = out_dir / MORRIS_RUNS
    design = morris_design()
    _check_or_write_design(run_dir, design)
    lesions = list(lesions or design["lesions"])
    inputs = morris_inputs()
    base = ModelInputs.baseline()
    X = np.asarray(design["X"], dtype=float)
    k = len(inputs)
    r = len(X) // (k + 1)
    done = []
    for j in range(r):
        for lid in lesions:
            for s in range(k + 1):
                i = j * (k + 1) + s
                stem = _morris_stem(lid, i)
                path = run_dir / f"{stem}.json"
                x = [float(v) for v in X[i]]
                old = _read(path)
                if old and old.get("x") is not None and old["x"] != x:
                    raise ValueError(f"{path}: stored sample differs from the design")
                head = {"lesion": lid, "index": i, "trajectory": j, "step": s, "x": x,
                        "factors": dict(zip(design["problem"]["names"], x)), "seed": design["seed"]}
                if _do_run(path, run_dir / "locks" / f"{stem}.lock", head, morris_point(x, inputs, base),
                           resume=resume, n_steps=n_steps, h_ca=h_ca):
                    done.append(stem)
                    if max_runs is not None and len(done) >= max_runs:
                        return done
    if all(_complete(run_dir / f"{_morris_stem(lid, i)}.json") for lid in design["lesions"] for i in range(len(X))):
        analyze(out_dir)
    return done


def _y(rec: dict | None, name: str) -> float:
    if not rec or not rec.get("complete"):
        return float("nan")
    v = (rec.get("outputs") or {}).get(name)
    return float("nan") if v is None else float(v)


def _flag(rec: dict | None, name: str) -> bool:
    return bool(rec and rec.get("complete") and (rec.get("outputs") or {}).get(name))


def _salib(problem: dict, X: np.ndarray, Y: np.ndarray, cfg: dict) -> dict:
    from SALib.analyze import morris as SA

    res = SA.analyze(problem, X, Y, num_resamples=cfg["conf_resamples"], conf_level=cfg["conf_level"],
                     num_levels=cfg["levels"], seed=cfg["seed"])
    return {name: {k: float(np.asarray(res[k], dtype=float)[n]) for k in ("mu", "mu_star", "sigma", "mu_star_conf")}
            for n, name in enumerate(problem["names"])}


def _flag_counts(names: list, X: np.ndarray, F: np.ndarray, ok: list, k: int) -> dict:
    """Over the trajectories ``ok``: flagged points (``F``: per-row bool), elementary effects with a flagged
    endpoint per factor (the factor changed by that step of the trajectory) and trajectories with every
    point flagged."""
    per, n_points, n_all = dict.fromkeys(names, 0), 0, 0
    for t in ok:
        rows = slice(t * (k + 1), (t + 1) * (k + 1))
        Ft = F[rows]
        moved = np.argmax(np.diff(X[rows], axis=0) != 0, axis=1)
        for s, c in enumerate(moved):
            per[names[c]] += int(Ft[s] or Ft[s + 1])
        n_points += int(Ft.sum())
        n_all += int(np.all(Ft))
    return {"n_points": n_points, "n_ee_with_flagged_endpoint": per, "n_trajectories_all_points_flagged": n_all}


def _analyze_set(problem, X, Y, k, cfg, flags: dict | None = None, step: np.ndarray | None = None,
                 step_name: str | None = None) -> dict:
    """Morris indices from the trajectories of (X, Y) without missing values.

    ``flags`` (flag name -> per-row bool array): :func:`_flag_counts` of each over these trajectories
    (``censoring``); ``step`` (per-row load step at onset, NaN without): its median over these
    trajectories (``median_<step_name>``, next to the indices).
    """
    n_t = len(Y) // (k + 1)
    ok = [t for t in range(n_t) if np.all(np.isfinite(Y[t * (k + 1):(t + 1) * (k + 1)]))]
    rows = np.concatenate([np.arange(t * (k + 1), (t + 1) * (k + 1)) for t in ok]) if ok else np.zeros(0, int)
    out = {"n_trajectories": len(ok), "excluded_trajectories": [t for t in range(n_t) if t not in ok]}
    if len(ok) < 2:
        out["indices"] = None
        out["note"] = "fewer than 2 complete trajectories"
    else:
        out["indices"] = _salib(problem, X[rows], Y[rows], cfg)
    if step is not None:
        st = step[rows]
        st = st[np.isfinite(st)]
        out[f"median_{step_name}"] = float(np.median(st)) if st.size else None
        out[f"n_{step_name}"] = int(st.size)
    if flags:
        out["censoring"] = {f: _flag_counts(problem["names"], X, F, ok, k) for f, F in flags.items()}
    return out


def _series(rs: list, name: str) -> dict:
    """Keyword arguments of :func:`_analyze_set` for output ``name`` over the sample records ``rs``: ``Y``,
    the per-row censoring flags (:data:`CENSORING_FLAGS`, and :data:`AT_P_TOP` = any of them) and load
    step at onset (:data:`ONSET_STEPS`)."""
    kw = {"Y": np.array([_y(r, name) for r in rs])}
    if name in CENSORING_FLAGS:
        kw["flags"] = {f: np.array([_flag(r, f) for r in rs], dtype=bool) for f in CENSORING_FLAGS[name]}
        kw["flags"][AT_P_TOP] = np.any(list(kw["flags"].values()), axis=0)
    if name in ONSET_STEPS:
        kw["step"] = np.array([_y(r, ONSET_STEPS[name]) for r in rs])
        kw["step_name"] = ONSET_STEPS[name]
    return kw


def _morris_note(r: int, n_lesions: int) -> str:
    return ("elementary effects with inputs scaled to [0, 1] (SALib): mu_star in output units per normalised input "
            "range; trajectories with a missing output value are excluded for that output (excluded_trajectories). "
            "Missing: a run that is not complete (a protocol error; resume retries it) or an outcome its protocol "
            "did not decide: ivl_fracture without fracture and ivl_onset_load without onset if the IVL protocol "
            "failed, before the peak or after it (n_ivl_failed_after_peak counts the failures after the peak), "
            "since they could first appear in the phases that never ran. ivl_onset_load: a run without onset "
            "enters censored at p_top = p_bal + T_equivalent x p_peak x pulse_scale, which moves with "
            "ivl_pulse_scale and with every input of T_equivalent, so these inputs can get a nonzero mu_star on a "
            "lesion that never reaches onset; 'censoring' gives per flag (ivl_onset_censored; "
            "ivl_onset_after_peak = onset only while unloading, value p_top; at_p_top = either) the number of "
            "flagged points, per input the number of elementary effects with a flagged endpoint and the number of "
            "trajectories with every point flagged; ivl_onset_load_uncensored sets censored and after-peak runs "
            "to missing, so only trajectories without such a point enter. Onset loads are known to the load step "
            "of the IVL ramps (ivl_onset_step = onset load minus ivl_onset_load_lower, the load of the step "
            "before; it scales with pulse_scale and T_equivalent): median_ivl_onset_step is given next to the "
            f"indices. Pooled indices concatenate the same {r} trajectories (one sample matrix X shared by the "
            f"{n_lesions} lesions) once per lesion; SALib's bootstrap treats the pooled elementary effects as "
            f"independent, so the pooled mu_star_conf is conditional on these {r} trajectories, not a confidence "
            f"interval over designs; per-lesion mu_star_conf rests on at most {r} elementary effects per input, "
            f"and with r = {r} mu_star_conf is unstable.")


def analyze(out_dir: str | Path | None = None) -> dict:
    """Morris indices (``mu``, ``mu_star``, ``sigma``, ``mu_star_conf``) per input, output and lesion,
    and pooled over lesions (trajectories of all lesions concatenated), from the sample files;
    writes ``<out_dir>/sensitivity_morris.json``. Uses whatever samples are complete. For the onset
    loads also the censoring counts per input and the median load step (:func:`_analyze_set`)."""
    out_dir = Path(out_dir) if out_dir is not None else RESULTS_DIR
    run_dir = out_dir / MORRIS_RUNS
    design = _read(run_dir / MORRIS_DESIGN)
    if design is None:
        raise FileNotFoundError(f"{run_dir / MORRIS_DESIGN} not found; run morris() first")
    cfg = {**morris_config(), "seed": design["seed"], "levels": design["levels"]}
    problem = design["problem"]
    X = np.asarray(design["X"], dtype=float)
    k = problem["num_vars"]
    lesions = design["lesions"]
    recs = {lid: [_read(run_dir / f"{_morris_stem(lid, i)}.json") for i in range(len(X))] for lid in lesions}
    out = {k_: design[k_] for k_ in ("method", "problem", "inputs", "outputs", "lesions", "r", "pool", "levels",
                                     "seed", "n_runs", "basis")}
    out["outputs"] = MORRIS_OUTPUTS               # the outputs analysed here (design.json keeps its first run's)
    out["conf_resamples"], out["conf_level"] = cfg["conf_resamples"], cfg["conf_level"]
    out["note"] = _morris_note(len(X) // (k + 1), len(lesions))
    out["n_runs_complete"] = sum(bool(r and r.get("complete")) for rs in recs.values() for r in rs)
    out["n_runs_error"] = sum(bool(r and not r.get("complete") and r.get("error")) for rs in recs.values() for r in rs)
    out["lesion_results"], out["pooled"] = {}, {}
    for lid, rs in recs.items():
        done = [r for r in rs if r and r.get("complete")]
        outs = [r.get("outputs") or {} for r in done]
        lres = {"n_complete": len(done),
                "n_ivl_onset_censored": sum(bool(o.get("ivl_onset_censored")) for o in outs),
                "n_ivl_onset_after_peak": sum(bool(o.get("ivl_onset_after_peak")) for o in outs),
                "n_ivl_failed_after_peak": sum(bool(o.get("ivl_failed_after_peak")) for o in outs),
                "n_status_not_ok": sum(1 for o in outs if o.get("ivl_status") != "ok" or o.get("balloon_status") != "ok"),
                "n_unconverged_runs": sum(1 for o in outs if (o.get("ivl_unconverged_steps") or 0)
                                          + (o.get("balloon_unconverged_steps") or 0) > 0),
                "n_cells": sorted({(r.get("mesh") or {}).get("n_cells") for r in done} - {None}),
                "outputs": {}}
        for name in MORRIS_OUTPUTS:
            lres["outputs"][name] = _analyze_set(problem, X, k=k, cfg=cfg, **_series(rs, name))
        out["lesion_results"][lid] = lres
    Xp = np.concatenate([X for _ in lesions])
    rs_p = [r for lid in lesions for r in recs[lid]]
    for name in MORRIS_OUTPUTS:
        out["pooled"][name] = _analyze_set(problem, Xp, k=k, cfg=cfg, **_series(rs_p, name))
    _write_atomic(out_dir / MORRIS_JSON, out)
    return out


def main(argv=None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("what", choices=("oat", "morris", "analyze", "collect"))
    ap.add_argument("--lesions", nargs="*", default=None)
    ap.add_argument("--scenarios", nargs="*", default=None, help="OAT scenario names (default all)")
    ap.add_argument("--max-runs", type=int, default=None, help="Morris: stop after this many runs")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    if a.what == "oat":
        one_at_a_time(a.lesions, a.scenarios)
    elif a.what == "morris":
        morris(a.lesions, max_runs=a.max_runs)
    elif a.what == "analyze":
        analyze()
    else:
        collect_oat()


if __name__ == "__main__":
    main()

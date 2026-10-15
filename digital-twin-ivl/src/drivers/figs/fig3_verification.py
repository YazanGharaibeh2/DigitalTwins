"""Figure 3: verification evidence, read from the verification and pilot results files.

(a) Linear Lame check (``results/verification_lame.json``): hoop stress at the sample radii, finite
    elements (angular mean) against the plane-strain Lame solution, which is drawn with
    :func:`dtivl.solvers.linear.lame_exact` from the case data of the file (a, b, p, lambda, mu) and
    checked against the exact values stored in the file.
(b) Same file: relative error of the radial displacement and the hoop stress per sample radius, with
    the file's tolerance (``tolerance_rel``).
(c) Mesh convergence (``results/verification_mesh_convergence.json``): each predictor at the default
    and the coarse mesh level relative to the finest level, with the acceptance band
    ``rel_change_accept``; ``mesh_dependent_quantities`` are marked as excluded.
(d) AT1 damage onset (``results/verification_phasefield.json``): tensile energy density and maximum
    damage along the loading steps, normalised by the theoretical onset of the file.
(e) Nonlinear checks (``results/verification_nonlinear.json``): hyperelastic Lame and rigid-cylinder
    contact errors as fractions of their acceptance limits (the limits are the constants of
    :mod:`dtivl.solvers.nonlinear` that set the ``passed`` flags of the file).
(f) IVL transfer factor ``T_equivalent`` against the calcium modulus for the pilot lesions
    (``results/pilot_linear_transient.json``), baseline modulus ``materials.calcium.E`` marked.
"""
from __future__ import annotations

import numpy as np

from dtivl import params as P
from dtivl.drivers.common import read_json
from dtivl.drivers.figstyle import GRID, SERIES, TEXT, TEXT_2, apply_style, save

NAME = "fig3_verification"
FIGSIZE = (7.1, 7.6)               # inches (max width 7.1 in)
MARKERS = ("o", "s", "^", "D")


def _titles(ax, letter: str, title: str) -> None:
    ax.set_title(f"({letter})", loc="left", fontsize=8, fontweight="bold", color=TEXT)
    ax.set_title(title, loc="center", fontsize=7, color=TEXT)


def _pct(x: float) -> str:
    return f"{100 * x:g} %"


# ---------------------------------------------------------------- (a), (b) linear Lame
def panel_lame_profile(ax, lame: dict) -> None:
    from dtivl.solvers.linear import lame_exact

    for i, c in enumerate(lame["cases"]):
        a, b, p = c["a_mm"], c["b_mm"], c["p_MPa"]
        rows = c["per_radius"]
        r = np.array([x["r_mm"] for x in rows])
        _, _, hoop = lame_exact(r, a, b, p, c["lambda_MPa"], c["mu_MPa"])
        if not np.allclose(hoop, [x["hoop_exact"] for x in rows], rtol=1e-9):
            raise ValueError(f"verification_lame.json: stored exact hoop stress of {c['material']} does not match "
                             "the Lame solution of its own case data")
        if i == 0:
            rr = np.linspace(a, b, 200)
            ax.plot(rr, lame_exact(rr, a, b, p, c["lambda_MPa"], c["mu_MPa"])[2] / p, color=TEXT_2, lw=1.0,
                    label="Lamé solution", zorder=2)
        # coincident values: first case small filled markers on top of larger open ones
        ax.plot(r, np.array([x["hoop_num_mean"] for x in rows]) / p, ls="none", marker=MARKERS[i],
                ms=3.0 if i == 0 else 6.0, mfc=SERIES[i] if i == 0 else "none", mec=SERIES[i], mew=0.9,
                color=SERIES[i], label=f"FE, {c['material']} (angular mean)", zorder=4 - i)
    ax.set_xlabel("Radius r (mm)")
    ax.set_ylabel("Hoop stress / lumen pressure (-)")
    ax.legend(frameon=False, loc="upper right")


def panel_lame_error(ax, lame: dict) -> None:
    k = 0
    for i, c in enumerate(lame["cases"]):
        rows = c["per_radius"]
        r = np.array([x["r_mm"] for x in rows])
        for q, (num, ex, ls, name) in enumerate((("ur_num_mean", "ur_exact", "-", "u_r"),
                                                  ("hoop_num_mean", "hoop_exact", "--", "σ_θθ"))):
            err = np.abs(np.array([x[num] for x in rows]) / np.array([x[ex] for x in rows]) - 1.0)
            ax.plot(r, err, color=SERIES[i], ls=ls, lw=1.0, marker=MARKERS[i], ms=3.5,
                    mfc="white" if q else SERIES[i], mec=SERIES[i], mew=0.8, label=f"{c['material']}, {name}")
            k += 1
    tol = lame["tolerance_rel"]
    ax.axhline(tol, color=TEXT, lw=0.9, ls=":")
    ax.text(0.02, tol * 1.15, f"tolerance {_pct(tol)}", fontsize=6, color=TEXT, va="bottom", ha="left",
            transform=ax.get_yaxis_transform())
    ax.set_yscale("log")
    lo = min(line.get_ydata().min() for line in ax.get_lines() if len(line.get_ydata()) > 2)
    ax.set_ylim(lo / 3, tol * 3)
    ax.set_xlabel("Radius r (mm)")
    ax.set_ylabel("Relative error of the angular mean (-)")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.86), ncol=2, columnspacing=0.8,
              handlelength=2.4)


# ---------------------------------------------------------------- (c) mesh convergence
def panel_mesh(ax, mc: dict):
    from matplotlib.lines import Line2D

    lesions = list(mc["lesions"])
    accept = mc["rel_change_accept"]
    excluded = set(mc.get("mesh_dependent_quantities", []))
    quants = list(next(iter(mc["lesions"].values()))["gci"])
    ax.axvspan(-100 * accept, 100 * accept, color=GRID, alpha=0.6, lw=0, zorder=0)
    ax.axvline(0, color=TEXT_2, lw=0.6, zorder=1)
    offs = np.linspace(-0.18, 0.18, len(lesions))
    for j, lid in enumerate(lesions):
        levels = sorted(mc["lesions"][lid]["levels"], key=lambda r: r["factor"])
        fine, rest = levels[0], levels[1:]
        for q_i, q in enumerate(quants):
            col = TEXT_2 if q in excluded else SERIES[j]
            for lv in rest:
                default = abs(lv["factor"] - 1.0) < 1e-12
                ax.plot(100 * (lv[q] / fine[q] - 1.0), q_i + offs[j], ls="none", marker=MARKERS[j], ms=3.8,
                        mfc=col if default else "white", mec=col, mew=0.8, zorder=3)
    ax.set_yticks(range(len(quants)))
    ax.set_yticklabels([f"{q} (excluded)" if q in excluded else q for q in quants])
    for lab, q in zip(ax.get_yticklabels(), quants):
        lab.set_color(TEXT_2 if q in excluded else TEXT)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"Change vs finest mesh (%); shaded: acceptance ±{_pct(accept)}")
    fac = sorted({lv["factor"] for L in mc["lesions"].values() for lv in L["levels"]})
    handles = [Line2D([], [], ls="none", marker=MARKERS[j], color=SERIES[j], ms=3.8, label=lid)
               for j, lid in enumerate(lesions)]
    handles += [Line2D([], [], ls="none", marker="o", mfc=TEXT_2, mec=TEXT_2, ms=3.8,
                       label="default mesh (h × 1)"),
                Line2D([], [], ls="none", marker="o", mfc="white", mec=TEXT_2, ms=3.8,
                       label=f"coarse mesh (h × {max(fac):g}); finest h × {min(fac):g}")]
    ax.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.4, -0.2), ncol=2,
              handletextpad=0.3, columnspacing=1.0, borderaxespad=0)


# ---------------------------------------------------------------- (d) phase field
def panel_phasefield(ax_top, ax_bot, pf: dict) -> None:
    x_lo, x_hi, y_lo = np.inf, -np.inf, np.inf
    for i, c in enumerate(pf["cases"]):
        st = c["steps"]
        x = np.array([s["strain"] for s in st]) / c["strain_onset_theory"]
        psi = np.array([s["psi_plus"] for s in st]) / c["psi_onset_theory_MPa"]
        d = np.array([s["d_max"] for s in st])
        # coincident normalised curves: first case small filled markers on top of larger open ones
        kw = dict(color=SERIES[i], marker=MARKERS[i], ms=2.6 if i == 0 else 5.0, lw=1.0 if i == 0 else 0.8,
                  mfc=SERIES[i] if i == 0 else "none", mec=SERIES[i], mew=0.8, zorder=4 - i)
        state = c["state"].split(",")[0].removeprefix(c["case"]).strip()
        ax_top.plot(x, psi, label=f"{c['case']}: {state}", **kw)
        ax_bot.plot(x, d, **kw)
        dx = np.diff(x)
        k = int(np.argmax(dx < 0.5 * dx[0]))          # first step of the refined increments near the onset
        x_lo, x_hi, y_lo = min(x_lo, x[k]), max(x_hi, x[-1]), min(y_lo, psi[k])
    pad = 0.04 * (x_hi - x_lo)
    ax_bot.set_xlim(x_lo - pad, x_hi + pad)      # the axis shows the refined steps; earlier steps have d = 0
    ax_top.set_ylim(y_lo - 0.1 * (1.0 - y_lo), 1.0 + 0.15 * (1.0 - y_lo))
    for ax in (ax_top, ax_bot):
        ax.axvline(1.0, color=TEXT, lw=0.8, ls=":", zorder=1)
    ax_top.axhline(1.0, color=TEXT, lw=0.8, ls=":", zorder=1)
    ax_top.text(0.02, 1.0, "theoretical onset ψ+ = ψ_crit", fontsize=5.5, color=TEXT, va="bottom",
                transform=ax_top.get_yaxis_transform())
    ax_top.set_ylabel("ψ+ / ψ_crit (-)")
    ax_top.legend(frameon=False, loc="lower right")
    ax_top.tick_params(labelbottom=False)
    ax_bot.set_ylabel("max d (-)")
    ax_bot.set_xlabel("Strain / theoretical onset strain (-)")


# ---------------------------------------------------------------- (e) nonlinear checks
def nonlinear_rows(nl: dict) -> list[tuple[str, float]]:
    """(label, error / acceptance limit) of the hyperelastic Lame and contact checks."""
    from dtivl.solvers.nonlinear import VERIF_CONTACT_LAME_TOL, VERIF_CONTACT_PEN_TOL, VERIF_LAME_TOL

    lm = nl["lame"]
    out = [(f"hyperelastic Lamé, u_r lumen\nlimit {_pct(VERIF_LAME_TOL)}", lm["max_rel_err_ur_lumen"] / VERIF_LAME_TOL),
           (f"hyperelastic Lamé, u_r outer\nlimit {_pct(VERIF_LAME_TOL)}", lm["max_rel_err_ur_outer"] / VERIF_LAME_TOL)]
    a = nl["contact"]["a_mm"]
    for r in nl["contact"]["rows"]:
        tag = f"contact R = {r['R_mm'] / a:.3g} a"
        out.append((f"{tag}, pressure vs Lamé\nlimit {_pct(VERIF_CONTACT_LAME_TOL)}",
                    r["rel_err_p_vs_lame"] / VERIF_CONTACT_LAME_TOL))
        out.append((f"{tag}, penetration\nlimit {1 + VERIF_CONTACT_PEN_TOL:g} × p/k_pen",
                    r["max_abs_R_minus_rho_mm"] / ((1 + VERIF_CONTACT_PEN_TOL) * r["penalty_penetration_mm"])))
    return out


def panel_nonlinear(ax, nl: dict) -> None:
    rows = nonlinear_rows(nl)
    y = np.arange(len(rows))
    vals = np.array([v for _, v in rows])
    ax.hlines(y, vals.min() / 3, vals, color=GRID, lw=1.0, zorder=1)
    for k, (lab, v) in enumerate(rows):
        i = 0 if k < 2 else 1
        ax.plot(v, k, ls="none", marker=MARKERS[i], ms=4.2, color=SERIES[i], zorder=3)
    ax.axvline(1.0, color=TEXT, lw=0.9, ls=":")
    ax.text(0.92, -0.6, "acceptance limit", fontsize=6, color=TEXT, ha="right", va="center")
    ax.set_xscale("log")
    ax.set_xlim(vals.min() / 3, 3.0)
    ax.set_yticks(y)
    ax.set_yticklabels([lab for lab, _ in rows], fontsize=5.5)
    ax.set_ylim(len(rows) - 0.5, -0.9)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Error / acceptance limit (-)")


# ---------------------------------------------------------------- (f) transfer factor
def panel_transfer(ax, lt: dict) -> None:
    for i, (lid, rec) in enumerate(lt["lesions"].items()):
        E = [r["ca_E_MPa"] for r in rec["by_E"]]
        T = [r["transient"]["T_equivalent"] for r in rec["by_E"]]
        ax.plot(E, T, color=SERIES[i], marker=MARKERS[i], ms=3.2, lw=1.0, label=lid)
    E0 = P.get("materials.calcium.E").value
    ax.axvline(E0, color=TEXT, lw=0.8, ls="--", zorder=1)
    ax.text(E0 / 1.2, 0.97, "baseline E_Ca (D8)", transform=ax.get_xaxis_transform(), fontsize=5.5, color=TEXT,
            va="top", ha="right")
    ax.set_xscale("log")
    ax.set_xlabel("Calcium modulus E_Ca (MPa)")
    ax.set_ylabel("IVL transfer factor T_equivalent (-)")
    ax.legend(frameon=False, loc="center left")


# ---------------------------------------------------------------- figure
def make() -> str:
    import matplotlib.pyplot as plt

    apply_style()
    lame = read_json("verification_lame")
    mc = read_json("verification_mesh_convergence")
    pf = read_json("verification_phasefield")
    nl = read_json("verification_nonlinear")
    lt = read_json("pilot_linear_transient")
    fig = plt.figure(figsize=FIGSIZE, layout="constrained")
    # side padding = the 0.1 in that the tight-bbox export adds, so the saved width stays <= FIGSIZE[0]
    fig.get_layout_engine().set(w_pad=0.1)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.25, 1.1])
    ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    sub = gs[1, 1].subgridspec(2, 1, height_ratios=[2.0, 1.0])
    ax_d1 = fig.add_subplot(sub[0])
    ax_d2 = fig.add_subplot(sub[1], sharex=ax_d1)
    ax_e, ax_f = fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])
    panel_lame_profile(ax_a, lame)
    panel_lame_error(ax_b, lame)
    panel_mesh(ax_c, mc)
    panel_phasefield(ax_d1, ax_d2, pf)
    panel_nonlinear(ax_e, nl)
    panel_transfer(ax_f, lt)
    _titles(ax_a, "a", "Linear elasticity: thick cylinder vs Lamé")
    _titles(ax_b, "b", "Linear elasticity: error per radius")
    _titles(ax_c, "c", "Mesh convergence of the predictors")
    _titles(ax_d1, "d", f"AT1 damage onset (tolerance {_pct(pf['tolerance_rel'])})")
    _titles(ax_e, "e", "Finite strain and contact checks")
    _titles(ax_f, "f", "IVL transfer factor, pilot lesions")
    return str(save(fig, NAME))

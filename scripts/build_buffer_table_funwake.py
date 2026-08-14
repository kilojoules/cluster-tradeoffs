"""FunWake-schedule buffer tables (K=2000, funwake_iter192), both wake models.

Same aggregation as build_buffer_table.py / build_buffer_table_tp.py but reads
the converged-schedule grids:
    analysis/buffer_table_funwake/a{a}_f{f}_d{d}/Nt50/results.json
    analysis/buffer_table_tp_funwake/a{a}_f{f}_d{d}/Nt50/results.json

Outputs (suffix _fw so the pixwake-era figures are not overwritten):
    paper_v3/figures/buffer_table_contour_fw.png
    paper_v3/figures/buffer_table_tp_contour_fw.png
    paper_v3/figures/buffer_table_decay_fw.png      -- bast + tp side by side
    paper_v3/tables/buffer_table_fw.tex
    paper_v3/tables/buffer_table_tp_fw.tex
    analysis/buffer_table_funwake/d_star_summary.json
    analysis/buffer_table_tp_funwake/d_star_summary.json
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator

D_M = 240.0
A_VALS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
F_VALS = [0.0, 0.25, 0.5, 0.75, 1.0]
D_VALS = [2, 5, 10, 15, 20, 30, 40]

SITES = [
    ("Horns Rev 1", 0.649, 0.384, "s", "cyan"),
    ("Lillgrund",   0.621, 0.544, "D", "lime"),
    ("Borssele",    0.728, 0.291, "^", "magenta"),
    ("Princess A.", 0.650, 0.182, "o", "yellow"),
    ("DEI",         0.637, 0.384, "P", "white"),
]

MODELS = {
    "bast": dict(dir="analysis/buffer_table_funwake",
                 taus=[0.1, 0.2, 0.3], tau_plot=0.2,
                 contour="paper_v3/figures/buffer_table_contour_fw.png",
                 table="paper_v3/tables/buffer_table_fw.tex",
                 title="Bastankhah (FunWake $K{=}2000$)"),
    "tp":   dict(dir="analysis/buffer_table_tp_funwake",
                 taus=[1.0, 2.0, 5.0], tau_plot=2.0,
                 contour="paper_v3/figures/buffer_table_tp_contour_fw.png",
                 table="paper_v3/tables/buffer_table_tp_fw.tex",
                 title="TurboPark (FunWake $K{=}2000$)"),
}


def load_peak(base):
    peak = np.full((len(A_VALS), len(F_VALS), len(D_VALS)), np.nan)
    for i, a in enumerate(A_VALS):
        for j, f in enumerate(F_VALS):
            for k, d in enumerate(D_VALS):
                p = Path(base) / f"a{a}_f{f}_d{d}/Nt50/results.json"
                if not p.exists():
                    continue
                data = json.load(open(p))
                g = np.array(data["regret_grid_gwh"])
                lib = data["liberal_aep_gwh"]
                peak[i, j, k] = 100 * g[0, :].max() / lib
    return peak


def invert(R, d, tau):
    R = np.asarray(R, dtype=float)
    d = np.asarray(d, dtype=float)
    valid = ~np.isnan(R)
    if valid.sum() < 2:
        return np.nan
    R = np.interp(d, d[valid], R[valid])
    R_mono = np.maximum.accumulate(R[::-1])[::-1]
    if R_mono[0] < tau:
        return d[0]
    if R_mono[-1] > tau:
        return np.inf
    for i in range(len(d) - 1):
        if R_mono[i] >= tau >= R_mono[i + 1]:
            r0, r1 = R_mono[i], R_mono[i + 1]
            return d[i] + (tau - r0) * (d[i + 1] - d[i]) / (r1 - r0)
    return np.nan


def fmt(v):
    if np.isnan(v):
        return "---"
    if np.isinf(v):
        return r"$>40$"
    return f"{v:.1f}"


results = {}
for key, cfg in MODELS.items():
    peak = load_peak(cfg["dir"])
    n_filled = int((~np.isnan(peak)).sum())
    print(f"[{key}] filled cells: {n_filled}/{peak.size}")

    d_star = {tau: np.full((len(A_VALS), len(F_VALS)), np.nan) for tau in cfg["taus"]}
    for i in range(len(A_VALS)):
        for j in range(len(F_VALS)):
            for tau in cfg["taus"]:
                d_star[tau][i, j] = invert(peak[i, j, :], D_VALS, tau)
    results[key] = dict(peak=peak, d_star=d_star)

    # contour
    tau = cfg["tau_plot"]
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 6))
    A_g, F_g = np.meshgrid(A_VALS, F_VALS, indexing="ij")
    # Censored cells (threshold never crossed within the swept range) are shown
    # as a hatched region, not painted with a placeholder value.
    Zc = d_star[tau]
    Z = np.ma.masked_invalid(np.where(np.isinf(Zc), np.nan, Zc))
    levels = [2, 5, 10, 15, 20, 30, 40]
    ax.contourf(A_g, F_g, np.isinf(Zc).astype(float), levels=[0.5, 1.5],
                colors="none", hatches=["////"])
    im = ax.contourf(A_g, F_g, Z, levels=levels, cmap="YlOrRd", extend="max")
    cs = ax.contour(A_g, F_g, Z, levels=levels, colors="k", linewidths=0.4, alpha=0.5)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.0f$D$")
    if np.isinf(Zc).any():
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(facecolor="white", edgecolor="k", hatch="////",
                                 label=f"$d^*$ censored ($>{max(D_VALS)}D$)")],
                  loc="lower left", fontsize=7.5, framealpha=0.95)
    fig.colorbar(im, ax=ax, label=fr"Required buffer $d^*$ ($D$) for $\tau={tau}\%$ AEP")
    for name, a, f, marker, color in SITES:
        ax.plot(a, f, marker, markerfacecolor=color, markeredgecolor="black",
                markersize=11, linewidth=0, zorder=10, label=name)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.95)
    ax.set_xlabel("$a$ (concentration)")
    ax.set_ylabel("$f$ (folding)")
    ax.set_title(fr"{cfg['title']}: $d^*(a, f; \tau={tau}\%$ AEP)")
    ax.set_xlim(0.25, 0.95)
    ax.set_ylim(-0.05, 1.05)
    fig.savefig(cfg["contour"], dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {cfg['contour']}")

    # LaTeX table
    lines = ["\\begin{tabular}{c|" + "c" * len(F_VALS) + "}", "\\toprule",
             "$a \\downarrow$, $f \\rightarrow$ & "
             + " & ".join(f"{f:.2f}" for f in F_VALS) + " \\\\", "\\midrule"]
    for i, a in enumerate(A_VALS):
        row = f"{a:.1f}"
        for j in range(len(F_VALS)):
            row += " & " + fmt(d_star[tau][i, j])
        lines.append(row + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    Path("paper_v3/tables").mkdir(exist_ok=True)
    with open(cfg["table"], "w") as f:
        f.write("\n".join(lines))
    print(f"Saved: {cfg['table']}")

    # console table + real sites
    print(f"[{key}] d* at tau={tau}% AEP:")
    print(f"{'a\\f':>5}", *[f"{f:>6.2f}" for f in F_VALS])
    for i, a in enumerate(A_VALS):
        row = [f"{a:>5.1f}"]
        for j in range(len(F_VALS)):
            v = d_star[tau][i, j]
            row.append("   --" if np.isnan(v) else (">40D " if np.isinf(v) else f"{v:>5.1f}"))
        print(*row)
    # Site read-out must NOT interpolate through censored cells. A cell marked
    # inf means the threshold was never crossed within the swept range (<=40D);
    # substituting a finite placeholder there manufactures a number. Report the
    # site as censored whenever any cell it interpolates from is censored.
    Zc = d_star[tau]
    finite = RegularGridInterpolator(
        (np.array(A_VALS), np.array(F_VALS)), np.where(np.isinf(Zc), np.nan, Zc),
        method="linear", bounds_error=False, fill_value=np.nan)
    censored_frac = RegularGridInterpolator(
        (np.array(A_VALS), np.array(F_VALS)), np.isinf(Zc).astype(float),
        method="linear", bounds_error=False, fill_value=np.nan)
    d_max = max(D_VALS)
    for name, a, f, _, _ in SITES:
        cf = float(censored_frac([[a, f]])[0])
        v = float(finite([[a, f]])[0])
        if cf > 0:
            note = ("all bracketing cells censored" if cf >= 1.0
                    else f"{100*cf:.0f}% of bracketing cells censored")
            print(f"  {name:<16} a={a:.3f} f={f:.3f}: d*({tau}%) > {d_max}D "
                  f"(> {d_max * D_M / 1000:.1f} km)  [{note}]")
        else:
            print(f"  {name:<16} a={a:.3f} f={f:.3f}: d*({tau}%) = {v:5.1f}D "
                  f"({v * D_M / 1000:.1f} km)")

    summary = {
        "schedule": "funwake_iter192", "k_inner": 2000,
        "A_VALS": A_VALS, "F_VALS": F_VALS, "D_VALS": D_VALS,
        "peak_pct": peak.tolist(),
        "d_star": {f"tau_{t}pct": d_star[t].tolist() for t in cfg["taus"]},
    }
    out = Path(cfg["dir"]) / "d_star_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2,
                  default=lambda x: None if (isinstance(x, float) and (np.isnan(x) or np.isinf(x))) else x)
    print(f"Saved: {out}")

# ---- decay figure: bast and tp side by side ----
fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True)
for ax, key in zip(axes, ["bast", "tp"]):
    peak = results[key]["peak"]
    for j, f in enumerate(F_VALS):
        color = plt.cm.viridis(j / max(len(F_VALS) - 1, 1))
        R_avg = np.nanmean(peak[:, j, :], axis=0)
        ax.plot(D_VALS, R_avg, "o-", color=color, label=f"$f$={f:.2f}",
                linewidth=2, markersize=6)
    for t in MODELS[key]["taus"]:
        ax.axhline(t, color="gray", ls=":", lw=0.8, alpha=0.6)
        ax.text(40, t, f"  $\\tau={t}\\%$", fontsize=8, color="gray", va="center")
    ax.set_xlabel("Buffer distance ($D$)")
    ax.set_ylabel("Peak regret (% of AEP)")
    ax.set_title(MODELS[key]["title"] + " (mean over $a$)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
out = "paper_v3/figures/buffer_table_decay_fw.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
print(f"Saved: {out}")

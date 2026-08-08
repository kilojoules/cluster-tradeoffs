"""Build wind-rose-conditional buffer lookup table.

For each (a, f) in the elliptical-rose grid, fit a smooth monotone decay
R(d) to peak-bearing regret over the 7 buffer distances, then invert at
thresholds τ ∈ {0.5, 1.0, 2.0}% AEP to obtain the required buffer d*.

Outputs:
    paper_v3/figures/buffer_table_contour.png  -- d*(a,f) contour at τ=1%, real sites overlaid
    paper_v3/figures/buffer_table_decay.png    -- per-case R(d) curves
    paper_v3/tables/buffer_table.tex            -- LaTeX table d*(a,f;1%)
    analysis/buffer_table/d_star_summary.json
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import PchipInterpolator

D_M = 240.0
A_VALS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
F_VALS = [0.0, 0.25, 0.5, 0.75, 1.0]
D_VALS = [2, 5, 10, 15, 20, 30, 40]
TAUS = [0.1, 0.2, 0.3]  # lower thresholds because Bastankhah realistic regret <0.5% AEP

# Real-site fits from main text Table
SITES = [
    ("Horns Rev 1",      0.649, 0.384),
    ("Lillgrund",        0.621, 0.544),
    ("Borssele",         0.728, 0.291),
    ("Princess Amalia",  0.650, 0.182),
    ("DEI",              0.637, 0.384),
]

# Load grid
peak = np.full((len(A_VALS), len(F_VALS), len(D_VALS)), np.nan)
for i, a in enumerate(A_VALS):
    for j, f in enumerate(F_VALS):
        for k, d in enumerate(D_VALS):
            p = Path(f"analysis/buffer_table/a{a}_f{f}_d{d}/Nt50/results.json")
            if not p.exists():
                continue
            data = json.load(open(p))
            g = np.array(data["regret_grid_gwh"])
            lib = data["liberal_aep_gwh"]
            peak[i, j, k] = 100 * g[0, :].max() / lib

print(f"Filled cells: {(~np.isnan(peak)).sum()}/{peak.size}")


def invert_one(R_arr, d_arr, tau):
    """Invert monotone-decay fit to find d* such that R(d*) = tau.
    Uses PCHIP with the right-most envelope (cumulative max from far end inward,
    enforcing monotone decay even when noise creates non-monotonicity).
    Returns d* in D, or NaN if curve never crosses tau within the buffer range."""
    R = R_arr.copy()
    d = np.asarray(d_arr, dtype=float)
    if np.all(np.isnan(R)):
        return np.nan
    # Replace NaN with linear interp in log-d
    valid = ~np.isnan(R)
    if valid.sum() < 2:
        return np.nan
    R = np.interp(d, d[valid], R[valid])
    # Enforce monotone decay: cumulative max from right (downstream → close)
    # Working from far end inward, set R[i] = max(R[i:])
    R_mono = np.maximum.accumulate(R[::-1])[::-1]
    # If even at d_min R < tau, regret is below threshold everywhere
    if R_mono[0] < tau:
        return d[0]  # already below threshold at closest distance
    # If at d_max R > tau, never crosses
    if R_mono[-1] > tau:
        return np.inf
    # Interpolate to find crossing
    pchip = PchipInterpolator(d, R_mono - tau)
    roots = []
    for i in range(len(d) - 1):
        if R_mono[i] >= tau >= R_mono[i + 1]:
            try:
                # Linear interp inside this interval
                r0, r1 = R_mono[i], R_mono[i + 1]
                d0, d1 = d[i], d[i + 1]
                d_star = d0 + (tau - r0) * (d1 - d0) / (r1 - r0)
                roots.append(d_star)
            except Exception:
                pass
    return min(roots) if roots else np.nan


# d_star table for each tau
d_star = {tau: np.full((len(A_VALS), len(F_VALS)), np.nan) for tau in TAUS}
for i, a in enumerate(A_VALS):
    for j, f in enumerate(F_VALS):
        R = peak[i, j, :]
        for tau in TAUS:
            d_star[tau][i, j] = invert_one(R, D_VALS, tau)

# ---- Figure: contour at tau=1% ----
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
A_g, F_g = np.meshgrid(A_VALS, F_VALS, indexing="ij")
TAU_PLOT = 0.2
Z = d_star[TAU_PLOT]
Z_clip = np.where(np.isinf(Z), 60, Z)
mask = np.isnan(Z_clip)
levels = [2, 5, 10, 15, 20, 30, 40, 60]
im = ax.contourf(A_g, F_g, np.where(mask, np.nan, Z_clip),
                 levels=levels, cmap="YlOrRd", extend="max")
cs = ax.contour(A_g, F_g, np.where(mask, np.nan, Z_clip),
                levels=levels, colors="k", linewidths=0.4, alpha=0.5)
ax.clabel(cs, inline=True, fontsize=7, fmt="%.0f$D$")
fig.colorbar(im, ax=ax, label=fr"Required buffer $d^*$ (in $D$) for $\tau={TAU_PLOT}\%$ AEP")

# Markers for fitted real sites
markers = ["s", "D", "^", "o", "P"]
colors = ["cyan", "lime", "magenta", "yellow", "white"]
for (name, a, f), m, c in zip(SITES, markers, colors):
    ax.plot(a, f, marker=m, markerfacecolor=c, markeredgecolor="black",
            markersize=10, linewidth=0, zorder=10, label=name)
ax.legend(loc="upper left", fontsize=8, framealpha=0.95)
ax.set_xlabel("$a$ (concentration)")
ax.set_ylabel("$f$ (folding)")
ax.set_title(fr"Required buffer $d^*(a, f; \tau={TAU_PLOT}\%$ AEP)")
ax.set_xlim(0.25, 0.95)
ax.set_ylim(-0.05, 1.05)
out = Path("paper_v3/figures/buffer_table_contour.png")
fig.savefig(str(out), dpi=180, bbox_inches="tight")
print(f"Saved: {out}")

# ---- Figure: decay curves at all (a, f) ----
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for ax, fixed_dim, fixed_label in [(axes[0], "a", "fixed $a$, varying $f$"),
                                     (axes[1], "f", "fixed $f$, varying $a$")]:
    if fixed_dim == "a":
        for j, f in enumerate(F_VALS):
            color = plt.cm.viridis(j / max(len(F_VALS) - 1, 1))
            R_avg = np.nanmean(peak[:, j, :], axis=0)
            ax.plot(D_VALS, R_avg, "o-", color=color, label=f"$f$={f:.2f}",
                    linewidth=2, markersize=6)
    else:
        for i, a in enumerate(A_VALS):
            color = plt.cm.plasma(i / max(len(A_VALS) - 1, 1))
            R_avg = np.nanmean(peak[i, :, :], axis=0)
            ax.plot(D_VALS, R_avg, "o-", color=color, label=f"$a$={a:.1f}",
                    linewidth=2, markersize=6)
    for t in TAUS:
        ax.axhline(t, color="gray", ls=":", lw=0.8, alpha=0.6)
        ax.text(40, t, f"  $\\tau={t}\\%$", fontsize=8, color="gray", va="center")
    ax.set_xlabel("Buffer distance ($D$)")
    ax.set_ylabel("Peak regret (\\% of AEP)")
    ax.set_title(fixed_label)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
fig.suptitle("Buffer-decay curves: peak regret vs.\\ buffer distance, "
             "averaged over the orthogonal axis of the rose grid")
out = Path("paper_v3/figures/buffer_table_decay.png")
fig.savefig(str(out), dpi=180, bbox_inches="tight")
print(f"Saved: {out}")

# ---- LaTeX table ----
def fmt(v):
    if np.isnan(v): return "---"
    if np.isinf(v): return r"$>40$"
    return f"{v:.1f}"

lines = []
lines.append("\\begin{tabular}{c|" + "c" * len(F_VALS) + "}")
lines.append("\\toprule")
lines.append("$a \\downarrow$, $f \\rightarrow$ & " + " & ".join(f"{f:.2f}" for f in F_VALS) + " \\\\")
lines.append("\\midrule")
for i, a in enumerate(A_VALS):
    row = f"{a:.1f}"
    for j in range(len(F_VALS)):
        row += " & " + fmt(d_star[TAU_PLOT][i, j])
    lines.append(row + " \\\\")
lines.append("\\bottomrule")
lines.append("\\end{tabular}")
tbl_path = Path("paper_v3/tables")
tbl_path.mkdir(exist_ok=True)
with open(tbl_path / "buffer_table.tex", "w") as f:
    f.write("\n".join(lines))
print(f"Saved: {tbl_path / 'buffer_table.tex'}")

# Summary
print(f"\nd* (in D) at tau={TAU_PLOT}% AEP:")
print(f"{'a\\\\f':>5}", *[f"{f:>6.2f}" for f in F_VALS])
for i, a in enumerate(A_VALS):
    row = [f"{a:>5.1f}"]
    for j in range(len(F_VALS)):
        v = d_star[TAU_PLOT][i, j]
        if np.isnan(v): row.append("   --")
        elif np.isinf(v): row.append(">40D")
        else: row.append(f"{v:>5.1f}")
    print(*row)

print("\nReal sites:")
for name, a, f in SITES:
    line = f"  {name:<18} a={a:.3f} f={f:.3f}: "
    for tau in TAUS:
        # Bilinear interp into d_star grid
        from scipy.interpolate import RegularGridInterpolator
        Z_t = d_star[tau].copy()
        Z_t = np.where(np.isinf(Z_t), 60, Z_t)
        interp = RegularGridInterpolator(
            (np.array(A_VALS), np.array(F_VALS)), Z_t,
            method="linear", bounds_error=False, fill_value=np.nan)
        v = float(interp([[a, f]])[0])
        line += f"d*({tau}%)={v:>5.1f}D ({v*D_M/1000:.1f}km)  "
    print(line)

# JSON summary
summary = {
    "A_VALS": A_VALS, "F_VALS": F_VALS, "D_VALS": D_VALS,
    "peak_pct": peak.tolist(),
    "d_star": {f"tau_{t}pct": d_star[t].tolist() for t in TAUS},
}
out = Path("analysis/buffer_table/d_star_summary.json")
with open(out, "w") as f:
    json.dump(summary, f, indent=2,
              default=lambda x: None if (isinstance(x, float) and (np.isnan(x) or np.isinf(x))) else x)
print(f"Saved: {out}")

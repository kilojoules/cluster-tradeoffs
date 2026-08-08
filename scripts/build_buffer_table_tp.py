"""TurboPark buffer table: same as build_buffer_table.py but on TP grid.
Outputs: paper_v3/figures/buffer_table_contour_tp.png + tables/buffer_table_tp.tex.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

D_M = 240.0
A_VALS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
F_VALS = [0.0, 0.25, 0.5, 0.75, 1.0]
D_VALS = [2, 5, 10, 15, 20, 30, 40]
TAUS = [1.0, 2.0, 5.0]
TAU_PLOT = 2.0

SITES = [
    ("Horns Rev 1", 0.649, 0.384, "s", "cyan"),
    ("Lillgrund",   0.621, 0.544, "D", "lime"),
    ("Borssele",    0.728, 0.291, "^", "magenta"),
    ("Princess A.", 0.650, 0.182, "o", "yellow"),
    ("DEI",         0.637, 0.384, "P", "white"),
]

peak = np.full((len(A_VALS), len(F_VALS), len(D_VALS)), np.nan)
for i, a in enumerate(A_VALS):
    for j, f in enumerate(F_VALS):
        for k, d in enumerate(D_VALS):
            p = Path(f"analysis/buffer_table_tp/a{a}_f{f}_d{d}/Nt50/results.json")
            if not p.exists(): continue
            data = json.load(open(p))
            g = np.array(data["regret_grid_gwh"]); lib = data["liberal_aep_gwh"]
            peak[i, j, k] = 100 * g[0, :].max() / lib

print(f"Filled: {(~np.isnan(peak)).sum()}/{peak.size}")


def invert(R, d, tau):
    R = np.asarray(R, dtype=float); d = np.asarray(d, dtype=float)
    valid = ~np.isnan(R)
    if valid.sum() < 2: return np.nan
    R = np.interp(d, d[valid], R[valid])
    R_mono = np.maximum.accumulate(R[::-1])[::-1]
    if R_mono[0] < tau: return d[0]
    if R_mono[-1] > tau: return np.inf
    for i in range(len(d) - 1):
        if R_mono[i] >= tau >= R_mono[i + 1]:
            r0, r1 = R_mono[i], R_mono[i + 1]
            return d[i] + (tau - r0) * (d[i + 1] - d[i]) / (r1 - r0)
    return np.nan


d_star = {tau: np.full((len(A_VALS), len(F_VALS)), np.nan) for tau in TAUS}
for i, a in enumerate(A_VALS):
    for j, f in enumerate(F_VALS):
        for tau in TAUS:
            d_star[tau][i, j] = invert(peak[i, j, :], np.array(D_VALS), tau)

# Plot contour
fig, ax = plt.subplots(1, 1, figsize=(8.5, 6))
A_g, F_g = np.meshgrid(A_VALS, F_VALS, indexing="ij")
Z = d_star[TAU_PLOT]
Z_clip = np.where(np.isinf(Z), 60, Z)
levels = [2, 5, 10, 15, 20, 30, 40, 60]
im = ax.contourf(A_g, F_g, Z_clip, levels=levels, cmap="YlOrRd", extend="max")
cs = ax.contour(A_g, F_g, Z_clip, levels=levels, colors="k", linewidths=0.4, alpha=0.5)
ax.clabel(cs, inline=True, fontsize=7, fmt="%.0f$D$")
fig.colorbar(im, ax=ax, label=fr"Required buffer $d^*$ ($D$) for $\tau={TAU_PLOT}\%$ AEP")
for name, a, f, marker, color in SITES:
    ax.plot(a, f, marker, markerfacecolor=color, markeredgecolor="black",
            markersize=11, linewidth=0, zorder=10, label=name)
ax.legend(loc="upper left", fontsize=8, framealpha=0.95)
ax.set_xlabel("$a$ (concentration)")
ax.set_ylabel("$f$ (folding)")
ax.set_title(fr"TurboPark buffer table: $d^*(a, f; \tau={TAU_PLOT}\%$ AEP)")
ax.set_xlim(0.25, 0.95); ax.set_ylim(-0.05, 1.05)
out = Path("paper_v3/figures/buffer_table_tp_contour.png")
fig.savefig(str(out), dpi=180, bbox_inches="tight")
print(f"Saved: {out}")


def fmt(v):
    if np.isnan(v): return "---"
    if np.isinf(v): return r"$>40$"
    return f"{v:.1f}"

lines = ["\\begin{tabular}{c|" + "c" * len(F_VALS) + "}", "\\toprule",
         "$a \\downarrow$, $f \\rightarrow$ & " + " & ".join(f"{f:.2f}" for f in F_VALS) + " \\\\",
         "\\midrule"]
for i, a in enumerate(A_VALS):
    row = f"{a:.1f}"
    for j in range(len(F_VALS)):
        row += " & " + fmt(d_star[TAU_PLOT][i, j])
    lines.append(row + " \\\\")
lines.append("\\bottomrule"); lines.append("\\end{tabular}")
with open("paper_v3/tables/buffer_table_tp.tex", "w") as f:
    f.write("\n".join(lines))
print("Saved: paper_v3/tables/buffer_table_tp.tex")

# Summary
print(f"\nTurboPark d* at tau={TAU_PLOT}% AEP:")
print(f'{"a\\\\f":>5}', *[f"{f:>6.2f}" for f in F_VALS])
for i, a in enumerate(A_VALS):
    row = [f"{a:>5.1f}"]
    for j in range(len(F_VALS)):
        v = d_star[TAU_PLOT][i, j]
        if np.isnan(v): row.append("   --")
        elif np.isinf(v): row.append(">40D")
        else: row.append(f"{v:>5.1f}")
    print(*row)

print("\nReal sites:")
from scipy.interpolate import RegularGridInterpolator
Z_t = np.where(np.isinf(d_star[TAU_PLOT]), 60, d_star[TAU_PLOT])
interp = RegularGridInterpolator(
    (np.array(A_VALS), np.array(F_VALS)), Z_t,
    method="linear", bounds_error=False, fill_value=np.nan)
for name, a, f, _, _ in SITES:
    v = float(interp([[a, f]])[0])
    print(f"  {name:<14} a={a:.3f} f={f:.3f}: d*({TAU_PLOT}%)={v:5.1f}D ({v*D_M/1000:.1f}km)")

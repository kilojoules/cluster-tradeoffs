"""Alternative view of Fig 6: regret contours over (a, f) at fixed minimum buffer.
Multi-panel, one panel per buffer distance."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

A_VALS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
F_VALS = [0.0, 0.25, 0.5, 0.75, 1.0]
D_VALS = [2, 5, 10, 15, 20, 30, 40]

SITES = [
    ("Horns Rev 1", 0.649, 0.384, "tab:blue", "s"),
    ("Lillgrund",   0.621, 0.544, "tab:green", "D"),
    ("Borssele",    0.728, 0.291, "tab:orange", "^"),
    ("DEI",         0.637, 0.384, "tab:red", "P"),
]

peak = np.full((len(A_VALS), len(F_VALS), len(D_VALS)), np.nan)
for i, a in enumerate(A_VALS):
    for j, f in enumerate(F_VALS):
        for k, d in enumerate(D_VALS):
            p = Path(f"analysis/buffer_table/a{a}_f{f}_d{d}/Nt50/results.json")
            if not p.exists(): continue
            data = json.load(open(p))
            g = np.array(data["regret_grid_gwh"]); lib = data["liberal_aep_gwh"]
            peak[i, j, k] = 100 * g[0, :].max() / lib

# 4 panels at 2D, 5D, 10D, 20D — most informative
panels = [2, 5, 10, 20]
fig, axes = plt.subplots(1, len(panels), figsize=(4*len(panels)+1, 4.8))
A_g, F_g = np.meshgrid(A_VALS, F_VALS, indexing="ij")
vmax = float(np.nanmax(peak))

for ax, d in zip(axes, panels):
    k = D_VALS.index(d)
    Z = peak[:, :, k]
    levels = np.linspace(0, vmax, 11)
    im = ax.contourf(A_g, F_g, Z, levels=levels, cmap="YlOrRd", extend="max")
    cs = ax.contour(A_g, F_g, Z, levels=levels, colors="k", linewidths=0.4, alpha=0.5)
    ax.clabel(cs, inline=True, fontsize=6, fmt="%.2f")
    site_offsets = {
        "Horns Rev 1": (8, 8),
        "Lillgrund":   (8, 4),
        "Borssele":    (8, 4),
        "DEI":         (-22, -10),
    }
    for name, a, f, color, marker in SITES:
        ax.plot(a, f, marker=marker, markerfacecolor=color, markeredgecolor="black",
                markersize=10, linewidth=0)
        dx, dy = site_offsets.get(name, (6, 4))
        ax.annotate(name, (a, f), textcoords="offset points",
                    xytext=(dx, dy), fontsize=7)
    ax.set_xlabel("$a$ (concentration)")
    if ax is axes[0]:
        ax.set_ylabel("$f$ (folding)")
    ax.set_title(f"buffer $d = {d}D$")
    ax.set_xlim(0.28, 0.92); ax.set_ylim(-0.05, 1.05)

cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02,
                    label="Peak design regret (\\% of AEP)")
fig.suptitle("Regret over wind-rose shape space at fixed minimum buffer distance",
             fontsize=12)
out = Path("paper_v3/figures/regret_contour_at_buffer.png")
fig.savefig(str(out), dpi=180, bbox_inches="tight")
print(f"Saved: {out}")

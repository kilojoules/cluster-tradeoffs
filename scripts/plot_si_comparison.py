"""Compare adversarial vs self-interested neighbor cross-sections."""

import jax
jax.config.update("jax_enable_x64", True)

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

D = 240.0

# Load both results
adv = json.load(open("analysis/cross_section/a0.9_f1.0/results.json"))
si = json.load(open("analysis/cross_section_si/a0.9_f1.0/results.json"))

adv_pct = np.array(adv["regret_grid_pct"])
si_pct = np.array(si["regret_grid_pct"])
bearings = np.array(adv["bearings_deg"])
distances_D = np.array(adv["distances_D"])

# Also load LinearSum cross-section
lin = json.load(open("analysis/cross_section_linearsum/a0.9_f1.0/results.json"))
lin_pct = np.array(lin["regret_grid_pct"])

# Wind rose for insets
from edrose import EllipticalWindRose
wr = EllipticalWindRose(a=0.9, f=1.0, theta_prev=270, n_sectors=24)
wr_dirs = np.array(wr.wind_directions)
wr_freq = np.array(wr.sector_frequencies)

# Build polar grid
dbear = np.radians(bearings[1] - bearings[0])
bear_edges = np.radians(np.concatenate([bearings - (bearings[1]-bearings[0])/2,
                                         [bearings[-1] + (bearings[1]-bearings[0])/2]]))
dist_edges = [distances_D[0] - (distances_D[1] - distances_D[0]) / 2]
for i in range(len(distances_D) - 1):
    dist_edges.append((distances_D[i] + distances_D[i+1]) / 2)
dist_edges.append(distances_D[-1] + (distances_D[-1] - distances_D[-2]) / 2)
dist_edges = np.maximum(np.array(dist_edges), 0)

theta_grid, r_grid = np.meshgrid(bear_edges, dist_edges)

global_max = max(np.nanmax(adv_pct), np.nanmax(si_pct), np.nanmax(lin_pct))

fig, axes = plt.subplots(1, 3, figsize=(18, 6),
                          subplot_kw={"projection": "polar"})

from matplotlib.projections.polar import PolarAxes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

panels = [
    ("Fixed reference farm\n(adversarial placement)", adv_pct),
    ("Self-interested neighbor\n(optimizes own AEP)", si_pct),
    ("Fixed reference farm\n(LinearSum superposition)", lin_pct),
]

for ax, (title, data) in zip(axes, panels):
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    im = ax.pcolormesh(theta_grid, r_grid, np.clip(data, 0, None),
                        cmap="YlOrRd", vmin=0, vmax=global_max,
                        shading="flat")

    idx = np.unravel_index(np.nanargmax(data), data.shape)
    ax.plot(np.radians(bearings[idx[1]]), distances_D[idx[0]],
            "k*", markersize=14, zorder=10)

    ax.set_title(title, fontsize=10, pad=15)
    ax.set_rlabel_position(135)
    ax.set_yticks([10, 20, 30, 40, 60])
    ax.set_yticklabels(["10D", "20D", "30D", "40D", "60D"], fontsize=7)
    ax.tick_params(axis="x", labelsize=8)

    # Wind rose inset
    ax_inset = inset_axes(ax, width="30%", height="30%", loc="lower right",
                          axes_class=PolarAxes,
                          axes_kwargs={"theta_zero_location": "N",
                                       "theta_direction": -1})
    wr_width = np.radians(360 / len(wr_dirs))
    ax_inset.bar(np.radians(wr_dirs), wr_freq, width=wr_width,
                 color="steelblue", alpha=0.7, edgecolor="navy", linewidth=0.3)
    ax_inset.set_yticks([])
    ax_inset.set_xticks([])
    ax_inset.patch.set_alpha(0.8)

cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
fig.colorbar(im, cax=cbar_ax, label="Design Regret (% of AEP)")

fig.suptitle("Cross-Section Comparison: $a$=0.9, $f$=1.0 (concentrated, unidirectional)\n"
             "Bastankhah $k$=0.04, DEI polygon, 50 IEA-15MW, K=5 multistart",
             fontsize=12, y=1.02)
plt.tight_layout(rect=[0, 0, 0.91, 0.95])

out = Path("analysis/cross_section_comparison.png")
fig.savefig(str(out), dpi=200, bbox_inches="tight")
print(f"Saved: {out}")

# Summary
print(f"\n{'='*60}")
print("COMPARISON SUMMARY (a=0.9, f=1.0)")
print(f"{'='*60}")
for title, data in panels:
    idx = np.unravel_index(np.nanargmax(data), data.shape)
    print(f"  {title.replace(chr(10), ' ')}")
    print(f"    Max: {np.nanmax(data):.2f}% at bearing={bearings[idx[1]]:.0f}°, dist={distances_D[idx[0]]:.0f}D")
    print(f"    Mean: {np.nanmean(data):.2f}%")

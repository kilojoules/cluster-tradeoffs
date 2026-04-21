"""Line plot of regret vs bearing for the single-direction case,
with distance as contour lines. Shows the dip at directly upwind."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

D = 240.0
distances_D = [2, 5, 10, 15, 20, 30, 40]

regrets = []
lib_aep = None
bearings = None
for d in distances_D:
    data = json.load(open(f"analysis/cross_section_fixed/unidir90_d{d}/Nt50/results.json"))
    g = np.array(data["regret_grid_gwh"])
    regrets.append(g[0, :])
    if lib_aep is None:
        lib_aep = data["liberal_aep_gwh"]
        bearings = np.array(data["bearings_deg"])

regret_grid = np.array(regrets)
regret_pct = 100 * regret_grid / lib_aep

# Color map for distances
colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(distances_D)))

fig, ax = plt.subplots(figsize=(10, 6))

for di, (d, color) in enumerate(zip(distances_D, colors)):
    ax.plot(bearings, regret_pct[di, :], "o-", color=color, linewidth=2,
            markersize=5, label=f"{d}$D$ ({d*D/1000:.1f} km)")

# Mark the wind direction
ax.axvline(90, color="steelblue", ls="--", lw=2, alpha=0.5)
ax.text(90, ax.get_ylim()[1] * 0.95, "Wind from\nhere (90$^\\circ$)",
        ha="center", va="top", fontsize=10, color="steelblue",
        fontweight="bold")

# Mark the downwind region
ax.axvspan(195, 345, alpha=0.05, color="gray")
ax.text(270, 0.05, "Downwind\n(zero regret)", ha="center", va="bottom",
        fontsize=9, color="gray", style="italic")

ax.set_xlabel("Neighbor bearing ($^\\circ$)", fontsize=12)
ax.set_ylabel("Design regret (% of AEP)", fontsize=12)
ax.set_title("Single Wind Direction (from 90$^\\circ$): Regret vs. Neighbor Bearing\n"
             "Identical farm copy, boundary-gap distance, $K_{lib}=K_{cons}=300$",
             fontsize=12)
ax.legend(title="Buffer distance", fontsize=9, title_fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 360)
ax.set_xticks(np.arange(0, 361, 45))
ax.set_ylim(bottom=0)

out = Path("paper_v3/figures/unidir_lineplot.png")
fig.savefig(str(out), dpi=200, bbox_inches="tight")
print(f"Saved: {out}")

"""Side-by-side radar plots: DEI cross-section regret with vs without blockage."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

distances = [2, 20]
fig, axes = plt.subplots(2, 2, figsize=(12, 11),
                         subplot_kw={"projection": "polar"})
all_pct = []
for tag in ["noblock", "block"]:
    for d in distances:
        p = Path(f"analysis/blockage/dei_{tag}_d{d}/Nt50/results.json")
        data = json.load(open(p))
        g = np.array(data["regret_grid_gwh"]); lib = data["liberal_aep_gwh"]
        all_pct.append(np.clip(100 * g[0, :] / lib, 0, None))

vmax = max(arr.max() for arr in all_pct)

for ri, tag in enumerate(["noblock", "block"]):
    for ci, d in enumerate(distances):
        ax = axes[ri, ci]
        p = Path(f"analysis/blockage/dei_{tag}_d{d}/Nt50/results.json")
        data = json.load(open(p))
        g = np.array(data["regret_grid_gwh"]); lib = data["liberal_aep_gwh"]
        bearings = np.array(data["bearings_deg"])
        regret_pct = np.clip(100 * g[0, :] / lib, 0, None)

        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        # bar plot for radar
        theta = np.deg2rad(bearings)
        width = np.deg2rad(360 / len(bearings)) * 0.95
        bars = ax.bar(theta, regret_pct, width=width,
                      color=plt.cm.YlOrRd(regret_pct / vmax),
                      edgecolor="gray", linewidth=0.3, alpha=1.0)
        ax.set_ylim(0, vmax * 1.05)
        ax.set_title(f"{'No blockage' if tag=='noblock' else 'With blockage'} | "
                     f"buffer = {d}$D$\npeak = {regret_pct.max():.3f}\\%",
                     fontsize=11)

fig.suptitle("DEI cross-section regret: with vs without self-similarity blockage\n"
             "$K_{lib}=K_{cons}=300$, Bastankhah $k=0.04$",
             fontsize=12)
sm = plt.cm.ScalarMappable(cmap="YlOrRd",
                           norm=plt.Normalize(vmin=0, vmax=vmax))
sm.set_array([])
cbar_ax = fig.add_axes([0.92, 0.20, 0.015, 0.6])
fig.colorbar(sm, cax=cbar_ax, label="Design regret (% of AEP)")

plt.tight_layout(rect=[0, 0, 0.91, 0.95])
out = Path("paper_v3/figures/blockage_ab.png")
fig.savefig(str(out), dpi=180, bbox_inches="tight")
print(f"Saved: {out}")

# Numeric summary
print(f"\n{'tag':>10} {'d':>4} {'peak %':>10} {'mean %':>10}")
for tag in ["noblock", "block"]:
    for d in distances:
        p = Path(f"analysis/blockage/dei_{tag}_d{d}/Nt50/results.json")
        data = json.load(open(p))
        g = np.array(data["regret_grid_gwh"]); lib = data["liberal_aep_gwh"]
        regret_pct = np.clip(100 * g[0, :] / lib, 0, None)
        print(f"{tag:>10} {d:>3}D {regret_pct.max():>9.3f}% {regret_pct.mean():>9.3f}%")

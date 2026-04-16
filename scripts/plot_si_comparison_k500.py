"""Compare adversarial vs self-interested neighbor cross-sections at K=500."""

import jax
jax.config.update("jax_enable_x64", True)

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.projections.polar import PolarAxes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pathlib import Path
from edrose import EllipticalWindRose

D = 240.0

cases = [
    ("$a$=0.5, $f$=0.0\n(moderate, bidir)", "a0.5_f0.0", 0.5, 0.0),
    ("$a$=0.7, $f$=0.5\n(mid-range)", "a0.7_f0.5", 0.7, 0.5),
    ("$a$=0.9, $f$=1.0\n(concentrated, unidir)", "a0.9_f1.0", 0.9, 1.0),
]

fig, axes = plt.subplots(2, 3, figsize=(18, 11),
                          subplot_kw={"projection": "polar"})

global_max_pct = 0
all_data = {}
for _, case_dir, a_val, f_val in cases:
    adv = json.load(open(f"analysis/cross_section_k500v/{case_dir}/results.json"))
    si = json.load(open(f"analysis/cross_section_si_k500/{case_dir}/results.json"))
    all_data[case_dir] = (adv, si)
    global_max_pct = max(global_max_pct,
                         np.nanmax(np.array(adv["regret_grid_pct"])),
                         np.nanmax(np.array(si["regret_grid_pct"])))

for col, (label, case_dir, a_val, f_val) in enumerate(cases):
    adv, si = all_data[case_dir]
    bearings = np.array(adv["bearings_deg"])
    distances_D = np.array(adv["distances_D"])

    dbear = bearings[1] - bearings[0]
    bear_edges = np.radians(np.concatenate([bearings - dbear/2,
                                             [bearings[-1] + dbear/2]]))
    dist_edges = [distances_D[0] - (distances_D[1] - distances_D[0]) / 2]
    for i in range(len(distances_D) - 1):
        dist_edges.append((distances_D[i] + distances_D[i+1]) / 2)
    dist_edges.append(distances_D[-1] + (distances_D[-1] - distances_D[-2]) / 2)
    dist_edges = np.maximum(np.array(dist_edges), 0)
    theta_grid, r_grid = np.meshgrid(bear_edges, dist_edges)

    wr = EllipticalWindRose(a=a_val, f=f_val, theta_prev=270, n_sectors=24)

    for row, (data, row_label) in enumerate([
        (adv, "Fixed reference farm"),
        (si, "Self-interested neighbor"),
    ]):
        ax = axes[row, col]
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)

        regret_pct = np.array(data["regret_grid_pct"])
        im = ax.pcolormesh(theta_grid, r_grid, np.clip(regret_pct, 0, None),
                            cmap="YlOrRd", vmin=0, vmax=global_max_pct,
                            shading="flat")

        idx = np.unravel_index(np.nanargmax(regret_pct), regret_pct.shape)
        ax.plot(np.radians(bearings[idx[1]]), distances_D[idx[0]],
                "k*", markersize=14, zorder=10)

        if row == 0:
            ax.set_title(label, fontsize=10, pad=15)
        ax.set_rlabel_position(135)
        ax.set_yticks([10, 20, 30, 40, 60])
        ax.set_yticklabels(["10D", "20D", "30D", "40D", "60D"], fontsize=7)
        ax.tick_params(axis="x", labelsize=8)

        # Peak annotation
        max_pct = np.nanmax(regret_pct)
        ax.text(0.02, 0.98, f"Peak: {max_pct:.1f}%",
                transform=ax.transAxes, fontsize=9, fontweight="bold",
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        # Wind rose inset
        ax_inset = inset_axes(ax, width="30%", height="30%", loc="lower right",
                              axes_class=PolarAxes)
        ax_inset.set_theta_zero_location("N")
        ax_inset.set_theta_direction(-1)
        wr_width = np.radians(360 / 24)
        ax_inset.bar(np.radians(wr.wind_directions), wr.sector_frequencies,
                     width=wr_width, color="steelblue", alpha=0.7,
                     edgecolor="navy", linewidth=0.3)
        ax_inset.set_yticks([])
        ax_inset.set_xticks([])
        ax_inset.patch.set_alpha(0.8)

# Row labels
axes[0, 0].text(-0.3, 0.5, "Fixed reference\nfarm (adversarial)",
                transform=axes[0, 0].transAxes, fontsize=12,
                fontweight="bold", va="center", ha="center", rotation=90)
axes[1, 0].text(-0.3, 0.5, "Self-interested\nneighbor",
                transform=axes[1, 0].transAxes, fontsize=12,
                fontweight="bold", va="center", ha="center", rotation=90)

cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
fig.colorbar(im, cax=cbar_ax, label="Design Regret (% of AEP)")

fig.suptitle("Adversarial vs Self-Interested Neighbor Cross-Sections at K=500\n"
             "Bastankhah $k$=0.04, DEI polygon, 50 IEA-15MW target turbines",
             fontsize=13, y=1.01)
plt.tight_layout(rect=[0.05, 0, 0.91, 0.96])

out = Path("paper_v3/figures/si_comparison_k500.png")
fig.savefig(str(out), dpi=200, bbox_inches="tight")
print(f"Saved: {out}")

# Summary
print(f"\n{'='*70}")
print("SUMMARY: Self-interested vs Adversarial at K=500")
print(f"{'='*70}")
for label, case_dir, a_val, f_val in cases:
    adv, si = all_data[case_dir]
    g_adv = np.array(adv['regret_grid_pct'])
    g_si = np.array(si['regret_grid_pct'])
    max_adv = np.nanmax(g_adv)
    max_si = np.nanmax(g_si)
    idx_adv = np.unravel_index(np.nanargmax(g_adv), g_adv.shape)
    idx_si = np.unravel_index(np.nanargmax(g_si), g_si.shape)
    print(f"\n{label.replace(chr(10), ' ')}:")
    print(f"  Adversarial: {max_adv:.2f}% at {bearings[idx_adv[1]]:.0f}°, {distances_D[idx_adv[0]]:.0f}D")
    print(f"  Self-inter:  {max_si:.2f}% at {bearings[idx_si[1]]:.0f}°, {distances_D[idx_si[0]]:.0f}D")
    print(f"  Ratio SI/Adv: {max_si/max_adv:.2f}x")

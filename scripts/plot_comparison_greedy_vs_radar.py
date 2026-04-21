"""Compare greedy individual turbine placement (upper bound) vs
identical farm copy cross-section (realistic scenario).

Side-by-side for key wind rose cases.
"""

import jax
jax.config.update("jax_enable_x64", True)

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from pathlib import Path

D = 240.0

# DEI polygon
_dk0w_raw = np.array([
    706694.3923283464, 6224158.532895836,
    703972.0844905999, 6226906.597455995,
    702624.6334635273, 6253853.5386425415,
    712771.6248419734, 6257704.934445341,
    715639.3355871611, 6260664.6846508905,
    721593.2420745814, 6257906.998015941,
]).reshape((-1, 2))
cx, cy = _dk0w_raw[:, 0].mean(), _dk0w_raw[:, 1].mean()
boundary_np = (_dk0w_raw - np.array([cx, cy]))
bnd_km = boundary_np / 1000

cases = [
    ("Single direction\n(wind from E)", "unidir90", None),
    ("Concentrated unidir\n($a$=0.9, $f$=1.0)", "a0.9_f1.0", "a0.9_f1.0"),
    ("Moderate bidir\n($a$=0.5, $f$=0.0)", "a0.5_f0.0", "a0.5_f0.0"),
]

fig, axes = plt.subplots(len(cases), 2, figsize=(16, 6 * len(cases)))

for row, (label, xsec_case, greedy_case) in enumerate(cases):

    # --- LEFT: Greedy grid (upper bound) ---
    ax_left = axes[row, 0]
    if greedy_case:
        p = Path(f"analysis/edrose_sweep_k500/{greedy_case}/results.json")
        if p.exists():
            d = json.load(open(p))
            nx = np.array(d["neighbor_x"]) / 1000
            ny = np.array(d["neighbor_y"]) / 1000
            regret = d["regret_gwh"]
            lib_aep = d["liberal_aep_gwh"]

            poly = MplPolygon(bnd_km, closed=True, fill=True,
                              facecolor="#e0e0f0", edgecolor="black", lw=2, zorder=2)
            ax_left.add_patch(poly)
            ax_left.scatter(nx, ny, c="red", marker="D", s=40, edgecolors="darkred",
                            linewidths=0.5, zorder=5, label=f"30 greedy turbines")
            ax_left.set_xlim(bnd_km[:, 0].min() - 15, bnd_km[:, 0].max() + 15)
            ax_left.set_ylim(bnd_km[:, 1].min() - 15, bnd_km[:, 1].max() + 15)
            ax_left.set_aspect("equal")
            ax_left.legend(fontsize=9, loc="lower left")
            ax_left.text(0.98, 0.98,
                         f"Regret: {regret:.0f} GWh\n({100*regret/lib_aep:.1f}% of AEP)",
                         transform=ax_left.transAxes, fontsize=11, fontweight="bold",
                         va="top", ha="right",
                         bbox=dict(facecolor="white", edgecolor="red", alpha=0.9))
        else:
            ax_left.text(0.5, 0.5, "No data", transform=ax_left.transAxes,
                         ha="center", va="center")
    else:
        ax_left.text(0.5, 0.5, "N/A\n(not run for\nsingle direction)",
                     transform=ax_left.transAxes, ha="center", va="center",
                     fontsize=12, color="gray")
        ax_left.set_xlim(-20, 20)
        ax_left.set_ylim(-20, 20)

    if row == 0:
        ax_left.set_title("Greedy individual turbines\n(adversarial upper bound)",
                          fontsize=13, fontweight="bold")
    ax_left.set_ylabel(label, fontsize=12, fontweight="bold")
    ax_left.set_xlabel("x (km)")

    # --- RIGHT: Cross-section polar (realistic) ---
    ax_right = fig.add_subplot(len(cases), 2, row * 2 + 2, projection="polar")
    # Remove the non-polar axis
    axes[row, 1].set_visible(False)

    distances_D_arr = [2, 5, 10, 15, 20, 30, 40]
    regret_rows = []
    lib_aep_xsec = None
    bearings = None
    for dist in distances_D_arr:
        p = Path(f"analysis/cross_section_fixed/{xsec_case}_d{dist}/Nt50/results.json")
        if not p.exists():
            continue
        data = json.load(open(p))
        g = np.array(data["regret_grid_gwh"])
        regret_rows.append(g[0, :])
        if lib_aep_xsec is None:
            lib_aep_xsec = data["liberal_aep_gwh"]
            bearings = np.array(data["bearings_deg"])

    if regret_rows and bearings is not None:
        regret_grid = np.array(regret_rows)
        regret_pct = 100 * regret_grid / lib_aep_xsec

        ax_right.set_theta_zero_location("N")
        ax_right.set_theta_direction(-1)

        dbear = bearings[1] - bearings[0]
        bear_edges = np.radians(np.concatenate([bearings - dbear / 2,
                                                 [bearings[-1] + dbear / 2]]))
        dists = np.array(distances_D_arr[:len(regret_rows)])
        dist_edges = [dists[0] - (dists[1] - dists[0]) / 2]
        for i in range(len(dists) - 1):
            dist_edges.append((dists[i] + dists[i + 1]) / 2)
        dist_edges.append(dists[-1] + (dists[-1] - dists[-2]) / 2)
        dist_edges = np.maximum(np.array(dist_edges), 0)

        theta_grid, r_grid = np.meshgrid(bear_edges, dist_edges)
        im = ax_right.pcolormesh(theta_grid, r_grid, np.clip(regret_pct, 0, None),
                                  cmap="YlOrRd", shading="flat")

        idx = np.unravel_index(np.nanargmax(regret_pct), regret_pct.shape)
        ax_right.plot(np.radians(bearings[idx[1]]), dists[idx[0]],
                      "k*", markersize=12, zorder=10)

        ax_right.set_rlabel_position(135)
        ax_right.set_yticks([5, 10, 20, 30, 40])
        ax_right.set_yticklabels(["5D", "10D", "20D", "30D", "40D"], fontsize=7)
        ax_right.tick_params(axis="x", labelsize=8)

        max_pct = np.nanmax(regret_pct)
        ax_right.text(0.02, 0.98, f"Peak: {max_pct:.2f}%",
                      transform=ax_right.transAxes, fontsize=11, fontweight="bold",
                      va="top", ha="left",
                      bbox=dict(facecolor="white", edgecolor="black", alpha=0.9))

        fig.colorbar(im, ax=ax_right, shrink=0.7, pad=0.1,
                     label="Regret (% of AEP)")

    if row == 0:
        ax_right.set_title("Identical farm copy cross-section\n(realistic scenario)",
                           fontsize=13, fontweight="bold", pad=20)

fig.suptitle("Greedy Adversarial Placement vs. Realistic Neighbor Cross-Section",
             fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()

out = Path("paper_v3/figures/greedy_vs_radar.png")
fig.savefig(str(out), dpi=200, bbox_inches="tight")
print(f"Saved: {out}")

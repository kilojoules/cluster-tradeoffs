"""Plot corrected cross-sections with boundary-gap distance metric."""

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

D = 240.0

cases = [
    ("DEI real\nwind rose", "dei"),
    ("$a$=0.9, $f$=1.0\n(conc. unidir)", "a0.9_f1.0"),
    ("$a$=0.5, $f$=1.0\n(mod. unidir)", "a0.5_f1.0"),
    ("$a$=0.7, $f$=0.5\n(mid-range)", "a0.7_f0.5"),
    ("$a$=0.9, $f$=0.0\n(conc. bidir)", "a0.9_f0.0"),
    ("$a$=0.5, $f$=0.0\n(mod. bidir)", "a0.5_f0.0"),
]

distances_D = [2, 5, 10, 15, 20, 30, 40]


def get_wind_rose(case_dir):
    if case_dir == "dei":
        import pandas as pd
        csv_path = Path("energy_island_10y_daily_av_wind.csv")
        df = pd.read_csv(csv_path, sep=";")
        wd_ts = df["WD_150"].values
        n_bins = 24
        bin_edges = np.linspace(0, 360, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        w = np.zeros(n_bins)
        for i in range(n_bins):
            if i == n_bins - 1:
                mask = (wd_ts >= bin_edges[i]) | (wd_ts < bin_edges[0])
            else:
                mask = (wd_ts >= bin_edges[i]) & (wd_ts < bin_edges[i + 1])
            w[i] = mask.sum()
        w /= w.sum()
        return bin_centers, w
    parts = case_dir.replace("a", "").replace("f", "").split("_")
    a_val, f_val = float(parts[0]), float(parts[1])
    from edrose import EllipticalWindRose
    wr = EllipticalWindRose(a=a_val, f=f_val, theta_prev=270, n_sectors=24)
    return np.array(wr.wind_directions), np.array(wr.sector_frequencies)


# Load and merge per-distance results
all_data = {}
global_max_pct = 0
for label, case_dir in cases:
    regret_rows = []
    lib_aep = None
    bearings = None
    for d in distances_D:
        p = Path(f"analysis/cross_section_fixed/{case_dir}_d{d}/Nt50/results.json")
        data = json.load(open(p))
        g = np.array(data["regret_grid_gwh"])
        regret_rows.append(g[0, :])
        if lib_aep is None:
            lib_aep = data["liberal_aep_gwh"]
            bearings = np.array(data["bearings_deg"])
    regret_grid = np.array(regret_rows)
    regret_pct = 100 * regret_grid / lib_aep
    all_data[case_dir] = {
        "regret_grid_gwh": regret_grid,
        "regret_grid_pct": regret_pct,
        "bearings_deg": bearings,
        "distances_D": distances_D,
        "liberal_aep_gwh": lib_aep,
    }
    global_max_pct = max(global_max_pct, np.nanmax(regret_pct))

# Plot 6-panel polar heatmap
fig, axes = plt.subplots(1, 6, figsize=(30, 5.5),
                          subplot_kw={"projection": "polar"})

for ax, (label, case_dir) in zip(axes, cases):
    d = all_data[case_dir]
    bearings = d["bearings_deg"]
    dists = np.array(d["distances_D"])
    regret_pct = d["regret_grid_pct"]

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    dbear = bearings[1] - bearings[0]
    bear_edges = np.radians(np.concatenate([bearings - dbear/2,
                                             [bearings[-1] + dbear/2]]))
    dist_edges = [dists[0] - (dists[1] - dists[0]) / 2]
    for i in range(len(dists) - 1):
        dist_edges.append((dists[i] + dists[i+1]) / 2)
    dist_edges.append(dists[-1] + (dists[-1] - dists[-2]) / 2)
    dist_edges = np.maximum(np.array(dist_edges), 0)

    theta_grid, r_grid = np.meshgrid(bear_edges, dist_edges)
    im = ax.pcolormesh(theta_grid, r_grid, np.clip(regret_pct, 0, None),
                        cmap="YlOrRd", vmin=0, vmax=global_max_pct,
                        shading="flat")

    idx = np.unravel_index(np.nanargmax(regret_pct), regret_pct.shape)
    ax.plot(np.radians(bearings[idx[1]]), dists[idx[0]],
            "k*", markersize=14, zorder=10)

    ax.set_title(label, fontsize=10, pad=15)
    ax.set_rlabel_position(135)
    ax.set_yticks([5, 10, 20, 30, 40])
    ax.set_yticklabels(["5D", "10D", "20D", "30D", "40D"], fontsize=7)
    ax.tick_params(axis="x", labelsize=8)

    # Wind rose inset
    wr_dirs, wr_freq = get_wind_rose(case_dir)
    ax_inset = inset_axes(ax, width="35%", height="35%", loc="lower right",
                          axes_class=PolarAxes)
    ax_inset.set_theta_zero_location("N")
    ax_inset.set_theta_direction(-1)
    wr_width = np.radians(360 / len(wr_dirs))
    ax_inset.bar(np.radians(wr_dirs), wr_freq, width=wr_width,
                 color="steelblue", alpha=0.7, edgecolor="navy", linewidth=0.3)
    ax_inset.set_yticks([])
    ax_inset.set_xticks([])
    ax_inset.patch.set_alpha(0.8)

fig.suptitle("Regret Cross-Sections (corrected boundary-gap distance, K=300)\n"
             "Identical reference farm placed at boundary gap distances 2-40$D$",
             fontsize=12, y=1.02)

cbar_ax = fig.add_axes([0.92, 0.15, 0.012, 0.7])
fig.colorbar(im, cax=cbar_ax, label="Design Regret (% of AEP)")

plt.tight_layout(rect=[0, 0, 0.91, 0.95])
out = Path("paper_v3/figures/cross_section_fixed.png")
fig.savefig(str(out), dpi=200, bbox_inches="tight")
print(f"Saved: {out}")

# Summary
print(f"\n{'='*80}")
print("CORRECTED CROSS-SECTION SUMMARY (boundary-gap distance)")
print(f"{'='*80}")
print(f"{'Case':<25} {'Max GWh':>10} {'Max %':>8} {'Bearing':>10} {'Buffer':>8}")
print("-" * 65)
for label, case_dir in cases:
    d = all_data[case_dir]
    g = d['regret_grid_gwh']
    pct = d['regret_grid_pct']
    idx = np.unravel_index(np.nanargmax(g), g.shape)
    print(f"{label.replace(chr(10), ' '):<25} {g[idx]:>9.2f} {pct[idx]:>7.3f}% "
          f"{d['bearings_deg'][idx[1]]:>8.0f}° {d['distances_D'][idx[0]]:>6}D")

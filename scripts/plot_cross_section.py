"""Plot polar regret cross-section heatmaps from (bearing, distance) sweep."""

import jax
jax.config.update("jax_enable_x64", True)

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.projections.polar import PolarAxes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pathlib import Path

D = 240.0  # m

cases = [
    ("DEI real wind rose", "dei"),
    ("$a$=0.9, $f$=1.0\n(concentrated, unidir)", "a0.9_f1.0"),
    ("$a$=0.7, $f$=0.5\n(mid-range)", "a0.7_f0.5"),
    ("$a$=0.9, $f$=0.0\n(concentrated, bidir)", "a0.9_f0.0"),
    ("$a$=0.5, $f$=1.0\n(moderate, unidir)", "a0.5_f1.0"),
    ("$a$=0.5, $f$=0.0\n(moderate, bidir)", "a0.5_f0.0"),
]

def get_wind_rose(case_dir):
    """Generate wind rose (directions, frequencies) for a given case."""
    if case_dir == "dei":
        import pandas as pd
        csv_path = Path(__file__).parent.parent / "energy_island_10y_daily_av_wind.csv"
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
    else:
        # Parse a and f from directory name like "a0.9_f1.0"
        parts = case_dir.replace("a", "").replace("f", "").split("_")
        a_val, f_val = float(parts[0]), float(parts[1])
        from edrose import EllipticalWindRose
        wr = EllipticalWindRose(a=a_val, f=f_val, theta_prev=270, n_sectors=24)
        return np.array(wr.wind_directions), np.array(wr.sector_frequencies)


# Load all data
all_data = {}
global_max_pct = 0
for label, case_dir in cases:
    with open(f"analysis/cross_section_k100/{case_dir}/results.json") as f:
        d = json.load(f)
    all_data[case_dir] = d
    g = np.array(d["regret_grid_pct"])
    global_max_pct = max(global_max_pct, np.nanmax(g))

# --- Figure 1: 5-panel polar heatmaps ---
fig, axes = plt.subplots(1, 6, figsize=(28, 5.5),
                          subplot_kw={"projection": "polar"})

for ax, (label, case_dir) in zip(axes, cases):
    d = all_data[case_dir]
    bearings = np.array(d["bearings_deg"])
    distances_D = np.array(d["distances_D"])
    regret_pct = np.array(d["regret_grid_pct"])  # shape: (n_dist, n_bear)

    # Convert bearing to radians (meteorological: 0=N, 90=E, clockwise)
    # matplotlib polar: 0=E, CCW. Convert: theta_mpl = pi/2 - bearing_rad
    bearings_rad = np.radians(bearings)
    # For pcolormesh we need edges
    dbear = bearings_rad[1] - bearings_rad[0]
    bear_edges = np.concatenate([bearings_rad - dbear/2, [bearings_rad[-1] + dbear/2]])

    # Distance edges
    dist_edges = [distances_D[0] - (distances_D[1] - distances_D[0]) / 2]
    for i in range(len(distances_D) - 1):
        dist_edges.append((distances_D[i] + distances_D[i+1]) / 2)
    dist_edges.append(distances_D[-1] + (distances_D[-1] - distances_D[-2]) / 2)
    dist_edges = np.array(dist_edges)
    dist_edges = np.maximum(dist_edges, 0)

    # Meshgrid for pcolormesh
    theta_grid, r_grid = np.meshgrid(bear_edges, dist_edges)

    # Convert meteorological bearing to math angle for plotting
    # Met: 0=N(up), 90=E(right) clockwise
    # Polar plot: 0=E(right), CCW
    # theta_plot = pi/2 - theta_met
    theta_plot = np.pi / 2 - theta_grid

    im = ax.pcolormesh(theta_plot, r_grid, np.clip(regret_pct, 0, None),
                        cmap="YlOrRd", vmin=0, vmax=global_max_pct,
                        shading="flat")

    # Mark max regret
    idx = np.unravel_index(np.nanargmax(regret_pct), regret_pct.shape)
    max_bear_rad = np.pi / 2 - np.radians(bearings[idx[1]])
    ax.plot(max_bear_rad, distances_D[idx[0]], "k*", markersize=12, zorder=10)

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)  # clockwise
    # Undo our conversion since we set N at top and clockwise
    # Actually let's redo this properly with meteorological convention
    ax.clear()

    # Redo with proper meteorological convention
    # Set theta_zero_location to N and direction clockwise, then use bearings directly
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    theta_grid2, r_grid2 = np.meshgrid(np.radians(np.concatenate([bearings - dbear*180/np.pi/2,
                                                                    [bearings[-1] + dbear*180/np.pi/2]])),
                                         dist_edges)

    im = ax.pcolormesh(theta_grid2, r_grid2, np.clip(regret_pct, 0, None),
                        cmap="YlOrRd", vmin=0, vmax=global_max_pct,
                        shading="flat")

    # Mark max
    idx = np.unravel_index(np.nanargmax(regret_pct), regret_pct.shape)
    ax.plot(np.radians(bearings[idx[1]]), distances_D[idx[0]],
            "k*", markersize=14, zorder=10)

    ax.set_title(label, fontsize=10, pad=15)
    ax.set_rlabel_position(135)
    ax.set_yticks([10, 20, 30, 40, 60])
    ax.set_yticklabels(["10D", "20D", "30D", "40D", "60D"], fontsize=7)
    ax.tick_params(axis="x", labelsize=8)

    # Wind rose inset
    wr_dirs, wr_freq = get_wind_rose(case_dir)
    ax_inset = inset_axes(ax, width="35%", height="35%", loc="lower right",
                          axes_class=PolarAxes,
                          axes_kwargs={"theta_zero_location": "N",
                                       "theta_direction": -1})
    wr_width = np.radians(360 / len(wr_dirs))
    ax_inset.bar(np.radians(wr_dirs), wr_freq, width=wr_width,
                 color="steelblue", alpha=0.7, edgecolor="navy", linewidth=0.3)
    ax_inset.set_yticks([])
    ax_inset.set_xticks([])
    ax_inset.patch.set_alpha(0.8)

fig.suptitle("Regret Cross-Section: 5x5 Reference Neighbor Farm (25 turbines at 7D spacing)\n"
             "Bastankhah $k$=0.04, DEI polygon, 50 IEA-15MW target turbines, K=5 multistart",
             fontsize=12, y=1.02)

# Shared colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.012, 0.7])
fig.colorbar(im, cax=cbar_ax, label="Design Regret (% of AEP)")

plt.tight_layout(rect=[0, 0, 0.91, 0.95])
out = Path("analysis/cross_section_k100/regret_cross_section.png")
fig.savefig(str(out), dpi=200, bbox_inches="tight")
print(f"Saved: {out}")

# --- Figure 2: Decay curves by bearing for each case ---
fig2, axes2 = plt.subplots(1, 6, figsize=(28, 4.5), sharey=True)

for ax, (label, case_dir) in zip(axes2, cases):
    d = all_data[case_dir]
    bearings = np.array(d["bearings_deg"])
    distances_D = np.array(d["distances_D"])
    regret_pct = np.array(d["regret_grid_pct"])

    # Plot top 4 bearings
    mean_regret_by_bearing = np.nanmean(regret_pct, axis=0)
    top4 = np.argsort(-mean_regret_by_bearing)[:4]
    for bi in top4:
        ax.plot(distances_D, regret_pct[:, bi], "o-", label=f"{bearings[bi]:.0f}°",
                linewidth=1.5, markersize=4)
    # Also plot mean
    ax.plot(distances_D, np.nanmean(regret_pct, axis=1), "k--", linewidth=2,
            label="Mean", alpha=0.5)

    ax.set_xlabel("Distance ($D$)")
    if ax == axes2[0]:
        ax.set_ylabel("Design Regret (% of AEP)")
    ax.set_title(label, fontsize=10)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

fig2.suptitle("Regret Decay by Bearing — Top 4 Most Adverse Directions",
              fontsize=12)
plt.tight_layout()
out2 = Path("analysis/cross_section_k100/regret_decay_by_bearing.png")
fig2.savefig(str(out2), dpi=200, bbox_inches="tight")
print(f"Saved: {out2}")

# --- Print summary table ---
print(f"\n{'='*80}")
print("CROSS-SECTION SUMMARY")
print(f"{'='*80}")
print(f"{'Case':<30} {'Max Regret':>12} {'Max %':>8} {'Max Bearing':>12} {'Max Dist':>10}")
print("-" * 80)
for label, case_dir in cases:
    d = all_data[case_dir]
    g = np.array(d["regret_grid_gwh"])
    lib = d["liberal_aep_gwh"]
    idx = np.unravel_index(np.nanargmax(g), g.shape)
    bearing = d["bearings_deg"][idx[1]]
    dist = d["distances_D"][idx[0]]
    maxr = np.nanmax(g)
    print(f"{label.replace(chr(10), ' '):<30} {maxr:>10.1f} GWh {100*maxr/lib:>7.2f}% "
          f"{bearing:>10.0f}° {dist:>8.0f}D")

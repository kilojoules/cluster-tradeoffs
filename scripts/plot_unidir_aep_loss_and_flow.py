"""AEP-loss radar + flow field for unidir90 case.
- Radar: AEP of liberal layout with neighbor, normalized by no-neighbor AEP.
- Flow field: one chosen (bearing, distance) config, single wind direction.
Uses IEA 15MW turbine + Bastankhah + DEI polygon, matching run_regret_cross_section.
"""

import jax
jax.config.update("jax_enable_x64", True)

import json
import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.projections.polar import PolarAxes
from pathlib import Path

# Match cross_section setup
import sys
sys.path.insert(0, "scripts")
from run_regret_cross_section import (
    boundary_np, boundary, D, create_dei_turbine,
    centroid_offset_for_gap,
)
from pixwake.deficit import BastankhahGaussianDeficit
from pixwake.core import WakeSimulation

turbine = create_dei_turbine()

# Load unidir90 cross-section with all distances
cases_distances = [2, 5, 10, 15, 20, 30, 40]
all_data = {}
for d in cases_distances:
    p = Path(f"analysis/cross_section_fixed/unidir90_d{d}/Nt50/results.json")
    all_data[d] = json.load(open(p))

# Liberal layout (identical across distances — target farm liberal layout)
lib_x = np.array(all_data[2]["liberal_x"])
lib_y = np.array(all_data[2]["liberal_y"])
lib_aep_nonbr = all_data[2]["liberal_aep_gwh"]
bearings = np.array(all_data[2]["bearings_deg"])

# Single wind direction: 90° (east), 9 m/s
wd = jnp.array([90.0])
ws = jnp.array([9.0])
sector_freq = jnp.array([1.0])

deficit = BastankhahGaussianDeficit(k=0.04)
sim = WakeSimulation(turbine, deficit)


def aep_with_neighbor(tx, ty, nx, ny):
    x_all = jnp.concatenate([tx, nx])
    y_all = jnp.concatenate([ty, ny])
    res = sim(x_all, y_all, ws_amb=ws, wd_amb=wd)
    # res.power() returns kW; shape (n_wd, n_turbines)
    p_kw = res.power()
    tgt_p_kw = jnp.sum(p_kw[:, :len(tx)] * sector_freq[:, None], axis=0)
    return float(8760.0 * jnp.sum(tgt_p_kw) / 1e6)  # GWh (kWh -> GWh = /1e6)


# Build AEP-loss grid (bearings x distances)
print(f"Computing AEP loss for {len(cases_distances)} x {len(bearings)} configs...")
aep_grid = np.zeros((len(cases_distances), len(bearings)))
for di, d in enumerate(cases_distances):
    gap_m = d * D
    for bi, bear in enumerate(bearings):
        offset, direction, _ = centroid_offset_for_gap(boundary_np, bear, gap_m)
        nx = jnp.array(lib_x + offset * direction[0])
        ny = jnp.array(lib_y + offset * direction[1])
        aep_grid[di, bi] = aep_with_neighbor(jnp.array(lib_x), jnp.array(lib_y), nx, ny)
    print(f"  d={d}D done, AEP range [{aep_grid[di].min():.1f}, {aep_grid[di].max():.1f}] GWh")

aep_loss_pct = 100 * (lib_aep_nonbr - aep_grid) / lib_aep_nonbr

# ============ Radar: AEP loss ============
fig, axes = plt.subplots(1, 2, figsize=(14, 6.5),
                         subplot_kw={"projection": "polar"})

# Load regret for side-by-side (% of AEP)
regret_grid = np.zeros_like(aep_grid)
for di, d in enumerate(cases_distances):
    regret_grid[di] = np.array(all_data[d]["regret_grid_gwh"])[0]
regret_pct = 100 * regret_grid / lib_aep_nonbr

vmax = float(max(aep_loss_pct.max(), regret_pct.max()))

for ax, grid, title in [(axes[0], aep_loss_pct, "AEP loss from neighbor wake\n(liberal layout, no re-optimization)"),
                        (axes[1], regret_pct, "Design regret\n(cons - lib AEP, both with neighbor)")]:
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    dbear = bearings[1] - bearings[0]
    bear_edges = np.radians(np.concatenate([bearings - dbear/2,
                                             [bearings[-1] + dbear/2]]))
    dists = np.array(cases_distances)
    dist_edges = [dists[0] - (dists[1] - dists[0]) / 2]
    for i in range(len(dists) - 1):
        dist_edges.append((dists[i] + dists[i+1]) / 2)
    dist_edges.append(dists[-1] + (dists[-1] - dists[-2]) / 2)
    dist_edges = np.maximum(np.array(dist_edges), 0)
    theta_grid, r_grid = np.meshgrid(bear_edges, dist_edges)
    im = ax.pcolormesh(theta_grid, r_grid, np.clip(grid, 0, None),
                       cmap="YlOrRd", vmin=0, vmax=vmax, shading="flat")
    ax.set_title(title, fontsize=10, pad=15)
    ax.set_yticks([5, 10, 20, 30, 40])
    ax.set_yticklabels(["5D", "10D", "20D", "30D", "40D"], fontsize=8)

fig.suptitle("Single direction (wind from 90°, east): AEP loss vs.\\ design regret",
             fontsize=11)
fig.colorbar(im, ax=axes, shrink=0.6, label="% of no-neighbor AEP", pad=0.08)
out = Path("paper_v3/figures/unidir_aep_loss_vs_regret.png")
fig.savefig(str(out), dpi=180, bbox_inches="tight")
print(f"Saved: {out}")

# ============ Flow field: 135° / 40D ============
target_bearing = 135.0
target_distance_D = 40
gap_m = target_distance_D * D
offset, direction, _ = centroid_offset_for_gap(boundary_np, target_bearing, gap_m)
nx = lib_x + offset * direction[0]
ny = lib_y + offset * direction[1]

# Grid spanning both farms + buffer
all_x = np.concatenate([lib_x, nx])
all_y = np.concatenate([lib_y, ny])
pad = 6 * D
xmin, xmax = all_x.min() - pad, all_x.max() + pad
ymin, ymax = all_y.min() - pad, all_y.max() + pad
n = 240
fm_x = jnp.linspace(xmin, xmax, n)
fm_y = jnp.linspace(ymin, ymax, n)
gx, gy = jnp.meshgrid(fm_x, fm_y)
flat_x = gx.ravel()
flat_y = gy.ravel()

ws_scalar = 9.0
wd_scalar = 90.0
x_all_j = jnp.array(np.concatenate([lib_x, nx]))
y_all_j = jnp.array(np.concatenate([lib_y, ny]))

print("Computing flow map...")
u_flat, _ = sim.flow_map(x_all_j, y_all_j, fm_x=flat_x, fm_y=flat_y,
                         ws=ws_scalar, wd=wd_scalar)
u_grid = np.array(u_flat).reshape(n, n)

fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))
pcm = ax2.pcolormesh(np.array(gx), np.array(gy), u_grid,
                     cmap="viridis_r", shading="auto",
                     vmin=8.0, vmax=9.0)
cbar = fig2.colorbar(pcm, ax=ax2, label="Wind speed (m/s)")
ax2.scatter(lib_x, lib_y, s=25, c="white", edgecolor="black",
            linewidth=0.8, zorder=5, label="Target (liberal)")
ax2.scatter(nx, ny, s=25, c="red", edgecolor="black",
            linewidth=0.8, zorder=5, label=f"Neighbor (bearing 135°, 40$D$)")
# inflow arrow
xa = xmin + 0.15 * (xmax - xmin); ya = ymin + 0.9 * (ymax - ymin)
ax2.annotate("", xy=(xa + 1500, ya), xytext=(xa, ya),
             arrowprops=dict(arrowstyle="->", lw=2.5, color="black"))
ax2.text(xa, ya + 150, "wind (9 m/s from east)", fontsize=10, color="black")
ax2.set_xlabel("x (m)"); ax2.set_ylabel("y (m)")
ax2.set_aspect("equal")
ax2.legend(loc="lower left", fontsize=10)
ax2.set_title("Unidirectional case: neighbor at bearing 135°, 40$D$ buffer\n"
              "Wind from east (90°); note how neighbor wake (SE of target) "
              "drifts westward and grazes target's SW edge",
              fontsize=11)
out2 = Path("paper_v3/figures/unidir_flow_135deg_40D.png")
fig2.savefig(str(out2), dpi=180, bbox_inches="tight")
print(f"Saved: {out2}")

# report values at that config
i135 = int(np.argmin(np.abs(bearings - 135)))
i40 = cases_distances.index(40)
print(f"\n135°/40D: AEP loss = {aep_loss_pct[i40, i135]:.3f}% "
      f"(regret = {regret_pct[i40, i135]:.3f}%)")

"""Snapshot of greedy grid screening: mid-placement step showing AEP deficit
field across candidate grid with 30D pad (representative of actual runs).
Shows placed neighbors + target liberal layout + AEP loss per candidate."""

import jax
jax.config.update("jax_enable_x64", True)

import json
import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.projections.polar import PolarAxes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.spatial import ConvexHull
from pathlib import Path

import sys
sys.path.insert(0, "scripts")
from run_regret_cross_section import boundary_np, D, create_dei_turbine
from pixwake.deficit import BastankhahGaussianDeficit
from pixwake.core import WakeSimulation

GRID_SPACING = 5.0 * D  # 1200 m — matches production
GRID_PAD = 60.0 * D      # 60D — doubled from prior 30D
BUFFER = 30.0 * D        # 30D exclusion around target

def build_grid(boundary_np, spacing=GRID_SPACING, pad=GRID_PAD, buffer=BUFFER):
    from shapely.geometry import Polygon, Point
    x_lo = boundary_np[:, 0].min() - pad
    x_hi = boundary_np[:, 0].max() + pad
    y_lo = boundary_np[:, 1].min() - pad
    y_hi = boundary_np[:, 1].max() + pad
    gx = np.arange(x_lo, x_hi, spacing)
    gy = np.arange(y_lo, y_hi, spacing)
    gx2, gy2 = np.meshgrid(gx, gy)
    cand = np.column_stack([gx2.ravel(), gy2.ravel()])
    poly = Polygon(boundary_np)
    # Minkowski dilation: candidate must be at least `buffer` metres
    # from the nearest point on the target boundary.
    expanded_poly = poly.buffer(buffer)
    xx, yy = expanded_poly.exterior.xy
    expanded = np.column_stack([np.array(xx), np.array(yy)])
    excl_path = MplPath(expanded)
    outside = ~excl_path.contains_points(cand)
    return cand[outside], expanded

grid, exclusion = build_grid(boundary_np)
print(f"Grid (pad={GRID_PAD/D:.0f}D, spacing={GRID_SPACING/D:.0f}D): {len(grid)} candidates")

# Liberal layout from cross-section
lib = json.load(open("analysis/cross_section_fixed/dei_d5/Nt50/results.json"))
lib_x = np.array(lib["liberal_x"])
lib_y = np.array(lib["liberal_y"])

# Load DEI wind rose (24 bins, mean speed per bin)
import pandas as pd
df = pd.read_csv("energy_island_10y_daily_av_wind.csv", sep=";")
wd_ts, ws_ts = df["WD_150"].values, df["WS_150"].values
n_bins = 24
edges = np.linspace(0, 360, n_bins + 1)
centers = (edges[:-1] + edges[1:]) / 2
w = np.zeros(n_bins); means = np.zeros(n_bins)
for i in range(n_bins):
    if i == n_bins - 1:
        mask = (wd_ts >= edges[i]) | (wd_ts < edges[0])
    else:
        mask = (wd_ts >= edges[i]) & (wd_ts < edges[i+1])
    w[i] = mask.sum()
    means[i] = ws_ts[mask].mean() if mask.sum() > 0 else ws_ts.mean()
w /= w.sum()
wd_j = jnp.array(centers); ws_j = jnp.array(means); weights_j = jnp.array(w)

turbine = create_dei_turbine()
sim = WakeSimulation(turbine, BastankhahGaussianDeficit(k=0.04))

# Run mini-greedy using screening-only to populate placed positions
# consistent with the 30D exclusion. 5 placements is enough for illustration.
STEP = 5
placed_x = np.zeros(0)
placed_y = np.zeros(0)

n_target = len(lib_x)
lib_x_j = jnp.array(lib_x); lib_y_j = jnp.array(lib_y)

def aep_target_only(x_all, y_all, n_tgt):
    res = sim(x_all, y_all, ws_amb=ws_j, wd_amb=wd_j)
    p_kw = res.power()
    tgt_p = jnp.sum(p_kw[:, :n_tgt] * weights_j[:, None], axis=0)
    return 8760.0 * jnp.sum(tgt_p) / 1e6  # GWh

lib_aep = float(aep_target_only(lib_x_j, lib_y_j, n_target))
print(f"Liberal AEP = {lib_aep:.2f} GWh")

def compute_loss_field(placed_x_arr, placed_y_arr, baseline_aep):
    px_j = jnp.array(placed_x_arr); py_j = jnp.array(placed_y_arr)
    def loss_one(cx, cy):
        nb_x = jnp.concatenate([px_j, jnp.array([cx])])
        nb_y = jnp.concatenate([py_j, jnp.array([cy])])
        x_all = jnp.concatenate([lib_x_j, nb_x])
        y_all = jnp.concatenate([lib_y_j, nb_y])
        return baseline_aep - aep_target_only(x_all, y_all, n_target)
    loss_batch = jax.jit(jax.vmap(loss_one))
    out = np.zeros(len(grid))
    CHUNK = 100
    for i0 in range(0, len(grid), CHUNK):
        i1 = min(i0 + CHUNK, len(grid))
        cx = jnp.array(grid[i0:i1, 0]); cy = jnp.array(grid[i0:i1, 1])
        out[i0:i1] = np.array(loss_batch(cx, cy))
    return out

# Greedy-by-screening: 5 placements. Each iter: compute baseline with placed,
# compute field, place top-1, repeat.
placed_x = []
placed_y = []
for k in range(STEP):
    if len(placed_x) == 0:
        baseline_aep = lib_aep
    else:
        ba = jnp.concatenate([lib_x_j, jnp.array(placed_x)])
        bb = jnp.concatenate([lib_y_j, jnp.array(placed_y)])
        baseline_aep = float(aep_target_only(ba, bb, n_target))
    field = compute_loss_field(np.array(placed_x), np.array(placed_y), baseline_aep)
    # mask out positions too close to any already placed (one-grid-cell exclusion)
    if len(placed_x) > 0:
        dists = np.sqrt((grid[:, 0:1] - np.array(placed_x)[None]) ** 2
                        + (grid[:, 1:2] - np.array(placed_y)[None]) ** 2).min(axis=1)
        field = np.where(dists < GRID_SPACING * 0.5, -np.inf, field)
    best = int(np.argmax(field))
    placed_x.append(float(grid[best, 0]))
    placed_y.append(float(grid[best, 1]))
    print(f"  step {k+1}: placed at ({grid[best,0]/1000:.2f}, {grid[best,1]/1000:.2f}) km, top loss = {field[best]:.2f} GWh")

placed_x = np.array(placed_x); placed_y = np.array(placed_y)

# Final snapshot: compute field with STEP placements for plotting
ba = jnp.concatenate([lib_x_j, jnp.array(placed_x)])
bb = jnp.concatenate([lib_y_j, jnp.array(placed_y)])
baseline_aep = float(aep_target_only(ba, bb, n_target))
aep_loss = compute_loss_field(placed_x, placed_y, baseline_aep)

# Plot
fig, ax = plt.subplots(1, 1, figsize=(11, 11))
sc = ax.scatter(grid[:, 0] / 1000, grid[:, 1] / 1000, c=aep_loss,
                cmap="YlOrRd", s=50, marker="s",
                edgecolor="none", alpha=1.0, vmin=0, zorder=3)
cbar = fig.colorbar(sc, ax=ax, shrink=0.75,
                    label="Incremental AEP loss if placed here (GWh)")

poly = MplPolygon(boundary_np / 1000, fill=False, edgecolor="black",
                  linewidth=1.5, zorder=4)
ax.add_patch(poly)
ax.scatter(lib_x / 1000, lib_y / 1000, s=30, c="white",
           edgecolor="black", linewidth=0.8, zorder=6, label="Target (liberal)")
excl_poly = MplPolygon(exclusion / 1000, fill=False, edgecolor="gray",
                       linewidth=0.8, linestyle="--", zorder=4,
                       label="Exclusion zone ($30D$ buffer)")
ax.add_patch(excl_poly)
ax.scatter(placed_x / 1000, placed_y / 1000, s=150, c="crimson",
           marker="X", edgecolor="black", linewidth=1.2, zorder=7,
           label="Already placed")

ax.set_xlabel("x (km)"); ax.set_ylabel("y (km)")
ax.set_aspect("equal")
ax.legend(loc="upper left", fontsize=10, framealpha=0.95)
# Wind rose inset
ax_wr = inset_axes(ax, width="22%", height="22%", loc="upper right",
                   axes_class=PolarAxes)
ax_wr.set_theta_zero_location("N")
ax_wr.set_theta_direction(-1)
wr_width = np.radians(360 / len(centers)) * 0.9
ax_wr.bar(np.radians(centers), w, width=wr_width,
          color="steelblue", alpha=0.75, edgecolor="navy", linewidth=0.3)
ax_wr.set_yticks([])
ax_wr.set_xticks([])
ax_wr.patch.set_alpha(0.85)
ax_wr.set_title("DEI wind rose", fontsize=9, pad=4)

ax.set_title(f"Greedy-grid screening snapshot (step {STEP+1}): AEP deficit field\n"
             f"Grid: {len(grid)} candidates, $5D$ spacing, $30D$ exclusion buffer around target. "
             f"Top 20 (by AEP loss) advance to full $K$-start re-optimization.",
             fontsize=11)
ax.grid(True, alpha=0.25)

out = Path("paper_v3/figures/greedy_screening_snapshot.png")
fig.savefig(str(out), dpi=180, bbox_inches="tight")
print(f"Saved: {out}")

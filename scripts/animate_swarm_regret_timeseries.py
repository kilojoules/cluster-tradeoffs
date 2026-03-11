"""Cinematic animation: adversarial swarm with realistic time-series wind data.

Like animate_swarm_regret.py but uses 10 years of daily-average wind data
from the Danish Energy Island site (energy_island_10y_daily_av_wind.csv).
With multi-directional wind, adversarial neighbours can't just line up
upwind — they must find positions that maximize regret across all wind
directions weighted by frequency, producing more interesting "swarming"
behaviour.

Usage:
    pixi run python scripts/animate_swarm_regret_timeseries.py
"""

import jax

jax.config.update("jax_enable_x64", True)

import time
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jax import value_and_grad
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon as MplPolygon

from pixwake import WakeSimulation
from pixwake.definitions.v80 import vestas_v80
from pixwake.deficit import BastankhahGaussianDeficit
from pixwake.optim.sgd import (
    SGDSettings,
    sgd_solve_implicit_multistart,
    topfarm_sgd_solve_multistart,
)

# =============================================================================
# Configuration
# =============================================================================
D = vestas_v80.rotor_diameter  # 80 m
N_TARGET = 9
N_NEIGHBOR = 8
K_STARTS = 4       # fewer starts since 24 conditions (vs 8 for single)
OUTER_ITERS = 100
INNER_MAX_ITER = 500
LR_OUTER = 40.0    # aggressive but not oscillatory
BUFFER = 1.5 * D

FLOW_RES = 140
OUTPUT_DIR = Path("analysis")
OUTPUT_MP4 = OUTPUT_DIR / "swarm_regret_timeseries.mp4"
OUTPUT_PNG = OUTPUT_DIR / "swarm_regret_timeseries_final.png"

# =============================================================================
# Wind data loading + binning
# =============================================================================
CSV_PATH = Path("energy_island_10y_daily_av_wind.csv")

print(f"Loading wind data from {CSV_PATH} ...")
df = pd.read_csv(CSV_PATH, sep=";")
wd_ts = df["WD_150"].values
ws_ts = df["WS_150"].values
print(f"  Loaded {len(df)} daily samples")


def compute_binned_wind_rose(wd_ts, ws_ts, n_bins=24):
    """Compute binned wind rose from time series data."""
    bin_edges = np.linspace(0, 360, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    weights = np.zeros(n_bins)
    mean_speeds = np.zeros(n_bins)

    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (wd_ts >= bin_edges[i]) | (wd_ts < bin_edges[0])
        else:
            mask = (wd_ts >= bin_edges[i]) & (wd_ts < bin_edges[i + 1])

        weights[i] = mask.sum()
        if mask.sum() > 0:
            mean_speeds[i] = ws_ts[mask].mean()
        else:
            mean_speeds[i] = ws_ts.mean()

    weights = weights / weights.sum()

    return jnp.array(bin_centers), jnp.array(mean_speeds), jnp.array(weights)


wd_bins, ws_bins, weights = compute_binned_wind_rose(wd_ts, ws_ts, n_bins=24)

# Print wind rose summary
dominant_idx = int(jnp.argmax(weights))
print(f"  Wind rose: {len(wd_bins)} bins, weights sum = {float(weights.sum()):.4f}")
print(f"  Dominant direction: {float(wd_bins[dominant_idx]):.0f} deg "
      f"(weight = {float(weights[dominant_idx]):.3f})")
print(f"  Mean speed: {float(jnp.sum(ws_bins * weights)):.1f} m/s")

# =============================================================================
# Turbine & simulation
# =============================================================================
sim = WakeSimulation(vestas_v80, BastankhahGaussianDeficit(k=0.04))

# =============================================================================
# Target farm: 9 turbines in an 8D x 8D square
# =============================================================================
boundary_size = 8 * D  # 640 m
boundary = jnp.array([
    [0.0, 0.0],
    [boundary_size, 0.0],
    [boundary_size, boundary_size],
    [0.0, boundary_size],
])
min_spacing = 2 * D  # 160 m

# 3 x 3 grid with ~3D spacing, centred in the boundary
offsets = jnp.array([2 * D, 4 * D, 6 * D])
gx, gy = jnp.meshgrid(offsets, offsets)
init_target_x = gx.ravel()
init_target_y = gy.ravel()

# K random starting layouts for multistart inner solver
rng_starts = np.random.RandomState(42)
init_x_batch = jnp.array(np.stack([
    np.array(init_target_x) + rng_starts.uniform(-0.8 * D, 0.8 * D, N_TARGET)
    for _ in range(K_STARTS)
]))
init_y_batch = jnp.array(np.stack([
    np.array(init_target_y) + rng_starts.uniform(-0.8 * D, 0.8 * D, N_TARGET)
    for _ in range(K_STARTS)
]))

# Mutable warm-start batch (slot 0 is overwritten each iteration)
warm_x_batch = jnp.array(init_x_batch)
warm_y_batch = jnp.array(init_y_batch)

# =============================================================================
# Neighbour swarm: compact cluster upwind of dominant direction
# =============================================================================
rng = np.random.RandomState(7)
center_x = boundary_size / 2
center_y = boundary_size / 2
# Place cluster upwind of dominant direction (~278°, roughly west)
# Wind FROM 278° → upwind is TOWARD 278° from farm centre
cluster_dist = 6 * D  # distance from farm centre to cluster centre
dom_dir_rad = np.deg2rad(278.0)  # dominant wind direction
# Meteorological: 0°=N(+y), 90°=E(+x), so dx=sin(θ), dy=cos(θ)
cluster_cx = center_x + cluster_dist * np.sin(dom_dir_rad)
cluster_cy = center_y + cluster_dist * np.cos(dom_dir_rad)
# Scatter within ~3D radius with jitter
cluster_spread = 3 * D
init_nb_x = jnp.array(cluster_cx + rng.uniform(-cluster_spread, cluster_spread, N_NEIGHBOR))
init_nb_y = jnp.array(cluster_cy + rng.uniform(-cluster_spread, cluster_spread, N_NEIGHBOR))

# Neighbour search boundary — large box
nb_boundary = jnp.array([
    [-10 * D, -6 * D],
    [boundary_size + 10 * D, -6 * D],
    [boundary_size + 10 * D, boundary_size + 6 * D],
    [-10 * D, boundary_size + 6 * D],
])

sgd_settings = SGDSettings(learning_rate=10.0, max_iter=INNER_MAX_ITER, tol=1e-6)

# =============================================================================
# Flow-map grid (fixed, covers full area + wakes in all directions)
# =============================================================================
FM_PAD = 10 * D
FM_X_LO, FM_X_HI = -FM_PAD, boundary_size + FM_PAD
FM_Y_LO, FM_Y_HI = -FM_PAD, boundary_size + FM_PAD
fm_nx = FLOW_RES
fm_ny = int(FLOW_RES * (FM_Y_HI - FM_Y_LO) / (FM_X_HI - FM_X_LO))
fm_x1d = np.linspace(FM_X_LO, FM_X_HI, fm_nx)
fm_y1d = np.linspace(FM_Y_LO, FM_Y_HI, fm_ny)
fm_xx, fm_yy = np.meshgrid(fm_x1d, fm_y1d)
fm_x_flat = jnp.array(fm_xx.ravel())
fm_y_flat = jnp.array(fm_yy.ravel())


# =============================================================================
# Objective functions (frequency-weighted AEP)
# =============================================================================
def objective_with_neighbors(x, y, neighbor_params):
    n = neighbor_params.shape[0] // 2
    x_all = jnp.concatenate([x, neighbor_params[:n]])
    y_all = jnp.concatenate([y, neighbor_params[n:]])
    result = sim(x_all, y_all, ws_amb=ws_bins, wd_amb=wd_bins)
    power = result.power()[:, :N_TARGET]  # (24, n_target)
    aep = jnp.sum(power.sum(axis=1) * weights) * 8760 / 1e6
    return -aep


def liberal_objective(x, y):
    result = sim(x, y, ws_amb=ws_bins, wd_amb=wd_bins)
    power = result.power()  # (24, n_target)
    aep = jnp.sum(power.sum(axis=1) * weights) * 8760 / 1e6
    return -aep


# =============================================================================
# Phase 1 — liberal baseline (multistart, no neighbours)
# =============================================================================
print(f"\nPhase 1: computing liberal baseline (K={K_STARTS} starts) ...")
t0 = time.time()
lib_all_x, lib_all_y, lib_objs = topfarm_sgd_solve_multistart(
    liberal_objective, init_x_batch, init_y_batch,
    boundary, min_spacing, sgd_settings,
)
k_best = int(jnp.argmin(lib_objs))
liberal_x, liberal_y = lib_all_x[k_best], lib_all_y[k_best]

# Compute liberal AEP (weighted)
r_lib_isolated = sim(liberal_x, liberal_y, ws_amb=ws_bins, wd_amb=wd_bins)
p_lib_isolated = r_lib_isolated.power()  # (24, n_target)
liberal_aep = float(jnp.sum(p_lib_isolated.sum(axis=1) * weights) * 8760 / 1e6)
print(f"  Liberal AEP = {liberal_aep:.3f} GWh  (best of K={K_STARTS}, {time.time()-t0:.1f}s)")


# =============================================================================
# Phase 2 — bilevel optimisation with warm-started multistart inner solver
# =============================================================================
def compute_regret(neighbor_params):
    n = neighbor_params.shape[0] // 2
    nb_x, nb_y = neighbor_params[:n], neighbor_params[n:]
    # Liberal layout WITH neighbours (fixed layout, differentiable w.r.t. nb)
    x_lib = jnp.concatenate([liberal_x, nb_x])
    y_lib = jnp.concatenate([liberal_y, nb_y])
    r_lib = sim(x_lib, y_lib, ws_amb=ws_bins, wd_amb=wd_bins)
    p_lib = r_lib.power()[:, :N_TARGET]  # (24, n_target)
    liberal_aep_present = jnp.sum(p_lib.sum(axis=1) * weights) * 8760 / 1e6
    # Conservative layout via warm-started multistart IFT (envelope theorem)
    opt_x, opt_y = sgd_solve_implicit_multistart(
        objective_with_neighbors,
        warm_x_batch,
        warm_y_batch,
        neighbor_params,
        boundary, min_spacing, sgd_settings,
    )
    x_con = jnp.concatenate([opt_x, nb_x])
    y_con = jnp.concatenate([opt_y, nb_y])
    r_con = sim(x_con, y_con, ws_amb=ws_bins, wd_amb=wd_bins)
    p_con = r_con.power()[:, :N_TARGET]  # (24, n_target)
    conservative_aep = jnp.sum(p_con.sum(axis=1) * weights) * 8760 / 1e6
    return conservative_aep - liberal_aep_present


regret_and_grad = value_and_grad(compute_regret)

# ADAM state
neighbor_params = jnp.concatenate([init_nb_x, init_nb_y])
m_adam = jnp.zeros_like(neighbor_params)
v_adam = jnp.zeros_like(neighbor_params)
beta1, beta2, adam_eps = 0.9, 0.999, 1e-8

# History arrays
hist_regret = []
hist_grad_norm = []
hist_nb = []       # (nb_x, nb_y) numpy arrays per iteration
hist_target = []   # (opt_x, opt_y) numpy arrays per iteration

print(f"\nPhase 2: bilevel IFT + warm-started multistart "
      f"({OUTER_ITERS} outer x K={K_STARTS} inner, 24 wind dirs) ...")
print(f"{'iter':>4}  {'regret':>10}  {'best':>10}  {'|grad|':>10}  {'dt':>6}")
print("-" * 54)
t0 = time.time()
best_regret = -1e10

for i in range(OUTER_ITERS):
    t_step = time.time()
    regret, grad = regret_and_grad(neighbor_params)

    if not jnp.all(jnp.isfinite(grad)):
        print(f"  !! NaN gradient at iter {i}, stopping")
        break

    regret_val = float(regret)
    grad_norm = float(jnp.linalg.norm(grad))
    best_regret = max(best_regret, regret_val)
    nb_x_np = np.array(neighbor_params[:N_NEIGHBOR])
    nb_y_np = np.array(neighbor_params[N_NEIGHBOR:])
    hist_regret.append(regret_val)
    hist_grad_norm.append(grad_norm)
    hist_nb.append((nb_x_np.copy(), nb_y_np.copy()))

    # Recompute conservative layout for visualization
    opt_x, opt_y = sgd_solve_implicit_multistart(
        objective_with_neighbors,
        warm_x_batch, warm_y_batch, neighbor_params,
        boundary, min_spacing, sgd_settings,
    )
    hist_target.append((np.array(opt_x), np.array(opt_y)))

    # Warm-start: inject current best target layout into slot 0
    warm_x_batch = warm_x_batch.at[0].set(opt_x)
    warm_y_batch = warm_y_batch.at[0].set(opt_y)

    dt = time.time() - t_step
    print(f"{i:4d}  {regret_val:10.4f}  {best_regret:10.4f}  {grad_norm:10.4e}  {dt:5.1f}s")

    # ADAM update (gradient ascent)
    t = i + 1
    m_adam = beta1 * m_adam + (1 - beta1) * grad
    v_adam = beta2 * v_adam + (1 - beta2) * grad**2
    m_hat = m_adam / (1 - beta1**t)
    v_hat = v_adam / (1 - beta2**t)
    neighbor_params = neighbor_params + LR_OUTER * m_hat / (jnp.sqrt(v_hat) + adam_eps)

    # Clip to neighbour boundary
    nb_x_new = jnp.clip(neighbor_params[:N_NEIGHBOR],
                         nb_boundary[:, 0].min(), nb_boundary[:, 0].max())
    nb_y_new = jnp.clip(neighbor_params[N_NEIGHBOR:],
                         nb_boundary[:, 1].min(), nb_boundary[:, 1].max())

    # Buffer: push neighbours outside target boundary + BUFFER
    # Distance to each face of the buffered rectangle
    d_left = nb_x_new - (-BUFFER)
    d_right = (boundary_size + BUFFER) - nb_x_new
    d_bottom = nb_y_new - (-BUFFER)
    d_top = (boundary_size + BUFFER) - nb_y_new
    # Neighbor is inside forbidden zone if all four distances are positive
    inside = (d_left > 0) & (d_right > 0) & (d_bottom > 0) & (d_top > 0)
    # Find nearest face and project through it
    min_d = jnp.minimum(jnp.minimum(d_left, d_right),
                        jnp.minimum(d_bottom, d_top))
    nb_x_new = jnp.where(inside & (d_left <= min_d), -BUFFER, nb_x_new)
    nb_x_new = jnp.where(inside & (d_right <= min_d) & (d_left > min_d),
                          boundary_size + BUFFER, nb_x_new)
    nb_y_new = jnp.where(inside & (d_bottom <= min_d) & (d_left > min_d) & (d_right > min_d),
                          -BUFFER, nb_y_new)
    nb_y_new = jnp.where(inside & (d_top <= min_d) & (d_left > min_d) & (d_right > min_d) & (d_bottom > min_d),
                          boundary_size + BUFFER, nb_y_new)
    neighbor_params = jnp.concatenate([nb_x_new, nb_y_new])

elapsed = time.time() - t0
n_frames = len(hist_regret)
print(f"\nOptimisation done: {n_frames} iterations in {elapsed:.1f}s")
if n_frames:
    print(f"  Initial regret: {hist_regret[0]:.4f} GWh")
    print(f"  Final regret:   {hist_regret[-1]:.4f} GWh")
    print(f"  Best regret:    {max(hist_regret):.4f} GWh")

# Print initial -> final positions for verification
print("\nNeighbour displacement:")
for k in range(N_NEIGHBOR):
    x0, y0 = float(init_nb_x[k]), float(init_nb_y[k])
    x1, y1 = float(hist_nb[-1][0][k]), float(hist_nb[-1][1][k])
    d = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
    print(f"  nb[{k}]: ({x0:7.1f}, {y0:7.1f}) -> ({x1:7.1f}, {y1:7.1f})  "
          f"d={d:.0f}m = {d/D:.1f}D")

# =============================================================================
# Phase 3 — compute weighted-average wake flow maps for each frame
# =============================================================================
print(f"\nPhase 3: computing {n_frames} weighted flow maps (24 wind dirs each) ...")
t0 = time.time()
hist_deficit = []

DEFICIT_MAX = 0.15  # lower than single-direction (weighted avg dilutes peaks)

for i in range(n_frames):
    nb_x, nb_y = hist_nb[i]
    tgt_x, tgt_y = hist_target[i]
    x_all = jnp.concatenate([jnp.array(tgt_x), jnp.array(nb_x)])
    y_all = jnp.concatenate([jnp.array(tgt_y), jnp.array(nb_y)])
    flow_ws_all, _ = sim.flow_map(
        x_all, y_all, fm_x=fm_x_flat, fm_y=fm_y_flat,
        ws=ws_bins, wd=wd_bins,
    )
    # flow_ws_all shape: (24, n_flow_points)
    deficits = 1.0 - flow_ws_all / ws_bins[:, None]
    weighted_deficit = jnp.sum(deficits * weights[:, None], axis=0)
    hist_deficit.append(np.array(weighted_deficit).reshape(fm_ny, fm_nx))

print(f"  Flow maps done in {time.time()-t0:.1f}s")

# =============================================================================
# Phase 4 — cinematic MP4 rendering
# =============================================================================
print(f"\nPhase 4: rendering {OUTPUT_MP4} ...")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -- Dark theme --
plt.rcParams.update({
    "figure.facecolor": "#0a0a1a",
    "axes.facecolor": "#0d1b2e",
    "axes.edgecolor": "#334466",
    "axes.labelcolor": "#cccccc",
    "text.color": "#cccccc",
    "xtick.color": "#888888",
    "ytick.color": "#888888",
    "grid.color": "#1a2a44",
    "grid.alpha": 0.5,
    "font.family": "monospace",
})

# Custom colormap: dark blue (no deficit) -> teal -> orange -> yellow
wake_cmap = LinearSegmentedColormap.from_list("wake", [
    (0.00, "#0d1b2e"),
    (0.05, "#0f2847"),
    (0.15, "#1b4965"),
    (0.30, "#c06030"),
    (0.50, "#e8a030"),
    (1.00, "#ffe066"),
])

# Per-neighbour trail colors (6 distinct warm tones)
NB_COLORS = ["#ff4444", "#ff7733", "#ffaa22", "#ee55cc", "#cc66ff", "#ff6688",
             "#44ddaa", "#88bbff"]

# Pre-compute
lib_x_np = np.array(liberal_x)
lib_y_np = np.array(liberal_y)
bnd_np = np.array(boundary)
hist_best_regret = np.maximum.accumulate(hist_regret)
weights_np = np.array(weights)
wd_bins_np = np.array(wd_bins)
ws_bins_np = np.array(ws_bins)

# -- Figure layout: 4 rows, 2 columns --
fig = plt.figure(figsize=(18, 11))
gs = fig.add_gridspec(
    4, 2, width_ratios=[1.6, 1], hspace=0.45, wspace=0.22,
    left=0.04, right=0.97, top=0.93, bottom=0.05,
)
ax_wake = fig.add_subplot(gs[:, 0])
ax_reg = fig.add_subplot(gs[0, 1])
ax_aep = fig.add_subplot(gs[1, 1])
ax_rose = fig.add_subplot(gs[2, 1], projection="polar")
ax_grad = fig.add_subplot(gs[3, 1])

# Pre-compute fixed metric limits
reg_lo = min(hist_regret) * 0.9 if hist_regret else 0
reg_hi = max(hist_regret) * 1.15 if hist_regret else 1
gn_lo = min(hist_grad_norm) * 0.3 if hist_grad_norm else 1e-6
gn_hi = max(hist_grad_norm) * 3.0 if hist_grad_norm else 1

# Wind rose bar width (each bin spans 15 degrees = pi/12 radians)
rose_width = 2 * np.pi / len(wd_bins_np)
# Convert meteorological degrees to polar radians (N-up, clockwise)
rose_theta = np.deg2rad(wd_bins_np)


def draw_frame(idx):
    """Render a single animation frame."""
    for ax in (ax_wake, ax_reg, ax_aep, ax_grad):
        ax.clear()
    ax_rose.clear()

    nb_x, nb_y = hist_nb[idx]
    tgt_x, tgt_y = hist_target[idx]
    deficit = hist_deficit[idx]

    # -- Wake field heatmap (weighted average across all directions) --
    ax_wake.pcolormesh(
        fm_xx, fm_yy, deficit,
        cmap=wake_cmap, vmin=0, vmax=DEFICIT_MAX,
        shading="auto", rasterized=True,
    )

    # Target boundary
    bnd_patch = MplPolygon(
        bnd_np, closed=True, fill=False,
        edgecolor="#ffffff", linewidth=1.5, alpha=0.6, linestyle="-",
    )
    ax_wake.add_patch(bnd_patch)

    # Buffer zone (dashed)
    buf_verts = np.array([
        [-BUFFER, -BUFFER],
        [boundary_size + BUFFER, -BUFFER],
        [boundary_size + BUFFER, boundary_size + BUFFER],
        [-BUFFER, boundary_size + BUFFER],
    ])
    buf_patch = MplPolygon(
        buf_verts, closed=True, fill=False,
        edgecolor="#ffffff", linewidth=0.8, alpha=0.25, linestyle="--",
    )
    ax_wake.add_patch(buf_patch)

    # Liberal turbines (fixed, hollow circles)
    ax_wake.scatter(
        lib_x_np, lib_y_np, s=70, facecolors="none",
        edgecolors="#66ccff", linewidths=1.5, zorder=5, label="Liberal (naive)",
    )

    # Conservative turbines (filled circles)
    ax_wake.scatter(
        tgt_x, tgt_y, s=70, c="#44dd88",
        edgecolors="#225533", linewidths=0.8, zorder=6, label="Conservative",
    )

    # Displacement arrows (liberal -> conservative, actual scale)
    for k in range(N_TARGET):
        ddx = tgt_x[k] - lib_x_np[k]
        ddy = tgt_y[k] - lib_y_np[k]
        if np.sqrt(ddx**2 + ddy**2) > 1.0:
            ax_wake.annotate(
                "", xy=(tgt_x[k], tgt_y[k]),
                xytext=(lib_x_np[k], lib_y_np[k]),
                arrowprops=dict(arrowstyle="-|>", color="#44dd88",
                                lw=1.0, alpha=0.5),
                zorder=4,
            )

    # Neighbour trails (per-neighbour colored lines with fading segments)
    if idx > 0:
        for k in range(N_NEIGHBOR):
            trail_x = [hist_nb[j][0][k] for j in range(idx + 1)]
            trail_y = [hist_nb[j][1][k] for j in range(idx + 1)]
            col = NB_COLORS[k % len(NB_COLORS)]
            ax_wake.plot(trail_x, trail_y, color=col, lw=1.2, alpha=0.35, zorder=3)
            n_recent = min(8, idx)
            for j in range(idx - n_recent, idx):
                seg_alpha = 0.15 + 0.55 * ((j - (idx - n_recent)) / n_recent)
                ax_wake.plot(
                    [trail_x[j], trail_x[j + 1]],
                    [trail_y[j], trail_y[j + 1]],
                    color=col, lw=2.5, alpha=seg_alpha, zorder=7,
                    solid_capstyle="round",
                )

    # Current neighbour positions (bright diamonds)
    for k in range(N_NEIGHBOR):
        col = NB_COLORS[k % len(NB_COLORS)]
        ax_wake.scatter(
            [nb_x[k]], [nb_y[k]], s=130, c=col, marker="D",
            edgecolors="#ffffff", linewidths=1.2, zorder=8,
        )
    ax_wake.scatter([], [], s=80, c="#ff4444", marker="D",
                    edgecolors="#ffffff", linewidths=1.0,
                    label="Neighbours (swarm)")

    # Gradient arrows on neighbours (show next step direction)
    if idx + 1 < n_frames:
        next_x, next_y = hist_nb[idx + 1]
        for k in range(N_NEIGHBOR):
            ddx = next_x[k] - nb_x[k]
            ddy = next_y[k] - nb_y[k]
            norm = np.sqrt(ddx**2 + ddy**2)
            if norm > 1.0:
                arrow_len = min(norm, 3 * D)
                ax_wake.annotate(
                    "",
                    xy=(nb_x[k] + ddx / norm * arrow_len,
                        nb_y[k] + ddy / norm * arrow_len),
                    xytext=(nb_x[k], nb_y[k]),
                    arrowprops=dict(
                        arrowstyle="-|>", color="#ffcc00",
                        lw=1.8, alpha=0.7,
                    ),
                    zorder=9,
                )

    ax_wake.set_xlim(FM_X_LO, FM_X_HI)
    ax_wake.set_ylim(FM_Y_LO, FM_Y_HI)
    ax_wake.set_aspect("equal")
    ax_wake.set_xlabel("x  (m)")
    ax_wake.set_ylabel("y  (m)")
    ax_wake.legend(
        loc="lower left", fontsize=7.5, framealpha=0.6,
        facecolor="#0d1b2e", edgecolor="#334466",
    )

    # -- Regret convergence --
    ax_reg.plot(hist_regret[:idx + 1], "-o", color="#ffcc00",
                markersize=2.5, lw=1.2, markeredgewidth=0, alpha=0.6,
                label="Instantaneous")
    ax_reg.plot(hist_best_regret[:idx + 1], "-", color="#ff4444",
                lw=2.2, label="Running best")
    ax_reg.axvline(idx, color="#ffcc00", alpha=0.2, lw=0.8)
    ax_reg.set_xlim(-0.5, n_frames - 0.5)
    ax_reg.set_ylim(reg_lo, reg_hi)
    ax_reg.set_ylabel("Regret (GWh)")
    ax_reg.set_title(
        f"Regret = {hist_regret[idx]:.2f}  |  Best = {hist_best_regret[idx]:.2f} GWh",
        fontsize=9,
    )
    ax_reg.legend(loc="lower right", fontsize=7, framealpha=0.5,
                  facecolor="#0d1b2e", edgecolor="#334466")
    ax_reg.grid(True)

    # -- AEP bars (frequency-weighted) --
    nb_xj, nb_yj = jnp.array(nb_x), jnp.array(nb_y)
    x_lib_all = jnp.concatenate([liberal_x, nb_xj])
    y_lib_all = jnp.concatenate([liberal_y, nb_yj])
    r_lib = sim(x_lib_all, y_lib_all, ws_amb=ws_bins, wd_amb=wd_bins)
    p_lib = r_lib.power()[:, :N_TARGET]  # (24, n_target)
    lib_aep_present = float(jnp.sum(p_lib.sum(axis=1) * weights) * 8760 / 1e6)
    cons_aep = lib_aep_present + hist_regret[idx]

    bar_x = [0, 1, 2]
    bar_vals = [liberal_aep, lib_aep_present, cons_aep]
    bar_colors = ["#2277bb", "#cc4444", "#44dd88"]
    bar_labels = ["Liberal\n(isolated)", "Liberal\n(w/ neighbours)", "Conservative\n(w/ neighbours)"]
    bars = ax_aep.bar(bar_x, bar_vals, color=bar_colors, width=0.6,
                      edgecolor="#ffffff", linewidth=0.5)
    ax_aep.set_xticks(bar_x)
    ax_aep.set_xticklabels(bar_labels, fontsize=7)
    ax_aep.set_ylabel("AEP (GWh)")
    aep_floor = min(bar_vals) * 0.92
    aep_ceil = liberal_aep * 1.05
    ax_aep.set_ylim(aep_floor, aep_ceil)
    ax_aep.set_title("Annual Energy Production (weighted)", fontsize=10)
    for b, v in zip(bars, bar_vals):
        ax_aep.text(b.get_x() + b.get_width() / 2,
                    v + (aep_ceil - aep_floor) * 0.02,
                    f"{v:.2f}", ha="center", fontsize=7.5, color="#cccccc")

    # -- Wind rose (polar bar chart, N-up clockwise) --
    ax_rose.set_theta_zero_location("N")
    ax_rose.set_theta_direction(-1)
    rose_colors = ["#ffcc00" if j == dominant_idx else "#4488cc"
                   for j in range(len(wd_bins_np))]
    ax_rose.bar(rose_theta, weights_np, width=rose_width,
                color=rose_colors, edgecolor="#334466", linewidth=0.5, alpha=0.85)
    ax_rose.set_title("Wind Rose (frequency)", fontsize=9, pad=12)
    ax_rose.tick_params(labelsize=6)
    ax_rose.set_facecolor("#0d1b2e")

    # -- Gradient norm --
    ax_grad.semilogy(hist_grad_norm[:idx + 1], "-s", color="#ff8844",
                     markersize=2.5, lw=1.5, markeredgewidth=0)
    ax_grad.axvline(idx, color="#ff8844", alpha=0.2, lw=0.8)
    ax_grad.set_xlim(-0.5, n_frames - 0.5)
    ax_grad.set_ylim(gn_lo, gn_hi)
    ax_grad.set_xlabel("Outer iteration")
    ax_grad.set_ylabel("|grad|")
    ax_grad.set_title(f"|grad| = {hist_grad_norm[idx]:.2e}", fontsize=10)
    ax_grad.grid(True)

    ax_wake.figure.suptitle(
        f"Adversarial Swarm Search (Time-Series Wind)  --  Iteration {idx}/{n_frames - 1}",
        fontsize=14, fontweight="bold", color="#ffffff",
    )


# -- Animate --
hold_frames = 8
total_frames = n_frames + hold_frames


def draw_frame_with_hold(idx):
    draw_frame(min(idx, n_frames - 1))


anim = FuncAnimation(fig, draw_frame_with_hold, frames=total_frames,
                     interval=250, repeat=True)
anim.save(str(OUTPUT_MP4), writer="ffmpeg", fps=5, dpi=150)
plt.close(fig)
print(f"  Saved -> {OUTPUT_MP4}")

# -- Save high-res final frame --
fig2 = plt.figure(figsize=(18, 11))
gs2 = fig2.add_gridspec(
    4, 2, width_ratios=[1.6, 1], hspace=0.45, wspace=0.22,
    left=0.04, right=0.97, top=0.93, bottom=0.05,
)
ax_wake = fig2.add_subplot(gs2[:, 0])
ax_reg = fig2.add_subplot(gs2[0, 1])
ax_aep = fig2.add_subplot(gs2[1, 1])
ax_rose = fig2.add_subplot(gs2[2, 1], projection="polar")
ax_grad = fig2.add_subplot(gs2[3, 1])
draw_frame(n_frames - 1)
fig2.savefig(str(OUTPUT_PNG), dpi=200, facecolor=fig2.get_facecolor())
plt.close(fig2)
print(f"  Saved -> {OUTPUT_PNG}")

print("\nDone.")

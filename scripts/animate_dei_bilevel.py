"""Bilevel IFT adversarial search on the Danish Energy Island (DEI) case.

Applies the multistart envelope-theorem IFT framework to the DEI target farm
with realistic 15 MW turbines, polygon boundary, and 10-year wind data.
The adversarial swarm finds neighbor positions that maximize design regret.

Uses a 20-turbine subset of the full 66-turbine farm (for tractable IFT backward
pass) with the actual dk0w_tender_3 polygon boundary and DEI wind rose.

Usage:
    pixi run python scripts/animate_dei_bilevel.py
"""

import jax

jax.config.update("jax_enable_x64", True)

import gc
import sys
import time
from functools import partial
from pathlib import Path

# Unbuffered stdout so progress is visible when piped/backgrounded
print = partial(print, flush=True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jax import value_and_grad
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.path import Path as MplPath

from pixwake import Curve, Turbine, WakeSimulation
from pixwake.deficit import BastankhahGaussianDeficit
from pixwake.optim.sgd import (
    SGDSettings,
    sgd_solve_implicit_multistart,
    topfarm_sgd_solve_multistart,
)

# =============================================================================
# Configuration
# =============================================================================
N_TARGET = 50
N_NEIGHBOR = 50
K_STARTS = 2
OUTER_ITERS = 1000
INNER_MAX_ITER = 2000
LR_OUTER = 150.0     # km-scale problem needs aggressive LR
D = 240.0          # rotor diameter (m) — IEA 15MW class
MIN_SPACING = 3 * D  # 720 m (slightly under 4D for solver flexibility)
BUFFER = 2 * D       # 480 m exclusion from polygon boundary

FLOW_RES = 120
OUTPUT_DIR = Path("analysis")
OUTPUT_MP4 = OUTPUT_DIR / "dei_bilevel_50x50.mp4"
OUTPUT_PNG = OUTPUT_DIR / "dei_bilevel_50x50_final.png"


# =============================================================================
# DEI turbine (15 MW, D=240m, hub=150m) — matches PyWake GenericWindTurbine
# =============================================================================
def create_dei_turbine():
    ws = jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                    13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25.0])
    power = jnp.array([
        0.0, 0.0, 2.399, 209.258, 689.198, 1480.608,
        2661.238, 4308.929, 6501.057, 9260.516, 12081.404, 13937.297,
        14705.016, 14931.039, 14985.209, 14996.906, 14999.343, 14999.855,
        14999.966, 14999.992, 14999.998, 14999.999, 15000.0, 15000.0,
        15000.0, 15000.0,
    ])
    ct = jnp.array([
        0.889, 0.889, 0.889, 0.800, 0.800, 0.800,
        0.800, 0.800, 0.800, 0.793, 0.735, 0.610,
        0.476, 0.370, 0.292, 0.234, 0.191, 0.158,
        0.132, 0.112, 0.096, 0.083, 0.072, 0.063,
        0.055, 0.049,
    ])
    return Turbine(
        rotor_diameter=240.0,
        hub_height=150.0,
        power_curve=Curve(ws=ws, values=power),
        ct_curve=Curve(ws=ws, values=ct),
    )


# =============================================================================
# DEI polygon boundary (dk0w_tender_3, UTM) — centered at centroid
# =============================================================================
_dk0w_raw = np.array([
    706694.3923283464, 6224158.532895836,
    703972.0844905999, 6226906.597455995,
    702624.6334635273, 6253853.5386425415,
    712771.6248419734, 6257704.934445341,
    715639.3355871611, 6260664.6846508905,
    721593.2420745814, 6257906.998015941,
]).reshape((-1, 2))

# Center coordinates for numerical stability (FD epsilon scales with |param|)
CENTROID_X = _dk0w_raw[:, 0].mean()
CENTROID_Y = _dk0w_raw[:, 1].mean()

# Reverse vertex order: raw polygon is CW, containment_penalty requires CCW
_centered = (_dk0w_raw - np.array([CENTROID_X, CENTROID_Y]))[::-1]
boundary = jnp.array(_centered)  # (6, 2) CCW
_polygon_path = MplPath(_centered)

print(f"DEI polygon: {boundary.shape[0]} vertices, centered at "
      f"({CENTROID_X:.0f}, {CENTROID_Y:.0f}) UTM")
print(f"  x range: [{float(boundary[:,0].min())/1000:.1f}, {float(boundary[:,0].max())/1000:.1f}] km")
print(f"  y range: [{float(boundary[:,1].min())/1000:.1f}, {float(boundary[:,1].max())/1000:.1f}] km")


# =============================================================================
# Wind data loading + binning
# =============================================================================
CSV_PATH = Path("energy_island_10y_daily_av_wind.csv")

print(f"\nLoading wind data from {CSV_PATH} ...")
df = pd.read_csv(CSV_PATH, sep=";")
wd_ts = df["WD_150"].values
ws_ts = df["WS_150"].values
print(f"  Loaded {len(df)} daily samples")


def compute_binned_wind_rose(wd_ts, ws_ts, n_bins=24):
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
        mean_speeds[i] = ws_ts[mask].mean() if mask.sum() > 0 else ws_ts.mean()
    weights = weights / weights.sum()
    return jnp.array(bin_centers), jnp.array(mean_speeds), jnp.array(weights)


wd_bins, ws_bins, weights = compute_binned_wind_rose(wd_ts, ws_ts, n_bins=24)

dominant_idx = int(jnp.argmax(weights))
print(f"  Wind rose: {len(wd_bins)} bins, weights sum = {float(weights.sum()):.4f}")
print(f"  Dominant direction: {float(wd_bins[dominant_idx]):.0f} deg "
      f"(weight = {float(weights[dominant_idx]):.3f})")
print(f"  Mean speed: {float(jnp.sum(ws_bins * weights)):.1f} m/s")

# =============================================================================
# Turbine & simulation
# =============================================================================
turbine = create_dei_turbine()
sim = WakeSimulation(turbine, BastankhahGaussianDeficit(k=0.04))

# =============================================================================
# Target farm: N_TARGET turbines on a grid inside the polygon
# =============================================================================
# Create a grid at ~4D spacing inside the polygon, take N_TARGET points
x_lo, x_hi = float(boundary[:, 0].min()), float(boundary[:, 0].max())
y_lo, y_hi = float(boundary[:, 1].min()), float(boundary[:, 1].max())

grid_spacing = 4 * D  # 960 m
gx_1d = np.arange(x_lo + 2 * D, x_hi - 2 * D, grid_spacing)
gy_1d = np.arange(y_lo + 2 * D, y_hi - 2 * D, grid_spacing)
gx_2d, gy_2d = np.meshgrid(gx_1d, gy_1d)
candidates = np.column_stack([gx_2d.ravel(), gy_2d.ravel()])

# Filter to points inside the polygon
inside_mask = _polygon_path.contains_points(candidates)
inside_pts = candidates[inside_mask]
print(f"\nTarget grid: {len(inside_pts)} points inside polygon at {grid_spacing:.0f}m spacing")

# Subsample to N_TARGET (evenly spaced through the grid)
if len(inside_pts) >= N_TARGET:
    indices = np.round(np.linspace(0, len(inside_pts) - 1, N_TARGET)).astype(int)
    target_pts = inside_pts[indices]
else:
    # Fall back to denser grid
    grid_spacing = 3 * D
    gx_1d = np.arange(x_lo + D, x_hi - D, grid_spacing)
    gy_1d = np.arange(y_lo + D, y_hi - D, grid_spacing)
    gx_2d, gy_2d = np.meshgrid(gx_1d, gy_1d)
    candidates = np.column_stack([gx_2d.ravel(), gy_2d.ravel()])
    inside_mask = _polygon_path.contains_points(candidates)
    inside_pts = candidates[inside_mask]
    indices = np.round(np.linspace(0, len(inside_pts) - 1, N_TARGET)).astype(int)
    target_pts = inside_pts[indices]

init_target_x = jnp.array(target_pts[:, 0])
init_target_y = jnp.array(target_pts[:, 1])
print(f"  Selected {N_TARGET} target turbines")

# K random starting layouts (jitter around grid positions)
rng_starts = np.random.RandomState(42)
init_x_batch = jnp.array(np.stack([
    np.array(init_target_x) + rng_starts.uniform(-0.5 * D, 0.5 * D, N_TARGET)
    for _ in range(K_STARTS)
]))
init_y_batch = jnp.array(np.stack([
    np.array(init_target_y) + rng_starts.uniform(-0.5 * D, 0.5 * D, N_TARGET)
    for _ in range(K_STARTS)
]))

warm_x_batch = jnp.array(init_x_batch)
warm_y_batch = jnp.array(init_y_batch)


# =============================================================================
# Neighbour swarm: scattered around full polygon perimeter
# =============================================================================
_bnd_np = np.array(boundary)
_n_v = _bnd_np.shape[0]
rng = np.random.RandomState(7)

# Sample points along the polygon perimeter, then push outward
_bnd_np_closed = np.vstack([_bnd_np, _bnd_np[0]])  # close the polygon
_edge_lengths = np.sqrt(np.sum(np.diff(_bnd_np_closed, axis=0)**2, axis=1))
_cum_lengths = np.concatenate([[0], np.cumsum(_edge_lengths)])
_perimeter = _cum_lengths[-1]

# Evenly space N_NEIGHBOR points along the perimeter with jitter
nb_standoff = 3 * D  # ~720m outside the boundary
t_values = np.linspace(0, _perimeter, N_NEIGHBOR, endpoint=False)
t_values += rng.uniform(-0.3 * _perimeter / N_NEIGHBOR, 0.3 * _perimeter / N_NEIGHBOR, N_NEIGHBOR)
t_values = t_values % _perimeter

init_nb_x_arr = np.zeros(N_NEIGHBOR)
init_nb_y_arr = np.zeros(N_NEIGHBOR)
for k in range(N_NEIGHBOR):
    # Find which edge this t falls on
    edge_idx = np.searchsorted(_cum_lengths[1:], t_values[k])
    edge_idx = min(edge_idx, _n_v - 1)
    t_local = (t_values[k] - _cum_lengths[edge_idx]) / _edge_lengths[edge_idx]
    t_local = max(0.0, min(1.0, t_local))
    # Interpolate along edge
    ax, ay = _bnd_np[edge_idx]
    bx, by = _bnd_np[(edge_idx + 1) % _n_v]
    foot_x = ax + t_local * (bx - ax)
    foot_y = ay + t_local * (by - ay)
    # Push outward along edge normal (outward = (dy, -dx) for CCW)
    dx, dy = bx - ax, by - ay
    elen = np.sqrt(dx**2 + dy**2)
    out_nx, out_ny = dy / elen, -dx / elen
    standoff = nb_standoff + rng.uniform(0, 2 * D)
    init_nb_x_arr[k] = foot_x + standoff * out_nx
    init_nb_y_arr[k] = foot_y + standoff * out_ny

init_nb_x = jnp.array(init_nb_x_arr)
init_nb_y = jnp.array(init_nb_y_arr)

print(f"\nNeighbour scatter: {N_NEIGHBOR} around polygon perimeter")
print(f"  x range: [{float(init_nb_x.min())/1000:.1f}, {float(init_nb_x.max())/1000:.1f}] km")
print(f"  y range: [{float(init_nb_y.min())/1000:.1f}, {float(init_nb_y.max())/1000:.1f}] km")

# Outer search boundary (large box)
NB_MARGIN = 30 * D  # 7.2 km — wider search area for scattered init
nb_x_lo = x_lo - NB_MARGIN
nb_x_hi = x_hi + NB_MARGIN
nb_y_lo = y_lo - NB_MARGIN
nb_y_hi = y_hi + NB_MARGIN

sgd_settings = SGDSettings(learning_rate=30.0, max_iter=INNER_MAX_ITER, tol=1e-6)


# =============================================================================
# Polygon exclusion: push neighbours outside boundary + BUFFER
# =============================================================================
# The DEI polygon is slightly non-convex, so signed-distance-to-edge-lines
# fails. Use matplotlib ray-casting for the inside test and nearest-edge-segment
# projection for the push direction.


def exclude_from_polygon(nb_x, nb_y):
    """Push neighbour positions outside the target polygon + BUFFER."""
    nb_x_out = np.array(nb_x, dtype=np.float64)
    nb_y_out = np.array(nb_y, dtype=np.float64)

    for k in range(len(nb_x_out)):
        px, py = nb_x_out[k], nb_y_out[k]

        # Find nearest point on polygon boundary (edge segments, not lines)
        best_dist_sq = float("inf")
        best_foot_x, best_foot_y = px, py
        best_out_nx, best_out_ny = 0.0, 0.0

        for i in range(_n_v):
            j = (i + 1) % _n_v
            ax, ay = _bnd_np[i, 0], _bnd_np[i, 1]
            bx, by = _bnd_np[j, 0], _bnd_np[j, 1]
            dx, dy = bx - ax, by - ay
            edge_len_sq = dx * dx + dy * dy

            # Project onto edge segment, clamped to [0, 1]
            t = ((px - ax) * dx + (py - ay) * dy) / edge_len_sq
            t = max(0.0, min(1.0, t))
            foot_x = ax + t * dx
            foot_y = ay + t * dy
            dist_sq = (px - foot_x) ** 2 + (py - foot_y) ** 2

            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_foot_x, best_foot_y = foot_x, foot_y
                # Outward normal for CCW polygon: (dy, -dx) / len
                elen = np.sqrt(edge_len_sq)
                best_out_nx = dy / elen
                best_out_ny = -dx / elen

        best_dist = np.sqrt(best_dist_sq)

        # Ray-casting inside test (robust for any polygon shape)
        inside = _polygon_path.contains_point((px, py))

        if inside or best_dist < BUFFER:
            # Place at nearest boundary point + BUFFER along outward normal
            nb_x_out[k] = best_foot_x + BUFFER * best_out_nx
            nb_y_out[k] = best_foot_y + BUFFER * best_out_ny

    return jnp.array(nb_x_out), jnp.array(nb_y_out)


# =============================================================================
# Flow-map grid
# =============================================================================
FM_PAD = NB_MARGIN + 2 * D  # cover full search area + small margin
FM_X_LO, FM_X_HI = x_lo - FM_PAD, x_hi + FM_PAD
FM_Y_LO, FM_Y_HI = y_lo - FM_PAD, y_hi + FM_PAD
fm_nx = FLOW_RES
fm_ny = int(FLOW_RES * (FM_Y_HI - FM_Y_LO) / (FM_X_HI - FM_X_LO))
fm_x1d = np.linspace(FM_X_LO, FM_X_HI, fm_nx)
fm_y1d = np.linspace(FM_Y_LO, FM_Y_HI, fm_ny)
fm_xx, fm_yy = np.meshgrid(fm_x1d, fm_y1d)
fm_x_flat = jnp.array(fm_xx.ravel())
fm_y_flat = jnp.array(fm_yy.ravel())
print(f"\nFlow map: {fm_nx} x {fm_ny} = {fm_nx*fm_ny} points")


# =============================================================================
# Objective functions (frequency-weighted AEP)
# =============================================================================
def objective_with_neighbors(x, y, neighbor_params):
    n = neighbor_params.shape[0] // 2
    x_all = jnp.concatenate([x, neighbor_params[:n]])
    y_all = jnp.concatenate([y, neighbor_params[n:]])
    result = sim(x_all, y_all, ws_amb=ws_bins, wd_amb=wd_bins)
    power = result.power()[:, :N_TARGET]
    return -jnp.sum(power.sum(axis=1) * weights) * 8760 / 1e6


def liberal_objective(x, y):
    result = sim(x, y, ws_amb=ws_bins, wd_amb=wd_bins)
    power = result.power()
    return -jnp.sum(power.sum(axis=1) * weights) * 8760 / 1e6


# =============================================================================
# Phase 1 — liberal baseline
# =============================================================================
print(f"\nPhase 1: computing liberal baseline (K={K_STARTS} starts, {N_TARGET} turbines) ...")
t0 = time.time()
lib_all_x, lib_all_y, lib_objs = topfarm_sgd_solve_multistart(
    liberal_objective, init_x_batch, init_y_batch,
    boundary, MIN_SPACING, sgd_settings,
)
k_best = int(jnp.argmin(lib_objs))
liberal_x, liberal_y = lib_all_x[k_best], lib_all_y[k_best]

r_lib_isolated = sim(liberal_x, liberal_y, ws_amb=ws_bins, wd_amb=wd_bins)
p_lib_isolated = r_lib_isolated.power()
liberal_aep = float(jnp.sum(p_lib_isolated.sum(axis=1) * weights) * 8760 / 1e6)
print(f"  Liberal AEP = {liberal_aep:.3f} GWh  (best of K={K_STARTS}, {time.time()-t0:.1f}s)")


# =============================================================================
# Phase 2 — bilevel IFT optimisation
# =============================================================================
_captured_layout = {}


def _capture_layout(opt_x, opt_y):
    """Callback to capture the conservative layout during forward pass."""
    _captured_layout['x'] = np.array(opt_x)
    _captured_layout['y'] = np.array(opt_y)


def compute_regret(neighbor_params):
    n = neighbor_params.shape[0] // 2
    nb_x, nb_y = neighbor_params[:n], neighbor_params[n:]
    # Liberal layout WITH neighbours
    x_lib = jnp.concatenate([liberal_x, nb_x])
    y_lib = jnp.concatenate([liberal_y, nb_y])
    r_lib = sim(x_lib, y_lib, ws_amb=ws_bins, wd_amb=wd_bins)
    p_lib = r_lib.power()[:, :N_TARGET]
    liberal_aep_present = jnp.sum(p_lib.sum(axis=1) * weights) * 8760 / 1e6
    # Conservative layout via multistart IFT
    opt_x, opt_y = sgd_solve_implicit_multistart(
        objective_with_neighbors,
        warm_x_batch, warm_y_batch, neighbor_params,
        boundary, MIN_SPACING, sgd_settings,
    )
    # Capture layout for visualization (identity for AD, zero cost)
    jax.debug.callback(_capture_layout, opt_x, opt_y)
    x_con = jnp.concatenate([opt_x, nb_x])
    y_con = jnp.concatenate([opt_y, nb_y])
    r_con = sim(x_con, y_con, ws_amb=ws_bins, wd_amb=wd_bins)
    p_con = r_con.power()[:, :N_TARGET]
    conservative_aep = jnp.sum(p_con.sum(axis=1) * weights) * 8760 / 1e6
    return conservative_aep - liberal_aep_present


regret_and_grad = value_and_grad(compute_regret)

# ADAM state
neighbor_params = jnp.concatenate([init_nb_x, init_nb_y])
m_adam = jnp.zeros_like(neighbor_params)
v_adam = jnp.zeros_like(neighbor_params)
beta1, beta2, adam_eps = 0.9, 0.999, 1e-8

hist_regret = []
hist_grad_norm = []
hist_nb = []
hist_target = []

print(f"\nPhase 2: bilevel IFT + multistart "
      f"({OUTER_ITERS} outer x K={K_STARTS} inner, 24 wind dirs, {N_TARGET} targets) ...")
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
    if regret_val > best_regret:
        best_regret = regret_val
        best_neighbor_params = jnp.array(neighbor_params)
        best_iter = i
    nb_x_np = np.array(neighbor_params[:N_NEIGHBOR])
    nb_y_np = np.array(neighbor_params[N_NEIGHBOR:])
    hist_regret.append(regret_val)
    hist_grad_norm.append(grad_norm)
    hist_nb.append((nb_x_np.copy(), nb_y_np.copy()))

    # Layout was captured inside compute_regret via jax.debug.callback
    opt_x_np = _captured_layout['x']
    opt_y_np = _captured_layout['y']
    hist_target.append((opt_x_np.copy(), opt_y_np.copy()))

    warm_x_batch = warm_x_batch.at[0].set(jnp.array(opt_x_np))
    warm_y_batch = warm_y_batch.at[0].set(jnp.array(opt_y_np))

    dt = time.time() - t_step
    print(f"{i:4d}  {regret_val:10.4f}  {best_regret:10.4f}  {grad_norm:10.4e}  {dt:5.1f}s")

    # ADAM update (gradient ascent)
    t = i + 1
    m_adam = beta1 * m_adam + (1 - beta1) * grad
    v_adam = beta2 * v_adam + (1 - beta2) * grad**2
    m_hat = m_adam / (1 - beta1**t)
    v_hat = v_adam / (1 - beta2**t)
    neighbor_params = neighbor_params + LR_OUTER * m_hat / (jnp.sqrt(v_hat) + adam_eps)

    # Clip to outer search boundary
    nb_x_new = jnp.clip(neighbor_params[:N_NEIGHBOR], nb_x_lo, nb_x_hi)
    nb_y_new = jnp.clip(neighbor_params[N_NEIGHBOR:], nb_y_lo, nb_y_hi)

    # Push neighbours outside polygon + BUFFER
    nb_x_new, nb_y_new = exclude_from_polygon(nb_x_new, nb_y_new)
    neighbor_params = jnp.concatenate([nb_x_new, nb_y_new])

    # Free stale XLA buffers to prevent memory growth
    del regret, grad
    gc.collect()

elapsed = time.time() - t0
n_frames = len(hist_regret)
print(f"\nOptimisation done: {n_frames} iterations in {elapsed:.1f}s")
if n_frames:
    print(f"  Initial regret: {hist_regret[0]:.4f} GWh")
    print(f"  Final regret:   {hist_regret[-1]:.4f} GWh")
    print(f"  Best regret:    {best_regret:.4f} GWh  (iter {best_iter})")

print(f"\nNeighbour displacement (at best iter {best_iter}):")
for k in range(N_NEIGHBOR):
    x0, y0 = float(init_nb_x[k]), float(init_nb_y[k])
    x1, y1 = float(hist_nb[best_iter][0][k]), float(hist_nb[best_iter][1][k])
    d = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
    print(f"  nb[{k}]: ({x0/1000:7.2f}, {y0/1000:7.2f}) km -> "
          f"({x1/1000:7.2f}, {y1/1000:7.2f}) km  d={d:.0f}m = {d/D:.1f}D")


# =============================================================================
# Phase 3 — weighted flow maps
# =============================================================================
print(f"\nPhase 3: computing {n_frames} weighted flow maps ...")
t0 = time.time()
hist_deficit = []

DEFICIT_MAX = 0.12

for i in range(n_frames):
    nb_x, nb_y = hist_nb[i]
    tgt_x, tgt_y = hist_target[i]
    x_all = jnp.concatenate([jnp.array(tgt_x), jnp.array(nb_x)])
    y_all = jnp.concatenate([jnp.array(tgt_y), jnp.array(nb_y)])
    flow_ws_all, _ = sim.flow_map(
        x_all, y_all, fm_x=fm_x_flat, fm_y=fm_y_flat,
        ws=ws_bins, wd=wd_bins,
    )
    deficits = 1.0 - flow_ws_all / ws_bins[:, None]
    weighted_deficit = jnp.sum(deficits * weights[:, None], axis=0)
    hist_deficit.append(np.array(weighted_deficit).reshape(fm_ny, fm_nx))

print(f"  Flow maps done in {time.time()-t0:.1f}s")


# =============================================================================
# Phase 4 — animation rendering
# =============================================================================
print(f"\nPhase 4: rendering {OUTPUT_MP4} ...")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Display coordinates in km
SCALE = 1000.0  # m -> km

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

wake_cmap = LinearSegmentedColormap.from_list("wake", [
    (0.00, "#0d1b2e"),
    (0.05, "#0f2847"),
    (0.15, "#1b4965"),
    (0.30, "#c06030"),
    (0.50, "#e8a030"),
    (1.00, "#ffe066"),
])

# Generate N_NEIGHBOR distinct colors from HSV colormap
_nb_cmap = plt.cm.hsv(np.linspace(0, 0.9, N_NEIGHBOR))
NB_COLORS = [f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
             for r, g, b, _ in _nb_cmap]

lib_x_np = np.array(liberal_x)
lib_y_np = np.array(liberal_y)
bnd_np = np.array(boundary)
hist_best_regret = np.maximum.accumulate(hist_regret)
weights_np = np.array(weights)
wd_bins_np = np.array(wd_bins)

# Scaled coordinates for display
fm_xx_km = fm_xx / SCALE
fm_yy_km = fm_yy / SCALE
bnd_km = bnd_np / SCALE

# Wind rose
rose_width = 2 * np.pi / len(wd_bins_np)
rose_theta = np.deg2rad(wd_bins_np)

# Figure layout
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

reg_lo = min(hist_regret) * 0.9 if hist_regret else 0
reg_hi = max(hist_regret) * 1.15 if hist_regret else 1
gn_lo = min(hist_grad_norm) * 0.3 if hist_grad_norm else 1e-6
gn_hi = max(hist_grad_norm) * 3.0 if hist_grad_norm else 1


def draw_frame(idx):
    for ax in (ax_wake, ax_reg, ax_aep, ax_grad):
        ax.clear()
    ax_rose.clear()

    nb_x, nb_y = hist_nb[idx]
    tgt_x, tgt_y = hist_target[idx]
    deficit = hist_deficit[idx]

    # Wake field (in km)
    ax_wake.pcolormesh(
        fm_xx_km, fm_yy_km, deficit,
        cmap=wake_cmap, vmin=0, vmax=DEFICIT_MAX,
        shading="auto", rasterized=True,
    )

    # Polygon boundary
    bnd_patch = MplPolygon(
        bnd_km, closed=True, fill=False,
        edgecolor="#ffffff", linewidth=1.5, alpha=0.6, linestyle="-",
    )
    ax_wake.add_patch(bnd_patch)

    # Liberal turbines
    ax_wake.scatter(
        lib_x_np / SCALE, lib_y_np / SCALE, s=40, facecolors="none",
        edgecolors="#66ccff", linewidths=1.2, zorder=5, label="Liberal (naive)",
    )

    # Conservative turbines
    ax_wake.scatter(
        tgt_x / SCALE, tgt_y / SCALE, s=40, c="#44dd88",
        edgecolors="#225533", linewidths=0.6, zorder=6, label="Conservative",
    )

    # Displacement arrows
    for k in range(N_TARGET):
        ddx = tgt_x[k] - lib_x_np[k]
        ddy = tgt_y[k] - lib_y_np[k]
        if np.sqrt(ddx**2 + ddy**2) > 10.0:
            ax_wake.annotate(
                "", xy=(tgt_x[k] / SCALE, tgt_y[k] / SCALE),
                xytext=(lib_x_np[k] / SCALE, lib_y_np[k] / SCALE),
                arrowprops=dict(arrowstyle="-|>", color="#44dd88", lw=0.8, alpha=0.5),
                zorder=4,
            )

    # Neighbour trails
    if idx > 0:
        for k in range(N_NEIGHBOR):
            trail_x = [hist_nb[j][0][k] / SCALE for j in range(idx + 1)]
            trail_y = [hist_nb[j][1][k] / SCALE for j in range(idx + 1)]
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

    # Current neighbour positions
    for k in range(N_NEIGHBOR):
        col = NB_COLORS[k % len(NB_COLORS)]
        ax_wake.scatter(
            [nb_x[k] / SCALE], [nb_y[k] / SCALE], s=130, c=col, marker="D",
            edgecolors="#ffffff", linewidths=1.2, zorder=8,
        )
    ax_wake.scatter([], [], s=80, c="#ff4444", marker="D",
                    edgecolors="#ffffff", linewidths=1.0,
                    label="Neighbours (swarm)")

    # Gradient arrows
    if idx + 1 < n_frames:
        next_x, next_y = hist_nb[idx + 1]
        for k in range(N_NEIGHBOR):
            ddx = next_x[k] - nb_x[k]
            ddy = next_y[k] - nb_y[k]
            norm = np.sqrt(ddx**2 + ddy**2)
            if norm > 10.0:
                arrow_len = min(norm, 3 * D)
                ax_wake.annotate(
                    "",
                    xy=((nb_x[k] + ddx / norm * arrow_len) / SCALE,
                        (nb_y[k] + ddy / norm * arrow_len) / SCALE),
                    xytext=(nb_x[k] / SCALE, nb_y[k] / SCALE),
                    arrowprops=dict(arrowstyle="-|>", color="#ffcc00", lw=1.8, alpha=0.7),
                    zorder=9,
                )

    ax_wake.set_xlim(FM_X_LO / SCALE, FM_X_HI / SCALE)
    ax_wake.set_ylim(FM_Y_LO / SCALE, FM_Y_HI / SCALE)
    ax_wake.set_aspect("equal")
    ax_wake.set_xlabel("Easting offset (km)")
    ax_wake.set_ylabel("Northing offset (km)")
    ax_wake.legend(
        loc="lower left", fontsize=7.5, framealpha=0.6,
        facecolor="#0d1b2e", edgecolor="#334466",
    )

    # Regret convergence
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

    # AEP bars
    nb_xj, nb_yj = jnp.array(nb_x), jnp.array(nb_y)
    x_lib_all = jnp.concatenate([liberal_x, nb_xj])
    y_lib_all = jnp.concatenate([liberal_y, nb_yj])
    r_lib = sim(x_lib_all, y_lib_all, ws_amb=ws_bins, wd_amb=wd_bins)
    p_lib = r_lib.power()[:, :N_TARGET]
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
                    f"{v:.1f}", ha="center", fontsize=7.5, color="#cccccc")

    # Wind rose
    ax_rose.set_theta_zero_location("N")
    ax_rose.set_theta_direction(-1)
    rose_colors = ["#ffcc00" if j == dominant_idx else "#4488cc"
                   for j in range(len(wd_bins_np))]
    ax_rose.bar(rose_theta, weights_np, width=rose_width,
                color=rose_colors, edgecolor="#334466", linewidth=0.5, alpha=0.85)
    ax_rose.set_title("Wind Rose (frequency)", fontsize=9, pad=12)
    ax_rose.tick_params(labelsize=6)
    ax_rose.set_facecolor("#0d1b2e")

    # Gradient norm
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
        f"DEI Bilevel Adversarial Search (15 MW, {N_TARGET} targets)  --  Iteration {idx}/{n_frames - 1}",
        fontsize=14, fontweight="bold", color="#ffffff",
    )


# Animate
hold_frames = 8
total_frames = n_frames + hold_frames

def draw_frame_with_hold(idx):
    draw_frame(min(idx, n_frames - 1))

anim = FuncAnimation(fig, draw_frame_with_hold, frames=total_frames,
                     interval=250, repeat=True)
anim.save(str(OUTPUT_MP4), writer="ffmpeg", fps=5, dpi=150)
plt.close(fig)
print(f"  Saved -> {OUTPUT_MP4}")

# Best frame (high-res) — render the iteration with highest regret
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
draw_frame(best_iter)
fig2.savefig(str(OUTPUT_PNG), dpi=200, facecolor=fig2.get_facecolor())
plt.close(fig2)
print(f"  Saved -> {OUTPUT_PNG}  (best iter {best_iter})")

print("\nDone.")

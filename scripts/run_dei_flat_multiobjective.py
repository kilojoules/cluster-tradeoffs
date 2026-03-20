"""Flat multi-objective search for design regret on the DEI case.

Instead of the bi-level greedy approach (outer: place neighbors one at a time,
inner: re-optimize target layout), this script simultaneously optimizes TWO
target layouts and the neighbor positions in a single flat formulation:

    Decision variables:
        x_c, y_c  — conservative layout (n_target turbine positions)
        x_l, y_l  — liberal layout (n_target turbine positions)
        x_nb, y_nb — neighbor positions (n_neighbors positions)

    Objectives (maximize all three):
        AEP_liberal(x_l)              — liberal layout quality (no neighbors)
        AEP_conservative(x_c, nb)     — conservative layout quality (with neighbors)
        regret(x_c, x_l, nb) = AEP_conservative(x_c, nb) - AEP_conservative(x_l, nb)

The regret objective creates tension on x_l: the liberal objective wants x_l
to be good *without* neighbors, while regret wants x_l to be bad *with*
neighbors. If no layout can be simultaneously good without neighbors and bad
with neighbors, regret = 0 and there are no design tradeoffs — verified by
gradient, not just by comparing two independent optima.

We scalarize via weighted sum:
    L = w_lib * AEP_liberal(x_l)
      + w_con * AEP_conservative(x_c, nb)
      + w_reg * regret(x_c, x_l, nb)

and sweep the weights to trace out the Pareto front.

Optionally initializes from a greedy grid search result (warm start).

Usage:
    # Default weights (equal)
    pixi run python scripts/run_dei_flat_multiobjective.py

    # Sweep weights
    pixi run python scripts/run_dei_flat_multiobjective.py --sweep

    # Warm start from greedy result
    pixi run python scripts/run_dei_flat_multiobjective.py --warm-start analysis/dei_greedy_grid/results.json

    # Pure regret maximization (fix x_l to liberal optimum, only optimize x_c and neighbors)
    pixi run python scripts/run_dei_flat_multiobjective.py --regret-only
"""

import jax
jax.config.update("jax_enable_x64", True)

import argparse
import json
import time
from functools import partial
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd

from pixwake import Curve, Turbine, WakeSimulation
from pixwake.deficit import BastankhahGaussianDeficit
from pixwake.deficit.gaussian import TurboGaussianDeficit
from pixwake.optim.sgd import SGDSettings, topfarm_sgd_solve

# Unbuffered print
print = partial(print, flush=True)

# =============================================================================
# Configuration (matches run_dei_greedy_grid.py)
# =============================================================================
D = 240.0  # rotor diameter (m) — IEA 15 MW class
N_TARGET = 50
N_NEIGHBORS = 30  # number of external turbines
MIN_SPACING_D = 4.0
INNER_MAX_ITER = 500
INNER_LR = 50.0
GRID_PAD_D = 12.0

FLAT_MAX_ITER = 2000  # outer ADAM iterations for flat problem
FLAT_LR = 20.0        # ADAM learning rate (meters)

OUTPUT_DIR = Path("analysis/dei_flat_multiobjective")


# =============================================================================
# DEI turbine (15 MW, D=240m, hub=150m) — identical to greedy script
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
        rotor_diameter=240.0, hub_height=150.0,
        power_curve=Curve(ws=ws, values=power),
        ct_curve=Curve(ws=ws, values=ct),
    )


# =============================================================================
# DEI polygon boundary — identical to greedy script
# =============================================================================
_dk0w_raw = np.array([
    706694.3923283464, 6224158.532895836,
    703972.0844905999, 6226906.597455995,
    702624.6334635273, 6253853.5386425415,
    712771.6248419734, 6257704.934445341,
    715639.3355871611, 6260664.6846508905,
    721593.2420745814, 6257906.998015941,
]).reshape((-1, 2))

CENTROID_X = _dk0w_raw[:, 0].mean()
CENTROID_Y = _dk0w_raw[:, 1].mean()

from scipy.spatial import ConvexHull
_hull = ConvexHull(_dk0w_raw - np.array([CENTROID_X, CENTROID_Y]))
boundary_np = (_dk0w_raw - np.array([CENTROID_X, CENTROID_Y]))[_hull.vertices]
boundary = jnp.array(boundary_np)

from matplotlib.path import Path as MplPath
_polygon_path = MplPath(boundary_np)


# =============================================================================
# Wind data — identical to greedy script
# =============================================================================
def load_wind_data():
    csv_path = Path(__file__).parent.parent / "energy_island_10y_daily_av_wind.csv"
    df = pd.read_csv(csv_path, sep=";")
    wd_ts, ws_ts = df["WD_150"].values, df["WS_150"].values
    n_bins = 24
    bin_edges = np.linspace(0, 360, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    w = np.zeros(n_bins)
    mean_speeds = np.zeros(n_bins)
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (wd_ts >= bin_edges[i]) | (wd_ts < bin_edges[0])
        else:
            mask = (wd_ts >= bin_edges[i]) & (wd_ts < bin_edges[i + 1])
        w[i] = mask.sum()
        mean_speeds[i] = ws_ts[mask].mean() if mask.sum() > 0 else ws_ts.mean()
    w /= w.sum()
    return jnp.array(bin_centers), jnp.array(mean_speeds), jnp.array(w)


def generate_target_grid(boundary_np, n_target, spacing):
    """Create a grid of points inside the polygon, subsample to n_target."""
    x_lo, x_hi = boundary_np[:, 0].min(), boundary_np[:, 0].max()
    y_lo, y_hi = boundary_np[:, 1].min(), boundary_np[:, 1].max()
    gx = np.arange(x_lo + 2 * D, x_hi - 2 * D, spacing)
    gy = np.arange(y_lo + 2 * D, y_hi - 2 * D, spacing)
    gx_2d, gy_2d = np.meshgrid(gx, gy)
    candidates = np.column_stack([gx_2d.ravel(), gy_2d.ravel()])
    inside = _polygon_path.contains_points(candidates)
    pts = candidates[inside]

    if len(pts) < n_target:
        gx = np.arange(x_lo + D, x_hi - D, spacing * 0.7)
        gy = np.arange(y_lo + D, y_hi - D, spacing * 0.7)
        gx_2d, gy_2d = np.meshgrid(gx, gy)
        candidates = np.column_stack([gx_2d.ravel(), gy_2d.ravel()])
        inside = _polygon_path.contains_points(candidates)
        pts = candidates[inside]

    indices = np.round(np.linspace(0, len(pts) - 1, n_target)).astype(int)
    selected = pts[indices]
    return jnp.array(selected[:, 0]), jnp.array(selected[:, 1])


def generate_neighbor_ring(boundary_np, n_neighbors, pad_D):
    """Place initial neighbors in a ring outside the boundary.

    Uses the dominant wind direction (westerly) to bias placement upstream.
    """
    hull = ConvexHull(boundary_np)
    hull_pts = boundary_np[hull.vertices]
    centroid = hull_pts.mean(axis=0)

    # Ring at pad_D outside the boundary centroid
    radius = np.max(np.linalg.norm(hull_pts - centroid, axis=1)) + pad_D * D

    # Bias angles toward upstream (west = 270 deg = pi in math coords)
    # Mix: 60% upstream semicircle, 40% full circle
    n_upstream = int(n_neighbors * 0.6)
    n_rest = n_neighbors - n_upstream

    # Upstream: angles from 120 to 240 deg (math convention, = W/SW/NW)
    upstream_angles = np.linspace(2 * np.pi / 3, 4 * np.pi / 3, n_upstream, endpoint=False)
    rest_angles = np.linspace(0, 2 * np.pi, n_rest, endpoint=False)

    angles = np.concatenate([upstream_angles, rest_angles])
    nb_x = centroid[0] + radius * np.cos(angles)
    nb_y = centroid[1] + radius * np.sin(angles)

    return jnp.array(nb_x), jnp.array(nb_y)


# =============================================================================
# Constraint penalties (differentiable, for use in flat objective)
# =============================================================================
def spacing_penalty(x, y, min_spacing, sharpness=1.0):
    """Soft penalty for violating minimum inter-turbine spacing."""
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist_sq = dx**2 + dy**2 + jnp.eye(len(x)) * 1e10  # avoid self-pairs
    violations = jnp.maximum(0.0, min_spacing**2 - dist_sq)
    return sharpness * jnp.sum(violations) / (min_spacing**2)


def containment_penalty(x, y, boundary, sharpness=1.0):
    """Soft penalty for turbines outside the boundary polygon.

    Uses signed distance from each edge (positive = inside for CCW boundary).
    """
    n_verts = boundary.shape[0]
    penalty = jnp.zeros(())
    for i in range(n_verts):
        j = (i + 1) % n_verts
        edge_x = boundary[j, 0] - boundary[i, 0]
        edge_y = boundary[j, 1] - boundary[i, 1]
        # Signed distance (positive = left of edge = inside for CCW)
        signed_dist = edge_x * (y - boundary[i, 1]) - edge_y * (x - boundary[i, 0])
        penalty = penalty + jnp.sum(jnp.maximum(0.0, -signed_dist))
    return sharpness * penalty / (D * n_verts)


def neighbor_exclusion_penalty(nb_x, nb_y, boundary, buffer, sharpness=1.0):
    """Soft penalty pushing neighbors OUTSIDE the target boundary + buffer."""
    centroid = jnp.mean(boundary, axis=0)
    # For each neighbor, compute distance to centroid vs boundary radius
    # Simple version: penalize if inside expanded boundary
    n_verts = boundary.shape[0]
    penalty = jnp.zeros(())
    # Expand boundary outward by buffer
    dirs = boundary - centroid
    norms = jnp.sqrt(jnp.sum(dirs**2, axis=1, keepdims=True))
    expanded = boundary + dirs / norms * buffer
    for i in range(n_verts):
        j = (i + 1) % n_verts
        edge_x = expanded[j, 0] - expanded[i, 0]
        edge_y = expanded[j, 1] - expanded[i, 1]
        signed_dist = edge_x * (nb_y - expanded[i, 1]) - edge_y * (nb_x - expanded[i, 0])
        # Penalize if INSIDE expanded boundary (signed_dist > 0 for CCW)
        penalty = penalty + jnp.sum(jnp.maximum(0.0, signed_dist))
    return sharpness * penalty / (D * n_verts)


# =============================================================================
# Core: flat multi-objective function
# =============================================================================
def build_flat_objective(sim, boundary, min_spacing, ws_amb, wd_amb,
                         ti_amb, weights, n_target, n_neighbors,
                         w_lib, w_con, w_reg, penalty_weight,
                         regret_only=False):
    """Build the scalarized multi-objective function.

    Args:
        regret_only: If True, x_l is NOT a decision variable — it's the
            pre-computed liberal optimum. Only x_c and neighbors are optimized.

    Returns:
        objective_fn(params) -> scalar (to MAXIMIZE)
        unpack_fn(params) -> dict of named arrays
    """
    buffer = 2 * D  # min distance of neighbors from target boundary

    def compute_aep(target_x, target_y, nb_x=None, nb_y=None):
        """AEP of target turbines, optionally with neighbor interference."""
        n_t = target_x.shape[0]
        if nb_x is not None and nb_y is not None:
            x_all = jnp.concatenate([target_x, nb_x])
            y_all = jnp.concatenate([target_y, nb_y])
        else:
            x_all = target_x
            y_all = target_y

        result = sim(x_all, y_all, ws_amb=ws_amb, wd_amb=wd_amb, ti_amb=ti_amb)
        power = result.power()[:, :n_t]

        if weights is not None:
            weighted_power = jnp.sum(power * weights[:, None])
            return weighted_power * 8760 / 1e6  # GWh
        return jnp.sum(power) * 8760 / 1e6 / power.shape[0]

    def unpack(params):
        """Unpack flat parameter vector into named components."""
        idx = 0
        x_c = params[idx:idx + n_target]; idx += n_target
        y_c = params[idx:idx + n_target]; idx += n_target
        if not regret_only:
            x_l = params[idx:idx + n_target]; idx += n_target
            y_l = params[idx:idx + n_target]; idx += n_target
        else:
            x_l = None
            y_l = None
        nb_x = params[idx:idx + n_neighbors]; idx += n_neighbors
        nb_y = params[idx:idx + n_neighbors]; idx += n_neighbors
        return {"x_c": x_c, "y_c": y_c, "x_l": x_l, "y_l": y_l,
                "nb_x": nb_x, "nb_y": nb_y}

    def objective(params, liberal_x_fixed=None, liberal_y_fixed=None):
        d = unpack(params)
        x_c, y_c = d["x_c"], d["y_c"]
        nb_x, nb_y = d["nb_x"], d["nb_y"]

        if regret_only:
            x_l, y_l = liberal_x_fixed, liberal_y_fixed
        else:
            x_l, y_l = d["x_l"], d["y_l"]

        # --- Objectives (all to maximize) ---
        aep_conservative = compute_aep(x_c, y_c, nb_x, nb_y)
        aep_liberal = compute_aep(x_l, y_l)  # no neighbors
        aep_liberal_present = compute_aep(x_l, y_l, nb_x, nb_y)
        regret = aep_conservative - aep_liberal_present

        obj = w_lib * aep_liberal + w_con * aep_conservative + w_reg * regret

        # --- Constraint penalties (subtract from objective) ---
        pen = jnp.zeros(())

        # Conservative layout: inside boundary + min spacing
        pen = pen + containment_penalty(x_c, y_c, boundary)
        pen = pen + spacing_penalty(x_c, y_c, min_spacing)

        # Liberal layout: inside boundary + min spacing
        if not regret_only:
            pen = pen + containment_penalty(x_l, y_l, boundary)
            pen = pen + spacing_penalty(x_l, y_l, min_spacing)

        # Neighbors: outside boundary + buffer, min spacing among themselves
        pen = pen + neighbor_exclusion_penalty(nb_x, nb_y, boundary, buffer)
        pen = pen + spacing_penalty(nb_x, nb_y, min_spacing)

        return obj - penalty_weight * pen

    return objective, unpack


# =============================================================================
# ADAM optimizer
# =============================================================================
def adam_maximize(objective_fn, init_params, max_iter, lr,
                  verbose=True, log_every=50, snapshot_every=10,
                  **objective_kwargs):
    """ADAM gradient ascent with logging and parameter snapshots.

    Returns:
        best_params, best_val, history (list of obj values),
        snapshots (list of param arrays sampled every snapshot_every iters)
    """
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    params = init_params
    m = jnp.zeros_like(params)
    v = jnp.zeros_like(params)

    val_and_grad = jax.value_and_grad(lambda p: objective_fn(p, **objective_kwargs))

    best_val = -jnp.inf
    best_params = params
    history = []
    snapshots = [np.array(params)]  # always include initial state

    for i in range(max_iter):
        val, grad = val_and_grad(params)

        if not jnp.all(jnp.isfinite(grad)):
            if verbose:
                print(f"  Iter {i}: NaN gradient, stopping early")
            break

        history.append(float(val))

        if float(val) > float(best_val):
            best_val = val
            best_params = params

        if verbose and i % log_every == 0:
            grad_norm = float(jnp.linalg.norm(grad))
            print(f"  Iter {i:5d}: obj = {val:.4f}, |grad| = {grad_norm:.4f}")

        if i > 0 and i % snapshot_every == 0:
            snapshots.append(np.array(params))

        t = i + 1
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        params = params + lr * m_hat / (jnp.sqrt(v_hat) + eps)

    snapshots.append(np.array(best_params))  # always include final
    return best_params, float(best_val), history, snapshots


# =============================================================================
# Animation
# =============================================================================
def render_flat_animation(snapshots, history, unpack_fn, boundary_np,
                          wd_bins, ws_bins, wind_weights, output_path,
                          regret_only=False, liberal_x_fixed=None,
                          liberal_y_fixed=None, eval_fn=None):
    """Render MP4 showing all turbines moving over ADAM iterations.

    Three groups: conservative (green), liberal (blue), neighbors (red).
    Right panels: objective history and regret history.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Polygon as MplPolygon

    n_frames = len(snapshots)

    def to_km(x, y):
        return np.asarray(x) / 1000.0, np.asarray(y) / 1000.0

    bnd_km = np.column_stack(to_km(boundary_np[:, 0], boundary_np[:, 1]))

    # Unpack all frames
    frames_data = []
    for snap in snapshots:
        d = unpack_fn(jnp.array(snap))
        x_c, y_c = np.array(d["x_c"]), np.array(d["y_c"])
        nb_x, nb_y = np.array(d["nb_x"]), np.array(d["nb_y"])
        if regret_only:
            x_l = np.array(liberal_x_fixed)
            y_l = np.array(liberal_y_fixed)
        else:
            x_l, y_l = np.array(d["x_l"]), np.array(d["y_l"])
        frames_data.append((x_c, y_c, x_l, y_l, nb_x, nb_y))

    # Evaluate objectives at each snapshot if eval_fn provided
    obj_history = []
    regret_history = []
    aep_lib_history = []
    aep_con_history = []
    if eval_fn is not None:
        for x_c, y_c, x_l, y_l, nb_x, nb_y in frames_data:
            metrics = eval_fn(x_c, y_c, x_l, y_l, nb_x, nb_y)
            obj_history.append(metrics["obj"])
            regret_history.append(metrics["regret"])
            aep_lib_history.append(metrics["aep_lib"])
            aep_con_history.append(metrics["aep_con"])

    # Compute plot bounds from all frames
    all_x = np.concatenate([bnd_km[:, 0]] +
        [np.concatenate([to_km(fd[0], fd[1])[0], to_km(fd[2], fd[3])[0],
                         to_km(fd[4], fd[5])[0]]) for fd in frames_data])
    all_y = np.concatenate([bnd_km[:, 1]] +
        [np.concatenate([to_km(fd[0], fd[1])[1], to_km(fd[2], fd[3])[1],
                         to_km(fd[4], fd[5])[1]]) for fd in frames_data])
    pad_km = 2.0
    x_lo, x_hi = all_x.min() - pad_km, all_x.max() + pad_km
    y_lo, y_hi = all_y.min() - pad_km, all_y.max() + pad_km

    # Wind rose helper
    def draw_wind_rose(ax):
        n_bins = len(wd_bins)
        theta = np.deg2rad(np.array(wd_bins))
        bin_width = min(360 / n_bins, 30.0)
        width = np.deg2rad(bin_width) * 0.9
        ws_np = np.array(ws_bins)
        norm = plt.Normalize(vmin=max(ws_np.min() - 1, 0), vmax=ws_np.max() + 1)
        colors = plt.cm.YlOrRd(norm(ws_np))
        ax.bar(theta, np.array(wind_weights), width=width, bottom=0.0,
               color=colors, edgecolor="gray", linewidth=0.4, alpha=0.85)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_thetagrids([0, 90, 180, 270], ["N", "E", "S", "W"], fontsize=8)
        ax.set_rticks([])
        ax.set_title("Wind Rose", fontsize=10, pad=10)

    fig = plt.figure(figsize=(22, 10))
    gs = fig.add_gridspec(3, 2, width_ratios=[2.5, 1], height_ratios=[1, 1, 1],
                          hspace=0.4, wspace=0.25)
    ax_map = fig.add_subplot(gs[:, 0])
    ax_rose = fig.add_subplot(gs[0, 1], projection="polar")
    ax_obj = fig.add_subplot(gs[1, 1])
    ax_reg = fig.add_subplot(gs[2, 1])

    def draw(frame_idx):
        ax_map.clear()
        ax_rose.clear()
        ax_obj.clear()
        ax_reg.clear()

        x_c, y_c, x_l, y_l, nb_x, nb_y = frames_data[frame_idx]

        # --- Map panel ---
        poly = MplPolygon(bnd_km, closed=True, fill=True,
                          facecolor="#f5f5f0", edgecolor="black", lw=2.5,
                          alpha=0.9, zorder=2)
        ax_map.add_patch(poly)

        # Conservative layout (green)
        cx_km, cy_km = to_km(x_c, y_c)
        ax_map.scatter(cx_km, cy_km, c="forestgreen", marker="^", s=50,
                       edgecolors="darkgreen", linewidths=0.6,
                       label="Conservative", zorder=5)

        # Liberal layout (blue) — hollow if regret_only (fixed)
        lx_km, ly_km = to_km(x_l, y_l)
        if regret_only:
            ax_map.scatter(lx_km, ly_km, facecolors="none", edgecolors="royalblue",
                           marker="^", s=40, linewidths=0.8,
                           label="Liberal (fixed)", zorder=4, linestyle="--")
        else:
            ax_map.scatter(lx_km, ly_km, c="royalblue", marker="v", s=50,
                           edgecolors="darkblue", linewidths=0.6,
                           label="Liberal", zorder=4)

        # Neighbors (red diamonds)
        nx_km, ny_km = to_km(nb_x, nb_y)
        ax_map.scatter(nx_km, ny_km, c="red", marker="D", s=80,
                       edgecolors="darkred", linewidths=0.8,
                       label=f"Neighbors ({len(nb_x)})", zorder=6)

        # Trails: show previous frame positions as faded dots
        if frame_idx > 0:
            prev = frames_data[frame_idx - 1]
            pcx, pcy = to_km(prev[0], prev[1])
            ax_map.scatter(pcx, pcy, c="forestgreen", marker="^", s=15,
                           alpha=0.2, zorder=3)
            if not regret_only:
                plx, ply = to_km(prev[2], prev[3])
                ax_map.scatter(plx, ply, c="royalblue", marker="v", s=15,
                               alpha=0.2, zorder=3)
            pnx, pny = to_km(prev[4], prev[5])
            ax_map.scatter(pnx, pny, c="red", marker="D", s=20,
                           alpha=0.2, zorder=3)

        ax_map.set_xlim(x_lo, x_hi)
        ax_map.set_ylim(y_lo, y_hi)
        ax_map.set_aspect("equal")
        ax_map.set_xlabel("x (km)", fontsize=12)
        ax_map.set_ylabel("y (km)", fontsize=12)
        ax_map.legend(loc="lower left", fontsize=9, framealpha=0.9)

        # Title with metrics
        if regret_history:
            reg_val = regret_history[frame_idx]
            ax_map.set_title(
                f"Iteration {frame_idx * max(1, len(history) // n_frames)}"
                f" — Regret = {reg_val:.3f} GWh",
                fontsize=14, fontweight="bold")
        else:
            ax_map.set_title(f"Frame {frame_idx + 1}/{n_frames}",
                             fontsize=14, fontweight="bold")

        # --- Wind rose ---
        draw_wind_rose(ax_rose)

        # --- Objective history ---
        if obj_history:
            steps = np.arange(len(obj_history[:frame_idx + 1]))
            ax_obj.plot(steps, obj_history[:frame_idx + 1],
                        "o-", color="black", ms=3, lw=1.5)
            ax_obj.set_ylabel("Scalarized Obj", fontsize=10)
            ax_obj.set_xlabel("Snapshot", fontsize=10)
            ax_obj.set_title("Objective", fontsize=11)
            ax_obj.grid(True, alpha=0.3)

        # --- Regret history ---
        if regret_history:
            steps = np.arange(len(regret_history[:frame_idx + 1]))
            ax_reg.plot(steps, regret_history[:frame_idx + 1],
                        "o-", color="firebrick", ms=3, lw=1.5)
            ax_reg.fill_between(steps, 0, regret_history[:frame_idx + 1],
                                color="firebrick", alpha=0.1)
            ax_reg.set_ylabel("Regret (GWh)", fontsize=10)
            ax_reg.set_xlabel("Snapshot", fontsize=10)
            ax_reg.set_title("Design Regret", fontsize=11)
            ax_reg.grid(True, alpha=0.3)

            if frame_idx > 0:
                ax_reg.text(steps[-1], regret_history[frame_idx] * 1.02,
                            f"{regret_history[frame_idx]:.2f}",
                            ha="center", fontsize=9, fontweight="bold",
                            color="firebrick")

        fig.suptitle("Flat Multi-Objective Optimization — Turbine Positions",
                     fontsize=15, fontweight="bold", y=0.98)

    fps = max(1, min(10, n_frames // 5))
    anim = FuncAnimation(fig, draw, frames=n_frames,
                         interval=1000 // max(fps, 1), repeat=True)
    anim.save(str(output_path), writer="ffmpeg", fps=fps, dpi=150)
    plt.close(fig)
    print(f"Animation saved: {output_path}")

    # Save final frame as PNG
    png_path = output_path.with_suffix(".png")
    fig2, ax2 = plt.subplots(1, 1, figsize=(14, 10))
    x_c, y_c, x_l, y_l, nb_x, nb_y = frames_data[-1]
    poly = MplPolygon(bnd_km, closed=True, fill=True,
                      facecolor="#f5f5f0", edgecolor="black", lw=2.5,
                      alpha=0.9, zorder=2)
    ax2.add_patch(poly)
    cx_km, cy_km = to_km(x_c, y_c)
    ax2.scatter(cx_km, cy_km, c="forestgreen", marker="^", s=60,
                edgecolors="darkgreen", linewidths=0.6,
                label="Conservative", zorder=5)
    lx_km, ly_km = to_km(x_l, y_l)
    ax2.scatter(lx_km, ly_km, facecolors="none" if regret_only else "royalblue",
                edgecolors="royalblue" if regret_only else "darkblue",
                marker="^" if regret_only else "v", s=50, linewidths=0.8,
                label="Liberal" + (" (fixed)" if regret_only else ""), zorder=4)
    nx_km, ny_km = to_km(nb_x, nb_y)
    ax2.scatter(nx_km, ny_km, c="red", marker="D", s=100,
                edgecolors="darkred", linewidths=0.8,
                label=f"Neighbors ({len(nb_x)})", zorder=6)
    ax2.set_xlim(x_lo, x_hi)
    ax2.set_ylim(y_lo, y_hi)
    ax2.set_aspect("equal")
    ax2.set_xlabel("x (km)", fontsize=12)
    ax2.set_ylabel("y (km)", fontsize=12)
    ax2.legend(loc="lower left", fontsize=10)
    title = "Final Layout"
    if regret_history:
        title += f" — Regret = {regret_history[-1]:.3f} GWh"
    ax2.set_title(title, fontsize=14, fontweight="bold")
    fig2.savefig(str(png_path), dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Final frame saved: {png_path}")


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Flat multi-objective search for design regret (DEI case)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-target", type=int, default=N_TARGET)
    parser.add_argument("--n-neighbors", type=int, default=N_NEIGHBORS)
    parser.add_argument("--min-spacing-D", type=float, default=MIN_SPACING_D)
    parser.add_argument("--flat-max-iter", type=int, default=FLAT_MAX_ITER)
    parser.add_argument("--flat-lr", type=float, default=FLAT_LR)
    parser.add_argument("--inner-lr", type=float, default=INNER_LR,
                        help="SGD LR for initial liberal layout solve")
    parser.add_argument("--inner-max-iter", type=int, default=INNER_MAX_ITER,
                        help="SGD iterations for initial liberal layout solve")
    parser.add_argument("--grid-pad-D", type=float, default=GRID_PAD_D)
    parser.add_argument("--penalty-weight", type=float, default=100.0,
                        help="Weight on constraint penalties")
    parser.add_argument("--deficit", type=str, default="bastankhah",
                        choices=["bastankhah", "turbopark"])
    parser.add_argument("--wind-rose", type=str, default="dei",
                        choices=["dei", "unidirectional", "uniform"])
    parser.add_argument("--wind-dir", type=float, default=270.0)
    parser.add_argument("--wind-speed", type=float, default=9.0)
    parser.add_argument("--ti", type=float, default=0.06)

    parser.add_argument("--w-lib", type=float, default=1.0,
                        help="Weight on liberal AEP objective")
    parser.add_argument("--w-con", type=float, default=1.0,
                        help="Weight on conservative AEP objective")
    parser.add_argument("--w-reg", type=float, default=1.0,
                        help="Weight on regret objective")

    parser.add_argument("--sweep", action="store_true",
                        help="Sweep weight combinations to trace Pareto front")
    parser.add_argument("--regret-only", action="store_true",
                        help="Fix liberal layout, only optimize conservative + neighbors")
    parser.add_argument("--warm-start", type=str, default=None,
                        help="Path to greedy results.json for warm-starting neighbors")

    parser.add_argument("--no-animate", action="store_true",
                        help="Skip animation rendering (saves time on HPC)")
    parser.add_argument("--snapshot-every", type=int, default=None,
                        help="Save param snapshot every N iters (default: auto ~200 frames)")

    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_target = args.n_target
    n_neighbors = args.n_neighbors
    min_spacing = args.min_spacing_D * D

    # --- Setup (same as greedy script) ---
    turbine = create_dei_turbine()
    if args.wind_rose == "dei":
        wd, ws, weights = load_wind_data()
    elif args.wind_rose == "unidirectional":
        wd = jnp.array([args.wind_dir])
        ws = jnp.array([args.wind_speed])
        weights = jnp.array([1.0])
    elif args.wind_rose == "uniform":
        n_bins = 24
        wd = jnp.linspace(0, 360 - 360 / n_bins, n_bins)
        ws = jnp.full(n_bins, args.wind_speed)
        weights = jnp.full(n_bins, 1.0 / n_bins)

    if args.deficit == "bastankhah":
        deficit = BastankhahGaussianDeficit(k=0.04)
    elif args.deficit == "turbopark":
        deficit = TurboGaussianDeficit(A=0.04)
    ti_amb = args.ti if args.deficit == "turbopark" else None

    sim = WakeSimulation(turbine, deficit)

    print("=" * 70)
    print("FLAT MULTI-OBJECTIVE SEARCH")
    print(f"  {n_target} target turbines, {n_neighbors} neighbor turbines")
    print(f"  Weights: lib={args.w_lib}, con={args.w_con}, reg={args.w_reg}")
    print(f"  Regret-only mode: {args.regret_only}")
    print(f"  Warm start: {args.warm_start}")
    print(f"  ADAM: lr={args.flat_lr}, max_iter={args.flat_max_iter}")
    print(f"  Penalty weight: {args.penalty_weight}")
    print("=" * 70)

    # --- Step 1: Compute liberal layout (SGD, no neighbors) ---
    print("\n--- Computing liberal layout (SGD, no neighbors) ---")
    init_x, init_y = generate_target_grid(boundary_np, n_target, spacing=4 * D)

    sgd_settings = SGDSettings(
        learning_rate=args.inner_lr,
        max_iter=args.inner_max_iter,
        additional_constant_lr_iterations=args.inner_max_iter,
        tol=1e-6,
    )

    def liberal_objective(x, y):
        n_t = x.shape[0]
        result = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=ti_amb)
        power = result.power()[:, :n_t]
        if weights is not None:
            return -jnp.sum(power * weights[:, None]) * 8760 / 1e6
        return -jnp.sum(power) * 8760 / 1e6 / power.shape[0]

    liberal_x, liberal_y = topfarm_sgd_solve(
        liberal_objective, init_x, init_y, boundary, min_spacing, sgd_settings,
    )

    # Evaluate liberal AEP
    lib_result = sim(liberal_x, liberal_y, ws_amb=ws, wd_amb=wd, ti_amb=ti_amb)
    lib_power = lib_result.power()[:, :n_target]
    liberal_aep = float(jnp.sum(lib_power * weights[:, None]) * 8760 / 1e6)
    print(f"  Liberal AEP: {liberal_aep:.2f} GWh")

    # --- Step 2: Initialize neighbor positions ---
    if args.warm_start:
        print(f"\n--- Warm-starting neighbors from {args.warm_start} ---")
        with open(args.warm_start) as f:
            greedy_result = json.load(f)
        nb_x_init = jnp.array(greedy_result["neighbor_x"][:n_neighbors])
        nb_y_init = jnp.array(greedy_result["neighbor_y"][:n_neighbors])
        # Pad if greedy placed fewer than n_neighbors
        if len(nb_x_init) < n_neighbors:
            extra_x, extra_y = generate_neighbor_ring(
                boundary_np, n_neighbors - len(nb_x_init), args.grid_pad_D)
            nb_x_init = jnp.concatenate([nb_x_init, extra_x])
            nb_y_init = jnp.concatenate([nb_y_init, extra_y])
        print(f"  Loaded {len(nb_x_init)} neighbor positions")
    else:
        print("\n--- Generating initial neighbor ring ---")
        nb_x_init, nb_y_init = generate_neighbor_ring(
            boundary_np, n_neighbors, args.grid_pad_D)
        print(f"  Placed {n_neighbors} neighbors in ring")

    # --- Step 3: Build and run flat optimization ---
    def run_flat(w_lib, w_con, w_reg, tag=""):
        """Run one flat optimization with given weights."""
        print(f"\n{'='*50}")
        print(f"Running flat optimization {tag}")
        print(f"  w_lib={w_lib:.2f}, w_con={w_con:.2f}, w_reg={w_reg:.2f}")
        print(f"{'='*50}")

        objective_fn, unpack_fn = build_flat_objective(
            sim, boundary, min_spacing, ws, wd, ti_amb, weights,
            n_target, n_neighbors,
            w_lib=w_lib, w_con=w_con, w_reg=w_reg,
            penalty_weight=args.penalty_weight,
            regret_only=args.regret_only,
        )

        # Pack initial parameters
        if args.regret_only:
            # Only x_c, y_c, nb_x, nb_y
            init_params = jnp.concatenate([
                liberal_x, liberal_y,  # x_c, y_c — start from liberal
                nb_x_init, nb_y_init,
            ])
        else:
            # x_c, y_c, x_l, y_l, nb_x, nb_y
            init_params = jnp.concatenate([
                liberal_x, liberal_y,  # x_c starts at liberal solution
                liberal_x, liberal_y,  # x_l starts at liberal solution
                nb_x_init, nb_y_init,
            ])

        n_params = len(init_params)
        print(f"  Decision variables: {n_params} "
              f"({2*n_target} conservative + "
              f"{'0' if args.regret_only else str(2*n_target)} liberal + "
              f"{2*n_neighbors} neighbors)")

        t0 = time.time()
        objective_kwargs = {}
        if args.regret_only:
            objective_kwargs = {
                "liberal_x_fixed": liberal_x,
                "liberal_y_fixed": liberal_y,
            }
        snapshot_every = args.snapshot_every or max(1, args.flat_max_iter // 200)
        best_params, best_val, history, snapshots = adam_maximize(
            objective_fn, init_params,
            max_iter=args.flat_max_iter, lr=args.flat_lr,
            snapshot_every=snapshot_every,
            **objective_kwargs,
        )
        elapsed = time.time() - t0

        # Unpack results
        d = unpack_fn(best_params)
        x_c, y_c = d["x_c"], d["y_c"]
        nb_x, nb_y = d["nb_x"], d["nb_y"]
        if args.regret_only:
            x_l, y_l = liberal_x, liberal_y
        else:
            x_l, y_l = d["x_l"], d["y_l"]

        # Evaluate final objectives
        def compute_aep(tx, ty, nbx=None, nby=None):
            n_t = tx.shape[0]
            if nbx is not None:
                xa = jnp.concatenate([tx, nbx])
                ya = jnp.concatenate([ty, nby])
            else:
                xa, ya = tx, ty
            r = sim(xa, ya, ws_amb=ws, wd_amb=wd, ti_amb=ti_amb)
            p = r.power()[:, :n_t]
            if weights is not None:
                return float(jnp.sum(p * weights[:, None]) * 8760 / 1e6)
            return float(jnp.sum(p) * 8760 / 1e6 / p.shape[0])

        aep_lib = compute_aep(x_l, y_l)
        aep_con = compute_aep(x_c, y_c, nb_x, nb_y)
        aep_lib_present = compute_aep(x_l, y_l, nb_x, nb_y)
        regret = aep_con - aep_lib_present

        print(f"\n  Results:")
        print(f"    Liberal AEP (no neighbors):   {aep_lib:.2f} GWh")
        print(f"    Liberal AEP (w/ neighbors):   {aep_lib_present:.2f} GWh")
        print(f"    Conservative AEP (w/ nb):     {aep_con:.2f} GWh")
        print(f"    Regret:                       {regret:.4f} GWh")
        print(f"    Time:                         {elapsed:.1f}s ({elapsed/60:.1f} min)")

        # --- Render animation ---
        def eval_snapshot(xc, yc, xl, yl, nbx, nby):
            """Evaluate all metrics for one snapshot (for animation panels)."""
            al = compute_aep(jnp.array(xl), jnp.array(yl))
            ac = compute_aep(jnp.array(xc), jnp.array(yc),
                             jnp.array(nbx), jnp.array(nby))
            alp = compute_aep(jnp.array(xl), jnp.array(yl),
                              jnp.array(nbx), jnp.array(nby))
            reg = ac - alp
            obj_val = w_lib * al + w_con * ac + w_reg * reg
            return {"obj": obj_val, "regret": reg,
                    "aep_lib": al, "aep_con": ac}

        if not args.no_animate:
            mp4_path = output_dir / f"flat_{tag.replace(' ', '_')}.mp4"
            print(f"\n  Rendering animation ({len(snapshots)} frames)...")
            render_flat_animation(
                snapshots, history, unpack_fn, boundary_np,
                wd, ws, weights, mp4_path,
                regret_only=args.regret_only,
                liberal_x_fixed=liberal_x if args.regret_only else None,
                liberal_y_fixed=liberal_y if args.regret_only else None,
                eval_fn=eval_snapshot,
            )

        # Save snapshots for re-rendering without re-running
        npz_path = output_dir / f"snapshots_{tag.replace(' ', '_')}.npz"
        np.savez(str(npz_path),
                 **{f"snap_{i}": s for i, s in enumerate(snapshots)})
        print(f"  Snapshots saved: {npz_path}")

        return {
            "w_lib": w_lib, "w_con": w_con, "w_reg": w_reg,
            "aep_liberal": aep_lib,
            "aep_conservative": aep_con,
            "aep_liberal_present": aep_lib_present,
            "regret": regret,
            "elapsed_s": elapsed,
            "history": history,
            "x_c": [float(v) for v in x_c],
            "y_c": [float(v) for v in y_c],
            "x_l": [float(v) for v in x_l],
            "y_l": [float(v) for v in y_l],
            "nb_x": [float(v) for v in nb_x],
            "nb_y": [float(v) for v in nb_y],
        }

    # --- Run ---
    if args.sweep:
        # Sweep weight combinations
        weight_configs = [
            (1.0, 1.0, 0.0, "lib+con (no regret)"),
            (1.0, 0.0, 0.0, "lib only"),
            (0.0, 1.0, 0.0, "con only"),
            (0.0, 0.0, 1.0, "regret only"),
            (1.0, 1.0, 1.0, "equal"),
            (1.0, 1.0, 5.0, "regret-heavy"),
            (1.0, 1.0, 10.0, "regret-dominant"),
            (0.5, 0.5, 1.0, "regret-balanced"),
        ]
        results = []
        for w_lib, w_con, w_reg, tag in weight_configs:
            r = run_flat(w_lib, w_con, w_reg, tag=tag)
            results.append(r)

        # Save sweep results
        sweep_path = output_dir / "sweep_results.json"
        # Strip history for compact storage
        for r in results:
            r["history"] = r["history"][-1:] if r["history"] else []
        with open(sweep_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSweep results saved: {sweep_path}")

        # Print summary table
        print(f"\n{'='*80}")
        print(f"{'Weights':>20s} {'Lib AEP':>10s} {'Con AEP':>10s} {'Regret':>10s}")
        print(f"{'='*80}")
        for r in results:
            wstr = f"({r['w_lib']:.1f},{r['w_con']:.1f},{r['w_reg']:.1f})"
            print(f"{wstr:>20s} {r['aep_liberal']:>10.2f} {r['aep_conservative']:>10.2f} {r['regret']:>10.4f}")

    else:
        # Single run
        r = run_flat(args.w_lib, args.w_con, args.w_reg, tag="single")

        # Save
        result_path = output_dir / "results.json"
        with open(result_path, "w") as f:
            json.dump(r, f, indent=2)
        print(f"\nResults saved: {result_path}")


if __name__ == "__main__":
    main()

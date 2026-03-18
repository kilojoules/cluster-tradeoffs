"""Greedy grid search for adversarial neighbor placement on the DEI case.

Places external turbines one at a time on a discrete grid, choosing the
location that maximizes design regret at each step.  Produces an MP4
animation with a regret heatmap at each greedy step.

Usage:
    pixi run python scripts/run_dei_greedy_grid.py
    pixi run python scripts/run_dei_greedy_grid.py --n-place=3 --grid-spacing=3
"""

import jax
jax.config.update("jax_enable_x64", True)

import argparse
import json
import time
from functools import partial
from pathlib import Path

import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.path import Path as MplPath

from pixwake import Curve, Turbine, WakeSimulation
from pixwake.deficit import BastankhahGaussianDeficit
from pixwake.optim.sgd import SGDSettings
from pixwake.optim.adversarial import GreedyGridSearch, GreedyGridSettings

# Unbuffered print
print = partial(print, flush=True)

# =============================================================================
# Configuration
# =============================================================================
D = 240.0  # rotor diameter (m) — IEA 15 MW class
N_TARGET = 50
N_PLACE = 30  # number of external turbines to place greedily
MIN_SPACING_D = 4.0
INNER_MAX_ITER = 500
INNER_LR = 50.0
GRID_SPACING_D = 5.0  # grid spacing in rotor diameters
GRID_PAD_D = 12.0     # how far outside the boundary to extend the grid

OUTPUT_DIR = Path("analysis/dei_greedy_grid")

# =============================================================================
# DEI turbine (15 MW, D=240m, hub=150m)
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
# DEI polygon boundary (dk0w_tender_3) — centered at centroid
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

# CCW ordering for containment_penalty
from scipy.spatial import ConvexHull
_hull = ConvexHull(_dk0w_raw - np.array([CENTROID_X, CENTROID_Y]))
boundary_np = (_dk0w_raw - np.array([CENTROID_X, CENTROID_Y]))[_hull.vertices]
boundary = jnp.array(boundary_np)
_polygon_path = MplPath(boundary_np)


# =============================================================================
# Wind data
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


# =============================================================================
# Initial target layout inside polygon
# =============================================================================
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
        # Denser grid fallback
        gx = np.arange(x_lo + D, x_hi - D, spacing * 0.7)
        gy = np.arange(y_lo + D, y_hi - D, spacing * 0.7)
        gx_2d, gy_2d = np.meshgrid(gx, gy)
        candidates = np.column_stack([gx_2d.ravel(), gy_2d.ravel()])
        inside = _polygon_path.contains_points(candidates)
        pts = candidates[inside]

    indices = np.round(np.linspace(0, len(pts) - 1, n_target)).astype(int)
    selected = pts[indices]
    return jnp.array(selected[:, 0]), jnp.array(selected[:, 1])


# =============================================================================
# Candidate neighbour grid (outside polygon + buffer)
# =============================================================================
def build_neighbor_grid(boundary_np, grid_spacing, pad):
    """Build a grid of candidate positions OUTSIDE the target polygon + buffer."""
    buffer = 2 * D  # minimum distance from boundary
    x_lo = boundary_np[:, 0].min() - pad
    x_hi = boundary_np[:, 0].max() + pad
    y_lo = boundary_np[:, 1].min() - pad
    y_hi = boundary_np[:, 1].max() + pad

    gx = np.arange(x_lo, x_hi, grid_spacing)
    gy = np.arange(y_lo, y_hi, grid_spacing)
    gx_2d, gy_2d = np.meshgrid(gx, gy)
    candidates = np.column_stack([gx_2d.ravel(), gy_2d.ravel()])

    # Offset polygon outward by buffer for exclusion zone
    from scipy.spatial import ConvexHull
    hull = ConvexHull(boundary_np)
    hull_pts = boundary_np[hull.vertices]
    centroid = hull_pts.mean(axis=0)

    # Expand hull outward by buffer
    expanded = centroid + (hull_pts - centroid) * (1 + buffer / np.linalg.norm(hull_pts - centroid, axis=1, keepdims=True))
    exclusion_path = MplPath(expanded)

    # Keep only points outside the exclusion zone
    outside = ~exclusion_path.contains_points(candidates)
    grid = candidates[outside]

    print(f"Neighbor grid: {len(grid)} candidates outside exclusion zone "
          f"(spacing={grid_spacing/D:.1f}D, pad={pad/D:.0f}D)")
    return grid, gx, gy


# =============================================================================
# Wind rose drawing helper
# =============================================================================
def draw_wind_rose(ax, wd_bins, ws_bins, weights):
    """Draw a wind rose on a polar axes."""
    n_bins = len(wd_bins)
    theta = np.deg2rad(90 - np.array(wd_bins))  # met convention -> math convention
    width = np.deg2rad(360 / n_bins) * 0.9

    # Color by wind speed
    ws_np = np.array(ws_bins)
    norm = plt.Normalize(vmin=ws_np.min(), vmax=ws_np.max())
    colors = plt.cm.YlOrRd(norm(ws_np))

    bars = ax.bar(theta, np.array(weights), width=width, bottom=0.0,
                  color=colors, edgecolor="gray", linewidth=0.4, alpha=0.85)

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)  # clockwise
    ax.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315],
                      ["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
                      fontsize=8)
    ax.set_rticks([])
    ax.set_title("Wind Rose", fontsize=11, pad=12)


# =============================================================================
# Animation
# =============================================================================
def render_animation(result, grid, boundary_np, output_path, gx_1d, gy_1d,
                     wd_bins, ws_bins, weights):
    """Render MP4 with regret heatmap + wind rose at each greedy step."""
    n_steps = len(result.regret_maps)
    wd_np = np.array(wd_bins)
    ws_np = np.array(ws_bins)
    w_np = np.array(weights)

    # Coordinate transform: centered coords in km
    def to_km(x, y):
        return np.asarray(x) / 1000.0, np.asarray(y) / 1000.0

    bnd_km = np.column_stack(to_km(boundary_np[:, 0], boundary_np[:, 1]))
    grid_km_x, grid_km_y = to_km(grid[:, 0], grid[:, 1])

    # Bounds for the plot
    pad_km = 1.0
    all_x_km = np.concatenate([grid_km_x, bnd_km[:, 0]])
    all_y_km = np.concatenate([grid_km_y, bnd_km[:, 1]])
    x_lo = all_x_km.min() - pad_km
    x_hi = all_x_km.max() + pad_km
    y_lo = all_y_km.min() - pad_km
    y_hi = all_y_km.max() + pad_km

    cmap = plt.cm.RdYlBu_r

    # Per-frame color ranges
    frame_vmin = []
    frame_vmax = []
    for rm in result.regret_maps:
        finite = np.array(rm)[np.isfinite(np.array(rm))]
        frame_vmin.append(float(finite.min()) if len(finite) else 0.0)
        frame_vmax.append(float(finite.max()) if len(finite) else 1.0)

    # Build interpolated regret grids for smooth heatmap
    from scipy.interpolate import griddata
    heatmap_res = 200
    xi = np.linspace(x_lo, x_hi, heatmap_res)
    yi = np.linspace(y_lo, y_hi, heatmap_res)
    xi_2d, yi_2d = np.meshgrid(xi, yi)

    heatmaps = []
    for step_idx in range(n_steps):
        rm = np.array(result.regret_maps[step_idx])
        mask = np.isfinite(rm)
        if mask.sum() > 3:
            zi = griddata(
                (grid_km_x[mask], grid_km_y[mask]),
                rm[mask],
                (xi_2d, yi_2d),
                method="cubic",
                fill_value=np.nan,
            )
        else:
            zi = np.full_like(xi_2d, np.nan)
        heatmaps.append(zi)

    placed_x_steps = []
    placed_y_steps = []
    for s in range(n_steps):
        idxs = result.placement_order[:s + 1]
        px = np.array([float(grid[i, 0]) for i in idxs])
        py = np.array([float(grid[i, 1]) for i in idxs])
        placed_x_steps.append(px)
        placed_y_steps.append(py)

    # --- Layout: heatmap (large), colorbar, wind rose (upper-right), regret bar (lower-right)
    fig = plt.figure(figsize=(20, 9))
    gs = fig.add_gridspec(2, 3, width_ratios=[2.3, 0.07, 1], height_ratios=[1, 1],
                          hspace=0.35, wspace=0.3)
    ax_map = fig.add_subplot(gs[:, 0])
    ax_cbar = fig.add_subplot(gs[:, 1])
    ax_rose = fig.add_subplot(gs[0, 2], projection="polar")
    ax_bar = fig.add_subplot(gs[1, 2])

    def draw(frame):
        ax_map.clear()
        ax_cbar.clear()
        ax_rose.clear()
        ax_bar.clear()

        step_idx = frame

        # --- Left panel: heatmap + layout ---
        im = ax_map.pcolormesh(
            xi, yi, heatmaps[step_idx],
            cmap=cmap, vmin=frame_vmin[step_idx], vmax=frame_vmax[step_idx],
            shading="auto", alpha=0.85,
        )

        poly = MplPolygon(bnd_km, closed=True, fill=True,
                          facecolor="white", edgecolor="black", lw=2.5,
                          alpha=0.9, zorder=3)
        ax_map.add_patch(poly)

        sc_lib = ax_map.scatter(*to_km(result.liberal_x, result.liberal_y),
                       facecolors="none", edgecolors="royalblue",
                       marker="^", s=40, linewidths=0.8,
                       label="Liberal layout", zorder=5)
        sc_lib.set_linestyle("--")

        ax_map.scatter(*to_km(result.target_x, result.target_y),
                       c="forestgreen", marker="^", s=40,
                       edgecolors="darkgreen", linewidths=0.5,
                       label="Conservative layout", zorder=5)

        px, py = placed_x_steps[step_idx], placed_y_steps[step_idx]
        px_km, py_km = to_km(px, py)

        if step_idx > 0:
            prev_px, prev_py = placed_x_steps[step_idx - 1], placed_y_steps[step_idx - 1]
            ppx_km, ppy_km = to_km(prev_px, prev_py)
            ax_map.scatter(ppx_km, ppy_km, c="red", marker="D", s=120,
                           edgecolors="darkred", linewidths=1.0, alpha=0.5,
                           zorder=6)

        new_idx = result.placement_order[step_idx]
        nx_km, ny_km = to_km(grid[new_idx, 0], grid[new_idx, 1])
        ax_map.scatter([nx_km], [ny_km], c="red", marker="*", s=350,
                       edgecolors="darkred", linewidths=1.5,
                       label=f"Placed (step {step_idx+1})", zorder=7)

        ax_map.scatter(px_km, py_km, c="red", marker="D", s=100,
                       edgecolors="darkred", linewidths=1.0,
                       label=f"Neighbors ({len(px)})", zorder=6)

        ax_map.set_xlim(x_lo, x_hi)
        ax_map.set_ylim(y_lo, y_hi)
        ax_map.set_aspect("equal")
        ax_map.set_xlabel("x (km)", fontsize=12)
        ax_map.set_ylabel("y (km)", fontsize=12)
        ax_map.legend(loc="lower left", fontsize=9, framealpha=0.9)

        regret_val = result.regret_history[step_idx]
        ax_map.set_title(
            f"Greedy Step {step_idx + 1}/{n_steps} — "
            f"Regret = {regret_val:.3f} GWh",
            fontsize=14, fontweight="bold",
        )

        # Colorbar in dedicated axes — just redraw each frame
        norm = plt.Normalize(vmin=frame_vmin[step_idx], vmax=frame_vmax[step_idx])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        fig.colorbar(sm, cax=ax_cbar, label="AEP Loss (GWh)")

        # --- Upper-right: wind rose ---
        draw_wind_rose(ax_rose, wd_np, ws_np, w_np)

        # --- Lower-right: regret vs step ---
        steps = np.arange(1, step_idx + 2)
        regrets = result.regret_history[:step_idx + 1]
        ax_bar.plot(steps, regrets, "o-", color="firebrick", ms=5, lw=2)
        ax_bar.fill_between(steps, 0, regrets, color="firebrick", alpha=0.15)
        ax_bar.set_xlabel("Neighbors Placed", fontsize=12)
        ax_bar.set_ylabel("Regret (GWh)", fontsize=12)
        ax_bar.set_xlim(0.5, n_steps + 0.5)
        ax_bar.set_ylim(0, max(result.regret_history) * 1.15)
        ax_bar.set_title("Regret vs. Neighbors Placed", fontsize=13)
        ax_bar.grid(True, alpha=0.3)
        ax_bar.text(steps[-1], regrets[-1] + max(result.regret_history) * 0.03,
                    f"{regrets[-1]:.1f}", ha="center", fontsize=10, fontweight="bold",
                    color="firebrick")

        fig.suptitle(
            f"DEI Greedy Grid Search — {N_TARGET} target turbines, "
            f"placing {n_steps} neighbors",
            fontsize=15, fontweight="bold", y=0.98,
        )

    fps = 0.5 if n_steps <= 5 else 2
    anim = FuncAnimation(fig, draw, frames=n_steps, interval=1000 // max(fps, 1), repeat=True)
    anim.save(str(output_path), writer="ffmpeg", fps=fps, dpi=150)
    plt.close(fig)
    print(f"Animation saved: {output_path}")

    # Also save final frame as PNG
    png_path = output_path.with_suffix(".png")
    fig2 = plt.figure(figsize=(20, 9))
    gs2 = fig2.add_gridspec(2, 3, width_ratios=[2.3, 0.07, 1], height_ratios=[1, 1],
                            hspace=0.35, wspace=0.3)
    ax_map = fig2.add_subplot(gs2[:, 0])
    ax_cbar2 = fig2.add_subplot(gs2[:, 1])
    ax_rose = fig2.add_subplot(gs2[0, 2], projection="polar")
    ax_bar = fig2.add_subplot(gs2[1, 2])

    last = n_steps - 1
    im = ax_map.pcolormesh(xi, yi, heatmaps[last], cmap=cmap,
                           vmin=frame_vmin[last], vmax=frame_vmax[last],
                           shading="auto", alpha=0.85)
    poly = MplPolygon(bnd_km, closed=True, fill=True,
                      facecolor="white", edgecolor="black", lw=2.5, alpha=0.9, zorder=3)
    ax_map.add_patch(poly)
    sc_lib = ax_map.scatter(*to_km(result.liberal_x, result.liberal_y),
                   facecolors="none", edgecolors="royalblue",
                   marker="^", s=40, linewidths=0.8, label="Liberal layout", zorder=5)
    sc_lib.set_linestyle("--")
    ax_map.scatter(*to_km(result.target_x, result.target_y),
                   c="forestgreen", marker="^", s=40,
                   edgecolors="darkgreen", linewidths=0.5,
                   label="Conservative layout", zorder=5)
    px, py = placed_x_steps[last], placed_y_steps[last]
    px_km, py_km = to_km(px, py)
    ax_map.scatter(px_km, py_km, c="red", marker="D", s=120,
                   edgecolors="darkred", linewidths=1.0, label=f"Neighbors ({len(px)})", zorder=6)
    for i, idx in enumerate(result.placement_order):
        nx_km, ny_km = to_km(grid[idx, 0], grid[idx, 1])
        ax_map.annotate(f"{i+1}", (nx_km, ny_km), fontsize=11, fontweight="bold",
                        ha="center", va="bottom", color="darkred",
                        xytext=(0, 8), textcoords="offset points")
    ax_map.set_xlim(x_lo, x_hi)
    ax_map.set_ylim(y_lo, y_hi)
    ax_map.set_aspect("equal")
    ax_map.set_xlabel("x (km)", fontsize=12)
    ax_map.set_ylabel("y (km)", fontsize=12)
    ax_map.legend(loc="lower left", fontsize=9)
    ax_map.set_title(f"Final Regret = {result.regret:.3f} GWh", fontsize=14, fontweight="bold")
    fig2.colorbar(im, cax=ax_cbar2, label="AEP Loss (GWh)")

    draw_wind_rose(ax_rose, wd_np, ws_np, w_np)

    steps = np.arange(1, n_steps + 1)
    ax_bar.plot(steps, result.regret_history, "o-", color="firebrick", ms=5, lw=2)
    ax_bar.fill_between(steps, 0, result.regret_history, color="firebrick", alpha=0.15)
    ax_bar.set_xlabel("Neighbors Placed", fontsize=12)
    ax_bar.set_ylabel("Regret (GWh)", fontsize=12)
    ax_bar.set_xlim(0.5, n_steps + 0.5)
    ax_bar.set_ylim(0, max(result.regret_history) * 1.15)
    ax_bar.set_title("Regret vs. Neighbors Placed", fontsize=13)
    ax_bar.grid(True, alpha=0.3)
    ax_bar.text(steps[-1], result.regret_history[-1] + max(result.regret_history) * 0.03,
                f"{result.regret_history[-1]:.1f}", ha="center", fontsize=10,
                fontweight="bold", color="firebrick")
    fig2.suptitle(f"DEI Greedy Grid Search — {N_TARGET} targets, {n_steps} neighbors placed",
                  fontsize=15, fontweight="bold", y=0.98)
    fig2.savefig(str(png_path), dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Final frame saved: {png_path}")


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="DEI greedy grid search for adversarial neighbor placement",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-target", type=int, default=N_TARGET)
    parser.add_argument("--n-place", type=int, default=N_PLACE)
    parser.add_argument("--min-spacing-D", type=float, default=MIN_SPACING_D)
    parser.add_argument("--inner-lr", type=float, default=INNER_LR)
    parser.add_argument("--inner-max-iter", type=int, default=INNER_MAX_ITER)
    parser.add_argument("--grid-spacing-D", type=float, default=GRID_SPACING_D)
    parser.add_argument("--grid-pad-D", type=float, default=GRID_PAD_D)
    parser.add_argument("--screen-top-k", type=int, default=10)
    parser.add_argument("--screen-chunk-size", type=int, default=100)
    parser.add_argument("--n-inner-starts", type=int, default=1,
                        help="Number of random inner starts per candidate (best-of-K)")
    parser.add_argument("--eval-parallel", action="store_true",
                        help="Run inner SGDs in parallel via vmap (use on GPU)")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_target = args.n_target
    n_place = args.n_place
    min_spacing = args.min_spacing_D * D
    grid_spacing = args.grid_spacing_D * D
    grid_pad = args.grid_pad_D * D

    print("=" * 70)
    print("DEI GREEDY GRID SEARCH")
    print(f"  {n_target} target turbines, placing {n_place} external turbines")
    print(f"  Grid spacing: {args.grid_spacing_D:.1f}D = {grid_spacing:.0f} m")
    print(f"  Grid pad: {args.grid_pad_D:.0f}D = {grid_pad:.0f} m")
    print(f"  Min spacing: {args.min_spacing_D:.1f}D = {min_spacing:.0f} m")
    print(f"  Inner SGD: lr={args.inner_lr}, max_iter={args.inner_max_iter}")
    print(f"  Inner starts: {args.n_inner_starts} (best-of-K)")
    print(f"  Eval parallel: {args.eval_parallel}")
    print("=" * 70)

    # Load data
    turbine = create_dei_turbine()
    wd, ws, weights = load_wind_data()
    sim = WakeSimulation(turbine, BastankhahGaussianDeficit(k=0.04))

    print(f"\nBoundary: {boundary_np.shape[0]} vertices (CCW)")
    dominant_idx = int(jnp.argmax(weights))
    print(f"Wind rose: {len(wd)} bins, dominant ~{float(wd[dominant_idx]):.0f} deg")

    # Target layout
    init_x, init_y = generate_target_grid(boundary_np, n_target, spacing=4 * D)
    print(f"Target: {n_target} turbines on grid inside polygon")

    # Neighbour grid
    grid, gx_1d, gy_1d = build_neighbor_grid(boundary_np, grid_spacing, grid_pad)
    grid_jax = jnp.array(grid)

    # SGD settings — constant LR for all iterations
    sgd_settings = SGDSettings(
        learning_rate=args.inner_lr,
        max_iter=args.inner_max_iter,
        additional_constant_lr_iterations=args.inner_max_iter,
        tol=1e-6,
    )
    settings = GreedyGridSettings(
        sgd_settings=sgd_settings,
        n_inner_starts=args.n_inner_starts,
        screen_top_k=args.screen_top_k,
        screen_chunk_size=args.screen_chunk_size,
        eval_parallel=args.eval_parallel,
        verbose=True,
    )

    # Run greedy search
    searcher = GreedyGridSearch(
        sim, boundary, min_spacing,
        ws_amb=ws, wd_amb=wd,
        weights=weights,
    )

    print(f"\nStarting greedy grid search ({n_place} placements, "
          f"{len(grid)} candidates each)...")
    t0 = time.time()

    result = searcher.search(
        init_x, init_y,
        grid=grid_jax,
        n_neighbors=n_place,
        settings=settings,
    )

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Save results
    results_data = {
        "n_target": n_target,
        "n_placed": n_place,
        "n_grid_candidates": len(grid),
        "grid_spacing_m": float(grid_spacing),
        "min_spacing_m": float(min_spacing),
        "liberal_aep_gwh": float(result.liberal_aep),
        "conservative_aep_gwh": float(result.conservative_aep),
        "regret_gwh": float(result.regret),
        "placement_order": result.placement_order,
        "regret_history": result.regret_history,
        "neighbor_x": [float(x) for x in result.neighbor_x],
        "neighbor_y": [float(y) for y in result.neighbor_y],
        "elapsed_s": elapsed,
    }
    json_path = output_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"Results saved: {json_path}")

    # Save regret maps for re-rendering without re-running
    npz_path = output_dir / "regret_maps.npz"
    np.savez(str(npz_path), **{f"step_{i}": np.array(rm) for i, rm in enumerate(result.regret_maps)})
    print(f"Regret maps saved: {npz_path}")

    # Render animation
    mp4_path = output_dir / "dei_greedy_grid.mp4"
    render_animation(result, grid, boundary_np, mp4_path, gx_1d, gy_1d,
                     wd, ws, weights)


if __name__ == "__main__":
    main()

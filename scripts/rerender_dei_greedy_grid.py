"""Re-render the DEI greedy grid animation from saved results.

Reconstructs regret heatmaps by re-evaluating screening AEP loss at each step
(cheap forward eval only, no inner SGD). Uses saved placement order from results.json.

Usage:
    pixi run python scripts/rerender_dei_greedy_grid.py
    pixi run python scripts/rerender_dei_greedy_grid.py --results-dir analysis/dei_greedy_grid_30
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
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.path import Path as MplPath
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull

from pixwake import Curve, Turbine, WakeSimulation
from pixwake.deficit import BastankhahGaussianDeficit
from pixwake.optim.sgd import SGDSettings, topfarm_sgd_solve

print = partial(print, flush=True)

D = 240.0
N_TARGET = 50
MIN_SPACING_D = 4.0
GRID_SPACING_D = 5.0
GRID_PAD_D = 12.0


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

_hull = ConvexHull(_dk0w_raw - np.array([CENTROID_X, CENTROID_Y]))
boundary_np = (_dk0w_raw - np.array([CENTROID_X, CENTROID_Y]))[_hull.vertices]
boundary = jnp.array(boundary_np)
_polygon_path = MplPath(boundary_np)


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


def build_neighbor_grid(boundary_np, grid_spacing, pad):
    buffer = 2 * D
    x_lo = boundary_np[:, 0].min() - pad
    x_hi = boundary_np[:, 0].max() + pad
    y_lo = boundary_np[:, 1].min() - pad
    y_hi = boundary_np[:, 1].max() + pad
    gx = np.arange(x_lo, x_hi, grid_spacing)
    gy = np.arange(y_lo, y_hi, grid_spacing)
    gx_2d, gy_2d = np.meshgrid(gx, gy)
    candidates = np.column_stack([gx_2d.ravel(), gy_2d.ravel()])
    hull = ConvexHull(boundary_np)
    hull_pts = boundary_np[hull.vertices]
    centroid = hull_pts.mean(axis=0)
    expanded = centroid + (hull_pts - centroid) * (
        1 + buffer / np.linalg.norm(hull_pts - centroid, axis=1, keepdims=True)
    )
    exclusion_path = MplPath(expanded)
    outside = ~exclusion_path.contains_points(candidates)
    grid = candidates[outside]
    print(f"Neighbor grid: {len(grid)} candidates outside exclusion zone "
          f"(spacing={grid_spacing/D:.1f}D, pad={pad/D:.0f}D)")
    return grid, gx, gy


def draw_wind_rose(ax, wd_bins, ws_bins, weights):
    n_bins = len(wd_bins)
    theta = np.deg2rad(90 - np.array(wd_bins))
    width = np.deg2rad(360 / n_bins) * 0.9
    ws_np = np.array(ws_bins)
    norm = plt.Normalize(vmin=ws_np.min(), vmax=ws_np.max())
    colors = plt.cm.YlOrRd(norm(ws_np))
    ax.bar(theta, np.array(weights), width=width, bottom=0.0,
           color=colors, edgecolor="gray", linewidth=0.4, alpha=0.85)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315],
                      ["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
                      fontsize=8)
    ax.set_rticks([])
    ax.set_title("Wind Rose", fontsize=11, pad=12)


def compute_aep(sim, target_x, target_y, ws_amb, wd_amb, weights,
                neighbor_x=None, neighbor_y=None):
    """Compute AEP in GWh for the target turbines."""
    n_target = len(target_x)
    if neighbor_x is not None and len(neighbor_x) > 0:
        all_x = jnp.concatenate([target_x, neighbor_x])
        all_y = jnp.concatenate([target_y, neighbor_y])
    else:
        all_x = target_x
        all_y = target_y
    result = sim(all_x, all_y, ws_amb=ws_amb, wd_amb=wd_amb)
    power = result.power()[:, :n_target]  # (n_dirs, n_target)
    weighted_power = jnp.sum(power * weights[:, None])
    return weighted_power * 8760 / 1e6  # kW -> GWh


def reconstruct_regret_maps(sim, grid, placement_order, liberal_x, liberal_y,
                            ws_amb, wd_amb, weights):
    """Reconstruct screening-based regret heatmaps from saved placement data."""
    n_steps = len(placement_order)
    n_cands = len(grid)

    # Liberal AEP (no neighbors)
    liberal_aep = compute_aep(sim, liberal_x, liberal_y, ws_amb, wd_amb, weights)
    print(f"Liberal AEP (no neighbors): {liberal_aep:.3f} GWh")

    regret_maps = []
    placed_indices = []

    for step in range(n_steps):
        print(f"  Reconstructing heatmap for step {step + 1}/{n_steps}...", end=" ")
        t0 = time.time()

        # Already-placed neighbors
        placed_x = jnp.array([grid[i, 0] for i in placed_indices]) if placed_indices else jnp.array([])
        placed_y = jnp.array([grid[i, 1] for i in placed_indices]) if placed_indices else jnp.array([])

        regret_map = np.full(n_cands, np.nan)

        for c in range(n_cands):
            if c in placed_indices:
                continue
            cx = jnp.array([grid[c, 0]])
            cy = jnp.array([grid[c, 1]])
            if len(placed_x) > 0:
                nx = jnp.concatenate([placed_x, cx])
                ny = jnp.concatenate([placed_y, cy])
            else:
                nx, ny = cx, cy

            aep_with = compute_aep(sim, liberal_x, liberal_y, ws_amb, wd_amb,
                                   weights, neighbor_x=nx, neighbor_y=ny)
            regret_map[c] = float(liberal_aep - aep_with)

        regret_maps.append(regret_map)
        placed_indices.append(placement_order[step])
        elapsed = time.time() - t0
        print(f"{elapsed:.1f}s")

    return regret_maps


def render_animation(regret_maps, results, grid, boundary_np, output_path,
                     wd_bins, ws_bins, weights, liberal_x, liberal_y,
                     conservative_xs=None, conservative_ys=None):
    """Render MP4 with per-frame color range."""
    n_steps = len(regret_maps)
    placement_order = results["placement_order"]
    regret_history = results["regret_history"]
    wd_np = np.array(wd_bins)
    ws_np = np.array(ws_bins)
    w_np = np.array(weights)

    def to_km(x, y):
        return np.asarray(x) / 1000.0, np.asarray(y) / 1000.0

    bnd_km = np.column_stack(to_km(boundary_np[:, 0], boundary_np[:, 1]))
    grid_km_x, grid_km_y = to_km(grid[:, 0], grid[:, 1])

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
    for rm in regret_maps:
        finite = np.array(rm)[np.isfinite(np.array(rm))]
        frame_vmin.append(float(finite.min()) if len(finite) else 0.0)
        frame_vmax.append(float(finite.max()) if len(finite) else 1.0)

    # Interpolated heatmaps
    heatmap_res = 200
    xi = np.linspace(x_lo, x_hi, heatmap_res)
    yi = np.linspace(y_lo, y_hi, heatmap_res)
    xi_2d, yi_2d = np.meshgrid(xi, yi)

    heatmaps = []
    for step_idx in range(n_steps):
        rm = np.array(regret_maps[step_idx])
        mask = np.isfinite(rm)
        if mask.sum() > 3:
            zi = griddata(
                (grid_km_x[mask], grid_km_y[mask]), rm[mask],
                (xi_2d, yi_2d), method="cubic", fill_value=np.nan,
            )
        else:
            zi = np.full_like(xi_2d, np.nan)
        heatmaps.append(zi)

    placed_x_steps = []
    placed_y_steps = []
    for s in range(n_steps):
        idxs = placement_order[:s + 1]
        px = np.array([float(grid[i, 0]) for i in idxs])
        py = np.array([float(grid[i, 1]) for i in idxs])
        placed_x_steps.append(px)
        placed_y_steps.append(py)

    lib_x_np = np.array(liberal_x)
    lib_y_np = np.array(liberal_y)

    fig = plt.figure(figsize=(20, 9))
    # Use explicit colorbar axes to avoid remove() issues with gridspec
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

        im = ax_map.pcolormesh(
            xi, yi, heatmaps[step_idx],
            cmap=cmap, vmin=frame_vmin[step_idx], vmax=frame_vmax[step_idx],
            shading="auto", alpha=0.85,
        )

        poly = MplPolygon(bnd_km, closed=True, fill=True,
                          facecolor="white", edgecolor="black", lw=2.5,
                          alpha=0.9, zorder=3)
        ax_map.add_patch(poly)

        sc_lib = ax_map.scatter(*to_km(lib_x_np, lib_y_np),
                       facecolors="none", edgecolors="royalblue",
                       marker="^", s=40, linewidths=0.8,
                       label="Liberal layout", zorder=5)
        sc_lib.set_linestyle("--")

        if conservative_xs is not None:
            cx = np.array(conservative_xs[step_idx])
            cy = np.array(conservative_ys[step_idx])
            ax_map.scatter(*to_km(cx, cy),
                           c="forestgreen", marker="^", s=40,
                           edgecolors="darkgreen", linewidths=0.5,
                           label="Conservative layout", zorder=5)

        px, py = placed_x_steps[step_idx], placed_y_steps[step_idx]
        px_km, py_km = to_km(px, py)

        if step_idx > 0:
            prev_px, prev_py = placed_x_steps[step_idx - 1], placed_y_steps[step_idx - 1]
            ppx_km, ppy_km = to_km(prev_px, prev_py)
            ax_map.scatter(ppx_km, ppy_km, c="red", marker="D", s=120,
                           edgecolors="darkred", linewidths=1.0, alpha=0.5, zorder=6)

        new_idx = placement_order[step_idx]
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

        regret_val = regret_history[step_idx]
        ax_map.set_title(
            f"Greedy Step {step_idx + 1}/{n_steps} — "
            f"Regret = {regret_val:.3f} GWh",
            fontsize=14, fontweight="bold",
        )

        # Colorbar in dedicated axes — just redraw each frame
        norm = plt.Normalize(vmin=frame_vmin[step_idx], vmax=frame_vmax[step_idx])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        fig.colorbar(sm, cax=ax_cbar, label="AEP Loss (GWh)")

        # Wind rose
        draw_wind_rose(ax_rose, wd_np, ws_np, w_np)

        # Regret vs step
        steps = np.arange(1, step_idx + 2)
        regrets = regret_history[:step_idx + 1]
        ax_bar.plot(steps, regrets, "o-", color="firebrick", ms=5, lw=2)
        ax_bar.fill_between(steps, 0, regrets, color="firebrick", alpha=0.15)
        ax_bar.set_xlabel("Neighbors Placed", fontsize=12)
        ax_bar.set_ylabel("Regret (GWh)", fontsize=12)
        ax_bar.set_xlim(0.5, n_steps + 0.5)
        ax_bar.set_ylim(0, max(regret_history) * 1.15)
        ax_bar.set_title("Regret vs. Neighbors Placed", fontsize=13)
        ax_bar.grid(True, alpha=0.3)
        ax_bar.text(steps[-1], regrets[-1] + max(regret_history) * 0.03,
                    f"{regrets[-1]:.1f}", ha="center", fontsize=10,
                    fontweight="bold", color="firebrick")

        fig.suptitle(
            f"DEI Greedy Grid Search — {N_TARGET} target turbines, "
            f"placing {n_steps} neighbors",
            fontsize=15, fontweight="bold", y=0.98,
        )

    fps = 0.5 if n_steps <= 5 else 2
    anim = FuncAnimation(fig, draw, frames=n_steps,
                         interval=1000 // max(fps, 1), repeat=True)
    anim.save(str(output_path), writer="ffmpeg", fps=fps, dpi=150)
    plt.close(fig)
    print(f"Animation saved: {output_path}")

    # Final frame as PNG
    png_path = output_path.with_suffix(".png")
    fig2 = plt.figure(figsize=(20, 9))
    gs2 = fig2.add_gridspec(2, 2, width_ratios=[2.4, 1], height_ratios=[1, 1],
                            hspace=0.35, wspace=0.25)
    ax_map = fig2.add_subplot(gs2[:, 0])
    ax_rose = fig2.add_subplot(gs2[0, 1], projection="polar")
    ax_bar = fig2.add_subplot(gs2[1, 1])

    last = n_steps - 1
    im = ax_map.pcolormesh(xi, yi, heatmaps[last], cmap=cmap,
                           vmin=frame_vmin[last], vmax=frame_vmax[last],
                           shading="auto", alpha=0.85)
    poly = MplPolygon(bnd_km, closed=True, fill=True,
                      facecolor="white", edgecolor="black", lw=2.5,
                      alpha=0.9, zorder=3)
    ax_map.add_patch(poly)
    sc_lib = ax_map.scatter(*to_km(lib_x_np, lib_y_np),
                   facecolors="none", edgecolors="royalblue",
                   marker="^", s=40, linewidths=0.8, label="Liberal layout", zorder=5)
    sc_lib.set_linestyle("--")
    if conservative_xs is not None:
        cx = np.array(conservative_xs[last])
        cy = np.array(conservative_ys[last])
        ax_map.scatter(*to_km(cx, cy),
                       c="forestgreen", marker="^", s=40,
                       edgecolors="darkgreen", linewidths=0.5,
                       label="Conservative layout", zorder=5)
    px, py = placed_x_steps[last], placed_y_steps[last]
    px_km, py_km = to_km(px, py)
    ax_map.scatter(px_km, py_km, c="red", marker="D", s=120,
                   edgecolors="darkred", linewidths=1.0,
                   label=f"Neighbors ({len(px)})", zorder=6)
    for i, idx in enumerate(placement_order):
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
    ax_map.set_title(f"Final Regret = {regret_history[-1]:.3f} GWh",
                     fontsize=14, fontweight="bold")
    fig2.colorbar(im, ax=ax_map, shrink=0.8, pad=0.02, label="AEP Loss (GWh)")

    draw_wind_rose(ax_rose, wd_np, ws_np, w_np)

    steps = np.arange(1, n_steps + 1)
    ax_bar.plot(steps, regret_history, "o-", color="firebrick", ms=5, lw=2)
    ax_bar.fill_between(steps, 0, regret_history, color="firebrick", alpha=0.15)
    ax_bar.set_xlabel("Neighbors Placed", fontsize=12)
    ax_bar.set_ylabel("Regret (GWh)", fontsize=12)
    ax_bar.set_xlim(0.5, n_steps + 0.5)
    ax_bar.set_ylim(0, max(regret_history) * 1.15)
    ax_bar.set_title("Regret vs. Neighbors Placed", fontsize=13)
    ax_bar.grid(True, alpha=0.3)
    ax_bar.text(steps[-1], regret_history[-1] + max(regret_history) * 0.03,
                f"{regret_history[-1]:.1f}", ha="center", fontsize=10,
                fontweight="bold", color="firebrick")
    fig2.suptitle(f"DEI Greedy Grid Search — {N_TARGET} targets, {n_steps} neighbors placed",
                  fontsize=15, fontweight="bold", y=0.98)
    fig2.savefig(str(png_path), dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Final frame saved: {png_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Re-render DEI greedy grid animation from saved results",
    )
    parser.add_argument("--results-dir", type=str,
                        default="analysis/dei_greedy_grid_30")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    json_path = results_dir / "results.json"
    with open(json_path) as f:
        results = json.load(f)

    print(f"Loaded results: {json_path}")
    print(f"  {results['n_placed']} neighbors placed, "
          f"final regret = {results['regret_gwh']:.3f} GWh")

    # Rebuild infrastructure
    turbine = create_dei_turbine()
    wd, ws, weights = load_wind_data()
    sim = WakeSimulation(turbine, BastankhahGaussianDeficit(k=0.04))

    grid, gx_1d, gy_1d = build_neighbor_grid(
        boundary_np, GRID_SPACING_D * D, GRID_PAD_D * D)

    init_x, init_y = generate_target_grid(boundary_np, N_TARGET, spacing=4 * D)

    # Compute liberal layout (no neighbors)
    print("Computing liberal layout...")
    sgd_settings = SGDSettings(learning_rate=50.0, max_iter=5000, additional_constant_lr_iterations=5000, tol=1e-6)

    def liberal_objective(x, y):
        return -compute_aep(sim, x, y, ws, wd, weights)

    liberal_x, liberal_y = topfarm_sgd_solve(
        liberal_objective, init_x, init_y, boundary, MIN_SPACING_D * D,
        settings=sgd_settings,
    )
    print(f"Liberal layout optimized: {len(liberal_x)} turbines")

    # Check if regret maps already saved
    npz_path = results_dir / "regret_maps.npz"
    if npz_path.exists():
        print(f"Loading cached regret maps from {npz_path}")
        data = np.load(str(npz_path))
        regret_maps = [data[f"step_{i}"] for i in range(results["n_placed"])]
    else:
        print("Reconstructing screening heatmaps (forward eval only)...")
        regret_maps = reconstruct_regret_maps(
            sim, grid, results["placement_order"],
            liberal_x, liberal_y, ws, wd, weights,
        )
        # Save for future re-renders
        np.savez(str(npz_path),
                 **{f"step_{i}": rm for i, rm in enumerate(regret_maps)})
        print(f"Regret maps cached: {npz_path}")

    # Compute per-step conservative layouts (re-optimized with neighbors placed so far)
    cons_cache_path = results_dir / "conservative_layouts.npz"
    n_placed = results["n_placed"]
    placement_order = results["placement_order"]
    if cons_cache_path.exists():
        print(f"Loading cached conservative layouts from {cons_cache_path}")
        cons_data = np.load(str(cons_cache_path))
        conservative_xs = [jnp.array(cons_data[f"cx_{i}"]) for i in range(n_placed)]
        conservative_ys = [jnp.array(cons_data[f"cy_{i}"]) for i in range(n_placed)]
    else:
        print(f"Computing conservative layouts at each step ({n_placed} steps)...")
        conservative_xs = []
        conservative_ys = []
        placed_indices = []
        for step in range(n_placed):
            placed_indices.append(placement_order[step])
            nx = jnp.array([grid[i, 0] for i in placed_indices])
            ny = jnp.array([grid[i, 1] for i in placed_indices])

            def cons_obj(x, y, _nx=nx, _ny=ny):
                return -compute_aep(sim, x, y, ws, wd, weights,
                                    neighbor_x=_nx, neighbor_y=_ny)

            t0 = time.time()
            cx, cy = topfarm_sgd_solve(
                cons_obj, init_x, init_y, boundary, MIN_SPACING_D * D,
                settings=sgd_settings,
            )
            conservative_xs.append(cx)
            conservative_ys.append(cy)
            elapsed = time.time() - t0
            print(f"  Step {step+1}/{n_placed}: {elapsed:.1f}s")

        np.savez(str(cons_cache_path),
                 **{f"cx_{i}": np.array(cx) for i, cx in enumerate(conservative_xs)},
                 **{f"cy_{i}": np.array(cy) for i, cy in enumerate(conservative_ys)})
        print(f"Conservative layouts cached: {cons_cache_path}")

    # Render
    mp4_path = results_dir / "dei_greedy_grid.mp4"
    render_animation(
        regret_maps, results, grid, boundary_np, mp4_path,
        wd, ws, weights, liberal_x, liberal_y,
        conservative_xs, conservative_ys,
    )


if __name__ == "__main__":
    main()

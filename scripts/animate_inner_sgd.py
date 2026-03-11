"""Animate the inner SGD optimization trajectory for one greedy step.

Shows how the target layout evolves iteration-by-iteration as the inner
SGD converges, given a fixed set of neighbors. Runs the SGD at many
iteration counts to capture snapshots of the trajectory.

Usage:
    pixi run python scripts/animate_inner_sgd.py
    pixi run python scripts/animate_inner_sgd.py --step=15 --max-iter=5000
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
from scipy.spatial import ConvexHull

from pixwake import Curve, Turbine, WakeSimulation
from pixwake.deficit import BastankhahGaussianDeficit
from pixwake.optim.sgd import SGDSettings, topfarm_sgd_solve

print = partial(print, flush=True)

D = 240.0
N_TARGET = 50
MIN_SPACING_D = 4.0


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
    print(f"Neighbor grid: {len(grid)} candidates")
    return grid, gx, gy


def compute_aep(sim, target_x, target_y, ws_amb, wd_amb, weights,
                neighbor_x=None, neighbor_y=None):
    n_target = len(target_x)
    if neighbor_x is not None and len(neighbor_x) > 0:
        all_x = jnp.concatenate([target_x, neighbor_x])
        all_y = jnp.concatenate([target_y, neighbor_y])
    else:
        all_x = target_x
        all_y = target_y
    result = sim(all_x, all_y, ws_amb=ws_amb, wd_amb=wd_amb)
    power = result.power()[:, :n_target]
    weighted_power = jnp.sum(power * weights[:, None])
    return weighted_power * 8760 / 1e6


def main():
    parser = argparse.ArgumentParser(
        description="Animate inner SGD trajectory for one greedy step",
    )
    parser.add_argument("--results-dir", type=str,
                        default="analysis/dei_greedy_grid_5k")
    parser.add_argument("--step", type=int, default=15,
                        help="Which greedy step to visualize (1-indexed)")
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=50.0)
    args = parser.parse_args()

    output_dir = Path("analysis/inner_sgd_animation")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    results_path = Path(args.results_dir) / "results.json"
    with open(results_path) as f:
        results = json.load(f)

    step = args.step  # 1-indexed
    placement_order = results["placement_order"]
    n_placed = len(placement_order)
    assert 1 <= step <= n_placed, f"Step must be 1..{n_placed}"

    # Setup
    turbine = create_dei_turbine()
    wd, ws, weights = load_wind_data()
    sim = WakeSimulation(turbine, BastankhahGaussianDeficit(k=0.04))
    init_x, init_y = generate_target_grid(boundary_np, N_TARGET, spacing=4 * D)
    min_spacing = MIN_SPACING_D * D

    # Build neighbor grid to get positions
    grid, _, _ = build_neighbor_grid(boundary_np, 5.0 * D, 12.0 * D)

    # Neighbors placed up to this step
    placed_indices = placement_order[:step]
    neighbor_x = jnp.array([grid[i, 0] for i in placed_indices])
    neighbor_y = jnp.array([grid[i, 1] for i in placed_indices])
    print(f"Step {step}: {len(neighbor_x)} neighbors placed")

    def conservative_objective(x, y):
        return -compute_aep(sim, x, y, ws, wd, weights,
                            neighbor_x=neighbor_x, neighbor_y=neighbor_y)

    # Sample iteration counts: dense early, sparser later
    max_iter = args.max_iter
    iter_counts = sorted(set(
        [1, 2, 5] +
        list(range(10, 100, 10)) +
        list(range(100, 500, 25)) +
        list(range(500, min(2001, max_iter + 1), 50)) +
        list(range(2000, max_iter + 1, 100)) +
        [max_iter]
    ))
    iter_counts = [n for n in iter_counts if n <= max_iter]
    print(f"Sampling {len(iter_counts)} iteration counts up to {max_iter}")

    # Run SGD at each iteration count
    snapshots = []  # (n_iter, aep, x, y)
    for i, n_iter in enumerate(iter_counts):
        t0 = time.time()
        settings = SGDSettings(
            learning_rate=args.lr,
            max_iter=n_iter,
            additional_constant_lr_iterations=n_iter,
            tol=0.0,
        )
        opt_x, opt_y = topfarm_sgd_solve(
            conservative_objective, init_x, init_y, boundary, min_spacing,
            settings=settings,
        )
        aep = float(compute_aep(sim, opt_x, opt_y, ws, wd, weights,
                                neighbor_x=neighbor_x, neighbor_y=neighbor_y))
        elapsed = time.time() - t0
        snapshots.append((n_iter, aep, np.array(opt_x), np.array(opt_y)))
        if (i + 1) % 10 == 0 or n_iter == max_iter:
            print(f"  iter={n_iter:5d}  AEP={aep:.3f} GWh  ({elapsed:.1f}s)")

    # Initial AEP
    init_aep = float(compute_aep(sim, init_x, init_y, ws, wd, weights,
                                 neighbor_x=neighbor_x, neighbor_y=neighbor_y))
    final_aep = snapshots[-1][1]
    print(f"\nInitial AEP: {init_aep:.3f} GWh")
    print(f"Final AEP ({max_iter} iter): {final_aep:.3f} GWh")
    print(f"Improvement: {final_aep - init_aep:+.3f} GWh")

    # ---- Render animation ----
    def to_km(x, y):
        return np.asarray(x) / 1000.0, np.asarray(y) / 1000.0

    bnd_km = np.column_stack(to_km(boundary_np[:, 0], boundary_np[:, 1]))
    nb_km_x, nb_km_y = to_km(neighbor_x, neighbor_y)
    init_km_x, init_km_y = to_km(init_x, init_y)

    pad_km = 1.0
    all_pts_x = np.concatenate([bnd_km[:, 0], nb_km_x])
    all_pts_y = np.concatenate([bnd_km[:, 1], nb_km_y])
    x_lo, x_hi = all_pts_x.min() - pad_km, all_pts_x.max() + pad_km
    y_lo, y_hi = all_pts_y.min() - pad_km, all_pts_y.max() + pad_km

    iters_all = [s[0] for s in snapshots]
    aeps_all = [s[1] for s in snapshots]

    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1], wspace=0.25)
    ax_layout = fig.add_subplot(gs[0])
    ax_conv = fig.add_subplot(gs[1])

    def draw(frame):
        ax_layout.clear()
        ax_conv.clear()

        snap = snapshots[frame]
        cur_iter = snap[0]
        cur_aep = snap[1]
        cur_x, cur_y = snap[2], snap[3]

        # ---- Layout panel ----
        poly = MplPolygon(bnd_km, closed=True, fill=True,
                          facecolor="lightyellow", edgecolor="black",
                          lw=2, alpha=0.9, zorder=2)
        ax_layout.add_patch(poly)

        # Neighbors
        ax_layout.scatter(nb_km_x, nb_km_y, c="red", marker="D", s=70,
                          edgecolors="darkred", linewidths=0.5, alpha=0.6,
                          label=f"Neighbors ({len(neighbor_x)})", zorder=3)

        # Initial layout (dashed outlines)
        sc_init = ax_layout.scatter(init_km_x, init_km_y,
                                     facecolors="none", edgecolors="gray",
                                     marker="^", s=40, linewidths=0.6,
                                     label="Initial layout", zorder=4)
        sc_init.set_linestyle("--")

        # Current layout (solid)
        ax_layout.scatter(*to_km(cur_x, cur_y), c="forestgreen", marker="^",
                          s=50, edgecolors="darkgreen", linewidths=0.5,
                          label=f"Iter {cur_iter}", zorder=5)

        # Draw arrows from initial to current for each turbine
        for j in range(len(cur_x)):
            dx = float(cur_x[j] - init_x[j]) / 1000
            dy = float(cur_y[j] - init_y[j]) / 1000
            if np.sqrt(dx**2 + dy**2) > 0.05:  # only show if moved > 50m
                ax_layout.annotate(
                    "", xy=(init_km_x[j] + dx, init_km_y[j] + dy),
                    xytext=(init_km_x[j], init_km_y[j]),
                    arrowprops=dict(arrowstyle="-", color="forestgreen",
                                    lw=0.5, alpha=0.4),
                    zorder=4,
                )

        ax_layout.set_xlim(x_lo, x_hi)
        ax_layout.set_ylim(y_lo, y_hi)
        ax_layout.set_aspect("equal")
        ax_layout.set_xlabel("x (km)", fontsize=12)
        ax_layout.set_ylabel("y (km)", fontsize=12)
        ax_layout.legend(loc="lower left", fontsize=9, framealpha=0.9)
        ax_layout.set_title(
            f"Inner SGD — Step {step}, Iter {cur_iter}/{max_iter}\n"
            f"AEP = {cur_aep:.2f} GWh",
            fontsize=14, fontweight="bold",
        )

        # ---- Convergence panel ----
        # Full curve in light gray
        ax_conv.plot(iters_all, aeps_all, "-", color="lightgray", lw=1.5, zorder=1)
        # Progress up to current
        mask = [i for i, it in enumerate(iters_all) if it <= cur_iter]
        ax_conv.plot([iters_all[i] for i in mask],
                     [aeps_all[i] for i in mask],
                     "o-", color="forestgreen", ms=3, lw=2.5, zorder=3)
        ax_conv.scatter([cur_iter], [cur_aep], c="forestgreen", s=100,
                        edgecolors="darkgreen", linewidths=1.5, zorder=5)

        ax_conv.set_xlabel("SGD Iterations", fontsize=12)
        ax_conv.set_ylabel("Conservative AEP (GWh)", fontsize=12)
        ax_conv.set_title("Convergence", fontsize=13, fontweight="bold")
        ax_conv.set_xlim(0, max_iter * 1.05)
        ax_conv.set_ylim(min(aeps_all) - 3, max(aeps_all) + 3)
        ax_conv.grid(True, alpha=0.3)

        improvement = cur_aep - init_aep
        ax_conv.text(0.95, 0.05,
                     f"$\\Delta$AEP = {improvement:+.2f} GWh",
                     transform=ax_conv.transAxes, fontsize=12,
                     fontweight="bold", color="forestgreen",
                     ha="right", va="bottom",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                               edgecolor="forestgreen", alpha=0.9))

        fig.suptitle(
            f"Inner Optimization Trajectory — Greedy Step {step}/{n_placed}, "
            f"{N_TARGET} targets, {len(neighbor_x)} neighbors, lr={args.lr}",
            fontsize=14, fontweight="bold", y=0.98,
        )

    n_frames = len(snapshots)
    fps = 4
    anim = FuncAnimation(fig, draw, frames=n_frames,
                         interval=1000 // fps, repeat=True)
    mp4_path = output_dir / f"inner_sgd_step{step}.mp4"
    anim.save(str(mp4_path), writer="ffmpeg", fps=fps, dpi=150)
    plt.close(fig)
    print(f"\nAnimation saved: {mp4_path}")

    # Static final frame
    png_path = output_dir / f"inner_sgd_step{step}.png"
    fig2, (ax_l, ax_c) = plt.subplots(1, 2, figsize=(18, 8),
                                       gridspec_kw={"width_ratios": [1.3, 1]})
    final = snapshots[-1]
    poly = MplPolygon(bnd_km, closed=True, fill=True,
                      facecolor="lightyellow", edgecolor="black",
                      lw=2, alpha=0.9, zorder=2)
    ax_l.add_patch(poly)
    ax_l.scatter(nb_km_x, nb_km_y, c="red", marker="D", s=70,
                 edgecolors="darkred", linewidths=0.5, alpha=0.6,
                 label=f"Neighbors ({len(neighbor_x)})", zorder=3)
    sc_init = ax_l.scatter(init_km_x, init_km_y,
                            facecolors="none", edgecolors="gray",
                            marker="^", s=40, linewidths=0.6,
                            label="Initial layout", zorder=4)
    sc_init.set_linestyle("--")
    ax_l.scatter(*to_km(final[2], final[3]), c="forestgreen", marker="^",
                 s=50, edgecolors="darkgreen", linewidths=0.5,
                 label=f"Converged ({max_iter} iter)", zorder=5)
    for j in range(len(final[2])):
        dx = float(final[2][j] - init_x[j]) / 1000
        dy = float(final[3][j] - init_y[j]) / 1000
        if np.sqrt(dx**2 + dy**2) > 0.05:
            ax_l.annotate(
                "", xy=(init_km_x[j] + dx, init_km_y[j] + dy),
                xytext=(init_km_x[j], init_km_y[j]),
                arrowprops=dict(arrowstyle="->", color="forestgreen",
                                lw=0.7, alpha=0.5),
                zorder=4,
            )
    ax_l.set_xlim(x_lo, x_hi)
    ax_l.set_ylim(y_lo, y_hi)
    ax_l.set_aspect("equal")
    ax_l.set_xlabel("x (km)", fontsize=12)
    ax_l.set_ylabel("y (km)", fontsize=12)
    ax_l.legend(loc="lower left", fontsize=9)
    ax_l.set_title(f"Converged Layout — AEP = {final[1]:.2f} GWh",
                   fontsize=14, fontweight="bold")

    ax_c.plot(iters_all, aeps_all, "o-", color="forestgreen", ms=3, lw=2.5)
    ax_c.set_xlabel("SGD Iterations", fontsize=12)
    ax_c.set_ylabel("Conservative AEP (GWh)", fontsize=12)
    ax_c.set_title("Convergence", fontsize=13, fontweight="bold")
    ax_c.set_xlim(0, max_iter * 1.05)
    ax_c.set_ylim(min(aeps_all) - 3, max(aeps_all) + 3)
    ax_c.grid(True, alpha=0.3)
    ax_c.text(0.95, 0.05,
              f"$\\Delta$AEP = {final[1] - init_aep:+.2f} GWh",
              transform=ax_c.transAxes, fontsize=12,
              fontweight="bold", color="forestgreen",
              ha="right", va="bottom",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                        edgecolor="forestgreen", alpha=0.9))
    fig2.suptitle(
        f"Inner Optimization — Step {step}/{n_placed}, "
        f"{N_TARGET} targets, {len(neighbor_x)} neighbors",
        fontsize=14, fontweight="bold", y=0.98,
    )
    fig2.savefig(str(png_path), dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Static frame saved: {png_path}")


if __name__ == "__main__":
    main()

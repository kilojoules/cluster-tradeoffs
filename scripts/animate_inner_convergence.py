"""Animate inner SGD convergence: 500 decaying-LR vs 2000 constant-LR iterations.

Runs the conservative inner optimization (with all 30 placed neighbors)
under two schedules, capturing layout snapshots and AEP values at many
iteration counts. Produces a side-by-side animation.

Usage:
    pixi run python scripts/animate_inner_convergence.py
"""

import jax
jax.config.update("jax_enable_x64", True)

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

# ---- DEI turbine ----
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


# ---- DEI boundary ----
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


def sweep_iterations(objective_fn, init_x, init_y, boundary, min_spacing,
                     iter_counts, lr, constant_lr_iters, sim, ws, wd, weights,
                     neighbor_x, neighbor_y, label):
    """Run SGD at many iteration counts, return list of (n_iter, aep, x, y)."""
    snapshots = []
    for i, n_iter in enumerate(iter_counts):
        t0 = time.time()
        settings = SGDSettings(
            learning_rate=lr,
            max_iter=n_iter,
            tol=0.0,
            additional_constant_lr_iterations=min(constant_lr_iters, n_iter),
        )
        opt_x, opt_y = topfarm_sgd_solve(
            objective_fn, init_x, init_y, boundary, min_spacing,
            settings=settings,
        )
        aep = float(compute_aep(sim, opt_x, opt_y, ws, wd, weights,
                                neighbor_x=neighbor_x, neighbor_y=neighbor_y))
        elapsed = time.time() - t0
        snapshots.append((n_iter, aep, np.array(opt_x), np.array(opt_y)))
        if (i + 1) % 10 == 0 or n_iter == iter_counts[-1]:
            print(f"  [{label}] iter={n_iter:5d}  AEP={aep:.3f} GWh  ({elapsed:.1f}s)")
    return snapshots


def main():
    output_dir = Path("analysis/inner_convergence")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results for neighbor positions
    results_path = Path("analysis/dei_greedy_grid_30/results.json")
    with open(results_path) as f:
        results = json.load(f)
    neighbor_x = jnp.array(results["neighbor_x"])
    neighbor_y = jnp.array(results["neighbor_y"])
    print(f"Loaded {len(neighbor_x)} neighbor positions")

    # Setup
    turbine = create_dei_turbine()
    wd, ws, weights = load_wind_data()
    sim = WakeSimulation(turbine, BastankhahGaussianDeficit(k=0.04))
    init_x, init_y = generate_target_grid(boundary_np, N_TARGET, spacing=4 * D)
    min_spacing = MIN_SPACING_D * D

    def conservative_objective(x, y):
        return -compute_aep(sim, x, y, ws, wd, weights,
                            neighbor_x=neighbor_x, neighbor_y=neighbor_y)

    # Shared iteration sample points
    iter_counts = sorted(set(
        list(range(10, 100, 10)) +
        list(range(100, 500, 25)) +
        list(range(500, 2001, 50))
    ))

    # ---- Schedule A: 500 iter, decaying LR (additional_constant_lr_iterations=0) ----
    iter_counts_a = [n for n in iter_counts if n <= 500]
    print(f"\n--- Schedule A: 500 iter, decaying LR ({len(iter_counts_a)} samples) ---")
    snaps_a = sweep_iterations(
        conservative_objective, init_x, init_y, boundary, min_spacing,
        iter_counts_a, lr=50.0, constant_lr_iters=0,
        sim=sim, ws=ws, wd=wd, weights=weights,
        neighbor_x=neighbor_x, neighbor_y=neighbor_y,
        label="500/decay",
    )

    # ---- Schedule B: 2000 iter, constant LR ----
    print(f"\n--- Schedule B: 2000 iter, constant LR ({len(iter_counts)} samples) ---")
    snaps_b = sweep_iterations(
        conservative_objective, init_x, init_y, boundary, min_spacing,
        iter_counts, lr=50.0, constant_lr_iters=2000,
        sim=sim, ws=ws, wd=wd, weights=weights,
        neighbor_x=neighbor_x, neighbor_y=neighbor_y,
        label="2000/const",
    )

    final_a = snaps_a[-1]
    final_b = snaps_b[-1]
    print(f"\nSchedule A final (500 iter, decay LR):    AEP = {final_a[1]:.3f} GWh")
    print(f"Schedule B final (2000 iter, constant LR): AEP = {final_b[1]:.3f} GWh")
    print(f"Difference: {final_b[1] - final_a[1]:+.3f} GWh")

    # ---- Render animation ----
    def to_km(x, y):
        return np.asarray(x) / 1000.0, np.asarray(y) / 1000.0

    bnd_km = np.column_stack(to_km(boundary_np[:, 0], boundary_np[:, 1]))
    nb_km_x, nb_km_y = to_km(neighbor_x, neighbor_y)

    pad_km = 1.0
    all_pts_x = np.concatenate([bnd_km[:, 0], nb_km_x])
    all_pts_y = np.concatenate([bnd_km[:, 1], nb_km_y])
    x_lo, x_hi = all_pts_x.min() - pad_km, all_pts_x.max() + pad_km
    y_lo, y_hi = all_pts_y.min() - pad_km, all_pts_y.max() + pad_km

    iters_a = [s[0] for s in snaps_a]
    aeps_a = [s[1] for s in snaps_a]
    iters_b = [s[0] for s in snaps_b]
    aeps_b = [s[1] for s in snaps_b]

    # Unified y-axis for convergence
    all_aeps = aeps_a + aeps_b
    aep_lo = min(all_aeps) - 5
    aep_hi = max(all_aeps) + 5

    # Animation frames: one per sample in schedule B (the longer one)
    # For schedule A, freeze at final once we pass 500
    n_frames = len(snaps_b)

    fig, axes = plt.subplots(1, 3, figsize=(24, 8),
                              gridspec_kw={"width_ratios": [1, 1, 1]})
    ax_a, ax_b, ax_conv = axes

    def draw_layout(ax, snap, label):
        poly = MplPolygon(bnd_km, closed=True, fill=True,
                          facecolor="lightyellow", edgecolor="black",
                          lw=2, alpha=0.9, zorder=2)
        ax.add_patch(poly)
        ax.scatter(nb_km_x, nb_km_y, c="red", marker="D", s=60,
                   edgecolors="darkred", linewidths=0.5, alpha=0.6, zorder=3)
        ax.scatter(*to_km(snap[2], snap[3]), c="forestgreen", marker="^", s=45,
                   edgecolors="darkgreen", linewidths=0.5, zorder=5)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_aspect("equal")
        ax.set_xlabel("x (km)", fontsize=11)
        ax.set_ylabel("y (km)", fontsize=11)
        ax.set_title(f"{label}\nAEP = {snap[1]:.2f} GWh",
                     fontsize=13, fontweight="bold")

    def draw(frame):
        ax_a.clear()
        ax_b.clear()
        ax_conv.clear()

        cur_iter_b = snaps_b[frame][0]

        # Schedule A: find matching snapshot or freeze at final
        snap_a_idx = 0
        for i, s in enumerate(snaps_a):
            if s[0] <= cur_iter_b:
                snap_a_idx = i
        snap_a = snaps_a[snap_a_idx]
        snap_b = snaps_b[frame]

        draw_layout(ax_a, snap_a,
                    f"500 iter, decaying LR\niter {snap_a[0]}")
        draw_layout(ax_b, snap_b,
                    f"2000 iter, constant LR\niter {snap_b[0]}")

        # Convergence curves
        # Full curves in light color
        ax_conv.plot(iters_a, aeps_a, "-", color="lightskyblue", lw=1.5, zorder=1)
        ax_conv.plot(iters_b, aeps_b, "-", color="lightsalmon", lw=1.5, zorder=1)

        # Progress up to current iteration
        mask_a = [i for i, it in enumerate(iters_a) if it <= cur_iter_b]
        mask_b = [i for i, it in enumerate(iters_b) if it <= cur_iter_b]
        if mask_a:
            ax_conv.plot([iters_a[i] for i in mask_a],
                         [aeps_a[i] for i in mask_a],
                         "o-", color="royalblue", ms=3, lw=2.5, zorder=3,
                         label=f"500 / decay LR")
        if mask_b:
            ax_conv.plot([iters_b[i] for i in mask_b],
                         [aeps_b[i] for i in mask_b],
                         "s-", color="firebrick", ms=3, lw=2.5, zorder=3,
                         label=f"2000 / const LR")

        # Vertical line at current iteration
        ax_conv.axvline(cur_iter_b, color="gray", ls=":", lw=1, alpha=0.5)

        ax_conv.set_xlabel("SGD Iterations", fontsize=12)
        ax_conv.set_ylabel("Conservative AEP (GWh)", fontsize=12)
        ax_conv.set_title("Inner SGD Convergence", fontsize=13, fontweight="bold")
        ax_conv.set_xlim(0, 2100)
        ax_conv.set_ylim(aep_lo, aep_hi)
        ax_conv.legend(loc="lower right", fontsize=10)
        ax_conv.grid(True, alpha=0.3)

        fig.suptitle(
            f"Inner Optimization: 500 (decay) vs 2000 (constant) LR — "
            f"{N_TARGET} targets, {len(neighbor_x)} neighbors",
            fontsize=15, fontweight="bold", y=0.98,
        )

    fps = 4
    anim = FuncAnimation(fig, draw, frames=n_frames,
                         interval=1000 // fps, repeat=True)
    mp4_path = output_dir / "inner_convergence.mp4"
    anim.save(str(mp4_path), writer="ffmpeg", fps=fps, dpi=150)
    plt.close(fig)
    print(f"\nAnimation saved: {mp4_path}")

    # ---- Static comparison PNG ----
    fig2, axes2 = plt.subplots(1, 3, figsize=(24, 8),
                                gridspec_kw={"width_ratios": [1, 1, 1]})
    ax_as, ax_bs, ax_cs = axes2

    draw_layout(ax_as, final_a,
                f"500 iter, decaying LR\nAEP = {final_a[1]:.2f} GWh")
    draw_layout(ax_bs, final_b,
                f"2000 iter, constant LR\nAEP = {final_b[1]:.2f} GWh")

    ax_cs.plot(iters_a, aeps_a, "o-", color="royalblue", ms=3, lw=2.5,
               label=f"500 / decay: {final_a[1]:.2f} GWh")
    ax_cs.plot(iters_b, aeps_b, "s-", color="firebrick", ms=3, lw=2.5,
               label=f"2000 / const: {final_b[1]:.2f} GWh")

    delta = final_b[1] - final_a[1]
    ax_cs.annotate(
        f"$\\Delta$ = {delta:+.2f} GWh",
        xy=(2000, final_b[1]), xytext=(-80, -40),
        textcoords="offset points", fontsize=12, fontweight="bold",
        color="firebrick",
        arrowprops=dict(arrowstyle="->", color="firebrick"),
    )

    ax_cs.set_xlabel("SGD Iterations", fontsize=12)
    ax_cs.set_ylabel("Conservative AEP (GWh)", fontsize=12)
    ax_cs.set_title("Inner SGD Convergence", fontsize=13, fontweight="bold")
    ax_cs.set_xlim(0, 2100)
    ax_cs.set_ylim(aep_lo, aep_hi)
    ax_cs.legend(loc="lower right", fontsize=10)
    ax_cs.grid(True, alpha=0.3)
    fig2.suptitle(
        f"Inner Optimization: 500 (decay) vs 2000 (constant) LR — "
        f"{N_TARGET} targets, {len(neighbor_x)} neighbors",
        fontsize=15, fontweight="bold", y=0.98,
    )
    png_path = output_dir / "inner_convergence.png"
    fig2.savefig(str(png_path), dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Static comparison saved: {png_path}")

    # Save data
    data = {
        "schedule_a": {"label": "500 iter, decaying LR",
                       "iters": iters_a, "aeps": aeps_a,
                       "final_aep": final_a[1]},
        "schedule_b": {"label": "2000 iter, constant LR",
                       "iters": iters_b, "aeps": aeps_b,
                       "final_aep": final_b[1]},
        "delta_gwh": delta,
    }
    with open(output_dir / "convergence_data.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"Data saved: {output_dir / 'convergence_data.json'}")


if __name__ == "__main__":
    main()

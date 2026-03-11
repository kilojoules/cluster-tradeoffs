"""Regret landscape: sweep bearing × distance for a fixed neighbor cluster.

Produces polar heatmap and line plots showing how regret depends on
neighbor cluster placement relative to the target farm and wind direction.

Setup matches run_bilevel_ift.py:
- 16 target turbines, 16D×16D boundary, D=200m, 4D min spacing
- BastankhahGaussianDeficit(k=0.04), single wind 270°/9 m/s
- 4×4 grid initial target layout

Sweeps:
1. Bearing × Distance grid: 24 bearings × 18 distances = 432 configs
2. Cluster size sweep at peak-regret bearing/distance

Outputs → analysis/regret_landscape/
- polar_heatmap.png, distance_sweep.png, bearing_sweep.png, size_sweep.png
- results.json

Usage:
    pixi run python scripts/regret_landscape.py
"""

import jax

jax.config.update("jax_enable_x64", True)

import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pixwake import Curve, Turbine, WakeSimulation
from pixwake.deficit import BastankhahGaussianDeficit
from pixwake.optim.sgd import (
    SGDSettings,
    generate_random_starts,
    topfarm_sgd_solve,
    topfarm_sgd_solve_multistart,
)

# ── Configuration ─────────────────────────────────────────────────────────

D = 200.0
TARGET_SIZE = 16 * D
MIN_SPACING = 4 * D
N_TARGET = 16
WS = jnp.array([9.0])
WD = jnp.array([270.0])

# Sweep parameters
N_BEARINGS = 24
N_DISTANCES = 18
BEARING_STEP = 15.0  # degrees
DIST_MIN_D = 3.0
DIST_MAX_D = 20.0
DIST_STEP_D = 1.0

# Buffer: neighbors must be outside boundary + buffer
BUFFER_D = 0.0

# Default neighbor cluster
DEFAULT_N_ROWS = 3
DEFAULT_N_COLS = 3
DEFAULT_SPACING_D = 5.0

# Multistart
K_STARTS = 3
SEED = 42

OUTPUT_DIR = Path("analysis/regret_landscape")


# ── Turbine and simulation ────────────────────────────────────────────────


def create_turbine(rotor_diameter: float = 200.0) -> Turbine:
    """Create a 10 MW class turbine (matches run_bilevel_ift.py)."""
    ws = jnp.array([0.0, 4.0, 10.0, 15.0, 25.0])
    power = jnp.array([0.0, 0.0, 10000.0, 10000.0, 0.0])
    ct = jnp.array([0.0, 0.8, 0.8, 0.4, 0.0])
    return Turbine(
        rotor_diameter=rotor_diameter,
        hub_height=120.0,
        power_curve=Curve(ws=ws, values=power),
        ct_curve=Curve(ws=ws, values=ct),
    )


# ── Cluster placement ─────────────────────────────────────────────────────


def make_rectangular_cluster(
    bearing_deg: float,
    distance_D: float,
    n_rows: int,
    n_cols: int,
    spacing_D: float,
    D: float,
    center: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Place a rectangular cluster at (bearing, distance) from center.

    Args:
        bearing_deg: Compass bearing (0=N, 90=E, 270=W).
        distance_D: Distance from center in rotor diameters.
        n_rows: Rows along radial direction.
        n_cols: Columns perpendicular to radial direction.
        spacing_D: Spacing between turbines in rotor diameters.
        D: Rotor diameter in meters.
        center: Target farm center (x, y).

    Returns:
        (nb_x, nb_y) arrays of neighbor positions.
    """
    # Bearing to math angle (compass: 0=N CW, math: 0=E CCW)
    angle_rad = np.radians(90.0 - bearing_deg)

    # Cluster center
    dist_m = distance_D * D
    cx = center[0] + dist_m * np.cos(angle_rad)
    cy = center[1] + dist_m * np.sin(angle_rad)

    # Radial direction unit vector
    radial = np.array([np.cos(angle_rad), np.sin(angle_rad)])
    # Perpendicular (90° CCW from radial)
    perp = np.array([-radial[1], radial[0]])

    spacing_m = spacing_D * D
    positions = []
    for r in range(n_rows):
        for c in range(n_cols):
            # Center the grid on the cluster center
            r_offset = (r - (n_rows - 1) / 2.0) * spacing_m
            c_offset = (c - (n_cols - 1) / 2.0) * spacing_m
            pos = np.array([cx, cy]) + r_offset * radial + c_offset * perp
            positions.append(pos)

    positions = np.array(positions)
    return positions[:, 0], positions[:, 1]


# ── Regret computation ────────────────────────────────────────────────────


def compute_regret(
    nb_x: jnp.ndarray,
    nb_y: jnp.ndarray,
    liberal_x: jnp.ndarray,
    liberal_y: jnp.ndarray,
    sim: WakeSimulation,
    init_x: jnp.ndarray,
    init_y: jnp.ndarray,
    boundary: jnp.ndarray,
    min_spacing: float,
    sgd_settings: SGDSettings,
    k_starts: int = 3,
    rng_key: jnp.ndarray | None = None,
) -> dict:
    """Compute regret for given neighbor positions with multistart inner solve.

    Returns dict with regret, conservative_aep, liberal_aep_present.
    """
    n_target = len(init_x)

    def compute_aep(target_x, target_y, nb_x_=None, nb_y_=None):
        if nb_x_ is not None:
            x_all = jnp.concatenate([target_x, nb_x_])
            y_all = jnp.concatenate([target_y, nb_y_])
        else:
            x_all = target_x
            y_all = target_y
        result = sim(x_all, y_all, ws_amb=WS, wd_amb=WD)
        power = result.power()[:, :n_target]
        return jnp.sum(power) * 8760.0 / 1e6  # GWh/yr

    def obj_fn(x, y):
        return -compute_aep(x, y, nb_x, nb_y)

    # Multistart inner solve — always include the liberal layout as a candidate
    # so that conservative_aep >= liberal_aep_present (regret >= 0 by construction)
    if k_starts > 1 and rng_key is not None:
        rand_x, rand_y = generate_random_starts(
            rng_key, k_starts - 1, n_target, boundary, min_spacing
        )
        init_x_batch = jnp.concatenate([liberal_x[None, :], init_x[None, :], rand_x], axis=0)
        init_y_batch = jnp.concatenate([liberal_y[None, :], init_y[None, :], rand_y], axis=0)
        all_x, all_y, all_objs = topfarm_sgd_solve_multistart(
            obj_fn, init_x_batch, init_y_batch, boundary, min_spacing, sgd_settings
        )
        best_idx = jnp.argmin(all_objs)
        opt_x = all_x[best_idx]
        opt_y = all_y[best_idx]
    else:
        # Single start from liberal layout
        opt_x, opt_y = topfarm_sgd_solve(
            obj_fn, liberal_x, liberal_y, boundary, min_spacing, sgd_settings
        )

    conservative_aep = float(compute_aep(opt_x, opt_y, nb_x, nb_y))
    liberal_aep_present = float(compute_aep(liberal_x, liberal_y, nb_x, nb_y))
    regret = conservative_aep - liberal_aep_present

    return {
        "regret": regret,
        "conservative_aep": conservative_aep,
        "liberal_aep_present": liberal_aep_present,
    }


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Setup
    turbine = create_turbine(D)
    deficit = BastankhahGaussianDeficit(k=0.04)
    sim = WakeSimulation(turbine, deficit)

    target_boundary = jnp.array([
        [0.0, 0.0],
        [TARGET_SIZE, 0.0],
        [TARGET_SIZE, TARGET_SIZE],
        [0.0, TARGET_SIZE],
    ])
    target_center = np.array([TARGET_SIZE / 2, TARGET_SIZE / 2])

    # 4×4 grid initial layout
    grid_side = int(np.sqrt(N_TARGET))
    grid_spacing = TARGET_SIZE / (grid_side + 1)
    init_x = jnp.array([grid_spacing * (i + 1) for i in range(grid_side)] * grid_side)
    init_y = jnp.array(
        [grid_spacing * (j + 1) for j in range(grid_side) for _ in range(grid_side)]
    )

    sgd_settings = SGDSettings(
        learning_rate=D / 5,
        max_iter=3000,
        tol=1e-8,
    )

    # Compute liberal layout ONCE (no neighbors)
    print("Computing liberal layout (no neighbors)...", flush=True)
    t0 = time.time()

    def liberal_obj(x, y):
        x_all = x
        y_all = y
        result = sim(x_all, y_all, ws_amb=WS, wd_amb=WD)
        power = result.power()[:, :N_TARGET]
        return -jnp.sum(power) * 8760.0 / 1e6

    liberal_x, liberal_y = topfarm_sgd_solve(
        liberal_obj, init_x, init_y, target_boundary, MIN_SPACING, sgd_settings
    )
    liberal_aep_result = sim(liberal_x, liberal_y, ws_amb=WS, wd_amb=WD)
    liberal_aep = float(jnp.sum(liberal_aep_result.power()[:, :N_TARGET]) * 8760.0 / 1e6)
    print(f"  Liberal AEP: {liberal_aep:.2f} GWh ({time.time() - t0:.1f}s)", flush=True)

    # ── Sweep 1: Bearing × Distance ──────────────────────────────────────

    bearings = np.arange(0, 360, BEARING_STEP)
    distances = np.arange(DIST_MIN_D, DIST_MAX_D + 0.5, DIST_STEP_D)
    n_total = len(bearings) * len(distances)

    print(f"\nSweep 1: {len(bearings)} bearings x {len(distances)} distances = {n_total} configs")
    print(f"  Cluster: {DEFAULT_N_ROWS}x{DEFAULT_N_COLS}, {DEFAULT_SPACING_D:.0f}D spacing")
    print(f"  Inner starts: K={K_STARTS}", flush=True)

    regret_grid = np.full((len(bearings), len(distances)), np.nan)
    conservative_aep_grid = np.full_like(regret_grid, np.nan)
    liberal_aep_present_grid = np.full_like(regret_grid, np.nan)

    rng_key = jax.random.PRNGKey(SEED)
    t_sweep_start = time.time()
    count = 0

    # Boundary exclusion: skip configs where any neighbor is inside boundary + buffer
    bnd_x_min = float(target_boundary[:, 0].min())
    bnd_x_max = float(target_boundary[:, 0].max())
    bnd_y_min = float(target_boundary[:, 1].min())
    bnd_y_max = float(target_boundary[:, 1].max())
    buffer = BUFFER_D * D
    skipped = 0

    for bi, bearing in enumerate(bearings):
        for di, dist_D in enumerate(distances):
            nb_x_np, nb_y_np = make_rectangular_cluster(
                bearing, dist_D,
                DEFAULT_N_ROWS, DEFAULT_N_COLS, DEFAULT_SPACING_D,
                D, target_center,
            )

            # Check: all neighbors must be outside boundary + buffer
            inside = (
                (nb_x_np > bnd_x_min - buffer) & (nb_x_np < bnd_x_max + buffer) &
                (nb_y_np > bnd_y_min - buffer) & (nb_y_np < bnd_y_max + buffer)
            )
            if inside.any():
                skipped += 1
                count += 1
                continue  # leave as NaN

            nb_x = jnp.array(nb_x_np)
            nb_y = jnp.array(nb_y_np)

            rng_key, subkey = jax.random.split(rng_key)
            result = compute_regret(
                nb_x, nb_y, liberal_x, liberal_y,
                sim, init_x, init_y, target_boundary, MIN_SPACING,
                sgd_settings, k_starts=K_STARTS, rng_key=subkey,
            )

            regret_grid[bi, di] = result["regret"]
            conservative_aep_grid[bi, di] = result["conservative_aep"]
            liberal_aep_present_grid[bi, di] = result["liberal_aep_present"]
            count += 1

            if count % 24 == 0 or count == n_total:
                elapsed = time.time() - t_sweep_start
                rate = count / elapsed if elapsed > 0 else 1
                eta = (n_total - count) / rate if rate > 0 else 0
                regret_str = f"{result['regret']:.4f}" if not inside.any() else "SKIP"
                print(
                    f"  [{count:4d}/{n_total}] bearing={bearing:.0f}° dist={dist_D:.0f}D "
                    f"regret={regret_str} GWh  "
                    f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)",
                    flush=True,
                )

    sweep1_elapsed = time.time() - t_sweep_start
    print(f"  Sweep 1 done in {sweep1_elapsed:.0f}s ({skipped} configs skipped — neighbors inside boundary)", flush=True)

    # Find peak regret location
    peak_idx = np.unravel_index(np.nanargmax(regret_grid), regret_grid.shape)
    peak_bearing = bearings[peak_idx[0]]
    peak_dist = distances[peak_idx[1]]
    peak_regret = regret_grid[peak_idx]
    print(f"\n  Peak regret: {peak_regret:.4f} GWh at bearing={peak_bearing:.0f}°, distance={peak_dist:.0f}D")

    # ── Sweep 2: Cluster size at peak location ───────────────────────────

    cluster_sizes = [(1, 1), (1, 2), (2, 2), (2, 3), (3, 3), (3, 4), (4, 4), (4, 5), (5, 5)]
    n_turbines_list = [r * c for r, c in cluster_sizes]
    size_regrets = []

    print(f"\nSweep 2: Cluster size at bearing={peak_bearing:.0f}°, distance={peak_dist:.0f}D")
    t_size_start = time.time()

    for rows, cols in cluster_sizes:
        nb_x_np, nb_y_np = make_rectangular_cluster(
            peak_bearing, peak_dist,
            rows, cols, DEFAULT_SPACING_D, D, target_center,
        )

        # Check boundary exclusion
        inside = (
            (nb_x_np > bnd_x_min - buffer) & (nb_x_np < bnd_x_max + buffer) &
            (nb_y_np > bnd_y_min - buffer) & (nb_y_np < bnd_y_max + buffer)
        )
        if inside.any():
            size_regrets.append(float("nan"))
            print(
                f"  {rows}x{cols} ({rows*cols:2d} turbines): SKIPPED (neighbors inside boundary)",
                flush=True,
            )
            continue

        nb_x = jnp.array(nb_x_np)
        nb_y = jnp.array(nb_y_np)

        rng_key, subkey = jax.random.split(rng_key)
        result = compute_regret(
            nb_x, nb_y, liberal_x, liberal_y,
            sim, init_x, init_y, target_boundary, MIN_SPACING,
            sgd_settings, k_starts=K_STARTS, rng_key=subkey,
        )
        size_regrets.append(result["regret"])
        print(
            f"  {rows}x{cols} ({rows*cols:2d} turbines): regret={result['regret']:.4f} GWh",
            flush=True,
        )

    size_elapsed = time.time() - t_size_start
    print(f"  Sweep 2 done in {size_elapsed:.0f}s", flush=True)

    # ── Save results ─────────────────────────────────────────────────────

    results = {
        "config": {
            "D": D,
            "target_size": TARGET_SIZE,
            "min_spacing": MIN_SPACING,
            "n_target": N_TARGET,
            "ws": [float(x) for x in WS],
            "wd": [float(x) for x in WD],
            "k_starts": K_STARTS,
            "default_cluster": {
                "n_rows": DEFAULT_N_ROWS,
                "n_cols": DEFAULT_N_COLS,
                "spacing_D": DEFAULT_SPACING_D,
            },
        },
        "liberal_aep": liberal_aep,
        "liberal_x": [float(x) for x in liberal_x],
        "liberal_y": [float(x) for x in liberal_y],
        "sweep1": {
            "bearings_deg": bearings.tolist(),
            "distances_D": distances.tolist(),
            "regret_grid": regret_grid.tolist(),
            "conservative_aep_grid": conservative_aep_grid.tolist(),
            "liberal_aep_present_grid": liberal_aep_present_grid.tolist(),
            "peak_bearing_deg": float(peak_bearing),
            "peak_distance_D": float(peak_dist),
            "peak_regret": float(peak_regret),
            "elapsed_seconds": sweep1_elapsed,
        },
        "sweep2": {
            "cluster_sizes": [[r, c] for r, c in cluster_sizes],
            "n_turbines": n_turbines_list,
            "regrets": size_regrets,
            "bearing_deg": float(peak_bearing),
            "distance_D": float(peak_dist),
            "elapsed_seconds": size_elapsed,
        },
    }

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR / 'results.json'}")

    # ── Plot 1: Polar heatmap ────────────────────────────────────────────

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(9, 9))

    # Convert bearing (compass, CW from N) to polar angle (CCW from E)
    theta = np.radians(90.0 - bearings)
    # meshgrid: theta x distance
    theta_grid, dist_grid = np.meshgrid(theta, distances, indexing="ij")

    pcm = ax.pcolormesh(
        theta_grid, dist_grid, regret_grid,
        cmap="hot_r", shading="auto",
    )
    cbar = fig.colorbar(pcm, ax=ax, shrink=0.8, pad=0.08)
    cbar.set_label("Regret (GWh)")

    # Mark peak
    peak_theta = np.radians(90.0 - peak_bearing)
    ax.plot(peak_theta, peak_dist, "w*", markersize=18, markeredgecolor="black", zorder=10)

    # Wind direction arrow (270° = from west = blowing east)
    wind_theta = np.radians(90.0 - 270.0)  # 270° bearing → polar angle
    ax.annotate(
        "", xy=(wind_theta, distances[-1] * 0.4),
        xytext=(wind_theta, distances[-1] * 0.9),
        arrowprops=dict(arrowstyle="->", color="blue", lw=2.5),
    )
    ax.text(
        wind_theta, distances[-1] * 1.0, "Wind 270°",
        ha="center", va="center", fontsize=9, color="blue", fontweight="bold",
    )

    # Compass labels
    ax.set_theta_zero_location("E")  # 0° polar = East
    ax.set_theta_direction(1)  # CCW
    # Override tick labels to show compass bearings
    compass_angles = [0, 45, 90, 135, 180, 225, 270, 315]
    compass_labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    polar_angles = [np.radians(90 - b) for b in compass_angles]
    ax.set_xticks(polar_angles)
    ax.set_xticklabels(compass_labels)

    ax.set_ylabel("Distance (D)", labelpad=30)
    ax.set_title(
        f"Regret vs Neighbor Placement\n"
        f"{DEFAULT_N_ROWS}x{DEFAULT_N_COLS} cluster, {DEFAULT_SPACING_D:.0f}D spacing, "
        f"wind 270° @ 9 m/s\n"
        f"Peak: {peak_regret:.4f} GWh at {peak_bearing:.0f}°, {peak_dist:.0f}D",
        pad=20,
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "polar_heatmap.png", dpi=150, bbox_inches="tight")
    print(f"Saved {OUTPUT_DIR / 'polar_heatmap.png'}")
    plt.close()

    # ── Plot 2: Distance sweep at upwind bearing ─────────────────────────

    fig, ax = plt.subplots(figsize=(8, 5))
    upwind_bi = np.argmin(np.abs(bearings - peak_bearing))
    ax.plot(distances, regret_grid[upwind_bi, :], "o-", linewidth=2, markersize=5, color="darkred")
    ax.set_xlabel("Distance (D)")
    ax.set_ylabel("Regret (GWh)")
    ax.set_title(f"Regret vs Distance at bearing={peak_bearing:.0f}° (upwind)")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "distance_sweep.png", dpi=150, bbox_inches="tight")
    print(f"Saved {OUTPUT_DIR / 'distance_sweep.png'}")
    plt.close()

    # ── Plot 3: Bearing sweep at optimal distance ────────────────────────

    fig, ax = plt.subplots(figsize=(8, 5))
    peak_di = np.argmin(np.abs(distances - peak_dist))
    ax.plot(bearings, regret_grid[:, peak_di], "o-", linewidth=2, markersize=5, color="navy")
    ax.set_xlabel("Bearing (°)")
    ax.set_ylabel("Regret (GWh)")
    ax.set_title(f"Regret vs Bearing at distance={peak_dist:.0f}D")
    ax.set_xticks(np.arange(0, 361, 45))
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    # Mark wind direction
    ax.axvline(270, color="blue", linestyle=":", alpha=0.7, label="Wind dir (270°)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "bearing_sweep.png", dpi=150, bbox_inches="tight")
    print(f"Saved {OUTPUT_DIR / 'bearing_sweep.png'}")
    plt.close()

    # ── Plot 4: Size sweep ───────────────────────────────────────────────

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(n_turbines_list, size_regrets, "s-", linewidth=2, markersize=7, color="forestgreen")
    for i, (r, c) in enumerate(cluster_sizes):
        ax.annotate(
            f"{r}x{c}", (n_turbines_list[i], size_regrets[i]),
            textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8,
        )
    ax.set_xlabel("Number of Neighbor Turbines")
    ax.set_ylabel("Regret (GWh)")
    ax.set_title(
        f"Regret vs Cluster Size at bearing={peak_bearing:.0f}°, distance={peak_dist:.0f}D"
    )
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "size_sweep.png", dpi=150, bbox_inches="tight")
    print(f"Saved {OUTPUT_DIR / 'size_sweep.png'}")
    plt.close()

    # ── Summary ──────────────────────────────────────────────────────────

    total_elapsed = sweep1_elapsed + size_elapsed
    print(f"\nTotal time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"Peak regret: {peak_regret:.4f} GWh at bearing={peak_bearing:.0f}°, distance={peak_dist:.0f}D")
    print(f"Liberal AEP: {liberal_aep:.2f} GWh")
    if liberal_aep > 0:
        print(f"Peak regret as % of liberal AEP: {peak_regret / liberal_aep * 100:.2f}%")


if __name__ == "__main__":
    main()

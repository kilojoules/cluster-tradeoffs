"""Multistart bilevel IFT adversarial search with K inner × M outer parallelism.

Demonstrates the GPU-parallel bilevel optimization pipeline:
- K inner multistarts: find robust inner optima via vmap over initial layouts
- M outer multistarts: explore neighbor configuration space from multiple seeds

Setup matches run_bilevel_ift.py:
- 16 target turbines in 16D x 16D area, D=200m, 4D min spacing
- BastankhahGaussianDeficit(k=0.04), single wind direction 270°, 9 m/s
- 4 initial neighbors placed upwind

Usage:
    pixi run python scripts/run_bilevel_multistart.py
"""

import json
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from pixwake import Curve, Turbine, WakeSimulation
from pixwake.deficit import BastankhahGaussianDeficit
from pixwake.optim.adversarial import (
    AdversarialSearchSettings,
    GradientAdversarialSearch,
)
from pixwake.optim.sgd import SGDSettings, generate_random_starts


def create_turbine(rotor_diameter: float = 200.0) -> Turbine:
    """Create a 10 MW class turbine (matches run_regret_discovery.py)."""
    ws = jnp.array([0.0, 4.0, 10.0, 15.0, 25.0])
    power = jnp.array([0.0, 0.0, 10000.0, 10000.0, 0.0])  # 10 MW
    ct = jnp.array([0.0, 0.8, 0.8, 0.4, 0.0])
    return Turbine(
        rotor_diameter=rotor_diameter,
        hub_height=120.0,
        power_curve=Curve(ws=ws, values=power),
        ct_curve=Curve(ws=ws, values=ct),
    )


def main():
    output_dir = Path("analysis/bilevel_multistart")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configuration
    D = 200.0
    target_size = 16 * D
    min_spacing = 4 * D
    ws = jnp.array([9.0])
    wd = jnp.array([270.0])
    n_neighbors = 4

    # Multistart settings
    inner_k = 5   # K inner starts for layout optimization
    outer_m = 5   # M outer starts for neighbor search

    # Create turbine and simulation
    turbine = create_turbine(D)
    deficit = BastankhahGaussianDeficit(k=0.04)
    sim = WakeSimulation(turbine, deficit)

    # Target farm boundary
    target_boundary = jnp.array([
        [0.0, 0.0],
        [target_size, 0.0],
        [target_size, target_size],
        [0.0, target_size],
    ])

    # Initial target positions (4x4 grid)
    n_target = 16
    grid_side = int(np.sqrt(n_target))
    grid_spacing = target_size / (grid_side + 1)
    init_target_x = jnp.array(
        [grid_spacing * (i + 1) for i in range(grid_side)] * grid_side
    )
    init_target_y = jnp.array(
        [grid_spacing * (j + 1) for j in range(grid_side) for _ in range(grid_side)]
    )

    # Generate K inner random starts for target layout
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    inner_rand_x, inner_rand_y = generate_random_starts(
        subkey, inner_k - 1, n_target, target_boundary, min_spacing
    )
    # Include deterministic grid start as first
    inner_init_x_batch = jnp.concatenate(
        [init_target_x[None, :], inner_rand_x], axis=0
    )
    inner_init_y_batch = jnp.concatenate(
        [init_target_y[None, :], inner_rand_y], axis=0
    )

    # Generate M outer neighbor starting configs
    base_nb_x = jnp.array([-3 * D, -4 * D, -5 * D, -6 * D])
    base_nb_y = jnp.array([
        target_size * 0.2, target_size * 0.4,
        target_size * 0.6, target_size * 0.8,
    ])

    nb_x_starts = [base_nb_x]
    nb_y_starts = [base_nb_y]
    for m in range(1, outer_m):
        key, k1, k2 = jax.random.split(key, 3)
        # Perturb base positions
        dx = jax.random.uniform(k1, (n_neighbors,), minval=-2 * D, maxval=2 * D)
        dy = jax.random.uniform(k2, (n_neighbors,), minval=-2 * D, maxval=2 * D)
        nb_x_starts.append(base_nb_x + dx)
        nb_y_starts.append(base_nb_y + dy)

    init_nb_x_batch = jnp.stack(nb_x_starts)
    init_nb_y_batch = jnp.stack(nb_y_starts)

    # Neighbor boundary for clipping
    neighbor_bound = jnp.array([
        [-10 * D, -5 * D],
        [target_size + 5 * D, -5 * D],
        [target_size + 5 * D, target_size + 5 * D],
        [-10 * D, target_size + 5 * D],
    ])

    # Settings
    sgd_settings = SGDSettings(
        learning_rate=D / 5,
        max_iter=3000,
        tol=1e-8,
    )
    search_settings = AdversarialSearchSettings(
        max_iter=50,
        learning_rate=50.0,
        tol=1e-5,
        neighbor_boundary=neighbor_bound,
        sgd_settings=sgd_settings,
        verbose=True,
    )

    print("=" * 60)
    print("Multistart Bilevel IFT Adversarial Search")
    print("=" * 60)
    print(f"Target: {n_target} turbines in {target_size/D:.0f}D x {target_size/D:.0f}D")
    print(f"Min spacing: {min_spacing/D:.0f}D = {min_spacing:.0f}m")
    print(f"Neighbors: {n_neighbors}")
    print(f"Inner multistarts (K): {inner_k}")
    print(f"Outer multistarts (M): {outer_m}")
    print(f"Wind: {float(ws[0]):.0f} m/s @ {float(wd[0]):.0f} deg")
    print(f"Outer iters per start: {search_settings.max_iter}")
    print("=" * 60, flush=True)

    # Run multistart search
    searcher = GradientAdversarialSearch(
        sim=sim,
        target_boundary=target_boundary,
        target_min_spacing=min_spacing,
        ws_amb=ws,
        wd_amb=wd,
    )

    start_time = time.time()
    result = searcher.search_multistart(
        init_target_x, init_target_y,
        init_nb_x_batch, init_nb_y_batch,
        settings=search_settings,
        inner_k=inner_k,
        inner_init_x_batch=inner_init_x_batch,
        inner_init_y_batch=inner_init_y_batch,
    )
    elapsed_multi = time.time() - start_time
    print(f"\nMultistart elapsed: {elapsed_multi:.1f}s")

    # Run single-start baseline for comparison
    print("\n--- Single-start baseline ---")
    single_settings = AdversarialSearchSettings(
        max_iter=50,
        learning_rate=50.0,
        tol=1e-5,
        neighbor_boundary=neighbor_bound,
        sgd_settings=sgd_settings,
        verbose=True,
    )
    start_time = time.time()
    result_single = searcher.search(
        init_target_x, init_target_y,
        base_nb_x, base_nb_y,
        settings=single_settings,
    )
    elapsed_single = time.time() - start_time
    print(f"Single-start elapsed: {elapsed_single:.1f}s")

    # Save results
    results_data = {
        "method": "ift_bilevel_multistart",
        "inner_k": inner_k,
        "outer_m": outer_m,
        "n_target": n_target,
        "n_neighbors": n_neighbors,
        "D": D,
        "multistart": {
            "regret": float(result.regret),
            "liberal_aep": float(result.liberal_aep),
            "conservative_aep": float(result.conservative_aep),
            "neighbor_x": [float(x) for x in result.neighbor_x],
            "neighbor_y": [float(y) for y in result.neighbor_y],
            "elapsed_seconds": elapsed_multi,
        },
        "single_start": {
            "regret": float(result_single.regret),
            "liberal_aep": float(result_single.liberal_aep),
            "conservative_aep": float(result_single.conservative_aep),
            "neighbor_x": [float(x) for x in result_single.neighbor_x],
            "neighbor_y": [float(y) for y in result_single.neighbor_y],
            "elapsed_seconds": elapsed_single,
        },
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved to {output_dir / 'results.json'}")

    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Regret comparison bar chart
    ax = axes[0]
    labels = ["Single-start", f"Multistart\n(K={inner_k}, M={outer_m})"]
    regrets = [result_single.regret, result.regret]
    colors = ["steelblue", "darkorange"]
    ax.bar(labels, regrets, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Regret (GWh)")
    ax.set_title("Regret Comparison")
    ax.grid(True, alpha=0.3, axis="y")

    # Timing comparison
    ax = axes[1]
    times = [elapsed_single, elapsed_multi]
    ax.bar(labels, times, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title("Timing Comparison")
    ax.grid(True, alpha=0.3, axis="y")

    # Layout comparison
    ax = axes[2]
    bnd = np.array(target_boundary)
    bnd = np.vstack([bnd, bnd[0]])
    ax.plot(bnd[:, 0], bnd[:, 1], "k-", linewidth=2, label="Boundary")

    ax.scatter(result.liberal_x, result.liberal_y, c="blue", marker="^", s=60,
               label="Liberal layout", zorder=5)
    ax.scatter(result.target_x, result.target_y, c="green", marker="s", s=60,
               label="Conservative (multi)", zorder=5)
    ax.scatter(result_single.target_x, result_single.target_y, c="lime",
               marker="o", s=40, alpha=0.5, label="Conservative (single)")

    ax.scatter(result.neighbor_x, result.neighbor_y, c="red", marker="D", s=80,
               label="Neighbors (multi)", zorder=5)
    ax.scatter(result_single.neighbor_x, result_single.neighbor_y, c="orange",
               marker="x", s=80, linewidths=2, label="Neighbors (single)")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Layouts — Multi regret={result.regret:.2f}, Single={result_single.regret:.2f}")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "multistart_comparison.png", dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_dir / 'multistart_comparison.png'}")
    plt.close()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Single-start regret:  {result_single.regret:.4f} GWh ({elapsed_single:.1f}s)")
    print(f"Multistart regret:    {result.regret:.4f} GWh ({elapsed_multi:.1f}s)")
    improvement = result.regret - result_single.regret
    print(f"Improvement:          {improvement:+.4f} GWh")
    print(f"Speedup factor:       {elapsed_single / elapsed_multi:.2f}x" if elapsed_multi > 0 else "")
    print("=" * 60)


if __name__ == "__main__":
    main()

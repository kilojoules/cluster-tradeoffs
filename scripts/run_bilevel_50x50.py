"""IFT bilevel adversarial search: 50 targets × 50 neighbors.

Uses the pure-AD IFT backward pass with K=10 inner multistarts
and 1,000 inner SGD steps per start.

Usage:
    pixi run python scripts/run_bilevel_50x50.py
"""

import json
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pixwake import Curve, Turbine, WakeSimulation
from pixwake.deficit import BastankhahGaussianDeficit
from pixwake.optim.adversarial import (
    AdversarialSearchSettings,
    GradientAdversarialSearch,
)
from pixwake.optim.sgd import SGDSettings, generate_random_starts


def create_turbine(rotor_diameter: float = 200.0) -> Turbine:
    """Create a 10 MW class turbine."""
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
    output_dir = Path("analysis/bilevel_50x50")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configuration
    D = 200.0
    n_target = 50
    n_neighbors = 50
    inner_k = 10  # inner multistarts
    inner_max_iter = 1000
    outer_max_iter = 50

    # Farm sizing: ~8x7 grid at 4D spacing → 32D x 32D
    target_size = 32 * D
    min_spacing = 4 * D
    ws = jnp.array([9.0])
    wd = jnp.array([270.0])

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

    # Initial target positions: Poisson-disc-like random placement
    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)
    # Use generate_random_starts for the deterministic first start
    from matplotlib.path import Path as MplPath

    poly = MplPath(np.array(target_boundary))
    rng = np.random.default_rng(42)
    margin = min_spacing / 2
    pts = []
    while len(pts) < n_target:
        cands = rng.uniform(
            [margin, margin],
            [float(target_size) - margin, float(target_size) - margin],
            size=(n_target * 10, 2),
        )
        inside = poly.contains_points(cands)
        pts.extend(cands[inside].tolist())
    pts = np.array(pts[:n_target])
    init_target_x = jnp.array(pts[:, 0])
    init_target_y = jnp.array(pts[:, 1])

    # Generate K inner random starts
    key, subkey = jax.random.split(key)
    inner_rand_x, inner_rand_y = generate_random_starts(
        subkey, inner_k - 1, n_target, target_boundary, min_spacing
    )
    inner_init_x_batch = jnp.concatenate(
        [init_target_x[None, :], inner_rand_x], axis=0
    )
    inner_init_y_batch = jnp.concatenate(
        [init_target_y[None, :], inner_rand_y], axis=0
    )

    # 50 neighbors: spread around the target farm boundary
    # West side (upwind): 20 neighbors
    # North side: 10 neighbors
    # South side: 10 neighbors
    # East side: 10 neighbors
    rng_nb = np.random.default_rng(123)
    nb_x_list, nb_y_list = [], []
    # West (upwind) — most impactful
    for _ in range(20):
        nb_x_list.append(rng_nb.uniform(-8 * D, -3 * D))
        nb_y_list.append(rng_nb.uniform(0, float(target_size)))
    # North
    for _ in range(10):
        nb_x_list.append(rng_nb.uniform(0, float(target_size)))
        nb_y_list.append(rng_nb.uniform(float(target_size) + 3 * D, float(target_size) + 8 * D))
    # South
    for _ in range(10):
        nb_x_list.append(rng_nb.uniform(0, float(target_size)))
        nb_y_list.append(rng_nb.uniform(-8 * D, -3 * D))
    # East
    for _ in range(10):
        nb_x_list.append(rng_nb.uniform(float(target_size) + 3 * D, float(target_size) + 8 * D))
        nb_y_list.append(rng_nb.uniform(0, float(target_size)))

    init_neighbor_x = jnp.array(nb_x_list)
    init_neighbor_y = jnp.array(nb_y_list)

    # Neighbor boundary for clipping
    neighbor_bound = jnp.array([
        [-12 * D, -12 * D],
        [target_size + 12 * D, -12 * D],
        [target_size + 12 * D, target_size + 12 * D],
        [-12 * D, target_size + 12 * D],
    ])

    # Settings
    sgd_settings = SGDSettings(
        learning_rate=D / 5,
        max_iter=inner_max_iter,
        tol=1e-6,
    )
    search_settings = AdversarialSearchSettings(
        max_iter=outer_max_iter,
        learning_rate=50.0,
        tol=1e-5,
        neighbor_boundary=neighbor_bound,
        target_buffer=min_spacing,
        sgd_settings=sgd_settings,
        verbose=True,
    )

    print("=" * 60)
    print("IFT Bilevel Search — 50 Targets × 50 Neighbors")
    print("=" * 60)
    print(f"Target farm: {n_target} turbines in {target_size/D:.0f}D × {target_size/D:.0f}D")
    print(f"Min spacing: {min_spacing/D:.0f}D = {min_spacing:.0f}m")
    print(f"Neighbors: {n_neighbors}")
    print(f"Inner multistarts (K): {inner_k}")
    print(f"Inner SGD steps: {inner_max_iter}")
    print(f"Outer iterations: {outer_max_iter}")
    print(f"Wind: {float(ws[0]):.0f} m/s @ {float(wd[0]):.0f} deg")
    print(f"Total turbines per wake sim call: {n_target + n_neighbors}")
    print("=" * 60, flush=True)

    # Run search
    searcher = GradientAdversarialSearch(
        sim=sim,
        target_boundary=target_boundary,
        target_min_spacing=min_spacing,
        ws_amb=ws,
        wd_amb=wd,
    )

    # Use search_multistart with M=1 outer start, K=10 inner starts
    init_nb_x_batch = init_neighbor_x[None, :]  # (1, 50)
    init_nb_y_batch = init_neighbor_y[None, :]  # (1, 50)

    start_time = time.time()
    result = searcher.search_multistart(
        init_target_x,
        init_target_y,
        init_nb_x_batch,
        init_nb_y_batch,
        settings=search_settings,
        inner_k=inner_k,
        inner_init_x_batch=inner_init_x_batch,
        inner_init_y_batch=inner_init_y_batch,
    )
    elapsed = time.time() - start_time

    print(f"\nTotal elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Per outer iteration: {elapsed/outer_max_iter:.1f}s")

    # Save results
    results_data = {
        "method": "ift_bilevel_50x50_pure_ad",
        "n_target": n_target,
        "n_neighbors": n_neighbors,
        "inner_k": inner_k,
        "inner_max_iter": inner_max_iter,
        "outer_max_iter": outer_max_iter,
        "D": D,
        "target_size": float(target_size),
        "min_spacing": float(min_spacing),
        "liberal_aep": float(result.liberal_aep),
        "conservative_aep": float(result.conservative_aep),
        "regret": float(result.regret),
        "regret_pct": float(result.regret / result.liberal_aep * 100)
        if result.liberal_aep > 0
        else 0.0,
        "neighbor_x": [float(x) for x in result.neighbor_x],
        "neighbor_y": [float(y) for y in result.neighbor_y],
        "target_x": [float(x) for x in result.target_x],
        "target_y": [float(y) for y in result.target_y],
        "liberal_x": [float(x) for x in result.liberal_x],
        "liberal_y": [float(y) for y in result.liberal_y],
        "elapsed_seconds": elapsed,
        "per_outer_step_seconds": elapsed / max(len(result.history), 1),
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"Results saved to {output_dir / 'results.json'}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Convergence
    ax = axes[0]
    regrets = [h[0] for h in result.history]
    ax.plot(regrets, "o-", linewidth=2, markersize=3)
    ax.set_xlabel("Outer Iteration")
    ax.set_ylabel("Regret (GWh)")
    ax.set_title("Convergence — 50×50 Pure AD")
    ax.grid(True, alpha=0.3)

    # Layout
    ax = axes[1]
    bnd = np.array(target_boundary)
    bnd = np.vstack([bnd, bnd[0]])
    ax.plot(bnd[:, 0] / D, bnd[:, 1] / D, "k-", linewidth=2, label="Target boundary")

    ax.scatter(
        np.array(result.liberal_x) / D,
        np.array(result.liberal_y) / D,
        c="blue", marker="^", s=30, alpha=0.6, label="Liberal layout",
    )
    ax.scatter(
        np.array(result.target_x) / D,
        np.array(result.target_y) / D,
        c="green", marker="s", s=30, alpha=0.6, label="Conservative layout",
    )
    ax.scatter(
        np.array(init_neighbor_x) / D,
        np.array(init_neighbor_y) / D,
        c="orange", marker="x", s=40, linewidths=1, alpha=0.4, label="Initial neighbors",
    )
    ax.scatter(
        np.array(result.neighbor_x) / D,
        np.array(result.neighbor_y) / D,
        c="red", marker="D", s=40, label="Optimized neighbors", zorder=5,
    )

    ax.set_xlabel("x / D")
    ax.set_ylabel("y / D")
    ax.set_title(
        f"Layout — Regret = {result.regret:.2f} GWh "
        f"({results_data['regret_pct']:.1f}%)"
    )
    ax.legend(fontsize=7, loc="upper left")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "bilevel_50x50.png", dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_dir / 'bilevel_50x50.png'}")
    plt.close()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY — 50 Targets × 50 Neighbors")
    print("=" * 60)
    print(f"Liberal AEP:        {result.liberal_aep:.2f} GWh")
    print(f"Conservative AEP:   {result.conservative_aep:.2f} GWh")
    print(f"Regret:             {result.regret:.4f} GWh ({results_data['regret_pct']:.2f}%)")
    print(f"Total time:         {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Per outer step:     {elapsed/max(len(result.history),1):.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()

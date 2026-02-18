"""IFT-based bilevel adversarial search for maximum-regret neighbor configurations.

Uses GradientAdversarialSearch (IFT implicit differentiation through SGD)
to find neighbor turbine positions that maximize design regret via gradient ascent.

Setup matches run_regret_discovery.py:
- 16 target turbines in 16D x 16D area, D=200m, 4D min spacing
- BastankhahGaussianDeficit(k=0.04), single wind direction 270°, 9 m/s
- 4 initial neighbors placed upwind (-3D to -6D in x)

Usage:
    pixi run python scripts/run_bilevel_ift.py
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
from pixwake.optim.sgd import SGDSettings


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
    output_dir = Path("analysis/bilevel_ift")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configuration (matches run_regret_discovery.py)
    D = 200.0
    target_size = 16 * D
    min_spacing = 4 * D
    ws = jnp.array([9.0])
    wd = jnp.array([270.0])

    # Create turbine and simulation
    turbine = create_turbine(D)
    deficit = BastankhahGaussianDeficit(k=0.04)
    sim = WakeSimulation(turbine, deficit)

    # Target farm boundary (16D x 16D)
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
    init_target_x = jnp.array([grid_spacing * (i + 1) for i in range(grid_side)] * grid_side)
    init_target_y = jnp.array([grid_spacing * (j + 1) for j in range(grid_side) for _ in range(grid_side)])

    # 4 initial neighbors placed upwind (-3D to -6D in x)
    n_neighbors = 4
    nb_x_positions = [-3 * D, -4 * D, -5 * D, -6 * D]
    nb_y_positions = [
        target_size * 0.2,
        target_size * 0.4,
        target_size * 0.6,
        target_size * 0.8,
    ]
    init_neighbor_x = jnp.array(nb_x_positions)
    init_neighbor_y = jnp.array(nb_y_positions)

    # Neighbor boundary (large area around target for clipping)
    neighbor_bound = jnp.array([
        [-10 * D, -5 * D],
        [target_size + 5 * D, -5 * D],
        [target_size + 5 * D, target_size + 5 * D],
        [-10 * D, target_size + 5 * D],
    ])

    # Inner SGD settings (tighter tolerance for IFT accuracy)
    sgd_settings = SGDSettings(
        learning_rate=D / 5,
        max_iter=3000,
        tol=1e-8,
    )

    # Outer search settings
    search_settings = AdversarialSearchSettings(
        max_iter=50,
        learning_rate=50.0,
        tol=1e-5,
        neighbor_boundary=neighbor_bound,
        sgd_settings=sgd_settings,
        verbose=True,
    )

    print("=" * 60)
    print("IFT Bilevel Adversarial Search")
    print("=" * 60)
    print(f"Target farm: {n_target} turbines in {target_size/D:.0f}D x {target_size/D:.0f}D")
    print(f"Min spacing: {min_spacing/D:.0f}D = {min_spacing:.0f}m")
    print(f"Neighbors: {n_neighbors} turbines")
    print(f"Wind: {float(ws[0]):.0f} m/s @ {float(wd[0]):.0f}°")
    print(f"Deficit: BastankhahGaussian(k=0.04)")
    print(f"Outer iterations: {search_settings.max_iter}")
    print(f"Inner SGD iterations: {sgd_settings.max_iter}")
    print("=" * 60, flush=True)

    # Run search
    searcher = GradientAdversarialSearch(
        sim=sim,
        target_boundary=target_boundary,
        target_min_spacing=min_spacing,
        ws_amb=ws,
        wd_amb=wd,
    )

    start_time = time.time()
    result = searcher.search(
        init_target_x,
        init_target_y,
        init_neighbor_x,
        init_neighbor_y,
        settings=search_settings,
    )
    elapsed = time.time() - start_time

    print(f"\nElapsed: {elapsed:.1f}s")

    # Save results (same schema as blob_discovery/results.json)
    results_data = {
        "method": "ift_bilevel",
        "n_target": n_target,
        "n_neighbors": n_neighbors,
        "D": D,
        "target_size": target_size,
        "min_spacing": min_spacing,
        "ws": [float(x) for x in ws],
        "wd": [float(x) for x in wd],
        "liberal_aep": float(result.liberal_aep),
        "conservative_aep": float(result.conservative_aep),
        "regret": float(result.regret),
        "regret_pct": float(result.regret / result.liberal_aep * 100) if result.liberal_aep > 0 else 0.0,
        "liberal_x": [float(x) for x in result.liberal_x],
        "liberal_y": [float(x) for x in result.liberal_y],
        "target_x": [float(x) for x in result.target_x],
        "target_y": [float(x) for x in result.target_y],
        "neighbor_x": [float(x) for x in result.neighbor_x],
        "neighbor_y": [float(x) for x in result.neighbor_y],
        "init_neighbor_x": [float(x) for x in init_neighbor_x],
        "init_neighbor_y": [float(x) for x in init_neighbor_y],
        "history": [(r, [float(x) for x in nx], [float(y) for y in ny]) for r, nx, ny in result.history],
        "elapsed_seconds": elapsed,
        "outer_iterations": len(result.history),
        "search_settings": {
            "max_iter": search_settings.max_iter,
            "learning_rate": search_settings.learning_rate,
            "sgd_max_iter": sgd_settings.max_iter,
            "sgd_tol": sgd_settings.tol,
        },
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"Results saved to {output_dir / 'results.json'}")

    # Plot convergence curve
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Convergence
    ax = axes[0]
    regrets = [h[0] for h in result.history]
    ax.plot(regrets, "o-", linewidth=2, markersize=4)
    ax.set_xlabel("Outer Iteration")
    ax.set_ylabel("Regret (GWh)")
    ax.set_title("IFT Bilevel Search — Convergence")
    ax.grid(True, alpha=0.3)

    # Layout comparison
    ax = axes[1]
    # Target boundary
    bnd = np.array(target_boundary)
    bnd = np.vstack([bnd, bnd[0]])
    ax.plot(bnd[:, 0], bnd[:, 1], "k-", linewidth=2, label="Target boundary")

    # Liberal layout
    ax.scatter(result.liberal_x, result.liberal_y, c="blue", marker="^", s=80,
               label=f"Liberal (AEP={result.liberal_aep:.1f} GWh)", zorder=5)

    # Conservative layout
    ax.scatter(result.target_x, result.target_y, c="green", marker="s", s=80,
               label=f"Conservative (AEP={result.conservative_aep:.1f} GWh)", zorder=5)

    # Initial neighbors
    ax.scatter(init_neighbor_x, init_neighbor_y, c="red", marker="x", s=100,
               linewidths=2, alpha=0.4, label="Initial neighbors")

    # Final neighbors
    ax.scatter(result.neighbor_x, result.neighbor_y, c="red", marker="D", s=100,
               label="Optimized neighbors", zorder=5)

    # Arrows from initial to final neighbor positions
    for i in range(n_neighbors):
        dx = float(result.neighbor_x[i] - init_neighbor_x[i])
        dy = float(result.neighbor_y[i] - init_neighbor_y[i])
        ax.annotate("", xy=(float(result.neighbor_x[i]), float(result.neighbor_y[i])),
                     xytext=(float(init_neighbor_x[i]), float(init_neighbor_y[i])),
                     arrowprops=dict(arrowstyle="->", color="red", alpha=0.5, lw=1.5))

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Layout — Regret = {result.regret:.2f} GWh ({results_data['regret_pct']:.1f}%)")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "bilevel_ift_results.png", dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_dir / 'bilevel_ift_results.png'}")
    plt.close()


if __name__ == "__main__":
    main()

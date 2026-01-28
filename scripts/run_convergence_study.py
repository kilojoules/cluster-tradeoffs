"""Convergence study: verify regret stabilizes with sufficient optimization starts.

This script tests whether the computed Pareto regret values are converged by running
the same blob configurations with increasing numbers of random starts (5, 10, 20, 40).

The study focuses on high-regret configurations to verify that the observed tradeoffs
are real and not artifacts of insufficient optimization.

Usage:
    pixi run python scripts/run_convergence_study.py
    pixi run python scripts/run_convergence_study.py --n-starts-max=60
"""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

from pixwake import Curve, Turbine, WakeSimulation
from pixwake.deficit import BastankhahGaussianDeficit
from pixwake.optim.adversarial import PooledBlobDiscovery, PooledBlobDiscoverySettings
from pixwake.optim.geometry import sample_random_blob
from pixwake.optim.soft_packing import create_reference_grid
from pixwake.optim.sgd import SGDSettings

from run_regret_discovery import WindRoseConfig, generate_wind_rose


def create_turbine(rotor_diameter: float = 200.0) -> Turbine:
    """Create a 10 MW class turbine."""
    ws = jnp.array([0.0, 4.0, 10.0, 15.0, 25.0])
    power = jnp.array([0.0, 0.0, 10000.0, 10000.0, 0.0])
    ct = jnp.array([0.0, 0.8, 0.8, 0.4, 0.0])
    return Turbine(
        rotor_diameter=rotor_diameter,
        hub_height=120.0,
        power_curve=Curve(ws=ws, values=power),
        ct_curve=Curve(ws=ws, values=ct),
    )


def compute_pareto_regret(layouts: list[dict]) -> tuple[float, int]:
    """Compute Pareto regret from a set of layouts.

    Args:
        layouts: List of layout dictionaries with 'aep_absent' and 'aep_present' keys.

    Returns:
        Tuple of (regret in GWh, number of Pareto points).
    """
    aep_absent = np.array([l["aep_absent"] for l in layouts])
    aep_present = np.array([l["aep_present"] for l in layouts])

    # Find Pareto-optimal points (maximizing both objectives)
    pareto_mask = np.ones(len(layouts), dtype=bool)
    for i in range(len(layouts)):
        for j in range(len(layouts)):
            if i != j:
                # j dominates i if j is >= in both and > in at least one
                if aep_absent[j] >= aep_absent[i] and aep_present[j] >= aep_present[i]:
                    if aep_absent[j] > aep_absent[i] or aep_present[j] > aep_present[i]:
                        pareto_mask[i] = False
                        break

    pareto_indices = np.where(pareto_mask)[0]
    if len(pareto_indices) <= 1:
        return 0.0, len(pareto_indices)

    # Regret = difference in AEP_present between conservative-optimal and liberal-optimal
    pareto_absent = aep_absent[pareto_mask]
    pareto_present = aep_present[pareto_mask]
    lib_opt = np.argmax(pareto_absent)  # Best when neighbors absent
    con_opt = np.argmax(pareto_present)  # Best when neighbors present
    regret = pareto_present[con_opt] - pareto_present[lib_opt]

    return regret, len(pareto_indices)


def run_single_convergence_test(
    wind_config: WindRoseConfig,
    blob_seed: int,
    n_starts_list: list[int],
    D: float = 200.0,
) -> list[dict]:
    """Run discovery with varying number of starts to check convergence.

    Args:
        wind_config: Wind rose configuration.
        blob_seed: Random seed for blob generation.
        n_starts_list: List of n_starts values to test.
        D: Rotor diameter.

    Returns:
        List of result dictionaries, one per n_starts value.
    """
    target_size = 16 * D
    min_spacing = 4 * D

    turbine = create_turbine(D)
    deficit = BastankhahGaussianDeficit(k=0.04)
    sim = WakeSimulation(turbine, deficit)

    target_boundary = jnp.array([
        [0.0, 0.0],
        [target_size, 0.0],
        [target_size, target_size],
        [0.0, target_size],
    ])

    # Initial turbine positions (4x4 grid)
    n_target = 16
    grid_side = int(np.sqrt(n_target))
    spacing = target_size / (grid_side + 1)
    init_x = jnp.array([spacing * (i + 1) for i in range(grid_side)] * grid_side)
    init_y = jnp.array([spacing * (j + 1) for j in range(grid_side) for _ in range(grid_side)])

    # Neighbor grid
    neighbor_center = (-6 * D, target_size / 2)
    neighbor_grid = create_reference_grid(neighbor_center, 6 * D, 3 * D)

    # Generate wind rose
    wd, ws, weights = generate_wind_rose(wind_config)

    # Generate blob (same for all n_starts)
    key = jax.random.PRNGKey(42)
    for _ in range(blob_seed + 1):
        key, subkey = jax.random.split(key)
    control_points = sample_random_blob(
        subkey,
        center_bounds=((-10 * D, -4 * D), (target_size * 0.2, target_size * 0.8)),
        size_bounds=(5 * D, 10 * D),
        aspect_ratio_bounds=(0.6, 1.6),
        n_control=4,
    )

    results = []

    for n_starts in n_starts_list:
        print(f"  n_starts={n_starts}...", end=" ", flush=True)
        start_time = time.time()

        # Adjust iterations based on wind rose complexity
        n_dirs = len(wd)
        max_iter = 3000 if n_dirs <= 12 else 2000 if n_dirs <= 36 else 1500

        settings = PooledBlobDiscoverySettings(
            n_starts=n_starts,
            sgd_settings=SGDSettings(max_iter=max_iter, learning_rate=D / 5),
            verbose=False,
        )

        discoverer = PooledBlobDiscovery(
            sim=sim,
            target_boundary=target_boundary,
            target_min_spacing=min_spacing,
            neighbor_grid=neighbor_grid,
            ws_amb=ws,
            wd_amb=wd,
            weights=weights,
        )

        result = discoverer.discover(
            init_x,
            init_y,
            control_points,
            settings=settings,
            seed=blob_seed * 1000,
        )

        regret, n_pareto = compute_pareto_regret(result.all_layouts)
        elapsed = time.time() - start_time

        results.append({
            "n_starts": n_starts,
            "regret": float(regret),
            "n_pareto": n_pareto,
            "global_best_absent": float(result.global_best_aep_absent),
            "global_best_present": float(result.global_best_aep_present),
            "elapsed_seconds": elapsed,
        })
        print(f"regret={regret:.2f} GWh, {n_pareto} Pareto, {elapsed:.1f}s")

    return results


def plot_convergence(
    all_results: dict[str, list[dict]],
    output_dir: Path,
    labels: dict[str, str],
):
    """Create convergence plot.

    Args:
        all_results: Dictionary mapping config names to result lists.
        output_dir: Output directory for plots.
        labels: Dictionary mapping config names to display labels.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    markers = ["o-", "s-", "^-", "D-", "v-", "p-"]
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12", "#1abc9c"]

    for (name, data), marker, color in zip(all_results.items(), markers, colors):
        n_starts = [d["n_starts"] for d in data]
        regrets = [d["regret"] for d in data]
        label = labels.get(name, name)
        ax.plot(n_starts, regrets, marker, color=color, linewidth=2, markersize=10, label=label)

        # Add horizontal line at converged value
        ax.axhline(y=regrets[-1], color=color, linestyle="--", alpha=0.3)

    ax.set_xlabel("Number of Random Starts (per strategy)", fontsize=12)
    ax.set_ylabel("Pareto Regret [GWh]", fontsize=12)
    ax.set_title("Convergence of Regret with Number of Optimization Starts", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Set x-ticks to match the tested values
    if all_results:
        first_data = list(all_results.values())[0]
        ax.set_xticks([d["n_starts"] for d in first_data])

    plt.tight_layout()
    plt.savefig(output_dir / "convergence_plot.png", dpi=150, facecolor="white")
    plt.close()

    print(f"Convergence plot saved to {output_dir / 'convergence_plot.png'}")


def run_convergence_study(
    n_starts_max: int = 40,
    output_dir: str = "analysis/convergence_study",
):
    """Run full convergence study on high-regret configurations.

    Args:
        n_starts_max: Maximum number of starts to test.
        output_dir: Output directory for results.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define n_starts values to test (doubling sequence up to max)
    n_starts_list = [5]
    val = 10
    while val <= n_starts_max:
        n_starts_list.append(val)
        val *= 2

    # Test configurations (high-regret setups from initial analysis)
    test_cases = [
        (WindRoseConfig(rose_type="single", dominant_dir=270.0), 3, "single_blob3"),
        (WindRoseConfig(rose_type="uniform", n_directions=24), 3, "uniform_blob3"),
        (WindRoseConfig(rose_type="von_mises", n_directions=24, concentration=4.0, dominant_dir=270.0), 1, "vonmises_k4_blob1"),
    ]

    labels = {
        "single_blob3": "Single direction (blob 3)",
        "uniform_blob3": "Uniform (blob 3)",
        "vonmises_k4_blob1": "Von Mises κ=4 (blob 1)",
    }

    all_results = {}

    for wind_config, blob_seed, name in test_cases:
        print(f"\n{'='*50}")
        print(f"Testing: {labels[name]}")
        print(f"{'='*50}")

        results = run_single_convergence_test(wind_config, blob_seed, n_starts_list)
        all_results[name] = results

    # Save results
    with open(output_dir / "convergence_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Create plot
    plot_convergence(all_results, output_dir, labels)

    # Print summary
    print("\n" + "=" * 70)
    print("CONVERGENCE SUMMARY")
    print("=" * 70)

    header = f"{'Config':<25}" + "".join(f"n={n:>4}" for n in n_starts_list)
    print(header)
    print("-" * 70)

    for name, results in all_results.items():
        regrets = [r["regret"] for r in results]
        row = f"{labels[name]:<25}" + "".join(f"{r:>8.2f}" for r in regrets)
        print(row)

    print("=" * 70)

    # Convergence assessment
    print("\nCONVERGENCE ASSESSMENT:")
    for name, results in all_results.items():
        regrets = [r["regret"] for r in results]
        # Check if last two values are within 5% of each other
        if len(regrets) >= 2:
            diff = abs(regrets[-1] - regrets[-2])
            pct = diff / max(regrets[-1], 0.1) * 100
            converged = pct < 5
            status = "CONVERGED" if converged else "NOT CONVERGED"
            print(f"  {labels[name]}: {status} (Δ={diff:.2f} GWh, {pct:.1f}%)")

    print(f"\nResults saved to {output_dir}")
    return all_results


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run convergence study for regret analysis.",
    )
    parser.add_argument(
        "--n-starts-max",
        type=int,
        default=40,
        help="Maximum number of starts to test (default: 40)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="analysis/convergence_study",
        help="Output directory (default: analysis/convergence_study)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_convergence_study(
        n_starts_max=args.n_starts_max,
        output_dir=args.output_dir,
    )

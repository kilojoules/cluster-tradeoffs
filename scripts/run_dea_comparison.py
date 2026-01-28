"""Compare blob analysis framework with DEA Energy Island findings.

The OMAE 2026 paper found NO design tradeoffs for the Danish Energy Island
cluster. Key characteristics of that setup:
- Wind rose: κ≈0.59 (very diffuse), dominant direction ~250° (SW-W)
- 9 neighboring farms surrounding the target
- Result: single dominant Pareto point (0.7% variation in liberal AEP)

This script tests our blob analysis framework with:
1. A wind rose matching the DEA characteristics (κ=0.6, dominant dir=250°)
2. Compares with other κ values to understand the relationship
3. Verifies if low regret is found for diffuse wind roses

Usage:
    pixi run python scripts/run_dea_comparison.py
    pixi run python scripts/run_dea_comparison.py --n-blobs=10 --n-starts=10
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

jax.config.update("jax_enable_x64", True)

from pixwake import Curve, Turbine, WakeSimulation

# Import from the main script
import sys
sys.path.insert(0, str(Path(__file__).parent))
from run_regret_discovery import (
    WindRoseConfig,
    generate_wind_rose,
    plot_wind_rose,
    create_turbine,
    run_multistart_pooled_discovery,
    compute_pareto_and_regret,
)


def run_dea_comparison(
    n_blobs: int = 5,
    n_starts: int = 5,
    output_dir: str = "analysis/dea_comparison",
):
    """Run comparison across wind rose configurations matching DEA study."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Wind rose configurations to test
    # DEA site has κ≈0.59, dominant direction ~250°
    configs = [
        # DEA-like configuration
        WindRoseConfig(
            rose_type="von_mises",
            n_directions=24,
            dominant_dir=250.0,  # DEA circular mean
            concentration=0.6,   # DEA estimated κ
            mean_ws=10.6,        # DEA mean wind speed
        ),
        # Slightly more concentrated
        WindRoseConfig(
            rose_type="von_mises",
            n_directions=24,
            dominant_dir=250.0,
            concentration=1.0,
            mean_ws=10.6,
        ),
        # Our "sweet spot" from previous analysis
        WindRoseConfig(
            rose_type="von_mises",
            n_directions=24,
            dominant_dir=250.0,
            concentration=2.0,
            mean_ws=10.6,
        ),
        # Single direction (worst case)
        WindRoseConfig(
            rose_type="single",
            dominant_dir=250.0,
            mean_ws=10.6,
        ),
    ]

    # Run analysis for each configuration
    all_results = {}

    for config in configs:
        print(f"\n{'='*60}")
        print(f"Running: {config}")
        print(f"{'='*60}")

        config_name = str(config)
        config_output_dir = output_path / config_name.replace(".", "_")
        config_output_dir.mkdir(parents=True, exist_ok=True)

        # Generate wind rose
        wd, ws, weights = generate_wind_rose(config)

        # Save wind rose config
        config_dict = {
            "rose_type": config.rose_type,
            "n_directions": config.n_directions,
            "dominant_dir": config.dominant_dir,
            "concentration": config.concentration,
            "mean_ws": config.mean_ws,
        }
        with open(config_output_dir / "wind_rose_config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

        # Plot wind rose
        ax = plot_wind_rose(wd, ws, weights, title=f"Wind Rose: {config}")
        ax.figure.savefig(config_output_dir / "wind_rose.png", dpi=150, bbox_inches="tight")
        plt.close(ax.figure)

        # Run pooled multi-start discovery
        results = run_multistart_pooled_discovery(
            n_blobs=n_blobs,
            n_starts_per_strategy=n_starts,
            output_dir=str(config_output_dir),
            wind_rose_config=config,
        )

        # Compute summary statistics
        # Use min_liberal_regret as the primary regret metric
        regrets = [r["min_liberal_regret"] for r in results]
        max_regret = max(regrets)
        mean_regret = np.mean(regrets)
        # Count blobs where there's a true tradeoff (not same best layout)
        blobs_with_tradeoff = sum(1 for r in results if not r["same_best_layout"])

        all_results[config_name] = {
            "config": config_dict,
            "max_regret": max_regret,
            "mean_regret": mean_regret,
            "blobs_with_tradeoff": blobs_with_tradeoff,
            "n_blobs": n_blobs,
            "results": results,
        }

        print(f"\n  Max regret: {max_regret:.2f} GWh")
        print(f"  Mean regret: {mean_regret:.2f} GWh")
        print(f"  Blobs with tradeoff: {blobs_with_tradeoff}/{n_blobs}")

    # Save summary
    summary = {
        name: {
            "max_regret_gwh": data["max_regret"],
            "mean_regret_gwh": data["mean_regret"],
            "blobs_with_tradeoff": data["blobs_with_tradeoff"],
            "n_blobs": data["n_blobs"],
            "config": data["config"],
        }
        for name, data in all_results.items()
    }

    with open(output_path / "dea_comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Create comparison plot
    plot_dea_comparison(all_results, output_path)

    # Print summary table
    print("\n" + "="*70)
    print("DEA COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Wind Rose':<35} {'Max Regret':>12} {'Mean Regret':>12} {'Tradeoffs':>10}")
    print("-"*70)
    for name, data in all_results.items():
        print(f"{name:<35} {data['max_regret']:>10.2f} GWh {data['mean_regret']:>10.2f} GWh {data['blobs_with_tradeoff']:>6}/{n_blobs}")
    print("="*70)

    # Compare with DEA paper findings
    print("\n" + "="*70)
    print("COMPARISON WITH OMAE 2026 PAPER FINDINGS")
    print("="*70)
    print("""
DEA Energy Island Study (OMAE 2026):
- Wind rose: κ≈0.59, dominant direction ~250°
- Result: NO significant design tradeoffs
- Liberal AEP variation: 0.7%
- Conservative AEP variation: 0.6%
- Single dominant Pareto point

Our Blob Analysis with DEA-like wind rose (κ=0.6):""")

    dea_like_key = [k for k in all_results.keys() if "k0.6" in k][0]
    dea_result = all_results[dea_like_key]
    print(f"- Max regret: {dea_result['max_regret']:.2f} GWh")
    print(f"- Mean regret: {dea_result['mean_regret']:.2f} GWh")
    print(f"- Blobs with tradeoff: {dea_result['blobs_with_tradeoff']}/{n_blobs}")

    if dea_result['max_regret'] < 5.0:
        print("\n✓ CONSISTENT: Low regret found, similar to DEA findings")
    else:
        print("\n⚠ NOTE: Some regret found with blob configurations")
        print("  This may be because our blob analysis can find adversarial")
        print("  neighbor configurations not present in the fixed DEA geometry.")

    return all_results


def plot_dea_comparison(all_results: dict, output_path: Path):
    """Create comparison visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Bar chart of max/mean regret by configuration
    ax1 = axes[0]
    names = list(all_results.keys())
    max_regrets = [all_results[n]["max_regret"] for n in names]
    mean_regrets = [all_results[n]["mean_regret"] for n in names]

    x = np.arange(len(names))
    width = 0.35

    ax1.bar(x - width/2, max_regrets, width, label='Max Regret', color='coral')
    ax1.bar(x + width/2, mean_regrets, width, label='Mean Regret', color='steelblue')

    ax1.set_ylabel('Regret (GWh)')
    ax1.set_title('Design Regret by Wind Rose Configuration')
    ax1.set_xticks(x)
    ax1.set_xticklabels([n.replace('_', '\n') for n in names], fontsize=8)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Add horizontal line at DEA-like result
    dea_like_key = [k for k in names if "k0.6" in k][0]
    ax1.axhline(y=all_results[dea_like_key]["max_regret"], color='green',
                linestyle='--', alpha=0.5, label=f'DEA-like (κ=0.6)')

    # Right: Regret vs concentration (for von_mises configs)
    ax2 = axes[1]
    vm_configs = [(n, all_results[n]) for n in names if "von_mises" in n]

    if vm_configs:
        kappas = [d["config"]["concentration"] for n, d in vm_configs]
        max_regs = [d["max_regret"] for n, d in vm_configs]
        mean_regs = [d["mean_regret"] for n, d in vm_configs]

        ax2.plot(kappas, max_regs, 'o-', color='coral', label='Max Regret', markersize=10)
        ax2.plot(kappas, mean_regs, 's-', color='steelblue', label='Mean Regret', markersize=10)

        # Add single direction as κ→∞ reference
        if "single" in str(names):
            single_key = [k for k in names if "single" in k][0]
            ax2.axhline(y=all_results[single_key]["max_regret"], color='coral',
                       linestyle=':', alpha=0.7)
            ax2.axhline(y=all_results[single_key]["mean_regret"], color='steelblue',
                       linestyle=':', alpha=0.7)
            ax2.annotate('Single direction\n(κ→∞)', xy=(max(kappas)+0.3, all_results[single_key]["max_regret"]),
                        fontsize=8, color='coral')

        # Mark DEA κ value
        ax2.axvline(x=0.59, color='green', linestyle='--', alpha=0.5, label='DEA κ≈0.59')

        ax2.set_xlabel('Von Mises Concentration (κ)')
        ax2.set_ylabel('Regret (GWh)')
        ax2.set_title('Regret vs Wind Rose Concentration')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, max(kappas) + 0.5)

    plt.tight_layout()
    fig.savefig(output_path / "dea_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved comparison plot to {output_path / 'dea_comparison.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare blob analysis with DEA Energy Island findings"
    )
    parser.add_argument(
        "--n-blobs", type=int, default=5,
        help="Number of blob configurations to test (default: 5)"
    )
    parser.add_argument(
        "--n-starts", type=int, default=5,
        help="Number of optimization starts per strategy (default: 5)"
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, default="analysis/dea_comparison",
        help="Output directory for results"
    )

    args = parser.parse_args()

    results = run_dea_comparison(
        n_blobs=args.n_blobs,
        n_starts=args.n_starts,
        output_dir=args.output_dir,
    )

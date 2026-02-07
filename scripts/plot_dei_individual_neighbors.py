"""Generate individual neighbor Pareto plots for DEI case study.

For each neighbor farm, shows scatter plot of:
- X-axis: AEP without that neighbor (liberal scenario)
- Y-axis: AEP with that neighbor (conservative scenario)

Each plot shows 100 points from 50 multi-start optimizations:
- 50 liberal starts (optimized ignoring neighbor, evaluated both ways)
- 50 conservative starts (optimized considering neighbor, evaluated both ways)
"""

import re
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).parent.parent / "analysis/dei_50starts_2000iter"
OUTPUT_DIR = Path(__file__).parent.parent / "docs/figures"


def parse_log(log_file):
    """Parse optimization results from log file.

    Returns list of (aep_absent, aep_present, strategy) tuples.
    """
    results = []

    with open(log_file, 'r') as f:
        for line in f:
            match = re.search(
                r'Start \d+: Liberal=([0-9.]+)/([0-9.]+), Conservative=([0-9.]+)/([0-9.]+)',
                line
            )
            if match:
                lib_absent, lib_present = float(match.group(1)), float(match.group(2))
                con_absent, con_present = float(match.group(3)), float(match.group(4))
                results.append((lib_absent, lib_present, 'liberal'))
                results.append((con_absent, con_present, 'conservative'))

    return results


def compute_pareto(points):
    """Find Pareto-optimal points (maximizing both objectives)."""
    points = np.array(points)
    n = len(points)
    pareto_mask = np.ones(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if i != j:
                if (points[j, 0] >= points[i, 0] and points[j, 1] >= points[i, 1] and
                    (points[j, 0] > points[i, 0] or points[j, 1] > points[i, 1])):
                    pareto_mask[i] = False
                    break

    return pareto_mask


def load_farm_info(farm_idx):
    """Load farm metadata from JSON."""
    json_file = RESULTS_DIR / f"dei_single_neighbor_bastankhah_farm{farm_idx}.json"
    if json_file.exists():
        with open(json_file) as f:
            data = json.load(f)
            return data[str(farm_idx)]
    return None


def plot_single_neighbor(ax, farm_idx, results, info):
    """Plot Pareto scatter for a single neighbor."""

    absent = np.array([r[0] for r in results])
    present = np.array([r[1] for r in results])
    strategies = [r[2] for r in results]

    lib_mask = np.array([s == 'liberal' for s in strategies])
    con_mask = np.array([s == 'conservative' for s in strategies])

    # Compute Pareto front
    pareto_mask = compute_pareto(list(zip(absent, present)))
    n_pareto = pareto_mask.sum()

    # Compute regret
    pareto_idx = np.where(pareto_mask)[0]
    if len(pareto_idx) > 0:
        lib_opt_idx = pareto_idx[np.argmax(absent[pareto_mask])]
        con_opt_idx = pareto_idx[np.argmax(present[pareto_mask])]
        regret = present[con_opt_idx] - present[lib_opt_idx]
    else:
        regret = 0

    # Plot dominated points (smaller, more transparent)
    ax.scatter(absent[lib_mask & ~pareto_mask], present[lib_mask & ~pareto_mask],
               s=30, c='#3498db', alpha=0.4, marker='o', label='Liberal')
    ax.scatter(absent[con_mask & ~pareto_mask], present[con_mask & ~pareto_mask],
               s=30, c='#e74c3c', alpha=0.4, marker='s', label='Conservative')

    # Plot Pareto points (larger, solid)
    ax.scatter(absent[lib_mask & pareto_mask], present[lib_mask & pareto_mask],
               s=100, c='#3498db', alpha=1.0, marker='o',
               edgecolors='black', linewidths=1.5, zorder=5)
    ax.scatter(absent[con_mask & pareto_mask], present[con_mask & pareto_mask],
               s=100, c='#e74c3c', alpha=1.0, marker='s',
               edgecolors='black', linewidths=1.5, zorder=5)

    # Draw Pareto front line
    if n_pareto > 1:
        pareto_absent = absent[pareto_mask]
        pareto_present = present[pareto_mask]
        sort_idx = np.argsort(pareto_absent)
        ax.plot(pareto_absent[sort_idx], pareto_present[sort_idx],
                'k--', linewidth=1.5, alpha=0.7, zorder=4)

    # Title
    direction = info['direction'] if info else 0
    name = info['name'].split('(')[1].rstrip(')') if info else ''

    title_color = '#e74c3c' if regret > 1 else '#333333'
    ax.set_title(f'Farm {farm_idx}: {name} ({direction:.0f}°)\n'
                 f'Regret: {regret:.1f} GWh | {n_pareto} Pareto pts',
                 fontsize=10, fontweight='bold', color=title_color)

    ax.set_xlabel('AEP without neighbor (GWh)', fontsize=9)
    ax.set_ylabel('AEP with neighbor (GWh)', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Equal aspect for small variations
    x_range = absent.max() - absent.min()
    y_range = present.max() - present.min()

    if x_range > 0:
        ax.set_xlim(absent.min() - x_range * 0.1, absent.max() + x_range * 0.1)
    if y_range > 0:
        ax.set_ylim(present.min() - y_range * 0.1, present.max() + y_range * 0.1)

    return n_pareto, regret


def main():
    # Create 3x3 grid
    fig, axes = plt.subplots(3, 3, figsize=(14, 14))
    axes = axes.flatten()

    for idx, farm_idx in enumerate(range(1, 10)):
        ax = axes[idx]
        log_file = RESULTS_DIR / f"farm{farm_idx}.log"

        if not log_file.exists():
            ax.text(0.5, 0.5, f'Farm {farm_idx}\nNo data',
                    ha='center', va='center', fontsize=12, transform=ax.transAxes)
            continue

        results = parse_log(log_file)
        if not results:
            ax.text(0.5, 0.5, f'Farm {farm_idx}\nNo data',
                    ha='center', va='center', fontsize=12, transform=ax.transAxes)
            continue

        info = load_farm_info(farm_idx)
        n_pareto, regret = plot_single_neighbor(ax, farm_idx, results, info)

        print(f"Farm {farm_idx}: {n_pareto} Pareto points, {regret:.1f} GWh regret")

        # Add legend to first plot only
        if idx == 0:
            ax.legend(loc='lower left', fontsize=8)

    plt.suptitle('Individual Neighbor Analysis: Pareto Frontiers\n'
                 '(50 liberal + 50 conservative multi-start optimizations per neighbor)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_file = OUTPUT_DIR / "dei_individual_neighbors.png"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"\nSaved: {output_file}")


if __name__ == "__main__":
    main()

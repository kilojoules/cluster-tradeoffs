"""Generate Pareto plots for all DEI neighbor cases."""

import re
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).parent.parent / "analysis/dei_50starts_2000iter"
OUTPUT_FILE = Path(__file__).parent.parent / "docs/figures/dei_all_pareto.png"


def parse_log(log_file):
    """Parse optimization results from log file."""
    liberal_results = []
    conservative_results = []

    with open(log_file, 'r') as f:
        for line in f:
            match = re.search(
                r'Start \d+: Liberal=([0-9.]+)/([0-9.]+), Conservative=([0-9.]+)/([0-9.]+)',
                line
            )
            if match:
                lib_absent, lib_present = float(match.group(1)), float(match.group(2))
                con_absent, con_present = float(match.group(3)), float(match.group(4))
                liberal_results.append((lib_absent, lib_present))
                conservative_results.append((con_absent, con_present))

    return liberal_results, conservative_results


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
    json_file = RESULTS_DIR / f"dea_single_neighbor_bastankhah_farm{farm_idx}.json"
    if json_file.exists():
        with open(json_file) as f:
            data = json.load(f)
            return data[str(farm_idx)]
    return None


def main():
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()

    for idx, farm_idx in enumerate(range(1, 10)):
        ax = axes[idx]
        log_file = RESULTS_DIR / f"farm{farm_idx}.log"

        if not log_file.exists():
            ax.text(0.5, 0.5, f'Farm {farm_idx}\nNo data', ha='center', va='center', fontsize=12)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            continue

        # Parse log
        liberal, conservative = parse_log(log_file)
        if not liberal:
            ax.text(0.5, 0.5, f'Farm {farm_idx}\nNo data', ha='center', va='center', fontsize=12)
            continue

        # Load metadata
        info = load_farm_info(farm_idx)
        direction = info['direction'] if info else 0
        regret = info['regret_gwh'] if info else 0
        name = info['name'].split('(')[1].rstrip(')') if info else ''

        # Combine all results
        all_points = liberal + conservative
        all_absent = np.array([p[0] for p in all_points])
        all_present = np.array([p[1] for p in all_points])
        strategies = ['liberal'] * len(liberal) + ['conservative'] * len(conservative)

        # Find Pareto front
        pareto_mask = compute_pareto(list(zip(all_absent, all_present)))
        n_pareto = pareto_mask.sum()

        lib_mask = np.array([s == 'liberal' for s in strategies])
        con_mask = np.array([s == 'conservative' for s in strategies])

        # Dominated points
        ax.scatter(all_absent[lib_mask & ~pareto_mask], all_present[lib_mask & ~pareto_mask],
                   s=40, c='#3498db', alpha=0.3, marker='o')
        ax.scatter(all_absent[con_mask & ~pareto_mask], all_present[con_mask & ~pareto_mask],
                   s=40, c='#e74c3c', alpha=0.3, marker='s')

        # Pareto points
        ax.scatter(all_absent[lib_mask & pareto_mask], all_present[lib_mask & pareto_mask],
                   s=100, c='#3498db', alpha=1.0, marker='o',
                   edgecolors='black', linewidths=1.5, zorder=5, label='Liberal')
        ax.scatter(all_absent[con_mask & pareto_mask], all_present[con_mask & pareto_mask],
                   s=100, c='#e74c3c', alpha=1.0, marker='s',
                   edgecolors='black', linewidths=1.5, zorder=5, label='Conservative')

        # Draw Pareto front line
        pareto_absent = all_absent[pareto_mask]
        pareto_present = all_present[pareto_mask]
        if len(pareto_absent) > 1:
            sort_idx = np.argsort(pareto_absent)
            ax.plot(pareto_absent[sort_idx], pareto_present[sort_idx],
                    'k--', linewidth=1.5, alpha=0.7, zorder=4)

        # Title with regret
        color = '#e74c3c' if regret > 1 else '#333333'
        ax.set_title(f'Farm {farm_idx} ({name}, {direction:.0f}°)\n'
                     f'Regret: {regret:.1f} GWh | {n_pareto} Pareto pts',
                     fontsize=10, fontweight='bold', color=color)

        ax.set_xlabel('AEP (alone)', fontsize=9)
        ax.set_ylabel('AEP (with neighbor)', fontsize=9)
        ax.grid(True, alpha=0.3)

        # Set axis limits with padding
        x_range = all_absent.max() - all_absent.min()
        y_range = all_present.max() - all_present.min()
        if x_range > 0:
            ax.set_xlim(all_absent.min() - x_range * 0.1, all_absent.max() + x_range * 0.1)
        if y_range > 0:
            ax.set_ylim(all_present.min() - y_range * 0.1, all_present.max() + y_range * 0.1)

        if idx == 0:
            ax.legend(loc='lower left', fontsize=8)

    plt.suptitle('DEI Pareto Frontiers by Neighbor\n(50 starts × 2 strategies = 100 optimizations each)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FILE, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

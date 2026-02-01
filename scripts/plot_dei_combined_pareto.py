"""Generate Pareto plot for DEI all-neighbors-combined case."""

import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

LOG_FILE = Path(__file__).parent.parent / "analysis/dei_50starts_2000iter_combined/all_neighbors.log"
OUTPUT_FILE = Path(__file__).parent.parent / "docs/figures/dei_pareto_combined.png"


def parse_log(log_file):
    """Parse optimization results from log file."""
    liberal_results = []
    conservative_results = []
    in_combined_section = False

    with open(log_file, 'r') as f:
        for line in f:
            if "ALL 9 NEIGHBORS TOGETHER" in line:
                in_combined_section = True
                continue

            if in_combined_section:
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


def main():
    print(f"Parsing {LOG_FILE}...")
    liberal, conservative = parse_log(LOG_FILE)

    print(f"Found {len(liberal)} liberal and {len(conservative)} conservative results")

    # Combine all results
    all_points = liberal + conservative
    all_absent = np.array([p[0] for p in all_points])
    all_present = np.array([p[1] for p in all_points])
    strategies = ['liberal'] * len(liberal) + ['conservative'] * len(conservative)

    # Find Pareto front
    pareto_mask = compute_pareto(list(zip(all_absent, all_present)))

    # Find liberal-optimal and conservative-optimal on Pareto front
    pareto_idx = np.where(pareto_mask)[0]
    lib_opt_idx = pareto_idx[np.argmax(all_absent[pareto_mask])]
    con_opt_idx = pareto_idx[np.argmax(all_present[pareto_mask])]
    regret = all_present[con_opt_idx] - all_present[lib_opt_idx]

    print(f"Pareto points: {pareto_mask.sum()}")
    print(f"Regret: {regret:.1f} GWh")

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    lib_mask = np.array([s == 'liberal' for s in strategies])
    con_mask = np.array([s == 'conservative' for s in strategies])

    # Dominated points
    ax.scatter(all_absent[lib_mask & ~pareto_mask], all_present[lib_mask & ~pareto_mask],
               s=80, c='#3498db', alpha=0.4, label='Liberal (dominated)', marker='o')
    ax.scatter(all_absent[con_mask & ~pareto_mask], all_present[con_mask & ~pareto_mask],
               s=80, c='#e74c3c', alpha=0.4, label='Conservative (dominated)', marker='s')

    # Pareto points
    ax.scatter(all_absent[lib_mask & pareto_mask], all_present[lib_mask & pareto_mask],
               s=180, c='#3498db', alpha=1.0, label='Liberal (Pareto)', marker='o',
               edgecolors='black', linewidths=2, zorder=5)
    ax.scatter(all_absent[con_mask & pareto_mask], all_present[con_mask & pareto_mask],
               s=180, c='#e74c3c', alpha=1.0, label='Conservative (Pareto)', marker='s',
               edgecolors='black', linewidths=2, zorder=5)

    # Draw Pareto front line
    pareto_absent = all_absent[pareto_mask]
    pareto_present = all_present[pareto_mask]
    sort_idx = np.argsort(pareto_absent)
    ax.plot(pareto_absent[sort_idx], pareto_present[sort_idx], 'k--', linewidth=2, alpha=0.7, zorder=4)

    # Annotate regret
    mid_x = all_absent[lib_opt_idx] - 4
    ax.annotate('', xy=(mid_x, all_present[con_opt_idx]),
                xytext=(mid_x, all_present[lib_opt_idx]),
                arrowprops=dict(arrowstyle='<->', color='#e74c3c', lw=2.5))
    ax.text(mid_x - 3, (all_present[lib_opt_idx] + all_present[con_opt_idx])/2,
            f'Regret\n{regret:.0f} GWh', fontsize=12, fontweight='bold',
            color='#e74c3c', va='center', ha='right')

    # Labels
    ax.set_xlabel('AEP without neighbors (GWh/year)', fontsize=12)
    ax.set_ylabel('AEP with all 9 neighbors (GWh/year)', fontsize=12)
    ax.set_title(f'Pareto Front: All 9 Neighbors Combined\n'
                 f'{pareto_mask.sum()} Pareto points from 100 optimizations | '
                 f'594 neighbor turbines',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Set axis limits
    x_range = all_absent.max() - all_absent.min()
    y_range = all_present.max() - all_present.min()
    ax.set_xlim(all_absent.min() - x_range * 0.15, all_absent.max() + x_range * 0.1)
    ax.set_ylim(all_present.min() - y_range * 0.1, all_present.max() + y_range * 0.1)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FILE, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

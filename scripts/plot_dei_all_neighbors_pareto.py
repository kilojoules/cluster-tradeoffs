"""Generate Pareto plot for DEI with ALL neighbors combined.

Shows scatter plot of:
- X-axis: AEP without any neighbors
- Y-axis: AEP with all 9 neighbors present

Each point is from 50 multi-start optimizations:
- Liberal: optimized ignoring all neighbors, evaluated both ways
- Conservative: optimized considering all 9 neighbors, evaluated both ways
"""

import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

LOG_FILE = Path(__file__).parent.parent / "analysis/dei_50starts_2000iter_combined/all_neighbors.log"
OUTPUT_FILE = Path(__file__).parent.parent / "docs/figures/dei_all_neighbors_pareto.png"


def parse_log(log_file):
    """Parse optimization results from log file.

    Returns list of (aep_absent, aep_present, strategy) tuples.
    """
    results = []
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


def main():
    print(f"Parsing {LOG_FILE}...")
    results = parse_log(LOG_FILE)

    n_liberal = sum(1 for r in results if r[2] == 'liberal')
    n_conservative = sum(1 for r in results if r[2] == 'conservative')
    print(f"Found {n_liberal} liberal and {n_conservative} conservative results")

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
    lib_opt_idx = pareto_idx[np.argmax(absent[pareto_mask])]
    con_opt_idx = pareto_idx[np.argmax(present[pareto_mask])]
    regret = present[con_opt_idx] - present[lib_opt_idx]

    print(f"Pareto points: {n_pareto}")
    print(f"Regret: {regret:.1f} GWh")
    print(f"Liberal-optimal: {absent[lib_opt_idx]:.1f} / {present[lib_opt_idx]:.1f} GWh")
    print(f"Conservative-optimal: {absent[con_opt_idx]:.1f} / {present[con_opt_idx]:.1f} GWh")

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot dominated points
    ax.scatter(absent[lib_mask & ~pareto_mask], present[lib_mask & ~pareto_mask],
               s=80, c='#3498db', alpha=0.4, marker='o', label=f'Liberal ({n_liberal} starts)')
    ax.scatter(absent[con_mask & ~pareto_mask], present[con_mask & ~pareto_mask],
               s=80, c='#e74c3c', alpha=0.4, marker='s', label=f'Conservative ({n_conservative} starts)')

    # Plot Pareto points
    ax.scatter(absent[lib_mask & pareto_mask], present[lib_mask & pareto_mask],
               s=200, c='#3498db', alpha=1.0, marker='o',
               edgecolors='black', linewidths=2, zorder=5)
    ax.scatter(absent[con_mask & pareto_mask], present[con_mask & pareto_mask],
               s=200, c='#e74c3c', alpha=1.0, marker='s',
               edgecolors='black', linewidths=2, zorder=5)

    # Draw Pareto front line
    if n_pareto > 1:
        pareto_absent = absent[pareto_mask]
        pareto_present = present[pareto_mask]
        sort_idx = np.argsort(pareto_absent)
        ax.plot(pareto_absent[sort_idx], pareto_present[sort_idx],
                'k--', linewidth=2, alpha=0.7, zorder=4)

    # Annotate regret
    mid_x = absent[lib_opt_idx] - 3
    ax.annotate('', xy=(mid_x, present[con_opt_idx]),
                xytext=(mid_x, present[lib_opt_idx]),
                arrowprops=dict(arrowstyle='<->', color='#e74c3c', lw=2.5))
    ax.text(mid_x - 2, (present[lib_opt_idx] + present[con_opt_idx]) / 2,
            f'Regret\n{regret:.0f} GWh', fontsize=12, fontweight='bold',
            color='#e74c3c', va='center', ha='right')

    # Annotate key points
    ax.annotate(f'Liberal-optimal\n({absent[lib_opt_idx]:.0f}, {present[lib_opt_idx]:.0f})',
                xy=(absent[lib_opt_idx], present[lib_opt_idx]),
                xytext=(absent[lib_opt_idx] + 3, present[lib_opt_idx] - 15),
                fontsize=9, ha='left',
                arrowprops=dict(arrowstyle='->', color='#3498db', lw=1))

    ax.annotate(f'Conservative-optimal\n({absent[con_opt_idx]:.0f}, {present[con_opt_idx]:.0f})',
                xy=(absent[con_opt_idx], present[con_opt_idx]),
                xytext=(absent[con_opt_idx] + 3, present[con_opt_idx] + 5),
                fontsize=9, ha='left',
                arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1))

    ax.set_xlabel('AEP without any neighbors (GWh/year)', fontsize=12)
    ax.set_ylabel('AEP with all 9 neighbors (GWh/year)', fontsize=12)
    ax.set_title(f'All 9 Neighbors Combined: Pareto Frontier\n'
                 f'{n_pareto} Pareto points | Regret: {regret:.1f} GWh | '
                 f'594 neighbor turbines',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='lower left', fontsize=11)
    ax.grid(True, alpha=0.3)

    # Set axis limits
    x_range = absent.max() - absent.min()
    y_range = present.max() - present.min()
    ax.set_xlim(absent.min() - x_range * 0.15, absent.max() + x_range * 0.15)
    ax.set_ylim(present.min() - y_range * 0.1, present.max() + y_range * 0.1)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FILE, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

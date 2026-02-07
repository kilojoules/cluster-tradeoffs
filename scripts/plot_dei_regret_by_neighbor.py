"""Generate polar plot of regret by neighbor direction for DEI case study."""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).parent.parent / "analysis/dei_50starts_2000iter"
OUTPUT_FILE = Path(__file__).parent.parent / "docs/figures/dei_regret_by_neighbor.png"


def load_results():
    """Load all farm results."""
    results = {}
    for farm_idx in range(1, 10):
        json_file = RESULTS_DIR / f"dei_single_neighbor_bastankhah_farm{farm_idx}.json"
        if json_file.exists():
            with open(json_file) as f:
                data = json.load(f)
                results[farm_idx] = data[str(farm_idx)]
    return results


def main():
    results = load_results()
    print(f"Loaded {len(results)} farm results")

    # Extract data
    farm_indices = sorted(results.keys())
    directions = [results[i]['direction'] for i in farm_indices]
    regrets = [results[i]['regret_gwh'] for i in farm_indices]
    names = [results[i]['name'].split('(')[1].rstrip(')') for i in farm_indices]

    # Create polar plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})

    # Convert directions to radians (meteorological convention: 0=N, clockwise)
    theta = np.deg2rad(directions)

    # Bar width
    width = 0.35

    # Colors: highlight Farm 8
    colors = ['#3498db' if r < 1 else '#e74c3c' for r in regrets]

    # Plot bars
    bars = ax.bar(theta, regrets, width=width, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)

    # Add labels
    max_regret = max(regrets) if max(regrets) > 0 else 1
    for i, (t, r, name, idx) in enumerate(zip(theta, regrets, names, farm_indices)):
        label_r = r + max_regret * 0.15 if r > 0.5 else max_regret * 0.2
        ax.annotate(f'Farm {idx}\n({name})\n{r:.1f} GWh',
                   xy=(t, label_r),
                   ha='center', va='bottom' if r > 0.5 else 'center',
                   fontsize=9, fontweight='bold' if r > 1 else 'normal')

    # Configure polar plot
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)  # Clockwise
    ax.set_ylim(0, max_regret * 1.5)

    # Add cardinal directions
    ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
    ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'], fontsize=11)

    ax.set_title('Design Regret by Neighbor Direction\nDanish Energy Island (50 starts, 2000 iter)',
                 fontsize=14, fontweight='bold', pad=20)

    # Add annotation about Farm 8
    ax.annotate('Farm 8 (South)\nis sole source\nof regret',
               xy=(np.deg2rad(163), 20.4),
               xytext=(np.deg2rad(120), 28),
               fontsize=10, color='#e74c3c',
               arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5),
               ha='center')

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FILE, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

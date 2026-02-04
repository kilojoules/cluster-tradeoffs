"""Generate clean polar plot of DEI regret by neighbor direction."""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def plot_polar_regret(results_file, output_file, title_suffix=""):
    """Create a clean polar plot of regret by direction."""

    with open(results_file) as f:
        results = json.load(f)

    # Extract farm data (exclude "all_neighbors")
    farms = []
    for key, data in results.items():
        if key != "all_neighbors" and isinstance(data, dict) and "direction" in data:
            farms.append({
                "idx": int(key),
                "name": data["name"],
                "direction": data["direction"],
                "regret": data["regret_gwh"],
            })

    farms.sort(key=lambda x: x["direction"])

    # Get combined regret if available
    combined_regret = results.get("all_neighbors", {}).get("regret_gwh", None)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})

    directions = np.array([f["direction"] for f in farms])
    regrets = np.array([f["regret"] for f in farms])
    directions_rad = np.radians(directions)

    # Color bars by regret magnitude
    colors = ['#d62728' if r > 10 else '#ff7f0e' if r > 0 else '#2ca02c' for r in regrets]

    # Plot bars with smaller width
    bars = ax.bar(directions_rad, regrets, width=0.25, alpha=0.8, color=colors, edgecolor='black', linewidth=1.5)

    # Set up the plot
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    # Add farm labels OUTSIDE the plot at fixed radius
    max_regret = max(regrets) if max(regrets) > 0 else 10
    label_radius = max_regret * 1.35

    for farm in farms:
        angle_rad = np.radians(farm["direction"])

        # Place label outside
        if farm["regret"] > 0:
            label = f"Farm {farm['idx']}\n{farm['regret']:.1f} GWh"
        else:
            label = f"Farm {farm['idx']}\n(0)"

        ax.annotate(label,
                   xy=(angle_rad, label_radius),
                   ha='center', va='center',
                   fontsize=10, fontweight='bold' if farm["regret"] > 0 else 'normal',
                   color='#d62728' if farm["regret"] > 10 else '#ff7f0e' if farm["regret"] > 0 else '#666666')

    # Add direction labels
    ax.set_xticks(np.radians([0, 45, 90, 135, 180, 225, 270, 315]))
    ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'], fontsize=11)

    # Set radial limits
    ax.set_ylim(0, max_regret * 1.5)

    # Title
    if combined_regret is not None:
        title = f'Design Regret by Neighbor Direction{title_suffix}\nCombined (all 9): {combined_regret:.1f} GWh'
    else:
        title = f'Design Regret by Neighbor Direction{title_suffix}'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=40)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#d62728', edgecolor='black', label='High regret (>10 GWh)'),
        Patch(facecolor='#ff7f0e', edgecolor='black', label='Low regret (>0 GWh)'),
        Patch(facecolor='#2ca02c', edgecolor='black', label='No regret (0 GWh)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.1, 1.0))

    plt.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="Input JSON file")
    parser.add_argument("--output", "-o", required=True, help="Output PNG file")
    parser.add_argument("--title", "-t", default="", help="Title suffix")
    args = parser.parse_args()

    plot_polar_regret(args.input, args.output, args.title)

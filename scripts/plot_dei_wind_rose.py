"""Generate clean wind rose for DEI site."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

# Paths
DEI_DIR = Path(__file__).parent.parent / "OMAE_neighbors"
WIND_DATA_FILE = DEI_DIR / "energy_island_10y_daily_av_wind.csv"
OUTPUT_DIR = Path(__file__).parent.parent / "docs/figures"


def plot_wind_rose(output_file=None):
    """Create a clean wind rose from DEI wind data."""

    # Load data
    df = pd.read_csv(WIND_DATA_FILE, sep=';')
    wd = df['WD_150'].values
    ws = df['WS_150'].values

    print(f"Loaded {len(wd)} samples")
    print(f"Wind speed range: {ws.min():.1f} - {ws.max():.1f} m/s")
    print(f"Mean wind speed: {ws.mean():.1f} m/s")

    # Bin settings
    n_dir_bins = 16
    ws_bins = [0, 6, 9, 12, 15, 25]
    ws_labels = ['0-6', '6-9', '9-12', '12-15', '>15']
    colors = ['#eff3ff', '#bdd7e7', '#6baed6', '#3182bd', '#08519c']

    # Compute direction bins
    dir_bin_edges = np.linspace(0, 360, n_dir_bins + 1)
    dir_bin_width = 360 / n_dir_bins
    dir_centers = (dir_bin_edges[:-1] + dir_bin_edges[1:]) / 2

    # Compute frequencies for each direction and speed bin
    freq = np.zeros((n_dir_bins, len(ws_bins) - 1))

    for i in range(n_dir_bins):
        if i == n_dir_bins - 1:
            dir_mask = (wd >= dir_bin_edges[i]) | (wd < dir_bin_edges[0])
        else:
            dir_mask = (wd >= dir_bin_edges[i]) & (wd < dir_bin_edges[i + 1])

        for j in range(len(ws_bins) - 1):
            ws_mask = (ws >= ws_bins[j]) & (ws < ws_bins[j + 1])
            freq[i, j] = np.sum(dir_mask & ws_mask) / len(wd) * 100

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})

    # Convert to radians
    dir_centers_rad = np.radians(dir_centers)
    bar_width = np.radians(dir_bin_width) * 0.9

    # Plot stacked bars
    bottom = np.zeros(n_dir_bins)
    for j in range(len(ws_bins) - 1):
        ax.bar(dir_centers_rad, freq[:, j], width=bar_width, bottom=bottom,
               color=colors[j], edgecolor='white', linewidth=0.5,
               label=f'{ws_labels[j]} m/s')
        bottom += freq[:, j]

    # Configure axes
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    # Cardinal direction labels
    ax.set_xticks(np.radians([0, 45, 90, 135, 180, 225, 270, 315]))
    ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'], fontsize=14, fontweight='bold')

    # Radial labels (frequency %)
    max_freq = bottom.max()
    ax.set_ylim(0, max_freq * 1.1)
    ax.set_yticks(np.arange(0, max_freq, 5))
    ax.set_yticklabels([f'{int(y)}%' for y in np.arange(0, max_freq, 5)], fontsize=10)

    # Title and legend
    ax.set_title('DEI Wind Rose\n10-year daily data (2012-2021)', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0), title='Wind Speed', fontsize=11)

    # Add annotation for dominant direction
    dominant_idx = np.argmax(bottom)
    dominant_dir = dir_centers[dominant_idx]
    dominant_freq = bottom[dominant_idx]

    # Find secondary (southern) direction
    south_idx = np.argmin(np.abs(dir_centers - 180))
    south_freq = bottom[south_idx]

    ax.annotate(f'Dominant: {dominant_freq:.0f}%',
                xy=(np.radians(dominant_dir), dominant_freq),
                xytext=(np.radians(dominant_dir - 30), dominant_freq + 3),
                fontsize=11, fontweight='bold', color='#08519c',
                arrowprops=dict(arrowstyle='->', color='#08519c', lw=1.5))

    plt.tight_layout()

    if output_file is None:
        output_file = OUTPUT_DIR / 'dei_wind_rose.png'

    fig.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {output_file}")

    # Print summary
    print(f"\nDominant direction: {dominant_dir:.0f}° ({dominant_freq:.1f}%)")
    print(f"South (180°): {south_freq:.1f}%")


if __name__ == "__main__":
    plot_wind_rose()

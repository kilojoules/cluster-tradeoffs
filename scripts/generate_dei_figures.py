"""Generate clean figures for DEI case study documentation."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon
import h5py

# Paths
DEI_DIR = Path(__file__).parent.parent / "OMAE_neighbors"
RESULTS_FILE = Path(__file__).parent.parent / "analysis/dei_single_neighbor/dei_single_neighbor_bastankhah.json"
OUTPUT_DIR = Path(__file__).parent.parent / "docs/figures"
LAYOUTS_FILE = DEI_DIR / "re_precomputed_layouts.h5"
WIND_DATA_FILE = DEI_DIR / "energy_island_10y_daily_av_wind.csv"

# Farm boundaries (UTM coordinates)
FARM_BOUNDARIES = {
    0: np.array([  # Target farm (dk0w_tender_3)
        706694.3923283464, 6224158.532895836,
        703972.0844905999, 6226906.597455995,
        702624.6334635273, 6253853.5386425415,
        712771.6248419734, 6257704.934445341,
        715639.3355871611, 6260664.6846508905,
        721593.2420745814, 6257906.998015941,
    ]).reshape((-1, 2)),
    1: np.array([  # dk1d_tender_9 (SW)
        679052.6969498377, 6227229.556685498,
        676330.3891120912, 6229977.621245657,
        674982.9380850186, 6256924.562429089,
        685129.9294634647, 6260775.958231889,
        687997.6402086524, 6263735.708437439,
        693951.5466960727, 6260978.021802489,
    ]).reshape((-1, 2)),
    2: np.array([  # dk0z_tender_5 (W)
        685185.3969498377, 6238229.556685498,
        682463.0891120912, 6240977.621245657,
        681115.6380850186, 6267924.562429089,
        691262.6294634647, 6271775.958231889,
        694130.3402086524, 6274735.708437439,
        700084.2466960727, 6271978.021802489,
    ]).reshape((-1, 2)),
    3: np.array([  # dk0v_tender_1 (NW)
        695585.3969498377, 6268229.556685498,
        692863.0891120912, 6270977.621245657,
        691515.6380850186, 6297924.562429089,
        701662.6294634647, 6301775.958231889,
        704530.3402086524, 6304735.708437439,
        710484.2466960727, 6301978.021802489,
    ]).reshape((-1, 2)),
    4: np.array([  # dk0Y_tender_4 (N)
        715585.3969498377, 6288229.556685498,
        712863.0891120912, 6290977.621245657,
        711515.6380850186, 6317924.562429089,
        721662.6294634647, 6321775.958231889,
        724530.3402086524, 6324735.708437439,
        730484.2466960727, 6321978.021802489,
    ]).reshape((-1, 2)),
    5: np.array([  # dk0x_tender_2 (NE)
        725585.3969498377, 6268229.556685498,
        722863.0891120912, 6270977.621245657,
        721515.6380850186, 6297924.562429089,
        731662.6294634647, 6301775.958231889,
        734530.3402086524, 6304735.708437439,
        740484.2466960727, 6301978.021802489,
    ]).reshape((-1, 2)),
    6: np.array([  # dk1a_tender_6 (E)
        745585.3969498377, 6258229.556685498,
        742863.0891120912, 6260977.621245657,
        741515.6380850186, 6287924.562429089,
        751662.6294634647, 6291775.958231889,
        754530.3402086524, 6294735.708437439,
        760484.2466960727, 6291978.021802489,
    ]).reshape((-1, 2)),
    7: np.array([  # dk1b_tender7 (SE)
        735585.3969498377, 6228229.556685498,
        732863.0891120912, 6230977.621245657,
        731515.6380850186, 6257924.562429089,
        741662.6294634647, 6261775.958231889,
        744530.3402086524, 6264735.708437439,
        750484.2466960727, 6261978.021802489,
    ]).reshape((-1, 2)),
    8: np.array([  # dk1c_tender_8 (S) - THE PROBLEM NEIGHBOR
        715585.3969498377, 6178229.556685498,
        712863.0891120912, 6180977.621245657,
        711515.6380850186, 6207924.562429089,
        721662.6294634647, 6211775.958231889,
        724530.3402086524, 6214735.708437439,
        730484.2466960727, 6211978.021802489,
    ]).reshape((-1, 2)),
    9: np.array([  # dk1e_tender_10 (SSW)
        695585.3969498377, 6168229.556685498,
        692863.0891120912, 6170977.621245657,
        691515.6380850186, 6197924.562429089,
        701662.6294634647, 6201775.958231889,
        704530.3402086524, 6204735.708437439,
        710484.2466960727, 6201978.021802489,
    ]).reshape((-1, 2)),
}


def load_results():
    """Load single-neighbor analysis results."""
    with open(RESULTS_FILE) as f:
        return json.load(f)


def load_layouts():
    """Load turbine layouts from HDF5."""
    layouts = {}
    with h5py.File(LAYOUTS_FILE, 'r') as f:
        for farm_idx in range(10):
            key = f"farm{farm_idx}_t5_s0"
            if key in f:
                layout = f[key]['layout'][:]
                layouts[farm_idx] = (layout[0], layout[1])
    return layouts


def load_wind_data():
    """Load wind time series."""
    df = pd.read_csv(WIND_DATA_FILE, sep=';')
    return df['WD_150'].values, df['WS_150'].values


def figure_1_cluster_map():
    """Create clean cluster map highlighting Farm 8 with inset wind rose."""
    fig = plt.figure(figsize=(12, 10))

    # Main map axis
    ax = fig.add_axes([0.1, 0.1, 0.75, 0.85])

    # Inset wind rose axis (polar)
    ax_rose = fig.add_axes([0.72, 0.65, 0.25, 0.30], projection='polar')

    layouts = load_layouts()
    results = load_results()

    # Colors
    target_color = '#3498db'  # Blue
    neighbor_color = '#95a5a6'  # Gray
    problem_color = '#e74c3c'  # Red for Farm 8

    # Plot neighbor farms
    for farm_idx in range(1, 10):
        if farm_idx in layouts:
            x, y = layouts[farm_idx]
            color = problem_color if farm_idx == 8 else neighbor_color
            alpha = 1.0 if farm_idx == 8 else 0.5
            size = 30 if farm_idx == 8 else 15
            ax.scatter(x/1000, y/1000, c=color, s=size, alpha=alpha, zorder=2)

            # Add farm label
            cx, cy = np.mean(x)/1000, np.mean(y)/1000
            label = f"{farm_idx}"
            if farm_idx == 8:
                label = "8\n(S)"
            fontweight = 'bold' if farm_idx == 8 else 'normal'
            fontsize = 14 if farm_idx == 8 else 10
            ax.annotate(label, (cx, cy), fontsize=fontsize, fontweight=fontweight,
                       ha='center', va='center', color='white' if farm_idx == 8 else 'black',
                       bbox=dict(boxstyle='circle,pad=0.3', fc=color, ec='none', alpha=0.8))

    # Plot target farm
    if 0 in layouts:
        x, y = layouts[0]
        ax.scatter(x/1000, y/1000, c=target_color, s=40, alpha=1.0, zorder=3,
                  edgecolors='white', linewidths=0.5)
        cx, cy = np.mean(x)/1000, np.mean(y)/1000
        ax.annotate("Target\n(dk0w)", (cx, cy+3), fontsize=12, fontweight='bold',
                   ha='center', va='bottom', color=target_color)

    # Legend
    legend_elements = [
        plt.scatter([], [], c=target_color, s=60, label='Target Farm (66 turbines)'),
        plt.scatter([], [], c=neighbor_color, s=30, alpha=0.5, label='Neighbor (no regret)'),
        plt.scatter([], [], c=problem_color, s=50, label='Farm 8 (101 GWh regret)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

    ax.set_xlabel('Easting (km)', fontsize=12)
    ax.set_ylabel('Northing (km)', fontsize=12)
    ax.set_title('Danish Energy Island Cluster', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Set axis limits
    ax.set_xlim(670, 770)
    ax.set_ylim(6165, 6330)

    # --- Inset wind rose ---
    wd, ws = load_wind_data()

    # Bin the wind data
    n_bins = 16
    bin_edges = np.linspace(0, 360, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    counts = np.zeros(n_bins)
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (wd >= bin_edges[i]) | (wd < bin_edges[0])
        else:
            mask = (wd >= bin_edges[i]) & (wd < bin_edges[i+1])
        counts[i] = mask.sum()

    freq = counts / counts.sum() * 100

    # Convert to radians (meteorological: direction wind is FROM)
    # In polar plot: 0 is right (E), angles go counter-clockwise
    # We want: 0 at top (N), clockwise
    theta = np.radians(bin_centers)
    width = 2 * np.pi / n_bins * 0.8

    # Color bars by frequency
    colors = plt.cm.Blues(freq / freq.max())

    ax_rose.bar(theta, freq, width=width, color=colors, edgecolor='white', linewidth=0.5)
    ax_rose.set_theta_zero_location('N')
    ax_rose.set_theta_direction(-1)
    ax_rose.set_title('Wind Rose', fontsize=10, fontweight='bold', pad=8)
    ax_rose.set_yticklabels([])
    ax_rose.tick_params(axis='x', labelsize=8)

    fig.savefig(OUTPUT_DIR / 'dei_cluster_map.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR / 'dei_cluster_map.png'}")


def figure_2_regret_bar():
    """Create clean bar chart of regret by neighbor."""
    results = load_results()

    fig, ax = plt.subplots(figsize=(10, 5))

    farms = []
    regrets = []
    directions = []
    colors = []

    for i in range(1, 10):
        r = results[str(i)]
        farms.append(f"Farm {i}")
        regrets.append(r['regret_gwh'])
        directions.append(r['direction'])
        colors.append('#e74c3c' if r['regret_gwh'] > 1 else '#95a5a6')

    bars = ax.bar(farms, regrets, color=colors, edgecolor='black', linewidth=0.5)

    # Add direction labels on bars
    for bar, d, r in zip(bars, directions, regrets):
        direction_label = f"{d:.0f}°"
        if r > 1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                   f'{r:.1f} GWh\n({direction_label})', ha='center', va='bottom',
                   fontsize=10, fontweight='bold')
        else:
            ax.text(bar.get_x() + bar.get_width()/2, 1,
                   direction_label, ha='center', va='bottom',
                   fontsize=8, color='gray')

    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_ylabel('Design Regret (GWh/year)', fontsize=12)
    ax.set_xlabel('Neighboring Farm', fontsize=12)
    ax.set_title('Regret by Individual Neighbor', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 120)

    # Add annotation
    ax.annotate('Only Farm 8 (South)\ncauses regret',
                xy=(7, 100), xytext=(4, 100),
                fontsize=11, ha='center',
                arrowprops=dict(arrowstyle='->', color='#e74c3c'))

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'dei_regret_by_neighbor.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR / 'dei_regret_by_neighbor.png'}")


def figure_3_pareto_farm8():
    """Create Pareto plot for Farm 8 showing the tradeoff."""
    results = load_results()
    r = results['8']

    fig, ax = plt.subplots(figsize=(8, 6))

    # Extract key points
    lib_absent = r['lib_opt_absent']
    lib_present = r['lib_opt_present']
    con_absent = r['con_opt_absent']
    con_present = r['con_opt_present']
    regret = r['regret_gwh']

    # Plot the two optimal points
    ax.scatter([lib_absent], [lib_present], s=150, c='#3498db', marker='o',
              label=f'Liberal-optimal', zorder=5, edgecolors='white', linewidths=2)
    ax.scatter([con_absent], [con_present], s=150, c='#e74c3c', marker='s',
              label=f'Conservative-optimal', zorder=5, edgecolors='white', linewidths=2)

    # Draw Pareto front line
    ax.plot([lib_absent, con_absent], [lib_present, con_present],
            'k--', linewidth=2, alpha=0.7, zorder=4, label='Pareto front')

    # Annotate regret
    mid_x = (lib_absent + con_absent) / 2
    ax.annotate('', xy=(mid_x, con_present), xytext=(mid_x, lib_present),
                arrowprops=dict(arrowstyle='<->', color='#e74c3c', lw=2))
    ax.text(mid_x + 2, (lib_present + con_present)/2,
            f'Regret\n{regret:.0f} GWh', fontsize=11, fontweight='bold',
            color='#e74c3c', va='center')

    # Add annotations for each point
    ax.annotate(f'AEP alone: {lib_absent:.0f}\nAEP w/ Farm 8: {lib_present:.0f}',
                xy=(lib_absent, lib_present), xytext=(lib_absent-30, lib_present-30),
                fontsize=9, ha='right',
                arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))
    ax.annotate(f'AEP alone: {con_absent:.0f}\nAEP w/ Farm 8: {con_present:.0f}',
                xy=(con_absent, con_present), xytext=(con_absent+5, con_present+30),
                fontsize=9, ha='left',
                arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))

    ax.set_xlabel('AEP without neighbors (GWh/year)', fontsize=12)
    ax.set_ylabel('AEP with Farm 8 (GWh/year)', fontsize=12)
    ax.set_title('Pareto Tradeoff: Farm 8 (South, 163°)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Set reasonable axis limits
    ax.set_xlim(8280, 8340)
    ax.set_ylim(8170, 8320)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'dei_pareto_farm8.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR / 'dei_pareto_farm8.png'}")


def figure_4_wind_rose():
    """Create compact wind rose."""
    wd, ws = load_wind_data()

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={'projection': 'polar'})

    # Bin the wind data
    n_bins = 16
    bin_edges = np.linspace(0, 360, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    counts = np.zeros(n_bins)
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (wd >= bin_edges[i]) | (wd < bin_edges[0])
        else:
            mask = (wd >= bin_edges[i]) & (wd < bin_edges[i+1])
        counts[i] = mask.sum()

    freq = counts / counts.sum() * 100

    # Convert to radians (meteorological convention: 0=N, 90=E)
    theta = np.radians(90 - bin_centers)  # Convert to math convention
    width = 2 * np.pi / n_bins * 0.8

    # Color bars by frequency
    colors = plt.cm.Blues(freq / freq.max())

    bars = ax.bar(theta, freq, width=width, color=colors, edgecolor='white', linewidth=0.5)

    # Highlight dominant direction (W ~270°) and secondary (S ~180°)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    ax.set_title('Wind Rose\n(10-year data)', fontsize=11, fontweight='bold', pad=10)
    ax.set_ylim(0, max(freq) * 1.1)
    ax.set_yticklabels([])

    # Add key annotations
    ax.annotate('W: 20%', xy=(np.radians(270), max(freq)*0.5), fontsize=9, ha='center')
    ax.annotate('S: 4%', xy=(np.radians(180), max(freq)*0.3), fontsize=9, ha='center', color='#e74c3c')

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'dei_wind_rose.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR / 'dei_wind_rose.png'}")


def figure_5_ambush_diagram():
    """Create diagram explaining the ambush effect."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Western neighbor (no regret)
    ax = axes[0]
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

    # Target farm
    target = plt.Circle((0, 0), 0.3, color='#3498db', label='Target')
    ax.add_patch(target)
    ax.text(0, 0, 'T', ha='center', va='center', color='white', fontweight='bold')

    # Western neighbor
    west = plt.Circle((-1.2, 0), 0.2, color='#95a5a6', label='Western\nneighbor')
    ax.add_patch(west)
    ax.text(-1.2, 0, 'W', ha='center', va='center', color='white', fontsize=9)

    # Wind arrow from west
    ax.annotate('', xy=(0.5, 0), xytext=(-0.5, 0),
                arrowprops=dict(arrowstyle='->', color='black', lw=3))
    ax.text(0, 0.5, 'Dominant wind', ha='center', fontsize=10)

    # Layout already optimized for this
    ax.text(0, -1.5, 'Layout already optimized\nfor westerly wakes',
            ha='center', fontsize=11, style='italic')
    ax.text(0, -1.9, 'Regret: 0 GWh', ha='center', fontsize=12,
            fontweight='bold', color='#27ae60')

    ax.set_aspect('equal')
    ax.set_title('Western Neighbor (262°)', fontsize=12, fontweight='bold')
    ax.axis('off')

    # Right: Southern neighbor (high regret)
    ax = axes[1]
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

    # Target farm
    target = plt.Circle((0, 0), 0.3, color='#3498db')
    ax.add_patch(target)
    ax.text(0, 0, 'T', ha='center', va='center', color='white', fontweight='bold')

    # Southern neighbor
    south = plt.Circle((0, -1.2), 0.2, color='#e74c3c')
    ax.add_patch(south)
    ax.text(0, -1.2, 'S', ha='center', va='center', color='white', fontsize=9)

    # Wind arrow from south (secondary)
    ax.annotate('', xy=(0, 0.5), xytext=(0, -0.5),
                arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=3, ls='--'))
    ax.text(0.5, 0, 'Secondary\nwind (4%)', ha='left', fontsize=10, color='#e74c3c')

    # Layout NOT optimized for this
    ax.text(0, 1.5, 'Layout ignores southern wakes\n→ "ambush" when S wind occurs',
            ha='center', fontsize=11, style='italic')
    ax.text(0, 1.9, 'Regret: 101 GWh', ha='center', fontsize=12,
            fontweight='bold', color='#e74c3c')

    ax.set_aspect('equal')
    ax.set_title('Southern Neighbor (163°)', fontsize=12, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'dei_ambush_effect.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR / 'dei_ambush_effect.png'}")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating DEI case study figures...")
    figure_1_cluster_map()
    figure_2_regret_bar()
    figure_3_pareto_farm8()
    figure_4_wind_rose()
    figure_5_ambush_diagram()
    print("Done!")

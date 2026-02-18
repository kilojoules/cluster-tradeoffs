"""Plot best liberal and conservative layouts from combined case at different A values.

Grid: rows = A values, columns = [Liberal, Conservative]
Shows polygon boundary + all 9 neighbor farms.
"""

from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt

DEI_DIR = Path(__file__).parent.parent
LAYOUTS_FILE = DEI_DIR / "re_precomputed_layouts.h5"
OUTPUT_DIR = DEI_DIR / "docs" / "figures"

FARM_COLORS = {
    1: '#8B4513', 2: '#FF6347', 3: '#228B22', 4: '#4169E1', 5: '#FF8C00',
    6: '#9932CC', 7: '#20B2AA', 8: '#DC143C', 9: '#708090',
}
FARM_DIRS = {
    1: 'SW', 2: 'W', 3: 'NW', 4: 'N', 5: 'NE', 6: 'E', 7: 'SE', 8: 'S', 9: 'SSW',
}


def load_target_boundary():
    return np.array([
        706694.3923283464, 6224158.532895836,
        703972.0844905999, 6226906.597455995,
        702624.6334635273, 6253853.5386425415,
        712771.6248419734, 6257704.934445341,
        715639.3355871611, 6260664.6846508905,
        721593.2420745814, 6257906.998015941,
    ]).reshape((-1, 2)).T


def load_all_neighbors():
    """Load all 9 neighbor farm layouts."""
    neighbors = {}
    with h5py.File(LAYOUTS_FILE, 'r') as f:
        for farm_idx in range(1, 10):
            key = f"farm{farm_idx}_t5_s0"
            if key in f:
                layout = f[key]['layout'][:]
                neighbors[farm_idx] = (layout[0], layout[1])
    return neighbors


def find_best_layout(h5_path, strategy, metric):
    """Find best layout by metric. Returns (x, y, attrs) or (None, None, None)."""
    best_aep = -np.inf
    best_x, best_y, best_attrs = None, None, None
    try:
        with h5py.File(h5_path, 'r') as f:
            for key in f.keys():
                if not key.startswith('layout_'):
                    continue
                grp = f[key]
                if str(grp.attrs['strategy']) != strategy:
                    continue
                aep = float(grp.attrs[metric])
                if aep > best_aep:
                    best_aep = aep
                    best_x = np.array(grp['x'][:])
                    best_y = np.array(grp['y'][:])
                    best_attrs = {
                        'aep_absent': float(grp.attrs['aep_absent']),
                        'aep_present': float(grp.attrs['aep_present']),
                        'seed': int(grp.attrs['seed']),
                    }
    except Exception:
        pass
    return best_x, best_y, best_attrs


def plot_layout(ax, x, y, boundary, neighbors, title, color):
    """Plot turbine layout with polygon boundary and all neighbor farms."""
    # Polygon boundary
    poly_x = np.append(boundary[0], boundary[0, 0])
    poly_y = np.append(boundary[1], boundary[1, 0])
    ax.plot(poly_x / 1e3, poly_y / 1e3, 'k-', linewidth=1.5, zorder=3)
    ax.fill(poly_x / 1e3, poly_y / 1e3, alpha=0.04, color='gray')

    # Neighbor farms
    for farm_idx, (xn, yn) in neighbors.items():
        ax.scatter(xn / 1e3, yn / 1e3, s=8, c=FARM_COLORS[farm_idx], alpha=0.4,
                   marker='^', label=f'Farm {farm_idx} ({FARM_DIRS[farm_idx]})', zorder=2)

    # Target turbines
    ax.scatter(x / 1e3, y / 1e3, s=20, c=color, edgecolors='k',
               linewidths=0.3, zorder=4)

    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)


def main():
    boundary = load_target_boundary()
    neighbors = load_all_neighbors()

    # Find available A values with combined data
    A_values = []
    for A in ['0.04', '0.08', '0.12']:
        h5 = DEI_DIR / f"analysis/dei_A{A}/layouts_combined.h5"
        if h5.exists():
            try:
                with h5py.File(h5, 'r') as f:
                    n = len([k for k in f.keys() if k.startswith('layout_')])
                if n > 0:
                    A_values.append(A)
                    print(f"A={A}: {n//2} seeds available")
            except:
                pass

    if not A_values:
        print("No combined case data available yet!")
        return

    n_rows = len(A_values)
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 5.5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # Axis limits from boundary
    pad = 8  # km — enough to show nearby neighbor farms
    xmin = boundary[0].min() / 1e3 - pad
    xmax = boundary[0].max() / 1e3 + pad
    ymin = boundary[1].min() / 1e3 - pad
    ymax = boundary[1].max() / 1e3 + pad

    for row, A in enumerate(A_values):
        h5 = DEI_DIR / f"analysis/dei_A{A}/layouts_combined.h5"

        # Best liberal (maximize aep_absent)
        x_lib, y_lib, attrs_lib = find_best_layout(h5, 'liberal', 'aep_absent')
        # Best conservative (maximize aep_present)
        x_con, y_con, attrs_con = find_best_layout(h5, 'conservative', 'aep_present')

        if x_lib is not None:
            title_lib = (f"Liberal (A={A})\n"
                         f"AEP absent={attrs_lib['aep_absent']:.0f}, "
                         f"present={attrs_lib['aep_present']:.0f} GWh")
            plot_layout(axes[row, 0], x_lib, y_lib, boundary, neighbors,
                        title_lib, '#3498db')
            print(f"  Liberal  A={A}: seed={attrs_lib['seed']}, "
                  f"absent={attrs_lib['aep_absent']:.1f}, present={attrs_lib['aep_present']:.1f}")

        if x_con is not None:
            title_con = (f"Conservative (A={A})\n"
                         f"AEP absent={attrs_con['aep_absent']:.0f}, "
                         f"present={attrs_con['aep_present']:.0f} GWh")
            plot_layout(axes[row, 1], x_con, y_con, boundary, neighbors,
                        title_con, '#e74c3c')
            print(f"  Conserv. A={A}: seed={attrs_con['seed']}, "
                  f"absent={attrs_con['aep_absent']:.1f}, present={attrs_con['aep_present']:.1f}")

        for col in range(2):
            axes[row, col].set_xlim(xmin, xmax)
            axes[row, col].set_ylim(ymin, ymax)
            if col == 0:
                axes[row, col].set_ylabel('Northing (km)')

    # Bottom row x-labels
    for col in range(2):
        axes[-1, col].set_xlabel('Easting (km)')

    # Legend on first panel
    axes[0, 0].legend(loc='lower left', fontsize=6, ncol=3, markerscale=1.5)

    fig.suptitle('Best layouts: Combined case (all 9 neighbors, polygon constraint)',
                 fontsize=14, fontweight='bold', y=1.0)
    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "combined_layouts_by_A.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

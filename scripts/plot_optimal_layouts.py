"""Plot optimal turbine positions with polygon boundary constraint."""

from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt

DEI_DIR = Path(__file__).parent.parent
LAYOUTS_FILE = DEI_DIR / "re_precomputed_layouts.h5"
OUTPUT_DIR = DEI_DIR / "docs" / "figures"


def load_target_boundary():
    return np.array([
        706694.3923283464, 6224158.532895836,
        703972.0844905999, 6226906.597455995,
        702624.6334635273, 6253853.5386425415,
        712771.6248419734, 6257704.934445341,
        715639.3355871611, 6260664.6846508905,
        721593.2420745814, 6257906.998015941,
    ]).reshape((-1, 2)).T


def load_neighbor_layout(farm_idx):
    with h5py.File(LAYOUTS_FILE, 'r') as f:
        key = f"farm{farm_idx}_t5_s0"
        if key in f:
            layout = f[key]['layout'][:]
            return layout[0], layout[1]
    return None, None


def find_best_layout(h5_path, strategy, metric):
    """Find best layout of given strategy by metric ('aep_present' or 'aep_absent')."""
    best_aep = -np.inf
    best_x, best_y = None, None
    best_attrs = {}
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
    return best_x, best_y, best_attrs


def plot_layout(ax, x, y, boundary, x_neigh, y_neigh, title, color,
                show_neighbor=True, neighbor_label=None):
    """Plot a single layout panel with polygon boundary."""
    D = 240  # rotor diameter

    # Draw polygon boundary (closed)
    poly_x = np.append(boundary[0], boundary[0, 0])
    poly_y = np.append(boundary[1], boundary[1, 0])
    ax.plot(poly_x / 1e3, poly_y / 1e3, 'k-', linewidth=1.5, zorder=3)
    ax.fill(poly_x / 1e3, poly_y / 1e3, alpha=0.05, color='gray')

    # Also show old bounding box for reference (dashed)
    x_min = boundary[0].min() + D / 2
    x_max = boundary[0].max() - D / 2
    y_min = boundary[1].min() + D / 2
    y_max = boundary[1].max() - D / 2
    rect_x = np.array([x_min, x_max, x_max, x_min, x_min])
    rect_y = np.array([y_min, y_min, y_max, y_max, y_min])
    ax.plot(rect_x / 1e3, rect_y / 1e3, 'r--', linewidth=0.8, alpha=0.4,
            zorder=2, label='Old bounding box')

    # Neighbor turbines
    if show_neighbor and x_neigh is not None:
        ax.scatter(x_neigh / 1e3, y_neigh / 1e3, s=12, c='gray', alpha=0.5,
                   marker='^', label=neighbor_label, zorder=2)

    # Target turbines
    ax.scatter(x / 1e3, y / 1e3, s=25, c=color, edgecolors='k',
               linewidths=0.3, zorder=4)

    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Easting (km)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)


def main():
    boundary = load_target_boundary()
    analysis_dir = DEI_DIR / "analysis" / "dei_A0.04_polygon"

    # Best conservative for farm 8 (polygon-constrained)
    x_con8, y_con8, attrs_con8 = find_best_layout(
        analysis_dir / "layouts_farm8.h5", 'conservative', 'aep_present')
    x_neigh8, y_neigh8 = load_neighbor_layout(8)

    # Best liberal (polygon-constrained)
    x_lib, y_lib, attrs_lib = find_best_layout(
        analysis_dir / "layouts_farm8.h5", 'liberal', 'aep_absent')

    print(f"Conservative Farm 8: seed={attrs_con8['seed']}, "
          f"AEP_absent={attrs_con8['aep_absent']:.1f}, AEP_present={attrs_con8['aep_present']:.1f}")
    print(f"Liberal:             seed={attrs_lib['seed']}, "
          f"AEP_absent={attrs_lib['aep_absent']:.1f}, AEP_present={attrs_lib['aep_present']:.1f}")

    # Compute axis limits from boundary with padding
    pad = 2  # km
    xmin = boundary[0].min() / 1e3 - pad
    xmax = boundary[0].max() / 1e3 + pad
    ymin = boundary[1].min() / 1e3 - pad
    ymax = boundary[1].max() / 1e3 + pad

    fig, axes = plt.subplots(1, 2, figsize=(12, 8))

    plot_layout(axes[0], x_con8, y_con8, boundary, x_neigh8, y_neigh8,
                f"Conservative (Farm 8 — S)\nAEP_present = {attrs_con8['aep_present']:.0f} GWh/yr",
                color='#e74c3c', neighbor_label='Farm 8 turbines')

    plot_layout(axes[1], x_lib, y_lib, boundary, x_neigh8, y_neigh8,
                f"Liberal (no neighbors)\nAEP_absent = {attrs_lib['aep_absent']:.0f} GWh/yr",
                color='#3498db', neighbor_label='Farm 8 turbines')

    for ax in axes:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.legend(loc='lower right', fontsize=8)

    axes[0].set_ylabel('Northing (km)')

    fig.suptitle('Polygon-constrained layouts (A = 0.04, Farm 8, seed 0)',
                 fontsize=14, fontweight='bold', y=0.98)
    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "polygon_test_layouts.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

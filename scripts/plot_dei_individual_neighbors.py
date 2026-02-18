"""Generate individual neighbor Pareto scatter plots from HDF5 data.

3x3 grid: one panel per neighbor farm (A=0.04, polygon-constrained).
X-axis: AEP without neighbor | Y-axis: AEP with neighbor
"""

from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt

ANALYSIS_DIR = Path(__file__).parent.parent / "analysis" / "dei_A0.04"
OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "figures"

FARM_NAMES = {
    1: "dk1d_tender_9 (SW)",
    2: "dk0z_tender_5 (W)",
    3: "dk0v_tender_1 (NW)",
    4: "dk0Y_tender_4 (N)",
    5: "dk0x_tender_2 (NE)",
    6: "dk1a_tender_6 (E)",
    7: "dk1b_tender7 (SE)",
    8: "dk1c_tender_8 (S)",
    9: "dk1e_tender_10 (SSW)",
}


def load_layouts(h5_path):
    """Load all layouts from HDF5 file. Returns list of dicts."""
    layouts = []
    with h5py.File(h5_path, 'r') as f:
        for key in f.keys():
            if not key.startswith('layout_'):
                continue
            grp = f[key]
            layouts.append({
                'aep_absent': float(grp.attrs['aep_absent']),
                'aep_present': float(grp.attrs['aep_present']),
                'strategy': str(grp.attrs['strategy']),
                'seed': int(grp.attrs['seed']),
            })
    return layouts


def compute_pareto(absent, present):
    """Find Pareto-optimal points (maximizing both objectives)."""
    n = len(absent)
    pareto_mask = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j:
                if (absent[j] >= absent[i] and present[j] >= present[i] and
                        (absent[j] > absent[i] or present[j] > present[i])):
                    pareto_mask[i] = False
                    break
    return pareto_mask


def plot_farm(ax, farm_idx, layouts):
    """Plot Pareto scatter for a single neighbor farm."""
    absent = np.array([l['aep_absent'] for l in layouts])
    present = np.array([l['aep_present'] for l in layouts])
    strategies = [l['strategy'] for l in layouts]

    lib_mask = np.array([s == 'liberal' for s in strategies])
    con_mask = np.array([s == 'conservative' for s in strategies])

    pareto_mask = compute_pareto(absent, present)
    n_pareto = pareto_mask.sum()

    # Regret
    pareto_idx = np.where(pareto_mask)[0]
    if len(pareto_idx) > 0:
        lib_opt_idx = pareto_idx[np.argmax(absent[pareto_mask])]
        con_opt_idx = pareto_idx[np.argmax(present[pareto_mask])]
        regret = present[con_opt_idx] - present[lib_opt_idx]
    else:
        regret = 0

    # Dominated points
    ax.scatter(absent[lib_mask & ~pareto_mask], present[lib_mask & ~pareto_mask],
               s=30, c='#3498db', alpha=0.4, marker='o', label='Liberal')
    ax.scatter(absent[con_mask & ~pareto_mask], present[con_mask & ~pareto_mask],
               s=30, c='#e74c3c', alpha=0.4, marker='s', label='Conservative')

    # Pareto points
    ax.scatter(absent[lib_mask & pareto_mask], present[lib_mask & pareto_mask],
               s=100, c='#3498db', alpha=1.0, marker='o',
               edgecolors='black', linewidths=1.5, zorder=5)
    ax.scatter(absent[con_mask & pareto_mask], present[con_mask & pareto_mask],
               s=100, c='#e74c3c', alpha=1.0, marker='s',
               edgecolors='black', linewidths=1.5, zorder=5)

    # Pareto front line
    if n_pareto > 1:
        pa = absent[pareto_mask]
        pp = present[pareto_mask]
        sort_idx = np.argsort(pa)
        ax.plot(pa[sort_idx], pp[sort_idx], 'k--', linewidth=1.5, alpha=0.7, zorder=4)

    # Title
    name = FARM_NAMES.get(farm_idx, f"Farm {farm_idx}")
    direction_part = name.split('(')[-1].rstrip(')') if '(' in name else ''
    n_seeds = len(set(l['seed'] for l in layouts))
    title_color = '#e74c3c' if regret > 1 else '#333333'
    ax.set_title(f'Farm {farm_idx}: {direction_part}\n'
                 f'Regret: {regret:.1f} GWh | {n_seeds} seeds | {n_pareto} Pareto pts',
                 fontsize=10, fontweight='bold', color=title_color)

    ax.set_xlabel('AEP without neighbor (GWh)', fontsize=9)
    ax.set_ylabel('AEP with neighbor (GWh)', fontsize=9)
    ax.grid(True, alpha=0.3)

    x_range = absent.max() - absent.min()
    y_range = present.max() - present.min()
    if x_range > 0:
        ax.set_xlim(absent.min() - x_range * 0.1, absent.max() + x_range * 0.1)
    if y_range > 0:
        ax.set_ylim(present.min() - y_range * 0.1, present.max() + y_range * 0.1)

    return n_pareto, regret


def main():
    fig, axes = plt.subplots(3, 3, figsize=(14, 14))
    axes = axes.flatten()

    for idx, farm_idx in enumerate(range(1, 10)):
        ax = axes[idx]
        h5_path = ANALYSIS_DIR / f"layouts_farm{farm_idx}.h5"

        if not h5_path.exists():
            ax.text(0.5, 0.5, f'Farm {farm_idx}\nNo data',
                    ha='center', va='center', fontsize=12, transform=ax.transAxes)
            continue

        layouts = load_layouts(h5_path)
        if not layouts:
            ax.text(0.5, 0.5, f'Farm {farm_idx}\nNo data',
                    ha='center', va='center', fontsize=12, transform=ax.transAxes)
            continue

        n_pareto, regret = plot_farm(ax, farm_idx, layouts)
        print(f"Farm {farm_idx}: {len(layouts)//2} seeds, {n_pareto} Pareto pts, {regret:.1f} GWh regret")

        if idx == 0:
            ax.legend(loc='lower left', fontsize=8)

    plt.suptitle('Individual Neighbor Analysis: Pareto Frontiers (A = 0.04, polygon constraint)\n'
                 'Blue = liberal (ignore neighbor), Red = conservative (account for neighbor)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "dei_A0.04_individual.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

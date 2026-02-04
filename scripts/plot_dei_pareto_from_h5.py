"""Plot DEI Pareto frontier from existing HDF5 layout data."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import h5py

# Paths
OUTPUT_DIR = Path(__file__).parent.parent / "docs/figures"
LAYOUTS_FILE = Path(__file__).parent.parent / "analysis/dei_full_ts/layouts_farm8.h5"


def main():
    print(f"Loading layouts from {LAYOUTS_FILE}...")

    with h5py.File(LAYOUTS_FILE, 'r') as f:
        n_layouts = f.attrs['n_layouts']
        print(f"Found {n_layouts} layouts")

        absent = []
        present = []
        strategies = []

        for i in range(n_layouts):
            layout = f[f'layout_{i}']
            absent.append(layout.attrs['aep_absent'])
            present.append(layout.attrs['aep_present'])
            strategies.append(layout.attrs['strategy'])

    absent = np.array(absent)
    present = np.array(present)

    # Find Pareto front
    pareto_mask = np.zeros(len(absent), dtype=bool)
    for i in range(len(absent)):
        dominated = False
        for j in range(len(absent)):
            if i != j:
                if (absent[j] >= absent[i] and present[j] > present[i]) or \
                   (absent[j] > absent[i] and present[j] >= present[i]):
                    dominated = True
                    break
        if not dominated:
            pareto_mask[i] = True

    # Find liberal-optimal and conservative-optimal
    pareto_idx = np.where(pareto_mask)[0]
    lib_opt_idx = pareto_idx[np.argmax(absent[pareto_mask])]
    con_opt_idx = pareto_idx[np.argmax(present[pareto_mask])]
    regret = present[con_opt_idx] - present[lib_opt_idx]

    print(f"Pareto points: {pareto_mask.sum()}")
    print(f"Liberal-optimal: absent={absent[lib_opt_idx]:.1f}, present={present[lib_opt_idx]:.1f}")
    print(f"Conservative-optimal: absent={absent[con_opt_idx]:.1f}, present={present[con_opt_idx]:.1f}")
    print(f"Regret: {regret:.1f} GWh")

    # --- Create figure ---
    fig, ax = plt.subplots(figsize=(8, 8))

    lib_mask = np.array([s == 'liberal' for s in strategies])
    con_mask = np.array([s == 'conservative' for s in strategies])

    # Dominated points
    ax.scatter(absent[lib_mask & ~pareto_mask], present[lib_mask & ~pareto_mask],
               s=100, c='#3498db', alpha=0.5, label='Liberal (dominated)', marker='o')
    ax.scatter(absent[con_mask & ~pareto_mask], present[con_mask & ~pareto_mask],
               s=100, c='#e74c3c', alpha=0.5, label='Conservative (dominated)', marker='s')

    # Pareto points
    ax.scatter(absent[lib_mask & pareto_mask], present[lib_mask & pareto_mask],
               s=200, c='#3498db', alpha=1.0, label='Liberal (Pareto)', marker='o',
               edgecolors='black', linewidths=2, zorder=5)
    ax.scatter(absent[con_mask & pareto_mask], present[con_mask & pareto_mask],
               s=200, c='#e74c3c', alpha=1.0, label='Conservative (Pareto)', marker='s',
               edgecolors='black', linewidths=2, zorder=5)

    # Pareto front line
    pareto_absent = absent[pareto_mask]
    pareto_present = present[pareto_mask]
    sort_idx = np.argsort(pareto_absent)
    ax.plot(pareto_absent[sort_idx], pareto_present[sort_idx], 'k--', linewidth=2, alpha=0.7, zorder=4)

    # Regret annotation
    mid_x = absent[lib_opt_idx] - 2
    ax.annotate('', xy=(mid_x, present[con_opt_idx]),
                xytext=(mid_x, present[lib_opt_idx]),
                arrowprops=dict(arrowstyle='<->', color='#e74c3c', lw=2.5))
    ax.text(mid_x - 1.5, (present[lib_opt_idx] + present[con_opt_idx])/2,
            f'Regret\n{regret:.0f} GWh', fontsize=11, fontweight='bold',
            color='#e74c3c', va='center', ha='right')

    # Axis limits with padding
    x_range = absent.max() - absent.min()
    y_range = present.max() - present.min()
    ax.set_xlim(absent.min() - x_range * 0.15, absent.max() + x_range * 0.15)
    ax.set_ylim(present.min() - y_range * 0.15, present.max() + y_range * 0.15)

    ax.set_xlabel('AEP without Farm 8 (GWh/year)', fontsize=12)
    ax.set_ylabel('AEP with Farm 8 (GWh/year)', fontsize=12)
    ax.set_title(f'Pareto Front: Farm 8 (South, 163°)\n{pareto_mask.sum()} Pareto points from {n_layouts} optimizations (50 starts × 2 strategies)', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / 'dei_pareto_farm8.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"\nSaved: {OUTPUT_DIR / 'dei_pareto_farm8.png'}")


if __name__ == "__main__":
    main()

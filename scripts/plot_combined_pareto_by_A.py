"""Pareto scatter plots for combined case across A values.

One panel per A value showing liberal vs conservative solutions,
with Pareto front highlighted.
"""

from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt

DEI_DIR = Path(__file__).parent.parent
OUTPUT_DIR = DEI_DIR / "docs" / "figures"


def load_layouts(h5_path):
    """Load all layouts from HDF5 file(s). Returns list of dicts."""
    layouts = []
    try:
        with h5py.File(h5_path, 'r') as f:
            for key in f.keys():
                if not key.startswith('layout_'):
                    continue
                try:
                    grp = f[key]
                    layouts.append({
                        'aep_absent': float(grp.attrs['aep_absent']),
                        'aep_present': float(grp.attrs['aep_present']),
                        'strategy': str(grp.attrs['strategy']),
                        'seed': int(grp.attrs['seed']),
                    })
                except Exception:
                    continue
    except Exception:
        pass
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


def plot_panel(ax, layouts, A):
    """Plot Pareto scatter for one A value."""
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

    n_seeds = len(set(l['seed'] for l in layouts))

    # Dominated points
    ax.scatter(absent[lib_mask & ~pareto_mask], present[lib_mask & ~pareto_mask],
               s=40, c='#3498db', alpha=0.4, marker='o', label='Liberal')
    ax.scatter(absent[con_mask & ~pareto_mask], present[con_mask & ~pareto_mask],
               s=40, c='#e74c3c', alpha=0.4, marker='s', label='Conservative')

    # Pareto points
    ax.scatter(absent[lib_mask & pareto_mask], present[lib_mask & pareto_mask],
               s=120, c='#3498db', alpha=1.0, marker='o',
               edgecolors='black', linewidths=1.5, zorder=5, label='Liberal (Pareto)')
    ax.scatter(absent[con_mask & pareto_mask], present[con_mask & pareto_mask],
               s=120, c='#e74c3c', alpha=1.0, marker='s',
               edgecolors='black', linewidths=1.5, zorder=5, label='Conservative (Pareto)')

    # Pareto front line
    if n_pareto > 1:
        pa = absent[pareto_mask]
        pp = present[pareto_mask]
        sort_idx = np.argsort(pa)
        ax.plot(pa[sort_idx], pp[sort_idx], 'k--', linewidth=1.5, alpha=0.7, zorder=4)

    ax.set_title(f'A = {A} | Regret = {regret:.1f} GWh',
                 fontsize=11)

    ax.set_xlabel('AEP without neighbors (GWh)', fontsize=10)
    ax.set_ylabel('AEP with all neighbors (GWh)', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Axis padding
    x_range = absent.max() - absent.min()
    y_range = present.max() - present.min()
    if x_range > 0:
        ax.set_xlim(absent.min() - x_range * 0.15, absent.max() + x_range * 0.15)
    if y_range > 0:
        ax.set_ylim(present.min() - y_range * 0.15, present.max() + y_range * 0.15)

    return n_pareto, regret


def main():
    A_values = []
    all_layouts = {}
    for A in ['0.04', '0.08', '0.12']:
        h5 = DEI_DIR / f"analysis/dei_A{A}/layouts_combined.h5"
        layouts = load_layouts(h5)
        if layouts:
            A_values.append(A)
            all_layouts[A] = layouts

    if not A_values:
        print("No combined case data available yet!")
        return

    n = len(A_values)
    fig, axes = plt.subplots(n, 1, figsize=(4.5, 3.5 * n))
    if n == 1:
        axes = [axes]

    for i, A in enumerate(A_values):
        n_pareto, regret = plot_panel(axes[i], all_layouts[A], A)
        n_seeds = len(set(l['seed'] for l in all_layouts[A]))
        print(f"A={A}: {n_seeds} seeds, {n_pareto} Pareto pts, regret={regret:.1f} GWh")

        if i == 0:
            axes[i].legend(loc='lower left', fontsize=8)

    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "combined_pareto_by_A.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

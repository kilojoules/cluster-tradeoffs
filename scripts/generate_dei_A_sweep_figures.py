"""Generate figures for DEI A parameter sweep analysis."""

import json
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Farm names for reference
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
    """Load layouts from HDF5 file."""
    layouts = []
    with h5py.File(h5_path, 'r') as f:
        n_layouts = f.attrs.get('n_layouts', 0)
        for i in range(n_layouts):
            grp = f[f'layout_{i}']
            layouts.append({
                'x': grp['x'][:],
                'y': grp['y'][:],
                'aep_absent': grp.attrs['aep_absent'],
                'aep_present': grp.attrs['aep_present'],
                'seed': grp.attrs['seed'],
                'strategy': grp.attrs['strategy'],
            })
    return layouts


def find_pareto(aep_absent, aep_present):
    """Find Pareto-optimal indices."""
    n = len(aep_absent)
    pareto_mask = np.zeros(n, dtype=bool)
    for i in range(n):
        dominated = False
        for j in range(n):
            if i != j:
                if (aep_absent[j] >= aep_absent[i] and aep_present[j] > aep_present[i]) or \
                   (aep_absent[j] > aep_absent[i] and aep_present[j] >= aep_present[i]):
                    dominated = True
                    break
        if not dominated:
            pareto_mask[i] = True
    return pareto_mask


def load_liberal_layouts(h5_path):
    """Load only liberal layouts from HDF5 file."""
    layouts = []
    with h5py.File(h5_path, 'r') as f:
        n_layouts = f.attrs.get('n_layouts', 0)
        for i in range(n_layouts):
            grp = f[f'layout_{i}']
            if grp.attrs['strategy'] == 'liberal':
                layouts.append({
                    'aep_absent': grp.attrs['aep_absent'],
                    'aep_present': grp.attrs['aep_present'],
                    'strategy': 'liberal',
                })
    return layouts


def load_conservative_layouts(h5_path):
    """Load only conservative layouts from HDF5 file."""
    layouts = []
    with h5py.File(h5_path, 'r') as f:
        n_layouts = f.attrs.get('n_layouts', 0)
        for i in range(n_layouts):
            grp = f[f'layout_{i}']
            if grp.attrs['strategy'] == 'conservative':
                layouts.append({
                    'aep_absent': grp.attrs['aep_absent'],
                    'aep_present': grp.attrs['aep_present'],
                    'strategy': 'conservative',
                })
    return layouts


def plot_individual_farms(analysis_dir, output_path, A_value):
    """Create 3x3 grid of individual farm Pareto plots."""
    fig, axes = plt.subplots(3, 3, figsize=(14, 12), sharex=True, sharey=True)

    for farm_idx in range(1, 10):
        ax = axes[(farm_idx - 1) // 3, (farm_idx - 1) % 3]

        # Load conservative layouts from farm-specific file
        h5_path = analysis_dir / f"layouts_farm{farm_idx}.h5"
        if not h5_path.exists():
            ax.text(0.5, 0.5, f"Farm {farm_idx}\nNo data", ha='center', va='center', transform=ax.transAxes)
            continue

        # Use re-evaluated liberal layouts if available (correct aep_present
        # for this farm's neighbor), otherwise fall back to farm's own file
        liberal_h5 = analysis_dir / f"liberal_standard_farm{farm_idx}.h5"
        if liberal_h5.exists():
            conservative = load_conservative_layouts(h5_path)
            liberal = load_liberal_layouts(liberal_h5)
            layouts = liberal + conservative
        else:
            layouts = load_layouts(h5_path)

        aep_absent = np.array([l['aep_absent'] for l in layouts])
        aep_present = np.array([l['aep_present'] for l in layouts])
        strategies = np.array([l['strategy'] for l in layouts])

        # Find Pareto front
        pareto_mask = find_pareto(aep_absent, aep_present)

        # Plot liberal (blue circles)
        lib_mask = strategies == 'liberal'
        ax.scatter(aep_absent[lib_mask & ~pareto_mask], aep_present[lib_mask & ~pareto_mask],
                   c='steelblue', alpha=0.5, s=30, label='Liberal')
        ax.scatter(aep_absent[lib_mask & pareto_mask], aep_present[lib_mask & pareto_mask],
                   c='steelblue', edgecolors='black', linewidths=1.5, s=60)

        # Plot conservative (red squares)
        con_mask = strategies == 'conservative'
        ax.scatter(aep_absent[con_mask & ~pareto_mask], aep_present[con_mask & ~pareto_mask],
                   c='indianred', alpha=0.5, s=30, marker='s', label='Conservative')
        ax.scatter(aep_absent[con_mask & pareto_mask], aep_present[con_mask & pareto_mask],
                   c='indianred', edgecolors='black', linewidths=1.5, s=60, marker='s')

        # Compute regret from actual loaded data
        n_pareto = int(pareto_mask.sum())
        pareto_indices = np.where(pareto_mask)[0]
        if len(pareto_indices) > 0:
            lib_opt_idx = pareto_indices[np.argmax(aep_absent[pareto_mask])]
            con_opt_idx = pareto_indices[np.argmax(aep_present[pareto_mask])]
            regret = aep_present[con_opt_idx] - aep_present[lib_opt_idx]
        else:
            regret = 0

        # Direction is a geometric constant — read from JSON if available
        direction = 0
        json_path = analysis_dir / f"dei_single_neighbor_turbopark_farm{farm_idx}.json"
        if json_path.exists():
            with open(json_path) as f:
                farm_data = json.load(f).get(str(farm_idx), {})
                direction = farm_data.get('direction', 0)

        ax.set_title(f"Farm {farm_idx}: {direction:.0f}°\nRegret: {regret:.1f} GWh ({n_pareto} Pareto)", fontsize=10)
        ax.grid(True, alpha=0.3)

        if farm_idx == 1:
            ax.legend(loc='lower right', fontsize=8)

    # Shared axis labels on outer edges only
    for ax in axes[-1, :]:
        ax.set_xlabel('AEP without neighbor (GWh)')
    for ax in axes[:, 0]:
        ax.set_ylabel('AEP with neighbor (GWh)')

    fig.suptitle(f'DEI Individual Neighbors Analysis (A={A_value})\nTurboGaussian Wake Model', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_combined(analysis_dir, output_path, A_value):
    """Create Pareto plot for combined case."""
    h5_path = analysis_dir / "layouts_combined.h5"
    if not h5_path.exists():
        print(f"Combined layouts not found: {h5_path}")
        return

    # Use re-evaluated liberal layouts if available, otherwise fall back
    liberal_h5 = analysis_dir / "liberal_standard_combined.h5"
    if liberal_h5.exists():
        conservative = load_conservative_layouts(h5_path)
        liberal = load_liberal_layouts(liberal_h5)
        layouts = liberal + conservative
    else:
        layouts = load_layouts(h5_path)

    aep_absent = np.array([l['aep_absent'] for l in layouts])
    aep_present = np.array([l['aep_present'] for l in layouts])
    strategies = np.array([l['strategy'] for l in layouts])

    pareto_mask = find_pareto(aep_absent, aep_present)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot liberal
    lib_mask = strategies == 'liberal'
    ax.scatter(aep_absent[lib_mask & ~pareto_mask], aep_present[lib_mask & ~pareto_mask],
               c='steelblue', alpha=0.5, s=50, label='Liberal (ignore neighbors)')
    ax.scatter(aep_absent[lib_mask & pareto_mask], aep_present[lib_mask & pareto_mask],
               c='steelblue', edgecolors='black', linewidths=2, s=100)

    # Plot conservative
    con_mask = strategies == 'conservative'
    ax.scatter(aep_absent[con_mask & ~pareto_mask], aep_present[con_mask & ~pareto_mask],
               c='indianred', alpha=0.5, s=50, marker='s', label='Conservative (all 594 turbines)')
    ax.scatter(aep_absent[con_mask & pareto_mask], aep_present[con_mask & pareto_mask],
               c='indianred', edgecolors='black', linewidths=2, s=100, marker='s')

    # Compute regret from actual loaded data
    n_pareto = int(pareto_mask.sum())
    pareto_indices = np.where(pareto_mask)[0]
    if len(pareto_indices) > 0:
        lib_opt_idx = pareto_indices[np.argmax(aep_absent[pareto_mask])]
        con_opt_idx = pareto_indices[np.argmax(aep_present[pareto_mask])]
        regret = aep_present[con_opt_idx] - aep_present[lib_opt_idx]
        regret_pct = regret / aep_present[con_opt_idx] * 100
    else:
        regret = 0
        regret_pct = 0

    ax.set_title(f'DEI All Neighbors Combined (A={A_value})\nRegret: {regret:.1f} GWh ({regret_pct:.2f}%), {n_pareto} Pareto points', fontsize=14)
    ax.set_xlabel('AEP without neighbors (GWh)', fontsize=12)
    ax.set_ylabel('AEP with all 594 neighbor turbines (GWh)', fontsize=12)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_polar_regret(analysis_dir, output_path, A_value):
    """Create polar plot of regret by direction."""
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})

    directions = []
    regrets = []
    farm_labels = []

    for farm_idx in range(1, 10):
        json_path = analysis_dir / f"dei_single_neighbor_turbopark_farm{farm_idx}.json"
        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)
                farm_data = data.get(str(farm_idx), {})
                directions.append(np.radians(farm_data.get('direction', 0)))
                regrets.append(farm_data.get('regret_gwh', 0))
                farm_labels.append(f"Farm {farm_idx}")

    if not directions:
        print(f"No farm data found in {analysis_dir}")
        return

    # Plot bars
    bars = ax.bar(directions, regrets, width=0.3, alpha=0.7, color='steelblue', edgecolor='black')

    # Add labels
    max_regret = max(regrets) if regrets else 1.0
    for d, r, label in zip(directions, regrets, farm_labels):
        if r > 0:
            ax.annotate(f'{label}\n{r:.1f} GWh', xy=(d, r), xytext=(d, r + max_regret*0.15),
                       ha='center', va='bottom', fontsize=9)

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_title(f'DEI Regret by Neighbor Direction (A={A_value})', fontsize=14, pad=20)
    ax.set_ylabel('Regret (GWh)', labelpad=30)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    for A_value in ['0.02', '0.04', '0.10']:
        analysis_dir = Path(f"analysis/dei_A{A_value}")
        output_dir = Path("docs/figures")
        output_dir.mkdir(exist_ok=True)

        if not analysis_dir.exists():
            print(f"Directory not found: {analysis_dir}")
            continue

        print(f"\n=== Generating figures for A={A_value} ===")

        # Individual farms plot
        plot_individual_farms(
            analysis_dir,
            output_dir / f"dei_A{A_value}_individual.png",
            A_value
        )

        # Combined plot
        plot_combined(
            analysis_dir,
            output_dir / f"dei_A{A_value}_combined.png",
            A_value
        )

        # Polar regret plot
        plot_polar_regret(
            analysis_dir,
            output_dir / f"dei_A{A_value}_polar.png",
            A_value
        )


if __name__ == "__main__":
    main()

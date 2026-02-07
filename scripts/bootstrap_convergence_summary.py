#!/usr/bin/env python
"""
Summary plot of bootstrap regret convergence across all farms.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=Path, default=Path('analysis/dei_A0.02'))
    parser.add_argument('--output', '-o', type=Path, default=None)
    args = parser.parse_args()

    # Load all results
    farm_labels = {
        1: 'Farm 1 (SW)',
        2: 'Farm 2 (W)',
        3: 'Farm 3 (NW)',
        4: 'Farm 4 (N)',
        5: 'Farm 5 (NE)',
        6: 'Farm 6 (E)',
        7: 'Farm 7 (SE)',
        8: 'Farm 8 (S)',
        9: 'Farm 9 (SSW)',
    }

    fig, axes = plt.subplots(2, 5, figsize=(16, 7), sharey=False)
    axes = axes.flatten()

    for idx, farm in enumerate(list(range(1, 10)) + ['combined']):
        ax = axes[idx]

        if farm == 'combined':
            npz_path = args.input / 'bootstrap_convergence_combined.npz'
            title = 'Combined (All 594)'
        else:
            npz_path = args.input / f'bootstrap_convergence_farm{farm}.npz'
            title = farm_labels[farm]

        if not npz_path.exists():
            print(f'Skipping {npz_path} (not found)')
            ax.set_title(title)
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue

        data = np.load(npz_path)
        n = data['n_starts']
        mean = data['mean']
        p5 = data['p5']
        p95 = data['p95']
        true_regret = float(data['true_regret'])

        # Plot
        ax.fill_between(n, p5, p95, alpha=0.3, color='C0')
        ax.plot(n, mean, 'C0-', linewidth=1.5, label='Bootstrap mean')
        ax.axhline(true_regret, color='C1', linestyle='--', linewidth=1.5)

        ax.set_title(title, fontsize=10)
        ax.set_xlim(1, 50)
        ax.grid(True, alpha=0.3)

        # Add true regret annotation
        if true_regret > 0:
            ax.text(0.95, 0.95, f'{true_regret:.1f} GWh',
                    transform=ax.transAxes, ha='right', va='top',
                    fontsize=9, color='C1')

        if idx >= 5:
            ax.set_xlabel('Starts', fontsize=9)
        if idx % 5 == 0:
            ax.set_ylabel('Regret (GWh)', fontsize=9)

    fig.suptitle('Regret Convergence vs. Number of Multi-Starts (A=0.02)', fontsize=14)
    plt.tight_layout()

    out_path = args.output or args.input / 'bootstrap_convergence_summary.png'
    plt.savefig(out_path, dpi=150)
    print(f'Saved to {out_path}')


if __name__ == '__main__':
    main()

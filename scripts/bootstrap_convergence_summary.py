#!/usr/bin/env python
"""
Summary plot of conservative AEP convergence (shuffle + cumulative max)
across all farms. Each subplot shows individual shuffle lines.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=Path, default=Path('analysis/dei_A0.04'))
    parser.add_argument('--output', '-o', type=Path, default=None)
    args = parser.parse_args()

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

    fig, axes = plt.subplots(2, 5, figsize=(16, 7), sharey=True, sharex=True)
    axes = axes.flatten()

    for idx, farm in enumerate(list(range(1, 10)) + ['farm1_liberal']):
        ax = axes[idx]

        if farm == 'farm1_liberal':
            npz_path = args.input / 'bootstrap_convergence_farm1_liberal.npz'
            title = 'Liberal'
        else:
            npz_path = args.input / f'bootstrap_convergence_farm{farm}.npz'
            title = farm_labels[farm]

        if not npz_path.exists():
            print(f'Skipping {npz_path} (not found)')
            ax.set_title(title)
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes)
            continue

        data = np.load(npz_path)
        n = data['n_starts']
        curves = data['curves']
        true_best = float(data['true_best'])

        p1 = np.percentile(curves, 1, axis=0)
        p5 = np.percentile(curves, 5, axis=0)
        p10 = np.percentile(curves, 10, axis=0)
        for i in range(curves.shape[0]):
            ax.plot(n, curves[i], color='C0', alpha=0.05, linewidth=0.3)
        ax.plot(n, p10, color='k', linewidth=1, linestyle=':', label='10th Percentile')
        ax.plot(n, p5, color='k', linewidth=1, linestyle='--', label='5th Percentile')
        ax.plot(n, p1, color='k', linewidth=1, linestyle='-', label='1st Percentile')
        #ax.axhline(true_best, color='C1', linestyle='--', linewidth=1.5)

        ax.set_title(title, fontsize=10)
        ax.set_xlim(1, len(n))
        ax.grid(True, alpha=0.3)

        #ax.text(0.95, 0.05, f'{true_best:.1f} GWh',
        #        transform=ax.transAxes, ha='right', va='bottom',
        #        fontsize=9, color='C1')

        if idx >= 5:
            ax.set_xlabel('Starts', fontsize=9)
        if idx % 5 == 0:
            ax.set_ylabel('Best AEP (GWh)', fontsize=9)

    # Extract A value from input directory name (e.g. "dei_A0.04" -> "0.04")
    a_str = args.input.name.replace('dei_A', '')
    fig.suptitle(f'Conservative AEP Convergence vs. Multi-Starts (A={a_str})',
                 fontsize=14)

    # Add a single figure-level legend using handles from the first subplot
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, 0.02))
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    out_path = args.output or args.input / 'bootstrap_convergence_summary.png'
    plt.savefig(out_path, dpi=150)
    print(f'Saved to {out_path}')


if __name__ == '__main__':
    main()

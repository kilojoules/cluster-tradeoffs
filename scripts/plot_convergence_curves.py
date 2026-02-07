#!/usr/bin/env python
"""
Plot convergence curves showing liberal vs conservative optimization difficulty.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    input_dir = Path('analysis/dei_A0.02')

    # Focus on cases with meaningful regret
    cases = [8, 1, 2, 'combined']
    labels = ['Farm 8 (S)', 'Farm 1 (SW)', 'Farm 2 (W)', 'Combined']

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['C0', 'C1', 'C2', 'C3']

    for case, label, color in zip(cases, labels, colors):
        if case == 'combined':
            npz_path = input_dir / 'bootstrap_convergence_combined.npz'
        else:
            npz_path = input_dir / f'bootstrap_convergence_farm{case}.npz'

        d = np.load(npz_path)
        n_starts = d['n_starts']

        true_lib = float(d['true_liberal_aep'])
        true_con = float(d['true_conservative_aep'])

        lib_mean = d['liberal_aep_mean']
        con_mean = d['conservative_aep_mean']

        # Gap from true value (in GWh)
        lib_gap = true_lib - lib_mean
        con_gap = true_con - con_mean

        ax.plot(n_starts, lib_gap, '-', color=color, linewidth=2, label=f'{label} - Liberal')
        ax.plot(n_starts, con_gap, '--', color=color, linewidth=2, label=f'{label} - Conservative')

    ax.set_xlabel('Number of optimization starts', fontsize=12)
    ax.set_ylabel('Gap from true optimal AEP (GWh)', fontsize=12)
    ax.set_title('Optimization Convergence: Liberal vs Conservative Strategies (A=0.02)', fontsize=14)

    # Create legend with two columns
    ax.legend(loc='upper right', ncol=2, fontsize=9)

    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, 50)
    ax.set_ylim(0, 35)

    # Add annotation
    ax.annotate('Solid = Liberal (easier)\nDashed = Conservative (harder)',
               xy=(0.02, 0.98), xycoords='axes fraction',
               ha='left', va='top', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    out_path = input_dir / 'convergence_lib_vs_con.png'
    plt.savefig(out_path, dpi=150)
    print(f'Saved to {out_path}')


if __name__ == '__main__':
    main()

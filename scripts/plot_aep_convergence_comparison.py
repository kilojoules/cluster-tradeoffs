#!/usr/bin/env python
"""
Plot comparing liberal vs conservative AEP convergence across all farms.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    input_dir = Path('analysis/dei_A0.02')

    # Load all results
    cases = list(range(1, 10)) + ['combined']
    labels = [f'Farm {i}' for i in range(1, 10)] + ['Combined']

    data = {}
    for case in cases:
        if case == 'combined':
            npz_path = input_dir / 'bootstrap_convergence_combined.npz'
        else:
            npz_path = input_dir / f'bootstrap_convergence_farm{case}.npz'
        data[case] = np.load(npz_path)

    n_starts = data[1]['n_starts']

    # Create figure with two panels
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Color map for farms
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # Left panel: Convergence curves (normalized gap)
    ax = axes[0]

    for idx, (case, label) in enumerate(zip(cases, labels)):
        d = data[case]
        true_lib = float(d['true_liberal_aep'])
        true_con = float(d['true_conservative_aep'])

        # Normalize: gap as fraction of total range
        lib_mean = d['liberal_aep_mean']
        con_mean = d['conservative_aep_mean']

        # Gap from true value (in GWh)
        lib_gap = true_lib - lib_mean
        con_gap = true_con - con_mean

        color = colors[idx]
        ax.plot(n_starts, lib_gap, '-', color=color, alpha=0.7, linewidth=1.5)
        ax.plot(n_starts, con_gap, '--', color=color, alpha=0.7, linewidth=1.5)

    # Add legend entries for line styles
    ax.plot([], [], 'k-', linewidth=2, label='Liberal (solid)')
    ax.plot([], [], 'k--', linewidth=2, label='Conservative (dashed)')
    ax.legend(loc='upper right')

    ax.set_xlabel('Number of optimization starts', fontsize=12)
    ax.set_ylabel('Gap from true optimum (GWh)', fontsize=12)
    ax.set_title('AEP Convergence: Liberal vs Conservative', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, 50)
    ax.set_ylim(bottom=0)

    # Right panel: Bar chart of gaps at n=50
    ax = axes[1]

    x = np.arange(len(cases))
    width = 0.35

    lib_gaps = []
    con_gaps = []
    for case in cases:
        d = data[case]
        lib_gaps.append(float(d['true_liberal_aep']) - d['liberal_aep_mean'][-1])
        con_gaps.append(float(d['true_conservative_aep']) - d['conservative_aep_mean'][-1])

    bars1 = ax.bar(x - width/2, lib_gaps, width, label='Liberal', color='C0', alpha=0.7)
    bars2 = ax.bar(x + width/2, con_gaps, width, label='Conservative', color='C1', alpha=0.7)

    ax.set_xlabel('Case', fontsize=12)
    ax.set_ylabel('Gap from true optimum at n=50 (GWh)', fontsize=12)
    ax.set_title('Optimization Difficulty by Strategy', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        if height > 1:
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    out_path = input_dir / 'aep_convergence_comparison.png'
    plt.savefig(out_path, dpi=150)
    print(f'Saved to {out_path}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python
"""
Convergence analysis of conservative AEP w.r.t. number of multi-starts.

For many random shuffles, the conservative optimization results are shuffled
and the cumulative maximum is plotted. Each shuffle is shown as an individual
line, illustrating how the best-found-so-far improves with more starts.
"""

import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_conservative_aep(h5_path: Path) -> np.ndarray:
    """Load aep_present values for conservative layouts from HDF5 file."""
    values = []
    with h5py.File(h5_path, 'r') as f:
        for key in f.keys():
            if not key.startswith('layout_'):
                continue
            layout = f[key]
            if layout.attrs['strategy'] == 'conservative':
                values.append(float(layout.attrs['aep_present']))
    return np.array(values)


def main():
    parser = argparse.ArgumentParser(
        description='Convergence analysis via shuffle + cumulative max')
    parser.add_argument('--input', '-i', type=Path,
                        default=Path('analysis/dei_A0.02'),
                        help='Input directory with layout HDF5 files')
    parser.add_argument('--farm', type=int, default=8,
                        help='Farm number to analyze (default: 8)')
    parser.add_argument('--combined', action='store_true',
                        help='Analyze combined case instead of single farm')
    parser.add_argument('--n-samples', type=int, default=1000,
                        help='Number of random shuffles (default: 1000)')
    parser.add_argument('--output', '-o', type=Path, default=None,
                        help='Output figure path')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    # Load layouts
    if args.combined:
        h5_path = args.input / 'layouts_combined.h5'
        title = 'Combined (All 594 Neighbors)'
    else:
        h5_path = args.input / f'layouts_farm{args.farm}.h5'
        title = f'Farm {args.farm}'

    print(f'Loading conservative layouts from {h5_path}')
    aep = load_conservative_aep(h5_path)
    n_total = len(aep)
    print(f'  {n_total} conservative layouts')

    rng = np.random.default_rng(args.seed)
    n_starts = np.arange(1, n_total + 1)

    # Generate cumulative-max curves for each shuffle
    curves = np.zeros((args.n_samples, n_total))
    for i in range(args.n_samples):
        perm = rng.permutation(n_total)
        curves[i] = np.maximum.accumulate(aep[perm])

    true_best = aep.max()
    print(f'  Best conservative AEP: {true_best:.2f} GWh')

    # Plot
    p1 = np.percentile(curves, 1, axis=0)
    p5 = np.percentile(curves, 5, axis=0)
    p10 = np.percentile(curves, 10, axis=0)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i in range(args.n_samples):
        ax.plot(n_starts, curves[i], color='C0', alpha=0.05, linewidth=0.5)
    ax.plot(n_starts, p10, color='k', linewidth=2, linestyle='--', label='p10')
    ax.plot(n_starts, p5, color='k', linewidth=2.5, label='p5')
    ax.plot(n_starts, p1, color='k', linewidth=3, linestyle=':', label='p1')
    ax.axhline(true_best, color='C1', linestyle='--', linewidth=2,
               label=f'Best ({true_best:.1f} GWh)')

    ax.set_xlabel('Number of optimization starts', fontsize=12)
    ax.set_ylabel('Best conservative AEP found (GWh)', fontsize=12)
    ax.set_title(f'Conservative AEP Convergence - {title}', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, n_total)
    plt.tight_layout()

    if args.output:
        out_path = args.output
    else:
        if args.combined:
            out_path = args.input / 'bootstrap_convergence_combined.png'
        else:
            out_path = args.input / f'bootstrap_convergence_farm{args.farm}.png'

    plt.savefig(out_path, dpi=150)
    print(f'\nSaved figure to {out_path}')

    # Save raw curves
    results_path = out_path.with_suffix('.npz')
    np.savez(results_path, n_starts=n_starts, curves=curves,
             true_best=true_best)
    print(f'Saved results to {results_path}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python
"""
Convergence analysis of AEP w.r.t. number of multi-starts.

For many random shuffles, the optimization results are shuffled
and the cumulative maximum is plotted. Each shuffle is shown as an individual
line, illustrating how the best-found-so-far improves with more starts.
"""

import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_aep(h5_path: Path, strategy: str = 'conservative',
             aep_key: str = 'aep_present') -> np.ndarray:
    """Load AEP values for the given strategy from HDF5 file."""
    values = []
    try:
        with h5py.File(h5_path, 'r') as f:
            for key in f.keys():
                if not key.startswith('layout_'):
                    continue
                try:
                    layout = f[key]
                    if layout.attrs['strategy'] == strategy:
                        values.append(float(layout.attrs[aep_key]))
                except (KeyError, RuntimeError, OSError):
                    continue
    except (KeyError, RuntimeError, OSError) as e:
        print(f'  WARNING: skipping {h5_path.name}: {e}')
    return np.array(values)


def main():
    parser = argparse.ArgumentParser(
        description='Convergence analysis via shuffle + cumulative max')
    parser.add_argument('--input', '-i', type=Path,
                        default=Path('analysis/dei_A0.04'),
                        help='Input directory with layout HDF5 files')
    parser.add_argument('--farm', type=int, default=8,
                        help='Farm number to analyze (default: 8)')
    parser.add_argument('--combined', action='store_true',
                        help='Analyze combined case instead of single farm')
    parser.add_argument('--pool-liberal', action='store_true',
                        help='Pool liberal seeds from all 9 farms')
    parser.add_argument('--strategy', type=str, default='conservative',
                        choices=['conservative', 'liberal'],
                        help='Strategy to analyze (default: conservative)')
    parser.add_argument('--n-samples', type=int, default=1000,
                        help='Number of random shuffles (default: 1000)')
    parser.add_argument('--output', '-o', type=Path, default=None,
                        help='Output figure path')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    # Load layouts
    if args.pool_liberal:
        args.strategy = 'liberal'
        title = 'Pooled Liberal (All Runs)'
        aep_list = []
        # Individual farm files
        for farm_idx in range(1, 10):
            h5_path = args.input / f'layouts_farm{farm_idx}.h5'
            if h5_path.exists():
                vals = load_aep(h5_path, strategy='liberal', aep_key='aep_absent')
                print(f'  Farm {farm_idx}: {len(vals)} liberal seeds')
                aep_list.append(vals)
        # Combined file (no shards — those are pre-polygon)
        for h5_path in [args.input / 'layouts_combined.h5']:
            if h5_path.exists():
                vals = load_aep(h5_path, strategy='liberal', aep_key='aep_absent')
                if len(vals) > 0:
                    print(f'  {h5_path.name}: {len(vals)} liberal seeds')
                    aep_list.append(vals)
        aep = np.concatenate(aep_list)
        n_total = len(aep)
        print(f'  Total: {n_total} pooled liberal layouts')
    elif args.combined:
        h5_path = args.input / 'layouts_combined.h5'
        title = 'Combined (All 594 Neighbors)'
        aep_key = 'aep_absent' if args.strategy == 'liberal' else 'aep_present'
        print(f'Loading {args.strategy} layouts from {h5_path} (using {aep_key})')
        aep = load_aep(h5_path, strategy=args.strategy, aep_key=aep_key)
        n_total = len(aep)
        print(f'  {n_total} {args.strategy} layouts')
    else:
        h5_path = args.input / f'layouts_farm{args.farm}.h5'
        title = f'Farm {args.farm}'
        aep_key = 'aep_absent' if args.strategy == 'liberal' else 'aep_present'
        print(f'Loading {args.strategy} layouts from {h5_path} (using {aep_key})')
        aep = load_aep(h5_path, strategy=args.strategy, aep_key=aep_key)
        n_total = len(aep)
        print(f'  {n_total} {args.strategy} layouts')

    # Cap at 1000 seeds
    if n_total > 1000:
        aep = aep[:1000]
        n_total = 1000
        print(f'  Capped to {n_total} seeds')

    rng = np.random.default_rng(args.seed)
    n_starts = np.arange(1, n_total + 1)

    # Generate cumulative-max curves for each shuffle
    curves = np.zeros((args.n_samples, n_total))
    for i in range(args.n_samples):
        perm = rng.permutation(n_total)
        curves[i] = np.maximum.accumulate(aep[perm])

    true_best = aep.max()
    print(f'  Best {args.strategy} AEP: {true_best:.2f} GWh')

    # Plot
    p1 = np.percentile(curves, 1, axis=0)
    p5 = np.percentile(curves, 5, axis=0)
    p10 = np.percentile(curves, 10, axis=0)

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    for i in range(args.n_samples):
        ax.plot(n_starts, curves[i], color='C0', alpha=0.05, linewidth=0.5)
    ax.plot(n_starts, p10, color='k', linewidth=1, linestyle=':', label='10th pctl.')
    ax.plot(n_starts, p5, color='k', linewidth=1, label='5th pctl.', linestyle='--')
    ax.plot(n_starts, p1, color='k', linewidth=1, linestyle='-', label='1st pctl.')

    ax.set_xlabel('Number of optimization starts')
    ax.set_ylabel(f'Best {args.strategy} AEP found (GWh)')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, n_total)
    plt.tight_layout()

    if args.output:
        out_path = args.output
    elif args.pool_liberal:
        out_path = args.input / 'bootstrap_convergence_pooled_liberal.png'
    else:
        suffix = '' if args.strategy == 'conservative' else f'_{args.strategy}'
        if args.combined:
            out_path = args.input / f'bootstrap_convergence_combined{suffix}.png'
        else:
            out_path = args.input / f'bootstrap_convergence_farm{args.farm}{suffix}.png'

    plt.savefig(out_path, dpi=150)
    print(f'\nSaved figure to {out_path}')

    # Save raw curves
    results_path = out_path.with_suffix('.npz')
    np.savez(results_path, n_starts=n_starts, curves=curves,
             true_best=true_best)
    print(f'Saved results to {results_path}')


if __name__ == '__main__':
    main()

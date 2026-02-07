#!/usr/bin/env python
"""
Bootstrap analysis of regret convergence w.r.t. number of multi-starts.

For each number of starts n, we bootstrap sample n layouts from the available
optimization results and compute the resulting regret. This shows how regret
estimates stabilize as more optimization starts are used.
"""

import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_layouts(h5_path: Path) -> tuple[list[dict], list[dict]]:
    """
    Load liberal and conservative layouts from HDF5 file.

    Returns two lists, one for each strategy. Each list contains the layouts
    produced by that optimization strategy (50 multi-starts each).
    """
    liberal = []
    conservative = []

    with h5py.File(h5_path, 'r') as f:
        for key in f.keys():
            if not key.startswith('layout_'):
                continue
            layout = f[key]
            data = {
                'aep_absent': float(layout.attrs['aep_absent']),
                'aep_present': float(layout.attrs['aep_present']),
                'strategy': layout.attrs['strategy'],
                'seed': int(layout.attrs['seed'])
            }
            if data['strategy'] == 'liberal':
                liberal.append(data)
            else:
                conservative.append(data)

    return liberal, conservative


def load_all_layouts(h5_path: Path) -> list[dict]:
    """Load all layouts from HDF5 file regardless of strategy."""
    layouts = []

    with h5py.File(h5_path, 'r') as f:
        for key in f.keys():
            if not key.startswith('layout_'):
                continue
            layout = f[key]
            layouts.append({
                'aep_absent': float(layout.attrs['aep_absent']),
                'aep_present': float(layout.attrs['aep_present']),
                'strategy': layout.attrs['strategy'],
                'seed': int(layout.attrs['seed'])
            })

    return layouts


def compute_regret(layouts: list[dict]) -> float:
    """
    Compute regret given a set of layouts.

    The Pareto frontier is found by considering ALL layouts (from both strategies).
    - Liberal-optimal: layout with highest AEP when neighbor is absent
    - Conservative-optimal: layout with highest AEP when neighbor is present
    - Regret: AEP loss when liberal layout is used but neighbor is present
    """
    # Best layout for "alone" scenario (liberal objective)
    liberal_best = max(layouts, key=lambda x: x['aep_absent'])

    # Best layout for "with neighbor" scenario (conservative objective)
    conservative_best = max(layouts, key=lambda x: x['aep_present'])

    # Regret: what we lose by using liberal layout when neighbor is present
    regret = conservative_best['aep_present'] - liberal_best['aep_present']

    return regret


def bootstrap_regret(liberal: list[dict], conservative: list[dict],
                     n_starts: int, n_bootstrap: int = 1000,
                     rng: np.random.Generator = None) -> dict:
    """
    Bootstrap estimate of regret and optimal AEPs for a given number of starts.

    Samples n_starts layouts (with replacement) from each strategy,
    pools them together, then finds Pareto-optimal layouts and computes regret.
    Repeats n_bootstrap times.

    Returns dict with arrays for 'regret', 'liberal_aep', 'conservative_aep'.
    """
    if rng is None:
        rng = np.random.default_rng()

    results = {'regret': [], 'liberal_aep': [], 'conservative_aep': []}
    for _ in range(n_bootstrap):
        # Sample n_starts from each strategy independently
        lib_sample = rng.choice(liberal, size=n_starts, replace=True).tolist()
        con_sample = rng.choice(conservative, size=n_starts, replace=True).tolist()

        # Pool all layouts and find best for each objective
        all_layouts = lib_sample + con_sample

        # Best layout for "alone" scenario (liberal objective)
        liberal_best = max(all_layouts, key=lambda x: x['aep_absent'])
        # Best layout for "with neighbor" scenario (conservative objective)
        conservative_best = max(all_layouts, key=lambda x: x['aep_present'])

        results['liberal_aep'].append(liberal_best['aep_absent'])
        results['conservative_aep'].append(conservative_best['aep_present'])
        results['regret'].append(conservative_best['aep_present'] - liberal_best['aep_present'])

    return {k: np.array(v) for k, v in results.items()}


def main():
    parser = argparse.ArgumentParser(description='Bootstrap regret convergence analysis')
    parser.add_argument('--input', '-i', type=Path, default=Path('analysis/dei_A0.02'),
                        help='Input directory with layout HDF5 files')
    parser.add_argument('--farm', type=int, default=8,
                        help='Farm number to analyze (default: 8)')
    parser.add_argument('--combined', action='store_true',
                        help='Analyze combined case instead of single farm')
    parser.add_argument('--n-bootstrap', type=int, default=1000,
                        help='Number of bootstrap samples (default: 1000)')
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

    print(f'Loading layouts from {h5_path}')
    liberal, conservative = load_layouts(h5_path)
    print(f'  Liberal layouts: {len(liberal)}')
    print(f'  Conservative layouts: {len(conservative)}')

    n_total = len(liberal)
    rng = np.random.default_rng(args.seed)

    # Compute regret for different numbers of starts
    start_counts = list(range(1, n_total + 1))

    # Store results for regret and both AEP metrics
    metrics = ['regret', 'liberal_aep', 'conservative_aep']
    results = {'n_starts': []}
    for metric in metrics:
        results[metric] = {'mean': [], 'std': [], 'p5': [], 'p25': [], 'p50': [], 'p75': [], 'p95': []}

    print(f'\nBootstrapping for n_starts = 1 to {n_total}...')
    for n in start_counts:
        bootstrap_results = bootstrap_regret(liberal, conservative, n, args.n_bootstrap, rng)

        results['n_starts'].append(n)
        for metric in metrics:
            data = bootstrap_results[metric]
            results[metric]['mean'].append(np.mean(data))
            results[metric]['std'].append(np.std(data))
            results[metric]['p5'].append(np.percentile(data, 5))
            results[metric]['p25'].append(np.percentile(data, 25))
            results[metric]['p50'].append(np.percentile(data, 50))
            results[metric]['p75'].append(np.percentile(data, 75))
            results[metric]['p95'].append(np.percentile(data, 95))

        if n % 10 == 0 or n == 1:
            print(f'  n={n:2d}: liberal={results["liberal_aep"]["mean"][-1]:.1f}, '
                  f'conservative={results["conservative_aep"]["mean"][-1]:.1f}, '
                  f'regret={results["regret"]["mean"][-1]:.2f} GWh')

    # Compute "true" values using all starts
    all_layouts = liberal + conservative
    true_liberal = max(all_layouts, key=lambda x: x['aep_absent'])
    true_conservative = max(all_layouts, key=lambda x: x['aep_present'])
    true_liberal_aep = true_liberal['aep_absent']
    true_conservative_aep = true_conservative['aep_present']
    true_regret = true_conservative_aep - true_liberal['aep_present']

    print(f'\nTrue values (all {n_total} starts per strategy):')
    print(f'  Liberal-optimal AEP (alone): {true_liberal_aep:.2f} GWh')
    print(f'  Conservative-optimal AEP (with neighbor): {true_conservative_aep:.2f} GWh')
    print(f'  Regret: {true_regret:.2f} GWh')

    # Plot - two panels: AEP convergence and regret
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    n = np.array(results['n_starts'])

    # Left panel: AEP convergence
    ax = axes[0]
    for metric, color, label, true_val in [
        ('liberal_aep', 'C0', 'Liberal (max AEP alone)', true_liberal_aep),
        ('conservative_aep', 'C1', 'Conservative (max AEP w/neighbor)', true_conservative_aep),
    ]:
        mean = np.array(results[metric]['mean'])
        p5 = np.array(results[metric]['p5'])
        p95 = np.array(results[metric]['p95'])

        ax.fill_between(n, p5, p95, alpha=0.2, color=color)
        ax.plot(n, mean, color=color, linewidth=2, label=label)
        ax.axhline(true_val, color=color, linestyle='--', linewidth=1.5, alpha=0.7)

    ax.set_xlabel('Number of optimization starts', fontsize=12)
    ax.set_ylabel('Optimal AEP (GWh)', fontsize=12)
    ax.set_title(f'AEP Convergence - {title}', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, n_total)

    # Right panel: Regret
    ax = axes[1]
    mean = np.array(results['regret']['mean'])
    p5 = np.array(results['regret']['p5'])
    p25 = np.array(results['regret']['p25'])
    p75 = np.array(results['regret']['p75'])
    p95 = np.array(results['regret']['p95'])

    ax.fill_between(n, p5, p95, alpha=0.2, color='C2', label='90% CI')
    ax.fill_between(n, p25, p75, alpha=0.4, color='C2', label='50% CI')
    ax.plot(n, mean, 'C2-', linewidth=2, label='Bootstrap mean')
    ax.axhline(true_regret, color='C3', linestyle='--', linewidth=2,
               label=f'True regret ({true_regret:.2f} GWh)')

    ax.set_xlabel('Number of optimization starts', fontsize=12)
    ax.set_ylabel('Design regret (GWh)', fontsize=12)
    ax.set_title(f'Regret Convergence - {title}', fontsize=14)
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

    # Save numerical results
    results_path = out_path.with_suffix('.npz')
    save_dict = {'n_starts': np.array(results['n_starts'])}
    for metric in metrics:
        for stat, vals in results[metric].items():
            save_dict[f'{metric}_{stat}'] = np.array(vals)
    save_dict['true_liberal_aep'] = true_liberal_aep
    save_dict['true_conservative_aep'] = true_conservative_aep
    save_dict['true_regret'] = true_regret
    np.savez(results_path, **save_dict)
    print(f'Saved results to {results_path}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python
"""
Bootstrap convergence analysis of conservative AEP for A=0.04 individual farms.

For each farm, loads conservative layouts (capped at 500 seeds), generates
random shuffles, computes cumulative maximum curves, and plots a 3x3 summary.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

INPUT_DIR = Path('analysis/dei_A0.04')
MAX_SEEDS = 500
N_SAMPLES = 1000
RNG_SEED = 42

FARM_LABELS = {
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


def load_conservative_aep(h5_path: Path, max_seeds: int) -> np.ndarray:
    """Load aep_present for conservative layouts, sorted by seed, capped."""
    entries = []
    with h5py.File(h5_path, 'r') as f:
        for key in f.keys():
            if not key.startswith('layout_'):
                continue
            grp = f[key]
            if grp.attrs['strategy'] == 'conservative':
                entries.append((int(grp.attrs['seed']), float(grp.attrs['aep_present'])))
    # Sort by seed so "first N seeds" is deterministic
    entries.sort(key=lambda x: x[0])
    values = np.array([v for _, v in entries[:max_seeds]])
    return values


def bootstrap_cummax(aep: np.ndarray, n_samples: int, rng) -> np.ndarray:
    """Generate cumulative-max curves from random shuffles."""
    n = len(aep)
    curves = np.zeros((n_samples, n))
    for i in range(n_samples):
        perm = rng.permutation(n)
        curves[i] = np.maximum.accumulate(aep[perm])
    return curves


def main():
    rng = np.random.default_rng(RNG_SEED)

    fig, axes = plt.subplots(3, 3, figsize=(14, 11), sharey=True, sharex=True)
    axes = axes.flatten()

    for idx, farm in enumerate(range(1, 10)):
        ax = axes[idx]
        h5_path = INPUT_DIR / f'layouts_farm{farm}.h5'

        if not h5_path.exists():
            ax.set_title(FARM_LABELS[farm], fontsize=11)
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='gray')
            continue

        aep = load_conservative_aep(h5_path, MAX_SEEDS)
        n_total = len(aep)
        print(f'{FARM_LABELS[farm]}: {n_total} conservative seeds, '
              f'AEP range [{aep.min():.2f}, {aep.max():.2f}] GWh')

        curves = bootstrap_cummax(aep, N_SAMPLES, rng)
        true_best = aep.max()
        n_starts = np.arange(1, n_total + 1)

        # Percentile envelopes
        p1 = np.percentile(curves, 1, axis=0)
        p5 = np.percentile(curves, 5, axis=0)
        p10 = np.percentile(curves, 10, axis=0)
        p50 = np.percentile(curves, 50, axis=0)

        # Individual shuffle lines
        for i in range(N_SAMPLES):
            ax.plot(n_starts, curves[i], color='C0', alpha=0.03, linewidth=0.4)

        # Percentile lines
        #ax.plot(n_starts, p50, color='0.2', linewidth=1, linestyle='-',
        #        label='median')
        ax.plot(n_starts, p10, color='k', linewidth=1.5, linestyle=':',
                label='p10')
        ax.plot(n_starts, p5, color='k', linewidth=2, label='p5', linestyle='--')
        ax.plot(n_starts, p1, color='k', linewidth=1.5, linestyle='-',
                label='p1')

        # True best
        #ax.axhline(true_best, color='C1', linestyle='--', linewidth=1.5)

        ax.set_title(FARM_LABELS[farm], fontsize=11)
        ax.set_xlim(1, n_total)
        ax.grid(True, alpha=0.3)

        # Annotate best value
        #ax.text(0.95, 0.05, f'{true_best:.2f} GWh',
        #        transform=ax.transAxes, ha='right', va='bottom',
        #        fontsize=9, color='C1', fontweight='bold')
        ax.text(0.95, 0.15, f'n={n_total}',
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=8, color='0.4')

        if idx >= 6:
            ax.set_xlabel('Number of starts', fontsize=10)
        if idx % 3 == 0:
            ax.set_ylabel('Best conservative AEP (GWh)', fontsize=10)

        # Add legend only to first subplot
        if idx == 0:
            ax.legend(loc='lower right', fontsize=7, ncol=2)

    fig.suptitle(
        'Bootstrap Convergence — Conservative AEP, A = 0.04 (Individual Farms)\n'
        f'{N_SAMPLES} shuffles per farm, capped at {MAX_SEEDS} seeds',
        fontsize=13)
    plt.tight_layout()

    out_path = INPUT_DIR / 'bootstrap_convergence_farms_A0.04.png'
    plt.savefig(out_path, dpi=150)
    print(f'\nSaved to {out_path}')


if __name__ == '__main__':
    main()

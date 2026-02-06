"""Verify A=0.02 pixwake results using PyWake.

This script loads optimized layouts from the A=0.02 analysis and evaluates them
using PyWake with the matching TurboGaussian configuration to verify Pareto points.

Configuration (matching dei-methodology.md):
- Model: TurboGaussianDeficit
- A: 0.02
- ct2a: ct2a_mom1d
- ctlim: 0.96
- superposition: SquaredSum
- use_effective_ws: False
- use_effective_ti: False
- Turbulence: None
- Ambient TI: 0.06
"""

import argparse
import h5py
import numpy as np
import pandas as pd
from pathlib import Path

from py_wake.wind_turbines.generic_wind_turbines import GenericWindTurbine
from py_wake.site import UniformSite
from py_wake.deficit_models.gaussian import TurboGaussianDeficit
from py_wake.deficit_models.utils import ct2a_mom1d
from py_wake.superposition_models import SquaredSum
from py_wake.wind_farm_models import PropagateDownwind
from py_wake.rotor_avg_models import RotorCenter


# Configuration
DEI_DIR = Path(__file__).parent.parent / "OMAE_neighbors"
WIND_DATA_FILE = DEI_DIR / "energy_island_10y_daily_av_wind.csv"
LAYOUTS_FILE = DEI_DIR / "re_precomputed_layouts.h5"


def load_wind_data():
    """Load DEI wind time series data."""
    df = pd.read_csv(WIND_DATA_FILE, sep=';')
    wd = df['WD_150'].values
    ws = df['WS_150'].values
    return wd, ws


def create_turbine():
    """Create turbine matching DEI specification."""
    return GenericWindTurbine(
        name='DEI_15MW',
        diameter=240,
        hub_height=150,
        power_norm=15000,  # kW
    )


def load_neighbor_layout(farm_idx, type_idx=5, seed=0):
    """Load a single neighbor farm layout."""
    with h5py.File(LAYOUTS_FILE, 'r') as f:
        key = f"farm{farm_idx}_t{type_idx}_s{seed}"
        if key in f:
            layout = f[key]['layout'][:]
            return layout[0], layout[1]
    return None, None


def load_all_neighbor_layouts():
    """Load all 9 neighbor farm layouts."""
    x_all = []
    y_all = []
    for farm_idx in range(1, 10):
        x, y = load_neighbor_layout(farm_idx)
        if x is not None:
            x_all.extend(x)
            y_all.extend(y)
    return np.array(x_all), np.array(y_all)


def load_optimized_layouts(layouts_file):
    """Load optimized layouts from pixwake analysis."""
    layouts = []
    with h5py.File(layouts_file, 'r') as f:
        n_layouts = f.attrs.get('n_layouts', 100)
        for i in range(n_layouts):
            key = f"layout_{i}"
            if key in f:
                grp = f[key]
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


def compute_aep_timeseries(wfm, x, y, wd_ts, ws_ts, ti_amb=0.06):
    """Compute AEP using full time series evaluation.

    This correctly computes E[P(v)] instead of P(E[v]).
    Uses time=True to treat wd/ws as paired time series values.
    """
    # Run simulation with time=True to treat wd/ws as paired time series
    # Result shape will be (n_turbines, n_timesteps)
    sim_result = wfm(x, y, wd=wd_ts, ws=ws_ts, time=True, TI=ti_amb)

    # Get power: shape is (n_turbines, n_timesteps)
    # NOTE: PyWake returns power in W (watts), not kW
    power_w = sim_result.Power.values  # W

    # Sum over turbines, then average over time steps
    total_power_per_step = power_w.sum(axis=0)  # W per time step (sum over turbines)
    avg_power_kw = total_power_per_step.mean() / 1000  # Convert W to kW

    # Convert to AEP (GWh/year)
    # Each sample represents one day (24 hours)
    aep_gwh = avg_power_kw * 8760 / 1e6

    return aep_gwh


def verify_farm(farm_idx, A=0.02, ti_amb=0.06):
    """Verify results for a specific farm."""
    print(f"\n{'='*70}")
    print(f"VERIFYING FARM {farm_idx} (A={A})")
    print("="*70)

    # Load wind data
    wd_ts, ws_ts = load_wind_data()
    print(f"Wind data: {len(wd_ts)} samples (10 years daily)")

    # Create turbine
    turbine = create_turbine()

    # Create site (uniform - we'll pass wd/ws directly)
    site = UniformSite(p_wd=[1], ti=ti_amb)

    # Create PyWake model matching methodology
    wake_deficit = TurboGaussianDeficit(
        ct2a=ct2a_mom1d,
        use_effective_ws=False,
        use_effective_ti=False,
        rotorAvgModel=RotorCenter(),
        ctlim=0.96,
        A=A,
    )
    wfm = PropagateDownwind(
        site, turbine,
        wake_deficitModel=wake_deficit,
        superpositionModel=SquaredSum(),
    )

    # Load neighbor layout
    x_neighbor, y_neighbor = load_neighbor_layout(farm_idx)
    if x_neighbor is None:
        print(f"No neighbor layout found for farm {farm_idx}")
        return None
    print(f"Neighbor: {len(x_neighbor)} turbines")

    # Load optimized layouts
    layouts_file = Path(f"analysis/dei_A{A:.2f}/layouts_farm{farm_idx}.h5")
    if not layouts_file.exists():
        print(f"Layouts file not found: {layouts_file}")
        return None

    layouts = load_optimized_layouts(layouts_file)
    print(f"Loaded {len(layouts)} layouts from pixwake analysis")

    # Evaluate each layout with PyWake
    print("\nEvaluating layouts with PyWake (this may take a while)...")
    pywake_results = []

    for i, layout in enumerate(layouts):
        # AEP without neighbor
        aep_absent = compute_aep_timeseries(
            wfm, layout['x'], layout['y'], wd_ts, ws_ts, ti_amb
        )

        # AEP with neighbor
        x_all = np.concatenate([layout['x'], x_neighbor])
        y_all = np.concatenate([layout['y'], y_neighbor])

        # Run with all turbines, extract target turbine power
        sim_all = wfm(x_all, y_all, wd=wd_ts, ws=ws_ts, time=True, TI=ti_amb)
        power_all_w = sim_all.Power.values  # shape: (n_turbines, n_timesteps), in W
        n_target = len(layout['x'])
        # Target turbines are first n_target in the array
        target_power_per_step = power_all_w[:n_target, :].sum(axis=0)  # sum over target turbines
        avg_target_power_kw = target_power_per_step.mean() / 1000  # W to kW
        aep_present = avg_target_power_kw * 8760 / 1e6

        pywake_results.append({
            'aep_absent': aep_absent,
            'aep_present': aep_present,
            'pixwake_absent': layout['aep_absent'],
            'pixwake_present': layout['aep_present'],
            'strategy': layout['strategy'],
        })

        if i % 20 == 0 or i == len(layouts) - 1:
            print(f"  Layout {i:3d}: PyWake={aep_absent:.1f}/{aep_present:.1f}, "
                  f"pixwake={layout['aep_absent']:.1f}/{layout['aep_present']:.1f} GWh")

    # Find Pareto front from PyWake results
    aep_absent = np.array([r['aep_absent'] for r in pywake_results])
    aep_present = np.array([r['aep_present'] for r in pywake_results])
    pareto_mask = find_pareto(aep_absent, aep_present)

    pareto_indices = np.where(pareto_mask)[0]
    if len(pareto_indices) == 0:
        print("No Pareto points found!")
        return None

    lib_opt_idx = pareto_indices[np.argmax(aep_absent[pareto_mask])]
    con_opt_idx = pareto_indices[np.argmax(aep_present[pareto_mask])]
    regret_pywake = aep_present[con_opt_idx] - aep_present[lib_opt_idx]

    # Get pixwake Pareto for comparison
    px_absent = np.array([r['pixwake_absent'] for r in pywake_results])
    px_present = np.array([r['pixwake_present'] for r in pywake_results])
    px_pareto = find_pareto(px_absent, px_present)
    px_pareto_idx = np.where(px_pareto)[0]
    px_lib_idx = px_pareto_idx[np.argmax(px_absent[px_pareto])]
    px_con_idx = px_pareto_idx[np.argmax(px_present[px_pareto])]
    regret_pixwake = px_present[px_con_idx] - px_present[px_lib_idx]

    print("\n" + "-"*70)
    print("RESULTS COMPARISON")
    print("-"*70)
    print(f"{'Metric':<30} {'PyWake':>15} {'pixwake':>15} {'Diff':>10}")
    print("-"*70)
    print(f"{'Pareto points':<30} {pareto_mask.sum():>15} {px_pareto.sum():>15}")
    print(f"{'Liberal-opt AEP (alone)':<30} {aep_absent[lib_opt_idx]:>15.1f} {px_absent[px_lib_idx]:>15.1f} {aep_absent[lib_opt_idx]-px_absent[px_lib_idx]:>10.1f}")
    print(f"{'Liberal-opt AEP (w/neighbor)':<30} {aep_present[lib_opt_idx]:>15.1f} {px_present[px_lib_idx]:>15.1f} {aep_present[lib_opt_idx]-px_present[px_lib_idx]:>10.1f}")
    print(f"{'Conserv-opt AEP (alone)':<30} {aep_absent[con_opt_idx]:>15.1f} {px_absent[px_con_idx]:>15.1f} {aep_absent[con_opt_idx]-px_absent[px_con_idx]:>10.1f}")
    print(f"{'Conserv-opt AEP (w/neighbor)':<30} {aep_present[con_opt_idx]:>15.1f} {px_present[px_con_idx]:>15.1f} {aep_present[con_opt_idx]-px_present[px_con_idx]:>10.1f}")
    print("-"*70)
    print(f"{'REGRET (GWh)':<30} {regret_pywake:>15.2f} {regret_pixwake:>15.2f} {regret_pywake-regret_pixwake:>10.2f}")
    print(f"{'REGRET (%)':<30} {regret_pywake/aep_present[con_opt_idx]*100:>14.2f}% {regret_pixwake/px_present[px_con_idx]*100:>14.2f}%")

    return {
        'farm_idx': farm_idx,
        'pywake_regret': regret_pywake,
        'pixwake_regret': regret_pixwake,
        'pywake_pareto': int(pareto_mask.sum()),
        'pixwake_pareto': int(px_pareto.sum()),
        'pywake_results': pywake_results,
    }


def verify_combined(A=0.02, ti_amb=0.06):
    """Verify combined (all neighbors) results."""
    print(f"\n{'='*70}")
    print(f"VERIFYING COMBINED CASE (A={A})")
    print("="*70)

    # Load wind data
    wd_ts, ws_ts = load_wind_data()
    print(f"Wind data: {len(wd_ts)} samples (10 years daily)")

    # Create turbine
    turbine = create_turbine()

    # Create site
    site = UniformSite(p_wd=[1], ti=ti_amb)

    # Create PyWake model
    wake_deficit = TurboGaussianDeficit(
        ct2a=ct2a_mom1d,
        use_effective_ws=False,
        use_effective_ti=False,
        rotorAvgModel=RotorCenter(),
        ctlim=0.96,
        A=A,
    )
    wfm = PropagateDownwind(
        site, turbine,
        wake_deficitModel=wake_deficit,
        superpositionModel=SquaredSum(),
    )

    # Load all neighbor layouts
    x_neighbor, y_neighbor = load_all_neighbor_layouts()
    print(f"All neighbors: {len(x_neighbor)} turbines")

    # Load optimized layouts
    layouts_file = Path(f"analysis/dei_A{A:.2f}/layouts_combined.h5")
    if not layouts_file.exists():
        print(f"Layouts file not found: {layouts_file}")
        return None

    layouts = load_optimized_layouts(layouts_file)
    print(f"Loaded {len(layouts)} layouts from pixwake analysis")

    # Evaluate each layout with PyWake
    print("\nEvaluating layouts with PyWake (this may take a while)...")
    pywake_results = []

    for i, layout in enumerate(layouts):
        # AEP without neighbor
        aep_absent = compute_aep_timeseries(
            wfm, layout['x'], layout['y'], wd_ts, ws_ts, ti_amb
        )

        # AEP with all neighbors
        x_all = np.concatenate([layout['x'], x_neighbor])
        y_all = np.concatenate([layout['y'], y_neighbor])
        sim_all = wfm(x_all, y_all, wd=wd_ts, ws=ws_ts, time=True, TI=ti_amb)
        power_all_w = sim_all.Power.values  # shape: (n_turbines, n_timesteps), in W
        n_target = len(layout['x'])
        target_power_per_step = power_all_w[:n_target, :].sum(axis=0)
        avg_target_power_kw = target_power_per_step.mean() / 1000  # W to kW
        aep_present = avg_target_power_kw * 8760 / 1e6

        pywake_results.append({
            'aep_absent': aep_absent,
            'aep_present': aep_present,
            'pixwake_absent': layout['aep_absent'],
            'pixwake_present': layout['aep_present'],
            'strategy': layout['strategy'],
        })

        if i % 20 == 0 or i == len(layouts) - 1:
            print(f"  Layout {i:3d}: PyWake={aep_absent:.1f}/{aep_present:.1f}, "
                  f"pixwake={layout['aep_absent']:.1f}/{layout['aep_present']:.1f} GWh")

    # Find Pareto front
    aep_absent = np.array([r['aep_absent'] for r in pywake_results])
    aep_present = np.array([r['aep_present'] for r in pywake_results])
    pareto_mask = find_pareto(aep_absent, aep_present)

    pareto_indices = np.where(pareto_mask)[0]
    lib_opt_idx = pareto_indices[np.argmax(aep_absent[pareto_mask])]
    con_opt_idx = pareto_indices[np.argmax(aep_present[pareto_mask])]
    regret_pywake = aep_present[con_opt_idx] - aep_present[lib_opt_idx]

    # Get pixwake Pareto
    px_absent = np.array([r['pixwake_absent'] for r in pywake_results])
    px_present = np.array([r['pixwake_present'] for r in pywake_results])
    px_pareto = find_pareto(px_absent, px_present)
    px_pareto_idx = np.where(px_pareto)[0]
    px_lib_idx = px_pareto_idx[np.argmax(px_absent[px_pareto])]
    px_con_idx = px_pareto_idx[np.argmax(px_present[px_pareto])]
    regret_pixwake = px_present[px_con_idx] - px_present[px_lib_idx]

    print("\n" + "-"*70)
    print("COMBINED RESULTS COMPARISON")
    print("-"*70)
    print(f"{'Metric':<30} {'PyWake':>15} {'pixwake':>15} {'Diff':>10}")
    print("-"*70)
    print(f"{'Pareto points':<30} {pareto_mask.sum():>15} {px_pareto.sum():>15}")
    print(f"{'Liberal-opt AEP (alone)':<30} {aep_absent[lib_opt_idx]:>15.1f} {px_absent[px_lib_idx]:>15.1f} {aep_absent[lib_opt_idx]-px_absent[px_lib_idx]:>10.1f}")
    print(f"{'Liberal-opt AEP (w/neighbors)':<30} {aep_present[lib_opt_idx]:>15.1f} {px_present[px_lib_idx]:>15.1f} {aep_present[lib_opt_idx]-px_present[px_lib_idx]:>10.1f}")
    print(f"{'Conserv-opt AEP (alone)':<30} {aep_absent[con_opt_idx]:>15.1f} {px_absent[px_con_idx]:>15.1f} {aep_absent[con_opt_idx]-px_absent[px_con_idx]:>10.1f}")
    print(f"{'Conserv-opt AEP (w/neighbors)':<30} {aep_present[con_opt_idx]:>15.1f} {px_present[px_con_idx]:>15.1f} {aep_present[con_opt_idx]-px_present[px_con_idx]:>10.1f}")
    print("-"*70)
    print(f"{'REGRET (GWh)':<30} {regret_pywake:>15.2f} {regret_pixwake:>15.2f} {regret_pywake-regret_pixwake:>10.2f}")
    print(f"{'REGRET (%)':<30} {regret_pywake/aep_present[con_opt_idx]*100:>14.2f}% {regret_pixwake/px_present[px_con_idx]*100:>14.2f}%")

    return {
        'pywake_regret': regret_pywake,
        'pixwake_regret': regret_pixwake,
        'pywake_pareto': int(pareto_mask.sum()),
        'pixwake_pareto': int(px_pareto.sum()),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify A=0.02 results with PyWake")
    parser.add_argument("--farm", type=int, default=8,
                        help="Farm index to verify (1-9, default: 8)")
    parser.add_argument("--combined", action="store_true",
                        help="Verify combined case instead of individual farm")
    parser.add_argument("--A", type=float, default=0.02,
                        help="Wake expansion coefficient (default: 0.02)")

    args = parser.parse_args()

    if args.combined:
        verify_combined(A=args.A)
    else:
        verify_farm(args.farm, A=args.A)

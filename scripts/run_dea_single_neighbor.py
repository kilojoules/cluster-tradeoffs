"""Test DEA case with each individual neighboring wind farm separately.

This script tests whether having just ONE neighboring wind farm (instead of all 9)
leads to design regret. If the symmetric "ring" geometry is what eliminates tradeoffs,
then individual neighbors from asymmetric directions should create regret.

Uses actual gradient-based optimization to find liberal vs conservative optimal layouts.

Usage:
    pixi run python scripts/run_dea_single_neighbor.py
    pixi run python scripts/run_dea_single_neighbor.py --wake-model=turbopark
    pixi run python scripts/run_dea_single_neighbor.py --n-starts=5 --max-iter=500
"""

import argparse
import json
import time
from functools import partial
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from pixwake import Curve, Turbine, WakeSimulation
from pixwake.deficit import BastankhahGaussianDeficit, TurboGaussianDeficit
from pixwake.turbulence import CrespoHernandez


# =============================================================================
# DEA Study Configuration
# =============================================================================

DEA_DIR = Path(__file__).parent.parent / "DEA_neighbors"
WIND_DATA_FILE = DEA_DIR / "energy_island_10y_daily_av_wind.csv"
LAYOUTS_FILE = DEA_DIR / "re_precomputed_layouts.h5"

TARGET_FARM_IDX = 0
TARGET_N_TURBINES = 66
TARGET_RATED_POWER = 15  # MW
TARGET_ROTOR_DIAMETER = 240  # m
TARGET_HUB_HEIGHT = 150  # m

N_NEIGHBOR_FARMS = 9

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


def load_wind_data():
    """Load DEA wind time series data."""
    df = pd.read_csv(WIND_DATA_FILE, sep=';')
    wd = df['WD_150'].values
    ws = df['WS_150'].values
    return wd, ws


def compute_full_timeseries_arrays(wd_ts, ws_ts):
    """Return full time series as arrays for simulation."""
    return jnp.array(wd_ts), jnp.array(ws_ts)


def load_neighbor_layout(farm_idx, type_idx=5, seed=0):
    """Load a single neighbor farm layout."""
    with h5py.File(LAYOUTS_FILE, 'r') as f:
        key = f"farm{farm_idx}_t{type_idx}_s{seed}"
        if key in f:
            layout = f[key]['layout'][:]
            return layout[0], layout[1]
    return None, None


def load_target_boundary():
    """Load target farm boundary."""
    dk0w_tender_3 = np.array([
        706694.3923283464, 6224158.532895836,
        703972.0844905999, 6226906.597455995,
        702624.6334635273, 6253853.5386425415,
        712771.6248419734, 6257704.934445341,
        715639.3355871611, 6260664.6846508905,
        721593.2420745814, 6257906.998015941,
    ]).reshape((-1, 2)).T
    return dk0w_tender_3


def create_dea_turbine():
    """Create turbine matching DEA specification."""
    ws = jnp.array([0.0, 4.0, 10.0, 15.0, 25.0])
    power = jnp.array([0.0, 0.0, 15000.0, 15000.0, 0.0])
    ct = jnp.array([0.0, 0.8, 0.8, 0.4, 0.0])

    return Turbine(
        rotor_diameter=float(TARGET_ROTOR_DIAMETER),
        hub_height=float(TARGET_HUB_HEIGHT),
        power_curve=Curve(ws=ws, values=power),
        ct_curve=Curve(ws=ws, values=ct),
    )


def compute_neighbor_direction(target_boundary, x_neighbor, y_neighbor):
    """Compute the direction from target farm center to neighbor farm center."""
    target_center_x = target_boundary[0].mean()
    target_center_y = target_boundary[1].mean()

    neighbor_center_x = np.mean(x_neighbor)
    neighbor_center_y = np.mean(y_neighbor)

    dx = neighbor_center_x - target_center_x
    dy = neighbor_center_y - target_center_y

    direction = np.degrees(np.arctan2(dx, dy)) % 360  # North = 0°
    distance = np.sqrt(dx**2 + dy**2) / 1000  # km

    return direction, distance


def compute_binned_wind_rose(wd_ts, ws_ts, n_bins=24):
    """Compute binned wind rose from time series data."""
    bin_edges = np.linspace(0, 360, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    weights = np.zeros(n_bins)
    mean_speeds = np.zeros(n_bins)

    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (wd_ts >= bin_edges[i]) | (wd_ts < bin_edges[0])
        else:
            mask = (wd_ts >= bin_edges[i]) & (wd_ts < bin_edges[i+1])

        weights[i] = mask.sum()
        if mask.sum() > 0:
            mean_speeds[i] = ws_ts[mask].mean()
        else:
            mean_speeds[i] = ws_ts.mean()

    weights = weights / weights.sum()

    return jnp.array(bin_centers), jnp.array(mean_speeds), jnp.array(weights)


def run_single_neighbor_analysis(
    wake_model: str = "bastankhah",
    ti_amb: float = 0.06,
    n_starts: int = 5,
    max_iter: int = 500,
    output_dir: str = "analysis/dea_single_neighbor",
):
    """Run DEA analysis with each individual neighbor using gradient-based optimization."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"DEA SINGLE NEIGHBOR ANALYSIS ({wake_model.upper()}) - OPTIMIZED")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    wd_ts, ws_ts = load_wind_data()

    # Use binned wind rose for faster optimization (24 directions)
    wd, ws, weights = compute_binned_wind_rose(wd_ts, ws_ts, n_bins=24)
    print(f"Binned wind rose: {len(wd)} directions")
    print(f"Dominant direction: {float(wd[jnp.argmax(weights)]):.0f}° ({float(jnp.max(weights)*100):.1f}% frequency)")

    target_boundary = load_target_boundary()
    turbine = create_dea_turbine()

    # Create wake model
    if wake_model == "turbopark":
        deficit = TurboGaussianDeficit(A=0.02)
        sim = WakeSimulation(turbine, deficit, turbulence=CrespoHernandez())
        ti_array = jnp.full_like(ws, ti_amb)
    else:
        deficit = BastankhahGaussianDeficit(k=0.04)
        sim = WakeSimulation(turbine, deficit)
        ti_array = None

    # Get boundary extent for layout generation
    x_min, x_max = float(target_boundary[0].min()), float(target_boundary[0].max())
    y_min, y_max = float(target_boundary[1].min()), float(target_boundary[1].max())
    D = TARGET_ROTOR_DIAMETER

    def generate_layout(seed):
        """Generate random initial layout."""
        rng = np.random.default_rng(seed)
        x = rng.uniform(x_min, x_max, TARGET_N_TURBINES)
        y = rng.uniform(y_min, y_max, TARGET_N_TURBINES)
        return jnp.array(x), jnp.array(y)

    def compute_aep_binned(x_target, y_target, x_neighbor=None, y_neighbor=None):
        """Compute AEP using binned wind rose (for optimization)."""
        if x_neighbor is not None:
            x_all = jnp.concatenate([x_target, jnp.array(x_neighbor)])
            y_all = jnp.concatenate([y_target, jnp.array(y_neighbor)])
        else:
            x_all = x_target
            y_all = y_target

        n_target = len(x_target)

        if ti_array is not None:
            result = sim(x_all, y_all, ws_amb=ws, wd_amb=wd, ti_amb=ti_array)
        else:
            result = sim(x_all, y_all, ws_amb=ws, wd_amb=wd)

        power = result.power()
        target_power = power[:, :n_target].sum(axis=1)

        # Weighted AEP (GWh)
        aep_gwh = jnp.sum(target_power * weights) * 8760 / 1e6

        return aep_gwh

    def make_optimizer(x_neighbor_arr=None, y_neighbor_arr=None):
        """Create a JIT-compiled optimizer for given neighbor configuration."""

        if x_neighbor_arr is not None:
            x_neigh = jnp.array(x_neighbor_arr)
            y_neigh = jnp.array(y_neighbor_arr)

            def loss_fn(x, y):
                x_all = jnp.concatenate([x, x_neigh])
                y_all = jnp.concatenate([y, y_neigh])
                n_target = len(x)
                if ti_array is not None:
                    result = sim(x_all, y_all, ws_amb=ws, wd_amb=wd, ti_amb=ti_array)
                else:
                    result = sim(x_all, y_all, ws_amb=ws, wd_amb=wd)
                power = result.power()
                target_power = power[:, :n_target].sum(axis=1)
                aep = jnp.sum(target_power * weights) * 8760 / 1e6
                return -aep
        else:
            def loss_fn(x, y):
                if ti_array is not None:
                    result = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=ti_array)
                else:
                    result = sim(x, y, ws_amb=ws, wd_amb=wd)
                power = result.power()
                target_power = power.sum(axis=1)
                aep = jnp.sum(target_power * weights) * 8760 / 1e6
                return -aep

        grad_fn = jax.grad(loss_fn, argnums=(0, 1))

        @partial(jax.jit, static_argnums=(2,))
        def run_adam(x_init, y_init, max_iterations, lr):
            beta1, beta2, eps = 0.9, 0.999, 1e-8

            def adam_step(carry, t):
                x, y, m_x, m_y, v_x, v_y = carry
                t_float = t.astype(jnp.float64) + 1.0

                grad_x, grad_y = grad_fn(x, y)
                grad_x = jnp.clip(grad_x, -100, 100)
                grad_y = jnp.clip(grad_y, -100, 100)

                m_x = beta1 * m_x + (1 - beta1) * grad_x
                m_y = beta1 * m_y + (1 - beta1) * grad_y
                v_x = beta2 * v_x + (1 - beta2) * grad_x**2
                v_y = beta2 * v_y + (1 - beta2) * grad_y**2

                m_x_hat = m_x / (1 - beta1**t_float)
                m_y_hat = m_y / (1 - beta1**t_float)
                v_x_hat = v_x / (1 - beta2**t_float)
                v_y_hat = v_y / (1 - beta2**t_float)

                x = x - lr * m_x_hat / (jnp.sqrt(v_x_hat) + eps)
                y = y - lr * m_y_hat / (jnp.sqrt(v_y_hat) + eps)

                x = jnp.clip(x, x_min + D/2, x_max - D/2)
                y = jnp.clip(y, y_min + D/2, y_max - D/2)

                return (x, y, m_x, m_y, v_x, v_y), None

            m_x = jnp.zeros_like(x_init)
            m_y = jnp.zeros_like(y_init)
            v_x = jnp.zeros_like(x_init)
            v_y = jnp.zeros_like(y_init)

            init_carry = (x_init, y_init, m_x, m_y, v_x, v_y)
            (x_opt, y_opt, _, _, _, _), _ = jax.lax.scan(adam_step, init_carry, jnp.arange(max_iterations))

            final_aep = -loss_fn(x_opt, y_opt)
            return x_opt, y_opt, final_aep

        return run_adam

    # Pre-compile optimizer without neighbors (liberal)
    print("Compiling liberal optimizer...", flush=True)
    liberal_optimizer = make_optimizer(None, None)

    def optimize_layout(x_init, y_init, x_neighbor=None, y_neighbor=None, max_iterations=500, lr=50.0):
        """Optimize layout using JIT-compiled Adam."""
        if x_neighbor is not None:
            optimizer = make_optimizer(x_neighbor, y_neighbor)
        else:
            optimizer = liberal_optimizer

        x_opt, y_opt, aep = optimizer(x_init, y_init, max_iterations, lr)
        return x_opt, y_opt, float(aep)

    # Test each neighbor individually
    results = {}

    print("\n" + "=" * 70)
    print(f"Testing each neighbor farm individually (n_starts={n_starts}, max_iter={max_iter})")
    print("=" * 70)

    for farm_idx in range(1, N_NEIGHBOR_FARMS + 1):
        x_neighbor, y_neighbor = load_neighbor_layout(farm_idx)
        if x_neighbor is None:
            print(f"Skipping farm {farm_idx} - no layout found")
            continue

        direction, distance = compute_neighbor_direction(target_boundary, x_neighbor, y_neighbor)
        farm_name = FARM_NAMES.get(farm_idx, f"Farm {farm_idx}")

        print(f"\n--- Farm {farm_idx}: {farm_name} ---")
        print(f"Direction: {direction:.0f}°, Distance: {distance:.1f} km, Turbines: {len(x_neighbor)}")

        start_time = time.time()

        # Run multi-start optimization
        liberal_layouts = []  # Optimize without neighbor
        conservative_layouts = []  # Optimize with neighbor

        for seed in range(n_starts):
            x0, y0 = generate_layout(seed)

            # Liberal: optimize ignoring neighbor
            x_lib, y_lib, aep_lib_absent = optimize_layout(x0, y0, None, None, max_iter)
            aep_lib_present = float(compute_aep_binned(x_lib, y_lib, x_neighbor, y_neighbor))
            liberal_layouts.append({
                'x': x_lib, 'y': y_lib,
                'aep_absent': float(aep_lib_absent),
                'aep_present': aep_lib_present,
            })

            # Conservative: optimize considering neighbor
            x_con, y_con, aep_con_present = optimize_layout(x0, y0, x_neighbor, y_neighbor, max_iter)
            aep_con_absent = float(compute_aep_binned(x_con, y_con, None, None))
            conservative_layouts.append({
                'x': x_con, 'y': y_con,
                'aep_absent': aep_con_absent,
                'aep_present': float(aep_con_present),
            })

            print(f"  Start {seed}: Liberal={aep_lib_absent:.1f}/{aep_lib_present:.1f}, "
                  f"Conservative={aep_con_absent:.1f}/{aep_con_present:.1f} GWh")

        elapsed = time.time() - start_time

        # Compute Pareto and regret from all optimized layouts
        all_layouts = liberal_layouts + conservative_layouts
        aep_absent = np.array([r['aep_absent'] for r in all_layouts])
        aep_present = np.array([r['aep_present'] for r in all_layouts])

        # Find Pareto front
        pareto_mask = np.zeros(len(all_layouts), dtype=bool)
        for i in range(len(all_layouts)):
            dominated = False
            for j in range(len(all_layouts)):
                if i != j:
                    if (aep_absent[j] >= aep_absent[i] and aep_present[j] > aep_present[i]) or \
                       (aep_absent[j] > aep_absent[i] and aep_present[j] >= aep_present[i]):
                        dominated = True
                        break
            if not dominated:
                pareto_mask[i] = True

        pareto_indices = np.where(pareto_mask)[0]
        if len(pareto_indices) > 0:
            lib_opt_idx = pareto_indices[np.argmax(aep_absent[pareto_mask])]
            con_opt_idx = pareto_indices[np.argmax(aep_present[pareto_mask])]
            regret = aep_present[con_opt_idx] - aep_present[lib_opt_idx]
        else:
            regret = 0.0

        n_pareto = pareto_mask.sum()

        print(f"Pareto points: {n_pareto}")
        print(f"Liberal-optimal: {aep_absent[lib_opt_idx]:.1f} / {aep_present[lib_opt_idx]:.1f} GWh")
        print(f"Conservative-optimal: {aep_absent[con_opt_idx]:.1f} / {aep_present[con_opt_idx]:.1f} GWh")
        print(f"REGRET: {regret:.2f} GWh ({regret/aep_present[con_opt_idx]*100:.2f}%)")
        print(f"Time: {elapsed:.1f}s")

        results[farm_idx] = {
            "name": farm_name,
            "direction": float(direction),
            "distance_km": float(distance),
            "n_turbines": len(x_neighbor),
            "n_pareto": int(n_pareto),
            "regret_gwh": float(regret),
            "regret_pct": float(regret/aep_present[con_opt_idx]*100) if aep_present[con_opt_idx] > 0 else 0.0,
            "lib_opt_absent": float(aep_absent[lib_opt_idx]),
            "lib_opt_present": float(aep_present[lib_opt_idx]),
            "con_opt_absent": float(aep_absent[con_opt_idx]),
            "con_opt_present": float(aep_present[con_opt_idx]),
            "elapsed_seconds": elapsed,
        }

    # Also test all neighbors together for comparison
    print("\n--- ALL 9 NEIGHBORS TOGETHER ---")
    x_all_neighbors = []
    y_all_neighbors = []
    for farm_idx in range(1, N_NEIGHBOR_FARMS + 1):
        x_n, y_n = load_neighbor_layout(farm_idx)
        if x_n is not None:
            x_all_neighbors.extend(x_n)
            y_all_neighbors.extend(y_n)

    x_all_neighbors = np.array(x_all_neighbors)
    y_all_neighbors = np.array(y_all_neighbors)

    start_time = time.time()

    all_layouts = []
    for seed in range(n_starts):
        x0, y0 = generate_layout(seed)

        # Liberal
        x_lib, y_lib, aep_lib_absent = optimize_layout(x0, y0, None, None, max_iter)
        aep_lib_present = float(compute_aep_binned(x_lib, y_lib, x_all_neighbors, y_all_neighbors))
        all_layouts.append({'aep_absent': float(aep_lib_absent), 'aep_present': aep_lib_present})

        # Conservative
        x_con, y_con, aep_con_present = optimize_layout(x0, y0, x_all_neighbors, y_all_neighbors, max_iter)
        aep_con_absent = float(compute_aep_binned(x_con, y_con, None, None))
        all_layouts.append({'aep_absent': aep_con_absent, 'aep_present': float(aep_con_present)})

        print(f"  Start {seed}: Liberal={aep_lib_absent:.1f}/{aep_lib_present:.1f}, "
              f"Conservative={aep_con_absent:.1f}/{aep_con_present:.1f} GWh")

    elapsed = time.time() - start_time

    aep_absent = np.array([r['aep_absent'] for r in all_layouts])
    aep_present = np.array([r['aep_present'] for r in all_layouts])

    # Pareto
    pareto_mask = np.zeros(len(all_layouts), dtype=bool)
    for i in range(len(all_layouts)):
        dominated = False
        for j in range(len(all_layouts)):
            if i != j:
                if (aep_absent[j] >= aep_absent[i] and aep_present[j] > aep_present[i]) or \
                   (aep_absent[j] > aep_absent[i] and aep_present[j] >= aep_present[i]):
                    dominated = True
                    break
        if not dominated:
            pareto_mask[i] = True

    pareto_indices = np.where(pareto_mask)[0]
    if len(pareto_indices) > 0:
        lib_opt_idx = pareto_indices[np.argmax(aep_absent[pareto_mask])]
        con_opt_idx = pareto_indices[np.argmax(aep_present[pareto_mask])]
        all_regret = aep_present[con_opt_idx] - aep_present[lib_opt_idx]
    else:
        all_regret = 0.0

    print(f"Pareto points: {pareto_mask.sum()}")
    print(f"REGRET (all 9): {all_regret:.2f} GWh")

    results["all_neighbors"] = {
        "name": "All 9 neighbors",
        "n_pareto": int(pareto_mask.sum()),
        "regret_gwh": float(all_regret),
        "total_turbines": len(x_all_neighbors),
        "elapsed_seconds": elapsed,
    }

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Farm':<30} {'Direction':>10} {'Regret (GWh)':>15}")
    print("-" * 70)

    for farm_idx in range(1, N_NEIGHBOR_FARMS + 1):
        if farm_idx in results:
            r = results[farm_idx]
            print(f"{r['name']:<30} {r['direction']:>10.0f}° {r['regret_gwh']:>15.2f}")

    print("-" * 70)
    print(f"{'All 9 neighbors':<30} {'(ring)':<10} {results['all_neighbors']['regret_gwh']:>15.2f}")

    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': 'polar'})

    farm_indices = [i for i in range(1, N_NEIGHBOR_FARMS + 1) if i in results]
    directions_rad = [np.radians(results[i]['direction']) for i in farm_indices]
    regrets = [results[i]['regret_gwh'] for i in farm_indices]

    # Plot bars
    bars = ax.bar(directions_rad, regrets, width=0.3, alpha=0.7, color='steelblue', edgecolor='black')

    # Add labels
    for i, (d, r, idx) in enumerate(zip(directions_rad, regrets, farm_indices)):
        ax.annotate(f'Farm {idx}\n{r:.1f} GWh',
                   xy=(d, r),
                   xytext=(d, r + max(regrets)*0.1),
                   ha='center', va='bottom', fontsize=9)

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_title(f'DEA Single Neighbor Regret by Direction ({wake_model.upper()})\n(All 9 together: {all_regret:.2f} GWh)',
                 fontsize=14, pad=20)
    ax.set_ylabel('Regret (GWh)', labelpad=30)

    plt.tight_layout()
    fig.savefig(output_path / f'dea_single_neighbor_{wake_model}.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {output_path / f'dea_single_neighbor_{wake_model}.png'}")
    plt.close(fig)

    # Save results
    with open(output_path / f'dea_single_neighbor_{wake_model}.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test DEA with individual neighbors (optimized)")
    parser.add_argument("--wake-model", type=str, default="bastankhah",
                        choices=["bastankhah", "turbopark"])
    parser.add_argument("--ti", type=float, default=0.06,
                        help="Ambient turbulence intensity (for turbopark)")
    parser.add_argument("--n-starts", type=int, default=5,
                        help="Number of optimization starts per strategy")
    parser.add_argument("--max-iter", type=int, default=500,
                        help="Maximum optimization iterations per start")
    parser.add_argument("--output-dir", "-o", type=str, default="analysis/dea_single_neighbor")

    args = parser.parse_args()

    run_single_neighbor_analysis(
        wake_model=args.wake_model,
        ti_amb=args.ti,
        n_starts=args.n_starts,
        max_iter=args.max_iter,
        output_dir=args.output_dir,
    )

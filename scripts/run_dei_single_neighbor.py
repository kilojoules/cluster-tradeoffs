"""Test DEI case with each individual neighboring wind farm separately.

This script tests whether having just ONE neighboring wind farm (instead of all 9)
leads to design regret. If the symmetric "ring" geometry is what eliminates tradeoffs,
then individual neighbors from asymmetric directions should create regret.

Uses actual gradient-based optimization to find liberal vs conservative optimal layouts.

Usage:
    pixi run python scripts/run_dei_single_neighbor.py
    pixi run python scripts/run_dei_single_neighbor.py --wake-model=turbopark
    pixi run python scripts/run_dei_single_neighbor.py --n-starts=5 --max-iter=500
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
from pixwake.superposition import SquaredSum
from pixwake.utils import ct2a_mom1d


# =============================================================================
# DEI Study Configuration
# =============================================================================

DEI_DIR = Path(__file__).parent.parent / "OMAE_neighbors"
WIND_DATA_FILE = DEI_DIR / "energy_island_10y_daily_av_wind.csv"
LAYOUTS_FILE = DEI_DIR / "re_precomputed_layouts.h5"

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
    """Load DEI wind time series data."""
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


def create_dei_turbine():
    """Create turbine matching DEI specification.

    Uses EXACT power/CT curves from PyWake's GenericWindTurbine(diameter=240, hub_height=150, power_norm=15000).
    This ensures pixwake and PyWake produce identical results.
    """
    # Exact curves extracted from PyWake GenericWindTurbine at 1 m/s resolution (high precision)
    ws = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                    13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0])

    # Power in kW (extracted from PyWake GenericWindTurbine, high precision)
    power = jnp.array([
        0.0000000000, 0.0000000000, 2.3985813556, 209.2580932713, 689.1977407908, 1480.6084745032,
        2661.2377234914, 4308.9290345041, 6501.0566168182, 9260.5162758057, 12081.4039288222, 13937.2966269349,
        14705.0159806425, 14931.0392395129, 14985.2085175128, 14996.9062345265, 14999.3433209739, 14999.8550035258,
        14999.9662091698, 14999.9916237998, 14999.9977839699, 14999.9993738862, 14999.9998112694, 14999.9999393881,
        14999.9999792501, 14999.9999923911,
    ])

    # CT curve (extracted from PyWake GenericWindTurbine, high precision)
    ct = jnp.array([
        0.8888888889, 0.8888888889, 0.8888888889, 0.8003233124, 0.8000001158, 0.8000000002,
        0.8000000000, 0.7999999845, 0.7999170517, 0.7930486829, 0.7353646478, 0.6099800647,
        0.4763545657, 0.3698364821, 0.2915403905, 0.2340695174, 0.1910465178, 0.1581411392,
        0.1324922850, 0.1121722012, 0.0958466811, 0.0825690747, 0.0716530730, 0.0625917527,
        0.0550045472, 0.0486016164,
    ])

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
    """Compute binned wind rose from time series data.

    NOTE: This is used for optimization only. For final AEP evaluation,
    use compute_aep_full_timeseries() which correctly handles the non-linear
    power curve by computing E[P(v)] instead of P(E[v]).
    """
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


def create_full_timeseries_aep_evaluator(sim, wd_ts, ws_ts, ti_amb=None, batch_size=500):
    """Create a function to evaluate AEP using full time series.

    This correctly computes E[P(v)] by evaluating power at each time step,
    avoiding the P(E[v]) error that occurs with binned wind roses.

    Args:
        sim: WakeSimulation instance
        wd_ts: Wind direction time series (numpy array)
        ws_ts: Wind speed time series (numpy array)
        ti_amb: Ambient turbulence intensity (scalar or None)
        batch_size: Number of time steps per batch (to manage memory)

    Returns:
        Function that computes AEP given turbine positions
    """
    n_samples = len(wd_ts)
    n_batches = (n_samples + batch_size - 1) // batch_size

    # Pre-convert to JAX arrays
    wd_full = jnp.array(wd_ts)
    ws_full = jnp.array(ws_ts)

    def compute_aep(x_target, y_target, x_neighbor=None, y_neighbor=None):
        """Compute AEP using full time series evaluation."""
        if x_neighbor is not None:
            x_all = jnp.concatenate([x_target, jnp.array(x_neighbor)])
            y_all = jnp.concatenate([y_target, jnp.array(y_neighbor)])
        else:
            x_all = x_target
            y_all = y_target

        n_target = len(x_target)
        total_power = 0.0

        # Process in batches to avoid memory issues
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_samples)

            wd_batch = wd_full[start_idx:end_idx]
            ws_batch = ws_full[start_idx:end_idx]

            if ti_amb is not None:
                ti_batch = jnp.full_like(ws_batch, ti_amb)
                result = sim(x_all, y_all, ws_amb=ws_batch, wd_amb=wd_batch, ti_amb=ti_batch)
            else:
                result = sim(x_all, y_all, ws_amb=ws_batch, wd_amb=wd_batch)

            power = result.power()  # shape: (n_conditions, n_turbines)
            target_power = power[:, :n_target].sum(axis=1)  # sum over target turbines
            total_power += float(target_power.sum())  # sum over time steps in batch

        # Convert to AEP (GWh/year)
        # Each sample represents one day (24 hours)
        # total_power is in kW summed over all samples
        # AEP = (total_power / n_samples) * 8760 hours/year / 1e6 kW->GWh
        avg_power = total_power / n_samples  # kW
        aep_gwh = avg_power * 8760 / 1e6

        return aep_gwh

    return compute_aep


def run_single_neighbor_analysis(
    wake_model: str = "bastankhah",
    ti_amb: float = 0.06,
    A: float = 0.04,
    n_starts: int = 5,
    max_iter: int = 500,
    output_dir: str = "analysis/dei_single_neighbor",
    farm_indices: list[int] | None = None,
    skip_combined: bool = False,
    seed_offset: int = 0,
):
    """Run DEI analysis with each individual neighbor using gradient-based optimization.

    Args:
        wake_model: Wake model to use ("bastankhah" or "turbopark")
        ti_amb: Ambient turbulence intensity (for turbopark)
        A: Wake expansion coefficient (for turbopark, default 0.04)
        n_starts: Number of optimization starts per strategy
        max_iter: Maximum optimization iterations per start
        output_dir: Output directory for results
        farm_indices: List of farm indices to run (1-9). If None, runs all farms. If empty list, skips individual farms.
        skip_combined: If True, skip the "all neighbors combined" analysis
        seed_offset: Starting seed number (for appending additional runs)
    """
    if farm_indices is None:
        farm_indices = list(range(1, N_NEIGHBOR_FARMS + 1))
    # If farm_indices is an empty list, we skip individual farms (for --only-combined)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"DEI SINGLE NEIGHBOR ANALYSIS ({wake_model.upper()}, A={A}) - OPTIMIZED")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    wd_ts, ws_ts = load_wind_data()

    # Use binned wind rose for faster optimization (24 directions)
    wd, ws, weights = compute_binned_wind_rose(wd_ts, ws_ts, n_bins=24)
    print(f"Binned wind rose: {len(wd)} directions")
    print(f"Dominant direction: {float(wd[jnp.argmax(weights)]):.0f}° ({float(jnp.max(weights)*100):.1f}% frequency)")

    target_boundary = load_target_boundary()
    turbine = create_dei_turbine()

    # Create wake model
    # Default is now TurboGaussian to match PyWake/OMAE pipeline
    if wake_model == "bastankhah":
        deficit = BastankhahGaussianDeficit(k=0.04)
        sim = WakeSimulation(turbine, deficit)
        ti_array = None
        ti_scalar = None
    else:  # turbopark (default) - matches PyWake Nygaard_2022 literature defaults
        deficit = TurboGaussianDeficit(
            A=A,  # Wake expansion coefficient (default 0.04 for Nygaard_2022)
            ct2a=ct2a_mom1d,
            ctlim=0.96,
            use_effective_ws=False,  # Nygaard_2022 uses ambient WS
            use_effective_ti=False,
            superposition=SquaredSum(),  # Nygaard_2022 default
        )
        # Nygaard_2022 doesn't use turbulence model, but TI is still required
        sim = WakeSimulation(turbine, deficit)
        ti_array = jnp.full_like(ws, ti_amb)
        ti_scalar = ti_amb

    # Create full time series evaluator for accurate AEP calculation
    # (The binned wind rose is still used for optimization speed)
    print("Creating full time series AEP evaluator...")
    compute_aep_full = create_full_timeseries_aep_evaluator(
        sim, wd_ts, ws_ts, ti_amb=ti_scalar, batch_size=500
    )
    print(f"Full time series: {len(wd_ts)} samples (10 years daily data)")

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
    print(f"Farms to run: {farm_indices}")
    print("=" * 70)

    for farm_idx in farm_indices:
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

        for seed in range(seed_offset, seed_offset + n_starts):
            x0, y0 = generate_layout(seed)

            # Liberal: optimize ignoring neighbor
            x_lib, y_lib, _ = optimize_layout(x0, y0, None, None, max_iter)
            # Re-evaluate with full time series for accurate AEP
            aep_lib_absent = compute_aep_full(x_lib, y_lib, None, None)
            aep_lib_present = compute_aep_full(x_lib, y_lib, x_neighbor, y_neighbor)
            liberal_layouts.append({
                'x': np.array(x_lib), 'y': np.array(y_lib),
                'aep_absent': float(aep_lib_absent),
                'aep_present': float(aep_lib_present),
                'seed': seed,
                'strategy': 'liberal',
            })

            # Conservative: optimize considering neighbor
            x_con, y_con, _ = optimize_layout(x0, y0, x_neighbor, y_neighbor, max_iter)
            # Re-evaluate with full time series for accurate AEP
            aep_con_absent = compute_aep_full(x_con, y_con, None, None)
            aep_con_present = compute_aep_full(x_con, y_con, x_neighbor, y_neighbor)
            conservative_layouts.append({
                'x': np.array(x_con), 'y': np.array(y_con),
                'aep_absent': float(aep_con_absent),
                'aep_present': float(aep_con_present),
                'seed': seed,
                'strategy': 'conservative',
            })

            print(f"  Start {seed}: Liberal={aep_lib_absent:.1f}/{aep_lib_present:.1f}, "
                  f"Conservative={aep_con_absent:.1f}/{aep_con_present:.1f} GWh", flush=True)

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

        # Save layouts for this farm (append if file exists)
        layouts_file = output_path / f"layouts_farm{farm_idx}.h5"
        mode = 'a' if layouts_file.exists() and seed_offset > 0 else 'w'
        with h5py.File(layouts_file, mode) as hf:
            # Count existing layouts to determine starting index
            existing_count = len([k for k in hf.keys() if k.startswith('layout_')]) if mode == 'a' else 0
            for i, layout in enumerate(all_layouts):
                layout_idx = existing_count + i
                grp = hf.create_group(f"layout_{layout_idx}")
                grp.create_dataset('x', data=layout['x'])
                grp.create_dataset('y', data=layout['y'])
                grp.attrs['aep_absent'] = layout['aep_absent']
                grp.attrs['aep_present'] = layout['aep_present']
                grp.attrs['seed'] = layout['seed']
                grp.attrs['strategy'] = layout['strategy']
            hf.attrs['farm_idx'] = farm_idx
            hf.attrs['n_layouts'] = existing_count + len(all_layouts)
        print(f"Saved {len(all_layouts)} layouts to {layouts_file} (total: {existing_count + len(all_layouts)})")

    # Also test all neighbors together for comparison
    if skip_combined:
        print("\n--- SKIPPING ALL 9 NEIGHBORS TOGETHER (--skip-combined) ---")
        all_regret = None
    else:
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
        for seed in range(seed_offset, seed_offset + n_starts):
            x0, y0 = generate_layout(seed)

            # Liberal
            x_lib, y_lib, _ = optimize_layout(x0, y0, None, None, max_iter)
            # Re-evaluate with full time series for accurate AEP
            aep_lib_absent = compute_aep_full(x_lib, y_lib, None, None)
            aep_lib_present = compute_aep_full(x_lib, y_lib, x_all_neighbors, y_all_neighbors)
            all_layouts.append({
                'x': np.array(x_lib), 'y': np.array(y_lib),
                'aep_absent': float(aep_lib_absent),
                'aep_present': float(aep_lib_present),
                'seed': seed,
                'strategy': 'liberal',
            })

            # Conservative
            x_con, y_con, _ = optimize_layout(x0, y0, x_all_neighbors, y_all_neighbors, max_iter)
            # Re-evaluate with full time series for accurate AEP
            aep_con_absent = compute_aep_full(x_con, y_con, None, None)
            aep_con_present = compute_aep_full(x_con, y_con, x_all_neighbors, y_all_neighbors)
            all_layouts.append({
                'x': np.array(x_con), 'y': np.array(y_con),
                'aep_absent': float(aep_con_absent),
                'aep_present': float(aep_con_present),
                'seed': seed,
                'strategy': 'conservative',
            })

            print(f"  Start {seed}: Liberal={aep_lib_absent:.1f}/{aep_lib_present:.1f}, "
                  f"Conservative={aep_con_absent:.1f}/{aep_con_present:.1f} GWh", flush=True)

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
            "regret_pct": float(all_regret / aep_present[con_opt_idx] * 100) if len(pareto_indices) > 0 else 0.0,
            "lib_opt_absent": float(aep_absent[lib_opt_idx]) if len(pareto_indices) > 0 else 0.0,
            "lib_opt_present": float(aep_present[lib_opt_idx]) if len(pareto_indices) > 0 else 0.0,
            "con_opt_absent": float(aep_absent[con_opt_idx]) if len(pareto_indices) > 0 else 0.0,
            "con_opt_present": float(aep_present[con_opt_idx]) if len(pareto_indices) > 0 else 0.0,
            "total_turbines": len(x_all_neighbors),
            "elapsed_seconds": elapsed,
        }

        # Save combined layouts for PyWake verification (append if file exists)
        layouts_file = output_path / "layouts_combined.h5"
        mode = 'a' if layouts_file.exists() and seed_offset > 0 else 'w'
        with h5py.File(layouts_file, mode) as hf:
            existing_count = len([k for k in hf.keys() if k.startswith('layout_')]) if mode == 'a' else 0
            for i, layout in enumerate(all_layouts):
                layout_idx = existing_count + i
                grp = hf.create_group(f"layout_{layout_idx}")
                grp.create_dataset('x', data=layout['x'])
                grp.create_dataset('y', data=layout['y'])
                grp.attrs['aep_absent'] = layout['aep_absent']
                grp.attrs['aep_present'] = layout['aep_present']
                grp.attrs['seed'] = layout['seed']
                grp.attrs['strategy'] = layout['strategy']
            hf.attrs['case'] = 'all_neighbors'
            hf.attrs['n_layouts'] = existing_count + len(all_layouts)
            hf.attrs['n_neighbor_turbines'] = len(x_all_neighbors)
        print(f"Saved {len(all_layouts)} layouts to {layouts_file} (total: {existing_count + len(all_layouts)})")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Farm':<30} {'Direction':>10} {'Regret (GWh)':>15}")
    print("-" * 70)

    for farm_idx in farm_indices:
        if farm_idx in results:
            r = results[farm_idx]
            print(f"{r['name']:<30} {r['direction']:>10.0f}° {r['regret_gwh']:>15.2f}")

    if not skip_combined:
        print("-" * 70)
        print(f"{'All 9 neighbors':<30} {'(ring)':<10} {results['all_neighbors']['regret_gwh']:>15.2f}")

    # Create visualization (only if we have results to plot)
    plotted_farms = [i for i in farm_indices if i in results]
    if plotted_farms:
        fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': 'polar'})

        directions_rad = [np.radians(results[i]['direction']) for i in plotted_farms]
        regrets = [results[i]['regret_gwh'] for i in plotted_farms]

        # Plot bars
        bars = ax.bar(directions_rad, regrets, width=0.3, alpha=0.7, color='steelblue', edgecolor='black')

        # Add labels
        max_regret = max(regrets) if regrets else 1.0
        for i, (d, r, idx) in enumerate(zip(directions_rad, regrets, plotted_farms)):
            ax.annotate(f'Farm {idx}\n{r:.1f} GWh',
                       xy=(d, r),
                       xytext=(d, r + max_regret*0.1),
                       ha='center', va='bottom', fontsize=9)

        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        if all_regret is not None:
            title = f'DEI Single Neighbor Regret by Direction ({wake_model.upper()})\n(All 9 together: {all_regret:.2f} GWh)'
        else:
            title = f'DEI Single Neighbor Regret by Direction ({wake_model.upper()})'
        ax.set_title(title, fontsize=14, pad=20)
        ax.set_ylabel('Regret (GWh)', labelpad=30)

        plt.tight_layout()
        if len(farm_indices) == N_NEIGHBOR_FARMS:
            plot_filename = f'dei_single_neighbor_{wake_model}.png'
        else:
            farm_suffix = '_'.join(str(f) for f in sorted(farm_indices))
            plot_filename = f'dei_single_neighbor_{wake_model}_farm{farm_suffix}.png'
        fig.savefig(output_path / plot_filename, dpi=150, bbox_inches='tight')
        print(f"\nSaved plot to {output_path / plot_filename}")
        plt.close(fig)

    # Save results (include farm indices in filename if running specific farms)
    if len(farm_indices) == N_NEIGHBOR_FARMS:
        results_filename = f'dei_single_neighbor_{wake_model}.json'
    else:
        farm_suffix = '_'.join(str(f) for f in sorted(farm_indices))
        results_filename = f'dei_single_neighbor_{wake_model}_farm{farm_suffix}.json'

    # Add configuration metadata
    results["_config"] = {
        "wake_model": wake_model,
        "A": A,
        "ti_amb": ti_amb,
        "n_starts": n_starts,
        "max_iter": max_iter,
    }

    with open(output_path / results_filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {output_path / results_filename}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test DEI with individual neighbors (optimized)")
    parser.add_argument("--wake-model", type=str, default="turbopark",
                        choices=["bastankhah", "turbopark"])
    parser.add_argument("--ti", type=float, default=0.06,
                        help="Ambient turbulence intensity (for turbopark)")
    parser.add_argument("--A", type=float, default=0.04,
                        help="Wake expansion coefficient A (for turbopark, default 0.04)")
    parser.add_argument("--n-starts", type=int, default=5,
                        help="Number of optimization starts per strategy")
    parser.add_argument("--seed-offset", type=int, default=0,
                        help="Starting seed number (for appending additional runs)")
    parser.add_argument("--max-iter", type=int, default=500,
                        help="Maximum optimization iterations per start")
    parser.add_argument("--output-dir", "-o", type=str, default="analysis/dei_single_neighbor")
    parser.add_argument("--farm", type=int, default=None,
                        help="Run only a specific farm index (1-9). If not set, runs all farms.")
    parser.add_argument("--skip-combined", action="store_true",
                        help="Skip the 'all neighbors combined' analysis")
    parser.add_argument("--only-combined", action="store_true",
                        help="Skip individual farms, only run 'all neighbors combined' analysis")

    args = parser.parse_args()

    if args.only_combined:
        farm_indices = []
        skip_combined = False
    else:
        farm_indices = [args.farm] if args.farm else None
        skip_combined = args.skip_combined

    run_single_neighbor_analysis(
        wake_model=args.wake_model,
        ti_amb=args.ti,
        A=args.A,
        n_starts=args.n_starts,
        max_iter=args.max_iter,
        output_dir=args.output_dir,
        farm_indices=farm_indices,
        skip_combined=skip_combined,
        seed_offset=args.seed_offset,
    )

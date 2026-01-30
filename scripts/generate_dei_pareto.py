"""Generate Pareto plot for DEI Farm 8 showing all multi-start results."""

import time
from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from pixwake import Curve, Turbine, WakeSimulation
from pixwake.deficit import BastankhahGaussianDeficit

# Paths
DEA_DIR = Path(__file__).parent.parent / "DEA_neighbors"
OUTPUT_DIR = Path(__file__).parent.parent / "docs/figures"
LAYOUTS_FILE = DEA_DIR / "re_precomputed_layouts.h5"
WIND_DATA_FILE = DEA_DIR / "energy_island_10y_daily_av_wind.csv"

# Configuration
TARGET_N_TURBINES = 66
TARGET_ROTOR_DIAMETER = 240
TARGET_HUB_HEIGHT = 150
N_STARTS = 5
MAX_ITER = 500


def load_wind_data():
    df = pd.read_csv(WIND_DATA_FILE, sep=';')
    return df['WD_150'].values, df['WS_150'].values


def compute_binned_wind_rose(wd_ts, ws_ts, n_bins=24):
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
        mean_speeds[i] = ws_ts[mask].mean() if mask.sum() > 0 else ws_ts.mean()
    weights = weights / weights.sum()
    return jnp.array(bin_centers), jnp.array(mean_speeds), jnp.array(weights)


def load_target_boundary():
    return np.array([
        706694.3923283464, 6224158.532895836,
        703972.0844905999, 6226906.597455995,
        702624.6334635273, 6253853.5386425415,
        712771.6248419734, 6257704.934445341,
        715639.3355871611, 6260664.6846508905,
        721593.2420745814, 6257906.998015941,
    ]).reshape((-1, 2)).T


def load_neighbor_layout(farm_idx):
    with h5py.File(LAYOUTS_FILE, 'r') as f:
        key = f"farm{farm_idx}_t5_s0"
        if key in f:
            layout = f[key]['layout'][:]
            return layout[0], layout[1]
    return None, None


def create_turbine():
    ws = jnp.array([0.0, 4.0, 10.0, 15.0, 25.0])
    power = jnp.array([0.0, 0.0, 15000.0, 15000.0, 0.0])
    ct = jnp.array([0.0, 0.8, 0.8, 0.4, 0.0])
    return Turbine(
        rotor_diameter=float(TARGET_ROTOR_DIAMETER),
        hub_height=float(TARGET_HUB_HEIGHT),
        power_curve=Curve(ws=ws, values=power),
        ct_curve=Curve(ws=ws, values=ct),
    )


def main():
    print("Loading data...")
    wd_ts, ws_ts = load_wind_data()
    wd, ws, weights = compute_binned_wind_rose(wd_ts, ws_ts, n_bins=24)

    target_boundary = load_target_boundary()
    x_neighbor, y_neighbor = load_neighbor_layout(8)  # Farm 8

    turbine = create_turbine()
    deficit = BastankhahGaussianDeficit(k=0.04)
    sim = WakeSimulation(turbine, deficit)

    x_min, x_max = float(target_boundary[0].min()), float(target_boundary[0].max())
    y_min, y_max = float(target_boundary[1].min()), float(target_boundary[1].max())
    D = TARGET_ROTOR_DIAMETER

    def generate_layout(seed):
        rng = np.random.default_rng(seed)
        x = rng.uniform(x_min, x_max, TARGET_N_TURBINES)
        y = rng.uniform(y_min, y_max, TARGET_N_TURBINES)
        return jnp.array(x), jnp.array(y)

    def compute_aep(x_target, y_target, x_neigh=None, y_neigh=None):
        if x_neigh is not None:
            x_all = jnp.concatenate([x_target, jnp.array(x_neigh)])
            y_all = jnp.concatenate([y_target, jnp.array(y_neigh)])
        else:
            x_all, y_all = x_target, y_target
        n_target = len(x_target)
        result = sim(x_all, y_all, ws_amb=ws, wd_amb=wd)
        power = result.power()
        target_power = power[:, :n_target].sum(axis=1)
        return float(jnp.sum(target_power * weights) * 8760 / 1e6)

    def make_optimizer(x_neigh=None, y_neigh=None):
        if x_neigh is not None:
            x_n, y_n = jnp.array(x_neigh), jnp.array(y_neigh)
            def loss_fn(x, y):
                x_all = jnp.concatenate([x, x_n])
                y_all = jnp.concatenate([y, y_n])
                result = sim(x_all, y_all, ws_amb=ws, wd_amb=wd)
                power = result.power()[:, :len(x)].sum(axis=1)
                return -jnp.sum(power * weights) * 8760 / 1e6
        else:
            def loss_fn(x, y):
                result = sim(x, y, ws_amb=ws, wd_amb=wd)
                power = result.power().sum(axis=1)
                return -jnp.sum(power * weights) * 8760 / 1e6

        grad_fn = jax.grad(loss_fn, argnums=(0, 1))

        @partial(jax.jit, static_argnums=(2,))
        def run_adam(x_init, y_init, max_iterations, lr=50.0):
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            def step(carry, t):
                x, y, m_x, m_y, v_x, v_y = carry
                t_f = t.astype(jnp.float64) + 1.0
                gx, gy = grad_fn(x, y)
                gx, gy = jnp.clip(gx, -100, 100), jnp.clip(gy, -100, 100)
                m_x = beta1 * m_x + (1 - beta1) * gx
                m_y = beta1 * m_y + (1 - beta1) * gy
                v_x = beta2 * v_x + (1 - beta2) * gx**2
                v_y = beta2 * v_y + (1 - beta2) * gy**2
                m_x_hat = m_x / (1 - beta1**t_f)
                m_y_hat = m_y / (1 - beta1**t_f)
                v_x_hat = v_x / (1 - beta2**t_f)
                v_y_hat = v_y / (1 - beta2**t_f)
                x = x - lr * m_x_hat / (jnp.sqrt(v_x_hat) + eps)
                y = y - lr * m_y_hat / (jnp.sqrt(v_y_hat) + eps)
                x = jnp.clip(x, x_min + D/2, x_max - D/2)
                y = jnp.clip(y, y_min + D/2, y_max - D/2)
                return (x, y, m_x, m_y, v_x, v_y), None

            init = (x_init, y_init, jnp.zeros_like(x_init), jnp.zeros_like(y_init),
                    jnp.zeros_like(x_init), jnp.zeros_like(y_init))
            (x_opt, y_opt, _, _, _, _), _ = jax.lax.scan(step, init, jnp.arange(max_iterations))
            return x_opt, y_opt, -loss_fn(x_opt, y_opt)

        return run_adam

    print("Compiling optimizers...")
    liberal_opt = make_optimizer(None, None)
    conservative_opt = make_optimizer(x_neighbor, y_neighbor)

    print(f"Running {N_STARTS} multi-start optimizations for Farm 8...")
    liberal_results = []
    conservative_results = []

    for seed in range(N_STARTS):
        x0, y0 = generate_layout(seed)

        # Liberal
        x_lib, y_lib, _ = liberal_opt(x0, y0, MAX_ITER)
        aep_absent = compute_aep(x_lib, y_lib, None, None)
        aep_present = compute_aep(x_lib, y_lib, x_neighbor, y_neighbor)
        liberal_results.append({'absent': aep_absent, 'present': aep_present, 'strategy': 'liberal'})

        # Conservative
        x_con, y_con, _ = conservative_opt(x0, y0, MAX_ITER)
        aep_absent = compute_aep(x_con, y_con, None, None)
        aep_present = compute_aep(x_con, y_con, x_neighbor, y_neighbor)
        conservative_results.append({'absent': aep_absent, 'present': aep_present, 'strategy': 'conservative'})

        print(f"  Start {seed+1}/{N_STARTS}: L={liberal_results[-1]['absent']:.1f}/{liberal_results[-1]['present']:.1f}, "
              f"C={conservative_results[-1]['absent']:.1f}/{conservative_results[-1]['present']:.1f}")

    # Combine and find Pareto front
    all_results = liberal_results + conservative_results
    absent = np.array([r['absent'] for r in all_results])
    present = np.array([r['present'] for r in all_results])
    strategies = [r['strategy'] for r in all_results]

    # Find Pareto front
    pareto_mask = np.zeros(len(all_results), dtype=bool)
    for i in range(len(all_results)):
        dominated = False
        for j in range(len(all_results)):
            if i != j:
                if (absent[j] >= absent[i] and present[j] > present[i]) or \
                   (absent[j] > absent[i] and present[j] >= present[i]):
                    dominated = True
                    break
        if not dominated:
            pareto_mask[i] = True

    # Find liberal-optimal and conservative-optimal on Pareto front
    pareto_idx = np.where(pareto_mask)[0]
    lib_opt_idx = pareto_idx[np.argmax(absent[pareto_mask])]
    con_opt_idx = pareto_idx[np.argmax(present[pareto_mask])]
    regret = present[con_opt_idx] - present[lib_opt_idx]

    print(f"\nPareto points: {pareto_mask.sum()}")
    print(f"Regret: {regret:.1f} GWh")

    # --- Create figure ---
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot all multi-start results
    lib_mask = np.array([s == 'liberal' for s in strategies])
    con_mask = np.array([s == 'conservative' for s in strategies])

    # Dominated points (smaller, transparent)
    ax.scatter(absent[lib_mask & ~pareto_mask], present[lib_mask & ~pareto_mask],
               s=100, c='#3498db', alpha=0.5, label='Liberal (dominated)', marker='o')
    ax.scatter(absent[con_mask & ~pareto_mask], present[con_mask & ~pareto_mask],
               s=100, c='#e74c3c', alpha=0.5, label='Conservative (dominated)', marker='s')

    # Pareto points (larger, solid)
    ax.scatter(absent[lib_mask & pareto_mask], present[lib_mask & pareto_mask],
               s=200, c='#3498db', alpha=1.0, label='Liberal (Pareto)', marker='o',
               edgecolors='black', linewidths=2, zorder=5)
    ax.scatter(absent[con_mask & pareto_mask], present[con_mask & pareto_mask],
               s=200, c='#e74c3c', alpha=1.0, label='Conservative (Pareto)', marker='s',
               edgecolors='black', linewidths=2, zorder=5)

    # Draw Pareto front line
    pareto_absent = absent[pareto_mask]
    pareto_present = present[pareto_mask]
    sort_idx = np.argsort(pareto_absent)
    ax.plot(pareto_absent[sort_idx], pareto_present[sort_idx], 'k--', linewidth=2, alpha=0.7, zorder=4)

    # Annotate regret with vertical arrow
    mid_x = absent[lib_opt_idx] - 2
    ax.annotate('', xy=(mid_x, present[con_opt_idx]),
                xytext=(mid_x, present[lib_opt_idx]),
                arrowprops=dict(arrowstyle='<->', color='#e74c3c', lw=2.5))
    ax.text(mid_x - 1.5, (present[lib_opt_idx] + present[con_opt_idx])/2,
            f'Regret\n{regret:.0f} GWh', fontsize=11, fontweight='bold',
            color='#e74c3c', va='center', ha='right')

    # Add axis padding
    x_range = absent.max() - absent.min()
    y_range = present.max() - present.min()
    ax.set_xlim(absent.min() - x_range * 0.15, absent.max() + x_range * 0.15)
    ax.set_ylim(present.min() - y_range * 0.15, present.max() + y_range * 0.15)

    ax.set_xlabel('AEP without Farm 8 (GWh/year)', fontsize=12)
    ax.set_ylabel('AEP with Farm 8 (GWh/year)', fontsize=12)
    ax.set_title('Pareto Front: Farm 8 (South, 163°)\nPooled Multi-Start Optimization', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / 'dei_pareto_farm8.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"\nSaved: {OUTPUT_DIR / 'dei_pareto_farm8.png'}")


if __name__ == "__main__":
    main()

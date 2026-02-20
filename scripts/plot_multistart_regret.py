"""Multistart regret convergence: envelope theorem IFT bilevel.

For each outer iteration with K inner starts:
  1. Run topfarm_sgd_solve K times with different random initial layouts (forward-only)
  2. Evaluate regret for each start, identify winner (highest regret)
  3. Run sgd_solve_implicit once with winning start's init layout -> IFT gradient
  4. ADAM gradient ascent on neighbor positions

The K forward screening solves are parallelized via jax.vmap in chunks,
giving significant speedup on multi-core CPUs while bounding RAM usage.

Cost: K x forward + 1 x (forward + backward) per outer step.

Produces:
  - analysis/multistart_regret/multistart_regret.png      (regret vs iteration)
  - analysis/multistart_regret/results_K{k}.json           (per-K results)

Usage:
    # Quick smoke test
    pixi run python scripts/plot_multistart_regret.py --n-outer=2 --k-values=1,3

    # Full run with large K (vmap parallelism)
    pixi run python scripts/plot_multistart_regret.py --k-values=1,10,50,100 --chunk-size=10
"""

import jax
jax.config.update("jax_enable_x64", True)

import argparse
import json
import time
from dataclasses import replace as _dc_replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pixwake import Curve, Turbine, WakeSimulation
from pixwake.deficit import BastankhahGaussianDeficit
from pixwake.optim.sgd import (
    SGDSettings, _compute_mid_bisection,
    sgd_solve_implicit, topfarm_sgd_solve,
)


# =============================================================================
# DEI setup helpers
# =============================================================================

TARGET_ROTOR_DIAMETER = 240.0
TARGET_HUB_HEIGHT = 150.0
D = TARGET_ROTOR_DIAMETER


def load_target_boundary():
    from scipy.spatial import ConvexHull
    raw = np.array([
        706694.3923283464, 6224158.532895836,
        703972.0844905999, 6226906.597455995,
        702624.6334635273, 6253853.5386425415,
        712771.6248419734, 6257704.934445341,
        715639.3355871611, 6260664.6846508905,
        721593.2420745814, 6257906.998015941,
    ]).reshape((-1, 2))
    hull = ConvexHull(raw)
    return raw[hull.vertices]


def create_dei_turbine():
    ws = jnp.array([0.,1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.,
                    13.,14.,15.,16.,17.,18.,19.,20.,21.,22.,23.,24.,25.])
    power = jnp.array([
        0., 0., 2.3986, 209.2581, 689.1977, 1480.6085,
        2661.2377, 4308.9290, 6501.0566, 9260.5163, 12081.4039, 13937.2966,
        14705.0160, 14931.0392, 14985.2085, 14996.9062, 14999.3433, 14999.8550,
        14999.9662, 14999.9916, 14999.9978, 14999.9994, 14999.9998, 14999.9999,
        15000.0000, 15000.0000,
    ])
    ct = jnp.array([
        0.8889, 0.8889, 0.8889, 0.8003, 0.8000, 0.8000,
        0.8000, 0.8000, 0.7999, 0.7930, 0.7354, 0.6100,
        0.4764, 0.3698, 0.2915, 0.2341, 0.1910, 0.1581,
        0.1325, 0.1122, 0.0958, 0.0826, 0.0717, 0.0626,
        0.0550, 0.0486,
    ])
    return Turbine(
        rotor_diameter=TARGET_ROTOR_DIAMETER, hub_height=TARGET_HUB_HEIGHT,
        power_curve=Curve(ws=ws, values=power), ct_curve=Curve(ws=ws, values=ct),
    )


def load_wind_data():
    import pandas as pd
    csv_path = Path(__file__).parent.parent / "energy_island_10y_daily_av_wind.csv"
    df = pd.read_csv(csv_path, sep=';')
    wd_ts, ws_ts = df['WD_150'].values, df['WS_150'].values
    n_bins = 24
    bin_edges = np.linspace(0, 360, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    weights_arr = np.zeros(n_bins)
    mean_speeds = np.zeros(n_bins)
    for i in range(n_bins):
        mask = ((wd_ts >= bin_edges[i]) | (wd_ts < bin_edges[0])) if i == n_bins - 1 \
            else ((wd_ts >= bin_edges[i]) & (wd_ts < bin_edges[i + 1]))
        weights_arr[i] = mask.sum()
        mean_speeds[i] = ws_ts[mask].mean() if mask.sum() > 0 else ws_ts.mean()
    weights_arr /= weights_arr.sum()
    return jnp.array(bin_centers), jnp.array(mean_speeds), jnp.array(weights_arr)


def generate_initial_layout(boundary, n_turbines, seed=42):
    from matplotlib.path import Path as MplPath
    poly_path = MplPath(boundary)
    x_min, x_max = boundary[:, 0].min(), boundary[:, 0].max()
    y_min, y_max = boundary[:, 1].min(), boundary[:, 1].max()
    rng = np.random.default_rng(seed)
    pts = []
    while len(pts) < n_turbines:
        cands = rng.uniform([x_min, y_min], [x_max, y_max], size=(n_turbines * 5, 2))
        pts.extend(cands[poly_path.contains_points(cands)].tolist())
    pts = np.array(pts[:n_turbines])
    return jnp.array(pts[:, 0]), jnp.array(pts[:, 1])


def place_neighbors_close(boundary, n_neighbors, seed=123):
    """Place neighbors 3D outside the target boundary (close for wake interaction)."""
    cy = boundary[:, 1].mean()
    x_offset = boundary[:, 0].min() - 3 * D
    rng = np.random.default_rng(seed)
    y_spread = (boundary[:, 1].max() - boundary[:, 1].min()) * 0.5
    nb_x = np.full(n_neighbors, x_offset) + rng.uniform(-D, 0, n_neighbors)
    nb_y = cy + rng.uniform(-y_spread, y_spread, n_neighbors)
    return jnp.array(nb_x), jnp.array(nb_y)


# =============================================================================
# ADAM + buffer helpers
# =============================================================================

def adam_step(params, grad, m, v, t, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    """Standard ADAM update (gradient ascent: add lr * update)."""
    m_new = beta1 * m + (1 - beta1) * grad
    v_new = beta2 * v + (1 - beta2) * grad ** 2
    m_hat = m_new / (1 - beta1 ** t)
    v_hat = v_new / (1 - beta2 ** t)
    params_new = params + lr * m_hat / (jnp.sqrt(v_hat) + eps)
    return params_new, m_new, v_new


def enforce_target_buffer(nb_params, boundary, buffer):
    """Project neighbors outside target boundary bounding box + buffer."""
    n_nb = nb_params.shape[0] // 2
    nb_x = nb_params[:n_nb]
    nb_y = nb_params[n_nb:]
    cx = jnp.clip(nb_x, boundary[:, 0].min(), boundary[:, 0].max())
    cy = jnp.clip(nb_y, boundary[:, 1].min(), boundary[:, 1].max())
    dx = nb_x - cx
    dy = nb_y - cy
    dist = jnp.sqrt(dx**2 + dy**2)
    scale = jnp.where(dist < 1e-6, 1.0, buffer / dist)
    nb_x = jnp.where(dist < buffer, cx + dx * scale, nb_x)
    nb_y = jnp.where(dist < buffer, cy + dy * scale, nb_y)
    return jnp.concatenate([nb_x, nb_y])


# =============================================================================
# Core: envelope theorem multistart with vmap parallelism
# =============================================================================

def run_envelope_multistart(
    K,
    n_outer,
    lr,
    init_layouts,
    neighbor_params_init,
    liberal_aep,
    liberal_x,
    liberal_y,
    objective_with_neighbors,
    sim,
    ws,
    wd,
    weights,
    boundary,
    min_spacing,
    sgd_settings,
    n_target,
    buffer,
    chunk_size=10,
    verbose=True,
):
    """Run envelope-theorem multistart IFT bilevel optimization.

    The K forward screening solves are parallelized via jax.vmap in chunks
    of `chunk_size`. Each chunk runs as a single vectorized XLA program,
    giving ~Nx speedup on N-core CPUs while bounding RAM to chunk_size
    simultaneous solves.
    """
    nb_params = neighbor_params_init.copy()
    m = jnp.zeros_like(nb_params)
    v = jnp.zeros_like(nb_params)

    # Pre-compute mid to avoid Python bisection during JAX tracing
    if sgd_settings.mid is None:
        computed_mid = _compute_mid_bisection(
            sgd_settings.learning_rate, sgd_settings.gamma_min_factor,
            sgd_settings.max_iter, sgd_settings.bisect_lower,
            sgd_settings.bisect_upper,
        )
        sgd_settings = _dc_replace(sgd_settings, mid=computed_mid)

    # Stack initial layouts: (K, n_target)
    all_init_x = jnp.stack([lay[0] for lay in init_layouts])
    all_init_y = jnp.stack([lay[1] for lay in init_layouts])

    # --- Define single-start screening function ---
    def screen_one(init_x, init_y, nb_p):
        """Forward-only solve + regret evaluation for one start."""
        def obj_fn(x, y):
            return objective_with_neighbors(x, y, nb_p)
        opt_x, opt_y = topfarm_sgd_solve(
            obj_fn, init_x, init_y, boundary, min_spacing, sgd_settings,
        )
        n_nb = nb_p.shape[0] // 2
        nb_x, nb_y = nb_p[:n_nb], nb_p[n_nb:]
        x_all = jnp.concatenate([opt_x, nb_x])
        y_all = jnp.concatenate([opt_y, nb_y])
        result = sim(x_all, y_all, ws_amb=ws, wd_amb=wd)
        power = result.power()[:, :n_target]
        conservative_aep = jnp.sum(power * weights[:, None]) * 8760 / 1e6
        # Liberal layout evaluated WITH neighbors
        x_lib_all = jnp.concatenate([liberal_x, nb_x])
        y_lib_all = jnp.concatenate([liberal_y, nb_y])
        result_lib = sim(x_lib_all, y_lib_all, ws_amb=ws, wd_amb=wd)
        power_lib = result_lib.power()[:, :n_target]
        liberal_aep_present = jnp.sum(power_lib * weights[:, None]) * 8760 / 1e6
        return conservative_aep - liberal_aep_present

    # --- Setup parallel screening via vmap ---
    use_vmap = K > 1
    batched_screen = None

    if use_vmap:
        if verbose:
            print(f"  Compiling vmapped screening (chunk_size={chunk_size})...",
                  end=" ", flush=True)
        try:
            batched_screen = jax.vmap(screen_one, in_axes=(0, 0, None))
            # Warmup: compile with one chunk
            warmup_size = min(chunk_size, K)
            t0 = time.time()
            test_r = batched_screen(
                all_init_x[:warmup_size], all_init_y[:warmup_size], nb_params,
            )
            warmup_time = time.time() - t0
            if verbose:
                top3 = ", ".join(f"{float(r):.1f}" for r in test_r[:3])
                print(f"OK ({warmup_time:.1f}s, regrets: {top3}...)")
        except Exception as e:
            if verbose:
                print(f"FAILED: {e}")
                print(f"  Falling back to sequential screening")
            use_vmap = False

    def screen_all(nb_p):
        """Screen all K starts, return regret array."""
        if use_vmap:
            results = []
            for c_start in range(0, K, chunk_size):
                c_end = min(c_start + chunk_size, K)
                actual = c_end - c_start
                cx = all_init_x[c_start:c_end]
                cy = all_init_y[c_start:c_end]
                # Pad to chunk_size for consistent XLA compilation shapes
                if actual < chunk_size:
                    pad_n = chunk_size - actual
                    cx = jnp.concatenate([cx, all_init_x[:pad_n]])
                    cy = jnp.concatenate([cy, all_init_y[:pad_n]])
                chunk_r = batched_screen(cx, cy, nb_p)
                results.append(chunk_r[:actual])
            return jnp.concatenate(results)
        else:
            return jnp.array([
                screen_one(all_init_x[k], all_init_y[k], nb_p)
                for k in range(K)
            ])

    # --- History tracking ---
    regret_history = []
    best_regret_history = []
    grad_norm_history = []
    winner_history = []
    time_history = []
    best_regret_overall = -np.inf
    best_nb_params = nb_params.copy()
    total_t0 = time.time()

    for outer_i in range(n_outer):
        iter_t0 = time.time()

        # --- Screen all K starts ---
        screen_t0 = time.time()
        regrets_k = screen_all(nb_params)
        screen_time = time.time() - screen_t0
        best_k = int(jnp.argmax(regrets_k))

        # --- IFT backward through winner ---
        winning_start_x, winning_start_y = init_layouts[best_k]

        def regret_fn(p):
            opt_x, opt_y = sgd_solve_implicit(
                objective_with_neighbors, winning_start_x, winning_start_y,
                boundary, min_spacing, sgd_settings, p,
            )
            n_nb = p.shape[0] // 2
            nb_x, nb_y = p[:n_nb], p[n_nb:]
            x_all = jnp.concatenate([opt_x, nb_x])
            y_all = jnp.concatenate([opt_y, nb_y])
            result = sim(x_all, y_all, ws_amb=ws, wd_amb=wd)
            power = result.power()[:, :n_target]
            conservative_aep = jnp.sum(power * weights[:, None]) * 8760 / 1e6
            # Liberal layout (optimized in isolation) evaluated WITH neighbors
            x_lib_all = jnp.concatenate([liberal_x, nb_x])
            y_lib_all = jnp.concatenate([liberal_y, nb_y])
            result_lib = sim(x_lib_all, y_lib_all, ws_amb=ws, wd_amb=wd)
            power_lib = result_lib.power()[:, :n_target]
            liberal_aep_present = jnp.sum(power_lib * weights[:, None]) * 8760 / 1e6
            return conservative_aep - liberal_aep_present

        regret_val, grad = jax.value_and_grad(regret_fn)(nb_params)
        regret_val = float(regret_val)
        grad_norm = float(jnp.linalg.norm(grad))

        if not jnp.all(jnp.isfinite(grad)):
            if verbose:
                print(f"  K={K} iter {outer_i}: NaN gradient, stopping early")
            break

        # --- ADAM ascent ---
        t = outer_i + 1
        nb_params, m, v = adam_step(nb_params, grad, m, v, t, lr)
        nb_params = enforce_target_buffer(nb_params, boundary, buffer)

        if regret_val > best_regret_overall:
            best_regret_overall = regret_val
            best_nb_params = nb_params.copy()

        iter_time = time.time() - iter_t0
        regret_history.append(regret_val)
        best_regret_history.append(best_regret_overall)
        grad_norm_history.append(grad_norm)
        winner_history.append(best_k)
        time_history.append(iter_time)

        if verbose and (outer_i % 5 == 0 or outer_i == n_outer - 1):
            print(f"  K={K} iter {outer_i:3d}: regret={regret_val:.4f} GWh, "
                  f"best_so_far={best_regret_overall:.4f}, "
                  f"|grad|={grad_norm:.4e}, winner=k{best_k}, "
                  f"screen={screen_time:.1f}s, total={iter_time:.1f}s")

    total_time = time.time() - total_t0

    return {
        "K": K,
        "regret_history": regret_history,
        "best_regret_history": best_regret_history,
        "grad_norm_history": grad_norm_history,
        "winner_history": winner_history,
        "time_history": time_history,
        "final_neighbor_params": np.array(best_nb_params).tolist(),
        "total_time": total_time,
        "liberal_aep": liberal_aep,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multistart regret convergence (envelope theorem IFT)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-outer", type=int, default=20)
    parser.add_argument("--lr", type=float, default=10.0)
    parser.add_argument("--n-target", type=int, default=50)
    parser.add_argument("--n-neighbors", type=int, default=50)
    parser.add_argument("--k-values", type=str, default="1,10,50,100",
                        help="Comma-separated K values")
    parser.add_argument("--chunk-size", type=int, default=10,
                        help="Vmap chunk size for parallel screening (limits RAM)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inner-lr", type=float, default=50.0)
    parser.add_argument("--inner-max-iter", type=int, default=500)
    parser.add_argument("--inner-tol", type=float, default=1e-10)
    parser.add_argument("--output-dir", type=str, default="analysis/multistart_regret")
    args = parser.parse_args()

    k_values = [int(k) for k in args.k_values.split(",")]
    max_K = max(k_values)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    N_TARGET = args.n_target
    N_NEIGHBOR = args.n_neighbors
    min_spacing = 4.0 * D

    print("=" * 70)
    print("MULTISTART REGRET CONVERGENCE (Envelope Theorem)")
    print(f"  K values: {k_values}")
    print(f"  {args.n_outer} outer iterations, lr={args.lr}")
    print(f"  {N_TARGET} targets, {N_NEIGHBOR} neighbors")
    print(f"  Inner: lr={args.inner_lr}, max_iter={args.inner_max_iter}, tol={args.inner_tol}")
    print(f"  Vmap chunk size: {args.chunk_size}")
    print("=" * 70)

    # --- Setup ---
    boundary_np = load_target_boundary()
    boundary = jnp.array(boundary_np)
    turbine = create_dei_turbine()
    wd, ws, weights = load_wind_data()
    sim = WakeSimulation(turbine, BastankhahGaussianDeficit(k=0.04))

    sgd_settings = SGDSettings(
        learning_rate=args.inner_lr,
        max_iter=args.inner_max_iter,
        tol=args.inner_tol,
    )

    # --- Pre-generate pool of initial layouts ---
    print(f"\nGenerating {max_K} initial layouts (seeds {args.seed}..{args.seed + max_K - 1})...")
    t0 = time.time()
    init_layouts = []
    for k in range(max_K):
        ix, iy = generate_initial_layout(boundary_np, N_TARGET, seed=args.seed + k)
        init_layouts.append((ix, iy))
    print(f"  Done ({time.time()-t0:.1f}s)")

    # --- Initial neighbors ---
    init_nb_x, init_nb_y = place_neighbors_close(boundary_np, N_NEIGHBOR, seed=args.seed + 100)
    neighbor_params_init = jnp.concatenate([init_nb_x, init_nb_y])
    print(f"Initial neighbors placed 3D west of boundary")
    print(f"  nb_x range: [{float(init_nb_x.min()):.0f}, {float(init_nb_x.max()):.0f}]")
    print(f"  nb_y range: [{float(init_nb_y.min()):.0f}, {float(init_nb_y.max()):.0f}]")

    # --- Liberal layout (shared across all K) ---
    def liberal_objective(x, y):
        result = sim(x, y, ws_amb=ws, wd_amb=wd)
        power = result.power()[:, :N_TARGET]
        return -jnp.sum(power * weights[:, None]) * 8760 / 1e6

    print("\nComputing liberal layout (no neighbors)...")
    t0 = time.time()
    liberal_x, liberal_y = topfarm_sgd_solve(
        liberal_objective, init_layouts[0][0], init_layouts[0][1],
        boundary, min_spacing, sgd_settings,
    )
    liberal_aep = float(-liberal_objective(liberal_x, liberal_y))
    print(f"Liberal AEP: {liberal_aep:.2f} GWh ({time.time()-t0:.1f}s)")

    # --- Objective with neighbors ---
    def objective_with_neighbors(x, y, neighbor_params):
        n_nb = neighbor_params.shape[0] // 2
        nb_x, nb_y = neighbor_params[:n_nb], neighbor_params[n_nb:]
        x_all = jnp.concatenate([x, nb_x])
        y_all = jnp.concatenate([y, nb_y])
        result = sim(x_all, y_all, ws_amb=ws, wd_amb=wd)
        power = result.power()[:, :N_TARGET]
        return -jnp.sum(power * weights[:, None]) * 8760 / 1e6

    buffer = 2 * D

    # --- Run for each K ---
    all_results = {}
    for K in k_values:
        print(f"\n{'='*60}")
        print(f"Running K={K} ({args.n_outer} outer iterations)")
        print(f"{'='*60}")

        result = run_envelope_multistart(
            K=K,
            n_outer=args.n_outer,
            lr=args.lr,
            init_layouts=init_layouts[:K],
            neighbor_params_init=neighbor_params_init,
            liberal_aep=liberal_aep,
            liberal_x=liberal_x,
            liberal_y=liberal_y,
            objective_with_neighbors=objective_with_neighbors,
            sim=sim,
            ws=ws,
            wd=wd,
            weights=weights,
            boundary=boundary,
            min_spacing=min_spacing,
            sgd_settings=sgd_settings,
            n_target=N_TARGET,
            buffer=buffer,
            chunk_size=args.chunk_size,
            verbose=True,
        )

        all_results[K] = result

        # Save per-K JSON
        json_path = output_dir / f"results_K{K}.json"
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved {json_path}")

    # --- Summary ---
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'K':>6}  {'Best Regret':>12}  {'Final Regret':>14}  {'Total Time':>12}")
    print("-" * 54)
    for K in k_values:
        r = all_results[K]
        best = max(r["regret_history"]) if r["regret_history"] else 0
        final = r["regret_history"][-1] if r["regret_history"] else 0
        print(f"{K:6d}  {best:12.4f}  {final:14.4f}  {r['total_time']:10.1f}s")

    # =========================================================================
    # Plot: Regret vs iteration for each K
    # =========================================================================
    cmap = plt.cm.viridis
    n_lines = len(k_values)

    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, K in enumerate(k_values):
        r = all_results[K]
        iters = range(len(r["regret_history"]))
        c = cmap(idx / max(n_lines - 1, 1))
        ax.plot(iters, r["regret_history"], "-o", color=c, lw=2, ms=3,
                label=f"K={K}")

    ax.set_xlabel("Outer Iteration", fontsize=12)
    ax.set_ylabel("Regret (GWh)", fontsize=12)
    ax.set_title(f"Regret vs Iteration — {N_TARGET} targets, {N_NEIGHBOR} neighbors",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "multistart_regret.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {output_dir / 'multistart_regret.png'}")

    print(f"\nAll outputs in {output_dir}/")


if __name__ == "__main__":
    import sys
    # Force unbuffered stdout for background execution
    sys.stdout.reconfigure(line_buffering=True)
    main()

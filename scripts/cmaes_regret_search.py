"""CMA-ES search for maximum-regret neighbor configurations.

Two phases:
  Phase 1: 2D CMA-ES over (bearing, distance) with fixed 3×3 cluster.
  Phase 2: Full-position CMA-ES over individual neighbor turbine positions.

Optionally overlays CMA-ES trajectory on polar heatmap from regret_landscape.py,
and computes IFT gradient magnitude at the optimum for comparison.

Setup matches regret_landscape.py / run_bilevel_ift.py:
- 16 target turbines, 16D×16D boundary, D=200m, 4D min spacing
- BastankhahGaussianDeficit(k=0.04), single wind 270°/9 m/s

Outputs → analysis/cmaes_regret/
- convergence.png — regret + sigma vs generation (both phases)
- layout.png — boundary + liberal + conservative + neighbors
- overlay.png — CMA-ES trajectory on polar heatmap (if landscape results exist)
- gradient_comparison.png — IFT gradient magnitude analysis
- results.json

Usage:
    pixi run python scripts/cmaes_regret_search.py
"""

import jax

jax.config.update("jax_enable_x64", True)

import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cma

from pixwake import Curve, Turbine, WakeSimulation
from pixwake.deficit import BastankhahGaussianDeficit
from pixwake.optim.sgd import (
    SGDSettings,
    generate_random_starts,
    sgd_solve_implicit,
    topfarm_sgd_solve,
    topfarm_sgd_solve_multistart,
)

# ── Configuration ─────────────────────────────────────────────────────────

D = 200.0
TARGET_SIZE = 16 * D
MIN_SPACING = 4 * D
N_TARGET = 16
WS = jnp.array([9.0])
WD = jnp.array([270.0])

# Default neighbor cluster
DEFAULT_N_ROWS = 3
DEFAULT_N_COLS = 3
DEFAULT_SPACING_D = 5.0
N_NEIGHBORS = DEFAULT_N_ROWS * DEFAULT_N_COLS

# Multistart for inner solve
K_STARTS = 3
SEED = 42

# Phase 1: 2D CMA-ES (Cartesian center)
P1_POPSIZE = 14
P1_SIGMA0 = 5.0 * D  # ~1000m initial step size
P1_MAXITER = 50

# Phase 2: Full-position CMA-ES
P2_POPSIZE = 20
P2_SIGMA0 = 2.0 * D  # ~400m initial step size
P2_MAXITER = 100

# Buffer: neighbors must be outside target boundary (no overlap with lease area)
BUFFER_D = 0.0
BUFFER = BUFFER_D * D

OUTPUT_DIR = Path("analysis/cmaes_regret")
LANDSCAPE_RESULTS = Path("analysis/regret_landscape/results.json")


# ── Turbine and simulation ────────────────────────────────────────────────


def create_turbine(rotor_diameter: float = 200.0) -> Turbine:
    ws = jnp.array([0.0, 4.0, 10.0, 15.0, 25.0])
    power = jnp.array([0.0, 0.0, 10000.0, 10000.0, 0.0])
    ct = jnp.array([0.0, 0.8, 0.8, 0.4, 0.0])
    return Turbine(
        rotor_diameter=rotor_diameter,
        hub_height=120.0,
        power_curve=Curve(ws=ws, values=power),
        ct_curve=Curve(ws=ws, values=ct),
    )


# ── Cluster placement (same as regret_landscape.py) ──────────────────────


def make_rectangular_cluster(
    bearing_deg: float,
    distance_D: float,
    n_rows: int,
    n_cols: int,
    spacing_D: float,
    D_: float,
    center: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    angle_rad = np.radians(90.0 - bearing_deg)
    dist_m = distance_D * D_
    cx = center[0] + dist_m * np.cos(angle_rad)
    cy = center[1] + dist_m * np.sin(angle_rad)
    radial = np.array([np.cos(angle_rad), np.sin(angle_rad)])
    perp = np.array([-radial[1], radial[0]])
    spacing_m = spacing_D * D_
    positions = []
    for r in range(n_rows):
        for c in range(n_cols):
            r_offset = (r - (n_rows - 1) / 2.0) * spacing_m
            c_offset = (c - (n_cols - 1) / 2.0) * spacing_m
            pos = np.array([cx, cy]) + r_offset * radial + c_offset * perp
            positions.append(pos)
    positions = np.array(positions)
    return positions[:, 0], positions[:, 1]


# ── Boundary projection ──────────────────────────────────────────────────


def _is_outside_boundary(nb_x, nb_y, bnd, buffer_dist):
    """Check if ALL neighbors are outside boundary + buffer (rectangular)."""
    x_min = float(bnd[:, 0].min()) - buffer_dist
    x_max = float(bnd[:, 0].max()) + buffer_dist
    y_min = float(bnd[:, 1].min()) - buffer_dist
    y_max = float(bnd[:, 1].max()) + buffer_dist
    inside = (nb_x > x_min) & (nb_x < x_max) & (nb_y > y_min) & (nb_y < y_max)
    return not inside.any()


# ── AEP / regret helpers ─────────────────────────────────────────────────


def setup_farm():
    """Create turbine, sim, boundary, initial layout, liberal layout."""
    turbine = create_turbine(D)
    deficit = BastankhahGaussianDeficit(k=0.04)
    sim = WakeSimulation(turbine, deficit)

    target_boundary = jnp.array([
        [0.0, 0.0],
        [TARGET_SIZE, 0.0],
        [TARGET_SIZE, TARGET_SIZE],
        [0.0, TARGET_SIZE],
    ])
    target_center = np.array([TARGET_SIZE / 2, TARGET_SIZE / 2])

    # 4×4 grid initial layout
    grid_side = int(np.sqrt(N_TARGET))
    grid_spacing = TARGET_SIZE / (grid_side + 1)
    init_x = jnp.array([grid_spacing * (i + 1) for i in range(grid_side)] * grid_side)
    init_y = jnp.array(
        [grid_spacing * (j + 1) for j in range(grid_side) for _ in range(grid_side)]
    )

    sgd_settings = SGDSettings(
        learning_rate=D / 5,
        max_iter=3000,
        tol=1e-8,
    )

    return sim, target_boundary, target_center, init_x, init_y, sgd_settings


def compute_aep(sim, target_x, target_y, nb_x=None, nb_y=None):
    n_target = len(target_x)
    if nb_x is not None:
        x_all = jnp.concatenate([target_x, nb_x])
        y_all = jnp.concatenate([target_y, nb_y])
    else:
        x_all = target_x
        y_all = target_y
    result = sim(x_all, y_all, ws_amb=WS, wd_amb=WD)
    power = result.power()[:, :n_target]
    return float(jnp.sum(power) * 8760.0 / 1e6)


def compute_regret_full(
    nb_x, nb_y, liberal_x, liberal_y,
    sim, init_x, init_y, boundary, min_spacing, sgd_settings,
    k_starts=3, rng_key=None,
):
    """Full regret with multistart inner solve. Returns dict."""
    n_nb = len(nb_x)
    nb_x_j = jnp.array(nb_x) if not isinstance(nb_x, jnp.ndarray) else nb_x
    nb_y_j = jnp.array(nb_y) if not isinstance(nb_y, jnp.ndarray) else nb_y

    def obj_fn(x, y):
        x_all = jnp.concatenate([x, nb_x_j])
        y_all = jnp.concatenate([y, nb_y_j])
        result = sim(x_all, y_all, ws_amb=WS, wd_amb=WD)
        power = result.power()[:, :N_TARGET]
        return -jnp.sum(power) * 8760.0 / 1e6

    # Always include the liberal layout as a candidate start so that
    # conservative_aep >= liberal_aep_present (regret >= 0 by construction)
    if k_starts > 1 and rng_key is not None:
        rand_x, rand_y = generate_random_starts(
            rng_key, k_starts - 1, N_TARGET, boundary, min_spacing
        )
        init_x_batch = jnp.concatenate([liberal_x[None, :], init_x[None, :], rand_x], axis=0)
        init_y_batch = jnp.concatenate([liberal_y[None, :], init_y[None, :], rand_y], axis=0)
        all_x, all_y, all_objs = topfarm_sgd_solve_multistart(
            obj_fn, init_x_batch, init_y_batch, boundary, min_spacing, sgd_settings
        )
        best_idx = jnp.argmin(all_objs)
        opt_x = all_x[best_idx]
        opt_y = all_y[best_idx]
    else:
        opt_x, opt_y = topfarm_sgd_solve(
            obj_fn, liberal_x, liberal_y, boundary, min_spacing, sgd_settings
        )

    conservative_aep = compute_aep(sim, opt_x, opt_y, nb_x_j, nb_y_j)
    liberal_aep_present = compute_aep(sim, liberal_x, liberal_y, nb_x_j, nb_y_j)
    regret = conservative_aep - liberal_aep_present

    return {
        "regret": regret,
        "conservative_aep": conservative_aep,
        "liberal_aep_present": liberal_aep_present,
        "opt_x": opt_x,
        "opt_y": opt_y,
    }


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sim, target_boundary, target_center, init_x, init_y, sgd_settings = setup_farm()
    boundary_np = np.array(target_boundary)

    # Liberal layout
    print("Computing liberal layout (no neighbors)...", flush=True)
    t0 = time.time()

    def liberal_obj(x, y):
        result = sim(x, y, ws_amb=WS, wd_amb=WD)
        power = result.power()[:, :N_TARGET]
        return -jnp.sum(power) * 8760.0 / 1e6

    liberal_x, liberal_y = topfarm_sgd_solve(
        liberal_obj, init_x, init_y, target_boundary, MIN_SPACING, sgd_settings
    )
    liberal_aep = compute_aep(sim, liberal_x, liberal_y)
    print(f"  Liberal AEP: {liberal_aep:.2f} GWh ({time.time() - t0:.1f}s)", flush=True)

    rng_key = jax.random.PRNGKey(SEED)

    # ══════════════════════════════════════════════════════════════════════
    # Phase 1: 2D CMA-ES over (cx, cy) — cluster center in Cartesian
    # ══════════════════════════════════════════════════════════════════════

    print(f"\n{'='*60}")
    print("Phase 1: 2D CMA-ES (cluster center position)")
    print(f"{'='*60}")
    print(f"  Cluster: {DEFAULT_N_ROWS}x{DEFAULT_N_COLS}, {DEFAULT_SPACING_D:.0f}D spacing")
    print(f"  popsize={P1_POPSIZE}, sigma0={P1_SIGMA0:.0f}m, maxiter={P1_MAXITER}")
    print(f"  Inner starts: K={K_STARTS}", flush=True)

    # Neighbor bounding box
    pad = 25.0 * D
    nb_lo = np.array([boundary_np[:, 0].min() - pad, boundary_np[:, 1].min() - pad])
    nb_hi = np.array([boundary_np[:, 0].max() + pad, boundary_np[:, 1].max() + pad])

    # Initial: cluster upwind at 14D (where landscape shows peak valid regret)
    init_bearing = 270.0
    init_dist = 14.0
    init_angle = np.radians(90.0 - init_bearing)
    x0_p1 = np.array([
        target_center[0] + init_dist * D * np.cos(init_angle),
        target_center[1] + init_dist * D * np.sin(init_angle),
    ])

    p1_history_regret = []
    p1_history_sigma = []
    p1_history_bearing = []
    p1_history_dist = []
    best_p1_regret = -np.inf
    best_p1_params = None
    best_p1_bearing = None
    best_p1_dist = None

    es1 = cma.CMAEvolutionStrategy(x0_p1, P1_SIGMA0, {
        "bounds": [nb_lo.tolist(), nb_hi.tolist()],
        "maxiter": P1_MAXITER,
        "popsize": P1_POPSIZE,
        "seed": SEED,
        "verbose": -1,
        "tolfun": 0,
        "tolfunhist": 0,
        "tolflatfitness": 0,
        "tolstagnation": 200,
    })

    print(f"\n{'Gen':>4}  {'Gen Best':>10}  {'Best Ever':>10}  {'Sigma':>10}  {'Bearing':>8}  {'Dist(D)':>8}  {'Time':>6}")
    print("-" * 70, flush=True)

    t_p1_start = time.time()
    gen = 0

    while not es1.stop():
        t_gen = time.time()
        solutions = es1.ask()
        fitnesses = []
        gen_bearings = []
        gen_dists = []

        for s in solutions:
            cx, cy = s
            dx = cx - target_center[0]
            dy = cy - target_center[1]
            dist_m = np.sqrt(dx**2 + dy**2)
            dist_D_ = dist_m / D
            bearing_ = (90.0 - np.degrees(np.arctan2(dy, dx))) % 360

            nb_x_np, nb_y_np = make_rectangular_cluster(
                bearing_, dist_D_,
                DEFAULT_N_ROWS, DEFAULT_N_COLS, DEFAULT_SPACING_D,
                D, target_center,
            )

            # Skip if any neighbor is inside boundary + buffer
            if not _is_outside_boundary(nb_x_np, nb_y_np, boundary_np, BUFFER):
                fitnesses.append(0.0)  # CMA-ES minimizes; 0 = zero regret
                gen_bearings.append(bearing_)
                gen_dists.append(dist_D_)
                continue

            rng_key, subkey = jax.random.split(rng_key)
            result = compute_regret_full(
                jnp.array(nb_x_np), jnp.array(nb_y_np),
                liberal_x, liberal_y,
                sim, init_x, init_y, target_boundary, MIN_SPACING,
                sgd_settings, k_starts=K_STARTS, rng_key=subkey,
            )
            fitnesses.append(-result["regret"])  # CMA-ES minimizes
            gen_bearings.append(bearing_)
            gen_dists.append(dist_D_)

        es1.tell(solutions, fitnesses)

        best_idx = int(np.argmin(fitnesses))
        gen_best_regret = -fitnesses[best_idx]
        if gen_best_regret > best_p1_regret:
            best_p1_regret = gen_best_regret
            best_p1_params = solutions[best_idx].copy()
            best_p1_bearing = gen_bearings[best_idx]
            best_p1_dist = gen_dists[best_idx]

        p1_history_regret.append(best_p1_regret)
        p1_history_sigma.append(es1.sigma)
        p1_history_bearing.append(best_p1_bearing)
        p1_history_dist.append(best_p1_dist)
        gen += 1

        elapsed_gen = time.time() - t_gen
        print(
            f"{gen:4d}  {gen_best_regret:10.4f}  {best_p1_regret:10.4f}  "
            f"{es1.sigma:10.1f}  {best_p1_bearing:7.1f}°  {best_p1_dist:7.1f}  "
            f"{elapsed_gen:5.1f}s",
            flush=True,
        )

    p1_elapsed = time.time() - t_p1_start
    print(f"\nPhase 1 done: {gen} generations in {p1_elapsed:.0f}s ({p1_elapsed/60:.1f} min)")
    print(f"  Best regret: {best_p1_regret:.4f} GWh")
    print(f"  Best bearing: {best_p1_bearing:.1f}°, distance: {best_p1_dist:.1f}D", flush=True)

    # Build Phase 1 optimal cluster
    p1_nb_x, p1_nb_y = make_rectangular_cluster(
        best_p1_bearing, best_p1_dist,
        DEFAULT_N_ROWS, DEFAULT_N_COLS, DEFAULT_SPACING_D,
        D, target_center,
    )
    # p1 cluster is already outside boundary (only valid configs evaluated)

    # ══════════════════════════════════════════════════════════════════════
    # Phase 2: Full-position CMA-ES over individual neighbor positions
    # ══════════════════════════════════════════════════════════════════════

    print(f"\n{'='*60}")
    print("Phase 2: Full-position CMA-ES (individual neighbor positions)")
    print(f"{'='*60}")
    print(f"  {N_NEIGHBORS} neighbors = {2*N_NEIGHBORS}D search")
    print(f"  popsize={P2_POPSIZE}, sigma0={P2_SIGMA0:.0f}m, maxiter={P2_MAXITER}")
    print(f"  Initialized from Phase 1 optimum", flush=True)

    x0_p2 = np.concatenate([p1_nb_x, p1_nb_y])
    lower_p2 = np.full(2 * N_NEIGHBORS, float(nb_lo[0]))
    lower_p2[N_NEIGHBORS:] = float(nb_lo[1])
    upper_p2 = np.full(2 * N_NEIGHBORS, float(nb_hi[0]))
    upper_p2[N_NEIGHBORS:] = float(nb_hi[1])

    p2_history_regret = []
    p2_history_sigma = []
    best_p2_regret = -np.inf
    best_p2_params = None
    best_p2_result = None

    es2 = cma.CMAEvolutionStrategy(x0_p2, P2_SIGMA0, {
        "bounds": [lower_p2.tolist(), upper_p2.tolist()],
        "maxiter": P2_MAXITER,
        "popsize": P2_POPSIZE,
        "seed": SEED + 1,
        "verbose": -1,
        "tolfun": 0,
        "tolfunhist": 0,
        "tolflatfitness": 0,
        "tolstagnation": 200,
    })

    print(f"\n{'Gen':>4}  {'Gen Best':>10}  {'Best Ever':>10}  {'Sigma':>10}  {'Time':>6}")
    print("-" * 50, flush=True)

    t_p2_start = time.time()
    gen = 0

    while not es2.stop():
        t_gen = time.time()
        solutions = es2.ask()
        fitnesses = []

        for s in solutions:
            nb_x_s = s[:N_NEIGHBORS].copy()
            nb_y_s = s[N_NEIGHBORS:].copy()

            # Skip if any neighbor is inside boundary + buffer
            if not _is_outside_boundary(nb_x_s, nb_y_s, boundary_np, BUFFER):
                fitnesses.append(0.0)  # zero regret penalty
                continue

            rng_key, subkey = jax.random.split(rng_key)
            result = compute_regret_full(
                jnp.array(nb_x_s), jnp.array(nb_y_s),
                liberal_x, liberal_y,
                sim, init_x, init_y, target_boundary, MIN_SPACING,
                sgd_settings, k_starts=K_STARTS, rng_key=subkey,
            )
            fitnesses.append(-result["regret"])

        es2.tell(solutions, fitnesses)

        best_idx = int(np.argmin(fitnesses))
        gen_best_regret = -fitnesses[best_idx]
        if gen_best_regret > best_p2_regret:
            best_p2_regret = gen_best_regret
            best_p2_params = solutions[best_idx].copy()

        p2_history_regret.append(best_p2_regret)
        p2_history_sigma.append(es2.sigma)
        gen += 1

        elapsed_gen = time.time() - t_gen
        if gen % 5 == 0 or gen == 1:
            print(
                f"{gen:4d}  {gen_best_regret:10.4f}  {best_p2_regret:10.4f}  "
                f"{es2.sigma:10.1f}  {elapsed_gen:5.1f}s",
                flush=True,
            )

    p2_elapsed = time.time() - t_p2_start
    print(f"\nPhase 2 done: {gen} generations in {p2_elapsed:.0f}s ({p2_elapsed/60:.1f} min)")
    print(f"  Best regret: {best_p2_regret:.4f} GWh", flush=True)

    # Final evaluation of best Phase 2 result
    final_nb_x = best_p2_params[:N_NEIGHBORS].copy()
    final_nb_y = best_p2_params[N_NEIGHBORS:].copy()
    # best_p2_params already passed boundary check during search

    rng_key, subkey = jax.random.split(rng_key)
    final_result = compute_regret_full(
        jnp.array(final_nb_x), jnp.array(final_nb_y),
        liberal_x, liberal_y,
        sim, init_x, init_y, target_boundary, MIN_SPACING,
        sgd_settings, k_starts=K_STARTS, rng_key=subkey,
    )

    print(f"\nFinal result:")
    print(f"  Regret:           {final_result['regret']:.4f} GWh")
    print(f"  Conservative AEP: {final_result['conservative_aep']:.2f} GWh")
    print(f"  Liberal AEP (present): {final_result['liberal_aep_present']:.2f} GWh")
    print(f"  Liberal AEP (alone):   {liberal_aep:.2f} GWh")
    if liberal_aep > 0:
        print(f"  Regret %: {final_result['regret'] / liberal_aep * 100:.2f}%")

    # ══════════════════════════════════════════════════════════════════════
    # Gradient magnitude analysis at the CMA-ES optimum
    # ══════════════════════════════════════════════════════════════════════

    print(f"\n{'='*60}")
    print("Gradient Analysis at CMA-ES Optimum")
    print(f"{'='*60}", flush=True)

    nb_x_j = jnp.array(final_nb_x)
    nb_y_j = jnp.array(final_nb_y)
    neighbor_params = jnp.concatenate([nb_x_j, nb_y_j])

    # Inner objective (takes params as third arg for IFT)
    def inner_obj_with_params(x, y, params):
        n_nb = params.shape[0] // 2
        nb_x_ = params[:n_nb]
        nb_y_ = params[n_nb:]
        x_all = jnp.concatenate([x, nb_x_])
        y_all = jnp.concatenate([y, nb_y_])
        result = sim(x_all, y_all, ws_amb=WS, wd_amb=WD)
        power = result.power()[:, :N_TARGET]
        return -jnp.sum(power) * 8760.0 / 1e6

    # Use sgd_solve_implicit to get differentiable conservative layout
    def compute_regret_differentiable(params):
        n_nb = params.shape[0] // 2
        nb_x_ = params[:n_nb]
        nb_y_ = params[n_nb:]
        opt_x, opt_y = sgd_solve_implicit(
            inner_obj_with_params,
            init_x, init_y,
            target_boundary, MIN_SPACING, sgd_settings,
            params,
        )
        # Conservative AEP
        x_all_c = jnp.concatenate([opt_x, nb_x_])
        y_all_c = jnp.concatenate([opt_y, nb_y_])
        result_c = sim(x_all_c, y_all_c, ws_amb=WS, wd_amb=WD)
        conservative_aep = jnp.sum(result_c.power()[:, :N_TARGET]) * 8760.0 / 1e6

        # Liberal AEP with neighbors present
        x_all_l = jnp.concatenate([liberal_x, nb_x_])
        y_all_l = jnp.concatenate([liberal_y, nb_y_])
        result_l = sim(x_all_l, y_all_l, ws_amb=WS, wd_amb=WD)
        liberal_aep_present = jnp.sum(result_l.power()[:, :N_TARGET]) * 8760.0 / 1e6

        return conservative_aep - liberal_aep_present

    print("Computing IFT gradient (this may take a minute)...", flush=True)
    t_grad = time.time()
    try:
        regret_val, grad_outer = jax.value_and_grad(compute_regret_differentiable)(neighbor_params)
        grad_outer_np = np.array(grad_outer)
        outer_grad_norm = float(np.linalg.norm(grad_outer_np))
        grad_elapsed = time.time() - t_grad
        print(f"  IFT regret: {float(regret_val):.4f} GWh")
        print(f"  Outer gradient norm: {outer_grad_norm:.6f}")
        print(f"  Gradient components: {grad_outer_np}")
        print(f"  Time: {grad_elapsed:.1f}s", flush=True)
        has_gradient = True
    except Exception as e:
        print(f"  IFT gradient failed: {e}")
        outer_grad_norm = None
        grad_outer_np = None
        has_gradient = False

    # Inner gradient norm (at convergence)
    def inner_obj_no_params(x, y):
        x_all = jnp.concatenate([x, nb_x_j])
        y_all = jnp.concatenate([y, nb_y_j])
        result = sim(x_all, y_all, ws_amb=WS, wd_amb=WD)
        power = result.power()[:, :N_TARGET]
        return -jnp.sum(power) * 8760.0 / 1e6

    opt_x = final_result["opt_x"]
    opt_y = final_result["opt_y"]
    inner_gx, inner_gy = jax.grad(inner_obj_no_params, argnums=(0, 1))(opt_x, opt_y)
    inner_grad_norm = float(jnp.sqrt(jnp.sum(inner_gx**2) + jnp.sum(inner_gy**2)))
    print(f"  Inner gradient norm at convergence: {inner_grad_norm:.6e}")

    if has_gradient and inner_grad_norm > 0:
        ratio = outer_grad_norm / inner_grad_norm
        print(f"  |grad_outer| / |grad_inner| = {ratio:.4f}")
    else:
        ratio = None

    # ══════════════════════════════════════════════════════════════════════
    # Save results
    # ══════════════════════════════════════════════════════════════════════

    results = {
        "config": {
            "D": D,
            "target_size": TARGET_SIZE,
            "min_spacing": MIN_SPACING,
            "n_target": N_TARGET,
            "n_neighbors": N_NEIGHBORS,
            "ws": [float(x) for x in WS],
            "wd": [float(x) for x in WD],
            "k_starts": K_STARTS,
            "buffer_D": BUFFER_D,
        },
        "liberal_aep": liberal_aep,
        "liberal_x": [float(x) for x in liberal_x],
        "liberal_y": [float(x) for x in liberal_y],
        "phase1": {
            "popsize": P1_POPSIZE,
            "sigma0": P1_SIGMA0,
            "maxiter": P1_MAXITER,
            "best_regret": best_p1_regret,
            "best_bearing_deg": best_p1_bearing,
            "best_distance_D": best_p1_dist,
            "history_regret": p1_history_regret,
            "history_sigma": p1_history_sigma,
            "elapsed_seconds": p1_elapsed,
        },
        "phase2": {
            "popsize": P2_POPSIZE,
            "sigma0": P2_SIGMA0,
            "maxiter": P2_MAXITER,
            "best_regret": best_p2_regret,
            "neighbor_x": final_nb_x.tolist(),
            "neighbor_y": final_nb_y.tolist(),
            "conservative_aep": final_result["conservative_aep"],
            "liberal_aep_present": final_result["liberal_aep_present"],
            "history_regret": p2_history_regret,
            "history_sigma": p2_history_sigma,
            "elapsed_seconds": p2_elapsed,
            "opt_target_x": [float(x) for x in final_result["opt_x"]],
            "opt_target_y": [float(x) for x in final_result["opt_y"]],
        },
        "gradient_analysis": {
            "outer_grad_norm": outer_grad_norm,
            "inner_grad_norm": inner_grad_norm,
            "ratio": ratio,
            "gradient_components": grad_outer_np.tolist() if grad_outer_np is not None else None,
        },
    }

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR / 'results.json'}")

    # ══════════════════════════════════════════════════════════════════════
    # Plots
    # ══════════════════════════════════════════════════════════════════════

    # ── Plot 1: Convergence (both phases) ────────────────────────────────

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(p1_history_regret, "o-", markersize=3, linewidth=1.5, color="darkred")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Regret (GWh)")
    ax.set_title("Phase 1: 2D CMA-ES (Bearing/Distance)")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(p1_history_sigma, "-", linewidth=1.5, color="darkorange")
    ax.set_xlabel("Generation")
    ax.set_ylabel("CMA-ES sigma (m)")
    ax.set_title("Phase 1: Step Size")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(p2_history_regret, "o-", markersize=2, linewidth=1.5, color="darkred")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Regret (GWh)")
    ax.set_title(f"Phase 2: Full-Position CMA-ES ({2*N_NEIGHBORS}D)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(p2_history_sigma, "-", linewidth=1.5, color="darkorange")
    ax.set_xlabel("Generation")
    ax.set_ylabel("CMA-ES sigma (m)")
    ax.set_title("Phase 2: Step Size")
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"CMA-ES Regret Search\n"
        f"Phase 1: {best_p1_regret:.4f} GWh → Phase 2: {best_p2_regret:.4f} GWh",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "convergence.png", dpi=150, bbox_inches="tight")
    print(f"Saved {OUTPUT_DIR / 'convergence.png'}")
    plt.close()

    # ── Plot 2: Final layout ─────────────────────────────────────────────

    fig, ax = plt.subplots(figsize=(10, 10))

    # Target boundary
    bnd = np.vstack([boundary_np, boundary_np[0]])
    ax.plot(bnd[:, 0] / D, bnd[:, 1] / D, "k-", linewidth=2, label="Target boundary")
    ax.fill(boundary_np[:, 0] / D, boundary_np[:, 1] / D, alpha=0.05, color="gray")

    # Liberal layout
    ax.scatter(
        np.array(liberal_x) / D, np.array(liberal_y) / D,
        c="royalblue", marker="^", s=80, alpha=0.6,
        label=f"Liberal (AEP={liberal_aep:.2f})", zorder=5,
    )

    # Conservative layout
    ax.scatter(
        np.array(final_result["opt_x"]) / D, np.array(final_result["opt_y"]) / D,
        c="seagreen", marker="s", s=80, alpha=0.8,
        label=f"Conservative (AEP={final_result['conservative_aep']:.2f})", zorder=5,
    )

    # Neighbors
    ax.scatter(
        final_nb_x / D, final_nb_y / D,
        c="red", marker="D", s=100, edgecolors="black", linewidths=0.8,
        label=f"Neighbors ({N_NEIGHBORS} turbines)", zorder=6,
    )

    # Phase 1 cluster (faded)
    ax.scatter(
        p1_nb_x / D, p1_nb_y / D,
        c="red", marker="x", s=60, alpha=0.3,
        label="Phase 1 cluster", zorder=4,
    )

    # Wind arrow
    arrow_x = float(boundary_np[:, 0].min() / D) - 3
    arrow_y = float(TARGET_SIZE / (2 * D))
    ax.annotate(
        "", xy=(arrow_x + 3, arrow_y), xytext=(arrow_x, arrow_y),
        arrowprops=dict(arrowstyle="->", color="blue", lw=2),
    )
    ax.text(arrow_x + 1.5, arrow_y + 1, "Wind 270°", ha="center", color="blue", fontsize=9)

    ax.set_xlabel("x (D)")
    ax.set_ylabel("y (D)")
    ax.set_title(
        f"CMA-ES Optimal Layout — Regret = {final_result['regret']:.4f} GWh "
        f"({final_result['regret']/liberal_aep*100:.2f}%)"
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "layout.png", dpi=150, bbox_inches="tight")
    print(f"Saved {OUTPUT_DIR / 'layout.png'}")
    plt.close()

    # ── Plot 3: Overlay on polar heatmap (if landscape results exist) ────

    if LANDSCAPE_RESULTS.exists():
        print("Loading landscape results for overlay...", flush=True)
        with open(LANDSCAPE_RESULTS) as f:
            landscape = json.load(f)

        bearings_ls = np.array(landscape["sweep1"]["bearings_deg"])
        distances_ls = np.array(landscape["sweep1"]["distances_D"])
        regret_grid_ls = np.array(landscape["sweep1"]["regret_grid"])

        fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(9, 9))

        theta_ls = np.radians(90.0 - bearings_ls)
        theta_grid_ls, dist_grid_ls = np.meshgrid(theta_ls, distances_ls, indexing="ij")

        pcm = ax.pcolormesh(
            theta_grid_ls, dist_grid_ls, regret_grid_ls,
            cmap="hot_r", shading="auto",
        )
        cbar = fig.colorbar(pcm, ax=ax, shrink=0.8, pad=0.08)
        cbar.set_label("Regret (GWh)")

        # CMA-ES Phase 1 trajectory
        for i, (b, d) in enumerate(zip(p1_history_bearing, p1_history_dist)):
            if b is not None and d is not None:
                t = np.radians(90.0 - b)
                alpha = 0.2 + 0.8 * (i / max(len(p1_history_bearing) - 1, 1))
                ax.plot(t, d, "c.", markersize=6, alpha=alpha)

        # Phase 1 optimum
        if best_p1_bearing is not None:
            t_opt = np.radians(90.0 - best_p1_bearing)
            ax.plot(t_opt, best_p1_dist, "c*", markersize=15,
                    markeredgecolor="white", label="Phase 1 optimum", zorder=10)

        # Phase 2 optimum (compute bearing/distance of centroid)
        p2_cx = np.mean(final_nb_x)
        p2_cy = np.mean(final_nb_y)
        p2_dx = p2_cx - target_center[0]
        p2_dy = p2_cy - target_center[1]
        p2_dist = np.sqrt(p2_dx**2 + p2_dy**2) / D
        p2_bearing = (90.0 - np.degrees(np.arctan2(p2_dy, p2_dx))) % 360
        p2_theta = np.radians(90.0 - p2_bearing)
        ax.plot(p2_theta, p2_dist, "w*", markersize=18,
                markeredgecolor="black", label="Phase 2 optimum", zorder=10)

        # Landscape peak
        peak_bearing_ls = landscape["sweep1"]["peak_bearing_deg"]
        peak_dist_ls = landscape["sweep1"]["peak_distance_D"]
        peak_theta_ls = np.radians(90.0 - peak_bearing_ls)
        ax.plot(peak_theta_ls, peak_dist_ls, "y*", markersize=15,
                markeredgecolor="black", label="Grid-search peak", zorder=10)

        compass_angles = [0, 45, 90, 135, 180, 225, 270, 315]
        compass_labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        polar_angles = [np.radians(90 - b) for b in compass_angles]
        ax.set_xticks(polar_angles)
        ax.set_xticklabels(compass_labels)

        ax.set_ylabel("Distance (D)", labelpad=30)
        ax.set_title(
            "CMA-ES Trajectory on Regret Landscape\n"
            f"Grid peak: {landscape['sweep1']['peak_regret']:.4f} GWh, "
            f"CMA-ES: {best_p2_regret:.4f} GWh",
            pad=20,
        )
        ax.legend(fontsize=8, loc="lower right")

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "overlay.png", dpi=150, bbox_inches="tight")
        print(f"Saved {OUTPUT_DIR / 'overlay.png'}")
        plt.close()
    else:
        print(f"No landscape results at {LANDSCAPE_RESULTS} — skipping overlay plot")

    # ── Plot 4: Gradient comparison ──────────────────────────────────────

    if has_gradient:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Bar chart: outer vs inner gradient norms
        ax = axes[0]
        bars = ax.bar(
            ["Inner\n(at convergence)", "Outer\n(IFT regret)"],
            [inner_grad_norm, outer_grad_norm],
            color=["seagreen", "darkred"],
            width=0.5,
        )
        ax.set_ylabel("Gradient Norm")
        ax.set_title("Gradient Magnitudes")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, axis="y")
        for bar, val in zip(bars, [inner_grad_norm, outer_grad_norm]):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.5,
                f"{val:.2e}", ha="center", va="bottom", fontsize=9,
            )

        # Per-component outer gradient
        ax = axes[1]
        n_nb = N_NEIGHBORS
        grad_x = grad_outer_np[:n_nb]
        grad_y = grad_outer_np[n_nb:]
        x_pos = np.arange(n_nb)
        width = 0.35
        ax.bar(x_pos - width / 2, grad_x, width, label="d(regret)/d(nb_x)", color="salmon")
        ax.bar(x_pos + width / 2, grad_y, width, label="d(regret)/d(nb_y)", color="cornflowerblue")
        ax.set_xlabel("Neighbor index")
        ax.set_ylabel("Gradient component")
        ax.set_title("IFT Gradient Components at CMA-ES Optimum")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_xticks(x_pos)

        ratio_str = f"|outer|/|inner| = {ratio:.4f}" if ratio else ""
        fig.suptitle(f"Gradient Analysis  {ratio_str}", fontsize=12)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "gradient_comparison.png", dpi=150, bbox_inches="tight")
        print(f"Saved {OUTPUT_DIR / 'gradient_comparison.png'}")
        plt.close()

    # ── Summary ──────────────────────────────────────────────────────────

    total_elapsed = p1_elapsed + p2_elapsed
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"  Phase 1 (2D):   {best_p1_regret:.4f} GWh in {p1_elapsed:.0f}s")
    print(f"  Phase 2 (full): {best_p2_regret:.4f} GWh in {p2_elapsed:.0f}s")
    print(f"  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    if has_gradient:
        print(f"  Outer gradient norm: {outer_grad_norm:.6f}")
        print(f"  Inner gradient norm: {inner_grad_norm:.6e}")
        if ratio:
            print(f"  Ratio |outer|/|inner|: {ratio:.4f}")


if __name__ == "__main__":
    main()

"""Comprehensive convergence study for design regret computation.

Following Quick et al. (OMAE 2026) which found ~400-500 starts needed for
66-turbine liberal layout convergence, this study sweeps:

1. K (conservative multistart): 1 to 500
2. K_lib (liberal multistart): 1 to 200
3. max_iter (SGD iterations): 100 to 10000
4. Multiple (bearing, distance) test points
5. Multiple wind roses

All with proper liberal AEP pooling (conservative AEP >= liberal AEP with neighbors).

Usage:
    pixi run python scripts/run_regret_convergence.py
"""

import jax
jax.config.update("jax_enable_x64", True)

import argparse
import json
import time
from functools import partial
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from pixwake import Curve, Turbine, WakeSimulation
from pixwake.deficit import BastankhahGaussianDeficit
from pixwake.optim.sgd import SGDSettings, topfarm_sgd_solve

print = partial(print, flush=True)

D = 240.0
N_TARGET = 50

# DEI polygon
_dk0w_raw = np.array([
    706694.3923283464, 6224158.532895836,
    703972.0844905999, 6226906.597455995,
    702624.6334635273, 6253853.5386425415,
    712771.6248419734, 6257704.934445341,
    715639.3355871611, 6260664.6846508905,
    721593.2420745814, 6257906.998015941,
]).reshape((-1, 2))
CENTROID_X = _dk0w_raw[:, 0].mean()
CENTROID_Y = _dk0w_raw[:, 1].mean()
from scipy.spatial import ConvexHull
from matplotlib.path import Path as MplPath
_hull = ConvexHull(_dk0w_raw - np.array([CENTROID_X, CENTROID_Y]))
boundary_np = (_dk0w_raw - np.array([CENTROID_X, CENTROID_Y]))[_hull.vertices]
boundary = jnp.array(boundary_np)
_polygon_path = MplPath(boundary_np)


def create_dei_turbine():
    ws = jnp.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25.0])
    power = jnp.array([0,0,2.399,209.258,689.198,1480.608,2661.238,4308.929,6501.057,
        9260.516,12081.404,13937.297,14705.016,14931.039,14985.209,14996.906,
        14999.343,14999.855,14999.966,14999.992,14999.998,14999.999,15000,15000,15000,15000.0])
    ct = jnp.array([0.889,0.889,0.889,0.8,0.8,0.8,0.8,0.8,0.8,0.793,0.735,0.610,
        0.476,0.370,0.292,0.234,0.191,0.158,0.132,0.112,0.096,0.083,0.072,0.063,0.055,0.049])
    return Turbine(rotor_diameter=240.0, hub_height=150.0,
        power_curve=Curve(ws=ws, values=power), ct_curve=Curve(ws=ws, values=ct))


def generate_target_grid(boundary_np, n_target, spacing):
    x_lo, x_hi = boundary_np[:, 0].min(), boundary_np[:, 0].max()
    y_lo, y_hi = boundary_np[:, 1].min(), boundary_np[:, 1].max()
    gx = np.arange(x_lo + 2*D, x_hi - 2*D, spacing)
    gy = np.arange(y_lo + 2*D, y_hi - 2*D, spacing)
    gx_2d, gy_2d = np.meshgrid(gx, gy)
    candidates = np.column_stack([gx_2d.ravel(), gy_2d.ravel()])
    inside = _polygon_path.contains_points(candidates)
    pts = candidates[inside]
    if len(pts) < n_target:
        gx = np.arange(x_lo + D, x_hi - D, spacing * 0.7)
        gy = np.arange(y_lo + D, y_hi - D, spacing * 0.7)
        gx_2d, gy_2d = np.meshgrid(gx, gy)
        candidates = np.column_stack([gx_2d.ravel(), gy_2d.ravel()])
        inside = _polygon_path.contains_points(candidates)
        pts = candidates[inside]
    indices = np.round(np.linspace(0, len(pts) - 1, n_target)).astype(int)
    return jnp.array(pts[indices, 0]), jnp.array(pts[indices, 1])


def compute_aep(sim, x, y, ws_amb, wd_amb, weights, ti_amb=None,
                neighbor_x=None, neighbor_y=None):
    if neighbor_x is not None and neighbor_x.shape[0] > 0:
        x_all = jnp.concatenate([x, neighbor_x])
        y_all = jnp.concatenate([y, neighbor_y])
    else:
        x_all, y_all = x, y
    n_target = x.shape[0]
    result = sim(x_all, y_all, ws_amb=ws_amb, wd_amb=wd_amb, ti_amb=ti_amb)
    power = result.power()[:, :n_target]
    return jnp.sum(power * weights[:, None]) * 8760 / 1e6


def random_start_inside_polygon(key, n_turbines):
    """Generate one random layout inside the polygon via rejection sampling."""
    pts = []
    while len(pts) < n_turbines:
        rx = jax.random.uniform(key, (n_turbines * 3,),
            minval=boundary_np[:, 0].min(), maxval=boundary_np[:, 0].max())
        key, _ = jax.random.split(key)
        ry = jax.random.uniform(key, (n_turbines * 3,),
            minval=boundary_np[:, 1].min(), maxval=boundary_np[:, 1].max())
        key, _ = jax.random.split(key)
        cands = np.column_stack([np.array(rx), np.array(ry)])
        inside = _polygon_path.contains_points(cands)
        pts.extend(cands[inside].tolist())
    pts = np.array(pts[:n_turbines])
    return jnp.array(pts[:, 0]), jnp.array(pts[:, 1])


def run_multistart_optimization(sim, objective_fn, boundary, min_spacing,
                                init_x, init_y, starts_x, starts_y,
                                sgd_settings):
    """Run K-start optimization, return best AEP and all individual AEPs."""
    all_aeps = []
    best_aep = -np.inf
    best_x, best_y = None, None

    for sx, sy in zip(starts_x, starts_y):
        opt_x, opt_y = topfarm_sgd_solve(
            objective_fn, sx, sy, boundary, min_spacing, sgd_settings)
        aep = float(-objective_fn(opt_x, opt_y))
        all_aeps.append(aep)
        if aep > best_aep:
            best_aep = aep
            best_x, best_y = opt_x, opt_y

    return best_aep, best_x, best_y, all_aeps


def place_reference_farm(bearing_deg, distance_D):
    """Place a 5x5 reference farm at (bearing, distance) from centroid."""
    spacing = 7 * D
    xs = np.arange(5) * spacing - 2 * spacing
    ys = np.arange(5) * spacing - 2 * spacing
    gx, gy = np.meshgrid(xs, ys)
    ref_x, ref_y = gx.ravel(), gy.ravel()

    bearing_rad = np.radians(bearing_deg)
    dist_m = distance_D * D
    cx = dist_m * np.sin(bearing_rad)
    cy = dist_m * np.cos(bearing_rad)
    return jnp.array(ref_x + cx), jnp.array(ref_y + cy)


def main():
    parser = argparse.ArgumentParser(description="Comprehensive convergence study")
    parser.add_argument("--output-dir", type=str, default="analysis/convergence_study")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    turbine = create_dei_turbine()
    deficit = BastankhahGaussianDeficit(k=0.04)
    sim = WakeSimulation(turbine, deficit)
    init_x, init_y = generate_target_grid(boundary_np, N_TARGET, spacing=4*D)

    # Test configurations: multiple wind roses × (bearing, distance)
    test_configs = [
        {"name": "concentrated_unidir_close", "ed_a": 0.9, "ed_f": 1.0,
         "bearing": 105, "distance_D": 5},
        {"name": "concentrated_unidir_far", "ed_a": 0.9, "ed_f": 1.0,
         "bearing": 105, "distance_D": 20},
        {"name": "moderate_bidir", "ed_a": 0.5, "ed_f": 0.0,
         "bearing": 210, "distance_D": 5},
    ]

    # Pre-generate a large pool of random starts (shared across all configs)
    K_MAX = 500
    print(f"Pre-generating {K_MAX} random starts...")
    all_random_x, all_random_y = [], []
    for k in range(K_MAX):
        key = jax.random.PRNGKey(k * 7919 + 42)
        rx, ry = random_start_inside_polygon(key, N_TARGET)
        all_random_x.append(rx)
        all_random_y.append(ry)

    all_results = {}

    for cfg in test_configs:
        name = cfg["name"]
        print(f"\n{'#'*70}")
        print(f"# CONFIG: {name}")
        print(f"# a={cfg['ed_a']}, f={cfg['ed_f']}, "
              f"bearing={cfg['bearing']}°, dist={cfg['distance_D']}D")
        print(f"{'#'*70}")

        # Wind rose
        from edrose import EllipticalWindRose
        wr = EllipticalWindRose(a=cfg["ed_a"], f=cfg["ed_f"],
                                theta_prev=270, n_sectors=24)
        wd = jnp.array(wr.wind_directions)
        weights = jnp.array(wr.sector_frequencies)
        ws = jnp.full_like(wd, 9.0)

        # =====================================================================
        # PART A: Liberal layout convergence (K_lib sweep)
        # =====================================================================
        print(f"\n{'='*60}")
        print("PART A: Liberal layout convergence")
        print(f"{'='*60}")

        def lib_obj(x, y):
            return -compute_aep(sim, x, y, ws, wd, weights)

        K_lib_values = [1, 2, 5, 10, 20, 50, 100, 200]
        lib_settings = SGDSettings(learning_rate=50.0, max_iter=10000,
                                   additional_constant_lr_iterations=10000, tol=1e-6)

        # Build liberal starts: grid init + randoms
        lib_starts_x = [init_x] + all_random_x[:K_MAX-1]
        lib_starts_y = [init_y] + all_random_y[:K_MAX-1]

        # Run all liberal starts once, accumulate
        print("Running 200 liberal optimizations...")
        lib_all_aeps = []
        lib_all_xs, lib_all_ys = [], []
        for k in range(max(K_lib_values)):
            t0 = time.time()
            opt_x, opt_y = topfarm_sgd_solve(
                lib_obj, lib_starts_x[k], lib_starts_y[k],
                boundary, D * 4, lib_settings)
            aep = float(-lib_obj(opt_x, opt_y))
            lib_all_aeps.append(aep)
            lib_all_xs.append(opt_x)
            lib_all_ys.append(opt_y)
            if (k + 1) % 50 == 0 or k == 0:
                elapsed = time.time() - t0
                print(f"  Start {k+1}/{max(K_lib_values)}: "
                      f"AEP={aep:.2f} GWh, best so far={max(lib_all_aeps):.2f} GWh "
                      f"({elapsed:.1f}s)")

        # Report best-of-K for each K_lib
        lib_results = []
        print(f"\n{'K_lib':>6} {'Best Liberal AEP':>18} {'Δ from K=200':>14}")
        print("-" * 42)
        best_lib_200 = max(lib_all_aeps[:200])
        for K_lib in K_lib_values:
            best = max(lib_all_aeps[:K_lib])
            delta = best - best_lib_200
            print(f"{K_lib:>6} {best:>18.2f} {delta:>+13.2f} GWh")
            lib_results.append({"K_lib": K_lib, "best_aep": best,
                                "all_aeps": lib_all_aeps[:K_lib]})

        # Use K_lib=200 liberal layout as the reference
        best_lib_idx = int(np.argmax(lib_all_aeps[:200]))
        liberal_x = lib_all_xs[best_lib_idx]
        liberal_y = lib_all_ys[best_lib_idx]
        liberal_aep = lib_all_aeps[best_lib_idx]
        print(f"\nUsing liberal layout from start {best_lib_idx} "
              f"(AEP={liberal_aep:.2f} GWh)")

        # =====================================================================
        # PART B: Conservative convergence (K sweep at fixed max_iter=5000)
        # =====================================================================
        print(f"\n{'='*60}")
        print("PART B: Conservative convergence (K sweep, max_iter=5000)")
        print(f"{'='*60}")

        nx, ny = place_reference_farm(cfg["bearing"], cfg["distance_D"])
        lib_aep_present = float(compute_aep(
            sim, liberal_x, liberal_y, ws, wd, weights,
            neighbor_x=nx, neighbor_y=ny))
        print(f"Liberal AEP with neighbors: {lib_aep_present:.2f} GWh")

        cons_settings = SGDSettings(learning_rate=50.0, max_iter=5000,
                                    additional_constant_lr_iterations=5000, tol=1e-6)

        def cons_obj(x, y):
            return -compute_aep(sim, x, y, ws, wd, weights,
                                neighbor_x=nx, neighbor_y=ny)

        # Build conservative starts: grid, liberal, then randoms
        cons_starts_x = [init_x, liberal_x] + all_random_x[:K_MAX-2]
        cons_starts_y = [init_y, liberal_y] + all_random_y[:K_MAX-2]

        # Run all conservative starts, accumulate
        K_cons_values = [1, 2, 3, 5, 10, 20, 50, 100, 200, 300, 400, 500]
        max_K_cons = max(K_cons_values)
        print(f"Running {max_K_cons} conservative optimizations...")

        cons_all_aeps = []
        for k in range(max_K_cons):
            t0 = time.time()
            opt_x, opt_y = topfarm_sgd_solve(
                cons_obj, cons_starts_x[k], cons_starts_y[k],
                boundary, D * 4, cons_settings)
            aep = float(-cons_obj(opt_x, opt_y))
            cons_all_aeps.append(aep)
            if (k + 1) % 50 == 0 or k < 5:
                elapsed = time.time() - t0
                best_so_far = max(cons_all_aeps)
                regret_so_far = max(best_so_far, lib_aep_present) - lib_aep_present
                print(f"  Start {k+1}/{max_K_cons}: "
                      f"AEP={aep:.2f}, best={best_so_far:.2f}, "
                      f"regret={regret_so_far:.2f} GWh ({elapsed:.1f}s)")

        # Report best-of-K for each K (with pooling)
        cons_results = []
        print(f"\n{'K':>5} {'Best Cons AEP':>15} {'Regret (GWh)':>14} {'Regret (%)':>12} {'Δ from K=500':>14}")
        print("-" * 65)
        ref_regret = max(max(cons_all_aeps[:500]), lib_aep_present) - lib_aep_present
        for K in K_cons_values:
            best = max(cons_all_aeps[:K])
            best_pooled = max(best, lib_aep_present)
            regret = best_pooled - lib_aep_present
            pct = 100 * regret / liberal_aep
            delta = regret - ref_regret
            print(f"{K:>5} {best_pooled:>15.2f} {regret:>14.2f} {pct:>11.3f}% {delta:>+13.2f}")
            cons_results.append({
                "K": K, "best_cons_aep": float(best_pooled),
                "regret_gwh": float(regret), "regret_pct": float(pct),
                "best_raw_aep": float(best),
                "all_aeps": [float(a) for a in cons_all_aeps[:K]],
            })

        # =====================================================================
        # PART C: SGD iteration convergence (vary max_iter at K=200)
        # =====================================================================
        print(f"\n{'='*60}")
        print("PART C: SGD iteration convergence (K=200, vary max_iter)")
        print(f"{'='*60}")

        iter_values = [100, 500, 1000, 2000, 5000, 10000]
        iter_results = []

        for max_iter in iter_values:
            iter_settings = SGDSettings(
                learning_rate=50.0, max_iter=max_iter,
                additional_constant_lr_iterations=max_iter, tol=1e-6)

            iter_aeps = []
            t0 = time.time()
            for k in range(200):
                opt_x, opt_y = topfarm_sgd_solve(
                    cons_obj, cons_starts_x[k], cons_starts_y[k],
                    boundary, D * 4, iter_settings)
                aep = float(-cons_obj(opt_x, opt_y))
                iter_aeps.append(aep)

            elapsed = time.time() - t0
            best = max(iter_aeps)
            best_pooled = max(best, lib_aep_present)
            regret = best_pooled - lib_aep_present
            pct = 100 * regret / liberal_aep
            print(f"  iter={max_iter:>6}: regret={regret:.2f} GWh ({pct:.3f}%), "
                  f"best_cons={best_pooled:.2f}, time={elapsed:.0f}s")
            iter_results.append({
                "max_iter": max_iter, "regret_gwh": float(regret),
                "regret_pct": float(pct), "best_cons_aep": float(best_pooled),
                "elapsed_s": elapsed,
            })

        # =====================================================================
        # PART D: Bootstrap analysis — stability of regret estimate
        # =====================================================================
        print(f"\n{'='*60}")
        print("PART D: Bootstrap analysis (1000 reshuffles of 500 starts)")
        print(f"{'='*60}")

        n_bootstrap = 1000
        aeps_array = np.array(cons_all_aeps[:500])

        for K_test in [5, 10, 20, 50, 100, 200]:
            bootstrap_regrets = []
            for _ in range(n_bootstrap):
                sample = np.random.choice(aeps_array, size=K_test, replace=True)
                best = sample.max()
                best_pooled = max(best, lib_aep_present)
                bootstrap_regrets.append(best_pooled - lib_aep_present)
            br = np.array(bootstrap_regrets)
            print(f"  K={K_test:>4}: regret = {br.mean():.2f} ± {br.std():.2f} GWh "
                  f"(p5={np.percentile(br, 5):.2f}, p50={np.percentile(br, 50):.2f}, "
                  f"p95={np.percentile(br, 95):.2f})")

        # Save everything
        config_results = {
            "config": cfg,
            "liberal_aep_gwh": float(liberal_aep),
            "liberal_aep_present_gwh": float(lib_aep_present),
            "liberal_convergence": lib_results,
            "conservative_convergence": cons_results,
            "iteration_convergence": iter_results,
            "bootstrap": {
                "n_bootstrap": n_bootstrap,
                "n_starts_pool": 500,
            },
        }
        all_results[name] = config_results

    # Save all results
    with open(output_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{'='*70}")
    print(f"All results saved: {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()

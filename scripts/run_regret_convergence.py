"""Convergence study: regret vs number of multistart restarts and SGD iterations.

Tests whether K=5 with 5000 iterations is sufficient for accurate regret computation
with 50 turbines. Runs at a single (bearing, distance) configuration with a 5x5
reference neighbor farm, sweeping K and max_iter.

Usage:
    pixi run python scripts/run_regret_convergence.py
    pixi run python scripts/run_regret_convergence.py --bearing 105 --distance-D 5
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


def run_conservative(sim, starts_x, starts_y, liberal_x, liberal_y,
                     nx, ny, ws, wd, weights, ti_amb, sgd_settings):
    """Run K conservative optimizations and return best AEP (pooled with liberal)."""
    K = len(starts_x)
    lib_aep_present = float(compute_aep(
        sim, liberal_x, liberal_y, ws, wd, weights, ti_amb,
        neighbor_x=nx, neighbor_y=ny))

    best_cons_aep = lib_aep_present  # Pool: floor is liberal with neighbors
    all_aeps = [lib_aep_present]  # Track all AEPs including liberal floor

    for k in range(K):
        def cons_obj(x, y):
            return -compute_aep(sim, x, y, ws, wd, weights, ti_amb,
                                neighbor_x=nx, neighbor_y=ny)
        cx, cy = topfarm_sgd_solve(cons_obj, starts_x[k], starts_y[k],
                                   boundary, D * 4, sgd_settings)
        aep = float(compute_aep(sim, cx, cy, ws, wd, weights, ti_amb,
                                neighbor_x=nx, neighbor_y=ny))
        all_aeps.append(aep)
        if aep > best_cons_aep:
            best_cons_aep = aep

    regret = best_cons_aep - lib_aep_present
    return regret, best_cons_aep, lib_aep_present, all_aeps


def generate_starts(init_x, init_y, liberal_x, liberal_y, K):
    """Generate K initial positions: grid, liberal, then random."""
    starts_x, starts_y = [], []
    for k in range(K):
        if k == 0:
            starts_x.append(init_x)
            starts_y.append(init_y)
        elif k == 1:
            starts_x.append(liberal_x)
            starts_y.append(liberal_y)
        else:
            key = jax.random.PRNGKey(k * 7919)
            pts = []
            while len(pts) < N_TARGET:
                rx = jax.random.uniform(key, (N_TARGET * 3,),
                    minval=boundary_np[:, 0].min(), maxval=boundary_np[:, 0].max())
                key, _ = jax.random.split(key)
                ry = jax.random.uniform(key, (N_TARGET * 3,),
                    minval=boundary_np[:, 1].min(), maxval=boundary_np[:, 1].max())
                key, _ = jax.random.split(key)
                cands = np.column_stack([np.array(rx), np.array(ry)])
                inside = _polygon_path.contains_points(cands)
                pts.extend(cands[inside].tolist())
            pts = np.array(pts[:N_TARGET])
            starts_x.append(jnp.array(pts[:, 0]))
            starts_y.append(jnp.array(pts[:, 1]))
    return starts_x, starts_y


def main():
    parser = argparse.ArgumentParser(description="Convergence study for regret")
    parser.add_argument("--bearing", type=float, default=105.0)
    parser.add_argument("--distance-D", type=float, default=5.0)
    parser.add_argument("--ed-a", type=float, default=0.9)
    parser.add_argument("--ed-f", type=float, default=1.0)
    parser.add_argument("--wind-dir", type=float, default=270.0)
    parser.add_argument("--wind-speed", type=float, default=9.0)
    parser.add_argument("--output-dir", type=str, default="analysis/convergence_study")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from edrose import EllipticalWindRose
    wr = EllipticalWindRose(a=args.ed_a, f=args.ed_f,
                            theta_prev=args.wind_dir, n_sectors=24)
    wd = jnp.array(wr.wind_directions)
    weights = jnp.array(wr.sector_frequencies)
    ws = jnp.full_like(wd, args.wind_speed)

    turbine = create_dei_turbine()
    deficit = BastankhahGaussianDeficit(k=0.04)
    sim = WakeSimulation(turbine, deficit)

    init_x, init_y = generate_target_grid(boundary_np, N_TARGET, spacing=4*D)

    # Liberal layout with generous iterations
    print("Computing liberal layout (K=1, 10000 iter)...")
    lib_settings = SGDSettings(learning_rate=50.0, max_iter=10000,
                               additional_constant_lr_iterations=10000, tol=1e-6)
    def lib_obj(x, y):
        return -compute_aep(sim, x, y, ws, wd, weights)
    liberal_x, liberal_y = topfarm_sgd_solve(lib_obj, init_x, init_y, boundary, D*4, lib_settings)
    liberal_aep = float(compute_aep(sim, liberal_x, liberal_y, ws, wd, weights))
    print(f"Liberal AEP: {liberal_aep:.2f} GWh")

    # Place reference neighbor
    ref_x_local = np.arange(5) * 7 * D - 2 * 7 * D
    ref_y_local = np.arange(5) * 7 * D - 2 * 7 * D
    ref_x_local, ref_y_local = np.meshgrid(ref_x_local, ref_y_local)
    ref_x_local, ref_y_local = ref_x_local.ravel(), ref_y_local.ravel()
    bearing_rad = np.radians(args.bearing)
    dist_m = args.distance_D * D
    cx = dist_m * np.sin(bearing_rad)
    cy = dist_m * np.cos(bearing_rad)
    nx = jnp.array(ref_x_local + cx)
    ny = jnp.array(ref_y_local + cy)

    print(f"Neighbor: 5x5 farm at bearing={args.bearing}°, dist={args.distance_D}D")

    # Pre-generate a large pool of starts (reuse across K values)
    K_max = 200
    all_starts_x, all_starts_y = generate_starts(init_x, init_y, liberal_x, liberal_y, K_max)

    # === Sweep 1: Vary K at fixed max_iter=5000 ===
    K_values = [1, 2, 3, 5, 10, 20, 50, 100, 200]
    print(f"\n{'='*70}")
    print(f"SWEEP 1: Vary K (inner starts) at max_iter=5000, lr=50")
    print(f"{'='*70}")
    print(f"{'K':>5} {'Regret (GWh)':>14} {'Regret (%)':>12} {'Cons AEP':>12} {'Time (s)':>10}")
    print("-" * 60)

    results_K = []
    for K in K_values:
        sgd_s = SGDSettings(learning_rate=50.0, max_iter=5000,
                            additional_constant_lr_iterations=5000, tol=1e-6)
        t0 = time.time()
        regret, cons_aep, lib_pres, all_aeps = run_conservative(
            sim, all_starts_x[:K], all_starts_y[:K],
            liberal_x, liberal_y, nx, ny, ws, wd, weights, None, sgd_s)
        elapsed = time.time() - t0
        pct = 100 * regret / liberal_aep
        print(f"{K:>5} {regret:>14.2f} {pct:>11.3f}% {cons_aep:>12.2f} {elapsed:>10.1f}")
        results_K.append({"K": K, "regret_gwh": float(regret), "regret_pct": float(pct),
                          "conservative_aep": float(cons_aep),
                          "liberal_aep_present": float(lib_pres),
                          "elapsed_s": elapsed,
                          "all_aeps": [float(a) for a in all_aeps]})

    # === Sweep 2: Vary max_iter at fixed K=50 ===
    iter_values = [100, 500, 1000, 2000, 5000, 10000]
    print(f"\n{'='*70}")
    print(f"SWEEP 2: Vary max_iter at K=50, lr=50")
    print(f"{'='*70}")
    print(f"{'Iter':>7} {'Regret (GWh)':>14} {'Regret (%)':>12} {'Cons AEP':>12} {'Time (s)':>10}")
    print("-" * 60)

    results_iter = []
    for max_iter in iter_values:
        sgd_s = SGDSettings(learning_rate=50.0, max_iter=max_iter,
                            additional_constant_lr_iterations=max_iter, tol=1e-6)
        t0 = time.time()
        regret, cons_aep, lib_pres, all_aeps = run_conservative(
            sim, all_starts_x[:50], all_starts_y[:50],
            liberal_x, liberal_y, nx, ny, ws, wd, weights, None, sgd_s)
        elapsed = time.time() - t0
        pct = 100 * regret / liberal_aep
        print(f"{max_iter:>7} {regret:>14.2f} {pct:>11.3f}% {cons_aep:>12.2f} {elapsed:>10.1f}")
        results_iter.append({"max_iter": max_iter, "regret_gwh": float(regret),
                             "regret_pct": float(pct), "conservative_aep": float(cons_aep),
                             "elapsed_s": elapsed})

    # Save
    results = {
        "bearing_deg": args.bearing,
        "distance_D": args.distance_D,
        "liberal_aep_gwh": float(liberal_aep),
        "ed_a": args.ed_a, "ed_f": args.ed_f,
        "sweep_K": results_K,
        "sweep_iter": results_iter,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()

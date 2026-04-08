"""Regret cross-section: place a reference neighbor farm at (bearing, distance)
positions around the DEI target farm and measure design regret at each.

Produces a polar heatmap showing the farm's directional vulnerability.

Usage:
    pixi run python scripts/run_regret_cross_section.py
    pixi run python scripts/run_regret_cross_section.py --wind-rose elliptical --ed-a 0.8 --ed-f 0.5
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
from pixwake.deficit.gaussian import TurboGaussianDeficit
from pixwake.optim.sgd import SGDSettings, topfarm_sgd_solve

print = partial(print, flush=True)

D = 240.0
N_TARGET = 50

# =============================================================================
# DEI polygon boundary (same as run_dei_greedy_grid.py)
# =============================================================================
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
    ws = jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                    13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25.0])
    power = jnp.array([
        0.0, 0.0, 2.399, 209.258, 689.198, 1480.608,
        2661.238, 4308.929, 6501.057, 9260.516, 12081.404, 13937.297,
        14705.016, 14931.039, 14985.209, 14996.906, 14999.343, 14999.855,
        14999.966, 14999.992, 14999.998, 14999.999, 15000.0, 15000.0,
        15000.0, 15000.0,
    ])
    ct = jnp.array([
        0.889, 0.889, 0.889, 0.800, 0.800, 0.800,
        0.800, 0.800, 0.800, 0.793, 0.735, 0.610,
        0.476, 0.370, 0.292, 0.234, 0.191, 0.158,
        0.132, 0.112, 0.096, 0.083, 0.072, 0.063,
        0.055, 0.049,
    ])
    return Turbine(
        rotor_diameter=240.0, hub_height=150.0,
        power_curve=Curve(ws=ws, values=power),
        ct_curve=Curve(ws=ws, values=ct),
    )


def generate_target_grid(boundary_np, n_target, spacing):
    x_lo, x_hi = boundary_np[:, 0].min(), boundary_np[:, 0].max()
    y_lo, y_hi = boundary_np[:, 1].min(), boundary_np[:, 1].max()
    gx = np.arange(x_lo + 2 * D, x_hi - 2 * D, spacing)
    gy = np.arange(y_lo + 2 * D, y_hi - 2 * D, spacing)
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
    selected = pts[indices]
    return jnp.array(selected[:, 0]), jnp.array(selected[:, 1])


def load_wind_data():
    import pandas as pd
    csv_path = Path(__file__).parent.parent / "energy_island_10y_daily_av_wind.csv"
    df = pd.read_csv(csv_path, sep=";")
    wd_ts, ws_ts = df["WD_150"].values, df["WS_150"].values
    n_bins = 24
    bin_edges = np.linspace(0, 360, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    w = np.zeros(n_bins)
    mean_speeds = np.zeros(n_bins)
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (wd_ts >= bin_edges[i]) | (wd_ts < bin_edges[0])
        else:
            mask = (wd_ts >= bin_edges[i]) & (wd_ts < bin_edges[i + 1])
        w[i] = mask.sum()
        mean_speeds[i] = ws_ts[mask].mean() if mask.sum() > 0 else ws_ts.mean()
    w /= w.sum()
    return jnp.array(bin_centers), jnp.array(mean_speeds), jnp.array(w)


def build_reference_farm(n_rows, n_cols, spacing_D):
    """Build a rectangular reference farm centered at origin."""
    spacing = spacing_D * D
    xs = np.arange(n_cols) * spacing - (n_cols - 1) * spacing / 2
    ys = np.arange(n_rows) * spacing - (n_rows - 1) * spacing / 2
    gx, gy = np.meshgrid(xs, ys)
    return gx.ravel(), gy.ravel()


def place_reference_farm(ref_x, ref_y, bearing_deg, distance_m, centroid_x=0, centroid_y=0):
    """Place the reference farm at (bearing, distance) from centroid.

    Bearing is meteorological convention: 0=N, 90=E, 180=S, 270=W.
    """
    bearing_rad = np.radians(bearing_deg)
    # Meteorological: 0=N means +y, 90=E means +x
    cx = centroid_x + distance_m * np.sin(bearing_rad)
    cy = centroid_y + distance_m * np.cos(bearing_rad)
    return jnp.array(ref_x + cx), jnp.array(ref_y + cy)


def compute_aep(sim, x, y, ws_amb, wd_amb, weights, ti_amb=None,
                neighbor_x=None, neighbor_y=None, n_target=None):
    """Compute AEP for target turbines, optionally with neighbors.

    Returns a JAX scalar (not float) so this can be used inside jax.grad.
    """
    if n_target is None:
        n_target = x.shape[0]
    if neighbor_x is not None and neighbor_x.shape[0] > 0:
        x_all = jnp.concatenate([x, neighbor_x])
        y_all = jnp.concatenate([y, neighbor_y])
    else:
        x_all = x
        y_all = y
    result = sim(x_all, y_all, ws_amb=ws_amb, wd_amb=wd_amb, ti_amb=ti_amb)
    power = result.power()[:, :n_target]
    return jnp.sum(power * weights[:, None]) * 8760 / 1e6


def main():
    parser = argparse.ArgumentParser(
        description="Regret cross-section: reference farm at (bearing, distance)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Target farm
    parser.add_argument("--n-target", type=int, default=N_TARGET)
    parser.add_argument("--inner-lr", type=float, default=50.0)
    parser.add_argument("--inner-max-iter", type=int, default=5000)
    parser.add_argument("--n-inner-starts", type=int, default=5)

    # Reference neighbor farm
    parser.add_argument("--ref-rows", type=int, default=5)
    parser.add_argument("--ref-cols", type=int, default=5)
    parser.add_argument("--ref-spacing-D", type=float, default=7.0)

    # Sweep parameters
    parser.add_argument("--n-bearings", type=int, default=24)
    parser.add_argument("--distances-D", type=str, default="5,10,15,20,30,40,60",
                        help="Comma-separated distances in D")

    # Wind rose
    parser.add_argument("--wind-rose", type=str, default="dei",
                        choices=["dei", "unidirectional", "uniform", "elliptical", "mixture"])
    parser.add_argument("--wind-dir", type=float, default=270.0)
    parser.add_argument("--wind-speed", type=float, default=9.0)
    parser.add_argument("--n-bins", type=int, default=24)
    parser.add_argument("--ed-a", type=float, default=0.8)
    parser.add_argument("--ed-f", type=float, default=1.0)
    parser.add_argument("--ed-a2", type=float, default=0.8)
    parser.add_argument("--ed-f2", type=float, default=1.0)
    parser.add_argument("--wind-dir2", type=float, default=90.0)
    parser.add_argument("--mixture-weight", type=float, default=0.7)

    # Wake model
    parser.add_argument("--deficit", type=str, default="bastankhah",
                        choices=["bastankhah", "turbopark"])
    parser.add_argument("--superposition", type=str, default="squaredsum",
                        choices=["squaredsum", "linearsum"])
    parser.add_argument("--ti", type=float, default=0.06)

    parser.add_argument("--output-dir", type=str, default="analysis/regret_cross_section")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    distances_D = [float(x) for x in args.distances_D.split(",")]
    distances_m = [d * D for d in distances_D]
    bearings = np.linspace(0, 360 - 360 / args.n_bearings, args.n_bearings)

    print("=" * 70)
    print("REGRET CROSS-SECTION")
    print(f"  Target: {args.n_target} turbines in DEI polygon")
    print(f"  Reference farm: {args.ref_rows}x{args.ref_cols} = "
          f"{args.ref_rows * args.ref_cols} turbines at {args.ref_spacing_D}D spacing")
    print(f"  Bearings: {args.n_bearings} ({bearings[1]-bearings[0]:.0f} deg steps)")
    print(f"  Distances: {distances_D} D")
    print(f"  Total evaluations: {args.n_bearings * len(distances_D)}")
    print(f"  Wind rose: {args.wind_rose}")
    print(f"  Deficit: {args.deficit}")
    print(f"  K={args.n_inner_starts} inner starts, {args.inner_max_iter} iter")
    print("=" * 70)

    # Setup
    turbine = create_dei_turbine()
    if args.wind_rose == "dei":
        wd, ws, weights = load_wind_data()
    elif args.wind_rose == "unidirectional":
        wd = jnp.array([args.wind_dir])
        ws = jnp.array([args.wind_speed])
        weights = jnp.array([1.0])
    elif args.wind_rose == "uniform":
        n_bins = args.n_bins
        wd = jnp.linspace(0, 360 - 360 / n_bins, n_bins)
        ws = jnp.full(n_bins, args.wind_speed)
        weights = jnp.full(n_bins, 1.0 / n_bins)
    elif args.wind_rose == "elliptical":
        from edrose import EllipticalWindRose
        wr = EllipticalWindRose(a=args.ed_a, f=args.ed_f,
                                theta_prev=args.wind_dir, n_sectors=args.n_bins)
        wd = jnp.array(wr.wind_directions)
        weights = jnp.array(wr.sector_frequencies)
        ws = jnp.full_like(wd, args.wind_speed)
    elif args.wind_rose == "mixture":
        from edrose import EllipticalWindRose, MixtureEllipticalWindRose
        c1 = EllipticalWindRose(a=args.ed_a, f=args.ed_f,
                                theta_prev=args.wind_dir, n_sectors=args.n_bins)
        c2 = EllipticalWindRose(a=args.ed_a2, f=args.ed_f2,
                                theta_prev=args.wind_dir2, n_sectors=args.n_bins)
        mix = MixtureEllipticalWindRose([c1, c2],
                                        weights=[args.mixture_weight, 1 - args.mixture_weight])
        wd = jnp.array(mix.wind_directions)
        weights = jnp.array(mix.sector_frequencies)
        ws = jnp.full_like(wd, args.wind_speed)

    from pixwake.superposition import LinearSum, SquaredSum
    sup = LinearSum() if args.superposition == "linearsum" else SquaredSum()
    if args.deficit == "bastankhah":
        deficit = BastankhahGaussianDeficit(k=0.04, superposition=sup)
    elif args.deficit == "turbopark":
        deficit = TurboGaussianDeficit(A=0.04, superposition=sup)
    print(f"  Superposition: {type(sup).__name__}")
    sim = WakeSimulation(turbine, deficit)
    ti_amb = args.ti if args.deficit == "turbopark" else None

    # Target layout + liberal optimization
    init_x, init_y = generate_target_grid(boundary_np, args.n_target, spacing=4 * D)
    sgd_settings = SGDSettings(
        learning_rate=args.inner_lr,
        max_iter=args.inner_max_iter,
        additional_constant_lr_iterations=args.inner_max_iter,
        tol=1e-6,
    )

    # Liberal layout with multistart
    print(f"\nComputing liberal layout (K_lib={min(args.n_inner_starts, 100)} starts)...")
    def liberal_objective(x, y):
        return -compute_aep(sim, x, y, ws, wd, weights, ti_amb)

    K_lib = min(args.n_inner_starts, 100)
    lib_best_aep = -np.inf
    liberal_x, liberal_y = init_x, init_y
    for k in range(K_lib):
        if k == 0:
            sx, sy = init_x, init_y
        else:
            key = jax.random.PRNGKey(k * 7919 + 42)
            n_turb = args.n_target
            pts = []
            while len(pts) < n_turb:
                rx = jax.random.uniform(key, (n_turb * 3,),
                                        minval=boundary_np[:, 0].min(),
                                        maxval=boundary_np[:, 0].max())
                key, _ = jax.random.split(key)
                ry = jax.random.uniform(key, (n_turb * 3,),
                                        minval=boundary_np[:, 1].min(),
                                        maxval=boundary_np[:, 1].max())
                key, _ = jax.random.split(key)
                cands = np.column_stack([np.array(rx), np.array(ry)])
                inside = _polygon_path.contains_points(cands)
                pts.extend(cands[inside].tolist())
            pts = np.array(pts[:n_turb])
            sx, sy = jnp.array(pts[:, 0]), jnp.array(pts[:, 1])
        lx, ly = topfarm_sgd_solve(liberal_objective, sx, sy, boundary, D * 4, sgd_settings)
        lib_aep = float(-liberal_objective(lx, ly))
        if lib_aep > lib_best_aep:
            lib_best_aep = lib_aep
            liberal_x, liberal_y = lx, ly
    liberal_aep = lib_best_aep
    print(f"Liberal AEP: {liberal_aep:.2f} GWh (best of {K_lib} starts)")

    # Build reference farm
    ref_x_local, ref_y_local = build_reference_farm(
        args.ref_rows, args.ref_cols, args.ref_spacing_D)
    n_ref = len(ref_x_local)
    print(f"Reference farm: {n_ref} turbines")

    # =========================================================================
    # VMAP-BATCHED SWEEP
    # Batch all (position, start) pairs and vmap in chunks for GPU efficiency
    # =========================================================================
    n_positions = len(distances_m) * args.n_bearings
    K = args.n_inner_starts

    # Build all neighbor positions: (n_positions, n_ref) arrays
    all_nx_np = np.zeros((n_positions, n_ref))
    all_ny_np = np.zeros((n_positions, n_ref))
    pos_info = []  # (di, bi, bearing, dist_m) for each position
    pi = 0
    for di, dist_m in enumerate(distances_m):
        for bi, bearing in enumerate(bearings):
            nx, ny = place_reference_farm(ref_x_local, ref_y_local, bearing, dist_m)
            all_nx_np[pi] = np.array(nx)
            all_ny_np[pi] = np.array(ny)
            pos_info.append((di, bi, float(bearing), float(dist_m)))
            pi += 1
    all_nx = jnp.array(all_nx_np)
    all_ny = jnp.array(all_ny_np)

    # Build K starts (shared across all positions): grid, liberal, random
    print(f"Generating {K} start layouts...")
    start_xs_list = [init_x, liberal_x]
    start_ys_list = [init_y, liberal_y]
    for k in range(2, K):
        key = jax.random.PRNGKey(k * 7919 + 42)
        n_turb = args.n_target
        pts = []
        while len(pts) < n_turb:
            rx = jax.random.uniform(key, (n_turb * 3,),
                                    minval=boundary_np[:, 0].min(),
                                    maxval=boundary_np[:, 0].max())
            key, _ = jax.random.split(key)
            ry = jax.random.uniform(key, (n_turb * 3,),
                                    minval=boundary_np[:, 1].min(),
                                    maxval=boundary_np[:, 1].max())
            key, _ = jax.random.split(key)
            cands = np.column_stack([np.array(rx), np.array(ry)])
            inside = _polygon_path.contains_points(cands)
            pts.extend(cands[inside].tolist())
        pts = np.array(pts[:n_turb])
        start_xs_list.append(jnp.array(pts[:, 0]))
        start_ys_list.append(jnp.array(pts[:, 1]))
    start_xs = jnp.stack(start_xs_list)  # (K, n_target)
    start_ys = jnp.stack(start_ys_list)

    # Build batched (position × start) arrays: (n_positions * K, ...)
    # For each position, repeat K starts
    # For each start, repeat n_positions neighbor positions
    batch_nx = jnp.repeat(all_nx, K, axis=0)  # (n_positions * K, n_ref)
    batch_ny = jnp.repeat(all_ny, K, axis=0)
    batch_start_x = jnp.tile(start_xs, (n_positions, 1))  # (n_positions * K, n_target)
    batch_start_y = jnp.tile(start_ys, (n_positions, 1))

    n_total = n_positions * K
    print(f"Total batch: {n_positions} positions × {K} starts = {n_total} solves")

    # Pre-compute mid for SGD settings
    from dataclasses import replace as _dc_replace
    from pixwake.optim.sgd import _compute_mid_bisection
    if sgd_settings.mid is None:
        computed_mid = _compute_mid_bisection(
            learning_rate=sgd_settings.learning_rate,
            gamma_min=sgd_settings.gamma_min_factor,
            max_iter=sgd_settings.max_iter,
            lower=sgd_settings.bisect_lower,
            upper=sgd_settings.bisect_upper,
        )
        sgd_settings = _dc_replace(sgd_settings, mid=computed_mid)

    n_target = init_x.shape[0]

    def solve_one(sx, sy, nx_local, ny_local):
        def objective(x, y):
            x_all = jnp.concatenate([x, nx_local])
            y_all = jnp.concatenate([y, ny_local])
            result = sim(x_all, y_all, ws_amb=ws, wd_amb=wd, ti_amb=ti_amb)
            power = result.power()[:, :n_target]
            return -jnp.sum(power * weights[:, None]) * 8760 / 1e6

        opt_x, opt_y = topfarm_sgd_solve(
            objective, sx, sy, boundary, D * 4, sgd_settings)
        aep = -objective(opt_x, opt_y)
        return aep

    # Chunked vmap
    CHUNK = 50
    print(f"Running vmap in chunks of {CHUNK}...")
    t0 = time.time()
    all_cons_aeps = np.zeros(n_total)
    for start in range(0, n_total, CHUNK):
        end = min(start + CHUNK, n_total)
        chunk_aeps = jax.vmap(solve_one)(
            batch_start_x[start:end], batch_start_y[start:end],
            batch_nx[start:end], batch_ny[start:end])
        all_cons_aeps[start:end] = np.array(chunk_aeps)
        elapsed = time.time() - t0
        rate = elapsed / end
        remaining = rate * (n_total - end)
        pct_done = 100 * end / n_total
        print(f"  Chunk {start:>6}-{end:>6}  "
              f"({pct_done:>5.1f}%)  "
              f"elapsed={elapsed/60:.1f}min  eta={remaining/60:.1f}min")

    # Reshape: (n_positions, K)
    cons_aeps = all_cons_aeps.reshape(n_positions, K)

    # For each position, compute regret with pooling
    results_grid = np.full((len(distances_D), args.n_bearings), np.nan)
    liberal_aep_present_grid = np.full_like(results_grid, np.nan)
    conservative_aep_grid = np.full_like(results_grid, np.nan)
    all_evals = []

    print(f"\n{'Position':>10} {'bearing':>8} {'dist':>6} {'regret':>10} {'pct':>8}")
    for pi, (di, bi, bearing, dist_m) in enumerate(pos_info):
        nx_p = all_nx[pi]
        ny_p = all_ny[pi]
        lib_aep_present = float(compute_aep(
            sim, liberal_x, liberal_y, ws, wd, weights, ti_amb,
            neighbor_x=nx_p, neighbor_y=ny_p))
        best_cons = float(cons_aeps[pi].max())
        # Pool
        best_cons = max(best_cons, lib_aep_present)
        regret = best_cons - lib_aep_present
        results_grid[di, bi] = regret
        liberal_aep_present_grid[di, bi] = lib_aep_present
        conservative_aep_grid[di, bi] = best_cons
        pct = 100 * regret / liberal_aep
        all_evals.append({
            "bearing_deg": bearing,
            "distance_D": distances_D[di],
            "distance_m": dist_m,
            "regret_gwh": float(regret),
            "regret_pct": float(pct),
            "liberal_aep_present_gwh": lib_aep_present,
            "conservative_aep_gwh": best_cons,
        })
    t_total = time.time() - t0
    print(f"\nTotal sweep time: {t_total/60:.1f} min")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Save results
    results_data = {
        "n_target": args.n_target,
        "ref_rows": args.ref_rows,
        "ref_cols": args.ref_cols,
        "ref_spacing_D": args.ref_spacing_D,
        "n_ref_turbines": n_ref,
        "liberal_aep_gwh": float(liberal_aep),
        "bearings_deg": bearings.tolist(),
        "distances_D": distances_D,
        "regret_grid_gwh": results_grid.tolist(),
        "regret_grid_pct": (100 * results_grid / liberal_aep).tolist(),
        "evaluations": all_evals,
        "elapsed_s": elapsed,
        "config": {
            "wind_rose": args.wind_rose,
            "wind_dir": args.wind_dir,
            "wind_speed": args.wind_speed,
            "n_bins": args.n_bins,
            "deficit": args.deficit,
            "ti": args.ti,
            "n_inner_starts": args.n_inner_starts,
            "inner_max_iter": args.inner_max_iter,
            "ed_a": args.ed_a if args.wind_rose in ("elliptical", "mixture") else None,
            "ed_f": args.ed_f if args.wind_rose in ("elliptical", "mixture") else None,
        },
    }

    json_path = output_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"Results saved: {json_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Liberal AEP: {liberal_aep:.2f} GWh")
    print(f"Max regret: {np.nanmax(results_grid):.2f} GWh "
          f"({100*np.nanmax(results_grid)/liberal_aep:.3f}%)")
    idx = np.unravel_index(np.nanargmax(results_grid), results_grid.shape)
    print(f"  at bearing={bearings[idx[1]]:.0f} deg, distance={distances_D[idx[0]]:.0f}D")
    print(f"Min regret: {np.nanmin(results_grid):.2f} GWh")
    idx_min = np.unravel_index(np.nanargmin(results_grid), results_grid.shape)
    print(f"  at bearing={bearings[idx_min[1]]:.0f} deg, distance={distances_D[idx_min[0]]:.0f}D")


if __name__ == "__main__":
    main()

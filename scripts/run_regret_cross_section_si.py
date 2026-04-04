"""Regret cross-section with SELF-INTERESTED neighbor.

Like run_regret_cross_section.py, but the reference neighbor farm is first
optimized to maximize its OWN AEP (given the target at its liberal position),
before the target farm is re-optimized. This tests whether the "ambush effect"
persists when the neighbor acts rationally rather than adversarially.

Usage:
    pixi run python scripts/run_regret_cross_section_si.py --wind-rose elliptical --ed-a 0.9 --ed-f 1.0
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


def compute_neighbor_aep(sim, target_x, target_y, neighbor_x, neighbor_y,
                         ws_amb, wd_amb, weights, ti_amb=None):
    """Compute AEP for NEIGHBOR turbines only, with target farm present."""
    n_target = target_x.shape[0]
    x_all = jnp.concatenate([target_x, neighbor_x])
    y_all = jnp.concatenate([target_y, neighbor_y])
    result = sim(x_all, y_all, ws_amb=ws_amb, wd_amb=wd_amb, ti_amb=ti_amb)
    power = result.power()[:, n_target:]  # only neighbor turbines
    return jnp.sum(power * weights[:, None]) * 8760 / 1e6


def create_neighbor_boundary(ref_x, ref_y, margin=500.0):
    """Create a rectangular boundary around the reference farm."""
    x_min = float(jnp.min(ref_x)) - margin
    x_max = float(jnp.max(ref_x)) + margin
    y_min = float(jnp.min(ref_y)) - margin
    y_max = float(jnp.max(ref_y)) + margin
    return jnp.array([[x_min, y_min], [x_max, y_min],
                      [x_max, y_max], [x_min, y_max]])


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

    parser.add_argument("--output-dir", type=str, default="analysis/cross_section_si")
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

    print("\nComputing liberal layout (no neighbors)...")
    def liberal_objective(x, y):
        return -compute_aep(sim, x, y, ws, wd, weights, ti_amb)

    liberal_x, liberal_y = topfarm_sgd_solve(
        liberal_objective, init_x, init_y, boundary, D * 4, sgd_settings)
    liberal_aep = compute_aep(sim, liberal_x, liberal_y, ws, wd, weights, ti_amb)
    print(f"Liberal AEP: {liberal_aep:.2f} GWh")

    # Build reference farm
    ref_x_local, ref_y_local = build_reference_farm(
        args.ref_rows, args.ref_cols, args.ref_spacing_D)
    n_ref = len(ref_x_local)
    print(f"Reference farm: {n_ref} turbines")

    # Sweep
    results_grid = np.full((len(distances_D), args.n_bearings), np.nan)
    liberal_aep_present_grid = np.full_like(results_grid, np.nan)
    conservative_aep_grid = np.full_like(results_grid, np.nan)
    all_evals = []

    total = len(distances_D) * args.n_bearings
    count = 0
    t0 = time.time()

    for di, dist_m in enumerate(distances_m):
        for bi, bearing in enumerate(bearings):
            count += 1
            nx_init, ny_init = place_reference_farm(ref_x_local, ref_y_local, bearing, dist_m)

            # Step 0: Optimize neighbor farm for its OWN AEP
            # (with target farm at liberal positions, fixed)
            neighbor_boundary = create_neighbor_boundary(nx_init, ny_init)

            def neighbor_objective(n_x, n_y):
                return -compute_neighbor_aep(
                    sim, liberal_x, liberal_y, n_x, n_y,
                    ws, wd, weights, ti_amb)

            nx, ny = topfarm_sgd_solve(
                neighbor_objective, nx_init, ny_init,
                neighbor_boundary, D * 4, sgd_settings)

            # Liberal AEP with this self-interested neighbor present
            lib_aep_present = compute_aep(
                sim, liberal_x, liberal_y, ws, wd, weights, ti_amb,
                neighbor_x=nx, neighbor_y=ny)

            # Conservative: re-optimize target with this neighbor
            # K multistart: grid init + liberal layout + random starts
            best_cons_aep = -np.inf
            for k in range(args.n_inner_starts):
                if k == 0:
                    start_x, start_y = init_x, init_y
                elif k == 1:
                    start_x, start_y = liberal_x, liberal_y
                else:
                    key = jax.random.PRNGKey(hash((bearing, dist_m, k)) % (2**31))
                    # Random inside polygon
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
                    start_x, start_y = jnp.array(pts[:, 0]), jnp.array(pts[:, 1])

                def cons_objective(x, y):
                    return -compute_aep(sim, x, y, ws, wd, weights, ti_amb,
                                        neighbor_x=nx, neighbor_y=ny)

                cx, cy = topfarm_sgd_solve(
                    cons_objective, start_x, start_y, boundary, D * 4, sgd_settings)
                cons_aep = compute_aep(
                    sim, cx, cy, ws, wd, weights, ti_amb,
                    neighbor_x=nx, neighbor_y=ny)
                if cons_aep > best_cons_aep:
                    best_cons_aep = cons_aep

            regret = best_cons_aep - lib_aep_present
            results_grid[di, bi] = regret
            liberal_aep_present_grid[di, bi] = lib_aep_present
            conservative_aep_grid[di, bi] = best_cons_aep

            elapsed = time.time() - t0
            rate = elapsed / count
            remaining = rate * (total - count)
            print(f"  [{count}/{total}] bearing={bearing:.0f} dist={distances_D[di]:.0f}D "
                  f"regret={regret:.2f} GWh ({100*regret/liberal_aep:.3f}%) "
                  f"[{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining]")

            all_evals.append({
                "bearing_deg": float(bearing),
                "distance_D": float(distances_D[di]),
                "distance_m": float(dist_m),
                "regret_gwh": float(regret),
                "regret_pct": float(100 * regret / liberal_aep),
                "liberal_aep_present_gwh": float(lib_aep_present),
                "conservative_aep_gwh": float(best_cons_aep),
            })

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

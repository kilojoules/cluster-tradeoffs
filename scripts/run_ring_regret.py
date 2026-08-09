"""Ring regret: n identical-copy farms evenly spaced around the target.

Tests how design regret scales with the NUMBER of neighboring farms, isolating
the N effect from greedy placement path-dependence.  For each n in --n-farms,
n copies of the liberal layout are placed at bearings
    base_bearing + i * 360/n,  i = 0..n-1,
each at the specified boundary-gap distance.  If the neighbor copies would
overlap each other (inter-neighbor boundary gap < 2D), all centroid offsets are
scaled up by a common multiplier until every pairwise neighbor gap is >= 2D;
the realized target gaps are recorded.

Per n, records: pooled regret, AEP loss of the liberal layout, per-start
conservative AEPs (for subsampling convergence analysis), and realized gaps.

Usage:
    pixi run python scripts/run_ring_regret.py \
        --n-farms 1,2,3,4,5,6 --distance-D 2 --base-bearing 270 \
        --wind-rose elliptical --ed-a 0.9 --ed-f 1.0 --wind-dir 270 \
        --deficit turbopark --ti 0.06 --schedule funwake_iter192 \
        --n-inner-starts 500 --k-liberal 500 --chunk-size 10 \
        --output-dir analysis/ring_regret_tp/a0.9_f1.0_d2
"""

import run_regret_cross_section as cs  # sets x64 BEFORE pixwake import

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from pixwake.deficit import BastankhahGaussianDeficit
from pixwake.deficit.gaussian import TurboGaussianDeficit
from pixwake.optim.sgd import SGDSettings, topfarm_sgd_solve
from pixwake import WakeSimulation

from schedules import funwake_iter192
from scheduled_sgd import scheduled_sgd_solve

print = cs.print
D = cs.D
boundary_np = cs.boundary_np
boundary = cs.boundary
_polygon_path = cs._polygon_path

MIN_NEIGHBOR_GAP_M = 2 * D


def offset_for_gap_poly(target_poly, nbr_poly, bearing_deg, gap_m,
                        lo=0.0, hi=None, iters=60):
    """Centroid offset along `bearing_deg` so the boundary gap between
    `target_poly` and the shifted `nbr_poly` equals `gap_m`.

    Generalizes cs.centroid_offset_for_gap to a neighbor polygon that differs
    from the target (used for scaled split-neighbor rings).
    """
    th = np.radians(bearing_deg)
    direction = np.array([np.sin(th), np.cos(th)])  # bearing: 0=N, cw
    if hi is None:
        span = np.ptp(target_poly, axis=0).max() + np.ptp(nbr_poly, axis=0).max()
        hi = span + gap_m * 3 + 1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        g = cs.compute_boundary_gap(target_poly, nbr_poly + mid * direction)
        if g < gap_m:
            lo = mid
        else:
            hi = mid
    off = 0.5 * (lo + hi)
    return off, direction, cs.compute_boundary_gap(target_poly, nbr_poly + off * direction)


def ring_offsets_poly(n, base_bearing, gap_m, nbr_poly, min_neighbor_gap_m):
    """Ring offsets for n copies of an arbitrary neighbor polygon."""
    bearings = [(base_bearing + i * 360.0 / n) % 360.0 for i in range(n)]
    base = [offset_for_gap_poly(boundary_np, nbr_poly, b, gap_m)[:2] for b in bearings]
    mult = 1.0
    for _ in range(200):
        shifted = [nbr_poly + mult * off * dirn for off, dirn in base]
        ok = all(cs.compute_boundary_gap(shifted[i], shifted[j]) >= min_neighbor_gap_m
                 for i in range(n) for j in range(i + 1, n))
        if ok:
            break
        mult *= 1.02
    else:
        raise RuntimeError(f"split ring n={n}: could not separate neighbors")
    gaps = [cs.compute_boundary_gap(boundary_np, nbr_poly + mult * off * dirn)
            for off, dirn in base]
    return bearings, [mult * o for o, _ in base], [d for _, d in base], gaps, mult


def ring_offsets(n, base_bearing, gap_m, min_neighbor_gap_m=None):
    """Centroid offsets for n evenly-spaced copies at nominal boundary gap.

    Returns (offsets, directions, realized_target_gaps, multiplier).
    Offsets are scaled by a common multiplier so every pairwise
    neighbor-neighbor boundary gap is >= min_neighbor_gap_m.
    """
    if min_neighbor_gap_m is None:
        min_neighbor_gap_m = MIN_NEIGHBOR_GAP_M
    bearings = [(base_bearing + i * 360.0 / n) % 360.0 for i in range(n)]
    base = []
    for b in bearings:
        offset, direction, _ = cs.centroid_offset_for_gap(boundary_np, b, gap_m)
        base.append((offset, np.asarray(direction)))

    def feasible(m):
        shifted = [boundary_np + m * off * dirn for off, dirn in base]
        return all(cs.compute_boundary_gap(shifted[i], shifted[j]) >= min_neighbor_gap_m
                   for i in range(n) for j in range(i + 1, n))

    if feasible(1.0):
        mult = 1.0
    else:
        # bracket, then bisect for the *minimal* feasible expansion — a coarse
        # multiplicative search overshoots and makes the realized gap depend on
        # step size rather than geometry.
        lo, hi = 1.0, 2.0
        while not feasible(hi):
            lo, hi = hi, hi * 2.0
            if hi > 1e4:
                raise RuntimeError(f"ring n={n}: could not separate neighbors")
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if feasible(mid):
                hi = mid
            else:
                lo = mid
        mult = hi

    target_gaps = [cs.compute_boundary_gap(boundary_np, boundary_np + mult * off * dirn)
                   for off, dirn in base]
    offsets = [mult * off for off, _ in base]
    directions = [dirn for _, dirn in base]
    return bearings, offsets, directions, target_gaps, mult


def main():
    parser = argparse.ArgumentParser(
        description="Ring regret: n identical-copy neighbors evenly spaced",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-target", type=int, default=50)
    parser.add_argument("--n-farms", type=str, default="1,2,3,4",
                        help="Comma-separated ring sizes n")
    parser.add_argument("--distance-D", type=float, default=2.0,
                        help="Nominal boundary-gap distance (D)")
    parser.add_argument("--base-bearing", type=float, default=270.0)
    parser.add_argument("--inner-lr", type=float, default=50.0)
    parser.add_argument("--inner-max-iter", type=int, default=5000)
    parser.add_argument("--n-inner-starts", type=int, default=500)
    parser.add_argument("--k-liberal", type=int, default=500)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--wind-rose", type=str, default="dei",
                        choices=["dei", "elliptical"])
    parser.add_argument("--wind-dir", type=float, default=270.0)
    parser.add_argument("--wind-speed", type=float, default=9.0)
    parser.add_argument("--n-bins", type=int, default=24)
    parser.add_argument("--ed-a", type=float, default=0.9)
    parser.add_argument("--ed-f", type=float, default=1.0)
    parser.add_argument("--deficit", type=str, default="bastankhah",
                        choices=["bastankhah", "turbopark"])
    parser.add_argument("--ti", type=float, default=0.06)
    parser.add_argument("--schedule", type=str, default="funwake_iter192",
                        choices=["sgd_baseline", "funwake_iter192"])
    parser.add_argument("--min-neighbor-gap-D", type=float, default=2.0,
                        help="Minimum boundary gap between neighbor copies. "
                             "Lower values let dense rings stay at the nominal "
                             "target gap, separating encirclement from distance.")
    parser.add_argument("--split-neighbors", action="store_true",
                        help="Hold TOTAL neighbor capacity and area fixed: each of "
                             "the n neighbors is the target polygon scaled by "
                             "1/sqrt(n) containing ~n_target/n turbines. Isolates "
                             "angular spread from amount-of-wake and distance.")
    parser.add_argument("--save-layouts", action="store_true",
                        help="Store the best conservative layout per ring "
                             "(for displacement-field plots)")
    parser.add_argument("--output-dir", type=str, default="analysis/ring_regret")
    args = parser.parse_args()
    min_nbr_gap_m = args.min_neighbor_gap_D * D

    n_target = args.n_target
    n_list = [int(x) for x in args.n_farms.split(",")]
    gap_m = args.distance_D * D

    turbine = cs.create_dei_turbine()
    if args.wind_rose == "dei":
        wd, ws, weights = cs.load_wind_data()
    else:
        from edrose import EllipticalWindRose
        wr = EllipticalWindRose(a=args.ed_a, f=args.ed_f,
                                theta_prev=args.wind_dir, n_sectors=args.n_bins)
        wd = jnp.array(wr.wind_directions)
        weights = jnp.array(wr.sector_frequencies)
        ws = jnp.full_like(wd, args.wind_speed)

    from pixwake.superposition import SquaredSum
    sup = SquaredSum()
    if args.deficit == "bastankhah":
        deficit = BastankhahGaussianDeficit(k=0.04, superposition=sup)
    else:
        deficit = TurboGaussianDeficit(A=0.04, superposition=sup)
    sim = WakeSimulation(turbine, deficit)
    ti_amb = args.ti if args.deficit == "turbopark" else None

    if args.schedule == "funwake_iter192":
        _apply = funwake_iter192(lr_init=args.inner_lr)

        def solve_layout(obj_fn, sx, sy):
            return scheduled_sgd_solve(obj_fn, sx, sy, boundary, D * 4,
                                       _apply, args.inner_max_iter,
                                       lr_init=args.inner_lr)
        print(f"Using FunWake iter_192 schedule (lr_init={args.inner_lr})")
    else:
        sgd_settings = SGDSettings(
            learning_rate=args.inner_lr, max_iter=args.inner_max_iter,
            additional_constant_lr_iterations=args.inner_max_iter, tol=1e-6)
        from dataclasses import replace as _dc_replace
        from pixwake.optim.sgd import _compute_mid_bisection
        sgd_settings = _dc_replace(sgd_settings, mid=_compute_mid_bisection(
            learning_rate=sgd_settings.learning_rate,
            gamma_min=sgd_settings.gamma_min_factor,
            max_iter=sgd_settings.max_iter,
            lower=sgd_settings.bisect_lower,
            upper=sgd_settings.bisect_upper))

        def solve_layout(obj_fn, sx, sy):
            return topfarm_sgd_solve(obj_fn, sx, sy, boundary, D * 4, sgd_settings)
        print("Using pixwake baseline SGD schedule")

    # ---- liberal layout (same seed scheme as cross-section runner) ----
    init_x, init_y = cs.generate_target_grid(boundary_np, n_target, spacing=4 * D)

    def liberal_objective(x, y):
        return -cs.compute_aep(sim, x, y, ws, wd, weights, ti_amb)

    print(f"Computing liberal layout (K={args.k_liberal})...")
    lib_best_aep = -np.inf
    liberal_x, liberal_y = init_x, init_y
    for k in range(args.k_liberal):
        if k == 0:
            sx, sy = init_x, init_y
        else:
            key = jax.random.PRNGKey(k * 7919 + 42)
            pts = []
            while len(pts) < n_target:
                rx = jax.random.uniform(key, (n_target * 3,),
                                        minval=boundary_np[:, 0].min(),
                                        maxval=boundary_np[:, 0].max())
                key, _ = jax.random.split(key)
                ry = jax.random.uniform(key, (n_target * 3,),
                                        minval=boundary_np[:, 1].min(),
                                        maxval=boundary_np[:, 1].max())
                key, _ = jax.random.split(key)
                cands = np.column_stack([np.array(rx), np.array(ry)])
                inside = _polygon_path.contains_points(cands)
                pts.extend(cands[inside].tolist())
            pts = np.array(pts[:n_target])
            sx, sy = jnp.array(pts[:, 0]), jnp.array(pts[:, 1])
        lx, ly = solve_layout(liberal_objective, sx, sy)
        lib_aep = float(-liberal_objective(lx, ly))
        if lib_aep > lib_best_aep:
            lib_best_aep = lib_aep
            liberal_x, liberal_y = lx, ly
        if (k + 1) % 100 == 0 or k == 0:
            print(f"  Start {k+1}/{args.k_liberal}: best AEP = {lib_best_aep:.2f} GWh")
    liberal_aep = lib_best_aep
    print(f"Liberal AEP: {liberal_aep:.2f} GWh")

    ref_x = np.array(liberal_x)
    ref_y = np.array(liberal_y)

    # ---- start layouts shared across rings ----
    K = args.n_inner_starts
    start_xs_list = [init_x, liberal_x]
    start_ys_list = [init_y, liberal_y]
    for k in range(2, K):
        key = jax.random.PRNGKey(k * 7919 + 42)
        pts = []
        while len(pts) < n_target:
            rx = jax.random.uniform(key, (n_target * 3,),
                                    minval=boundary_np[:, 0].min(),
                                    maxval=boundary_np[:, 0].max())
            key, _ = jax.random.split(key)
            ry = jax.random.uniform(key, (n_target * 3,),
                                    minval=boundary_np[:, 1].min(),
                                    maxval=boundary_np[:, 1].max())
            key, _ = jax.random.split(key)
            cands = np.column_stack([np.array(rx), np.array(ry)])
            inside = _polygon_path.contains_points(cands)
            pts.extend(cands[inside].tolist())
        pts = np.array(pts[:n_target])
        start_xs_list.append(jnp.array(pts[:, 0]))
        start_ys_list.append(jnp.array(pts[:, 1]))
    start_xs = jnp.stack(start_xs_list)
    start_ys = jnp.stack(start_ys_list)

    # ---- ring sweep ----
    ring_results = []
    t0 = time.time()
    for n in n_list:
        print(f"\n{'=' * 60}\nRING n={n}\n{'=' * 60}")
        if args.split_neighbors:
            scale = 1.0 / np.sqrt(n)
            nbr_poly = boundary_np * scale
            n_per = max(int(round(n_target / n)), 4)
            gx, gy = cs.generate_target_grid(nbr_poly, n_per, spacing=4 * D * scale)
            nbr_x_local, nbr_y_local = np.array(gx), np.array(gy)
            bearings, offsets, directions, target_gaps, mult = ring_offsets_poly(
                n, args.base_bearing, gap_m, nbr_poly, min_nbr_gap_m)
            print(f"  split mode: {n} x {len(nbr_x_local)} turbines "
                  f"(total {n*len(nbr_x_local)}), polygon scale {scale:.3f}")
        else:
            nbr_x_local, nbr_y_local = ref_x, ref_y
            bearings, offsets, directions, target_gaps, mult = ring_offsets(
                n, args.base_bearing, gap_m, min_nbr_gap_m)
        print(f"  bearings: {[f'{b:.0f}' for b in bearings]}")
        print(f"  separation multiplier: {mult:.3f}  "
              f"realized target gaps (D): {[f'{g/D:.1f}' for g in target_gaps]}")

        nx = np.concatenate([nbr_x_local + off * dirn[0]
                             for off, dirn in zip(offsets, directions)])
        ny = np.concatenate([nbr_y_local + off * dirn[1]
                             for off, dirn in zip(offsets, directions)])
        nx_j, ny_j = jnp.array(nx), jnp.array(ny)

        lib_present = float(cs.compute_aep(
            sim, liberal_x, liberal_y, ws, wd, weights, ti_amb,
            neighbor_x=nx_j, neighbor_y=ny_j))
        aep_loss = liberal_aep - lib_present

        def solve_one(sx, sy):
            def objective(x, y):
                x_all = jnp.concatenate([x, nx_j])
                y_all = jnp.concatenate([y, ny_j])
                result = sim(x_all, y_all, ws_amb=ws, wd_amb=wd, ti_amb=ti_amb)
                power = result.power()[:, :n_target]
                return -jnp.sum(power * weights[:, None]) * 8760 / 1e6
            opt_x, opt_y = solve_layout(objective, sx, sy)
            return -objective(opt_x, opt_y), opt_x, opt_y

        cons_aeps = np.zeros(K)
        cons_xs = np.zeros((K, n_target)) if args.save_layouts else None
        cons_ys = np.zeros((K, n_target)) if args.save_layouts else None
        CHUNK = args.chunk_size
        tn = time.time()
        for start in range(0, K, CHUNK):
            end = min(start + CHUNK, K)
            aeps_c, xs_c, ys_c = jax.vmap(solve_one)(
                start_xs[start:end], start_ys[start:end])
            cons_aeps[start:end] = np.array(aeps_c)
            if args.save_layouts:
                cons_xs[start:end] = np.array(xs_c)
                cons_ys[start:end] = np.array(ys_c)
            el = time.time() - tn
            print(f"  starts {start:>4}-{end:>4} ({100*end/K:5.1f}%)  "
                  f"elapsed={el/60:.1f}min  eta={el/end*(K-end)/60:.1f}min")

        best_cons = max(float(cons_aeps.max()), lib_present)
        regret = best_cons - lib_present
        print(f"  n={n}: loss={100*aep_loss/liberal_aep:.3f}%  "
              f"regret={100*regret/liberal_aep:.3f}%  "
              f"regret/loss={regret/aep_loss if aep_loss > 0 else float('nan'):.2f}")

        entry = {
            "n_farms": n,
            "bearings_deg": bearings,
            "nominal_gap_D": args.distance_D,
            "realized_target_gaps_D": [g / D for g in target_gaps],
            "separation_multiplier": mult,
            "n_turbines_per_neighbor": int(len(nbr_x_local)),
            "n_neighbor_turbines_total": int(len(nx)),
            "liberal_aep_present_gwh": lib_present,
            "aep_loss_gwh": float(aep_loss),
            "aep_loss_pct": float(100 * aep_loss / liberal_aep),
            "conservative_aep_gwh": best_cons,
            "regret_gwh": float(regret),
            "regret_pct": float(100 * regret / liberal_aep),
            "regret_over_loss": float(regret / aep_loss) if aep_loss > 0 else None,
            "all_cons_aeps_gwh": cons_aeps.tolist(),
        }
        if args.save_layouts:
            kbest = int(np.argmax(cons_aeps))
            entry["best_start_index"] = kbest
            entry["cons_x"] = cons_xs[kbest].tolist()
            entry["cons_y"] = cons_ys[kbest].tolist()
            entry["neighbor_x"] = nx.tolist()
            entry["neighbor_y"] = ny.tolist()
        ring_results.append(entry)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.json", "w") as f:
        json.dump({
            "n_target": n_target,
            "methodology": "ring_split_fixed_capacity" if args.split_neighbors
                           else "ring_identical_copy",
            "base_bearing_deg": args.base_bearing,
            "nominal_gap_D": args.distance_D,
            "min_neighbor_gap_D": args.min_neighbor_gap_D,
            "liberal_aep_gwh": float(liberal_aep),
            "liberal_x": ref_x.tolist(),
            "liberal_y": ref_y.tolist(),
            "k_liberal": args.k_liberal,
            "k_inner": K,
            "rings": ring_results,
            "elapsed_s": time.time() - t0,
            "config": {
                "wind_rose": args.wind_rose,
                "wind_dir": args.wind_dir,
                "wind_speed": args.wind_speed,
                "n_bins": args.n_bins,
                "deficit": args.deficit,
                "ti": args.ti,
                "inner_max_iter": args.inner_max_iter,
                "inner_lr": args.inner_lr,
                "schedule": args.schedule,
                "ed_a": args.ed_a if args.wind_rose == "elliptical" else None,
                "ed_f": args.ed_f if args.wind_rose == "elliptical" else None,
            },
        }, f, indent=2)
    print(f"\nResults saved: {out / 'results.json'}")
    print(f"Total time: {(time.time() - t0)/3600:.1f} h")


if __name__ == "__main__":
    main()

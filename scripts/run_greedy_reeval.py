"""Re-evaluate a stored greedy neighbour configuration under a different optimizer.

The greedy sweep reports regret for the configuration it ends up placing.  That
number is the output of an optimizer, so it can be depressed by an inner solver
that is too weak - and a depressed "adversarial ceiling" reads as safety.

This holds the neighbour configuration FIXED (loaded from an existing greedy
result) and varies only the inner optimizer, which isolates the optimizer's
effect on the reported number.  It costs ~2K layout solves instead of the
~300K needed to re-run the greedy search itself.

Usage:
    pixi run python scripts/run_greedy_reeval.py \
        --source analysis/edrose_sweep_k500/a0.9_f1.0 \
        --deficit bastankhah --wind-rose elliptical --ed-a 0.9 --ed-f 1.0 \
        --k 500 --inner-max-iter 5000 \
        --output-dir analysis/greedy_reeval/a0.9_f1.0_bast
"""

import run_regret_cross_section as cs  # sets x64 before pixwake import

import argparse
import json
import time
from dataclasses import replace as dc_replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from pixwake import WakeSimulation
from pixwake.deficit import BastankhahGaussianDeficit
from pixwake.deficit.gaussian import TurboGaussianDeficit
from pixwake.optim.sgd import SGDSettings, topfarm_sgd_solve, _compute_mid_bisection

from schedules import funwake_iter192
from scheduled_sgd import scheduled_sgd_solve

print = cs.print
D = cs.D
boundary_np = cs.boundary_np
boundary = cs.boundary
_polygon_path = cs._polygon_path


def random_start(seed, n):
    key = jax.random.PRNGKey(seed * 7919 + 42)
    pts = []
    while len(pts) < n:
        rx = jax.random.uniform(key, (n * 3,), minval=boundary_np[:, 0].min(),
                                maxval=boundary_np[:, 0].max())
        key, _ = jax.random.split(key)
        ry = jax.random.uniform(key, (n * 3,), minval=boundary_np[:, 1].min(),
                                maxval=boundary_np[:, 1].max())
        key, _ = jax.random.split(key)
        c = np.column_stack([np.array(rx), np.array(ry)])
        pts.extend(c[_polygon_path.contains_points(c)].tolist())
    p = np.array(pts[:n])
    return jnp.array(p[:, 0]), jnp.array(p[:, 1])


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--source", type=str, required=True,
                    help="Directory holding the greedy results.json to re-evaluate")
    ap.add_argument("--n-target", type=int, default=50)
    ap.add_argument("--k", type=int, default=500)
    ap.add_argument("--inner-max-iter", type=int, default=5000)
    ap.add_argument("--inner-lr", type=float, default=50.0)
    ap.add_argument("--chunk-size", type=int, default=25)
    ap.add_argument("--wind-rose", type=str, default="elliptical",
                    choices=["dei", "elliptical"])
    ap.add_argument("--wind-dir", type=float, default=270.0)
    ap.add_argument("--wind-speed", type=float, default=9.0)
    ap.add_argument("--n-bins", type=int, default=24)
    ap.add_argument("--ed-a", type=float, default=0.9)
    ap.add_argument("--ed-f", type=float, default=1.0)
    ap.add_argument("--deficit", type=str, default="bastankhah",
                    choices=["bastankhah", "turbopark"])
    ap.add_argument("--ti", type=float, default=0.06)
    ap.add_argument("--output-dir", type=str, default="analysis/greedy_reeval")
    args = ap.parse_args()

    src = json.load(open(Path(args.source) / "results.json"))
    nx = jnp.array(src["neighbor_x"])
    ny = jnp.array(src["neighbor_y"])
    print(f"Loaded {len(src['neighbor_x'])} neighbour turbines from {args.source}")
    print(f"  original reported regret: {src['regret_gwh']:.2f} GWh "
          f"({100*src['regret_gwh']/src['liberal_aep_gwh']:.3f}% AEP), "
          f"schedule={src['config'].get('schedule', 'sgd_baseline')}, "
          f"K={src['config'].get('n_inner_starts')}")

    n_target = args.n_target
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
    deficit = (BastankhahGaussianDeficit(k=0.04, superposition=SquaredSum())
               if args.deficit == "bastankhah"
               else TurboGaussianDeficit(A=0.04, superposition=SquaredSum()))
    sim = WakeSimulation(turbine, deficit)
    ti_amb = args.ti if args.deficit == "turbopark" else None

    # solvers
    sgd = SGDSettings(learning_rate=args.inner_lr, max_iter=args.inner_max_iter,
                      additional_constant_lr_iterations=args.inner_max_iter, tol=1e-6)
    sgd = dc_replace(sgd, mid=_compute_mid_bisection(
        learning_rate=sgd.learning_rate, gamma_min=sgd.gamma_min_factor,
        max_iter=sgd.max_iter, lower=sgd.bisect_lower, upper=sgd.bisect_upper))
    _fw = funwake_iter192(lr_init=args.inner_lr)

    SOLVERS = {
        "sgd_baseline": lambda obj, sx, sy: topfarm_sgd_solve(
            obj, sx, sy, boundary, D * 4, sgd),
        "funwake_iter192": lambda obj, sx, sy: scheduled_sgd_solve(
            obj, sx, sy, boundary, D * 4, _fw, args.inner_max_iter,
            lr_init=args.inner_lr),
    }

    init_x, init_y = cs.generate_target_grid(boundary_np, n_target, spacing=4 * D)
    starts = [(init_x, init_y)] + [random_start(k, n_target) for k in range(1, args.k)]
    start_xs = jnp.stack([s[0] for s in starts])
    start_ys = jnp.stack([s[1] for s in starts])

    def liberal_obj(x, y):
        return -cs.compute_aep(sim, x, y, ws, wd, weights, ti_amb)

    out = {}
    t0 = time.time()
    for name, solve in SOLVERS.items():
        print(f"\n=== {name}")
        # liberal, K starts
        best_lib, lx, ly = -np.inf, init_x, init_y
        for k in range(args.k):
            ox, oy = solve(liberal_obj, start_xs[k], start_ys[k])
            v = float(-liberal_obj(ox, oy))
            if v > best_lib:
                best_lib, lx, ly = v, ox, oy
            if (k + 1) % 100 == 0:
                print(f"  liberal {k+1}/{args.k}: best {best_lib:.2f} GWh")
        lib_present = float(cs.compute_aep(sim, lx, ly, ws, wd, weights, ti_amb,
                                           neighbor_x=nx, neighbor_y=ny))

        # conservative, K starts, neighbours fixed
        def cons_one(sx, sy):
            def obj(x, y):
                xa = jnp.concatenate([x, nx])
                ya = jnp.concatenate([y, ny])
                r = sim(xa, ya, ws_amb=ws, wd_amb=wd, ti_amb=ti_amb)
                p = r.power()[:, :n_target]
                return -jnp.sum(p * weights[:, None]) * 8760 / 1e6
            ox, oy = solve(obj, sx, sy)
            return -obj(ox, oy)

        cons = np.zeros(args.k)
        CH = args.chunk_size
        tc = time.time()
        for s in range(0, args.k, CH):
            e = min(s + CH, args.k)
            cons[s:e] = np.array(jax.vmap(cons_one)(start_xs[s:e], start_ys[s:e]))
            el = time.time() - tc
            print(f"  cons {s:>4}-{e:>4} ({100*e/args.k:5.1f}%) "
                  f"elapsed={el/60:.1f}min eta={el/e*(args.k-e)/60:.1f}min")
        best_cons = max(float(cons.max()), lib_present)
        regret = best_cons - lib_present
        out[name] = {
            "liberal_aep_gwh": best_lib,
            "liberal_aep_present_gwh": lib_present,
            "conservative_aep_gwh": best_cons,
            "regret_gwh": float(regret),
            "regret_pct": float(100 * regret / best_lib),
            "all_cons_aeps_gwh": cons.tolist(),
        }
        print(f"  -> regret {regret:.2f} GWh ({100*regret/best_lib:.3f}% AEP)")

    o = Path(args.output_dir)
    o.mkdir(parents=True, exist_ok=True)
    with open(o / "results.json", "w") as f:
        json.dump({
            "source": args.source,
            "source_regret_gwh": src["regret_gwh"],
            "source_regret_pct": 100 * src["regret_gwh"] / src["liberal_aep_gwh"],
            "source_schedule": src["config"].get("schedule", "sgd_baseline"),
            "n_neighbor_turbines": len(src["neighbor_x"]),
            "neighbor_x": src["neighbor_x"], "neighbor_y": src["neighbor_y"],
            "k": args.k, "inner_max_iter": args.inner_max_iter,
            "deficit": args.deficit, "wind_rose": args.wind_rose,
            "ed_a": args.ed_a if args.wind_rose == "elliptical" else None,
            "ed_f": args.ed_f if args.wind_rose == "elliptical" else None,
            "by_schedule": out,
            "elapsed_s": time.time() - t0,
        }, f, indent=2)
    print(f"\nSaved: {o / 'results.json'}")
    b, fw = out["sgd_baseline"], out["funwake_iter192"]
    print(f"baseline {b['regret_pct']:.3f}%   funwake {fw['regret_pct']:.3f}%   "
          f"ratio {fw['regret_pct']/b['regret_pct'] if b['regret_pct'] else float('nan'):.2f}x")


if __name__ == "__main__":
    main()

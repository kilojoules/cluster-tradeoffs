"""Bayesian-optimization tuning of the baseline (TopFarm-style) SGD optimizer.

Question: does the FunWake schedule matter, or would a well-tuned baseline
decay schedule reach the same layout quality at equal iteration budget?

Setup: single liberal-layout optimization case (fixed wind rose, fixed start),
total iteration budget fixed at --total-iter for every candidate.  Optuna TPE
tunes {lr_init, gamma_min_factor, frac_const, beta1, beta2} of SGDSettings.
References computed in the same run, same start, same budget:
  - production baseline settings (lr=50, gamma_min default, const phase = total)
  - FunWake iter_192 (lr_init=50)

Validation: best-3 BO configs + both references re-evaluated on --n-holdout
held-out random starts (max and mean AEP reported).

Usage (production):
    pixi run python scripts/run_sgd_bo_tuning.py \
        --wind-rose elliptical --ed-a 0.9 --ed-f 1.0 --wind-dir 270 \
        --n-trials 80 --total-iter 5000 --n-holdout 8 \
        --output-dir analysis/sgd_bo_tuning/a0.9_f1.0
"""

import run_regret_cross_section as cs  # sets x64 BEFORE pixwake import

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


def random_start(seed, n_target):
    key = jax.random.PRNGKey(seed * 7919 + 42)
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
    return jnp.array(pts[:, 0]), jnp.array(pts[:, 1])


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--n-target", type=int, default=50)
    parser.add_argument("--total-iter", type=int, default=5000)
    parser.add_argument("--n-trials", type=int, default=80)
    parser.add_argument("--n-holdout", type=int, default=8)
    parser.add_argument("--wind-rose", type=str, default="elliptical",
                        choices=["dei", "elliptical"])
    parser.add_argument("--wind-dir", type=float, default=270.0)
    parser.add_argument("--wind-speed", type=float, default=9.0)
    parser.add_argument("--n-bins", type=int, default=24)
    parser.add_argument("--ed-a", type=float, default=0.9)
    parser.add_argument("--ed-f", type=float, default=1.0)
    parser.add_argument("--deficit", type=str, default="bastankhah",
                        choices=["bastankhah", "turbopark"])
    parser.add_argument("--ti", type=float, default=0.06)
    parser.add_argument("--output-dir", type=str,
                        default="analysis/sgd_bo_tuning")
    args = parser.parse_args()

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    n_target = args.n_target
    T = args.total_iter

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
    if args.deficit == "bastankhah":
        deficit = BastankhahGaussianDeficit(k=0.04, superposition=SquaredSum())
    else:
        deficit = TurboGaussianDeficit(A=0.04, superposition=SquaredSum())
    sim = WakeSimulation(turbine, deficit)
    ti_amb = args.ti if args.deficit == "turbopark" else None

    def objective_fn(x, y):
        return -cs.compute_aep(sim, x, y, ws, wd, weights, ti_amb)

    # Fixed single tuning start: deterministic grid init
    init_x, init_y = cs.generate_target_grid(boundary_np, n_target, spacing=4 * D)

    def solve_baseline(params, sx, sy):
        """params: dict(lr, gamma_min, frac_const, beta1, beta2)."""
        n_const = int(round(params["frac_const"] * T))
        n_decay = max(T - n_const, 1)
        settings = SGDSettings(
            learning_rate=params["lr"],
            gamma_min_factor=params["gamma_min"],
            beta1=params["beta1"],
            beta2=params["beta2"],
            max_iter=n_decay,
            additional_constant_lr_iterations=n_const,
            tol=1e-6,
        )
        mid = _compute_mid_bisection(
            learning_rate=settings.learning_rate,
            gamma_min=settings.gamma_min_factor,
            max_iter=settings.max_iter,
            lower=settings.bisect_lower,
            upper=settings.bisect_upper,
        )
        settings = dc_replace(settings, mid=mid)
        ox, oy = topfarm_sgd_solve(objective_fn, sx, sy, boundary, D * 4, settings)
        return float(-objective_fn(ox, oy))

    _fw_apply = funwake_iter192(lr_init=50.0)

    def solve_funwake(sx, sy):
        ox, oy = scheduled_sgd_solve(objective_fn, sx, sy, boundary, D * 4,
                                     _fw_apply, T, lr_init=50.0)
        return float(-objective_fn(ox, oy))

    DEFAULT = {"lr": 50.0, "gamma_min": 0.01, "frac_const": 0.5,
               "beta1": 0.1, "beta2": 0.2}

    print("Reference evaluations (tuning start)...")
    t0 = time.time()
    aep_default = solve_baseline(DEFAULT, init_x, init_y)
    print(f"  baseline default: {aep_default:.3f} GWh  ({time.time()-t0:.0f}s)")
    aep_funwake = solve_funwake(init_x, init_y)
    print(f"  funwake iter192:  {aep_funwake:.3f} GWh")

    trials_log = []

    def objective(trial):
        params = {
            "lr": trial.suggest_float("lr", 2.0, 800.0, log=True),
            "gamma_min": trial.suggest_float("gamma_min", 1e-4, 1.0, log=True),
            "frac_const": trial.suggest_float("frac_const", 0.0, 0.8),
            "beta1": trial.suggest_float("beta1", 0.02, 0.95),
            "beta2": trial.suggest_float("beta2", 0.05, 0.99),
        }
        t = time.time()
        aep = solve_baseline(params, init_x, init_y)
        trials_log.append({"params": params, "aep_gwh": aep,
                           "elapsed_s": time.time() - t})
        print(f"  trial {trial.number:>3}: AEP={aep:.3f}  "
              f"(best so far incl refs: "
              f"{max([aep_funwake, aep_default] + [r['aep_gwh'] for r in trials_log]):.3f})  "
              f"lr={params['lr']:.1f} gmin={params['gamma_min']:.4f} "
              f"fc={params['frac_const']:.2f} b1={params['beta1']:.2f} "
              f"b2={params['beta2']:.2f}")
        return aep

    print(f"\nOptuna TPE, {args.n_trials} trials...")
    sampler = optuna.samplers.TPESampler(seed=0)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.enqueue_trial(DEFAULT)                       # seed with production default
    study.enqueue_trial({**DEFAULT, "lr": 200.0})      # and the FunWake-peak-like LR
    study.optimize(objective, n_trials=args.n_trials)

    best3 = sorted(trials_log, key=lambda r: -r["aep_gwh"])[:3]
    print(f"\nBest tuned baseline: {best3[0]['aep_gwh']:.3f} GWh")
    print(f"FunWake reference:   {aep_funwake:.3f} GWh")
    print(f"Default baseline:    {aep_default:.3f} GWh")

    # ---- hold-out validation ----
    print(f"\nHold-out validation on {args.n_holdout} random starts...")
    holdout = [random_start(100 + i, n_target) for i in range(args.n_holdout)]
    val = {}
    for name, solver in (
        [("funwake", solve_funwake),
         ("default_baseline", lambda sx, sy: solve_baseline(DEFAULT, sx, sy))]
        + [(f"bo_rank{i+1}", (lambda p: lambda sx, sy: solve_baseline(p, sx, sy))(b["params"]))
           for i, b in enumerate(best3)]
    ):
        aeps = [solver(sx, sy) for sx, sy in holdout]
        val[name] = {"aeps_gwh": aeps,
                     "mean_gwh": float(np.mean(aeps)),
                     "max_gwh": float(np.max(aeps))}
        print(f"  {name:18s} mean={val[name]['mean_gwh']:.3f}  "
              f"max={val[name]['max_gwh']:.3f}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.json", "w") as f:
        json.dump({
            "case": {"wind_rose": args.wind_rose, "ed_a": args.ed_a,
                     "ed_f": args.ed_f, "wind_dir": args.wind_dir,
                     "deficit": args.deficit, "n_target": n_target,
                     "total_iter": T},
            "tuning_start": "grid_init",
            "aep_default_baseline_gwh": aep_default,
            "aep_funwake_gwh": aep_funwake,
            "trials": trials_log,
            "best3": best3,
            "holdout_validation": val,
            "elapsed_s": time.time() - t0,
        }, f, indent=2)
    print(f"\nSaved: {out / 'results.json'}")


if __name__ == "__main__":
    main()

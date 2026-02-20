"""Plot multistart regret results from saved JSON + screening-only analysis for large K.

Loads completed results from analysis/multistart_regret/results_K*.json,
then runs forward-only screening at the initial neighbor position for large K
to show how best-of-K regret scales without the outer gradient loop.

Usage:
    pixi run python scripts/plot_multistart_results.py
    pixi run python scripts/plot_multistart_results.py --screening-only --max-k=500
"""

import jax
jax.config.update("jax_enable_x64", True)

import argparse
import json
import sys
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(line_buffering=True)


def main():
    parser = argparse.ArgumentParser(
        description="Plot multistart regret results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results-dir", type=str, default="analysis/multistart_regret")
    parser.add_argument("--screening-only", action="store_true",
                        help="Also run screening-only analysis for large K")
    parser.add_argument("--max-k", type=int, default=200,
                        help="Maximum K for screening-only analysis")
    parser.add_argument("--n-target", type=int, default=50)
    parser.add_argument("--n-neighbors", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    # --- Load completed results ---
    completed = {}
    for p in sorted(results_dir.glob("results_K*.json")):
        k = int(p.stem.split("K")[1])
        with open(p) as f:
            r = json.load(f)
        # Only use results with full 20 iterations
        if len(r["regret_history"]) >= 15:
            completed[k] = r
            print(f"Loaded K={k}: {len(r['regret_history'])} iters, "
                  f"best={max(r['regret_history']):.2f} GWh, "
                  f"time={r['total_time']:.0f}s")

    if not completed:
        print("No completed results found!")
        return

    # --- Plot 1: Regret vs Iteration (outer loop) ---
    fig, ax = plt.subplots(figsize=(10, 6))
    k_values = sorted(completed.keys())
    cmap = plt.cm.viridis
    n_lines = len(k_values)

    for idx, K in enumerate(k_values):
        r = completed[K]
        iters = range(len(r["regret_history"]))
        c = cmap(idx / max(n_lines - 1, 1))
        ax.plot(iters, r["regret_history"], "-o", color=c, lw=2, ms=3,
                label=f"K={K} (best={max(r['regret_history']):.1f} GWh, {r['total_time']:.0f}s)")

    ax.set_xlabel("Outer Iteration", fontsize=12)
    ax.set_ylabel("Regret (GWh)", fontsize=12)
    ax.set_title(f"Multistart Regret — Envelope Theorem IFT Bilevel\n"
                 f"{args.n_target} targets, {args.n_neighbors} neighbors, lr=10",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(results_dir / "multistart_regret.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {results_dir / 'multistart_regret.png'}")

    # --- Screening-only analysis for large K ---
    if args.screening_only:
        print(f"\n{'='*60}")
        print(f"SCREENING-ONLY ANALYSIS (forward solves, no gradient loop)")
        print(f"{'='*60}")

        from pixwake import Curve, Turbine, WakeSimulation
        from pixwake.deficit import BastankhahGaussianDeficit
        from pixwake.optim.sgd import SGDSettings, topfarm_sgd_solve

        D = 240.0

        # Reuse setup from plot_multistart_regret.py
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
        boundary_np = raw[hull.vertices]
        boundary = jnp.array(boundary_np)

        # Turbine
        ws_curve = jnp.array([0.,1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.,
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
        turbine = Turbine(
            rotor_diameter=D, hub_height=150.0,
            power_curve=Curve(ws=ws_curve, values=power),
            ct_curve=Curve(ws=ws_curve, values=ct),
        )
        sim = WakeSimulation(turbine, BastankhahGaussianDeficit(k=0.04))

        # Wind data
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
        wd = jnp.array(bin_centers)
        ws = jnp.array(mean_speeds)
        weights = jnp.array(weights_arr)

        N_TARGET = args.n_target
        N_NEIGHBOR = args.n_neighbors
        min_spacing = 4.0 * D

        sgd_settings = SGDSettings(learning_rate=50.0, max_iter=500, tol=1e-10)

        # Liberal layout
        from matplotlib.path import Path as MplPath
        poly_path = MplPath(boundary_np)
        x_min, x_max = boundary_np[:, 0].min(), boundary_np[:, 0].max()
        y_min, y_max = boundary_np[:, 1].min(), boundary_np[:, 1].max()

        def generate_layout(seed):
            rng = np.random.default_rng(seed)
            pts = []
            while len(pts) < N_TARGET:
                cands = rng.uniform([x_min, y_min], [x_max, y_max], size=(N_TARGET * 5, 2))
                pts.extend(cands[poly_path.contains_points(cands)].tolist())
            pts = np.array(pts[:N_TARGET])
            return jnp.array(pts[:, 0]), jnp.array(pts[:, 1])

        def liberal_objective(x, y):
            result = sim(x, y, ws_amb=ws, wd_amb=wd)
            pw = result.power()[:, :N_TARGET]
            return -jnp.sum(pw * weights[:, None]) * 8760 / 1e6

        init_x0, init_y0 = generate_layout(args.seed)
        print("Computing liberal AEP...", flush=True)
        liberal_x, liberal_y = topfarm_sgd_solve(
            liberal_objective, init_x0, init_y0, boundary, min_spacing, sgd_settings,
        )
        liberal_aep = float(-liberal_objective(liberal_x, liberal_y))
        print(f"  Liberal AEP: {liberal_aep:.2f} GWh")

        # Initial neighbors
        cy = boundary_np[:, 1].mean()
        x_offset = boundary_np[:, 0].min() - 3 * D
        rng_nb = np.random.default_rng(args.seed + 100)
        y_spread = (boundary_np[:, 1].max() - boundary_np[:, 1].min()) * 0.5
        nb_x = np.full(N_NEIGHBOR, x_offset) + rng_nb.uniform(-D, 0, N_NEIGHBOR)
        nb_y = cy + rng_nb.uniform(-y_spread, y_spread, N_NEIGHBOR)
        nb_params = jnp.concatenate([jnp.array(nb_x), jnp.array(nb_y)])

        # Run K forward solves sequentially, tracking best-so-far
        max_k = args.max_k
        print(f"\nRunning {max_k} forward screening solves...")

        regrets = []
        best_so_far = []
        times = []
        current_best = -np.inf
        total_t0 = time.time()

        for k in range(max_k):
            ix, iy = generate_layout(args.seed + k)

            def obj_fn(x, y):
                n_nb = nb_params.shape[0] // 2
                nb_x_, nb_y_ = nb_params[:n_nb], nb_params[n_nb:]
                x_all = jnp.concatenate([x, nb_x_])
                y_all = jnp.concatenate([y, nb_y_])
                result = sim(x_all, y_all, ws_amb=ws, wd_amb=wd)
                pw = result.power()[:, :N_TARGET]
                return -jnp.sum(pw * weights[:, None]) * 8760 / 1e6

            t0 = time.time()
            opt_x, opt_y = topfarm_sgd_solve(
                obj_fn, ix, iy, boundary, min_spacing, sgd_settings,
            )
            conservative_aep = float(-obj_fn(opt_x, opt_y))
            # Liberal layout (optimized in isolation) evaluated WITH neighbors
            n_nb = nb_params.shape[0] // 2
            nb_x_, nb_y_ = nb_params[:n_nb], nb_params[n_nb:]
            x_lib_all = jnp.concatenate([liberal_x, nb_x_])
            y_lib_all = jnp.concatenate([liberal_y, nb_y_])
            result_lib = sim(x_lib_all, y_lib_all, ws_amb=ws, wd_amb=wd)
            power_lib = result_lib.power()[:, :N_TARGET]
            liberal_aep_present = float(jnp.sum(power_lib * weights[:, None]) * 8760 / 1e6)
            regret_k = conservative_aep - liberal_aep_present
            dt = time.time() - t0

            regrets.append(regret_k)
            current_best = max(current_best, regret_k)
            best_so_far.append(current_best)
            times.append(dt)

            if k % 10 == 0 or k == max_k - 1:
                elapsed = time.time() - total_t0
                print(f"  k={k:4d}: regret={regret_k:.2f} GWh, "
                      f"best_so_far={current_best:.2f} GWh, "
                      f"solve={dt:.1f}s, elapsed={elapsed:.0f}s")

        total_time = time.time() - total_t0
        print(f"\nTotal screening time: {total_time:.0f}s ({total_time/60:.1f} min)")

        # Save screening results
        screening_results = {
            "regrets": regrets,
            "best_so_far": best_so_far,
            "times": times,
            "total_time": total_time,
            "liberal_aep": liberal_aep,
            "max_k": max_k,
        }
        with open(results_dir / "screening_results.json", "w") as f:
            json.dump(screening_results, f, indent=2)

        # --- Plot 2: Best-of-K regret vs K (screening only) ---
        fig, ax = plt.subplots(figsize=(10, 6))
        ks = range(1, max_k + 1)
        ax.plot(ks, best_so_far, "-", color="C0", lw=2, label="Best-of-K regret")
        ax.scatter(range(1, max_k + 1), regrets, s=8, alpha=0.3, color="C1",
                   label="Individual start regret", zorder=-1)

        # Mark specific K values
        for mark_k in [1, 10, 50, 100, max_k]:
            if mark_k <= max_k:
                ax.axvline(mark_k, color="gray", ls="--", alpha=0.3)
                ax.annotate(f"K={mark_k}\n{best_so_far[mark_k-1]:.1f} GWh",
                            xy=(mark_k, best_so_far[mark_k-1]),
                            fontsize=8, ha="left", va="bottom",
                            xytext=(5, 5), textcoords="offset points")

        ax.set_xlabel("Number of starts K", fontsize=12)
        ax.set_ylabel("Regret (GWh)", fontsize=12)
        ax.set_title(f"Screening-Only: Best-of-K Regret vs K\n"
                     f"{N_TARGET} targets, {N_NEIGHBOR} neighbors, initial neighbor position",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(results_dir / "screening_regret_vs_k.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {results_dir / 'screening_regret_vs_k.png'}")

    print(f"\nAll outputs in {results_dir}/")


if __name__ == "__main__":
    main()

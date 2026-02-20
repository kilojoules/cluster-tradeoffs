"""IFT gradient convergence study: sweep FD epsilons and inner tolerance.

At a fixed neighbor configuration, compares the IFT gradient (from
sgd_solve_implicit's custom_vjp) against the "ground truth" outer
finite-difference gradient for different combinations of:

  - ift_eps_base  (Jacobian FD epsilon)
  - ift_eps_hvp   (Hessian-vector product FD epsilon)
  - inner tol     (SGD convergence tolerance)

Outputs:
  - analysis/ift_convergence/results.json    (all metrics)
  - analysis/ift_convergence/heatmaps.png    (cosine sim & rel error)
  - analysis/ift_convergence/summary.txt     (human-readable table)

Usage:
    pixi run python scripts/run_ift_convergence_study.py
    pixi run python scripts/run_ift_convergence_study.py --n-neighbors=4 --fd-step=1.0
"""

import jax
jax.config.update("jax_enable_x64", True)

import argparse
import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pixwake import Curve, Turbine, WakeSimulation
from pixwake.deficit import BastankhahGaussianDeficit
from pixwake.optim.sgd import (
    SGDSettings, sgd_solve_implicit, topfarm_sgd_solve,
)


# =============================================================================
# DEI setup (shared with run_dei_ift_bilevel.py)
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


def place_initial_neighbors(boundary, n_neighbors, seed=123):
    cy = boundary[:, 1].mean()
    x_offset = boundary[:, 0].min() - 5 * D
    rng = np.random.default_rng(seed)
    y_spread = (boundary[:, 1].max() - boundary[:, 1].min()) * 0.5
    nb_x = np.full(n_neighbors, x_offset) + rng.uniform(-2 * D, 0, n_neighbors)
    nb_y = cy + rng.uniform(-y_spread, y_spread, n_neighbors)
    return jnp.array(nb_x), jnp.array(nb_y)


# =============================================================================
# Metrics
# =============================================================================

def cosine_similarity(a, b):
    """Cosine similarity between two vectors. Returns NaN if either is zero."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-30 or nb < 1e-30:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def relative_error(a, b):
    """Relative error ||a - b|| / ||b||. Returns NaN if b is zero."""
    nb = np.linalg.norm(b)
    if nb < 1e-30:
        return float("nan")
    return float(np.linalg.norm(a - b) / nb)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="IFT gradient convergence study",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-target", type=int, default=10)
    parser.add_argument("--n-neighbors", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inner-lr", type=float, default=50.0)
    parser.add_argument("--inner-max-iter", type=int, default=500)
    parser.add_argument("--fd-step", type=float, default=1.0,
                        help="Outer FD step size (meters) for ground-truth gradient")
    parser.add_argument("--output-dir", type=str, default="analysis/ift_convergence")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    N_TARGET = args.n_target
    N_NEIGHBOR = args.n_neighbors
    min_spacing = 4.0 * D

    # --- Sweep grid ---
    eps_base_values = [1e-5, 1e-4, 1e-3, 1e-2]
    eps_hvp_values = [1e-6, 1e-5, 1e-4, 1e-3]
    tol_values = [1e-6, 1e-8, 1e-10]

    print("=" * 70)
    print("IFT GRADIENT CONVERGENCE STUDY")
    print(f"  {N_TARGET} targets, {N_NEIGHBOR} neighbors")
    print(f"  Inner: lr={args.inner_lr}, max_iter={args.inner_max_iter}")
    print(f"  Outer FD step: {args.fd_step} m")
    print(f"  eps_base sweep: {eps_base_values}")
    print(f"  eps_hvp sweep:  {eps_hvp_values}")
    print(f"  tol sweep:      {tol_values}")
    print("=" * 70)

    # --- Setup ---
    boundary_np = load_target_boundary()
    boundary = jnp.array(boundary_np)
    turbine = create_dei_turbine()
    wd, ws, weights = load_wind_data()
    sim = WakeSimulation(turbine, BastankhahGaussianDeficit(k=0.04))

    init_x, init_y = generate_initial_layout(boundary_np, N_TARGET, seed=args.seed)
    init_nb_x, init_nb_y = place_initial_neighbors(boundary_np, N_NEIGHBOR, seed=args.seed + 1)
    neighbor_params = jnp.concatenate([init_nb_x, init_nb_y])

    n_params = neighbor_params.shape[0]
    print(f"\nNeighbor params: {n_params} values")
    print(f"  x range: [{float(init_nb_x.min()):.0f}, {float(init_nb_x.max()):.0f}]")
    print(f"  y range: [{float(init_nb_y.min()):.0f}, {float(init_nb_y.max()):.0f}]")

    def objective_with_neighbors(x, y, nb_params):
        n_nb = nb_params.shape[0] // 2
        nb_x, nb_y = nb_params[:n_nb], nb_params[n_nb:]
        x_all = jnp.concatenate([x, nb_x])
        y_all = jnp.concatenate([y, nb_y])
        result = sim(x_all, y_all, ws_amb=ws, wd_amb=wd)
        power = result.power()[:, :N_TARGET]
        return -jnp.sum(power * weights[:, None]) * 8760 / 1e6

    def liberal_objective(x, y):
        result = sim(x, y, ws_amb=ws, wd_amb=wd)
        power = result.power()[:, :N_TARGET]
        return -jnp.sum(power * weights[:, None]) * 8760 / 1e6

    # --- Liberal layout (fixed baseline) ---
    base_settings = SGDSettings(
        learning_rate=args.inner_lr,
        max_iter=args.inner_max_iter,
    )

    print("\nComputing liberal layout (no neighbors)...")
    t0 = time.time()
    liberal_x, liberal_y = topfarm_sgd_solve(
        liberal_objective, init_x, init_y, boundary, min_spacing, base_settings,
    )
    liberal_aep = float(-liberal_objective(liberal_x, liberal_y))
    print(f"  Liberal AEP: {liberal_aep:.2f} GWh ({time.time()-t0:.1f}s)")

    # --- Ground truth: outer finite differences ---
    # For each neighbor param, perturb by fd_step, re-solve inner SGD,
    # evaluate regret, compute central-difference gradient.
    fd_step = args.fd_step

    # Use tightest inner tolerance for ground truth
    gt_settings = SGDSettings(
        learning_rate=args.inner_lr,
        max_iter=args.inner_max_iter,
        tol=1e-10,
    )

    def compute_regret_for_fd(nb_params, settings):
        """Compute regret at given neighbor params (forward only, no grad)."""
        def obj_fn(x, y):
            return objective_with_neighbors(x, y, nb_params)
        opt_x, opt_y = topfarm_sgd_solve(
            obj_fn, liberal_x, liberal_y, boundary, min_spacing, settings,
        )
        n_nb = nb_params.shape[0] // 2
        nb_x, nb_y = nb_params[:n_nb], nb_params[n_nb:]
        x_all = jnp.concatenate([opt_x, nb_x])
        y_all = jnp.concatenate([opt_y, nb_y])
        result = sim(x_all, y_all, ws_amb=ws, wd_amb=wd)
        power = result.power()[:, :N_TARGET]
        conservative_aep = float(jnp.sum(power * weights[:, None]) * 8760 / 1e6)
        # Liberal layout evaluated WITH neighbors
        x_lib_all = jnp.concatenate([liberal_x, nb_x])
        y_lib_all = jnp.concatenate([liberal_y, nb_y])
        result_lib = sim(x_lib_all, y_lib_all, ws_amb=ws, wd_amb=wd)
        power_lib = result_lib.power()[:, :N_TARGET]
        liberal_aep_present = float(jnp.sum(power_lib * weights[:, None]) * 8760 / 1e6)
        return conservative_aep - liberal_aep_present

    print(f"\nComputing ground-truth gradient via outer FD (step={fd_step}m)...")
    print(f"  This requires {2 * n_params} inner SGD solves...")
    t0 = time.time()
    gt_regret_center = compute_regret_for_fd(neighbor_params, gt_settings)
    grad_fd = np.zeros(n_params)
    for i in range(n_params):
        e = jnp.zeros(n_params).at[i].set(fd_step)
        r_plus = compute_regret_for_fd(neighbor_params + e, gt_settings)
        r_minus = compute_regret_for_fd(neighbor_params - e, gt_settings)
        grad_fd[i] = (r_plus - r_minus) / (2 * fd_step)
        label = f"nb_x[{i}]" if i < N_NEIGHBOR else f"nb_y[{i-N_NEIGHBOR}]"
        print(f"    param {i} ({label}): FD grad = {grad_fd[i]:.8e}  "
              f"(r+={r_plus:.6f}, r-={r_minus:.6f})")
    gt_time = time.time() - t0
    gt_grad_norm = np.linalg.norm(grad_fd)
    print(f"  Ground truth: regret={gt_regret_center:.6f}, |grad|={gt_grad_norm:.6e}, "
          f"time={gt_time:.1f}s")

    # --- Sweep IFT parameters ---
    def compute_ift_gradient(nb_params, settings):
        """Compute regret and IFT gradient via value_and_grad."""
        def regret_fn(p):
            opt_x, opt_y = sgd_solve_implicit(
                objective_with_neighbors, liberal_x, liberal_y,
                boundary, min_spacing, settings, p,
            )
            n_nb = p.shape[0] // 2
            nb_x, nb_y = p[:n_nb], p[n_nb:]
            x_all = jnp.concatenate([opt_x, nb_x])
            y_all = jnp.concatenate([opt_y, nb_y])
            result = sim(x_all, y_all, ws_amb=ws, wd_amb=wd)
            power = result.power()[:, :N_TARGET]
            conservative_aep = jnp.sum(power * weights[:, None]) * 8760 / 1e6
            # Liberal layout evaluated WITH neighbors
            x_lib_all = jnp.concatenate([liberal_x, nb_x])
            y_lib_all = jnp.concatenate([liberal_y, nb_y])
            result_lib = sim(x_lib_all, y_lib_all, ws_amb=ws, wd_amb=wd)
            power_lib = result_lib.power()[:, :N_TARGET]
            liberal_aep_present = jnp.sum(power_lib * weights[:, None]) * 8760 / 1e6
            return conservative_aep - liberal_aep_present

        regret, grad = jax.value_and_grad(regret_fn)(nb_params)
        return float(regret), np.array(grad)

    results = []
    total_configs = len(eps_base_values) * len(eps_hvp_values) * len(tol_values)
    config_i = 0

    print(f"\nSweeping {total_configs} IFT configurations...")
    print(f"{'#':>3}  {'eps_base':>10}  {'eps_hvp':>10}  {'tol':>10}  "
          f"{'cos_sim':>8}  {'rel_err':>10}  {'|grad|':>12}  {'regret':>10}  {'time':>6}")
    print("-" * 95)

    for tol in tol_values:
        for eps_base in eps_base_values:
            for eps_hvp in eps_hvp_values:
                config_i += 1
                settings = SGDSettings(
                    learning_rate=args.inner_lr,
                    max_iter=args.inner_max_iter,
                    tol=tol,
                    ift_eps_base=eps_base,
                    ift_eps_hvp=eps_hvp,
                )

                t0 = time.time()
                try:
                    regret, grad_ift = compute_ift_gradient(neighbor_params, settings)
                    elapsed = time.time() - t0

                    grad_norm = float(np.linalg.norm(grad_ift))
                    cos_sim = cosine_similarity(grad_ift, grad_fd)
                    rel_err = relative_error(grad_ift, grad_fd)
                    has_nan = bool(np.any(~np.isfinite(grad_ift)))
                except Exception as e:
                    elapsed = time.time() - t0
                    regret = float("nan")
                    grad_ift = np.full(n_params, np.nan)
                    grad_norm = float("nan")
                    cos_sim = float("nan")
                    rel_err = float("nan")
                    has_nan = True
                    print(f"  !! Error: {e}")

                row = {
                    "eps_base": eps_base,
                    "eps_hvp": eps_hvp,
                    "tol": tol,
                    "regret": regret,
                    "grad_norm": grad_norm,
                    "cos_sim": cos_sim,
                    "rel_err": rel_err,
                    "has_nan": has_nan,
                    "time_s": elapsed,
                    "grad_ift": grad_ift.tolist(),
                }
                results.append(row)

                print(f"{config_i:3d}  {eps_base:10.0e}  {eps_hvp:10.0e}  {tol:10.0e}  "
                      f"{cos_sim:8.4f}  {rel_err:10.4f}  {grad_norm:12.6e}  "
                      f"{regret:10.4f}  {elapsed:5.1f}s")

    # --- Save results ---
    output = {
        "n_target": N_TARGET,
        "n_neighbors": N_NEIGHBOR,
        "inner_lr": args.inner_lr,
        "inner_max_iter": args.inner_max_iter,
        "fd_step": fd_step,
        "liberal_aep": liberal_aep,
        "gt_regret": gt_regret_center,
        "gt_grad": grad_fd.tolist(),
        "gt_grad_norm": gt_grad_norm,
        "gt_time_s": gt_time,
        "neighbor_params": neighbor_params.tolist(),
        "sweep": results,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to {output_dir / 'results.json'}")

    # --- Summary table ---
    print("\n" + "=" * 95)
    print("SUMMARY — Best configuration per tolerance level")
    print("=" * 95)
    for tol in tol_values:
        tol_rows = [r for r in results if r["tol"] == tol and not r["has_nan"]]
        if not tol_rows:
            print(f"  tol={tol:.0e}: all NaN")
            continue
        best = max(tol_rows, key=lambda r: r["cos_sim"] if np.isfinite(r["cos_sim"]) else -999)
        print(f"  tol={tol:.0e}:  best cos_sim={best['cos_sim']:.4f}  "
              f"(eps_base={best['eps_base']:.0e}, eps_hvp={best['eps_hvp']:.0e})  "
              f"rel_err={best['rel_err']:.4f}  |grad|={best['grad_norm']:.4e}  "
              f"time={best['time_s']:.1f}s")

    # --- Heatmap plots ---
    fig, axes = plt.subplots(len(tol_values), 2,
                             figsize=(14, 4 * len(tol_values)),
                             squeeze=False)

    for ti, tol in enumerate(tol_values):
        # Build matrices
        n_eb = len(eps_base_values)
        n_eh = len(eps_hvp_values)
        cos_mat = np.full((n_eb, n_eh), np.nan)
        err_mat = np.full((n_eb, n_eh), np.nan)

        for r in results:
            if r["tol"] != tol:
                continue
            i = eps_base_values.index(r["eps_base"])
            j = eps_hvp_values.index(r["eps_hvp"])
            cos_mat[i, j] = r["cos_sim"]
            err_mat[i, j] = r["rel_err"]

        # Cosine similarity heatmap
        ax = axes[ti, 0]
        im = ax.imshow(cos_mat, cmap="RdYlGn", vmin=-1, vmax=1,
                        aspect="auto", origin="lower")
        ax.set_xticks(range(n_eh))
        ax.set_xticklabels([f"{v:.0e}" for v in eps_hvp_values], fontsize=8)
        ax.set_yticks(range(n_eb))
        ax.set_yticklabels([f"{v:.0e}" for v in eps_base_values], fontsize=8)
        ax.set_xlabel("eps_hvp")
        ax.set_ylabel("eps_base")
        ax.set_title(f"Cosine similarity (tol={tol:.0e})")
        for i in range(n_eb):
            for j in range(n_eh):
                val = cos_mat[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=8, color="black" if abs(val) < 0.7 else "white")
        fig.colorbar(im, ax=ax)

        # Relative error heatmap (log scale)
        ax = axes[ti, 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            log_err = np.log10(np.clip(err_mat, 1e-10, None))
        im = ax.imshow(log_err, cmap="RdYlGn_r", vmin=-2, vmax=2,
                        aspect="auto", origin="lower")
        ax.set_xticks(range(n_eh))
        ax.set_xticklabels([f"{v:.0e}" for v in eps_hvp_values], fontsize=8)
        ax.set_yticks(range(n_eb))
        ax.set_yticklabels([f"{v:.0e}" for v in eps_base_values], fontsize=8)
        ax.set_xlabel("eps_hvp")
        ax.set_ylabel("eps_base")
        ax.set_title(f"Relative error (tol={tol:.0e})")
        for i in range(n_eb):
            for j in range(n_eh):
                val = err_mat[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                            fontsize=8, color="black" if val > 0.5 else "white")
        fig.colorbar(im, ax=ax)

    fig.suptitle(f"IFT Gradient Accuracy: {N_TARGET} targets, {N_NEIGHBOR} neighbors\n"
                 f"GT |grad|={gt_grad_norm:.4e}, FD step={fd_step}m",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_dir / "heatmaps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved heatmaps to {output_dir / 'heatmaps.png'}")

    # --- Per-component comparison plot ---
    # Show the best configuration's gradient vs ground truth, component by component
    best_overall = max(
        [r for r in results if not r["has_nan"]],
        key=lambda r: r["cos_sim"] if np.isfinite(r["cos_sim"]) else -999,
    )
    best_grad = np.array(best_overall["grad_ift"])

    fig2, ax = plt.subplots(figsize=(10, 5))
    x_pos = np.arange(n_params)
    width = 0.35
    ax.bar(x_pos - width / 2, grad_fd, width, label="Ground truth (outer FD)", color="steelblue")
    ax.bar(x_pos + width / 2, best_grad, width, label="IFT (best config)", color="coral")
    labels = [f"x{i}" for i in range(N_NEIGHBOR)] + [f"y{i}" for i in range(N_NEIGHBOR)]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("d(regret)/d(param)")
    ax.set_title(f"Best IFT config: eps_base={best_overall['eps_base']:.0e}, "
                 f"eps_hvp={best_overall['eps_hvp']:.0e}, tol={best_overall['tol']:.0e}\n"
                 f"cos_sim={best_overall['cos_sim']:.4f}, rel_err={best_overall['rel_err']:.4f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(output_dir / "grad_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved gradient comparison to {output_dir / 'grad_comparison.png'}")

    # --- Summary text file ---
    with open(output_dir / "summary.txt", "w") as f:
        f.write("IFT Gradient Convergence Study\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Setup: {N_TARGET} targets, {N_NEIGHBOR} neighbors\n")
        f.write(f"Inner SGD: lr={args.inner_lr}, max_iter={args.inner_max_iter}\n")
        f.write(f"Outer FD step: {fd_step} m\n\n")
        f.write(f"Ground truth: regret={gt_regret_center:.6f} GWh, "
                f"|grad|={gt_grad_norm:.6e}\n")
        f.write(f"Ground truth computation: {gt_time:.1f}s "
                f"({2*n_params} inner solves)\n\n")
        f.write(f"{'eps_base':>10}  {'eps_hvp':>10}  {'tol':>10}  "
                f"{'cos_sim':>8}  {'rel_err':>10}  {'|grad|':>12}  {'time':>6}\n")
        f.write("-" * 80 + "\n")
        for r in sorted(results, key=lambda r: -(r["cos_sim"] if np.isfinite(r["cos_sim"]) else -999)):
            f.write(f"{r['eps_base']:10.0e}  {r['eps_hvp']:10.0e}  {r['tol']:10.0e}  "
                    f"{r['cos_sim']:8.4f}  {r['rel_err']:10.4f}  "
                    f"{r['grad_norm']:12.6e}  {r['time_s']:5.1f}s\n")
    print(f"Saved summary to {output_dir / 'summary.txt'}")

    print(f"\nAll outputs in {output_dir}/")


if __name__ == "__main__":
    main()

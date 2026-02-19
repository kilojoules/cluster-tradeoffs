"""IFT bilevel optimization on the DEI case: 10 target turbines, polygon boundary.

Uses implicit differentiation through inner SGD to find adversarial neighbor
placements that maximize design regret for a 10-turbine subset of the Danish
Energy Island.

Produces:
  - analysis/dei_ift_bilevel/dei_ift_bilevel.mp4        (outer-loop animation)
  - analysis/dei_ift_bilevel/inner_sgd_first.mp4        (inner loop, first outer iter)
  - analysis/dei_ift_bilevel/inner_sgd_last.mp4         (inner loop, last outer iter)
  - analysis/dei_ift_bilevel/results.json               (regret + layout data)
  - analysis/dei_ift_bilevel/convergence.png            (regret vs iteration)

Usage:
    pixi run python scripts/run_dei_ift_bilevel.py
    pixi run python scripts/run_dei_ift_bilevel.py --n-outer=100 --lr=50
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
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon as MplPolygon

from pixwake import Curve, Turbine, WakeSimulation
from pixwake.deficit import BastankhahGaussianDeficit
from pixwake.optim.sgd import (
    SGDSettings, sgd_solve_implicit, topfarm_sgd_solve,
    _compute_mid_bisection, _init_sgd_state, _sgd_step,
    _signed_distance_to_edge, boundary_penalty, spacing_penalty,
)


# =============================================================================
# DEI Configuration
# =============================================================================

TARGET_ROTOR_DIAMETER = 240.0  # m
TARGET_HUB_HEIGHT = 150.0  # m
D = TARGET_ROTOR_DIAMETER
SNAPSHOT_EVERY = 5  # inner SGD snapshot frequency


def load_target_boundary():
    """Load DEI target farm boundary — convex hull, CCW order."""
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
    return raw[hull.vertices]  # (5, 2), CCW


def create_dei_turbine():
    """Create 15 MW DEI turbine with exact PyWake power/CT curves."""
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
    """Load DEI wind data → binned 24-sector wind rose."""
    import pandas as pd
    csv_path = Path(__file__).parent.parent / "energy_island_10y_daily_av_wind.csv"
    df = pd.read_csv(csv_path, sep=';')
    wd_ts, ws_ts = df['WD_150'].values, df['WS_150'].values
    n_bins = 24
    bin_edges = np.linspace(0, 360, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    weights = np.zeros(n_bins)
    mean_speeds = np.zeros(n_bins)
    for i in range(n_bins):
        mask = ((wd_ts >= bin_edges[i]) | (wd_ts < bin_edges[0])) if i == n_bins - 1 \
            else ((wd_ts >= bin_edges[i]) & (wd_ts < bin_edges[i + 1]))
        weights[i] = mask.sum()
        mean_speeds[i] = ws_ts[mask].mean() if mask.sum() > 0 else ws_ts.mean()
    weights /= weights.sum()
    return jnp.array(bin_centers), jnp.array(mean_speeds), jnp.array(weights)


def generate_initial_layout(boundary, n_turbines, seed=42):
    """Random layout inside polygon via rejection sampling."""
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
    """Place neighbors upwind (west) of the target boundary."""
    cy = boundary[:, 1].mean()
    x_offset = boundary[:, 0].min() - 5 * D
    rng = np.random.default_rng(seed)
    y_spread = (boundary[:, 1].max() - boundary[:, 1].min()) * 0.5
    nb_x = np.full(n_neighbors, x_offset) + rng.uniform(-2 * D, 0, n_neighbors)
    nb_y = cy + rng.uniform(-y_spread, y_spread, n_neighbors)
    return jnp.array(nb_x), jnp.array(nb_y)


# =============================================================================
# Inner SGD with history capture (Python loop, not while_loop)
# =============================================================================

def sgd_solve_with_history(objective_fn, init_x, init_y, boundary, min_spacing,
                           settings, snapshot_every=SNAPSHOT_EVERY):
    """Run inner SGD in Python loop, capturing snapshots of (x, y, obj, penalty).

    Total iterations = additional_constant_lr_iterations + max_iter, matching
    TopFarm's behavior where constant-LR steps are truly additional.
    """
    if settings.mid is None:
        gamma_min = settings.gamma_min_factor
        computed_mid = _compute_mid_bisection(
            settings.learning_rate, gamma_min, settings.max_iter,
            settings.bisect_lower, settings.bisect_upper,
        )
        settings = SGDSettings(
            learning_rate=settings.learning_rate,
            gamma_min_factor=settings.gamma_min_factor,
            beta1=settings.beta1, beta2=settings.beta2,
            max_iter=settings.max_iter, tol=settings.tol,
            mid=computed_mid,
            bisect_upper=settings.bisect_upper,
            bisect_lower=settings.bisect_lower,
            ks_rho=settings.ks_rho,
            spacing_weight=settings.spacing_weight,
            boundary_weight=settings.boundary_weight,
            additional_constant_lr_iterations=settings.additional_constant_lr_iterations,
        )

    total_iter = settings.max_iter + settings.additional_constant_lr_iterations
    rho = settings.ks_rho
    grad_obj_fn = jax.grad(objective_fn, argnums=(0, 1))

    def constraint_pen(x, y):
        return (settings.boundary_weight * boundary_penalty(x, y, boundary, rho)
                + settings.spacing_weight * spacing_penalty(x, y, min_spacing, rho))

    grad_con_fn = jax.grad(constraint_pen, argnums=(0, 1))

    x, y = init_x, init_y
    grad_obj_x, grad_obj_y = grad_obj_fn(x, y)
    state = _init_sgd_state(x, y, grad_obj_x, grad_obj_y, settings)

    snapshots = []
    prev_x, prev_y = x - 1.0, y - 1.0

    for step in range(total_iter):
        change = float(jnp.max(jnp.abs(x - prev_x)) + jnp.max(jnp.abs(y - prev_y)))
        if step > 0 and change < settings.tol:
            break

        if step % snapshot_every == 0:
            snapshots.append({
                "step": step,
                "x": np.array(x), "y": np.array(y),
                "obj": float(objective_fn(x, y)),
                "penalty": float(constraint_pen(x, y)),
            })

        prev_x, prev_y = x, y
        grad_obj_x, grad_obj_y = grad_obj_fn(x, y)
        grad_con_x, grad_con_y = grad_con_fn(x, y)
        x, y, state = _sgd_step(
            x, y, state, grad_obj_x, grad_obj_y, grad_con_x, grad_con_y, settings
        )

    snapshots.append({
        "step": step, "x": np.array(x), "y": np.array(y),
        "obj": float(objective_fn(x, y)),
        "penalty": float(constraint_pen(x, y)),
    })
    return np.array(x), np.array(y), snapshots


# =============================================================================
# Inner loop animation renderer
# =============================================================================

def render_inner_animation(snapshots, nb_params, n_neighbor, boundary_np,
                           liberal_x_np, liberal_y_np, label, output_path):
    """Render inner SGD animation to MP4."""
    nb_x = np.array(nb_params[:n_neighbor])
    nb_y = np.array(nb_params[n_neighbor:])
    n_frames = len(snapshots)

    cx_c = boundary_np[:, 0].mean()
    cy_c = boundary_np[:, 1].mean()
    def to_km(x, y):
        return (np.asarray(x) - cx_c) / 1000., (np.asarray(y) - cy_c) / 1000.

    bnd_km = np.column_stack(to_km(boundary_np[:, 0], boundary_np[:, 1]))
    lib_km_x, lib_km_y = to_km(liberal_x_np, liberal_y_np)
    nb_km_x, nb_km_y = to_km(nb_x, nb_y)

    all_x = np.concatenate([s["x"] for s in snapshots])
    all_y = np.concatenate([s["y"] for s in snapshots])
    ax_km, ay_km = to_km(all_x, all_y)
    pad = 3.0
    x_lo = min(ax_km.min(), bnd_km[:, 0].min(), nb_km_x.min()) - pad
    x_hi = max(ax_km.max(), bnd_km[:, 0].max(), nb_km_x.max()) + pad
    y_lo = min(ay_km.min(), bnd_km[:, 1].min(), nb_km_y.min()) - pad
    y_hi = max(ay_km.max(), bnd_km[:, 1].max(), nb_km_y.max()) + pad

    obj_vals = [s["obj"] for s in snapshots]
    pen_vals = [s["penalty"] for s in snapshots]
    obj_lo, obj_hi = min(obj_vals) * 1.05, max(obj_vals) * 0.95
    pen_hi = max(max(pen_vals) * 1.1, 1e-6)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ax_layout, ax_obj, ax_pen = axes

    def draw(fi):
        for ax in axes:
            ax.clear()
        s = snapshots[fi]
        tx_km, ty_km = to_km(s["x"], s["y"])

        # Layout
        poly = MplPolygon(bnd_km, closed=True, fill=True,
                          facecolor='lightyellow', edgecolor='black', lw=2)
        ax_layout.add_patch(poly)

        # Trail
        for j in range(0, fi + 1, max(1, fi // 15)):
            sp = snapshots[j]
            skx, sky = to_km(sp["x"], sp["y"])
            alpha = 0.05 + 0.4 * (j / max(fi, 1))
            ax_layout.scatter(skx, sky, c="seagreen", marker=".", s=12, alpha=alpha, zorder=3)

        ax_layout.scatter(tx_km, ty_km, c="seagreen", marker="s", s=80,
                          edgecolors="black", linewidths=0.8, label="Targets", zorder=6)
        ax_layout.scatter(lib_km_x, lib_km_y, c="royalblue", marker="^", s=50,
                          alpha=0.3, label="Liberal (start)", zorder=4)
        ax_layout.scatter(nb_km_x, nb_km_y, c="red", marker="D", s=100,
                          edgecolors="black", linewidths=0.8, label="Neighbors (fixed)", zorder=6)

        ax_layout.set_xlim(x_lo, x_hi)
        ax_layout.set_ylim(y_lo, y_hi)
        ax_layout.set_aspect("equal")
        ax_layout.set_xlabel("x (km)")
        ax_layout.set_ylabel("y (km)")
        ax_layout.set_title("Inner SGD — Layout")
        ax_layout.legend(fontsize=7, loc="lower left")

        # Objective
        ax_obj.plot([sn["step"] for sn in snapshots[:fi+1]],
                    [sn["obj"] for sn in snapshots[:fi+1]],
                    "o-", color="purple", ms=3, lw=1.5)
        ax_obj.set_xlim(-5, snapshots[-1]["step"] + 5)
        ax_obj.set_ylim(obj_lo, obj_hi)
        ax_obj.set_xlabel("SGD step")
        ax_obj.set_ylabel("Objective (neg AEP)")
        ax_obj.set_title(f"Obj = {s['obj']:.4f}")
        ax_obj.grid(True, alpha=0.3)

        # Penalty
        ax_pen.plot([sn["step"] for sn in snapshots[:fi+1]],
                    [sn["penalty"] for sn in snapshots[:fi+1]],
                    "s-", color="orangered", ms=3, lw=1.5)
        ax_pen.set_xlim(-5, snapshots[-1]["step"] + 5)
        ax_pen.set_ylim(-pen_hi * 0.05, pen_hi)
        ax_pen.set_xlabel("SGD step")
        ax_pen.set_ylabel("Constraint penalty")
        ax_pen.set_title(f"Penalty = {s['penalty']:.6f}")
        ax_pen.grid(True, alpha=0.3)

        fig.suptitle(f"{label} — SGD step {s['step']}/{snapshots[-1]['step']}", fontsize=13)
        plt.tight_layout(rect=[0, 0, 1, 0.94])

    anim = FuncAnimation(fig, draw, frames=n_frames, interval=100, repeat=True)
    anim.save(str(output_path), writer="ffmpeg", fps=10, dpi=120)
    plt.close(fig)
    print(f"  Saved → {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="DEI IFT bilevel optimization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Problem setup
    parser.add_argument("--n-target", type=int, default=10, help="Number of target turbines")
    parser.add_argument("--n-neighbors", type=int, default=10, help="Number of neighbor turbines")
    parser.add_argument("--min-spacing-D", type=float, default=4.0, help="Minimum spacing in rotor diameters")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initial layout")
    parser.add_argument("--output-dir", type=str, default="analysis/dei_ift_bilevel")

    # Outer loop SGD settings
    parser.add_argument("--n-outer", type=int, default=100, help="Outer loop iterations")
    parser.add_argument("--lr", type=float, default=200.0, help="Outer initial learning rate")
    parser.add_argument("--outer-gamma-min", type=float, default=0.01, help="Outer final LR as fraction of initial")
    parser.add_argument("--outer-beta1", type=float, default=0.1, help="Outer ADAM beta1")
    parser.add_argument("--outer-beta2", type=float, default=0.2, help="Outer ADAM beta2")
    parser.add_argument("--outer-constant-lr-iters", type=int, default=0,
                        help="Outer iterations at constant LR before decay")

    # Inner loop SGD settings
    parser.add_argument("--inner-lr", type=float, default=50.0, help="Inner SGD initial learning rate")
    parser.add_argument("--inner-max-iter", type=int, default=500, help="Inner SGD max iterations")
    parser.add_argument("--inner-gamma-min", type=float, default=0.01, help="Inner final LR as fraction of initial")
    parser.add_argument("--inner-beta1", type=float, default=0.1, help="Inner ADAM beta1")
    parser.add_argument("--inner-beta2", type=float, default=0.2, help="Inner ADAM beta2")
    parser.add_argument("--inner-tol", type=float, default=1e-6, help="Inner SGD convergence tolerance")
    parser.add_argument("--inner-constant-lr-iters", type=int, default=0,
                        help="Inner iterations at constant LR before decay")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    N_TARGET = args.n_target
    N_NEIGHBOR = args.n_neighbors
    min_spacing = args.min_spacing_D * D

    print("=" * 70)
    print("DEI IFT BILEVEL OPTIMIZATION")
    print(f"  {N_TARGET} target turbines, {N_NEIGHBOR} neighbor turbines")
    print(f"  {args.n_outer} outer iterations, outer lr={args.lr}")
    print(f"  Inner: lr={args.inner_lr}, max_iter={args.inner_max_iter}, "
          f"constant_lr_iters={args.inner_constant_lr_iters}")
    print(f"  Outer: lr={args.lr}, gamma_min={args.outer_gamma_min}, "
          f"constant_lr_iters={args.outer_constant_lr_iters}")
    print("=" * 70)

    # Load data
    boundary_np = load_target_boundary()
    boundary = jnp.array(boundary_np)
    turbine = create_dei_turbine()
    wd, ws, weights = load_wind_data()
    sim = WakeSimulation(turbine, BastankhahGaussianDeficit(k=0.04))

    print(f"\nBoundary: {boundary_np.shape[0]} vertices (convex hull, CCW)")
    print(f"Wind rose: {len(wd)} directions, dominant ~{float(wd[jnp.argmax(weights)]):.0f}°")

    # Initial layouts
    init_x, init_y = generate_initial_layout(boundary_np, N_TARGET, seed=args.seed)
    init_nb_x, init_nb_y = place_initial_neighbors(boundary_np, N_NEIGHBOR, seed=args.seed + 1)

    print(f"Target turbines: {N_TARGET}, min spacing: {min_spacing:.0f} m")
    print(f"Neighbor turbines: {N_NEIGHBOR}")

    sgd_settings = SGDSettings(
        learning_rate=args.inner_lr,
        gamma_min_factor=args.inner_gamma_min,
        beta1=args.inner_beta1,
        beta2=args.inner_beta2,
        max_iter=args.inner_max_iter,
        tol=args.inner_tol,
        additional_constant_lr_iterations=args.inner_constant_lr_iters,
    )

    # Objectives
    def objective_with_neighbors(x, y, neighbor_params):
        n_nb = neighbor_params.shape[0] // 2
        nb_x, nb_y = neighbor_params[:n_nb], neighbor_params[n_nb:]
        x_all = jnp.concatenate([x, nb_x])
        y_all = jnp.concatenate([y, nb_y])
        result = sim(x_all, y_all, ws_amb=ws, wd_amb=wd)
        power = result.power()[:, :N_TARGET]
        return -jnp.sum(power * weights[:, None]) * 8760 / 1e6

    def liberal_objective(x, y):
        result = sim(x, y, ws_amb=ws, wd_amb=wd)
        power = result.power()[:, :N_TARGET]
        return -jnp.sum(power * weights[:, None]) * 8760 / 1e6

    # AEP before/after inner optimization
    init_aep = float(-liberal_objective(init_x, init_y))
    print(f"\nAEP at random initial positions: {init_aep:.2f} GWh")

    print("Computing liberal layout (no neighbors)...")
    t0 = time.time()
    liberal_x, liberal_y = topfarm_sgd_solve(
        liberal_objective, init_x, init_y, boundary, min_spacing, sgd_settings
    )
    liberal_aep = float(-liberal_objective(liberal_x, liberal_y))
    init_disp = float(jnp.sqrt((liberal_x - init_x)**2 + (liberal_y - init_y)**2).mean())
    print(f"Liberal AEP: {liberal_aep:.2f} GWh ({time.time()-t0:.1f}s)")
    print(f"Mean displacement: {init_disp:.0f} m, AEP gain: {liberal_aep - init_aep:.2f} GWh")

    # Regret function — warm-start from liberal layout
    def compute_regret(neighbor_params):
        opt_x, opt_y = sgd_solve_implicit(
            objective_with_neighbors, liberal_x, liberal_y,
            boundary, min_spacing, sgd_settings, neighbor_params,
        )
        n_nb = neighbor_params.shape[0] // 2
        nb_x, nb_y = neighbor_params[:n_nb], neighbor_params[n_nb:]
        x_all = jnp.concatenate([opt_x, nb_x])
        y_all = jnp.concatenate([opt_y, nb_y])
        result = sim(x_all, y_all, ws_amb=ws, wd_amb=wd)
        power = result.power()[:, :N_TARGET]
        conservative_aep = jnp.sum(power * weights[:, None]) * 8760 / 1e6
        return liberal_aep - conservative_aep

    regret_and_grad = jax.value_and_grad(compute_regret)

    # =====================================================================
    # Outer loop constraint penalties (neighbors must satisfy spacing and
    # stay outside target boundary + buffer)
    # =====================================================================
    buffer = 2 * D

    def outer_boundary_penalty(nb_x, nb_y):
        """Penalize neighbors that are inside the target boundary + buffer.

        Uses the same signed-distance approach as boundary_penalty but
        *inverted*: positive penalty when a neighbor is too close to or
        inside the target polygon. We shrink the boundary inward by
        `buffer` and penalize points inside that expanded region.
        """
        n_vertices = boundary.shape[0]
        def edge_distances(i):
            x1, y1 = boundary[i]
            x2, y2 = boundary[(i + 1) % n_vertices]
            return _signed_distance_to_edge(nb_x, nb_y, x1, y1, x2, y2)
        all_distances = jax.vmap(edge_distances)(jnp.arange(n_vertices))
        min_distances = jnp.min(all_distances, axis=0)
        # Penalize if inside boundary + buffer (min_distance > -buffer)
        violations = jnp.maximum(0.0, min_distances + buffer)
        return jnp.sum(violations ** 2)

    def outer_constraint_penalty(neighbor_params):
        nb_x = neighbor_params[:N_NEIGHBOR]
        nb_y = neighbor_params[N_NEIGHBOR:]
        sp = spacing_penalty(nb_x, nb_y, min_spacing, rho=100.0)
        bp = outer_boundary_penalty(nb_x, nb_y)
        return sp + bp

    grad_constraint_fn = jax.grad(outer_constraint_penalty)

    # =====================================================================
    # Outer loop SGD settings (same annealing as inner loop)
    # =====================================================================
    outer_sgd_settings = SGDSettings(
        learning_rate=args.lr,
        gamma_min_factor=args.outer_gamma_min,
        beta1=args.outer_beta1,
        beta2=args.outer_beta2,
        max_iter=args.n_outer,
        tol=0.0,  # run all iterations, don't stop early
        spacing_weight=1.0,
        boundary_weight=1.0,
        additional_constant_lr_iterations=args.outer_constant_lr_iters,
    )
    # Compute mid for LR annealing
    outer_mid = _compute_mid_bisection(
        outer_sgd_settings.learning_rate,
        outer_sgd_settings.gamma_min_factor,
        outer_sgd_settings.max_iter,
        outer_sgd_settings.bisect_lower,
        outer_sgd_settings.bisect_upper,
    )

    # =====================================================================
    # Outer loop (TopFarm-style ADAM with LR annealing + constraint penalty)
    # =====================================================================
    neighbor_params = jnp.concatenate([init_nb_x, init_nb_y])
    first_nb_params = neighbor_params.copy()

    history_regret, history_grad_norm = [], []
    history_nb, history_target = [], []
    best_regret = -np.inf
    best_neighbor_params = neighbor_params

    # Initialize ADAM state using the same machinery as inner loop
    # We use a 1D param vector, so we treat x=params[:N], y=params[N:]
    # and reuse SGDState with (m_x, m_y, v_x, v_y) for the two halves.
    beta1_o, beta2_o = outer_sgd_settings.beta1, outer_sgd_settings.beta2
    m_params = jnp.zeros_like(neighbor_params)
    v_params = jnp.zeros_like(neighbor_params)

    # Total outer iterations = constant-LR + decaying (matching TopFarm)
    total_outer = args.n_outer + args.outer_constant_lr_iters
    print(f"\nRunning {total_outer} outer iterations "
          f"({args.outer_constant_lr_iters} constant + {args.n_outer} decaying, lr={args.lr})...")
    print(f"  LR annealing: {args.lr:.1f} → {args.lr * outer_sgd_settings.gamma_min_factor:.2f}"
          f"  (constant for first {args.outer_constant_lr_iters} iters)")
    print(f"  Constraint penalty: spacing + boundary buffer ({buffer:.0f} m)")
    print(f"{'Iter':>4}  {'Regret':>10}  {'Cons AEP':>10}  {'|grad|':>12}"
          f"  {'LR':>8}  {'Alpha':>10}  {'ConPen':>10}  {'Time':>6}")
    print("-" * 90)

    # First regret eval to calibrate alpha0
    total_t0 = time.time()
    regret_0, grad_0 = regret_and_grad(neighbor_params)
    # Negate regret gradient: we minimize (-regret) + alpha * constraint
    neg_grad_regret = -grad_0
    alpha0 = float(jnp.mean(jnp.abs(neg_grad_regret))) / outer_sgd_settings.learning_rate
    alpha = alpha0
    lr_current = outer_sgd_settings.learning_rate

    for i in range(total_outer):
        iter_t0 = time.time()

        if i == 0:
            regret, grad_regret = regret_0, grad_0
        else:
            regret, grad_regret = regret_and_grad(neighbor_params)

        if not jnp.all(jnp.isfinite(grad_regret)):
            print(f"  !! NaN gradient at iter {i}, stopping")
            break

        grad_con = grad_constraint_fn(neighbor_params)
        con_pen = float(outer_constraint_penalty(neighbor_params))

        grad_norm = float(jnp.linalg.norm(grad_regret))
        regret_val = float(regret)
        cons_aep = liberal_aep - regret_val

        history_regret.append(regret_val)
        history_grad_norm.append(grad_norm)
        history_nb.append((np.array(neighbor_params[:N_NEIGHBOR]).copy(),
                           np.array(neighbor_params[N_NEIGHBOR:]).copy()))

        opt_x, opt_y = sgd_solve_implicit(
            objective_with_neighbors, liberal_x, liberal_y,
            boundary, min_spacing, sgd_settings, neighbor_params,
        )
        history_target.append((np.array(opt_x), np.array(opt_y)))

        if regret_val > best_regret:
            best_regret = regret_val
            best_neighbor_params = neighbor_params

        elapsed = time.time() - iter_t0
        if i % 10 == 0 or i == total_outer - 1:
            print(f"{i:4d}  {regret_val:10.4f}  {cons_aep:10.4f}  {grad_norm:12.6f}"
                  f"  {lr_current:8.3f}  {alpha:10.4f}  {con_pen:10.2f}  {elapsed:5.1f}s")

        # Combined gradient: minimize (-regret) + alpha * constraint
        neg_grad = -grad_regret + alpha * grad_con
        it = i + 1

        # ADAM update (TopFarm-style: beta1=0.1, beta2=0.2)
        m_params = beta1_o * m_params + (1 - beta1_o) * neg_grad
        v_params = beta2_o * v_params + (1 - beta2_o) * neg_grad ** 2
        m_hat = m_params / (1 - beta1_o ** it)
        v_hat = v_params / (1 - beta2_o ** it)

        # Descent step on (-regret + alpha * constraint) = ascent on regret
        eps_adam = 1e-12
        neighbor_params = neighbor_params - lr_current * m_hat / (jnp.sqrt(v_hat) + eps_adam)

        # LR decay: lr *= 1/(1 + mid * t)  (same as inner loop)
        # During constant-LR phase, keep lr and alpha fixed (TopFarm behavior)
        n_const = args.outer_constant_lr_iters
        if it > n_const:
            decay_it = it - n_const
            lr_current = lr_current * 1.0 / (1.0 + outer_mid * decay_it)
            # Alpha update: alpha = alpha0 * lr0 / lr  (constraint weight increases)
            alpha = alpha0 * outer_sgd_settings.learning_rate / lr_current

    last_nb_params = neighbor_params.copy()
    total_elapsed = time.time() - total_t0
    n_frames = len(history_regret)
    print(f"\nDone: {n_frames} iterations in {total_elapsed:.1f}s ({total_elapsed/n_frames:.1f}s/iter)")
    print(f"Best regret: {best_regret:.4f} GWh ({best_regret/liberal_aep*100:.3f}%)")

    # =====================================================================
    # Inner loop animations (first and last outer iteration)
    # =====================================================================
    liberal_x_np = np.array(liberal_x)
    liberal_y_np = np.array(liberal_y)

    def make_inner_obj(nb_params):
        def obj(x, y):
            return objective_with_neighbors(x, y, nb_params)
        return obj

    print("\nCapturing inner SGD at FIRST outer iteration...")
    _, _, snaps_first = sgd_solve_with_history(
        make_inner_obj(first_nb_params), liberal_x, liberal_y,
        boundary, min_spacing, sgd_settings,
    )
    print(f"  {len(snaps_first)} snapshots, {snaps_first[-1]['step']} steps")

    print("Capturing inner SGD at LAST outer iteration...")
    _, _, snaps_last = sgd_solve_with_history(
        make_inner_obj(last_nb_params), liberal_x, liberal_y,
        boundary, min_spacing, sgd_settings,
    )
    print(f"  {len(snaps_last)} snapshots, {snaps_last[-1]['step']} steps")

    print("\nRendering inner loop animations...")
    render_inner_animation(
        snaps_first, first_nb_params, N_NEIGHBOR, boundary_np,
        liberal_x_np, liberal_y_np,
        "First outer iteration (initial neighbors)",
        output_dir / "inner_sgd_first.mp4",
    )
    render_inner_animation(
        snaps_last, last_nb_params, N_NEIGHBOR, boundary_np,
        liberal_x_np, liberal_y_np,
        "Last outer iteration (adversarial neighbors)",
        output_dir / "inner_sgd_last.mp4",
    )

    # =====================================================================
    # Save results
    # =====================================================================
    results = {
        "n_target": N_TARGET, "n_neighbors": N_NEIGHBOR,
        "n_outer": args.n_outer,
        "outer_constant_lr_iters": args.outer_constant_lr_iters,
        "total_outer": total_outer, "lr": args.lr,
        "liberal_aep": liberal_aep,
        "best_regret": float(best_regret),
        "best_regret_pct": float(best_regret / liberal_aep * 100),
        "final_regret": history_regret[-1] if history_regret else 0.0,
        "total_time_s": total_elapsed,
        "history_regret": history_regret,
        "history_grad_norm": history_grad_norm,
        "liberal_x": liberal_x_np.tolist(),
        "liberal_y": np.array(liberal_y).tolist(),
        "best_neighbor_x": np.array(best_neighbor_params[:N_NEIGHBOR]).tolist(),
        "best_neighbor_y": np.array(best_neighbor_params[N_NEIGHBOR:]).tolist(),
        "boundary": boundary_np.tolist(),
    }
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {output_dir / 'results.json'}")

    # =====================================================================
    # Convergence plot
    # =====================================================================
    fig_conv, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.plot(history_regret, 'b-o', ms=2)
    ax1.set_ylabel("Regret (GWh)")
    ax1.set_title(f"DEI IFT Bilevel: {N_TARGET} targets, {N_NEIGHBOR} neighbors")
    ax1.grid(True, alpha=0.3)
    ax2.semilogy(history_grad_norm, 'r-o', ms=2)
    ax2.set_xlabel("Outer Iteration")
    ax2.set_ylabel("|grad|")
    ax2.grid(True, alpha=0.3)
    fig_conv.tight_layout()
    fig_conv.savefig(output_dir / "convergence.png", dpi=150, bbox_inches='tight')
    plt.close(fig_conv)
    print(f"Saved convergence plot")

    # =====================================================================
    # Outer loop MP4 (subsample to ~200 frames max for manageable video)
    # =====================================================================
    max_anim_frames = 200
    if n_frames > max_anim_frames:
        frame_indices = np.round(np.linspace(0, n_frames - 1, max_anim_frames)).astype(int)
    else:
        frame_indices = np.arange(n_frames)
    n_anim_frames = len(frame_indices)
    print(f"\nRendering outer loop MP4 ({n_anim_frames} frames from {n_frames} iterations)...")
    mp4_path = output_dir / "dei_ift_bilevel.mp4"

    cx_center = boundary_np[:, 0].mean()
    cy_center = boundary_np[:, 1].mean()
    def to_km(x, y):
        return (np.asarray(x) - cx_center) / 1000., (np.asarray(y) - cy_center) / 1000.

    bnd_km = np.column_stack(to_km(boundary_np[:, 0], boundary_np[:, 1]))
    lib_km_x, lib_km_y = to_km(liberal_x_np, np.array(liberal_y))

    all_nb_x = np.concatenate([nb[0] for nb in history_nb])
    all_nb_y = np.concatenate([nb[1] for nb in history_nb])
    all_nb_km_x, all_nb_km_y = to_km(all_nb_x, all_nb_y)

    pad = 3.0
    layout_x_lo = min(bnd_km[:, 0].min(), all_nb_km_x.min()) - pad
    layout_x_hi = max(bnd_km[:, 0].max(), all_nb_km_x.max()) + pad
    layout_y_lo = min(bnd_km[:, 1].min(), all_nb_km_y.min()) - pad
    layout_y_hi = max(bnd_km[:, 1].max(), all_nb_km_y.max()) + pad

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    ax_layout, ax_regret = axes[0]
    ax_aep, ax_grad = axes[1]

    regret_lo = min(history_regret) * 0.9
    regret_hi = max(history_regret) * 1.1
    grad_lo = min(history_grad_norm) * 0.5
    grad_hi = max(history_grad_norm) * 2.0
    cons_aeps = [liberal_aep - r for r in history_regret]
    aep_lo = min(cons_aeps) * 0.999
    aep_hi = liberal_aep * 1.001

    # Precompute neighbor trail in km for efficiency
    nb_trail_km = [(to_km(nb[0], nb[1])) for nb in history_nb]

    def draw_frame(fi):
        for row in axes:
            for ax in row:
                ax.clear()

        i = frame_indices[fi]  # map animation frame to iteration index
        nb_x_cur, nb_y_cur = history_nb[i]
        opt_x, opt_y = history_target[i]

        poly_patch = MplPolygon(bnd_km, closed=True, fill=True,
                                facecolor='lightyellow', edgecolor='black', lw=2)
        ax_layout.add_patch(poly_patch)
        ax_layout.scatter(lib_km_x, lib_km_y, c="royalblue", marker="^",
                          s=60, label="Liberal", zorder=5, alpha=0.4)
        opt_km_x, opt_km_y = to_km(opt_x, opt_y)
        ax_layout.scatter(opt_km_x, opt_km_y, c="seagreen", marker="s", s=60,
                          label="Conservative", zorder=5)
        for k in range(N_TARGET):
            dxk = opt_km_x[k] - lib_km_x[k]
            dyk = opt_km_y[k] - lib_km_y[k]
            if np.sqrt(dxk**2 + dyk**2) > 0.01:
                ax_layout.annotate("", xy=(lib_km_x[k]+dxk*3, lib_km_y[k]+dyk*3),
                                   xytext=(lib_km_x[k], lib_km_y[k]),
                                   arrowprops=dict(arrowstyle="-|>", color="seagreen",
                                                   lw=1.5, alpha=0.7), zorder=4)

        # Draw neighbor trail (subsample to ~30 trail points max)
        trail_step = max(1, i // 30)
        for j in range(0, i + 1, trail_step):
            nbxk, nbyk = nb_trail_km[j]
            alpha_val = 0.05 + 0.6 * (j / max(i, 1))
            ax_layout.scatter(nbxk, nbyk, c="red", marker="x", s=12, alpha=alpha_val, zorder=3)
        nbkx, nbky = to_km(nb_x_cur, nb_y_cur)
        ax_layout.scatter(nbkx, nbky, c="red", marker="D", s=80,
                          edgecolors="black", linewidths=0.8, label="Neighbors", zorder=6)

        ax_layout.set_xlim(layout_x_lo, layout_x_hi)
        ax_layout.set_ylim(layout_y_lo, layout_y_hi)
        ax_layout.set_xlabel("x (km)")
        ax_layout.set_ylabel("y (km)")
        ax_layout.set_title(f"Layout (iter {i})")
        ax_layout.legend(loc="upper right", fontsize=7)
        ax_layout.set_aspect("equal")

        ax_regret.plot(range(i+1), history_regret[:i+1], "b-", lw=1.2)
        ax_regret.set_xlim(-0.5, n_frames-0.5)
        ax_regret.set_ylim(regret_lo, regret_hi)
        ax_regret.set_xlabel("Outer Iteration")
        ax_regret.set_ylabel("Regret (GWh)")
        ax_regret.set_title("Design Regret")
        ax_regret.grid(True, alpha=0.3)

        ax_aep.axhline(liberal_aep, color="royalblue", ls="--", alpha=0.6, label="Liberal")
        ax_aep.plot(range(i+1), cons_aeps[:i+1], "g-", lw=1.2, label="Conservative")
        ax_aep.set_xlim(-0.5, n_frames-0.5)
        ax_aep.set_ylim(aep_lo, aep_hi)
        ax_aep.set_xlabel("Outer Iteration")
        ax_aep.set_ylabel("AEP (GWh)")
        ax_aep.set_title("Target Farm AEP")
        ax_aep.legend(fontsize=7)
        ax_aep.grid(True, alpha=0.3)

        ax_grad.semilogy(range(i+1), history_grad_norm[:i+1], "r-", lw=1.2)
        ax_grad.set_xlim(-0.5, n_frames-0.5)
        ax_grad.set_ylim(grad_lo, grad_hi)
        ax_grad.set_xlabel("Outer Iteration")
        ax_grad.set_ylabel("|grad|")
        ax_grad.set_title("Gradient Norm")
        ax_grad.grid(True, alpha=0.3)

        fig.suptitle(
            f"DEI IFT Bilevel: {N_TARGET} targets, {N_NEIGHBOR} neighbors  |  "
            f"Regret = {history_regret[i]:.3f} GWh ({history_regret[i]/liberal_aep*100:.2f}%)",
            fontsize=13, fontweight="bold",
        )

    anim = FuncAnimation(fig, draw_frame, frames=n_anim_frames, interval=200, repeat=True)
    anim.save(str(mp4_path), writer="ffmpeg", fps=5, dpi=120)
    plt.close(fig)
    print(f"Saved outer loop animation to {mp4_path}")

    print(f"\nAll outputs in {output_dir}/")


if __name__ == "__main__":
    main()

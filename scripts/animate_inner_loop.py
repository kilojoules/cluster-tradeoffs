"""Animate the inner SGD optimization at the first and last outer iterations.

Replays the SGD steps in a Python loop (not jax.lax.while_loop) to capture
intermediate positions, then renders two MP4s.

Usage:
    pixi run python scripts/animate_inner_loop.py
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from jax import value_and_grad

from pixwake import WakeSimulation
from pixwake.definitions.v80 import vestas_v80
from pixwake.deficit import NOJDeficit
from pixwake.optim.sgd import (
    SGDSettings,
    _compute_mid_bisection,
    _init_sgd_state,
    _sgd_step,
    boundary_penalty,
    spacing_penalty,
    sgd_solve_implicit,
    topfarm_sgd_solve,
)

# ── Config (same as prototype) ──────────────────────────────────────────
D = vestas_v80.rotor_diameter  # 80m
N_TARGET = 4
N_NEIGHBOR = 2
INNER_MAX_ITER = 500
OUTER_ITERS = 20
LR = 10.0
SNAPSHOT_EVERY = 2  # capture positions every N inner steps

sim = WakeSimulation(vestas_v80, NOJDeficit(k=0.1))
ws = jnp.array([9.0])
wd = jnp.array([270.0])

spacing = 4 * D
boundary_size = spacing + 2 * D
boundary = jnp.array([
    [0.0, 0.0],
    [boundary_size, 0.0],
    [boundary_size, boundary_size],
    [0.0, boundary_size],
])
min_spacing = 2 * D

init_x = jnp.array([0.0, spacing, 0.0, spacing])
init_y = jnp.array([0.0, 0.0, spacing, spacing])

init_nb_x = jnp.array([-5 * D, -6 * D])
init_nb_y = jnp.array([spacing * 0.3, spacing * 0.7])

sgd_settings = SGDSettings(learning_rate=10.0, max_iter=INNER_MAX_ITER, tol=1e-6)


# ── Unrolled inner SGD with history capture ─────────────────────────────

def sgd_solve_with_history(
    objective_fn, init_x, init_y, boundary, min_spacing, settings,
    snapshot_every=SNAPSHOT_EVERY,
):
    """Run SGD in Python loop, capturing snapshots of (x, y, obj, penalty)."""
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
        )

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

    for step in range(settings.max_iter):
        # Check convergence
        change = float(jnp.max(jnp.abs(x - prev_x)) + jnp.max(jnp.abs(y - prev_y)))
        if step > 0 and change < settings.tol:
            break

        if step % snapshot_every == 0:
            obj_val = float(objective_fn(x, y))
            pen_val = float(constraint_pen(x, y))
            snapshots.append({
                "step": step,
                "x": np.array(x),
                "y": np.array(y),
                "obj": obj_val,
                "penalty": pen_val,
            })

        prev_x, prev_y = x, y
        grad_obj_x, grad_obj_y = grad_obj_fn(x, y)
        grad_con_x, grad_con_y = grad_con_fn(x, y)
        x, y, state = _sgd_step(
            x, y, state, grad_obj_x, grad_obj_y, grad_con_x, grad_con_y, settings
        )

    # Final snapshot
    snapshots.append({
        "step": step,
        "x": np.array(x),
        "y": np.array(y),
        "obj": float(objective_fn(x, y)),
        "penalty": float(constraint_pen(x, y)),
    })

    return np.array(x), np.array(y), snapshots


# ── Run outer loop to get first and last neighbor params ────────────────

def objective_with_neighbors(x, y, neighbor_params):
    n_nb = neighbor_params.shape[0] // 2
    nb_x, nb_y = neighbor_params[:n_nb], neighbor_params[n_nb:]
    x_all = jnp.concatenate([x, nb_x])
    y_all = jnp.concatenate([y, nb_y])
    result = sim(x_all, y_all, ws_amb=ws, wd_amb=wd)
    power = result.power()[:, :N_TARGET]
    return -jnp.sum(power) * 8760 / 1e6 / power.shape[0]


def liberal_objective(x, y):
    result = sim(x, y, ws_amb=ws, wd_amb=wd)
    return -result.aep()


print("Computing liberal layout...")
liberal_x, liberal_y = topfarm_sgd_solve(
    liberal_objective, init_x, init_y, boundary, min_spacing, sgd_settings
)
liberal_aep_result = sim(liberal_x, liberal_y, ws_amb=ws, wd_amb=wd)
liberal_aep = float(liberal_aep_result.aep())
print(f"Liberal AEP: {liberal_aep:.2f} GWh")


def compute_regret(neighbor_params):
    opt_x, opt_y = sgd_solve_implicit(
        objective_with_neighbors, init_x, init_y,
        boundary, min_spacing, sgd_settings, neighbor_params,
    )
    n_nb = neighbor_params.shape[0] // 2
    nb_x, nb_y = neighbor_params[:n_nb], neighbor_params[n_nb:]
    x_all = jnp.concatenate([opt_x, nb_x])
    y_all = jnp.concatenate([opt_y, nb_y])
    result = sim(x_all, y_all, ws_amb=ws, wd_amb=wd)
    power = result.power()[:, :N_TARGET]
    conservative_aep = jnp.sum(power) * 8760 / 1e6 / power.shape[0]
    return liberal_aep - conservative_aep


regret_and_grad = value_and_grad(compute_regret)

# Run outer loop
neighbor_params = jnp.concatenate([init_nb_x, init_nb_y])
beta1, beta2, adam_eps = 0.9, 0.999, 1e-8
m_adam = jnp.zeros_like(neighbor_params)
v_adam = jnp.zeros_like(neighbor_params)

first_nb_params = neighbor_params.copy()

print(f"\nRunning {OUTER_ITERS} outer iterations to get final neighbor positions...")
for i in range(OUTER_ITERS):
    regret, grad = regret_and_grad(neighbor_params)
    if not jnp.all(jnp.isfinite(grad)):
        break
    print(f"  outer {i}: regret={float(regret):.3f} GWh")
    t = i + 1
    m_adam = beta1 * m_adam + (1 - beta1) * grad
    v_adam = beta2 * v_adam + (1 - beta2) * grad ** 2
    m_hat = m_adam / (1 - beta1 ** t)
    v_hat = v_adam / (1 - beta2 ** t)
    neighbor_params = neighbor_params + LR * m_hat / (jnp.sqrt(v_hat) + adam_eps)

    # Enforce min_spacing buffer from target boundary
    nb_x_new = neighbor_params[:N_NEIGHBOR]
    nb_y_new = neighbor_params[N_NEIGHBOR:]
    cx = jnp.clip(nb_x_new, 0.0, boundary_size)
    cy = jnp.clip(nb_y_new, 0.0, boundary_size)
    dx = nb_x_new - cx
    dy = nb_y_new - cy
    dist = jnp.sqrt(dx**2 + dy**2)
    scale = jnp.where(dist < 1e-6, 1.0, min_spacing / dist)
    nb_x_new = jnp.where(dist < min_spacing, cx + dx * scale, nb_x_new)
    nb_y_new = jnp.where(dist < min_spacing, cy + dy * scale, nb_y_new)
    neighbor_params = jnp.concatenate([nb_x_new, nb_y_new])

last_nb_params = neighbor_params.copy()


# ── Capture inner SGD histories ─────────────────────────────────────────

def make_inner_objective(nb_params):
    """Create the inner objective for given neighbor positions."""
    def obj(x, y):
        return objective_with_neighbors(x, y, nb_params)
    return obj


print("\nCapturing inner SGD at FIRST outer iteration...")
first_obj = make_inner_objective(first_nb_params)
_, _, snaps_first = sgd_solve_with_history(
    first_obj, init_x, init_y, boundary, min_spacing, sgd_settings,
)
print(f"  {len(snaps_first)} snapshots, {snaps_first[-1]['step']} steps")

print("Capturing inner SGD at LAST outer iteration...")
last_obj = make_inner_objective(last_nb_params)
_, _, snaps_last = sgd_solve_with_history(
    last_obj, init_x, init_y, boundary, min_spacing, sgd_settings,
)
print(f"  {len(snaps_last)} snapshots, {snaps_last[-1]['step']} steps")


# ── Render helper ───────────────────────────────────────────────────────

bnd_closed = np.vstack([np.array(boundary), np.array(boundary)[0]])


def render_inner_animation(snapshots, nb_params, label, output_path):
    """Render inner SGD animation to MP4."""
    nb_x = np.array(nb_params[:N_NEIGHBOR])
    nb_y = np.array(nb_params[N_NEIGHBOR:])
    n_frames = len(snapshots)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    ax_layout, ax_obj, ax_pen = axes

    # Axis limits
    obj_vals = [s["obj"] for s in snapshots]
    pen_vals = [s["penalty"] for s in snapshots]
    obj_lo, obj_hi = min(obj_vals) * 1.05, max(obj_vals) * 0.95  # negative obj
    pen_hi = max(pen_vals) * 1.1

    # Gather all positions to set layout limits
    all_x = np.concatenate([s["x"] for s in snapshots])
    all_y = np.concatenate([s["y"] for s in snapshots])
    x_lo = min(all_x.min(), nb_x.min(), 0) - 2 * D
    x_hi = max(all_x.max(), boundary_size) + 2 * D
    y_lo = min(all_y.min(), nb_y.min(), 0) - 2 * D
    y_hi = max(all_y.max(), boundary_size) + 2 * D

    def draw(frame_idx):
        for ax in axes:
            ax.clear()

        s = snapshots[frame_idx]
        tx, ty = s["x"], s["y"]

        # ── Layout ──────────────────────────────────────────────────
        ax_layout.plot(bnd_closed[:, 0], bnd_closed[:, 1], "k-", lw=2)

        # Turbine trail
        for j in range(0, frame_idx + 1, max(1, frame_idx // 15)):
            sp = snapshots[j]
            alpha = 0.05 + 0.3 * (j / max(frame_idx, 1))
            ax_layout.scatter(sp["x"], sp["y"], c="seagreen", marker=".",
                              s=15, alpha=alpha, zorder=3)

        # Current target positions
        ax_layout.scatter(tx, ty, c="seagreen", marker="s", s=120,
                          edgecolors="black", linewidths=0.8,
                          label="Targets", zorder=6)

        # Starting positions (faded)
        ax_layout.scatter(snapshots[0]["x"], snapshots[0]["y"],
                          c="gray", marker="o", s=60, alpha=0.3,
                          label="Initial", zorder=4)

        # Fixed neighbor positions
        ax_layout.scatter(nb_x, nb_y, c="red", marker="D", s=140,
                          edgecolors="black", linewidths=0.8,
                          label="Neighbors (fixed)", zorder=6)

        # Wind arrow
        arrow_x = boundary_size * 0.85
        arrow_y = boundary_size * 0.92
        ax_layout.annotate("", xy=(arrow_x + D, arrow_y),
                           xytext=(arrow_x - D, arrow_y),
                           arrowprops=dict(arrowstyle="-|>", color="gray", lw=2))
        ax_layout.text(arrow_x, arrow_y + D * 0.5, "wind",
                       ha="center", color="gray", fontsize=9)

        # Spacing circles around each turbine
        for k in range(N_TARGET):
            circle = plt.Circle((tx[k], ty[k]), min_spacing / 2,
                                fill=False, color="seagreen", alpha=0.15,
                                linestyle="--", lw=0.8)
            ax_layout.add_patch(circle)

        ax_layout.set_xlim(x_lo, x_hi)
        ax_layout.set_ylim(y_lo, y_hi)
        ax_layout.set_aspect("equal")
        ax_layout.set_title("Inner SGD — Layout", fontsize=11)
        ax_layout.legend(fontsize=7, loc="lower left")
        ax_layout.grid(True, alpha=0.2)

        # ── Objective ───────────────────────────────────────────────
        ax_obj.plot(
            [sn["step"] for sn in snapshots[: frame_idx + 1]],
            [sn["obj"] for sn in snapshots[: frame_idx + 1]],
            "o-", color="purple", markersize=3, lw=1.5,
        )
        ax_obj.set_xlim(-5, snapshots[-1]["step"] + 5)
        ax_obj.set_ylim(obj_lo, obj_hi)
        ax_obj.set_xlabel("SGD step")
        ax_obj.set_ylabel("Objective (neg AEP)")
        ax_obj.set_title(f"Obj = {s['obj']:.4f}", fontsize=11)
        ax_obj.grid(True, alpha=0.3)

        # ── Penalty ─────────────────────────────────────────────────
        ax_pen.plot(
            [sn["step"] for sn in snapshots[: frame_idx + 1]],
            [sn["penalty"] for sn in snapshots[: frame_idx + 1]],
            "s-", color="orangered", markersize=3, lw=1.5,
        )
        ax_pen.set_xlim(-5, snapshots[-1]["step"] + 5)
        ax_pen.set_ylim(-pen_hi * 0.05, pen_hi)
        ax_pen.set_xlabel("SGD step")
        ax_pen.set_ylabel("Constraint penalty")
        ax_pen.set_title(f"Penalty = {s['penalty']:.4f}", fontsize=11)
        ax_pen.grid(True, alpha=0.3)

        fig.suptitle(
            f"{label} — SGD step {s['step']}/{snapshots[-1]['step']}",
            fontsize=13,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.94])

    anim = FuncAnimation(fig, draw, frames=n_frames, interval=100, repeat=True)
    anim.save(output_path, writer="ffmpeg", fps=10, dpi=150)
    plt.close(fig)
    print(f"Saved → {output_path}")


# ── Render ──────────────────────────────────────────────────────────────
print("\nRendering inner loop animations...")

render_inner_animation(
    snaps_first, first_nb_params,
    "First outer iteration (initial neighbors)",
    "analysis/inner_sgd_first.mp4",
)

render_inner_animation(
    snaps_last, last_nb_params,
    "Last outer iteration (optimized neighbors)",
    "analysis/inner_sgd_last.mp4",
)

print("\nDone!")

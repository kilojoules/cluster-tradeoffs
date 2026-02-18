"""Lightweight animated prototype of IFT bilevel search → MP4.

Small problem (4 V80 turbines, NOJ deficit, 2 neighbors). Runs all
iterations first, then renders an MP4 animation.

Usage:
    pixi run python scripts/prototype_bilevel_animated.py
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
from pixwake.optim.sgd import SGDSettings, sgd_solve_implicit, topfarm_sgd_solve

# ── Config ──────────────────────────────────────────────────────────────
D = vestas_v80.rotor_diameter  # 80m
N_TARGET = 4
N_NEIGHBOR = 2
OUTER_ITERS = 20
INNER_MAX_ITER = 500
LR = 10.0
OUTPUT_PATH = "analysis/prototype_bilevel.mp4"

# ── Setup ───────────────────────────────────────────────────────────────
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

# Neighbors: start further upwind so buffer doesn't immediately bind
init_nb_x = jnp.array([-5 * D, -6 * D])
init_nb_y = jnp.array([spacing * 0.3, spacing * 0.7])

sgd_settings = SGDSettings(learning_rate=10.0, max_iter=INNER_MAX_ITER, tol=1e-6)


# ── Objective functions ─────────────────────────────────────────────────

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


# ── Liberal baseline ────────────────────────────────────────────────────
print("Computing liberal layout (no neighbors)...")
liberal_x, liberal_y = topfarm_sgd_solve(
    liberal_objective, init_x, init_y, boundary, min_spacing, sgd_settings
)
liberal_aep_result = sim(liberal_x, liberal_y, ws_amb=ws, wd_amb=wd)
liberal_aep = float(liberal_aep_result.aep())
print(f"Liberal AEP: {liberal_aep:.2f} GWh")


# ── Regret function ────────────────────────────────────────────────────

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

# ── Run all iterations (collect data) ──────────────────────────────────
neighbor_params = jnp.concatenate([init_nb_x, init_nb_y])
beta1, beta2, adam_eps = 0.9, 0.999, 1e-8
m_adam = jnp.zeros_like(neighbor_params)
v_adam = jnp.zeros_like(neighbor_params)

history_regret = []
history_conservative_aep = []
history_grad_norm = []
history_nb = []
history_target = []

print(f"\nRunning {OUTER_ITERS} outer iterations...")
print(f"{'Iter':>4}  {'Regret':>10}  {'Cons AEP':>10}  {'|grad|':>10}  {'target Δ':>10}")
print("-" * 60)

for i in range(OUTER_ITERS):
    regret, grad = regret_and_grad(neighbor_params)

    if not jnp.all(jnp.isfinite(grad)):
        print(f"  !! NaN gradient at iter {i}, stopping")
        break

    grad_norm = float(jnp.linalg.norm(grad))
    history_regret.append(float(regret))
    history_grad_norm.append(grad_norm)

    nb_x_cur = np.array(neighbor_params[:N_NEIGHBOR])
    nb_y_cur = np.array(neighbor_params[N_NEIGHBOR:])
    history_nb.append((nb_x_cur, nb_y_cur))

    # Get conservative layout for this iteration
    opt_x, opt_y = sgd_solve_implicit(
        objective_with_neighbors, init_x, init_y,
        boundary, min_spacing, sgd_settings, neighbor_params,
    )
    opt_x_np, opt_y_np = np.array(opt_x), np.array(opt_y)
    history_target.append((opt_x_np, opt_y_np))

    # Compute conservative AEP and target displacement from liberal
    cons_aep = liberal_aep - float(regret)
    history_conservative_aep.append(cons_aep)
    target_disp = np.sqrt(
        (opt_x_np - np.array(liberal_x)) ** 2
        + (opt_y_np - np.array(liberal_y)) ** 2
    ).mean()

    print(
        f"{i:4d}  {float(regret):10.4f}  {cons_aep:10.4f}  "
        f"{grad_norm:10.6f}  {target_disp:8.1f} m"
    )

    # ADAM step
    t = i + 1
    m_adam = beta1 * m_adam + (1 - beta1) * grad
    v_adam = beta2 * v_adam + (1 - beta2) * grad ** 2
    m_hat = m_adam / (1 - beta1 ** t)
    v_hat = v_adam / (1 - beta2 ** t)
    neighbor_params = neighbor_params + LR * m_hat / (jnp.sqrt(v_hat) + adam_eps)

    # Enforce 2D buffer: neighbors must stay outside the target boundary
    # by at least min_spacing. For the square [0, bs] x [0, bs], project
    # each neighbor to the nearest point on the exclusion zone border.
    buffer = min_spacing
    nb_x_new = neighbor_params[:N_NEIGHBOR]
    nb_y_new = neighbor_params[N_NEIGHBOR:]
    # Nearest point on boundary box to each neighbor
    cx = jnp.clip(nb_x_new, 0.0, boundary_size)
    cy = jnp.clip(nb_y_new, 0.0, boundary_size)
    dx = nb_x_new - cx
    dy = nb_y_new - cy
    dist = jnp.sqrt(dx**2 + dy**2)
    # Where too close, push outward along (neighbor - nearest_boundary_pt)
    scale = jnp.where(dist < 1e-6, 1.0, buffer / dist)
    nb_x_new = jnp.where(dist < buffer, cx + dx * scale, nb_x_new)
    nb_y_new = jnp.where(dist < buffer, cy + dy * scale, nb_y_new)
    neighbor_params = jnp.concatenate([nb_x_new, nb_y_new])

n_frames = len(history_regret)
print(f"\nDone computing. {n_frames} frames collected.")
if n_frames > 0:
    print(f"Initial regret: {history_regret[0]:.3f} GWh")
    print(f"Final regret:   {history_regret[-1]:.3f} GWh")
    print(f"Best regret:    {max(history_regret):.3f} GWh")

# ── Render MP4 ──────────────────────────────────────────────────────────
print(f"\nRendering {OUTPUT_PATH} ...")

bnd = np.array(boundary)
bnd_closed = np.vstack([bnd, bnd[0]])
liberal_x_np = np.array(liberal_x)
liberal_y_np = np.array(liberal_y)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
ax_layout = axes[0, 0]
ax_regret = axes[0, 1]
ax_aep = axes[1, 0]
ax_grad = axes[1, 1]

# Fixed axis limits
regret_lo = min(history_regret) * 0.9 if history_regret else 0
regret_hi = max(history_regret) * 1.1 if history_regret else 1
grad_lo = min(history_grad_norm) * 0.5 if history_grad_norm else 1e-6
grad_hi = max(history_grad_norm) * 2.0 if history_grad_norm else 1
aep_lo = min(history_conservative_aep) * 0.95 if history_conservative_aep else 0
aep_hi = liberal_aep * 1.05

# Gather all neighbor positions to set axis limits
all_nb_x = np.concatenate([nb[0] for nb in history_nb])
all_nb_y = np.concatenate([nb[1] for nb in history_nb])
layout_x_lo = min(all_nb_x.min(), 0) - 2 * D
layout_x_hi = boundary_size + 2 * D
layout_y_lo = min(all_nb_y.min(), 0) - 2 * D
layout_y_hi = max(all_nb_y.max(), boundary_size) + 2 * D


def draw_frame(frame_idx):
    for row in axes:
        for ax in row:
            ax.clear()

    i = frame_idx
    nb_x_cur, nb_y_cur = history_nb[i]
    opt_x, opt_y = history_target[i]

    # ── Layout ──────────────────────────────────────────────────────
    ax_layout.plot(bnd_closed[:, 0], bnd_closed[:, 1], "k-", lw=2)

    # Buffer exclusion zone (dashed)
    buf = min_spacing
    buf_rect = plt.Rectangle((-buf, -buf), boundary_size + 2 * buf,
                              boundary_size + 2 * buf, fill=False,
                              edgecolor="gray", linestyle="--", lw=1, alpha=0.5)
    ax_layout.add_patch(buf_rect)

    # Liberal positions (fixed reference)
    ax_layout.scatter(liberal_x_np, liberal_y_np, c="royalblue", marker="^",
                      s=90, label="Liberal (fixed)", zorder=5, alpha=0.4)

    # Conservative positions (inner loop result)
    ax_layout.scatter(opt_x, opt_y, c="seagreen", marker="s", s=90,
                      label="Conservative (inner)", zorder=5)

    # Displacement arrows: liberal → conservative (amplified 3x for visibility)
    amp = 3.0
    for k in range(N_TARGET):
        dx = opt_x[k] - liberal_x_np[k]
        dy = opt_y[k] - liberal_y_np[k]
        if np.sqrt(dx**2 + dy**2) > 0.5:  # only show if > 0.5m
            ax_layout.annotate(
                "", xy=(liberal_x_np[k] + dx * amp, liberal_y_np[k] + dy * amp),
                xytext=(liberal_x_np[k], liberal_y_np[k]),
                arrowprops=dict(arrowstyle="-|>", color="seagreen",
                                lw=2, alpha=0.8),
                zorder=4,
            )

    # Neighbor trail (all past positions)
    for j in range(i + 1):
        nbx, nby = history_nb[j]
        alpha = 0.08 + 0.5 * (j / max(i, 1))
        ax_layout.scatter(nbx, nby, c="red", marker="x", s=20,
                          alpha=alpha, zorder=3)
    # Current neighbor position
    ax_layout.scatter(nb_x_cur, nb_y_cur, c="red", marker="D", s=140,
                      edgecolors="black", linewidths=0.8,
                      label="Neighbors (outer)", zorder=6)

    # Outer gradient arrows on neighbors
    if i + 1 < n_frames:
        next_nb_x, next_nb_y = history_nb[i + 1]
        grad_scale = D * 0.5
        for k in range(N_NEIGHBOR):
            dx = next_nb_x[k] - nb_x_cur[k]
            dy = next_nb_y[k] - nb_y_cur[k]
            norm = np.sqrt(dx**2 + dy**2)
            if norm > 1e-6:
                ax_layout.annotate(
                    "", xy=(nb_x_cur[k] + dx / norm * grad_scale,
                            nb_y_cur[k] + dy / norm * grad_scale),
                    xytext=(nb_x_cur[k], nb_y_cur[k]),
                    arrowprops=dict(arrowstyle="-|>", color="red",
                                    lw=1.5, alpha=0.7),
                )

    # Wind arrow
    arrow_x = boundary_size * 0.85
    arrow_y = boundary_size * 0.92
    ax_layout.annotate("", xy=(arrow_x + D, arrow_y),
                       xytext=(arrow_x - D, arrow_y),
                       arrowprops=dict(arrowstyle="-|>", color="gray", lw=2))
    ax_layout.text(arrow_x, arrow_y + D * 0.5, "wind", ha="center",
                   color="gray", fontsize=9)

    # Target displacement annotation
    mean_disp = np.sqrt(
        (opt_x - liberal_x_np) ** 2 + (opt_y - liberal_y_np) ** 2
    ).mean()
    ax_layout.text(
        0.02, 0.97, f"mean target shift: {mean_disp:.1f} m\n(arrows {amp:.0f}x amplified)",
        transform=ax_layout.transAxes, fontsize=8, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    ax_layout.set_xlim(layout_x_lo, layout_x_hi)
    ax_layout.set_ylim(layout_y_lo, layout_y_hi)
    ax_layout.set_aspect("equal")
    ax_layout.set_title("Layout", fontsize=11)
    ax_layout.legend(fontsize=7, loc="lower left")
    ax_layout.grid(True, alpha=0.2)

    # ── Regret ──────────────────────────────────────────────────────
    ax_regret.plot(history_regret[: i + 1], "o-", color="purple",
                   markersize=5, lw=2)
    ax_regret.set_xlim(-0.5, n_frames - 0.5)
    ax_regret.set_ylim(regret_lo, regret_hi)
    ax_regret.set_xlabel("Outer iteration")
    ax_regret.set_ylabel("Regret (GWh)")
    ax_regret.set_title(f"Regret = {history_regret[i]:.3f} GWh", fontsize=11)
    ax_regret.grid(True, alpha=0.3)

    # ── Conservative AEP ────────────────────────────────────────────
    ax_aep.plot(history_conservative_aep[: i + 1], "o-", color="seagreen",
                markersize=5, lw=2, label="Conservative AEP")
    ax_aep.axhline(liberal_aep, color="royalblue", ls="--", lw=1.5,
                   alpha=0.7, label=f"Liberal AEP = {liberal_aep:.1f}")
    ax_aep.set_xlim(-0.5, n_frames - 0.5)
    ax_aep.set_ylim(aep_lo, aep_hi)
    ax_aep.set_xlabel("Outer iteration")
    ax_aep.set_ylabel("AEP (GWh)")
    ax_aep.set_title(
        f"Conservative AEP = {history_conservative_aep[i]:.3f} GWh",
        fontsize=11,
    )
    ax_aep.legend(fontsize=8)
    ax_aep.grid(True, alpha=0.3)

    # ── Gradient norm ───────────────────────────────────────────────
    ax_grad.semilogy(history_grad_norm[: i + 1], "s-", color="darkorange",
                     markersize=5, lw=2)
    ax_grad.set_xlim(-0.5, n_frames - 0.5)
    ax_grad.set_ylim(grad_lo, grad_hi)
    ax_grad.set_xlabel("Outer iteration")
    ax_grad.set_ylabel("|grad|")
    ax_grad.set_title(f"|grad| = {history_grad_norm[i]:.2e}", fontsize=11)
    ax_grad.grid(True, alpha=0.3)

    fig.suptitle(
        f"IFT Bilevel Search — Iteration {i}/{n_frames - 1}",
        fontsize=13,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])


anim = FuncAnimation(fig, draw_frame, frames=n_frames, interval=800, repeat=True)
anim.save(OUTPUT_PATH, writer="ffmpeg", fps=2, dpi=150)
plt.close(fig)
print(f"Saved → {OUTPUT_PATH}")

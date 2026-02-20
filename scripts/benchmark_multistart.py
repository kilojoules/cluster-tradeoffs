"""Benchmark multistart strategies for IFT bilevel optimization.

Implements and compares four multistart approaches on the same benchmark case:
  1. Single-start IFT (baseline)
  2. Stochastic single-start (random init each outer iter)
  3. Envelope theorem (K forward, IFT through winner)
  4. LogSumExp soft selection (gradients through all K starts)

Each strategy runs 20 outer ADAM iterations. We measure wall-clock time,
final regret, and gradient quality.

Usage:
    pixi run python scripts/benchmark_multistart.py
"""

import jax

jax.config.update("jax_enable_x64", True)

import time
from functools import partial
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from jax import value_and_grad

from pixwake import WakeSimulation
from pixwake.definitions.v80 import vestas_v80
from pixwake.deficit import NOJDeficit
from pixwake.optim.sgd import (
    SGDSettings,
    sgd_solve_implicit,
    topfarm_sgd_solve,
)

OUTPUT_DIR = Path("analysis/multistart_benchmark")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Setup ────────────────────────────────────────────────────────────────
D = vestas_v80.rotor_diameter  # 80m
N_TARGET = 4
N_NEIGHBOR = 2

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

nb_x_init = jnp.array([-5 * D, -6 * D])
nb_y_init = jnp.array([spacing * 0.3, spacing * 0.7])
nb_params_init = jnp.concatenate([nb_x_init, nb_y_init])

sgd_settings = SGDSettings(learning_rate=10.0, max_iter=500, tol=1e-6)

OUTER_ITERS = 20
OUTER_LR = 10.0
K_STARTS = 5
LOGSUMEXP_TAU = 10.0  # temperature


# ── Objectives ───────────────────────────────────────────────────────────

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


# ── Liberal baseline ─────────────────────────────────────────────────────
print("Computing liberal layout...")
liberal_x, liberal_y = topfarm_sgd_solve(
    liberal_objective, init_x, init_y, boundary, min_spacing, sgd_settings,
)
liberal_aep = float(sim(liberal_x, liberal_y, ws_amb=ws, wd_amb=wd).aep())
print(f"Liberal AEP: {liberal_aep:.4f} GWh\n")


# ── Random initial layouts for multistarts ───────────────────────────────

def generate_random_inits(key, n_starts):
    """Generate n_starts random initial layouts within the boundary."""
    inits = []
    for i in range(n_starts):
        k1, k2, key = jax.random.split(key, 3)
        # Random positions within the boundary with some margin
        margin = min_spacing
        ix = jax.random.uniform(k1, (N_TARGET,), minval=margin, maxval=boundary_size - margin)
        iy = jax.random.uniform(k2, (N_TARGET,), minval=margin, maxval=boundary_size - margin)
        inits.append((ix, iy))
    return inits


# Pre-generate starts (same for all strategies that use K starts)
rng_key = jax.random.PRNGKey(42)
random_inits = generate_random_inits(rng_key, K_STARTS)
# Start 0 is always the default grid layout
random_inits[0] = (init_x, init_y)


# ── Helper: compute regret from a layout ─────────────────────────────────

def eval_conservative_aep(opt_x, opt_y, nb_params):
    """Evaluate AEP of a layout with neighbors (no gradient)."""
    n_nb = nb_params.shape[0] // 2
    nbx, nby = nb_params[:n_nb], nb_params[n_nb:]
    x_all = jnp.concatenate([opt_x, nbx])
    y_all = jnp.concatenate([opt_y, nby])
    result = sim(x_all, y_all, ws_amb=ws, wd_amb=wd)
    power = result.power()[:, :N_TARGET]
    return jnp.sum(power) * 8760 / 1e6 / power.shape[0]


def eval_liberal_aep_present(nb_params):
    """Evaluate liberal layout (optimized in isolation) WITH neighbors present."""
    n_nb = nb_params.shape[0] // 2
    nbx, nby = nb_params[:n_nb], nb_params[n_nb:]
    x_all = jnp.concatenate([liberal_x, nbx])
    y_all = jnp.concatenate([liberal_y, nby])
    result = sim(x_all, y_all, ws_amb=ws, wd_amb=wd)
    power = result.power()[:, :N_TARGET]
    return jnp.sum(power) * 8760 / 1e6 / power.shape[0]


# ── ADAM helper ──────────────────────────────────────────────────────────

def adam_init(params):
    return jnp.zeros_like(params), jnp.zeros_like(params)


def adam_step(params, grad, m, v, t, lr=OUTER_LR, beta1=0.9, beta2=0.999, eps=1e-8):
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad ** 2
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    # Gradient ASCENT (maximize regret)
    params = params + lr * m_hat / (jnp.sqrt(v_hat) + eps)
    return params, m, v


def enforce_buffer(nb_params, n_nb=N_NEIGHBOR):
    """Project neighbors outside min_spacing buffer from target boundary."""
    nb_x = nb_params[:n_nb]
    nb_y = nb_params[n_nb:]
    cx = jnp.clip(nb_x, 0.0, boundary_size)
    cy = jnp.clip(nb_y, 0.0, boundary_size)
    dx, dy = nb_x - cx, nb_y - cy
    dist = jnp.sqrt(dx**2 + dy**2)
    scale = jnp.where(dist < 1e-6, 1.0, min_spacing / dist)
    nb_x = jnp.where(dist < min_spacing, cx + dx * scale, nb_x)
    nb_y = jnp.where(dist < min_spacing, cy + dy * scale, nb_y)
    return jnp.concatenate([nb_x, nb_y])


# ========================================================================
# Strategy 1: Single-start IFT (baseline)
# ========================================================================

def run_single_start_ift():
    """Single fixed initial layout, IFT gradient through it."""
    print("Strategy 1: Single-start IFT")

    def compute_regret(nb_params):
        opt_x, opt_y = sgd_solve_implicit(
            objective_with_neighbors, init_x, init_y,
            boundary, min_spacing, sgd_settings, nb_params,
        )
        cons_aep = eval_conservative_aep(opt_x, opt_y, nb_params)
        lib_aep_present = eval_liberal_aep_present(nb_params)
        return cons_aep - lib_aep_present

    regret_and_grad = value_and_grad(compute_regret)

    nb_params = nb_params_init.copy()
    m, v = adam_init(nb_params)
    history = []

    t0 = time.perf_counter()
    for i in range(OUTER_ITERS):
        regret, grad = regret_and_grad(nb_params)
        regret = float(regret)
        grad_norm = float(jnp.linalg.norm(grad))
        history.append({"iter": i, "regret": regret, "grad_norm": grad_norm})
        print(f"  iter {i:3d}: regret={regret:.4f} GWh  |grad|={grad_norm:.4e}")

        if not jnp.all(jnp.isfinite(grad)):
            break

        nb_params, m, v = adam_step(nb_params, grad, m, v, i + 1)
        nb_params = enforce_buffer(nb_params)

    elapsed = time.perf_counter() - t0
    print(f"  Time: {elapsed:.2f}s\n")
    return history, elapsed


# ========================================================================
# Strategy 2: Stochastic single-start
# ========================================================================

def run_stochastic_single_start():
    """Random initial layout each outer iteration, IFT through it."""
    print("Strategy 2: Stochastic single-start")

    key = jax.random.PRNGKey(123)

    nb_params = nb_params_init.copy()
    m, v = adam_init(nb_params)
    history = []

    t0 = time.perf_counter()
    for i in range(OUTER_ITERS):
        # Random initial layout for this iteration
        k1, k2, key = jax.random.split(key, 3)
        margin = min_spacing
        rand_x = jax.random.uniform(k1, (N_TARGET,), minval=margin, maxval=boundary_size - margin)
        rand_y = jax.random.uniform(k2, (N_TARGET,), minval=margin, maxval=boundary_size - margin)

        def compute_regret(nb_params, ix=rand_x, iy=rand_y):
            opt_x, opt_y = sgd_solve_implicit(
                objective_with_neighbors, ix, iy,
                boundary, min_spacing, sgd_settings, nb_params,
            )
            cons_aep = eval_conservative_aep(opt_x, opt_y, nb_params)
            lib_aep_present = eval_liberal_aep_present(nb_params)
            return cons_aep - lib_aep_present

        regret, grad = value_and_grad(compute_regret)(nb_params)
        regret = float(regret)
        grad_norm = float(jnp.linalg.norm(grad))
        history.append({"iter": i, "regret": regret, "grad_norm": grad_norm})
        print(f"  iter {i:3d}: regret={regret:.4f} GWh  |grad|={grad_norm:.4e}")

        if not jnp.all(jnp.isfinite(grad)):
            break

        nb_params, m, v = adam_step(nb_params, grad, m, v, i + 1)
        nb_params = enforce_buffer(nb_params)

    elapsed = time.perf_counter() - t0
    print(f"  Time: {elapsed:.2f}s\n")
    return history, elapsed


# ========================================================================
# Strategy 3: Envelope theorem (K forward, IFT through winner)
# ========================================================================

def run_envelope_theorem():
    """K forward-only starts, IFT backward through the winner."""
    print(f"Strategy 3: Envelope theorem (K={K_STARTS})")

    nb_params = nb_params_init.copy()
    m, v = adam_init(nb_params)
    history = []

    t0 = time.perf_counter()
    for i in range(OUTER_ITERS):
        # Forward-only: run K starts, evaluate regret for each (no grad)
        start_regrets = []
        for k, (ix, iy) in enumerate(random_inits):
            opt_x, opt_y = topfarm_sgd_solve(
                lambda x, y, p=nb_params: objective_with_neighbors(x, y, p),
                ix, iy, boundary, min_spacing, sgd_settings,
            )
            cons_aep = float(eval_conservative_aep(opt_x, opt_y, nb_params))
            lib_aep_present = float(eval_liberal_aep_present(nb_params))
            start_regrets.append(cons_aep - lib_aep_present)

        best_k = int(np.argmax(start_regrets))
        best_ix, best_iy = random_inits[best_k]

        # Backward: IFT through the winning start only
        def compute_regret_winner(nb_params, ix=best_ix, iy=best_iy):
            opt_x, opt_y = sgd_solve_implicit(
                objective_with_neighbors, ix, iy,
                boundary, min_spacing, sgd_settings, nb_params,
            )
            cons_aep = eval_conservative_aep(opt_x, opt_y, nb_params)
            lib_aep_present = eval_liberal_aep_present(nb_params)
            return cons_aep - lib_aep_present

        regret, grad = value_and_grad(compute_regret_winner)(nb_params)
        regret = float(regret)
        grad_norm = float(jnp.linalg.norm(grad))
        history.append({
            "iter": i, "regret": regret, "grad_norm": grad_norm,
            "best_k": best_k, "all_regrets": start_regrets,
        })
        print(f"  iter {i:3d}: regret={regret:.4f} GWh  best_k={best_k}"
              f"  |grad|={grad_norm:.4e}  starts={[f'{r:.3f}' for r in start_regrets]}")

        if not jnp.all(jnp.isfinite(grad)):
            break

        nb_params, m, v = adam_step(nb_params, grad, m, v, i + 1)
        nb_params = enforce_buffer(nb_params)

    elapsed = time.perf_counter() - t0
    print(f"  Time: {elapsed:.2f}s\n")
    return history, elapsed


# ========================================================================
# Strategy 4: LogSumExp soft selection
# ========================================================================

def run_logsumexp():
    """K starts, gradients through all via LogSumExp aggregation."""
    print(f"Strategy 4: LogSumExp (K={K_STARTS}, tau={LOGSUMEXP_TAU})")

    nb_params = nb_params_init.copy()
    m, v = adam_init(nb_params)
    history = []

    def compute_soft_max_regret(nb_params):
        regrets = []
        for ix, iy in random_inits:
            opt_x, opt_y = sgd_solve_implicit(
                objective_with_neighbors, ix, iy,
                boundary, min_spacing, sgd_settings, nb_params,
            )
            cons_aep = eval_conservative_aep(opt_x, opt_y, nb_params)
            lib_aep_present = eval_liberal_aep_present(nb_params)
            regrets.append(cons_aep - lib_aep_present)
        regrets_arr = jnp.stack(regrets)
        # Smooth max: (1/tau) * logsumexp(tau * regrets)
        return (1.0 / LOGSUMEXP_TAU) * jax.nn.logsumexp(LOGSUMEXP_TAU * regrets_arr)

    soft_regret_and_grad = value_and_grad(compute_soft_max_regret)

    t0 = time.perf_counter()
    for i in range(OUTER_ITERS):
        soft_regret, grad = soft_regret_and_grad(nb_params)
        soft_regret = float(soft_regret)
        grad_norm = float(jnp.linalg.norm(grad))

        # Also compute true max regret for comparison
        true_regrets = []
        for ix, iy in random_inits:
            opt_x, opt_y = topfarm_sgd_solve(
                lambda x, y, p=nb_params: objective_with_neighbors(x, y, p),
                ix, iy, boundary, min_spacing, sgd_settings,
            )
            cons_aep = float(eval_conservative_aep(opt_x, opt_y, nb_params))
            lib_aep_present = float(eval_liberal_aep_present(nb_params))
            true_regrets.append(cons_aep - lib_aep_present)
        true_max = max(true_regrets)

        history.append({
            "iter": i, "regret": true_max, "soft_regret": soft_regret,
            "grad_norm": grad_norm, "all_regrets": true_regrets,
        })
        print(f"  iter {i:3d}: soft={soft_regret:.4f}  true_max={true_max:.4f} GWh"
              f"  |grad|={grad_norm:.4e}")

        if not jnp.all(jnp.isfinite(grad)):
            break

        nb_params, m, v = adam_step(nb_params, grad, m, v, i + 1)
        nb_params = enforce_buffer(nb_params)

    elapsed = time.perf_counter() - t0
    print(f"  Time: {elapsed:.2f}s\n")
    return history, elapsed


# ========================================================================
# Run all strategies
# ========================================================================

print("=" * 70)
print(f"Multistart Benchmark: {N_TARGET} targets, {N_NEIGHBOR} neighbors")
print(f"  K={K_STARTS} starts, {OUTER_ITERS} outer iterations, ADAM lr={OUTER_LR}")
print("=" * 70 + "\n")

results = {}

# Warm up JIT
print("Warming up JIT...")
_ = value_and_grad(lambda p: eval_conservative_aep(
    *sgd_solve_implicit(objective_with_neighbors, init_x, init_y,
                        boundary, min_spacing, sgd_settings, p),
    p) - eval_liberal_aep_present(p))(nb_params_init)
print("Done.\n")

for name, fn in [
    ("Single-start IFT", run_single_start_ift),
    ("Stochastic single-start", run_stochastic_single_start),
    ("Envelope theorem", run_envelope_theorem),
    ("LogSumExp", run_logsumexp),
]:
    history, elapsed = fn()
    results[name] = {"history": history, "elapsed": elapsed}


# ========================================================================
# Comparison plots
# ========================================================================

print("=" * 70)
print("Results")
print("=" * 70)

colors = {
    "Single-start IFT": "#2196F3",
    "Stochastic single-start": "#9C27B0",
    "Envelope theorem": "#4CAF50",
    "LogSumExp": "#F44336",
}

# ── Plot 1: Regret convergence ───────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
for name, res in results.items():
    regrets = [h["regret"] for h in res["history"]]
    ax.plot(regrets, "o-", color=colors[name], markersize=4, lw=2, label=name)
ax.set_xlabel("Outer iteration")
ax.set_ylabel("Regret (GWh)")
ax.set_title("Regret Convergence")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ── Plot 2: Gradient norm ────────────────────────────────────────────────
ax = axes[1]
for name, res in results.items():
    gnorms = [h["grad_norm"] for h in res["history"]]
    ax.semilogy(gnorms, "s-", color=colors[name], markersize=4, lw=1.5, label=name)
ax.set_xlabel("Outer iteration")
ax.set_ylabel("|grad|")
ax.set_title("Gradient Norm")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ── Plot 3: Wall-clock comparison bar chart ──────────────────────────────
ax = axes[2]
names = list(results.keys())
times = [results[n]["elapsed"] for n in names]
final_regrets = [results[n]["history"][-1]["regret"] for n in names]
bar_colors = [colors[n] for n in names]
bars = ax.bar(range(len(names)), times, color=bar_colors, edgecolor="black", linewidth=0.5)
for i, (bar, t, r) in enumerate(zip(bars, times, final_regrets)):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{t:.1f}s\n({r:.2f} GWh)", ha="center", va="bottom", fontsize=8)
ax.set_xticks(range(len(names)))
ax.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=8)
ax.set_ylabel("Wall-clock time (s)")
ax.set_title("Total Time (20 outer iters)")
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "multistart_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {OUTPUT_DIR / 'multistart_comparison.png'}")

# ── Summary table ────────────────────────────────────────────────────────
print(f"\n{'Strategy':<30} {'Time':>8} {'Final regret':>14} {'Per-iter':>10}")
print("-" * 65)
for name, res in results.items():
    t = res["elapsed"]
    final_r = res["history"][-1]["regret"]
    per_iter = t / len(res["history"])
    print(f"{name:<30} {t:7.1f}s {final_r:13.4f} GWh {per_iter:9.2f}s")

# ── Envelope: track winning start switches ───────────────────────────────
if "Envelope theorem" in results:
    env_hist = results["Envelope theorem"]["history"]
    if "best_k" in env_hist[0]:
        switches = 0
        for i in range(1, len(env_hist)):
            if env_hist[i]["best_k"] != env_hist[i - 1]["best_k"]:
                switches += 1
        print(f"\nEnvelope theorem: {switches} start switches over {len(env_hist)} iterations")
        winning_starts = [h["best_k"] for h in env_hist]
        print(f"  Winning starts: {winning_starts}")

print(f"\nAll outputs: {OUTPUT_DIR}/")

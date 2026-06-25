"""Scheduled SGD solver: per-step (lr, alpha, beta1, beta2) from a schedule.

Mirrors pixwake.optim.sgd._sgd_step semantics but reads per-step values from
schedule arrays precomputed inside JIT via vmap over step indices, so arbitrary
schedules (cosine, FunWake, Bayes-optimised) can be benchmarked apples-to-apples.

Pixwake baseline's running-best curve is reconstructed from the parent's
existing K=2000 conservative AEP pool (no fresh compute).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.lax import fori_loop

from pixwake.optim.sgd import boundary_penalty, spacing_penalty


def scheduled_sgd_solve(
    objective_fn,
    init_x,
    init_y,
    boundary,
    min_spacing,
    schedule_apply,
    total_iter,
    lr_init,
    ks_rho=100.0,
    boundary_weight=1.0,
    spacing_weight=1.0,
):
    """Run `total_iter` SGD steps under `schedule_apply`.

    schedule_apply: pure jnp callable (step, total_steps, lr0, alpha0) ->
                    (lr, alpha, beta1, beta2)
    lr_init       : reference initial LR used to compute alpha0 (matches pixwake)
    """
    def constraint_penalty(x, y):
        return boundary_weight * boundary_penalty(x, y, boundary, ks_rho) + \
               spacing_weight * spacing_penalty(x, y, min_spacing, ks_rho)

    grad_obj = jax.grad(objective_fn, argnums=(0, 1))
    grad_con = jax.grad(constraint_penalty, argnums=(0, 1))

    g0x, g0y = grad_obj(init_x, init_y)
    grad_mag0 = jnp.mean(jnp.concatenate([jnp.abs(g0x), jnp.abs(g0y)]))
    alpha0 = grad_mag0 / lr_init

    # Precompute schedule arrays via vmap over step indices (inside JIT-traceable).
    steps = jnp.arange(total_iter)
    def one_step(s):
        return schedule_apply(s, total_iter, jnp.float64(lr_init), alpha0)
    lr_arr, alpha_arr, b1_arr, b2_arr = jax.vmap(one_step)(steps)

    m_x = jnp.zeros_like(init_x)
    m_y = jnp.zeros_like(init_y)
    v_x = jnp.zeros_like(init_x)
    v_y = jnp.zeros_like(init_y)

    def body(t, carry):
        x, y, mx, my, vx, vy = carry
        lr = lr_arr[t]
        alpha = alpha_arr[t]
        b1 = b1_arr[t]
        b2 = b2_arr[t]
        gox, goy = grad_obj(x, y)
        gcx, gcy = grad_con(x, y)
        jx = gox + alpha * gcx
        jy = goy + alpha * gcy
        mx = b1 * mx + (1 - b1) * jx
        my = b1 * my + (1 - b1) * jy
        vx = b2 * vx + (1 - b2) * jx ** 2
        vy = b2 * vy + (1 - b2) * jy ** 2
        it = t + 1
        mhx = mx / (1 - b1 ** it)
        mhy = my / (1 - b1 ** it)
        vhx = vx / (1 - b2 ** it)
        vhy = vy / (1 - b2 ** it)
        eps = 1e-12
        x = x - lr * mhx / (jnp.sqrt(vhx) + eps)
        y = y - lr * mhy / (jnp.sqrt(vhy) + eps)
        return (x, y, mx, my, vx, vy)

    carry0 = (init_x, init_y, m_x, m_y, v_x, v_y)
    final = fori_loop(0, total_iter, body, carry0)
    return final[0], final[1]


def scheduled_vmap_solve_batch(
    sim,
    starts_x,
    starts_y,
    boundary,
    min_spacing,
    schedule_apply,
    total_iter,
    lr_init,
    ws_amb,
    wd_amb,
    weights,
    ti_amb=None,
    neighbor_x=None,
    neighbor_y=None,
    chunk_size=50,
):
    """Run K-start scheduled SGD via vmap (in chunks). Returns AEP per start."""
    n_target = starts_x.shape[1]
    has_neighbors = neighbor_x is not None and neighbor_x.shape[0] > 0

    def solve_one(sx, sy):
        if has_neighbors:
            def objective(x, y):
                x_all = jnp.concatenate([x, neighbor_x])
                y_all = jnp.concatenate([y, neighbor_y])
                result = sim(x_all, y_all, ws_amb=ws_amb, wd_amb=wd_amb, ti_amb=ti_amb)
                power = result.power()[:, :n_target]
                return -jnp.sum(power * weights[:, None]) * 8760 / 1e6
        else:
            def objective(x, y):
                result = sim(x, y, ws_amb=ws_amb, wd_amb=wd_amb, ti_amb=ti_amb)
                power = result.power()
                return -jnp.sum(power * weights[:, None]) * 8760 / 1e6

        ox, oy = scheduled_sgd_solve(
            objective, sx, sy, boundary, min_spacing,
            schedule_apply, total_iter, lr_init,
        )
        return -objective(ox, oy)

    n_starts = starts_x.shape[0]
    out = []
    for i in range(0, n_starts, chunk_size):
        j = min(i + chunk_size, n_starts)
        aeps = jax.vmap(solve_one)(starts_x[i:j], starts_y[i:j])
        out.append(aeps)
        print(f"    chunk {i}-{j}: best in chunk {float(jnp.max(aeps)):.2f} GWh",
              flush=True)
    return jnp.concatenate(out)

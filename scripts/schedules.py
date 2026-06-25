"""Schedule library for the multistart-convergence comparison.

Each schedule is a closure returning a `schedule_fn(step, total_steps, lr0, alpha0)`
callable that produces a (lr, alpha, beta1, beta2) tuple. Built on jnp so the
returned schedule can be vmapped over step indices inside a JIT.

alpha0 here is the same alpha0 pixwake computes from the init gradient
magnitude (mean(|grad_obj|) / lr0). It's passed in so schedules can build alpha
schedules relative to the natural penalty scale, exactly matching FunWake's
contract.
"""

from __future__ import annotations

import jax.numpy as jnp


def sgd_baseline(lr_init: float = 50.0):
    """Pixwake baseline. Constant LR, no decay (matches FunWake "iter_0" baseline).

    Use this only for sanity-check fresh runs; the paper's true baseline curve
    is reconstructed from the parent's K=2000 conservative AEP pool.
    """
    def apply(step, total_steps, lr0, alpha0):
        lr = jnp.full((), float(lr_init), dtype=jnp.float64)
        alpha = alpha0
        beta1 = jnp.float64(0.1)
        beta2 = jnp.float64(0.2)
        return lr, alpha, beta1, beta2
    return apply


def cosine_decay(
    lr_init: float = 50.0,
    lr_final_frac: float = 0.01,
    penalty_init_mult: float = 1.0,
    penalty_final_mult: float = 1.0,
    beta1_v: float = 0.1,
    beta2_v: float = 0.2,
):
    """Cosine LR decay (multiplicative penalty also cosine-interpolated).

    LR  : lr_init at t=0 -> lr_init * lr_final_frac at t=1
    alpha: penalty_init_mult*alpha0 at t=0 -> penalty_final_mult*alpha0 at t=1
    """
    def apply(step, total_steps, lr0, alpha0):
        t = step / jnp.maximum(total_steps - 1, 1)
        cos_t = 0.5 * (1.0 + jnp.cos(jnp.pi * t))
        lr = lr_init * (lr_final_frac + (1.0 - lr_final_frac) * cos_t)
        pen_mult = penalty_final_mult + (penalty_init_mult - penalty_final_mult) * cos_t
        alpha = pen_mult * alpha0
        beta1 = jnp.float64(beta1_v)
        beta2 = jnp.float64(beta2_v)
        return lr, alpha, beta1, beta2
    return apply


def funwake_iter192(lr_init: float = 50.0):
    """FunWake iter_192 verbatim (warmup + cosine + dual bumps + alpha dip).

    Source: ../../portfolio_copy/funwake/results_agent_schedule_only_5hr/iter_192.py
    """
    lr0_setting = float(lr_init)

    def apply(step, total_steps, lr0, alpha0):
        # lr0 here is the *runtime* solver LR scale — but FunWake's original code
        # uses a fixed lr0=50 reference. We honor lr_init as the reference.
        lr_ref = lr0_setting
        t = step / total_steps
        lr_peak = 4.0 * lr_ref
        lr_min = lr_peak / 10000.0
        warmup_end = 0.05
        warmup_lr = lr_peak * t / warmup_end
        cosine_t = (t - warmup_end) / (1.0 - warmup_end)
        cosine_lr = lr_min + (lr_peak - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * cosine_t))
        lr_base = jnp.where(t < warmup_end, warmup_lr, cosine_lr)
        bump1 = 0.2 * lr_peak * jnp.exp(-0.5 * ((t - 0.5) / 0.04) ** 2)
        bump2 = 0.3 * lr_peak * jnp.exp(-0.5 * ((t - 0.75) / 0.05) ** 2)
        lr = lr_base + bump1 + bump2

        alpha_base = 5.0 * alpha0 * lr_peak / jnp.maximum(lr, 1e-10)
        late = jnp.maximum(t - 0.5, 0.0) / 0.5
        alpha_extra = 3.0 * alpha0 * late ** 2
        dip = 0.5 * jnp.exp(-0.5 * ((t - 0.6) / 0.04) ** 2)
        alpha = (alpha_base + alpha_extra) * (1.0 - dip)

        beta1 = jnp.float64(0.3)
        beta2 = jnp.float64(0.5)
        return lr, alpha, beta1, beta2
    return apply


def bayes_optimised(params: dict):
    """Cosine variant with BO-tuned hyperparameters.

    `params` keys: lr_init, lr_final_frac, penalty_init_mult, penalty_final_mult.
    """
    return cosine_decay(
        lr_init=params["lr_init"],
        lr_final_frac=params["lr_final_frac"],
        penalty_init_mult=params["penalty_init_mult"],
        penalty_final_mult=params["penalty_final_mult"],
    )


SCHEDULES = {
    "sgd_baseline": sgd_baseline,
    "cosine": cosine_decay,
    "funwake_iter192": funwake_iter192,
}

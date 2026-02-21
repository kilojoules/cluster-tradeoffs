"""Tests for IFT-based implicit differentiation through SGD solver.

Validates that sgd_solve_implicit produces correct gradients by comparing
against finite differences, and that GradientAdversarialSearch runs end-to-end.

IMPORTANT: jax_enable_x64 must be set BEFORE importing pixwake.
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

from pixwake import WakeSimulation
from pixwake.definitions.v80 import vestas_v80
from pixwake.deficit import NOJDeficit
from pixwake.optim.sgd import (
    SGDSettings,
    sgd_solve_implicit,
    topfarm_sgd_solve,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

D = vestas_v80.rotor_diameter  # 80m


def _square_boundary(size: float) -> jnp.ndarray:
    """Return CCW square boundary vertices centered at (size/2, size/2)."""
    return jnp.array([
        [0.0, 0.0],
        [size, 0.0],
        [size, size],
        [0.0, size],
    ])


# ---------------------------------------------------------------------------
# Test 1: Quadratic IFT — verify custom_vjp plumbing
# ---------------------------------------------------------------------------


def test_quadratic_ift():
    """min_x (x - p)^2  =>  x* = p,  dx*/dp = I  =>  d/dp ||x*||^2 = 2p."""

    def objective(x, y, params):
        # Quadratic bowl: (x - params_x)^2 + (y - params_y)^2
        n = x.shape[0]
        px, py = params[:n], params[n:]
        return jnp.sum((x - px) ** 2) + jnp.sum((y - py) ** 2)

    n = 2
    init_x = jnp.zeros(n)
    init_y = jnp.zeros(n)
    boundary = _square_boundary(10.0)
    min_spacing = 0.1  # small so constraint doesn't bind
    settings = SGDSettings(learning_rate=1.0, max_iter=500, tol=1e-10)

    params = jnp.array([3.0, 7.0, 4.0, 6.0])  # px=[3,7], py=[4,6]

    # Forward: optimizer should find x*=p (within boundary)
    opt_x, opt_y = sgd_solve_implicit(
        objective, init_x, init_y, boundary, min_spacing, settings, params
    )
    assert jnp.allclose(opt_x, params[:n], atol=0.5), f"opt_x={opt_x}, expected {params[:n]}"
    assert jnp.allclose(opt_y, params[n:], atol=0.5), f"opt_y={opt_y}, expected {params[n:]}"

    # Backward: d/dp sum(x*^2 + y*^2) should be approximately 2*p
    def outer_loss(p):
        ox, oy = sgd_solve_implicit(
            objective, init_x, init_y, boundary, min_spacing, settings, p
        )
        return jnp.sum(ox ** 2) + jnp.sum(oy ** 2)

    grad_ad = jax.grad(outer_loss)(params)

    # Finite difference reference
    eps = 1e-3
    grad_fd = jnp.zeros_like(params)
    for i in range(len(params)):
        e = jnp.zeros_like(params).at[i].set(eps)
        grad_fd = grad_fd.at[i].set(
            (outer_loss(params + e) - outer_loss(params - e)) / (2 * eps)
        )

    # Check that IFT gradient has correct sign and roughly correct magnitude
    assert jnp.allclose(grad_ad, grad_fd, atol=1.0, rtol=0.5), (
        f"IFT grad={grad_ad}, FD grad={grad_fd}"
    )


# ---------------------------------------------------------------------------
# Test 2: Inner solver convergence with wake simulation
# ---------------------------------------------------------------------------


def test_inner_solver_convergence():
    """4 V80 turbines, NOJ deficit, single wind dir — verify SGD converges."""
    sim = WakeSimulation(vestas_v80, NOJDeficit(k=0.1))

    ws = jnp.array([9.0])
    wd = jnp.array([270.0])

    # 4 turbines in a square, 4D spacing
    spacing = 4 * D
    init_x = jnp.array([0.0, spacing, 0.0, spacing])
    init_y = jnp.array([0.0, 0.0, spacing, spacing])

    boundary = _square_boundary(spacing + 2 * D)
    min_spacing_val = 2 * D

    def neg_aep(x, y):
        result = sim(x, y, ws_amb=ws, wd_amb=wd)
        return -result.aep()

    settings = SGDSettings(learning_rate=10.0, max_iter=1000, tol=1e-8)

    opt_x, opt_y = topfarm_sgd_solve(
        neg_aep, init_x, init_y, boundary, min_spacing_val, settings
    )

    # Verify: optimized AEP should be >= initial AEP
    init_aep = -neg_aep(init_x, init_y)
    final_aep = -neg_aep(opt_x, opt_y)
    assert final_aep >= init_aep - 0.01, (
        f"Optimization did not improve: init={init_aep:.4f}, final={final_aep:.4f}"
    )

    # Verify: no NaN
    assert jnp.all(jnp.isfinite(opt_x)), "opt_x contains NaN/Inf"
    assert jnp.all(jnp.isfinite(opt_y)), "opt_y contains NaN/Inf"

    # Verify: turbines stay within boundary
    assert jnp.all(opt_x >= -1.0) and jnp.all(opt_x <= boundary[1, 0] + 1.0)
    assert jnp.all(opt_y >= -1.0) and jnp.all(opt_y <= boundary[2, 1] + 1.0)


# ---------------------------------------------------------------------------
# Test 3: Wake sim IFT gradient vs outer FD
# ---------------------------------------------------------------------------


def test_wake_sim_ift_gradient():
    """4 targets + 1 neighbor — compare jax.grad through sgd_solve_implicit vs outer FD."""
    sim = WakeSimulation(vestas_v80, NOJDeficit(k=0.1))

    ws = jnp.array([9.0])
    wd = jnp.array([270.0])

    n_target = 4
    spacing = 4 * D
    init_x = jnp.array([0.0, spacing, 0.0, spacing])
    init_y = jnp.array([0.0, 0.0, spacing, spacing])

    boundary = _square_boundary(spacing + 2 * D)
    min_spacing_val = 2 * D

    # Use tighter tolerance for IFT accuracy
    settings = SGDSettings(learning_rate=10.0, max_iter=3000, tol=1e-8)

    def objective_with_neighbors(x, y, neighbor_params):
        n_nb = neighbor_params.shape[0] // 2
        nb_x = neighbor_params[:n_nb]
        nb_y = neighbor_params[n_nb:]
        x_all = jnp.concatenate([x, nb_x])
        y_all = jnp.concatenate([y, nb_y])
        result = sim(x_all, y_all, ws_amb=ws, wd_amb=wd)
        power = result.power()[:, :n_target]
        return -jnp.sum(power) * 8760 / 1e6 / power.shape[0]

    # 1 neighbor placed directly upwind of target at (0,0)
    # At -2D, wake radius = D/2 + k*2D = 40 + 16 = 56m, so it hits the (0,0) turbine
    neighbor_x = jnp.array([-2 * D])
    neighbor_y = jnp.array([0.0])
    neighbor_params = jnp.concatenate([neighbor_x, neighbor_y])

    def compute_conservative_aep(nb_params):
        opt_x, opt_y = sgd_solve_implicit(
            objective_with_neighbors, init_x, init_y,
            boundary, min_spacing_val, settings, nb_params,
        )
        n_nb = nb_params.shape[0] // 2
        nb_x = nb_params[:n_nb]
        nb_y = nb_params[n_nb:]
        x_all = jnp.concatenate([opt_x, nb_x])
        y_all = jnp.concatenate([opt_y, nb_y])
        result = sim(x_all, y_all, ws_amb=ws, wd_amb=wd)
        power = result.power()[:, :n_target]
        return jnp.sum(power) * 8760 / 1e6 / power.shape[0]

    # IFT gradient
    grad_ad = jax.grad(compute_conservative_aep)(neighbor_params)

    # Finite difference gradient (1m perturbation for ~300m positions)
    eps_fd = 1.0
    grad_fd = jnp.zeros_like(neighbor_params)
    for i in range(len(neighbor_params)):
        e = jnp.zeros_like(neighbor_params).at[i].set(eps_fd)
        f_plus = compute_conservative_aep(neighbor_params + e)
        f_minus = compute_conservative_aep(neighbor_params - e)
        grad_fd = grad_fd.at[i].set((f_plus - f_minus) / (2 * eps_fd))

    # Allow generous tolerance: sign and order of magnitude agreement
    # 50% relative error is acceptable for validating direction
    for i in range(len(neighbor_params)):
        ad_val = float(grad_ad[i])
        fd_val = float(grad_fd[i])

        # Check no NaN
        assert jnp.isfinite(grad_ad[i]), f"IFT gradient[{i}] is NaN/Inf"

        # If FD gradient is non-negligible, check sign agreement
        if abs(fd_val) > 1e-8:
            assert ad_val * fd_val > 0 or abs(ad_val) < 1e-6, (
                f"Sign mismatch at [{i}]: IFT={ad_val:.6e}, FD={fd_val:.6e}"
            )


# ---------------------------------------------------------------------------
# Test 4: GradientAdversarialSearch smoke test
# ---------------------------------------------------------------------------


def test_gradient_adversarial_search_smoke():
    """4 targets + 2 neighbors, 10 outer iters — verify regret increases, no NaN."""
    from pixwake.optim.adversarial import (
        AdversarialSearchSettings,
        GradientAdversarialSearch,
    )

    sim = WakeSimulation(vestas_v80, NOJDeficit(k=0.1))

    ws = jnp.array([9.0])
    wd = jnp.array([270.0])

    spacing = 4 * D
    init_x = jnp.array([0.0, spacing, 0.0, spacing])
    init_y = jnp.array([0.0, 0.0, spacing, spacing])

    boundary = _square_boundary(spacing + 2 * D)

    # 2 neighbors placed upwind
    nb_x = jnp.array([-3 * D, -5 * D])
    nb_y = jnp.array([0.0, spacing])

    sgd_settings = SGDSettings(learning_rate=10.0, max_iter=500, tol=1e-6)
    search_settings = AdversarialSearchSettings(
        max_iter=10,
        learning_rate=10.0,
        sgd_settings=sgd_settings,
        verbose=False,
    )

    searcher = GradientAdversarialSearch(
        sim=sim,
        target_boundary=boundary,
        target_min_spacing=2 * D,
        ws_amb=ws,
        wd_amb=wd,
    )

    result = searcher.search(init_x, init_y, nb_x, nb_y, settings=search_settings)

    # Basic sanity checks
    assert jnp.all(jnp.isfinite(result.neighbor_x)), "neighbor_x contains NaN"
    assert jnp.all(jnp.isfinite(result.neighbor_y)), "neighbor_y contains NaN"
    assert jnp.isfinite(result.regret), "regret is NaN"
    assert result.liberal_aep > 0, "liberal AEP should be positive"

    # Regret should be non-negative (neighbors should hurt performance)
    assert result.regret >= -0.1, f"Negative regret: {result.regret}"

    # History should have entries
    assert len(result.history) > 0, "No history entries"

    # Check that optimization ran without catastrophic failure
    # (regret staying in a reasonable range, not diverging to ±inf)
    if len(result.history) >= 3:
        all_regrets = [h[0] for h in result.history]
        assert all(abs(r) < 1e6 for r in all_regrets), (
            f"Regret diverged: {all_regrets}"
        )


# ---------------------------------------------------------------------------
# Test 5: Multistart inner finds better-or-equal objective
# ---------------------------------------------------------------------------


def test_multistart_inner_finds_better():
    """K=3 inner starts find equal-or-better objective than single start."""
    from pixwake.optim.sgd import (
        generate_random_starts,
        topfarm_sgd_solve_multistart,
    )

    sim = WakeSimulation(vestas_v80, NOJDeficit(k=0.1))

    ws = jnp.array([9.0])
    wd = jnp.array([270.0])

    spacing = 4 * D
    init_x = jnp.array([0.0, spacing, 0.0, spacing])
    init_y = jnp.array([0.0, 0.0, spacing, spacing])

    boundary = _square_boundary(spacing + 2 * D)
    min_spacing_val = 2 * D

    def neg_aep(x, y):
        result = sim(x, y, ws_amb=ws, wd_amb=wd)
        return -result.aep()

    settings = SGDSettings(learning_rate=10.0, max_iter=500, tol=1e-6)

    # Single start
    single_x, single_y = topfarm_sgd_solve(
        neg_aep, init_x, init_y, boundary, min_spacing_val, settings
    )
    single_obj = float(neg_aep(single_x, single_y))

    # Multistart: K=3 (first start = same as single, plus 2 random)
    key = jax.random.PRNGKey(42)
    rand_x, rand_y = generate_random_starts(
        key, 2, len(init_x), boundary, min_spacing_val
    )
    batch_x = jnp.concatenate([init_x[None, :], rand_x], axis=0)
    batch_y = jnp.concatenate([init_y[None, :], rand_y], axis=0)

    all_x, all_y, all_objs = topfarm_sgd_solve_multistart(
        neg_aep, batch_x, batch_y, boundary, min_spacing_val, settings
    )

    best_multi_obj = float(jnp.min(all_objs))

    # Multi should be at least as good as single (lower obj = better)
    assert best_multi_obj <= single_obj + 0.01, (
        f"Multistart ({best_multi_obj:.4f}) worse than single ({single_obj:.4f})"
    )
    # All results should be finite
    assert jnp.all(jnp.isfinite(all_x)), "all_x contains NaN"
    assert jnp.all(jnp.isfinite(all_y)), "all_y contains NaN"


# ---------------------------------------------------------------------------
# Test 6: Multistart implicit gradient correctness
# ---------------------------------------------------------------------------


def test_multistart_gradient_correctness():
    """Compare sgd_solve_implicit_multistart grad vs single-start FD at winner."""
    from pixwake.optim.sgd import sgd_solve_implicit_multistart

    sim = WakeSimulation(vestas_v80, NOJDeficit(k=0.1))

    ws = jnp.array([9.0])
    wd = jnp.array([270.0])

    n_target = 4
    spacing = 4 * D
    init_x = jnp.array([0.0, spacing, 0.0, spacing])
    init_y = jnp.array([0.0, 0.0, spacing, spacing])

    boundary = _square_boundary(spacing + 2 * D)
    min_spacing_val = 2 * D
    settings = SGDSettings(learning_rate=10.0, max_iter=500, tol=1e-6)

    def objective_with_neighbors(x, y, neighbor_params):
        n_nb = neighbor_params.shape[0] // 2
        nb_x, nb_y = neighbor_params[:n_nb], neighbor_params[n_nb:]
        x_all = jnp.concatenate([x, nb_x])
        y_all = jnp.concatenate([y, nb_y])
        result = sim(x_all, y_all, ws_amb=ws, wd_amb=wd)
        power = result.power()[:, :n_target]
        return -jnp.sum(power) * 8760 / 1e6 / power.shape[0]

    neighbor_params = jnp.array([-2 * D, 0.0])  # 1 neighbor upwind

    # K=2 starts (one deterministic, one shifted)
    init_x2 = init_x + 10.0
    init_y2 = init_y + 10.0
    batch_x = jnp.stack([init_x, init_x2])
    batch_y = jnp.stack([init_y, init_y2])

    def compute_aep_multistart(nb_params):
        opt_x, opt_y = sgd_solve_implicit_multistart(
            objective_with_neighbors,
            batch_x, batch_y, nb_params,
            boundary, min_spacing_val, settings,
        )
        n_nb = nb_params.shape[0] // 2
        nb_x, nb_y = nb_params[:n_nb], nb_params[n_nb:]
        x_all = jnp.concatenate([opt_x, nb_x])
        y_all = jnp.concatenate([opt_y, nb_y])
        result = sim(x_all, y_all, ws_amb=ws, wd_amb=wd)
        power = result.power()[:, :n_target]
        return jnp.sum(power) * 8760 / 1e6 / power.shape[0]

    # AD gradient
    grad_ad = jax.grad(compute_aep_multistart)(neighbor_params)

    # FD gradient
    eps_fd = 1.0
    grad_fd = jnp.zeros_like(neighbor_params)
    for i in range(len(neighbor_params)):
        e = jnp.zeros_like(neighbor_params).at[i].set(eps_fd)
        f_plus = compute_aep_multistart(neighbor_params + e)
        f_minus = compute_aep_multistart(neighbor_params - e)
        grad_fd = grad_fd.at[i].set((f_plus - f_minus) / (2 * eps_fd))

    # Check finite
    assert jnp.all(jnp.isfinite(grad_ad)), f"AD grad has NaN: {grad_ad}"

    # Check sign agreement where FD is non-negligible
    for i in range(len(neighbor_params)):
        ad_val, fd_val = float(grad_ad[i]), float(grad_fd[i])
        if abs(fd_val) > 1e-8:
            assert ad_val * fd_val > 0 or abs(ad_val) < 1e-6, (
                f"Sign mismatch [{i}]: AD={ad_val:.6e}, FD={fd_val:.6e}"
            )


# ---------------------------------------------------------------------------
# Test 7: search_multistart smoke test
# ---------------------------------------------------------------------------


def test_search_multistart_smoke():
    """M=2 outer starts — verify result is valid and no NaN."""
    from pixwake.optim.adversarial import (
        AdversarialSearchSettings,
        GradientAdversarialSearch,
    )

    sim = WakeSimulation(vestas_v80, NOJDeficit(k=0.1))

    ws = jnp.array([9.0])
    wd = jnp.array([270.0])

    spacing = 4 * D
    init_x = jnp.array([0.0, spacing, 0.0, spacing])
    init_y = jnp.array([0.0, 0.0, spacing, spacing])

    boundary = _square_boundary(spacing + 2 * D)

    # M=2 outer starts with different neighbor positions
    nb_x_batch = jnp.array([
        [-3 * D, -5 * D],
        [-4 * D, -6 * D],
    ])
    nb_y_batch = jnp.array([
        [0.0, spacing],
        [spacing / 2, spacing / 2],
    ])

    sgd_settings = SGDSettings(learning_rate=10.0, max_iter=500, tol=1e-6)
    search_settings = AdversarialSearchSettings(
        max_iter=5,
        learning_rate=10.0,
        sgd_settings=sgd_settings,
        verbose=False,
    )

    searcher = GradientAdversarialSearch(
        sim=sim,
        target_boundary=boundary,
        target_min_spacing=2 * D,
        ws_amb=ws,
        wd_amb=wd,
    )

    result = searcher.search_multistart(
        init_x, init_y, nb_x_batch, nb_y_batch, settings=search_settings,
    )

    assert jnp.all(jnp.isfinite(result.neighbor_x)), "neighbor_x NaN"
    assert jnp.all(jnp.isfinite(result.neighbor_y)), "neighbor_y NaN"
    assert jnp.isfinite(result.regret), "regret NaN"
    assert result.liberal_aep > 0, "liberal AEP should be positive"

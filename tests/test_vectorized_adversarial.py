"""Tests for vectorized multi-start bilevel adversarial search.

Validates that n_starts > 1 correctly parallelizes and improves layout 
discovery compared to single-start.
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

from pixwake import WakeSimulation
from pixwake.definitions.v80 import vestas_v80
from pixwake.deficit import NOJDeficit
from pixwake.optim.sgd import SGDSettings
from pixwake.optim.adversarial import (
    AdversarialSearchSettings,
    GradientAdversarialSearch,
)

D = vestas_v80.rotor_diameter

def _square_boundary(size: float) -> jnp.ndarray:
    return jnp.array([[0.0, 0.0], [size, 0.0], [size, size], [0.0, size]])

def test_multi_start_improvement():
    """Verify that multi-start can find a better isolated layout than single-start."""
    sim = WakeSimulation(vestas_v80, NOJDeficit(k=0.1))
    
    # 4 turbines in a small boundary
    target_size = 6 * D
    boundary = _square_boundary(target_size)
    
    # Initial grid layout (likely a local minimum)
    init_x = jnp.array([target_size*0.3, target_size*0.7, target_size*0.3, target_size*0.7])
    init_y = jnp.array([target_size*0.3, target_size*0.3, target_size*0.7, target_size*0.7])
    
    # Neighbors placed to create a design gap
    nb_x = jnp.array([-2*D, -2*D])
    nb_y = jnp.array([target_size*0.2, target_size*0.8])
    
    searcher = GradientAdversarialSearch(
        sim=sim, target_boundary=boundary, target_min_spacing=2*D,
        ws_amb=jnp.array([9.0]), wd_amb=jnp.array([270.0])
    )
    
    sgd_settings = SGDSettings(max_iter=200, learning_rate=10.0)
    
    # Run with n_starts=1
    settings_1 = AdversarialSearchSettings(max_iter=2, n_starts=1, sgd_settings=sgd_settings, verbose=False)
    res1 = searcher.search(init_x, init_y, nb_x, nb_y, settings=settings_1)
    
    # Run with n_starts=10
    settings_10 = AdversarialSearchSettings(max_iter=2, n_starts=10, sgd_settings=sgd_settings, verbose=False)
    res10 = searcher.search(init_x, init_y, nb_x, nb_y, settings=settings_10)
    
    # Multi-start should find an isolated AEP >= single-start
    msg = f"Multi-start liberal AEP ({res10.liberal_aep}) < single-start ({res1.liberal_aep})"
    assert res10.liberal_aep >= res1.liberal_aep - 1e-6, msg
    
    # Regret should be non-negative
    assert res10.regret >= -1e-6, f"Negative regret: {res10.regret}"
    assert res1.regret >= -1e-6, f"Negative regret: {res1.regret}"

def test_vectorization_smoke():
    """Smoke test to ensure vmap doesn't crash with different n_starts."""
    sim = WakeSimulation(vestas_v80, NOJDeficit(k=0.1))
    boundary = _square_boundary(1000.0)
    init_x = jnp.array([200.0, 800.0])
    init_y = jnp.array([200.0, 200.0])
    nb_x, nb_y = jnp.array([-200.0]), jnp.array([200.0])
    
    searcher = GradientAdversarialSearch(
        sim=sim, target_boundary=boundary, target_min_spacing=200.0,
        ws_amb=jnp.array([10.0]), wd_amb=jnp.array([270.0])
    )
    
    # Use n_starts=3 to verify parallel logic
    settings = AdversarialSearchSettings(max_iter=1, n_starts=3, sgd_settings=SGDSettings(max_iter=10))
    res = searcher.search(init_x, init_y, nb_x, nb_y, settings=settings)
    
    assert jnp.isfinite(res.regret)
    assert res.neighbor_x.shape == nb_x.shape

"""Smoke test for GreedyGridSearch.

Validates that the greedy grid search runs end-to-end on a small problem
and produces sensible results.

IMPORTANT: jax_enable_x64 must be set BEFORE importing pixwake.
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
    GreedyGridSearch,
    GreedyGridSettings,
)


D = vestas_v80.rotor_diameter  # 80m


def _square_boundary(size: float) -> jnp.ndarray:
    return jnp.array([
        [0.0, 0.0],
        [size, 0.0],
        [size, size],
        [0.0, size],
    ])


def test_greedy_grid_smoke():
    """4 target turbines, small grid of 6 candidate neighbors, place 2."""
    sim = WakeSimulation(vestas_v80, NOJDeficit(k=0.1))

    boundary_size = 4 * D
    boundary = _square_boundary(boundary_size)

    # Initial target positions (2x2 grid inside boundary)
    init_x = jnp.array([1.0 * D, 3.0 * D, 1.0 * D, 3.0 * D])
    init_y = jnp.array([1.0 * D, 1.0 * D, 3.0 * D, 3.0 * D])

    # Candidate neighbor grid: 6 points upwind of the farm
    grid = jnp.array([
        [-3 * D, 0.5 * D],
        [-3 * D, 1.5 * D],
        [-3 * D, 2.5 * D],
        [-5 * D, 0.5 * D],
        [-5 * D, 1.5 * D],
        [-5 * D, 2.5 * D],
    ])

    # Wind from the west (270 deg) at 9 m/s — neighbors are upwind
    ws = jnp.array([9.0])
    wd = jnp.array([270.0])

    sgd_settings = SGDSettings(max_iter=200, learning_rate=5.0)
    settings = GreedyGridSettings(sgd_settings=sgd_settings, verbose=True)

    searcher = GreedyGridSearch(
        sim, boundary, 2 * D,
        ws_amb=ws, wd_amb=wd,
    )

    result = searcher.search(
        init_x, init_y,
        grid=grid,
        n_neighbors=2,
        settings=settings,
    )

    # Basic sanity checks
    assert result.neighbor_x.shape == (2,)
    assert result.neighbor_y.shape == (2,)
    assert len(result.placement_order) == 2
    assert len(result.regret_history) == 2
    assert len(result.regret_maps) == 2

    # Regret should be non-negative (conservative design >= liberal when neighbors present)
    assert result.regret >= 0.0, f"Regret should be >= 0, got {result.regret}"

    # Each regret map should have values at evaluated points
    for rm in result.regret_maps:
        assert rm.shape == (6,)
        # At least some non-NaN values
        assert jnp.any(jnp.isfinite(rm))

    # Placement order should be unique grid indices
    assert len(set(result.placement_order)) == 2
    assert all(0 <= idx < 6 for idx in result.placement_order)

    print(f"\nRegret history: {result.regret_history}")
    print(f"Placement order: {result.placement_order}")
    print(f"Final regret: {result.regret:.4f} GWh")

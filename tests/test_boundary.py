"""Tests for pixwake.optim.boundary — polygon SDF, containment, and exclusion.

IMPORTANT: jax_enable_x64 must be set BEFORE importing pixwake.
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

from pixwake.optim.boundary import (
    containment_penalty,
    exclusion_penalty,
    polygon_sdf,
)
from pixwake.optim.sgd import boundary_penalty


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _square(size: float = 100.0) -> jnp.ndarray:
    """CCW square centered at (size/2, size/2)."""
    return jnp.array([
        [0.0, 0.0],
        [size, 0.0],
        [size, size],
        [0.0, size],
    ])


def _l_shape() -> jnp.ndarray:
    """L-shaped concave polygon (CCW).

         (0,2)---(1,2)
           |       |
           |  (1,1)---(2,1)
           |           |
         (0,0)-------(2,0)

    Scaled by 100 for numerical comfort.
    """
    return jnp.array([
        [0.0, 0.0],
        [200.0, 0.0],
        [200.0, 100.0],
        [100.0, 100.0],
        [100.0, 200.0],
        [0.0, 200.0],
    ])


# ---------------------------------------------------------------------------
# Test 1: Convex containment parity with old boundary_penalty
# ---------------------------------------------------------------------------

def test_convex_containment_parity():
    """containment_penalty(convex=True) must match the original boundary_penalty."""
    sq = _square(200.0)

    # Points: some inside, some outside
    x = jnp.array([100.0, 250.0, -10.0, 100.0])
    y = jnp.array([100.0, 100.0, 100.0, -5.0])

    old = boundary_penalty(x, y, sq, rho=100.0)
    new = containment_penalty(x, y, sq, convex=True)

    assert jnp.allclose(old, new, atol=1e-12), f"old={old}, new={new}"


# ---------------------------------------------------------------------------
# Test 2: General polygon SDF — L-shaped concave polygon
# ---------------------------------------------------------------------------

def test_concave_polygon_sdf():
    """Point in the notch of an L-shape must be *outside* (positive SDF)."""
    L = _l_shape()

    # Point in the notch (150, 150) — outside the L
    x_notch = jnp.array([150.0])
    y_notch = jnp.array([150.0])
    sdf_notch = polygon_sdf(x_notch, y_notch, L)
    assert float(sdf_notch[0]) > 0, f"Notch point SDF should be positive, got {sdf_notch[0]}"

    # Point clearly inside the L (50, 50)
    x_in = jnp.array([50.0])
    y_in = jnp.array([50.0])
    sdf_in = polygon_sdf(x_in, y_in, L)
    assert float(sdf_in[0]) < 0, f"Interior point SDF should be negative, got {sdf_in[0]}"

    # Point far outside (300, 300)
    x_out = jnp.array([300.0])
    y_out = jnp.array([300.0])
    sdf_out = polygon_sdf(x_out, y_out, L)
    assert float(sdf_out[0]) > 0, f"Exterior point SDF should be positive, got {sdf_out[0]}"


# ---------------------------------------------------------------------------
# Test 3: Exclusion basic — inside penalized, far outside not
# ---------------------------------------------------------------------------

def test_exclusion_basic():
    """Point inside polygon has positive penalty; point far outside has zero."""
    sq = _square(100.0)

    # Inside
    x_in = jnp.array([50.0])
    y_in = jnp.array([50.0])
    pen_in = exclusion_penalty(x_in, y_in, sq, buffer=0.0, convex=True)
    assert float(pen_in) > 0, f"Inside point should have positive exclusion penalty, got {pen_in}"

    # Far outside
    x_out = jnp.array([500.0])
    y_out = jnp.array([500.0])
    pen_out = exclusion_penalty(x_out, y_out, sq, buffer=0.0, convex=True)
    assert float(pen_out) == 0.0, f"Far outside point should have zero penalty, got {pen_out}"


# ---------------------------------------------------------------------------
# Test 4: Exclusion with buffer — point just outside within buffer
# ---------------------------------------------------------------------------

def test_exclusion_with_buffer():
    """Point just outside the boundary (within buffer) has nonzero penalty."""
    sq = _square(100.0)
    buffer = 20.0

    # Point at (110, 50): 10m outside the right edge, within 20m buffer
    x = jnp.array([110.0])
    y = jnp.array([50.0])
    pen = exclusion_penalty(x, y, sq, buffer=buffer, convex=True)
    assert float(pen) > 0, f"Point within buffer should have positive penalty, got {pen}"

    # Point at (150, 50): 50m outside — well beyond buffer
    x_far = jnp.array([150.0])
    y_far = jnp.array([50.0])
    pen_far = exclusion_penalty(x_far, y_far, sq, buffer=buffer, convex=True)
    assert float(pen_far) == 0.0, f"Point beyond buffer should have zero penalty, got {pen_far}"


# ---------------------------------------------------------------------------
# Test 5: Differentiability — jax.grad produces finite values
# ---------------------------------------------------------------------------

def test_differentiability():
    """jax.grad through containment_penalty and exclusion_penalty produces finite values."""
    sq = _square(100.0)

    # Containment: point outside the boundary
    def loss_contain(xy):
        return containment_penalty(xy[:1], xy[1:], sq, convex=True)

    xy_outside = jnp.array([-10.0, 50.0])
    grad_contain = jax.grad(loss_contain)(xy_outside)
    assert jnp.all(jnp.isfinite(grad_contain)), f"containment grad not finite: {grad_contain}"
    assert jnp.any(grad_contain != 0.0), "containment grad should be nonzero for outside point"

    # Exclusion: point inside the boundary (off-center to break symmetry)
    def loss_exclude(xy):
        return exclusion_penalty(xy[:1], xy[1:], sq, buffer=10.0, convex=True)

    xy_inside = jnp.array([30.0, 50.0])
    grad_exclude = jax.grad(loss_exclude)(xy_inside)
    assert jnp.all(jnp.isfinite(grad_exclude)), f"exclusion grad not finite: {grad_exclude}"
    assert jnp.any(grad_exclude != 0.0), "exclusion grad should be nonzero for inside point"

    # General (concave) containment
    L = _l_shape()

    def loss_contain_concave(xy):
        return containment_penalty(xy[:1], xy[1:], L, convex=False)

    xy_notch = jnp.array([150.0, 150.0])  # outside the L
    grad_concave = jax.grad(loss_contain_concave)(xy_notch)
    assert jnp.all(jnp.isfinite(grad_concave)), f"concave containment grad not finite: {grad_concave}"

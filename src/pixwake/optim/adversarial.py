"""Gradient-based adversarial search using implicit differentiation.

This module enables finding neighbor configurations that maximize design regret
for a target wind farm using gradient-based optimization. The key insight is
that we can differentiate through the target farm's layout optimization using
the Implicit Function Theorem, avoiding the need to unroll thousands of SGD
iterations.

The bilevel optimization structure:
    - Outer loop: Gradient descent on neighbor positions to maximize regret
    - Inner loop: Target farm optimizes layout via SGD (differentiable via IFT)

Example:
    >>> searcher = GradientAdversarialSearch(sim, target_boundary, min_spacing, ws, wd)
    >>> result = searcher.search(init_target_x, init_target_y, init_neighbor_x, init_neighbor_y)
    >>> print(f"Max regret: {result.regret:.2f} GWh")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

import jax
import jax.numpy as jnp
from jax import value_and_grad

from pixwake.optim.sgd import (
    SGDSettings,
    sgd_solve_implicit,
)


class AdversarialSearchResult(NamedTuple):
    """Result of gradient-based adversarial search.

    Attributes:
        neighbor_x: Optimized neighbor x positions.
        neighbor_y: Optimized neighbor y positions.
        target_x: Target farm layout after optimization with adversarial neighbors.
        target_y: Target farm layout after optimization with adversarial neighbors.
        liberal_x: Target farm layout without neighbors (liberal case).
        liberal_y: Target farm layout without neighbors (liberal case).
        regret: AEP loss due to neighbors (liberal_aep - conservative_aep).
        liberal_aep: AEP without neighbors.
        conservative_aep: AEP with adversarial neighbors.
        history: List of (regret, neighbor_x, neighbor_y) tuples per iteration.
    """

    neighbor_x: jnp.ndarray
    neighbor_y: jnp.ndarray
    target_x: jnp.ndarray
    target_y: jnp.ndarray
    liberal_x: jnp.ndarray
    liberal_y: jnp.ndarray
    regret: float
    liberal_aep: float
    conservative_aep: float
    history: list


@dataclass
class AdversarialSearchSettings:
    """Settings for gradient-based adversarial search.

    Attributes:
        max_iter: Maximum outer loop iterations (default: 100).
        learning_rate: Learning rate for neighbor position updates (default: 10.0).
        tol: Convergence tolerance on regret change (default: 1e-4).
        neighbor_boundary: Boundary polygon for neighbor positions (optional).
        neighbor_min_spacing: Minimum spacing between neighbor turbines (optional).
        sgd_settings: Settings for inner SGD optimization.
        verbose: Print progress during optimization (default: True).
    """

    max_iter: int = 100
    learning_rate: float = 10.0
    tol: float = 1e-4
    neighbor_boundary: jnp.ndarray | None = None
    neighbor_min_spacing: float | None = None
    sgd_settings: SGDSettings | None = None
    verbose: bool = True


class GradientAdversarialSearch:
    """Gradient-based adversarial neighbor search using implicit differentiation.

    This class finds neighbor turbine configurations that maximize regret for
    a target wind farm. It uses JAX automatic differentiation through the
    target farm's layout optimization via the Implicit Function Theorem.

    The regret is defined as:
        regret = AEP_liberal - AEP_conservative

    where:
        - AEP_liberal: Target farm AEP when optimized without neighbors
        - AEP_conservative: Target farm AEP when optimized with neighbors present

    Parameters
    ----------
    sim : WakeSimulation
        Wake simulation engine.
    target_boundary : jnp.ndarray
        Target farm boundary polygon, shape (n_vertices, 2).
    target_min_spacing : float
        Minimum spacing between target turbines.
    ws_amb : jnp.ndarray
        Ambient wind speeds.
    wd_amb : jnp.ndarray
        Wind directions.
    ti_amb : jnp.ndarray | None
        Ambient turbulence intensity (optional).
    weights : jnp.ndarray | None
        Probability weights for wind conditions (optional).

    Example
    -------
    >>> from pixwake import WakeSimulation, Turbine
    >>> from pixwake.deficit import NOJDeficit
    >>> from pixwake.optim.adversarial import GradientAdversarialSearch
    >>>
    >>> sim = WakeSimulation(turbine, NOJDeficit())
    >>> searcher = GradientAdversarialSearch(
    ...     sim, target_boundary, min_spacing=160.0,
    ...     ws_amb=jnp.array([10.0]), wd_amb=jnp.array([270.0])
    ... )
    >>> result = searcher.search(init_x, init_y, neighbor_x, neighbor_y)
    """

    def __init__(
        self,
        sim,
        target_boundary: jnp.ndarray,
        target_min_spacing: float,
        ws_amb: jnp.ndarray,
        wd_amb: jnp.ndarray,
        ti_amb: jnp.ndarray | None = None,
        weights: jnp.ndarray | None = None,
    ):
        self.sim = sim
        self.target_boundary = target_boundary
        self.target_min_spacing = target_min_spacing
        self.ws_amb = ws_amb
        self.wd_amb = wd_amb
        self.ti_amb = ti_amb
        self.weights = weights

    def _compute_aep(
        self,
        target_x: jnp.ndarray,
        target_y: jnp.ndarray,
        neighbor_x: jnp.ndarray | None = None,
        neighbor_y: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Compute AEP for target turbines, optionally with neighbors."""
        n_target = target_x.shape[0]

        if neighbor_x is not None and neighbor_y is not None:
            x_all = jnp.concatenate([target_x, neighbor_x])
            y_all = jnp.concatenate([target_y, neighbor_y])
        else:
            x_all = target_x
            y_all = target_y

        result = self.sim(
            x_all, y_all, ws_amb=self.ws_amb, wd_amb=self.wd_amb, ti_amb=self.ti_amb
        )

        # Only count target turbines' AEP
        power = result.power()[:, :n_target]  # (n_cases, n_target)

        if self.weights is not None:
            weighted_power = jnp.sum(power * self.weights[:, None])
            return weighted_power * 8760 / 1e6  # GWh
        return jnp.sum(power) * 8760 / 1e6 / power.shape[0]

    def search(
        self,
        init_target_x: jnp.ndarray,
        init_target_y: jnp.ndarray,
        init_neighbor_x: jnp.ndarray,
        init_neighbor_y: jnp.ndarray,
        settings: AdversarialSearchSettings | None = None,
    ) -> AdversarialSearchResult:
        """Run gradient-based adversarial search.

        Parameters
        ----------
        init_target_x, init_target_y : jnp.ndarray
            Initial target farm turbine positions.
        init_neighbor_x, init_neighbor_y : jnp.ndarray
            Initial neighbor turbine positions.
        settings : AdversarialSearchSettings | None
            Search settings. Uses defaults if None.

        Returns
        -------
        AdversarialSearchResult
            Search results including optimized positions and regret.
        """
        if settings is None:
            settings = AdversarialSearchSettings()

        sgd_settings = settings.sgd_settings or SGDSettings()

        # Define objective function for inner SGD (with neighbor params)
        def objective_with_neighbors(
            x: jnp.ndarray, y: jnp.ndarray, neighbor_params: jnp.ndarray
        ) -> jnp.ndarray:
            n_neighbors = neighbor_params.shape[0] // 2
            neighbor_x = neighbor_params[:n_neighbors]
            neighbor_y = neighbor_params[n_neighbors:]
            return -self._compute_aep(x, y, neighbor_x, neighbor_y)

        # Compute liberal layout (no neighbors) - only once
        def liberal_objective(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
            return -self._compute_aep(x, y)

        from pixwake.optim.sgd import topfarm_sgd_solve

        liberal_x, liberal_y = topfarm_sgd_solve(
            liberal_objective,
            init_target_x,
            init_target_y,
            self.target_boundary,
            self.target_min_spacing,
            sgd_settings,
        )
        liberal_aep = float(self._compute_aep(liberal_x, liberal_y))

        if settings.verbose:
            print(f"Liberal AEP (no neighbors): {liberal_aep:.2f} GWh")

        # Define regret function (to maximize)
        def compute_regret(neighbor_params: jnp.ndarray) -> jnp.ndarray:
            """Compute regret = liberal_aep - conservative_aep."""
            # Optimize target layout given neighbors (differentiable via IFT)
            opt_x, opt_y = sgd_solve_implicit(
                objective_with_neighbors,
                init_target_x,
                init_target_y,
                self.target_boundary,
                self.target_min_spacing,
                sgd_settings,
                neighbor_params,
            )

            # Compute conservative AEP
            n_neighbors = neighbor_params.shape[0] // 2
            neighbor_x = neighbor_params[:n_neighbors]
            neighbor_y = neighbor_params[n_neighbors:]
            conservative_aep = self._compute_aep(opt_x, opt_y, neighbor_x, neighbor_y)

            # Regret = liberal - conservative (we want to maximize this)
            return liberal_aep - conservative_aep

        # Gradient ascent on neighbor positions to maximize regret
        neighbor_params = jnp.concatenate([init_neighbor_x, init_neighbor_y])
        history = []

        regret_and_grad = value_and_grad(compute_regret)

        for i in range(settings.max_iter):
            regret, grad = regret_and_grad(neighbor_params)

            history.append(
                (
                    float(regret),
                    neighbor_params[: len(init_neighbor_x)].copy(),
                    neighbor_params[len(init_neighbor_x) :].copy(),
                )
            )

            if settings.verbose and i % 10 == 0:
                print(f"Iter {i}: regret = {regret:.4f} GWh, |grad| = {jnp.linalg.norm(grad):.6f}")

            # Check convergence
            if i > 0 and abs(history[-1][0] - history[-2][0]) < settings.tol:
                if settings.verbose:
                    print(f"Converged at iteration {i}")
                break

            # Gradient ascent step (maximize regret)
            neighbor_params = neighbor_params + settings.learning_rate * grad

            # Apply neighbor boundary constraints if specified
            if settings.neighbor_boundary is not None:
                # Project back into boundary (simple clipping for now)
                n_neighbors = len(init_neighbor_x)
                neighbor_x = neighbor_params[:n_neighbors]
                neighbor_y = neighbor_params[n_neighbors:]

                x_min = settings.neighbor_boundary[:, 0].min()
                x_max = settings.neighbor_boundary[:, 0].max()
                y_min = settings.neighbor_boundary[:, 1].min()
                y_max = settings.neighbor_boundary[:, 1].max()

                neighbor_x = jnp.clip(neighbor_x, x_min, x_max)
                neighbor_y = jnp.clip(neighbor_y, y_min, y_max)
                neighbor_params = jnp.concatenate([neighbor_x, neighbor_y])

        # Final evaluation
        n_neighbors = len(init_neighbor_x)
        final_neighbor_x = neighbor_params[:n_neighbors]
        final_neighbor_y = neighbor_params[n_neighbors:]

        # Get final optimized target layout
        final_target_x, final_target_y = sgd_solve_implicit(
            objective_with_neighbors,
            init_target_x,
            init_target_y,
            self.target_boundary,
            self.target_min_spacing,
            sgd_settings,
            neighbor_params,
        )

        final_conservative_aep = float(
            self._compute_aep(final_target_x, final_target_y, final_neighbor_x, final_neighbor_y)
        )
        final_regret = liberal_aep - final_conservative_aep

        if settings.verbose:
            print(f"\nFinal Results:")
            print(f"  Liberal AEP: {liberal_aep:.2f} GWh")
            print(f"  Conservative AEP: {final_conservative_aep:.2f} GWh")
            print(f"  Regret: {final_regret:.2f} GWh ({100*final_regret/liberal_aep:.1f}%)")

        return AdversarialSearchResult(
            neighbor_x=final_neighbor_x,
            neighbor_y=final_neighbor_y,
            target_x=final_target_x,
            target_y=final_target_y,
            liberal_x=liberal_x,
            liberal_y=liberal_y,
            regret=final_regret,
            liberal_aep=liberal_aep,
            conservative_aep=final_conservative_aep,
            history=history,
        )


# =============================================================================
# Blob-Based Adversarial Discovery
# =============================================================================


class BlobDiscoveryResult(NamedTuple):
    """Result of blob-based adversarial discovery.

    Attributes:
        control_points: Final B-spline control points defining neighbor boundary.
        liberal_x: Liberal layout (optimized without neighbors).
        liberal_y: Liberal layout (optimized without neighbors).
        conservative_x: Conservative layout (optimized with neighbors).
        conservative_y: Conservative layout (optimized with neighbors).
        aep_L_absent: Liberal design AEP without neighbors.
        aep_L_present: Liberal design AEP with neighbors.
        aep_C_absent: Conservative design AEP without neighbors.
        aep_C_present: Conservative design AEP with neighbors.
        regret_liberal: AEP_C_present - AEP_L_present (benefit of conservative when neighbors appear).
        regret_conservative: AEP_L_absent - AEP_C_absent (cost of conservative when neighbors don't appear).
        history: List of (regret_liberal, control_points) tuples per iteration.
    """

    control_points: jnp.ndarray
    liberal_x: jnp.ndarray
    liberal_y: jnp.ndarray
    conservative_x: jnp.ndarray
    conservative_y: jnp.ndarray
    aep_L_absent: float
    aep_L_present: float
    aep_C_absent: float
    aep_C_present: float
    regret_liberal: float
    regret_conservative: float
    history: list


@dataclass
class BlobDiscoverySettings:
    """Settings for blob-based adversarial discovery.

    Attributes:
        max_iter: Maximum outer loop iterations.
        learning_rate: Learning rate for control point updates.
        tol: Convergence tolerance on regret change.
        temperature: Sigmoid sharpness for soft containment (annealed).
        temperature_final: Final temperature after annealing.
        sgd_settings: Settings for inner SGD optimization.
        objective: Which regret to maximize ('liberal' or 'conservative').
        verbose: Print progress during optimization.
    """

    max_iter: int = 100
    learning_rate: float = 50.0
    tol: float = 1e-4
    temperature: float = 1.0
    temperature_final: float = 20.0
    sgd_settings: SGDSettings | None = None
    objective: str = "liberal"  # 'liberal' or 'conservative'
    verbose: bool = True


class BlobAdversarialDiscovery:
    """Discover critical neighbor configurations using morphable blob geometry.

    Uses gradient ascent on regret with respect to B-spline control points
    to find neighbor farm shapes that maximize design tradeoffs.

    The neighbor farm is represented as a soft-packed region within a
    B-spline boundary, allowing gradients to flow through the effective
    number of neighbor turbines.

    Parameters
    ----------
    sim : WakeSimulation
        Wake simulation engine.
    target_boundary : jnp.ndarray
        Target farm boundary polygon.
    target_min_spacing : float
        Minimum spacing between target turbines.
    neighbor_grid : jnp.ndarray
        Dense reference grid for potential neighbor positions.
    ws_amb : jnp.ndarray
        Ambient wind speeds.
    wd_amb : jnp.ndarray
        Wind directions.
    ti_amb : jnp.ndarray | None
        Ambient turbulence intensity.
    weights : jnp.ndarray | None
        Probability weights for wind conditions.
    """

    def __init__(
        self,
        sim,
        target_boundary: jnp.ndarray,
        target_min_spacing: float,
        neighbor_grid: jnp.ndarray,
        ws_amb: jnp.ndarray,
        wd_amb: jnp.ndarray,
        ti_amb: jnp.ndarray | None = None,
        weights: jnp.ndarray | None = None,
    ):
        self.sim = sim
        self.target_boundary = target_boundary
        self.target_min_spacing = target_min_spacing
        self.neighbor_grid = neighbor_grid
        self.ws_amb = ws_amb
        self.wd_amb = wd_amb
        self.ti_amb = ti_amb
        self.weights = weights

    def _compute_aep_absent(
        self,
        target_x: jnp.ndarray,
        target_y: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute AEP for target turbines WITHOUT neighbors."""
        result = self.sim(
            target_x, target_y, ws_amb=self.ws_amb, wd_amb=self.wd_amb, ti_amb=self.ti_amb
        )
        power = result.power()

        if self.weights is not None:
            return jnp.sum(power * self.weights[:, None]) * 8760 / 1e6
        return jnp.sum(power) * 8760 / 1e6 / power.shape[0]

    def _compute_aep_present(
        self,
        target_x: jnp.ndarray,
        target_y: jnp.ndarray,
        neighbor_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute AEP for target turbines WITH soft-masked neighbors.

        Uses soft interpolation between isolated and full-neighbor cases:
            AEP = (1 - f) * AEP_absent + f * AEP_full
        where f = sum(mask) / n_neighbors.
        """
        n_target = target_x.shape[0]

        # AEP without neighbors
        aep_absent = self._compute_aep_absent(target_x, target_y)

        # AEP with all neighbors
        neighbor_x = self.neighbor_grid[:, 0]
        neighbor_y = self.neighbor_grid[:, 1]
        x_all = jnp.concatenate([target_x, neighbor_x])
        y_all = jnp.concatenate([target_y, neighbor_y])

        result_full = self.sim(
            x_all, y_all, ws_amb=self.ws_amb, wd_amb=self.wd_amb, ti_amb=self.ti_amb
        )
        power_full = result_full.power()[:, :n_target]

        if self.weights is not None:
            aep_full = jnp.sum(power_full * self.weights[:, None]) * 8760 / 1e6
        else:
            aep_full = jnp.sum(power_full) * 8760 / 1e6 / power_full.shape[0]

        # Soft interpolation
        n_neighbors = self.neighbor_grid.shape[0]
        effective_fraction = jnp.sum(neighbor_mask) / n_neighbors

        return aep_absent * (1 - effective_fraction) + aep_full * effective_fraction

    def _compute_aep(
        self,
        target_x: jnp.ndarray,
        target_y: jnp.ndarray,
        neighbor_mask: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Compute AEP (dispatches to absent/present based on mask)."""
        if neighbor_mask is None:
            return self._compute_aep_absent(target_x, target_y)
        return self._compute_aep_present(target_x, target_y, neighbor_mask)

    def discover(
        self,
        init_target_x: jnp.ndarray,
        init_target_y: jnp.ndarray,
        init_control_points: jnp.ndarray,
        settings: BlobDiscoverySettings | None = None,
    ) -> BlobDiscoveryResult:
        """Run blob-based adversarial discovery.

        Parameters
        ----------
        init_target_x, init_target_y : jnp.ndarray
            Initial target farm turbine positions.
        init_control_points : jnp.ndarray
            Initial B-spline control points for neighbor boundary.
        settings : BlobDiscoverySettings | None
            Discovery settings.

        Returns
        -------
        BlobDiscoveryResult
            Discovery results including optimized blob and regret.
        """
        from pixwake.optim.geometry import BSplineBoundary
        from pixwake.optim.sgd import topfarm_sgd_solve

        if settings is None:
            settings = BlobDiscoverySettings()

        sgd_settings = settings.sgd_settings or SGDSettings()

        # Compute LIBERAL layout (optimized WITHOUT neighbors)
        def liberal_objective(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
            return -self._compute_aep_absent(x, y)

        liberal_x, liberal_y = topfarm_sgd_solve(
            liberal_objective,
            init_target_x,
            init_target_y,
            self.target_boundary,
            self.target_min_spacing,
            sgd_settings,
        )

        # Compute AEP_L_absent (liberal design, no neighbors) - this is fixed
        aep_L_absent = float(self._compute_aep_absent(liberal_x, liberal_y))

        if settings.verbose:
            print(f"Liberal design AEP (no neighbors): {aep_L_absent:.2f} GWh", flush=True)

        # Define regret function
        def compute_regret(control_points: jnp.ndarray, temperature: float) -> tuple[jnp.ndarray, dict]:
            """Compute liberal regret for given blob configuration.

            Returns regret_liberal = AEP_C_present - AEP_L_present
            (positive when conservative design outperforms liberal under neighbor presence)
            """
            spline = BSplineBoundary(control_points)
            neighbor_mask = spline.contains(self.neighbor_grid, temperature)

            # AEP_L_present: Liberal design with neighbors
            aep_L_present = self._compute_aep_present(liberal_x, liberal_y, neighbor_mask)

            # Compute CONSERVATIVE layout (optimized WITH neighbors)
            # Start from liberal layout for better convergence
            def conservative_objective(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
                return -self._compute_aep_present(x, y, neighbor_mask)

            conservative_x, conservative_y = topfarm_sgd_solve(
                conservative_objective,
                liberal_x,  # Start from liberal layout
                liberal_y,
                self.target_boundary,
                self.target_min_spacing,
                sgd_settings,
            )

            # AEP_C_present: Conservative design with neighbors
            aep_C_present = self._compute_aep_present(conservative_x, conservative_y, neighbor_mask)

            # AEP_C_absent: Conservative design without neighbors
            aep_C_absent = self._compute_aep_absent(conservative_x, conservative_y)

            # Regrets
            regret_liberal = aep_C_present - aep_L_present  # Benefit of conservative when neighbors appear
            regret_conservative = aep_L_absent - aep_C_absent  # Cost of conservative when neighbors don't appear

            details = {
                'aep_L_present': aep_L_present,
                'aep_C_present': aep_C_present,
                'aep_C_absent': aep_C_absent,
                'conservative_x': conservative_x,
                'conservative_y': conservative_y,
                'regret_conservative': regret_conservative,
            }

            return regret_liberal, details

        # Gradient ascent on control points using finite differences
        # (to avoid custom_vjp closure issues with nested differentiation)
        control_points = init_control_points
        history = []

        def compute_gradient_fd(cp: jnp.ndarray, temp: float, eps: float = 100.0) -> tuple[jnp.ndarray, jnp.ndarray, dict]:
            """Compute gradient of liberal regret using finite differences."""
            grad = jnp.zeros_like(cp)
            base_regret, base_details = compute_regret(cp, temp)

            # Finite differences for each control point coordinate
            for i in range(cp.shape[0]):
                for j in range(cp.shape[1]):
                    cp_plus = cp.at[i, j].set(cp[i, j] + eps)
                    cp_minus = cp.at[i, j].set(cp[i, j] - eps)
                    regret_plus, _ = compute_regret(cp_plus, temp)
                    regret_minus, _ = compute_regret(cp_minus, temp)
                    grad = grad.at[i, j].set((regret_plus - regret_minus) / (2 * eps))

            return base_regret, grad, base_details

        for i in range(settings.max_iter):
            # Anneal temperature
            progress = i / max(settings.max_iter - 1, 1)
            temperature = (
                settings.temperature
                + progress * (settings.temperature_final - settings.temperature)
            )

            # Compute regret and gradient using finite differences
            regret, grad, details = compute_gradient_fd(control_points, temperature)

            history.append((float(regret), control_points.copy()))

            if settings.verbose and (i % 10 == 0 or i < 5):
                n_effective = float(
                    BSplineBoundary(control_points).contains(
                        self.neighbor_grid, temperature
                    ).sum()
                )
                step_size = settings.learning_rate * jnp.linalg.norm(grad)
                print(
                    f"Iter {i}: R_lib = {regret:.4f} GWh, "
                    f"R_con = {details['regret_conservative']:.4f} GWh, "
                    f"|grad| = {jnp.linalg.norm(grad):.6f}, "
                    f"n_eff = {n_effective:.1f}",
                    flush=True,
                )

            # Check convergence
            if i > 0 and abs(history[-1][0] - history[-2][0]) < settings.tol:
                if settings.verbose:
                    print(f"Converged at iteration {i}", flush=True)
                break

            # Gradient ascent step (maximize liberal regret)
            control_points = control_points + settings.learning_rate * grad

        # Final evaluation
        final_temperature = settings.temperature_final
        final_regret, final_details = compute_regret(control_points, final_temperature)

        final_mask = BSplineBoundary(control_points).contains(self.neighbor_grid, final_temperature)
        aep_L_present = float(final_details['aep_L_present'])
        aep_C_present = float(final_details['aep_C_present'])
        aep_C_absent = float(final_details['aep_C_absent'])
        regret_liberal = float(final_regret)
        regret_conservative = float(final_details['regret_conservative'])

        if settings.verbose:
            print(f"\n{'='*60}", flush=True)
            print("Final AEP Matrix:", flush=True)
            print(f"{'='*60}", flush=True)
            print(f"                      Neighbors Present | Neighbors Absent", flush=True)
            print(f"  Liberal Design:     {aep_L_present:>10.2f} GWh | {aep_L_absent:>10.2f} GWh", flush=True)
            print(f"  Conservative Design:{aep_C_present:>10.2f} GWh | {aep_C_absent:>10.2f} GWh", flush=True)
            print(f"{'='*60}", flush=True)
            print(f"Liberal Regret (R_lib = AEP_C_pres - AEP_L_pres): {regret_liberal:.2f} GWh", flush=True)
            print(f"Conservative Regret (R_con = AEP_L_abs - AEP_C_abs): {regret_conservative:.2f} GWh", flush=True)
            print(f"Effective neighbors: {float(final_mask.sum()):.1f}", flush=True)

        return BlobDiscoveryResult(
            control_points=control_points,
            liberal_x=liberal_x,
            liberal_y=liberal_y,
            conservative_x=final_details['conservative_x'],
            conservative_y=final_details['conservative_y'],
            aep_L_absent=aep_L_absent,
            aep_L_present=aep_L_present,
            aep_C_absent=aep_C_absent,
            aep_C_present=aep_C_present,
            regret_liberal=regret_liberal,
            regret_conservative=regret_conservative,
            history=history,
        )


# =============================================================================
# Pooled Multi-Start Blob Discovery (Proper Methodology)
# =============================================================================


class PooledBlobDiscoveryResult(NamedTuple):
    """Result of pooled multi-start blob discovery.

    The pooled methodology runs N random starts for both liberal and conservative
    strategies, cross-evaluates all layouts under both scenarios, and computes
    regret against pooled global bests. This ensures regret values reflect true
    fundamental tradeoffs rather than local minima artifacts.

    Attributes:
        control_points: B-spline control points defining neighbor boundary.
        global_best_aep_absent: Best AEP across all layouts when neighbors absent.
        global_best_aep_present: Best AEP across all layouts when neighbors present.
        min_liberal_regret: Minimum liberal regret achievable from the pool.
        min_conservative_regret: Minimum conservative regret achievable from the pool.
        best_liberal_layout: Layout achieving minimum liberal regret (x, y).
        best_conservative_layout: Layout achieving minimum conservative regret (x, y).
        all_layouts: List of all evaluated layouts with their AEPs.
        n_liberal_starts: Number of liberal optimization starts.
        n_conservative_starts: Number of conservative optimization starts.
        same_best_layout: Whether the same layout achieves both global bests.
    """

    control_points: jnp.ndarray
    global_best_aep_absent: float
    global_best_aep_present: float
    min_liberal_regret: float
    min_conservative_regret: float
    best_liberal_layout: tuple[jnp.ndarray, jnp.ndarray]
    best_conservative_layout: tuple[jnp.ndarray, jnp.ndarray]
    all_layouts: list[dict]
    n_liberal_starts: int
    n_conservative_starts: int
    same_best_layout: bool


@dataclass
class PooledBlobDiscoverySettings:
    """Settings for pooled multi-start blob discovery.

    Attributes:
        n_starts: Number of random starts per strategy (liberal/conservative).
        sgd_settings: Settings for inner SGD optimization.
        verbose: Print progress during optimization.
    """

    n_starts: int = 10
    sgd_settings: SGDSettings | None = None
    verbose: bool = True


class PooledBlobDiscovery:
    """Pooled multi-start blob discovery for proper regret estimation.

    This class implements the correct methodology for estimating design regret:
    1. For a given blob configuration, run N multi-start optimizations with
       liberal assumptions (ignoring neighbors).
    2. Run N multi-start optimizations with conservative assumptions
       (accounting for neighbors).
    3. Cross-evaluate ALL layouts from both pools under BOTH scenarios
       (neighbors present and absent).
    4. Compute regret against pooled global bests, ensuring regret values
       reflect true fundamental tradeoffs rather than local minima artifacts.

    Parameters
    ----------
    sim : WakeSimulation
        Wake simulation engine.
    target_boundary : jnp.ndarray
        Target farm boundary polygon.
    target_min_spacing : float
        Minimum spacing between target turbines.
    neighbor_grid : jnp.ndarray
        Dense reference grid for potential neighbor positions.
    ws_amb : jnp.ndarray
        Ambient wind speeds.
    wd_amb : jnp.ndarray
        Wind directions.
    ti_amb : jnp.ndarray | None
        Ambient turbulence intensity.
    weights : jnp.ndarray | None
        Probability weights for wind conditions.
    """

    def __init__(
        self,
        sim,
        target_boundary: jnp.ndarray,
        target_min_spacing: float,
        neighbor_grid: jnp.ndarray,
        ws_amb: jnp.ndarray,
        wd_amb: jnp.ndarray,
        ti_amb: jnp.ndarray | None = None,
        weights: jnp.ndarray | None = None,
    ):
        self.sim = sim
        self.target_boundary = target_boundary
        self.target_min_spacing = target_min_spacing
        self.neighbor_grid = neighbor_grid
        self.ws_amb = ws_amb
        self.wd_amb = wd_amb
        self.ti_amb = ti_amb
        self.weights = weights

    def _compute_aep_absent(
        self,
        target_x: jnp.ndarray,
        target_y: jnp.ndarray,
    ) -> float:
        """Compute AEP for target turbines WITHOUT neighbors."""
        result = self.sim(
            target_x, target_y, ws_amb=self.ws_amb, wd_amb=self.wd_amb, ti_amb=self.ti_amb
        )
        power = result.power()

        if self.weights is not None:
            aep = jnp.sum(power * self.weights[:, None]) * 8760 / 1e6
        else:
            aep = jnp.sum(power) * 8760 / 1e6 / power.shape[0]
        return float(aep)

    def _compute_aep_present(
        self,
        target_x: jnp.ndarray,
        target_y: jnp.ndarray,
        neighbor_mask: jnp.ndarray,
    ) -> float:
        """Compute AEP for target turbines WITH soft-masked neighbors."""
        n_target = target_x.shape[0]

        # AEP without neighbors
        aep_absent = self._compute_aep_absent(target_x, target_y)

        # AEP with all neighbors
        neighbor_x = self.neighbor_grid[:, 0]
        neighbor_y = self.neighbor_grid[:, 1]
        x_all = jnp.concatenate([target_x, neighbor_x])
        y_all = jnp.concatenate([target_y, neighbor_y])

        result_full = self.sim(
            x_all, y_all, ws_amb=self.ws_amb, wd_amb=self.wd_amb, ti_amb=self.ti_amb
        )
        power_full = result_full.power()[:, :n_target]

        if self.weights is not None:
            aep_full = float(jnp.sum(power_full * self.weights[:, None]) * 8760 / 1e6)
        else:
            aep_full = float(jnp.sum(power_full) * 8760 / 1e6 / power_full.shape[0])

        # Soft interpolation
        n_neighbors = self.neighbor_grid.shape[0]
        effective_fraction = float(jnp.sum(neighbor_mask)) / n_neighbors

        return aep_absent * (1 - effective_fraction) + aep_full * effective_fraction

    def _generate_random_init(
        self,
        key: jnp.ndarray,
        n_turbines: int,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Generate random initial positions within boundary."""
        x_min = float(self.target_boundary[:, 0].min())
        x_max = float(self.target_boundary[:, 0].max())
        y_min = float(self.target_boundary[:, 1].min())
        y_max = float(self.target_boundary[:, 1].max())

        margin = self.target_min_spacing / 2
        key1, key2 = jax.random.split(key)
        x = jax.random.uniform(key1, (n_turbines,), minval=x_min + margin, maxval=x_max - margin)
        y = jax.random.uniform(key2, (n_turbines,), minval=y_min + margin, maxval=y_max - margin)
        return x, y

    def discover(
        self,
        init_target_x: jnp.ndarray,
        init_target_y: jnp.ndarray,
        control_points: jnp.ndarray,
        settings: PooledBlobDiscoverySettings | None = None,
        seed: int = 42,
    ) -> PooledBlobDiscoveryResult:
        """Run pooled multi-start discovery for given blob configuration.

        Parameters
        ----------
        init_target_x, init_target_y : jnp.ndarray
            Base target farm turbine positions (used as one of the starts).
        control_points : jnp.ndarray
            B-spline control points for neighbor boundary.
        settings : PooledBlobDiscoverySettings | None
            Discovery settings.
        seed : int
            Random seed for generating random starts.

        Returns
        -------
        PooledBlobDiscoveryResult
            Discovery results with pooled regret values.
        """
        from pixwake.optim.geometry import BSplineBoundary
        from pixwake.optim.sgd import topfarm_sgd_solve

        if settings is None:
            settings = PooledBlobDiscoverySettings()

        sgd_settings = settings.sgd_settings or SGDSettings()

        # Compute neighbor mask for this blob
        spline = BSplineBoundary(control_points)
        neighbor_mask = spline.contains(self.neighbor_grid, temperature=10.0)

        n_turbines = len(init_target_x)
        key = jax.random.PRNGKey(seed)

        # Define objective functions
        def liberal_objective(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
            """Objective for liberal optimization (no neighbors)."""
            result = self.sim(x, y, ws_amb=self.ws_amb, wd_amb=self.wd_amb, ti_amb=self.ti_amb)
            power = result.power()
            if self.weights is not None:
                aep = jnp.sum(power * self.weights[:, None]) * 8760 / 1e6
            else:
                aep = jnp.sum(power) * 8760 / 1e6 / power.shape[0]
            return -aep

        def conservative_objective(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
            """Objective for conservative optimization (with neighbors)."""
            n_target = x.shape[0]
            neighbor_x = self.neighbor_grid[:, 0]
            neighbor_y = self.neighbor_grid[:, 1]
            x_all = jnp.concatenate([x, neighbor_x])
            y_all = jnp.concatenate([y, neighbor_y])

            result = self.sim(x_all, y_all, ws_amb=self.ws_amb, wd_amb=self.wd_amb, ti_amb=self.ti_amb)
            power_full = result.power()[:, :n_target]

            # AEP with soft-interpolated neighbors
            result_isolated = self.sim(x, y, ws_amb=self.ws_amb, wd_amb=self.wd_amb, ti_amb=self.ti_amb)
            power_isolated = result_isolated.power()

            if self.weights is not None:
                aep_full = jnp.sum(power_full * self.weights[:, None]) * 8760 / 1e6
                aep_isolated = jnp.sum(power_isolated * self.weights[:, None]) * 8760 / 1e6
            else:
                aep_full = jnp.sum(power_full) * 8760 / 1e6 / power_full.shape[0]
                aep_isolated = jnp.sum(power_isolated) * 8760 / 1e6 / power_isolated.shape[0]

            n_neighbors = self.neighbor_grid.shape[0]
            effective_fraction = jnp.sum(neighbor_mask) / n_neighbors
            aep = aep_isolated * (1 - effective_fraction) + aep_full * effective_fraction
            return -aep

        all_layouts = []

        if settings.verbose:
            print(f"Running pooled multi-start optimization ({settings.n_starts} starts per strategy)")
            print(f"  Effective neighbors in blob: {float(neighbor_mask.sum()):.1f}")

        # Run liberal optimizations
        if settings.verbose:
            print(f"\nLiberal optimizations (ignoring neighbors):")
        for i in range(settings.n_starts):
            if i == 0:
                start_x, start_y = init_target_x, init_target_y
            else:
                key, subkey = jax.random.split(key)
                start_x, start_y = self._generate_random_init(subkey, n_turbines)

            opt_x, opt_y = topfarm_sgd_solve(
                liberal_objective,
                start_x,
                start_y,
                self.target_boundary,
                self.target_min_spacing,
                sgd_settings,
            )

            # Cross-evaluate under both scenarios
            aep_absent = self._compute_aep_absent(opt_x, opt_y)
            aep_present = self._compute_aep_present(opt_x, opt_y, neighbor_mask)

            all_layouts.append({
                'strategy': 'liberal',
                'start_idx': i,
                'x': opt_x,
                'y': opt_y,
                'aep_absent': aep_absent,
                'aep_present': aep_present,
            })

            if settings.verbose:
                print(f"  Start {i}: AEP_absent={aep_absent:.2f}, AEP_present={aep_present:.2f} GWh")

        # Run conservative optimizations
        if settings.verbose:
            print(f"\nConservative optimizations (accounting for neighbors):")
        for i in range(settings.n_starts):
            if i == 0:
                start_x, start_y = init_target_x, init_target_y
            else:
                key, subkey = jax.random.split(key)
                start_x, start_y = self._generate_random_init(subkey, n_turbines)

            opt_x, opt_y = topfarm_sgd_solve(
                conservative_objective,
                start_x,
                start_y,
                self.target_boundary,
                self.target_min_spacing,
                sgd_settings,
            )

            # Cross-evaluate under both scenarios
            aep_absent = self._compute_aep_absent(opt_x, opt_y)
            aep_present = self._compute_aep_present(opt_x, opt_y, neighbor_mask)

            all_layouts.append({
                'strategy': 'conservative',
                'start_idx': i,
                'x': opt_x,
                'y': opt_y,
                'aep_absent': aep_absent,
                'aep_present': aep_present,
            })

            if settings.verbose:
                print(f"  Start {i}: AEP_absent={aep_absent:.2f}, AEP_present={aep_present:.2f} GWh")

        # Compute pooled global bests
        global_best_aep_absent = max(l['aep_absent'] for l in all_layouts)
        global_best_aep_present = max(l['aep_present'] for l in all_layouts)

        # Compute regrets for each layout
        for layout in all_layouts:
            # Liberal regret: how much worse than best when neighbors appear
            layout['liberal_regret'] = global_best_aep_present - layout['aep_present']
            # Conservative regret: how much worse than best when neighbors absent
            layout['conservative_regret'] = global_best_aep_absent - layout['aep_absent']

        # Find minimum regrets
        min_liberal_regret = min(l['liberal_regret'] for l in all_layouts)
        min_conservative_regret = min(l['conservative_regret'] for l in all_layouts)

        # Find layouts achieving minimum regrets
        best_liberal_layout = min(all_layouts, key=lambda l: l['liberal_regret'])
        best_conservative_layout = min(all_layouts, key=lambda l: l['conservative_regret'])

        # Check if same layout achieves both bests
        best_absent_layout = max(all_layouts, key=lambda l: l['aep_absent'])
        best_present_layout = max(all_layouts, key=lambda l: l['aep_present'])
        same_best = (
            best_absent_layout['strategy'] == best_present_layout['strategy'] and
            best_absent_layout['start_idx'] == best_present_layout['start_idx']
        )

        if settings.verbose:
            print(f"\n{'='*60}")
            print("Pooled Results:")
            print(f"{'='*60}")
            print(f"Global best AEP (absent):  {global_best_aep_absent:.2f} GWh")
            print(f"Global best AEP (present): {global_best_aep_present:.2f} GWh")
            print(f"Min liberal regret:        {min_liberal_regret:.2f} GWh")
            print(f"Min conservative regret:   {min_conservative_regret:.2f} GWh")
            print(f"Same layout achieves both: {same_best}")
            if not same_best:
                print("  -> TRUE TRADEOFF EXISTS")
            else:
                print("  -> No fundamental tradeoff (single layout dominates)")

        # Convert layouts for serialization
        serializable_layouts = []
        for l in all_layouts:
            serializable_layouts.append({
                'strategy': l['strategy'],
                'start_idx': l['start_idx'],
                'x': l['x'].tolist() if hasattr(l['x'], 'tolist') else list(l['x']),
                'y': l['y'].tolist() if hasattr(l['y'], 'tolist') else list(l['y']),
                'aep_absent': l['aep_absent'],
                'aep_present': l['aep_present'],
                'liberal_regret': l['liberal_regret'],
                'conservative_regret': l['conservative_regret'],
            })

        return PooledBlobDiscoveryResult(
            control_points=control_points,
            global_best_aep_absent=global_best_aep_absent,
            global_best_aep_present=global_best_aep_present,
            min_liberal_regret=min_liberal_regret,
            min_conservative_regret=min_conservative_regret,
            best_liberal_layout=(best_liberal_layout['x'], best_liberal_layout['y']),
            best_conservative_layout=(best_conservative_layout['x'], best_conservative_layout['y']),
            all_layouts=serializable_layouts,
            n_liberal_starts=settings.n_starts,
            n_conservative_starts=settings.n_starts,
            same_best_layout=same_best,
        )


# =============================================================================
# CMA-ES Bilevel Adversarial Search
# =============================================================================


class CMAAdversarialSearchResult(NamedTuple):
    """Result of CMA-ES bilevel adversarial search.

    Attributes:
        best_phi: Best 4D design vector [bearing, distance, rotation, aspect_ratio].
        best_control_points: B-spline control points for best blob.
        best_objective: Best irreducible tradeoff (GWh).
        best_inner_result: PooledBlobDiscoveryResult for best coarse evaluation.
        coarse_history: List of (phi, objective) from all CMA-ES evaluations.
        refined_result: PooledBlobDiscoveryResult from full-fidelity re-evaluation, or None.
    """

    best_phi: np.ndarray
    best_control_points: jnp.ndarray
    best_objective: float
    best_inner_result: PooledBlobDiscoveryResult
    coarse_history: list[tuple[np.ndarray, float]]
    refined_result: PooledBlobDiscoveryResult | None


@dataclass
class CMAAdversarialSearchSettings:
    """Settings for CMA-ES bilevel adversarial search.

    Attributes:
        phi_bounds: Box bounds for design vector as ((lo, lo, lo, lo), (hi, hi, hi, hi)).
        fixed_area: Fixed blob area in m^2.
        target_center: (x, y) center of the target farm.
        n_control: Number of B-spline control points.
        cma_sigma0: Initial CMA-ES step size.
        cma_popsize: CMA-ES population size (None for default).
        cma_maxiter: Maximum CMA-ES iterations.
        cma_seed: CMA-ES random seed.
        coarse_n_starts: Number of starts per strategy in coarse inner loop.
        coarse_sgd_settings: SGD settings for coarse inner loop.
        refine: Whether to run full-fidelity refinement at best_phi.
        refine_n_starts: Number of starts per strategy in refinement.
        refine_sgd_settings: SGD settings for refinement.
        neighbor_grid_extent: Half-extent of neighbor reference grid.
        neighbor_grid_spacing: Spacing of neighbor reference grid.
        verbose: Print progress.
    """

    phi_bounds: tuple[tuple[float, ...], tuple[float, ...]] = (
        (0.0, 2000.0, 0.0, 1.0),
        (2 * np.pi, 10000.0, 2 * np.pi, 3.0),
    )
    fixed_area: float = 50.0 * 200.0 ** 2
    target_center: tuple[float, float] = (1600.0, 1600.0)
    n_control: int = 8
    cma_sigma0: float = 0.5
    cma_popsize: int | None = None
    cma_maxiter: int = 50
    cma_seed: int = 42
    coarse_n_starts: int = 5
    coarse_sgd_settings: SGDSettings | None = None
    refine: bool = True
    refine_n_starts: int = 20
    refine_sgd_settings: SGDSettings | None = None
    neighbor_grid_extent: float = 1200.0
    neighbor_grid_spacing: float = 600.0
    verbose: bool = True


class CMAAdversarialSearch:
    """CMA-ES bilevel adversarial search for worst-case neighbor blobs.

    Uses CMA-ES to optimize over a 4D blob parameterization
    (bearing, distance, rotation, aspect_ratio) to find the neighbor
    configuration that maximizes the irreducible design tradeoff.
    The inner loop reuses PooledBlobDiscovery.discover() unchanged.

    Parameters
    ----------
    sim : WakeSimulation
        Wake simulation engine.
    target_boundary : jnp.ndarray
        Target farm boundary polygon, shape (n_vertices, 2).
    target_min_spacing : float
        Minimum spacing between target turbines.
    ws_amb : jnp.ndarray
        Ambient wind speeds.
    wd_amb : jnp.ndarray
        Wind directions.
    ti_amb : jnp.ndarray | None
        Ambient turbulence intensity.
    weights : jnp.ndarray | None
        Probability weights for wind conditions.
    """

    def __init__(
        self,
        sim,
        target_boundary: jnp.ndarray,
        target_min_spacing: float,
        ws_amb: jnp.ndarray,
        wd_amb: jnp.ndarray,
        ti_amb: jnp.ndarray | None = None,
        weights: jnp.ndarray | None = None,
    ):
        self.sim = sim
        self.target_boundary = target_boundary
        self.target_min_spacing = target_min_spacing
        self.ws_amb = ws_amb
        self.wd_amb = wd_amb
        self.ti_amb = ti_amb
        self.weights = weights

    @staticmethod
    def _extract_irreducible_tradeoff(result: PooledBlobDiscoveryResult) -> float:
        """Extract the irreducible tradeoff from an inner loop result.

        The irreducible tradeoff is:
            conservative_best_present - liberal_best_present

        where liberal_best is the layout maximizing AEP absent and
        conservative_best is the layout maximizing AEP present.
        A positive value means the liberal design suffers under neighbor presence.

        CMA-ES minimizes, so we return the negative.
        """
        all_layouts = result.all_layouts
        liberal_best = max(all_layouts, key=lambda l: l['aep_absent'])
        conservative_best = max(all_layouts, key=lambda l: l['aep_present'])
        tradeoff = conservative_best['aep_present'] - liberal_best['aep_present']
        return -tradeoff  # CMA-ES minimizes

    def search(
        self,
        init_target_x: jnp.ndarray,
        init_target_y: jnp.ndarray,
        settings: CMAAdversarialSearchSettings | None = None,
    ) -> CMAAdversarialSearchResult:
        """Run CMA-ES bilevel adversarial search.

        Parameters
        ----------
        init_target_x, init_target_y : jnp.ndarray
            Initial target farm turbine positions.
        settings : CMAAdversarialSearchSettings | None
            Search settings. Uses defaults if None.

        Returns
        -------
        CMAAdversarialSearchResult
            Search results.
        """
        import cma

        from pixwake.optim.geometry import phi_to_control_points
        from pixwake.optim.soft_packing import create_reference_grid

        if settings is None:
            settings = CMAAdversarialSearchSettings()

        # Coarse inner loop settings
        coarse_sgd = settings.coarse_sgd_settings or SGDSettings(
            max_iter=500, learning_rate=40.0,
        )
        coarse_pool_settings = PooledBlobDiscoverySettings(
            n_starts=settings.coarse_n_starts,
            sgd_settings=coarse_sgd,
            verbose=False,
        )

        # CMA-ES setup
        lo = np.array(settings.phi_bounds[0])
        hi = np.array(settings.phi_bounds[1])
        x0 = (lo + hi) / 2.0

        # Scale sigma0 to the normalized [0, 1] space
        opts = {
            'bounds': [list(lo), list(hi)],
            'maxiter': settings.cma_maxiter,
            'seed': settings.cma_seed,
            'verbose': -9,  # suppress CMA-ES internal output
        }
        if settings.cma_popsize is not None:
            opts['popsize'] = settings.cma_popsize

        es = cma.CMAEvolutionStrategy(x0.tolist(), settings.cma_sigma0, opts)

        # Track best
        best_phi = None
        best_objective = float('inf')  # CMA-ES minimizes
        best_control_points = None
        best_inner_result = None
        coarse_history = []
        eval_count = 0

        if settings.verbose:
            print("=" * 60)
            print("CMA-ES Bilevel Adversarial Search")
            print("=" * 60)
            print(f"  phi_bounds: {settings.phi_bounds}")
            print(f"  fixed_area: {settings.fixed_area:.0f} m^2")
            print(f"  cma_sigma0: {settings.cma_sigma0}, maxiter: {settings.cma_maxiter}")
            print(f"  coarse inner: {settings.coarse_n_starts} starts, "
                  f"{coarse_sgd.max_iter} SGD iters")
            print("=" * 60, flush=True)

        generation = 0
        while not es.stop():
            candidates = es.ask()
            fitnesses = []

            for phi_list in candidates:
                phi = np.array(phi_list)

                # Map to control points
                cp = phi_to_control_points(
                    phi,
                    target_center=settings.target_center,
                    fixed_area=settings.fixed_area,
                    n_control=settings.n_control,
                )

                # Build neighbor grid centered on this blob
                bearing, distance, rotation, aspect_ratio = phi
                blob_cx = settings.target_center[0] + distance * np.sin(bearing)
                blob_cy = settings.target_center[1] + distance * np.cos(bearing)
                neighbor_grid = create_reference_grid(
                    center=(float(blob_cx), float(blob_cy)),
                    extent=settings.neighbor_grid_extent,
                    spacing=settings.neighbor_grid_spacing,
                )

                # Run coarse inner loop
                discoverer = PooledBlobDiscovery(
                    sim=self.sim,
                    target_boundary=self.target_boundary,
                    target_min_spacing=self.target_min_spacing,
                    neighbor_grid=neighbor_grid,
                    ws_amb=self.ws_amb,
                    wd_amb=self.wd_amb,
                    ti_amb=self.ti_amb,
                    weights=self.weights,
                )

                inner_result = discoverer.discover(
                    init_target_x,
                    init_target_y,
                    cp,
                    settings=coarse_pool_settings,
                    seed=settings.cma_seed + eval_count,
                )

                obj = self._extract_irreducible_tradeoff(inner_result)
                fitnesses.append(obj)
                coarse_history.append((phi.copy(), float(-obj)))  # store positive tradeoff

                # Track best
                if obj < best_objective:
                    best_objective = obj
                    best_phi = phi.copy()
                    best_control_points = cp
                    best_inner_result = inner_result

                eval_count += 1

            es.tell(candidates, fitnesses)

            if settings.verbose:
                best_so_far = -best_objective
                gen_best = -min(fitnesses)
                print(
                    f"Gen {generation}: best_this_gen={gen_best:.3f} GWh, "
                    f"best_overall={best_so_far:.3f} GWh, evals={eval_count}",
                    flush=True,
                )
            generation += 1

        if settings.verbose:
            print(f"\nCMA-ES finished after {eval_count} evaluations")
            print(f"Best tradeoff: {-best_objective:.3f} GWh")
            print(f"Best phi: bearing={best_phi[0]:.3f} rad, dist={best_phi[1]:.0f} m, "
                  f"rot={best_phi[2]:.3f} rad, alpha={best_phi[3]:.2f}")

        # Refinement
        refined_result = None
        if settings.refine and best_phi is not None:
            if settings.verbose:
                print("\nRunning refinement at best_phi...", flush=True)

            refine_sgd = settings.refine_sgd_settings or SGDSettings(
                max_iter=3000, learning_rate=40.0,
            )
            refine_pool_settings = PooledBlobDiscoverySettings(
                n_starts=settings.refine_n_starts,
                sgd_settings=refine_sgd,
                verbose=settings.verbose,
            )

            # Rebuild grid and control points for best phi
            bearing, distance, rotation, aspect_ratio = best_phi
            blob_cx = settings.target_center[0] + distance * np.sin(bearing)
            blob_cy = settings.target_center[1] + distance * np.cos(bearing)
            neighbor_grid = create_reference_grid(
                center=(float(blob_cx), float(blob_cy)),
                extent=settings.neighbor_grid_extent,
                spacing=settings.neighbor_grid_spacing,
            )

            discoverer = PooledBlobDiscovery(
                sim=self.sim,
                target_boundary=self.target_boundary,
                target_min_spacing=self.target_min_spacing,
                neighbor_grid=neighbor_grid,
                ws_amb=self.ws_amb,
                wd_amb=self.wd_amb,
                ti_amb=self.ti_amb,
                weights=self.weights,
            )

            refined_result = discoverer.discover(
                init_target_x,
                init_target_y,
                best_control_points,
                settings=refine_pool_settings,
                seed=settings.cma_seed + 999999,
            )

            refined_tradeoff = -self._extract_irreducible_tradeoff(refined_result)
            if settings.verbose:
                print(f"Refined tradeoff: {refined_tradeoff:.3f} GWh")

        return CMAAdversarialSearchResult(
            best_phi=best_phi,
            best_control_points=best_control_points,
            best_objective=float(-best_objective),
            best_inner_result=best_inner_result,
            coarse_history=coarse_history,
            refined_result=refined_result,
        )

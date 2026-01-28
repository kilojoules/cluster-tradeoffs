"""
Adversarial Neighbor Optimization for Wind Farm Layout.

This module implements optimization schemes to find neighbor configurations
that maximize regret for a target wind farm.

Three optimization approaches are available:

1. **CMA-ES** (AdversarialNeighborOptimizer):
   - Optimizes cluster parameters (spline + density)
   - Variable turbine count via floor(area × density)
   - Gradient-free, handles discrete decisions

2. **Direct Gradient** (GradientAdversarialOptimizer):
   - Optimizes raw (x, y) positions for N turbines
   - Fully differentiable via JAX
   - No shape constraint

3. **Spline Gradient** (DifferentiableSplineOptimizer):
   - Optimizes spline parameters with fixed N turbines
   - Fully differentiable via JAX
   - Maintains cluster geometry

See docs/adversarial_optimization.md for detailed documentation.

Key components:
- SplineCluster: Spline-bounded cluster parameterization
- TurbinePacker: Hexagonal packing of turbines within boundaries
- evaluate_closed_bspline: JAX-differentiable B-spline evaluation
- differentiable_spline_positions: Spline params → turbine positions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.spatial import ConvexHull


# =============================================================================
# Spline Cluster Parameterization
# =============================================================================


def shoelace_area(vertices: np.ndarray) -> float:
    """Compute polygon area using shoelace formula.

    Parameters
    ----------
    vertices : np.ndarray
        Shape (N, 2) array of polygon vertices in order.

    Returns
    -------
    float
        Absolute area of the polygon.
    """
    n = len(vertices)
    if n < 3:
        return 0.0
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]) + x[-1] * y[0] - x[0] * y[-1])


def point_in_polygon(point: np.ndarray, vertices: np.ndarray) -> bool:
    """Check if point is inside polygon using ray casting.

    Parameters
    ----------
    point : np.ndarray
        Shape (2,) array [x, y].
    vertices : np.ndarray
        Shape (N, 2) array of polygon vertices.

    Returns
    -------
    bool
        True if point is inside polygon.
    """
    x, y = point
    n = len(vertices)
    inside = False

    j = n - 1
    for i in range(n):
        xi, yi = vertices[i]
        xj, yj = vertices[j]

        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i

    return inside


def points_in_polygon(points: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Vectorized point-in-polygon test.

    Parameters
    ----------
    points : np.ndarray
        Shape (M, 2) array of points to test.
    vertices : np.ndarray
        Shape (N, 2) array of polygon vertices.

    Returns
    -------
    np.ndarray
        Boolean array of shape (M,).
    """
    return np.array([point_in_polygon(p, vertices) for p in points])


class SplineCluster:
    """Spline-bounded cluster for neighbor wind farm representation.

    The cluster boundary is defined by a closed B-spline through K control
    points. This allows flexible, smooth shapes including convex and
    concave configurations.

    Parameters
    ----------
    control_points : np.ndarray
        Shape (K, 2) array of control points relative to center.
    center : tuple[float, float]
        Cluster center position (cx, cy).
    rotation : float
        Rotation angle in radians.
    n_boundary_samples : int
        Number of points to sample along the spline boundary.
    """

    def __init__(
        self,
        control_points: np.ndarray,
        center: tuple[float, float] = (0.0, 0.0),
        rotation: float = 0.0,
        n_boundary_samples: int = 100,
    ):
        self.control_points_raw = np.asarray(control_points)
        self.center = np.asarray(center)
        self.rotation = rotation
        self.n_boundary_samples = n_boundary_samples

        # Compute transformed boundary
        self._boundary = self._compute_boundary()
        self._area = shoelace_area(self._boundary)

    def _compute_boundary(self) -> np.ndarray:
        """Compute the spline boundary with rotation and translation."""
        pts = self.control_points_raw.copy()

        # Apply rotation
        if self.rotation != 0:
            cos_r = np.cos(self.rotation)
            sin_r = np.sin(self.rotation)
            rotation_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
            pts = pts @ rotation_matrix.T

        # Fit closed B-spline
        # Close the curve by appending first few points
        pts_closed = np.vstack([pts, pts[:3]])

        try:
            tck, _ = splprep([pts_closed[:, 0], pts_closed[:, 1]], s=0, per=True, k=3)
            u_new = np.linspace(0, 1, self.n_boundary_samples, endpoint=False)
            x, y = splev(u_new, tck)
            boundary = np.column_stack([x, y])
        except Exception:
            # Fall back to convex hull if spline fails
            if len(pts) >= 3:
                try:
                    hull = ConvexHull(pts)
                    boundary = pts[hull.vertices]
                except Exception:
                    boundary = pts
            else:
                boundary = pts

        # Translate to center
        boundary = boundary + self.center

        return boundary

    @property
    def boundary(self) -> np.ndarray:
        """Get the boundary polygon vertices."""
        return self._boundary

    @property
    def area(self) -> float:
        """Get the cluster area in square units."""
        return self._area

    def contains(self, points: np.ndarray) -> np.ndarray:
        """Check which points are inside the cluster boundary.

        Parameters
        ----------
        points : np.ndarray
            Shape (M, 2) array of points to test.

        Returns
        -------
        np.ndarray
            Boolean array of shape (M,).
        """
        return points_in_polygon(points, self._boundary)

    def get_bounding_box(self) -> tuple[float, float, float, float]:
        """Get axis-aligned bounding box.

        Returns
        -------
        tuple
            (x_min, y_min, x_max, y_max)
        """
        x_min = self._boundary[:, 0].min()
        x_max = self._boundary[:, 0].max()
        y_min = self._boundary[:, 1].min()
        y_max = self._boundary[:, 1].max()
        return x_min, y_min, x_max, y_max

    @classmethod
    def from_ellipse(
        cls,
        center: tuple[float, float],
        semi_major: float,
        semi_minor: float,
        rotation: float = 0.0,
        n_control_points: int = 8,
    ) -> "SplineCluster":
        """Create cluster from ellipse parameters.

        Parameters
        ----------
        center : tuple[float, float]
            Center position.
        semi_major : float
            Semi-major axis length.
        semi_minor : float
            Semi-minor axis length.
        rotation : float
            Rotation angle in radians.
        n_control_points : int
            Number of control points around the ellipse.

        Returns
        -------
        SplineCluster
            Cluster with elliptical boundary.
        """
        angles = np.linspace(0, 2 * np.pi, n_control_points, endpoint=False)
        control_points = np.column_stack([
            semi_major * np.cos(angles),
            semi_minor * np.sin(angles),
        ])
        return cls(control_points, center=center, rotation=rotation)

    @classmethod
    def from_rectangle(
        cls,
        center: tuple[float, float],
        width: float,
        height: float,
        rotation: float = 0.0,
    ) -> "SplineCluster":
        """Create cluster from rectangle parameters.

        Parameters
        ----------
        center : tuple[float, float]
            Center position.
        width : float
            Rectangle width.
        height : float
            Rectangle height.
        rotation : float
            Rotation angle in radians.

        Returns
        -------
        SplineCluster
            Cluster with approximately rectangular boundary.
        """
        hw, hh = width / 2, height / 2
        control_points = np.array([
            [-hw, -hh],
            [0, -hh],
            [hw, -hh],
            [hw, 0],
            [hw, hh],
            [0, hh],
            [-hw, hh],
            [-hw, 0],
        ])
        return cls(control_points, center=center, rotation=rotation)


# =============================================================================
# Turbine Packing
# =============================================================================


class TurbinePacker:
    """Pack turbines within a cluster boundary using hexagonal grid.

    Parameters
    ----------
    min_spacing : float
        Minimum spacing between turbines.
    """

    def __init__(self, min_spacing: float):
        self.min_spacing = min_spacing

    def hexagonal_grid(
        self, bounds: tuple[float, float, float, float], spacing: float
    ) -> np.ndarray:
        """Generate hexagonal grid points within bounding box.

        Parameters
        ----------
        bounds : tuple
            (x_min, y_min, x_max, y_max)
        spacing : float
            Distance between adjacent turbines.

        Returns
        -------
        np.ndarray
            Shape (N, 2) array of grid points.
        """
        x_min, y_min, x_max, y_max = bounds
        dx = spacing
        dy = spacing * np.sqrt(3) / 2

        points = []
        row = 0
        y = y_min
        while y <= y_max:
            x_offset = (row % 2) * dx / 2
            x = x_min + x_offset
            while x <= x_max:
                points.append([x, y])
                x += dx
            y += dy
            row += 1

        return np.array(points) if points else np.empty((0, 2))

    def pack_turbines(
        self, cluster: SplineCluster, n_turbines: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pack specified number of turbines inside cluster boundary.

        Uses hexagonal grid and filters to points inside the boundary.
        If more points are available than needed, selects points closest
        to center to maintain compact arrangement.

        Parameters
        ----------
        cluster : SplineCluster
            The cluster boundary.
        n_turbines : int
            Target number of turbines.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (x_positions, y_positions) arrays.
        """
        if n_turbines <= 0:
            return np.array([]), np.array([])

        bounds = cluster.get_bounding_box()

        # Generate hexagonal grid
        grid_points = self.hexagonal_grid(bounds, self.min_spacing)

        if len(grid_points) == 0:
            return np.array([]), np.array([])

        # Filter to points inside boundary
        inside_mask = cluster.contains(grid_points)
        inside_points = grid_points[inside_mask]

        if len(inside_points) == 0:
            return np.array([]), np.array([])

        # If we have more points than needed, select closest to center
        if len(inside_points) > n_turbines:
            center = cluster.center
            distances = np.sqrt(np.sum((inside_points - center) ** 2, axis=1))
            indices = np.argsort(distances)[:n_turbines]
            selected = inside_points[indices]
        else:
            selected = inside_points

        return selected[:, 0], selected[:, 1]

    def pack_by_density(
        self, cluster: SplineCluster, density: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pack turbines based on density specification.

        Parameters
        ----------
        cluster : SplineCluster
            The cluster boundary.
        density : float
            Turbines per unit area (e.g., turbines per km²).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (x_positions, y_positions) arrays.
        """
        n_turbines = int(cluster.area * density)
        return self.pack_turbines(cluster, n_turbines)


# =============================================================================
# Configuration Dataclasses
# =============================================================================


@dataclass
class ClusterParams:
    """Parameters defining a single neighbor cluster.

    Attributes
    ----------
    center : tuple[float, float]
        Cluster center position (cx, cy).
    control_points : np.ndarray
        Shape (K, 2) control points relative to center.
    density : float
        Turbines per unit area.
    rotation : float
        Rotation angle in radians.
    """

    center: tuple[float, float]
    control_points: np.ndarray
    density: float
    rotation: float = 0.0

    def to_cluster(self) -> SplineCluster:
        """Convert to SplineCluster instance."""
        return SplineCluster(
            control_points=self.control_points,
            center=self.center,
            rotation=self.rotation,
        )

    def n_params(self) -> int:
        """Get total number of scalar parameters."""
        # 2 for center + 2K for control points + 1 for density + 1 for rotation
        return 2 + 2 * len(self.control_points) + 2


@dataclass
class NeighborConfig:
    """Configuration of multiple neighbor clusters.

    Attributes
    ----------
    clusters : list[ClusterParams]
        List of cluster parameter sets.
    """

    clusters: list[ClusterParams] = field(default_factory=list)

    def to_turbines(
        self, packer: TurbinePacker
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert all clusters to turbine positions.

        Parameters
        ----------
        packer : TurbinePacker
            Packer instance for turbine placement.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Combined (x_positions, y_positions) from all clusters.
        """
        all_x, all_y = [], []

        for cluster_params in self.clusters:
            cluster = cluster_params.to_cluster()
            x, y = packer.pack_by_density(cluster, cluster_params.density)
            if len(x) > 0:
                all_x.extend(x)
                all_y.extend(y)

        return np.array(all_x), np.array(all_y)

    def total_params(self) -> int:
        """Get total number of scalar parameters across all clusters."""
        return sum(c.n_params() for c in self.clusters)


# =============================================================================
# Parameter Flattening/Unflattening
# =============================================================================


def flatten_config(config: NeighborConfig, n_control_points: int) -> np.ndarray:
    """Flatten NeighborConfig to parameter vector.

    Parameters
    ----------
    config : NeighborConfig
        Configuration to flatten.
    n_control_points : int
        Number of control points per cluster (must be consistent).

    Returns
    -------
    np.ndarray
        Flat parameter vector.
    """
    params = []
    for cluster in config.clusters:
        params.extend([cluster.center[0], cluster.center[1]])
        params.extend(cluster.control_points.flatten())
        params.append(cluster.density)
        params.append(cluster.rotation)
    return np.array(params)


def unflatten_config(
    params: np.ndarray, n_clusters: int, n_control_points: int
) -> NeighborConfig:
    """Unflatten parameter vector to NeighborConfig.

    Parameters
    ----------
    params : np.ndarray
        Flat parameter vector.
    n_clusters : int
        Number of clusters.
    n_control_points : int
        Control points per cluster.

    Returns
    -------
    NeighborConfig
        Reconstructed configuration.
    """
    # Parameters per cluster: 2 (center) + 2K (control points) + 1 (density) + 1 (rotation)
    params_per_cluster = 2 + 2 * n_control_points + 2

    clusters = []
    for i in range(n_clusters):
        offset = i * params_per_cluster
        cx = params[offset]
        cy = params[offset + 1]
        cp_flat = params[offset + 2 : offset + 2 + 2 * n_control_points]
        control_points = cp_flat.reshape(n_control_points, 2)
        density = params[offset + 2 + 2 * n_control_points]
        rotation = params[offset + 2 + 2 * n_control_points + 1]

        clusters.append(ClusterParams(
            center=(cx, cy),
            control_points=control_points,
            density=density,
            rotation=rotation,
        ))

    return NeighborConfig(clusters=clusters)


# =============================================================================
# Adversarial Optimizer
# =============================================================================


@dataclass
class OptimizationEntry:
    """Single entry in optimization history."""

    iteration: int
    config: NeighborConfig
    neighbor_x: np.ndarray
    neighbor_y: np.ndarray
    liberal_x: np.ndarray
    liberal_y: np.ndarray
    conservative_x: np.ndarray
    conservative_y: np.ndarray
    aep_lib_isolated: float
    aep_lib_neighbor: float
    aep_con_isolated: float
    aep_con_neighbor: float
    regret: float


@dataclass
class OptimizationHistory:
    """Full optimization history."""

    entries: list[OptimizationEntry] = field(default_factory=list)
    best_entry: OptimizationEntry | None = None

    def add(self, entry: OptimizationEntry):
        """Add entry and update best if applicable."""
        self.entries.append(entry)
        if self.best_entry is None or entry.regret > self.best_entry.regret:
            self.best_entry = entry


class AdversarialNeighborOptimizer:
    """Optimizer to find worst-case neighbor configurations.

    Uses CMA-ES outer loop to optimize neighbor cluster parameters,
    with inner optimization of target farm layouts for each candidate.

    Parameters
    ----------
    target_x : np.ndarray
        Initial target farm x positions.
    target_y : np.ndarray
        Initial target farm y positions.
    target_boundary : list[tuple[float, float]]
        Target farm boundary vertices.
    sim_engine : WakeSimulation
        Wake simulation engine.
    wd_arr : jnp.ndarray
        Wind directions array.
    weights : jnp.ndarray
        Wind rose probability weights.
    wind_speed : float
        Reference wind speed.
    rotor_diameter : float
        Turbine rotor diameter.
    n_clusters : int
        Number of neighbor clusters.
    n_control_points : int
        Control points per cluster spline.
    inner_optimizer : Callable
        Function to optimize target layout given neighbors.
    search_radius : float
        Maximum distance from target center for cluster centers.
    min_neighbor_distance : float
        Minimum distance from target boundary to cluster center.
    max_density : float
        Maximum turbine density (turbines per unit area).
    max_cluster_area : float
        Maximum cluster area.
    """

    def __init__(
        self,
        target_x: np.ndarray,
        target_y: np.ndarray,
        target_boundary: list[tuple[float, float]],
        sim_engine,
        wd_arr: jnp.ndarray,
        weights: jnp.ndarray,
        wind_speed: float,
        rotor_diameter: float,
        n_clusters: int = 1,
        n_control_points: int = 6,
        inner_optimizer: Callable | None = None,
        search_radius: float | None = None,
        min_neighbor_distance: float | None = None,
        max_density: float | None = None,
        max_cluster_area: float | None = None,
    ):
        self.target_x = np.asarray(target_x)
        self.target_y = np.asarray(target_y)
        self.target_boundary = target_boundary
        self.sim_engine = sim_engine
        self.wd_arr = wd_arr
        self.weights = weights
        self.wind_speed = wind_speed
        self.rotor_diameter = rotor_diameter
        self.n_clusters = n_clusters
        self.n_control_points = n_control_points
        self.inner_optimizer = inner_optimizer

        # Compute target farm center and radius
        self.target_center = (np.mean(target_x), np.mean(target_y))
        self.target_radius = np.max(np.sqrt(
            (target_x - self.target_center[0]) ** 2 +
            (target_y - self.target_center[1]) ** 2
        ))

        # Default constraints based on rotor diameter
        D = rotor_diameter
        self.search_radius = search_radius or 30 * D
        self.min_neighbor_distance = min_neighbor_distance or 3 * D
        self.max_density = max_density or 1.0 / (4 * D) ** 2  # ~1 turbine per (4D)²
        self.max_cluster_area = max_cluster_area or (10 * D) ** 2

        # Turbine packer
        self.packer = TurbinePacker(min_spacing=2 * D)

        # Parameters per cluster
        self.params_per_cluster = 2 + 2 * n_control_points + 2
        self.total_params = n_clusters * self.params_per_cluster

        # Cache for liberal layout (doesn't depend on neighbors)
        self._liberal_layout_cache = None

    def _default_cluster_size(self) -> float:
        """Get default cluster control point scale."""
        return 5 * self.rotor_diameter

    def _create_initial_config(self, angle: float = 0.0) -> NeighborConfig:
        """Create initial cluster configuration at given angle from target.

        Parameters
        ----------
        angle : float
            Angle in radians from target center.

        Returns
        -------
        NeighborConfig
            Initial configuration.
        """
        clusters = []
        cluster_scale = self._default_cluster_size()

        for i in range(self.n_clusters):
            # Position cluster at angle from target
            cluster_angle = angle + i * (2 * np.pi / self.n_clusters)
            dist = self.target_radius + self.min_neighbor_distance + cluster_scale

            cx = self.target_center[0] + dist * np.cos(cluster_angle)
            cy = self.target_center[1] + dist * np.sin(cluster_angle)

            # Elliptical control points
            angles = np.linspace(0, 2 * np.pi, self.n_control_points, endpoint=False)
            control_points = np.column_stack([
                cluster_scale * np.cos(angles),
                cluster_scale * 0.6 * np.sin(angles),
            ])

            clusters.append(ClusterParams(
                center=(cx, cy),
                control_points=control_points,
                density=self.max_density * 0.5,
                rotation=0.0,
            ))

        return NeighborConfig(clusters=clusters)

    def _get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Get parameter bounds for CMA-ES.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (lower_bounds, upper_bounds)
        """
        lower, upper = [], []
        cluster_scale = self._default_cluster_size()

        for _ in range(self.n_clusters):
            # Center bounds
            lower.extend([
                self.target_center[0] - self.search_radius,
                self.target_center[1] - self.search_radius,
            ])
            upper.extend([
                self.target_center[0] + self.search_radius,
                self.target_center[1] + self.search_radius,
            ])

            # Control point bounds (relative to center)
            for _ in range(self.n_control_points):
                lower.extend([-2 * cluster_scale, -2 * cluster_scale])
                upper.extend([2 * cluster_scale, 2 * cluster_scale])

            # Density bounds
            lower.append(0.0)
            upper.append(self.max_density)

            # Rotation bounds
            lower.append(-np.pi)
            upper.append(np.pi)

        return np.array(lower), np.array(upper)

    def _evaluate_aep(
        self,
        layout_x: np.ndarray,
        layout_y: np.ndarray,
        neighbor_x: np.ndarray | None,
        neighbor_y: np.ndarray | None,
    ) -> float:
        """Evaluate AEP for a layout with optional neighbors.

        Parameters
        ----------
        layout_x, layout_y : np.ndarray
            Target layout positions.
        neighbor_x, neighbor_y : np.ndarray or None
            Neighbor positions (None for isolated evaluation).

        Returns
        -------
        float
            AEP in GWh.
        """
        n_target = len(layout_x)

        if neighbor_x is not None and len(neighbor_x) > 0:
            x_all = jnp.concatenate([jnp.array(layout_x), jnp.array(neighbor_x)])
            y_all = jnp.concatenate([jnp.array(layout_y), jnp.array(neighbor_y)])
        else:
            x_all = jnp.array(layout_x)
            y_all = jnp.array(layout_y)

        ws_arr = jnp.full_like(self.wd_arr, self.wind_speed)
        result = self.sim_engine(x_all, y_all, ws_amb=ws_arr, wd_amb=self.wd_arr, ti_amb=0.06)

        power_kw = result.power()
        target_power = power_kw[:, :n_target]
        weighted_power = jnp.sum(target_power * self.weights[:, None])
        return float(weighted_power * 8760 / 1e6)

    def _check_constraints(self, config: NeighborConfig) -> bool:
        """Check if configuration satisfies constraints.

        Parameters
        ----------
        config : NeighborConfig
            Configuration to check.

        Returns
        -------
        bool
            True if all constraints satisfied.
        """
        for cluster_params in config.clusters:
            cluster = cluster_params.to_cluster()

            # Check cluster area
            if cluster.area > self.max_cluster_area:
                return False
            if cluster.area < 0:
                return False

            # Check center distance from target
            dx = cluster_params.center[0] - self.target_center[0]
            dy = cluster_params.center[1] - self.target_center[1]
            dist_from_center = np.sqrt(dx**2 + dy**2)

            if dist_from_center > self.search_radius:
                return False

            # Check minimum distance from target boundary
            min_dist_to_target = dist_from_center - self.target_radius
            if min_dist_to_target < self.min_neighbor_distance:
                return False

            # Check density
            if cluster_params.density < 0 or cluster_params.density > self.max_density:
                return False

        return True

    def _constraint_penalty(self, config: NeighborConfig) -> float:
        """Compute constraint violation penalty.

        Parameters
        ----------
        config : NeighborConfig
            Configuration to evaluate.

        Returns
        -------
        float
            Penalty value (0 if no violations).
        """
        penalty = 0.0

        for cluster_params in config.clusters:
            cluster = cluster_params.to_cluster()

            # Area penalty
            if cluster.area > self.max_cluster_area:
                penalty += (cluster.area - self.max_cluster_area) ** 2

            # Distance penalty
            dx = cluster_params.center[0] - self.target_center[0]
            dy = cluster_params.center[1] - self.target_center[1]
            dist_from_center = np.sqrt(dx**2 + dy**2)

            if dist_from_center > self.search_radius:
                penalty += (dist_from_center - self.search_radius) ** 2

            min_dist_to_target = dist_from_center - self.target_radius
            if min_dist_to_target < self.min_neighbor_distance:
                penalty += (self.min_neighbor_distance - min_dist_to_target) ** 2

            # Density penalty
            if cluster_params.density < 0:
                penalty += cluster_params.density ** 2
            if cluster_params.density > self.max_density:
                penalty += (cluster_params.density - self.max_density) ** 2

        return penalty * 1e6  # Scale penalty to dominate objective

    def objective(self, params_flat: np.ndarray) -> float:
        """Compute objective for minimization (negative regret + penalty).

        Parameters
        ----------
        params_flat : np.ndarray
            Flat parameter vector.

        Returns
        -------
        float
            Objective value (minimize this).
        """
        config = unflatten_config(params_flat, self.n_clusters, self.n_control_points)

        # Constraint penalty
        penalty = self._constraint_penalty(config)
        if penalty > 0:
            return penalty

        # Convert to turbine positions
        neighbor_x, neighbor_y = config.to_turbines(self.packer)

        if len(neighbor_x) == 0:
            return 0.0  # No neighbors = no regret

        # Get liberal layout (cached)
        if self._liberal_layout_cache is None:
            if self.inner_optimizer is not None:
                lib_x, lib_y = self.inner_optimizer(
                    self.target_x, self.target_y, self.target_boundary,
                    None, None
                )
            else:
                lib_x, lib_y = self.target_x.copy(), self.target_y.copy()
            self._liberal_layout_cache = (lib_x, lib_y)
        else:
            lib_x, lib_y = self._liberal_layout_cache

        # Get conservative layout (must reoptimize for each neighbor config)
        if self.inner_optimizer is not None:
            con_x, con_y = self.inner_optimizer(
                self.target_x, self.target_y, self.target_boundary,
                neighbor_x, neighbor_y
            )
        else:
            con_x, con_y = self.target_x.copy(), self.target_y.copy()

        # Evaluate AEPs
        aep_lib_neighbor = self._evaluate_aep(lib_x, lib_y, neighbor_x, neighbor_y)
        aep_con_neighbor = self._evaluate_aep(con_x, con_y, neighbor_x, neighbor_y)

        # Liberal regret: how much better conservative does when neighbors appear
        regret = aep_con_neighbor - aep_lib_neighbor

        # Return negative for minimization
        return -regret

    def objective_full(
        self, params_flat: np.ndarray
    ) -> tuple[float, NeighborConfig, np.ndarray, np.ndarray,
               np.ndarray, np.ndarray, np.ndarray, np.ndarray,
               float, float, float, float]:
        """Compute objective with full diagnostic information.

        Parameters
        ----------
        params_flat : np.ndarray
            Flat parameter vector.

        Returns
        -------
        tuple
            (regret, config, neighbor_x, neighbor_y, lib_x, lib_y, con_x, con_y,
             aep_lib_iso, aep_lib_nei, aep_con_iso, aep_con_nei)
        """
        config = unflatten_config(params_flat, self.n_clusters, self.n_control_points)
        neighbor_x, neighbor_y = config.to_turbines(self.packer)

        if len(neighbor_x) == 0:
            return (0.0, config, neighbor_x, neighbor_y,
                    self.target_x, self.target_y, self.target_x, self.target_y,
                    0.0, 0.0, 0.0, 0.0)

        # Liberal layout
        if self._liberal_layout_cache is None:
            if self.inner_optimizer is not None:
                lib_x, lib_y = self.inner_optimizer(
                    self.target_x, self.target_y, self.target_boundary,
                    None, None
                )
            else:
                lib_x, lib_y = self.target_x.copy(), self.target_y.copy()
            self._liberal_layout_cache = (lib_x, lib_y)
        else:
            lib_x, lib_y = self._liberal_layout_cache

        # Conservative layout
        if self.inner_optimizer is not None:
            con_x, con_y = self.inner_optimizer(
                self.target_x, self.target_y, self.target_boundary,
                neighbor_x, neighbor_y
            )
        else:
            con_x, con_y = self.target_x.copy(), self.target_y.copy()

        # Full 2x2 AEP matrix
        aep_lib_iso = self._evaluate_aep(lib_x, lib_y, None, None)
        aep_lib_nei = self._evaluate_aep(lib_x, lib_y, neighbor_x, neighbor_y)
        aep_con_iso = self._evaluate_aep(con_x, con_y, None, None)
        aep_con_nei = self._evaluate_aep(con_x, con_y, neighbor_x, neighbor_y)

        regret = aep_con_nei - aep_lib_nei

        return (regret, config, neighbor_x, neighbor_y,
                np.asarray(lib_x), np.asarray(lib_y),
                np.asarray(con_x), np.asarray(con_y),
                aep_lib_iso, aep_lib_nei, aep_con_iso, aep_con_nei)

    def optimize(
        self,
        n_iterations: int = 100,
        population_size: int | None = None,
        sigma0: float = 0.3,
        initial_angle: float | None = None,
        seed: int = 42,
        verbose: bool = True,
        callback: Callable[[int, OptimizationEntry], None] | None = None,
    ) -> OptimizationHistory:
        """Run CMA-ES optimization to find worst-case neighbors.

        Parameters
        ----------
        n_iterations : int
            Maximum number of CMA-ES iterations.
        population_size : int or None
            CMA-ES population size (default: auto).
        sigma0 : float
            Initial step size (relative to bounds).
        initial_angle : float or None
            Initial cluster angle from target (radians). If None, uses
            dominant wind direction.
        seed : int
            Random seed.
        verbose : bool
            Print progress.
        callback : Callable or None
            Called after each iteration with (iteration, entry).

        Returns
        -------
        OptimizationHistory
            Full optimization history.
        """
        try:
            import cma
        except ImportError:
            raise ImportError(
                "CMA-ES requires the 'cma' package. Install with: pip install cma"
            )

        # Initialize
        if initial_angle is None:
            # Default: upwind of dominant wind direction
            # Find dominant direction from weights
            dominant_idx = np.argmax(np.asarray(self.weights))
            dominant_wd = float(self.wd_arr[dominant_idx])
            initial_angle = np.deg2rad(90 - dominant_wd)  # Convert to bearing

        initial_config = self._create_initial_config(angle=initial_angle)
        x0 = flatten_config(initial_config, self.n_control_points)

        # Get bounds
        lower_bounds, upper_bounds = self._get_bounds()

        # Normalize initial guess to [0, 1] for CMA-ES
        ranges = upper_bounds - lower_bounds
        x0_normalized = (x0 - lower_bounds) / ranges

        # CMA-ES options
        opts = {
            'bounds': [0, 1],
            'maxiter': n_iterations,
            'seed': seed,
            'verbose': -9 if not verbose else 3,
        }
        if population_size is not None:
            opts['popsize'] = population_size

        # Create CMA-ES optimizer
        es = cma.CMAEvolutionStrategy(x0_normalized, sigma0, opts)

        history = OptimizationHistory()
        iteration = 0

        while not es.stop():
            # Get population
            solutions = es.ask()

            # Evaluate (denormalize first)
            fitnesses = []
            for sol in solutions:
                params = lower_bounds + sol * ranges
                fitness = self.objective(params)
                fitnesses.append(fitness)

            # Tell CMA-ES
            es.tell(solutions, fitnesses)

            # Get best of this generation
            best_idx = np.argmin(fitnesses)
            best_params = lower_bounds + solutions[best_idx] * ranges
            best_regret = -fitnesses[best_idx]

            # Full evaluation for history
            (regret, config, nei_x, nei_y, lib_x, lib_y, con_x, con_y,
             aep_lib_iso, aep_lib_nei, aep_con_iso, aep_con_nei) = self.objective_full(best_params)

            entry = OptimizationEntry(
                iteration=iteration,
                config=config,
                neighbor_x=nei_x,
                neighbor_y=nei_y,
                liberal_x=lib_x,
                liberal_y=lib_y,
                conservative_x=con_x,
                conservative_y=con_y,
                aep_lib_isolated=aep_lib_iso,
                aep_lib_neighbor=aep_lib_nei,
                aep_con_isolated=aep_con_iso,
                aep_con_neighbor=aep_con_nei,
                regret=regret,
            )
            history.add(entry)

            if callback is not None:
                callback(iteration, entry)

            if verbose:
                n_turbines = len(nei_x)
                print(f"Iter {iteration:3d}: regret = {regret:8.3f} GWh, "
                      f"n_neighbors = {n_turbines:3d}")

            iteration += 1

        return history


# =============================================================================
# Visualization
# =============================================================================


def plot_adversarial_state(
    entry: OptimizationEntry,
    target_boundary: list[tuple[float, float]],
    rotor_diameter: float,
    figsize: tuple[float, float] = (18, 6),
) -> plt.Figure:
    """Plot adversarial optimization state.

    Parameters
    ----------
    entry : OptimizationEntry
        Optimization state to plot.
    target_boundary : list
        Target farm boundary vertices.
    rotor_diameter : float
        Turbine rotor diameter.
    figsize : tuple
        Figure size.

    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Common plotting elements
    bx = [p[0] for p in target_boundary] + [target_boundary[0][0]]
    by = [p[1] for p in target_boundary] + [target_boundary[0][1]]

    # Panel 1: Cluster boundaries + turbine positions
    ax1 = axes[0]
    ax1.plot(bx, by, 'k-', linewidth=2, label='Target boundary')

    # Plot cluster boundaries
    for i, cluster_params in enumerate(entry.config.clusters):
        cluster = cluster_params.to_cluster()
        boundary = cluster.boundary
        boundary_closed = np.vstack([boundary, boundary[0]])
        ax1.plot(boundary_closed[:, 0], boundary_closed[:, 1], 'r--',
                linewidth=1.5, label=f'Cluster {i+1}' if i == 0 else None)
        ax1.fill(boundary[:, 0], boundary[:, 1], alpha=0.1, color='red')

    # Plot neighbor turbines
    if len(entry.neighbor_x) > 0:
        ax1.scatter(entry.neighbor_x, entry.neighbor_y, s=80, c='red',
                   marker='s', alpha=0.7, label=f'Neighbors (n={len(entry.neighbor_x)})')

    # Plot target turbines (both layouts)
    ax1.scatter(entry.liberal_x, entry.liberal_y, s=100, c='steelblue',
               marker='o', alpha=0.5, label='Liberal layout')
    ax1.scatter(entry.conservative_x, entry.conservative_y, s=100, c='darkorange',
               marker='^', alpha=0.5, label='Conservative layout')

    ax1.set_aspect('equal')
    ax1.set_title(f'Iteration {entry.iteration}\nRegret: {entry.regret:.2f} GWh')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Liberal layout detail
    ax2 = axes[1]
    ax2.plot(bx, by, 'k-', linewidth=2)

    for x, y in zip(entry.liberal_x, entry.liberal_y):
        circle = plt.Circle((x, y), rotor_diameter/2, fill=False,
                           color='steelblue', alpha=0.5)
        ax2.add_patch(circle)
    ax2.scatter(entry.liberal_x, entry.liberal_y, s=60, c='steelblue',
               marker='o', edgecolors='black', linewidths=1)

    if len(entry.neighbor_x) > 0:
        ax2.scatter(entry.neighbor_x, entry.neighbor_y, s=40, c='red',
                   marker='s', alpha=0.5)

    ax2.set_aspect('equal')
    ax2.set_title(f'Liberal Layout\nAEP (isolated): {entry.aep_lib_isolated:.1f} GWh\n'
                 f'AEP (neighbor): {entry.aep_lib_neighbor:.1f} GWh')
    ax2.grid(True, alpha=0.3)

    # Panel 3: Conservative layout detail
    ax3 = axes[2]
    ax3.plot(bx, by, 'k-', linewidth=2)

    for x, y in zip(entry.conservative_x, entry.conservative_y):
        circle = plt.Circle((x, y), rotor_diameter/2, fill=False,
                           color='darkorange', alpha=0.5)
        ax3.add_patch(circle)
    ax3.scatter(entry.conservative_x, entry.conservative_y, s=60, c='darkorange',
               marker='^', edgecolors='black', linewidths=1)

    if len(entry.neighbor_x) > 0:
        ax3.scatter(entry.neighbor_x, entry.neighbor_y, s=40, c='red',
                   marker='s', alpha=0.5)

    ax3.set_aspect('equal')
    ax3.set_title(f'Conservative Layout\nAEP (isolated): {entry.aep_con_isolated:.1f} GWh\n'
                 f'AEP (neighbor): {entry.aep_con_neighbor:.1f} GWh')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_animation(
    history: OptimizationHistory,
    target_boundary: list[tuple[float, float]],
    rotor_diameter: float,
    output_path: str = 'adversarial_morphing.mp4',
    fps: int = 5,
    dpi: int = 100,
) -> None:
    """Create animation of optimization progress.

    Parameters
    ----------
    history : OptimizationHistory
        Full optimization history.
    target_boundary : list
        Target farm boundary vertices.
    rotor_diameter : float
        Turbine rotor diameter.
    output_path : str
        Output file path (.mp4 or .gif).
    fps : int
        Frames per second.
    dpi : int
        Figure DPI.
    """
    try:
        import imageio
    except ImportError:
        raise ImportError(
            "Animation requires the 'imageio' package. "
            "Install with: pip install imageio[ffmpeg]"
        )

    import io

    frames = []
    for entry in history.entries:
        fig = plot_adversarial_state(entry, target_boundary, rotor_diameter)

        # Convert figure to image array
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
        buf.seek(0)
        img = imageio.imread(buf)
        frames.append(img)
        plt.close(fig)
        buf.close()

    # Write video/gif
    if output_path.endswith('.gif'):
        imageio.mimsave(output_path, frames, fps=fps)
    else:
        imageio.mimsave(output_path, frames, fps=fps, codec='libx264')

    print(f"Animation saved to {output_path}")


# =============================================================================
# Differentiable Spline-Based Optimization
# =============================================================================


def _bspline_basis(t: jnp.ndarray, i: int, k: int, knots: jnp.ndarray) -> jnp.ndarray:
    """Compute B-spline basis function (Cox-de Boor recursion).

    Parameters
    ----------
    t : jnp.ndarray
        Parameter values.
    i : int
        Basis function index.
    k : int
        Degree (0 = constant, 1 = linear, 3 = cubic).
    knots : jnp.ndarray
        Knot vector.

    Returns
    -------
    jnp.ndarray
        Basis function values at t.
    """
    if k == 0:
        return jnp.where((knots[i] <= t) & (t < knots[i + 1]), 1.0, 0.0)

    # Recursive case
    denom1 = knots[i + k] - knots[i]
    denom2 = knots[i + k + 1] - knots[i + 1]

    term1 = jnp.where(
        denom1 > 1e-10,
        (t - knots[i]) / denom1 * _bspline_basis(t, i, k - 1, knots),
        0.0
    )
    term2 = jnp.where(
        denom2 > 1e-10,
        (knots[i + k + 1] - t) / denom2 * _bspline_basis(t, i + 1, k - 1, knots),
        0.0
    )

    return term1 + term2


def evaluate_closed_bspline(
    control_points: jnp.ndarray,
    t: jnp.ndarray,
    degree: int = 3
) -> jnp.ndarray:
    """Evaluate closed B-spline curve at parameter values.

    This is a JAX-differentiable B-spline evaluation.

    Parameters
    ----------
    control_points : jnp.ndarray
        Shape (n, 2) control points. The curve will be closed by
        wrapping the first `degree` points.
    t : jnp.ndarray
        Parameter values in [0, 1).
    degree : int
        Spline degree (default: 3 for cubic).

    Returns
    -------
    jnp.ndarray
        Shape (len(t), 2) points on the curve.
    """
    n = len(control_points)

    # Wrap control points for closed curve
    cp_wrapped = jnp.concatenate([control_points, control_points[:degree]], axis=0)
    n_wrapped = n + degree

    # Create uniform knot vector for closed curve
    knots = jnp.linspace(0, 1, n_wrapped - degree + 2)

    # Scale t to knot range (excluding last knot to avoid boundary issues)
    t_scaled = t * (knots[-degree-1] - knots[degree]) + knots[degree]

    # Evaluate using matrix form for efficiency
    # For each t, compute weighted sum of control points
    points = jnp.zeros((len(t), 2))

    for i in range(n_wrapped):
        basis = _bspline_basis(t_scaled, i, degree, knots)
        points = points + basis[:, None] * cp_wrapped[i]

    return points


def differentiable_spline_positions(
    control_points: jnp.ndarray,
    center: jnp.ndarray,
    rotation: float,
    n_turbines: int,
    fill_fraction: float = 0.7,
) -> jnp.ndarray:
    """Compute turbine positions from spline parameters (differentiable).

    Places turbines in a pattern that fills the spline interior.
    Uses a combination of:
    1. Points along the boundary (scaled inward)
    2. Center point(s)

    Parameters
    ----------
    control_points : jnp.ndarray
        Shape (K, 2) spline control points (relative to center).
    center : jnp.ndarray
        Shape (2,) cluster center position.
    rotation : float
        Rotation angle in radians.
    n_turbines : int
        Number of turbines to place.
    fill_fraction : float
        How much to fill toward center (0 = boundary only, 1 = full fill).

    Returns
    -------
    jnp.ndarray
        Shape (n_turbines, 2) turbine positions.
    """
    # Apply rotation to control points
    cos_r = jnp.cos(rotation)
    sin_r = jnp.sin(rotation)
    rot_matrix = jnp.array([[cos_r, -sin_r], [sin_r, cos_r]])
    cp_rotated = control_points @ rot_matrix.T

    # Evaluate boundary at uniform parameter values
    t_boundary = jnp.linspace(0, 1, n_turbines * 3, endpoint=False)
    boundary_points = evaluate_closed_bspline(cp_rotated, t_boundary, degree=3)

    # Compute centroid of boundary
    centroid = jnp.mean(boundary_points, axis=0)

    # Strategy: place turbines at multiple "rings" from boundary to center
    # Ring 0: on boundary (scaled by 0.9 to be slightly inside)
    # Ring 1: 50% toward center
    # Ring 2: at center

    positions = []

    if n_turbines <= 6:
        # Small number: place along boundary
        t_vals = jnp.linspace(0, 1, n_turbines, endpoint=False)
        ring_points = evaluate_closed_bspline(cp_rotated, t_vals, degree=3)
        # Scale slightly inward
        ring_points = centroid + 0.85 * (ring_points - centroid)
        positions = ring_points
    else:
        # Multiple rings
        n_outer = min(n_turbines - 1, max(6, n_turbines // 2))
        n_inner = n_turbines - n_outer

        # Outer ring (85% from center to boundary)
        t_outer = jnp.linspace(0, 1, n_outer, endpoint=False)
        outer_points = evaluate_closed_bspline(cp_rotated, t_outer, degree=3)
        outer_ring = centroid + 0.85 * (outer_points - centroid)

        if n_inner <= 1:
            # Just add center
            inner_ring = centroid[None, :]
        elif n_inner <= 6:
            # Inner ring at 40% scale
            t_inner = jnp.linspace(0, 1, n_inner, endpoint=False)
            inner_points = evaluate_closed_bspline(cp_rotated, t_inner, degree=3)
            inner_ring = centroid + 0.4 * fill_fraction * (inner_points - centroid)
        else:
            # Two inner rings
            n_mid = n_inner // 2
            n_center = n_inner - n_mid

            t_mid = jnp.linspace(0, 1, n_mid, endpoint=False)
            mid_points = evaluate_closed_bspline(cp_rotated, t_mid, degree=3)
            mid_ring = centroid + 0.55 * fill_fraction * (mid_points - centroid)

            if n_center <= 1:
                center_ring = centroid[None, :]
            else:
                t_center = jnp.linspace(0, 1, n_center, endpoint=False)
                center_points = evaluate_closed_bspline(cp_rotated, t_center, degree=3)
                center_ring = centroid + 0.2 * fill_fraction * (center_points - centroid)

            inner_ring = jnp.concatenate([mid_ring, center_ring], axis=0)

        positions = jnp.concatenate([outer_ring, inner_ring], axis=0)

    # Translate to center
    positions = positions + center

    return positions[:n_turbines]


class DifferentiableSplineOptimizer:
    """Gradient-based optimizer using differentiable spline parameterization.

    Turbine positions are smooth functions of spline control points,
    enabling gradient-based optimization of cluster shape.

    Parameters
    ----------
    target_x : np.ndarray
        Target farm x positions.
    target_y : np.ndarray
        Target farm y positions.
    sim_engine : WakeSimulation
        Wake simulation engine.
    wd_arr : jnp.ndarray
        Wind directions array.
    weights : jnp.ndarray
        Wind rose probability weights.
    wind_speed : float
        Reference wind speed.
    rotor_diameter : float
        Turbine rotor diameter.
    n_neighbors : int
        Number of neighbor turbines.
    n_control_points : int
        Number of spline control points.
    search_radius : float
        Maximum distance from target center.
    min_neighbor_distance : float
        Minimum distance from target boundary.
    """

    def __init__(
        self,
        target_x: np.ndarray,
        target_y: np.ndarray,
        sim_engine,
        wd_arr: jnp.ndarray,
        weights: jnp.ndarray,
        wind_speed: float,
        rotor_diameter: float,
        n_neighbors: int = 16,
        n_control_points: int = 6,
        search_radius: float | None = None,
        min_neighbor_distance: float | None = None,
    ):
        self.target_x = jnp.array(target_x)
        self.target_y = jnp.array(target_y)
        self.n_target = len(target_x)
        self.sim_engine = sim_engine
        self.wd_arr = wd_arr
        self.weights = weights
        self.wind_speed = wind_speed
        self.rotor_diameter = rotor_diameter
        self.n_neighbors = n_neighbors
        self.n_control_points = n_control_points

        # Compute target farm geometry
        self.target_center = jnp.array([jnp.mean(target_x), jnp.mean(target_y)])
        self.target_radius = float(jnp.max(jnp.sqrt(
            (target_x - self.target_center[0]) ** 2 +
            (target_y - self.target_center[1]) ** 2
        )))

        D = rotor_diameter
        self.search_radius = search_radius or 30 * D
        self.min_neighbor_distance = min_neighbor_distance or 3 * D
        self.default_cluster_scale = 5 * D

        self._setup_jax_functions()

    def _setup_jax_functions(self):
        """Set up JIT-compiled functions."""

        def compute_aep_from_spline(control_points, center, rotation):
            """Compute target AEP given spline parameters."""
            # Get neighbor positions from spline (differentiable)
            neighbor_pos = differentiable_spline_positions(
                control_points, center, rotation,
                self.n_neighbors, fill_fraction=0.7
            )
            neighbor_x = neighbor_pos[:, 0]
            neighbor_y = neighbor_pos[:, 1]

            # Combine with target
            x_all = jnp.concatenate([self.target_x, neighbor_x])
            y_all = jnp.concatenate([self.target_y, neighbor_y])

            # Simulate
            ws_arr = jnp.full_like(self.wd_arr, self.wind_speed)
            result = self.sim_engine(x_all, y_all, ws_amb=ws_arr, wd_amb=self.wd_arr, ti_amb=0.06)

            # Target AEP only
            power_kw = result.power()
            target_power = power_kw[:, :self.n_target]
            weighted_power = jnp.sum(target_power * self.weights[:, None])
            return weighted_power * 8760 / 1e6

        self._compute_aep = jax.jit(compute_aep_from_spline)
        self._aep_and_grad = jax.jit(jax.value_and_grad(
            compute_aep_from_spline, argnums=(0, 1, 2)
        ))

    def _initialize_params(
        self, angle: float | None = None
    ) -> tuple[jnp.ndarray, jnp.ndarray, float]:
        """Initialize spline parameters.

        Returns
        -------
        tuple
            (control_points, center, rotation)
        """
        if angle is None:
            dominant_idx = int(jnp.argmax(self.weights))
            dominant_wd = float(self.wd_arr[dominant_idx])
            angle = np.deg2rad(90 - dominant_wd)

        # Place cluster center upwind
        dist = self.target_radius + self.min_neighbor_distance + self.default_cluster_scale
        center = self.target_center + dist * jnp.array([jnp.cos(angle), jnp.sin(angle)])

        # Elliptical control points
        cp_angles = jnp.linspace(0, 2 * jnp.pi, self.n_control_points, endpoint=False)
        control_points = jnp.column_stack([
            self.default_cluster_scale * jnp.cos(cp_angles),
            self.default_cluster_scale * 0.6 * jnp.sin(cp_angles),
        ])

        rotation = 0.0

        return control_points, center, rotation

    def _project_center(self, center: jnp.ndarray) -> jnp.ndarray:
        """Project center to satisfy distance constraints."""
        delta = center - self.target_center
        dist = jnp.sqrt(jnp.sum(delta ** 2)) + 1e-6

        # Minimum distance
        min_dist = self.target_radius + self.min_neighbor_distance
        # Maximum distance
        max_dist = self.search_radius

        # Clamp distance
        clamped_dist = jnp.clip(dist, min_dist, max_dist)
        return self.target_center + delta * (clamped_dist / dist)

    def get_turbine_positions(
        self,
        control_points: jnp.ndarray,
        center: jnp.ndarray,
        rotation: float,
    ) -> jnp.ndarray:
        """Get turbine positions from current parameters."""
        return differentiable_spline_positions(
            control_points, center, rotation,
            self.n_neighbors, fill_fraction=0.7
        )

    def optimize(
        self,
        n_iterations: int = 200,
        learning_rate: float | None = None,
        initial_angle: float | None = None,
        verbose: bool = True,
    ) -> tuple[jnp.ndarray, jnp.ndarray, float, list[float]]:
        """Optimize spline parameters to minimize target AEP.

        Parameters
        ----------
        n_iterations : int
            Number of gradient descent iterations.
        learning_rate : float or None
            Step size for control points/center (default: rotor_diameter / 10).
        initial_angle : float or None
            Initial cluster placement angle.
        verbose : bool
            Print progress.

        Returns
        -------
        tuple
            (control_points, center, rotation, aep_history)
        """
        if learning_rate is None:
            learning_rate = self.rotor_diameter / 10

        # Initialize
        control_points, center, rotation = self._initialize_params(initial_angle)
        aep_history = []

        # Learning rates for different parameters
        lr_cp = learning_rate  # Control points
        lr_center = learning_rate * 2  # Center (can move faster)
        lr_rotation = 0.01  # Rotation (radians)

        for i in range(n_iterations):
            # Compute AEP and gradients
            aep, (grad_cp, grad_center, grad_rotation) = self._aep_and_grad(
                control_points, center, rotation
            )
            aep_history.append(float(aep))

            # Gradient descent (minimize AEP)
            control_points = control_points - lr_cp * grad_cp
            center = center - lr_center * grad_center
            rotation = rotation - lr_rotation * grad_rotation

            # Project center to constraints
            center = self._project_center(center)

            # Keep rotation in [-pi, pi]
            rotation = jnp.arctan2(jnp.sin(rotation), jnp.cos(rotation))

            if verbose and i % 20 == 0:
                print(f"Iter {i:4d}: target AEP = {aep:.3f} GWh")

        return control_points, center, float(rotation), aep_history


# =============================================================================
# Gradient-Based Direct Position Optimization
# =============================================================================


class GradientAdversarialOptimizer:
    """Gradient-based adversarial optimizer using direct position optimization.

    Instead of parameterizing clusters with splines, this optimizer directly
    optimizes neighbor turbine positions using JAX autodiff. This is more
    efficient when the number of neighbor turbines is fixed.

    Parameters
    ----------
    target_x : np.ndarray
        Target farm x positions.
    target_y : np.ndarray
        Target farm y positions.
    sim_engine : WakeSimulation
        Wake simulation engine.
    wd_arr : jnp.ndarray
        Wind directions array.
    weights : jnp.ndarray
        Wind rose probability weights.
    wind_speed : float
        Reference wind speed.
    rotor_diameter : float
        Turbine rotor diameter.
    n_neighbors : int
        Number of neighbor turbines to optimize.
    search_radius : float
        Maximum distance from target center.
    min_neighbor_distance : float
        Minimum distance from target boundary.
    """

    def __init__(
        self,
        target_x: np.ndarray,
        target_y: np.ndarray,
        sim_engine,
        wd_arr: jnp.ndarray,
        weights: jnp.ndarray,
        wind_speed: float,
        rotor_diameter: float,
        n_neighbors: int = 16,
        search_radius: float | None = None,
        min_neighbor_distance: float | None = None,
    ):
        self.target_x = jnp.array(target_x)
        self.target_y = jnp.array(target_y)
        self.n_target = len(target_x)
        self.sim_engine = sim_engine
        self.wd_arr = wd_arr
        self.weights = weights
        self.wind_speed = wind_speed
        self.rotor_diameter = rotor_diameter
        self.n_neighbors = n_neighbors

        # Compute target farm center and radius
        self.target_center = (float(jnp.mean(target_x)), float(jnp.mean(target_y)))
        self.target_radius = float(jnp.max(jnp.sqrt(
            (target_x - self.target_center[0]) ** 2 +
            (target_y - self.target_center[1]) ** 2
        )))

        D = rotor_diameter
        self.search_radius = search_radius or 30 * D
        self.min_neighbor_distance = min_neighbor_distance or 3 * D
        self.min_spacing = 2 * D

        # JIT compile evaluation functions
        self._setup_jax_functions()

    def _setup_jax_functions(self):
        """Set up JIT-compiled JAX functions for gradient computation."""

        def compute_target_aep(target_x, target_y, neighbor_x, neighbor_y):
            """Compute AEP for target farm given neighbor positions."""
            x_all = jnp.concatenate([target_x, neighbor_x])
            y_all = jnp.concatenate([target_y, neighbor_y])
            ws_arr = jnp.full_like(self.wd_arr, self.wind_speed)
            result = self.sim_engine(x_all, y_all, ws_amb=ws_arr, wd_amb=self.wd_arr, ti_amb=0.06)
            power_kw = result.power()
            target_power = power_kw[:, :self.n_target]
            weighted_power = jnp.sum(target_power * self.weights[:, None])
            return weighted_power * 8760 / 1e6

        # For maximizing regret, we want to minimize target AEP with neighbors
        # Regret = AEP_conservative - AEP_liberal (with same neighbors)
        # If we fix target layout and optimize neighbors, we minimize AEP
        self._compute_aep = jax.jit(compute_target_aep)

        # Gradient of AEP w.r.t. neighbor positions
        self._aep_grad = jax.jit(jax.grad(compute_target_aep, argnums=(2, 3)))

        # Combined value and gradient
        self._aep_and_grad = jax.jit(jax.value_and_grad(compute_target_aep, argnums=(2, 3)))

    def _initialize_neighbors(self, angle: float | None = None) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Initialize neighbor positions in an arc around the target.

        Parameters
        ----------
        angle : float or None
            Central angle in radians. If None, uses dominant wind direction.

        Returns
        -------
        tuple[jnp.ndarray, jnp.ndarray]
            Initial (x, y) positions.
        """
        if angle is None:
            dominant_idx = int(jnp.argmax(self.weights))
            dominant_wd = float(self.wd_arr[dominant_idx])
            angle = np.deg2rad(90 - dominant_wd)

        # Place neighbors in an arc upwind
        dist = self.target_radius + self.min_neighbor_distance + 5 * self.rotor_diameter
        arc_span = np.pi / 3  # 60 degree arc

        angles = np.linspace(angle - arc_span/2, angle + arc_span/2, self.n_neighbors)
        x = self.target_center[0] + dist * np.cos(angles)
        y = self.target_center[1] + dist * np.sin(angles)

        return jnp.array(x), jnp.array(y)

    def _apply_constraints(
        self, neighbor_x: jnp.ndarray, neighbor_y: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Project positions to satisfy constraints.

        Enforces:
        1. Minimum distance from target center
        2. Maximum distance from target center
        3. Minimum spacing between neighbors
        """
        cx, cy = self.target_center

        # Distance from target center
        dx = neighbor_x - cx
        dy = neighbor_y - cy
        dist = jnp.sqrt(dx**2 + dy**2)

        # Enforce minimum distance
        min_dist = self.target_radius + self.min_neighbor_distance
        scale_up = jnp.maximum(1.0, min_dist / (dist + 1e-6))

        # Enforce maximum distance
        scale_down = jnp.minimum(1.0, self.search_radius / (dist + 1e-6))

        # Apply distance constraints
        scale = jnp.where(dist < min_dist, scale_up, scale_down)
        new_x = cx + dx * scale
        new_y = cy + dy * scale

        return new_x, new_y

    def compute_regret(
        self,
        neighbor_x: jnp.ndarray,
        neighbor_y: jnp.ndarray,
        liberal_x: jnp.ndarray | None = None,
        liberal_y: jnp.ndarray | None = None,
        conservative_x: jnp.ndarray | None = None,
        conservative_y: jnp.ndarray | None = None,
    ) -> float:
        """Compute regret for given neighbor configuration.

        Parameters
        ----------
        neighbor_x, neighbor_y : jnp.ndarray
            Neighbor turbine positions.
        liberal_x, liberal_y : jnp.ndarray or None
            Pre-optimized liberal layout. If None, uses target layout.
        conservative_x, conservative_y : jnp.ndarray or None
            Pre-optimized conservative layout. If None, uses target layout.

        Returns
        -------
        float
            Regret = AEP_conservative_with_neighbor - AEP_liberal_with_neighbor
        """
        if liberal_x is None:
            liberal_x, liberal_y = self.target_x, self.target_y
        if conservative_x is None:
            conservative_x, conservative_y = self.target_x, self.target_y

        aep_lib = self._compute_aep(liberal_x, liberal_y, neighbor_x, neighbor_y)
        aep_con = self._compute_aep(conservative_x, conservative_y, neighbor_x, neighbor_y)

        return float(aep_con - aep_lib)

    def optimize(
        self,
        n_iterations: int = 200,
        learning_rate: float | None = None,
        initial_angle: float | None = None,
        verbose: bool = True,
        target_layout: tuple[jnp.ndarray, jnp.ndarray] | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, list[float]]:
        """Optimize neighbor positions to minimize target AEP.

        This finds neighbors that cause maximum wake losses to the target farm.
        For computing actual regret, you still need to run inner optimizations.

        Parameters
        ----------
        n_iterations : int
            Number of gradient descent iterations.
        learning_rate : float or None
            Step size (default: rotor_diameter / 5).
        initial_angle : float or None
            Initial neighbor placement angle.
        verbose : bool
            Print progress.
        target_layout : tuple or None
            Fixed target layout (x, y). If None, uses initial target positions.

        Returns
        -------
        tuple
            (optimized_neighbor_x, optimized_neighbor_y, aep_history)
        """
        if learning_rate is None:
            learning_rate = self.rotor_diameter / 5

        if target_layout is None:
            target_x, target_y = self.target_x, self.target_y
        else:
            target_x, target_y = target_layout

        # Initialize
        neighbor_x, neighbor_y = self._initialize_neighbors(initial_angle)
        aep_history = []

        for i in range(n_iterations):
            # Compute AEP and gradients
            aep, (grad_x, grad_y) = self._aep_and_grad(
                target_x, target_y, neighbor_x, neighbor_y
            )
            aep_history.append(float(aep))

            # Gradient descent (minimize AEP = maximize wake impact)
            neighbor_x = neighbor_x - learning_rate * grad_x
            neighbor_y = neighbor_y - learning_rate * grad_y

            # Apply constraints
            neighbor_x, neighbor_y = self._apply_constraints(neighbor_x, neighbor_y)

            if verbose and i % 20 == 0:
                print(f"Iter {i:4d}: target AEP = {aep:.3f} GWh")

        return neighbor_x, neighbor_y, aep_history


def plot_regret_convergence(
    history: OptimizationHistory,
    output_path: str | None = None,
) -> plt.Figure:
    """Plot regret convergence over iterations.

    Parameters
    ----------
    history : OptimizationHistory
        Optimization history.
    output_path : str or None
        If provided, save figure to this path.

    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    iterations = [e.iteration for e in history.entries]
    regrets = [e.regret for e in history.entries]

    # Running best
    best_so_far = []
    current_best = float('-inf')
    for r in regrets:
        if r > current_best:
            current_best = r
        best_so_far.append(current_best)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(iterations, regrets, 'b-', alpha=0.5, label='Current regret')
    ax.plot(iterations, best_so_far, 'r-', linewidth=2, label='Best regret')
    ax.scatter([history.best_entry.iteration], [history.best_entry.regret],
               s=200, c='red', marker='*', zorder=5, label='Overall best')

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Regret (GWh)', fontsize=12)
    ax.set_title('Adversarial Optimization Convergence', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Convergence plot saved to {output_path}")

    return fig

"""Multi-start pooled regret discovery for finding critical phase transitions.

This script runs pooled multi-start optimization to properly estimate design regret
for different neighbor farm configurations (represented as morphable "blobs").

The proper methodology:
1. For each blob configuration, run N multi-start optimizations with liberal
   assumptions (ignoring neighbors).
2. Run N multi-start optimizations with conservative assumptions (with neighbors).
3. Cross-evaluate ALL layouts under BOTH scenarios (neighbors present/absent).
4. Compute regret against pooled global bests.

This ensures regret values reflect true fundamental tradeoffs rather than local
minima artifacts from single-shot optimization.

Extended to support different wind rose types:
- Single direction (baseline)
- Uniform (omnidirectional)
- Von Mises unimodal (realistic with dominant direction)
- Von Mises bimodal (two dominant directions)

Usage:
    pixi run python scripts/run_regret_discovery.py
    pixi run python scripts/run_regret_discovery.py --wind-rose=uniform
    pixi run python scripts/run_regret_discovery.py --wind-rose=von_mises --concentration=2.0
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

jax.config.update("jax_enable_x64", True)

from pixwake import Curve, Turbine, WakeSimulation


# =============================================================================
# Wind Rose Types and Generation
# =============================================================================

WindRoseType = Literal["single", "uniform", "von_mises", "bimodal"]


@dataclass
class WindRoseConfig:
    """Configuration for wind rose generation."""

    rose_type: WindRoseType = "single"
    n_directions: int = 72  # 5-degree bins
    dominant_dir: float = 270.0  # degrees (meteorological convention)
    concentration: float = 2.0  # Von Mises kappa (higher = more focused)
    secondary_dir: float = 90.0  # For bimodal: secondary direction
    secondary_weight: float = 0.3  # For bimodal: relative weight of secondary
    mean_ws: float = 9.0  # Mean wind speed [m/s]
    weibull_k: float = 2.0  # Weibull shape parameter
    speed_varies_with_direction: bool = False  # If True, speed varies with direction

    def __str__(self) -> str:
        if self.rose_type == "single":
            return f"single_{self.dominant_dir:.0f}deg"
        elif self.rose_type == "uniform":
            return f"uniform_{self.n_directions}dir"
        elif self.rose_type == "von_mises":
            return f"von_mises_{self.dominant_dir:.0f}deg_k{self.concentration:.1f}"
        elif self.rose_type == "bimodal":
            return f"bimodal_{self.dominant_dir:.0f}_{self.secondary_dir:.0f}deg"
        return self.rose_type


def generate_wind_rose(
    config: WindRoseConfig,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Generate wind rose arrays (wd, ws, weights) based on configuration.

    Args:
        config: Wind rose configuration specifying type and parameters.

    Returns:
        Tuple of (wd, ws, weights) where:
        - wd: Wind directions in degrees [n_directions]
        - ws: Wind speeds in m/s [n_directions]
        - weights: Probability weights summing to 1.0 [n_directions]
    """
    if config.rose_type == "single":
        return _generate_single_direction(config)
    elif config.rose_type == "uniform":
        return _generate_uniform(config)
    elif config.rose_type == "von_mises":
        return _generate_von_mises(config)
    elif config.rose_type == "bimodal":
        return _generate_bimodal(config)
    else:
        raise ValueError(f"Unknown wind rose type: {config.rose_type}")


def _generate_single_direction(
    config: WindRoseConfig,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Single wind direction (baseline case)."""
    wd = jnp.array([config.dominant_dir])
    ws = jnp.array([config.mean_ws])
    weights = jnp.array([1.0])
    return wd, ws, weights


def _generate_uniform(
    config: WindRoseConfig,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Uniform (omnidirectional) wind rose."""
    wd = jnp.linspace(0, 360, config.n_directions, endpoint=False)
    ws = jnp.full_like(wd, config.mean_ws)
    weights = jnp.ones(config.n_directions) / config.n_directions
    return wd, ws, weights


def _generate_von_mises(
    config: WindRoseConfig,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Von Mises (circular normal) distributed wind rose.

    The Von Mises distribution is the circular analog of the normal distribution,
    commonly used for directional data. It creates a smooth "bell curve" on the
    circle centered on the dominant direction.

    The concentration parameter (kappa) controls the spread:
    - kappa = 0: uniform distribution
    - kappa = 1: mild concentration
    - kappa = 2: moderate concentration (typical for offshore)
    - kappa = 4+: highly concentrated
    """
    wd = jnp.linspace(0, 360, config.n_directions, endpoint=False)
    wd_rad = jnp.deg2rad(wd)
    dom_rad = jnp.deg2rad(config.dominant_dir)

    # Von Mises: exp(kappa * cos(x - mu))
    # Normalize to sum to 1 (omitting the normalization constant 2*pi*I0(kappa))
    unnormalized_weights = jnp.exp(config.concentration * jnp.cos(wd_rad - dom_rad))
    weights = unnormalized_weights / jnp.sum(unnormalized_weights)

    # Wind speed: optionally varies with direction (higher in dominant direction)
    if config.speed_varies_with_direction:
        # Speed is 20% higher in dominant direction
        speed_factor = 1.0 + 0.2 * jnp.cos(wd_rad - dom_rad)
        ws = config.mean_ws * speed_factor
    else:
        ws = jnp.full_like(wd, config.mean_ws)

    return wd, ws, weights


def _generate_bimodal(
    config: WindRoseConfig,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Bimodal wind rose with two dominant directions.

    Common in coastal sites with land-sea breeze effects or sites influenced
    by two different weather systems.
    """
    wd = jnp.linspace(0, 360, config.n_directions, endpoint=False)
    wd_rad = jnp.deg2rad(wd)
    dom_rad = jnp.deg2rad(config.dominant_dir)
    sec_rad = jnp.deg2rad(config.secondary_dir)

    # Primary peak
    primary = jnp.exp(config.concentration * jnp.cos(wd_rad - dom_rad))

    # Secondary peak
    secondary = jnp.exp(config.concentration * jnp.cos(wd_rad - sec_rad))

    # Weighted combination
    combined = (1.0 - config.secondary_weight) * primary + config.secondary_weight * secondary
    weights = combined / jnp.sum(combined)

    ws = jnp.full_like(wd, config.mean_ws)

    return wd, ws, weights


def plot_wind_rose(
    wd: jnp.ndarray,
    ws: jnp.ndarray,
    weights: jnp.ndarray,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    """Plot wind rose as a polar bar chart.

    Args:
        wd: Wind directions in degrees
        ws: Wind speeds (used for color coding)
        weights: Probability weights
        ax: Matplotlib axes (will create if None)
        title: Plot title

    Returns:
        The matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 6))

    # Meteorological convention: 0=N, 90=E, 180=S, 270=W (clockwise)
    # With set_theta_zero_location("N") and set_theta_direction(-1),
    # matplotlib uses the same convention, so no transformation needed
    theta = jnp.deg2rad(wd)

    # Bar width depends on number of directions
    width = 2 * jnp.pi / len(wd) * 0.9

    # Color by wind speed
    colors = plt.cm.Blues(np.array(ws) / np.max(ws))

    bars = ax.bar(np.array(theta), np.array(weights), width=width, color=colors, edgecolor="k", linewidth=0.5)

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)  # Clockwise

    if title:
        ax.set_title(title, fontsize=10, pad=10)

    return ax


from pixwake.deficit import BastankhahGaussianDeficit
from pixwake.optim.adversarial import (
    PooledBlobDiscovery,
    PooledBlobDiscoverySettings,
)
from pixwake.optim.geometry import (
    BSplineBoundary,
    sample_random_blob,
)
from pixwake.optim.soft_packing import create_reference_grid
from pixwake.optim.sgd import SGDSettings


def create_turbine(rotor_diameter: float = 200.0) -> Turbine:
    """Create a 10 MW class turbine."""
    ws = jnp.array([0.0, 4.0, 10.0, 15.0, 25.0])
    power = jnp.array([0.0, 0.0, 10000.0, 10000.0, 0.0])  # 10 MW
    ct = jnp.array([0.0, 0.8, 0.8, 0.4, 0.0])

    return Turbine(
        rotor_diameter=rotor_diameter,
        hub_height=120.0,
        power_curve=Curve(ws=ws, values=power),
        ct_curve=Curve(ws=ws, values=ct),
    )


def run_single_pooled_discovery(
    sim: WakeSimulation,
    target_boundary: jnp.ndarray,
    target_min_spacing: float,
    neighbor_grid: jnp.ndarray,
    init_target_x: jnp.ndarray,
    init_target_y: jnp.ndarray,
    control_points: jnp.ndarray,
    ws: jnp.ndarray,
    wd: jnp.ndarray,
    weights: jnp.ndarray,
    blob_seed: int,
    settings: PooledBlobDiscoverySettings,
) -> dict:
    """Run pooled multi-start discovery for a single blob configuration."""
    print(f"\n{'='*60}", flush=True)
    print(f"Pooled Discovery for Blob {blob_seed}", flush=True)
    print("=" * 60, flush=True)

    start_time = time.time()

    discoverer = PooledBlobDiscovery(
        sim=sim,
        target_boundary=target_boundary,
        target_min_spacing=target_min_spacing,
        neighbor_grid=neighbor_grid,
        ws_amb=ws,
        wd_amb=wd,
        weights=weights,
    )

    result = discoverer.discover(
        init_target_x,
        init_target_y,
        control_points,
        settings=settings,
        seed=blob_seed * 1000,  # Different seed for each blob
    )

    elapsed = time.time() - start_time

    return {
        "blob_seed": blob_seed,
        "control_points": result.control_points.tolist(),
        "global_best_aep_absent": result.global_best_aep_absent,
        "global_best_aep_present": result.global_best_aep_present,
        "min_liberal_regret": result.min_liberal_regret,
        "min_conservative_regret": result.min_conservative_regret,
        "same_best_layout": result.same_best_layout,
        "n_liberal_starts": result.n_liberal_starts,
        "n_conservative_starts": result.n_conservative_starts,
        "best_liberal_x": result.best_liberal_layout[0].tolist(),
        "best_liberal_y": result.best_liberal_layout[1].tolist(),
        "best_conservative_x": result.best_conservative_layout[0].tolist(),
        "best_conservative_y": result.best_conservative_layout[1].tolist(),
        "all_layouts": result.all_layouts,
        "elapsed_seconds": elapsed,
    }


def run_multistart_pooled_discovery(
    n_blobs: int = 10,
    n_starts_per_strategy: int = 10,
    output_dir: str | None = None,
    wind_rose_config: WindRoseConfig | None = None,
):
    """Run pooled multi-start discovery across multiple blob configurations.

    Args:
        n_blobs: Number of random blob configurations to test.
        n_starts_per_strategy: Number of optimization starts per strategy (liberal/conservative).
        output_dir: Output directory path. If None, auto-generated based on wind rose type.
        wind_rose_config: Wind rose configuration. If None, uses single direction (270°).
    """
    # Default wind rose: single direction (baseline)
    if wind_rose_config is None:
        wind_rose_config = WindRoseConfig(rose_type="single", dominant_dir=270.0, mean_ws=9.0)

    # Auto-generate output directory if not specified
    if output_dir is None:
        output_dir = f"analysis/blob_discovery_{wind_rose_config}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configuration
    D = 200.0  # Rotor diameter
    target_size = 16 * D  # 16D x 16D target farm
    min_spacing = 4 * D  # 4D minimum spacing

    # Create turbine and simulation
    turbine = create_turbine(D)
    deficit = BastankhahGaussianDeficit(k=0.04)
    sim = WakeSimulation(turbine, deficit)

    # Target farm boundary
    target_boundary = jnp.array([
        [0.0, 0.0],
        [target_size, 0.0],
        [target_size, target_size],
        [0.0, target_size],
    ])

    # Initial target positions (4x4 grid)
    n_target = 16
    grid_side = int(np.sqrt(n_target))
    spacing = target_size / (grid_side + 1)
    xs = jnp.array([spacing * (i + 1) for i in range(grid_side)] * grid_side)
    ys = jnp.array([spacing * (j + 1) for j in range(grid_side) for _ in range(grid_side)])
    init_target_x = xs
    init_target_y = ys

    # Neighbor reference grid (upwind of target)
    neighbor_center = (-6 * D, target_size / 2)  # 6D upwind
    neighbor_extent = 6 * D
    neighbor_spacing = 3 * D
    neighbor_grid = create_reference_grid(neighbor_center, neighbor_extent, neighbor_spacing)

    # Generate wind rose
    print(f"Generating wind rose: {wind_rose_config}", flush=True)
    wd, ws, weights = generate_wind_rose(wind_rose_config)
    print(f"  Directions: {len(wd)}, Speed range: {float(ws.min()):.1f}-{float(ws.max()):.1f} m/s", flush=True)

    # Save wind rose configuration
    wind_rose_data = {
        "type": wind_rose_config.rose_type,
        "n_directions": int(len(wd)),
        "dominant_dir": wind_rose_config.dominant_dir,
        "concentration": wind_rose_config.concentration,
        "mean_ws": wind_rose_config.mean_ws,
        "wd": [float(x) for x in wd],
        "ws": [float(x) for x in ws],
        "weights": [float(x) for x in weights],
    }
    with open(output_dir / "wind_rose_config.json", "w") as f:
        json.dump(wind_rose_data, f, indent=2)

    # Discovery settings with proper pooled methodology
    # Adjust max_iter based on wind rose complexity (more directions = slower convergence)
    n_dirs = len(wd)
    max_iter = 3000 if n_dirs <= 12 else 2000 if n_dirs <= 36 else 1500
    settings = PooledBlobDiscoverySettings(
        n_starts=n_starts_per_strategy,
        sgd_settings=SGDSettings(max_iter=max_iter, learning_rate=D / 5),
        verbose=True,
    )

    print("=" * 60, flush=True)
    print("Pooled Multi-Start Blob Discovery", flush=True)
    print("=" * 60, flush=True)
    print(f"Wind rose: {wind_rose_config.rose_type} ({len(wd)} directions)", flush=True)
    print(f"Target farm: {n_target} turbines in {target_size/D:.0f}D x {target_size/D:.0f}D", flush=True)
    print(f"Neighbor grid: {len(neighbor_grid)} potential positions", flush=True)
    print(f"Number of blob configurations: {n_blobs}", flush=True)
    print(f"Starts per strategy per blob: {n_starts_per_strategy}", flush=True)
    print(f"Total optimizations: {n_blobs * n_starts_per_strategy * 2}", flush=True)
    print(f"Max iterations per optimization: {max_iter}", flush=True)
    print("=" * 60, flush=True)

    results = []
    key = jax.random.PRNGKey(42)

    for blob_seed in range(n_blobs):
        # Generate random blob configuration
        key, subkey = jax.random.split(key)
        control_points = sample_random_blob(
            subkey,
            center_bounds=((-10 * D, -4 * D), (target_size * 0.2, target_size * 0.8)),
            size_bounds=(5 * D, 10 * D),
            aspect_ratio_bounds=(0.6, 1.6),
            n_control=4,
        )

        try:
            result = run_single_pooled_discovery(
                sim=sim,
                target_boundary=target_boundary,
                target_min_spacing=min_spacing,
                neighbor_grid=neighbor_grid,
                init_target_x=init_target_x,
                init_target_y=init_target_y,
                control_points=control_points,
                ws=ws,
                wd=wd,
                weights=weights,
                blob_seed=blob_seed,
                settings=settings,
            )
            results.append(result)

            # Save intermediate results
            with open(output_dir / "results.json", "w") as f:
                json.dump(results, f, indent=2)

        except Exception as e:
            print(f"Discovery for blob {blob_seed} failed: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY (Pooled Methodology)")
    print("=" * 60)

    if not results:
        print("No results to summarize.")
        return results

    min_lib_regrets = [r["min_liberal_regret"] for r in results]
    min_con_regrets = [r["min_conservative_regret"] for r in results]
    tradeoffs = [not r["same_best_layout"] for r in results]

    print(f"Completed: {len(results)}/{n_blobs}")
    print(f"Min Liberal Regret range: {min(min_lib_regrets):.2f} - {max(min_lib_regrets):.2f} GWh")
    print(f"Min Conservative Regret range: {min(min_con_regrets):.2f} - {max(min_con_regrets):.2f} GWh")
    print(f"Configs with TRUE TRADEOFF: {sum(tradeoffs)}/{len(results)}")

    # Plot each blob result
    regrets = []
    for result in results:
        reg, n_pareto = plot_blob_result(
            result, target_boundary, neighbor_grid, D, output_dir,
            wd=wd, ws=ws, weights=weights
        )
        regrets.append(reg)
        print(f"Blob {result['blob_seed']}: regret={reg:.2f} GWh, {n_pareto} Pareto points")

    # Also save simple pareto_frontier.png for first blob
    plot_pareto_frontier(results, output_dir)

    print(f"\nMax regret across blobs: {max(regrets):.2f} GWh")

    return results


def plot_pooled_results(
    results: list[dict],
    target_boundary: jnp.ndarray,
    neighbor_grid: jnp.ndarray,
    D: float,
    output_dir: Path,
):
    """Plot the best result (highest min liberal regret with tradeoff)."""
    # Find most interesting result (tradeoff with significant regret)
    tradeoff_results = [r for r in results if not r["same_best_layout"]]
    if tradeoff_results:
        best = max(tradeoff_results, key=lambda r: r["min_liberal_regret"])
    else:
        best = max(results, key=lambda r: r["min_liberal_regret"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    control_points = jnp.array(best["control_points"])
    spline = BSplineBoundary(control_points)

    # Plot 1: Layout comparison (top-left)
    ax = axes[0, 0]
    tradeoff_str = "TRADEOFF" if not best["same_best_layout"] else "NO TRADEOFF"
    ax.set_title(f"Best Layouts for Blob {best['blob_seed']}\n({tradeoff_str})")

    # Target boundary
    ax.add_patch(Polygon(
        np.array(target_boundary),
        fill=False,
        edgecolor="gray",
        linewidth=2,
        linestyle="--",
        label="Target boundary",
    ))

    # Best liberal layout (minimizes liberal regret)
    ax.scatter(
        best["best_liberal_x"],
        best["best_liberal_y"],
        s=80,
        c="blue",
        marker="o",
        label="Best for neighbors present",
        zorder=5,
    )

    # Best conservative layout (minimizes conservative regret)
    ax.scatter(
        best["best_conservative_x"],
        best["best_conservative_y"],
        s=80,
        c="green",
        marker="s",
        label="Best for neighbors absent",
        zorder=4,
    )

    # Neighbor grid
    mask = spline.contains(neighbor_grid, temperature=10.0)
    scatter = ax.scatter(
        neighbor_grid[:, 0],
        neighbor_grid[:, 1],
        c=np.array(mask),
        cmap="Reds",
        s=40,
        alpha=0.7,
        vmin=0,
        vmax=1,
    )
    plt.colorbar(scatter, ax=ax, label="Neighbor weight")

    # Blob boundary
    t = jnp.linspace(0, 1, 100)
    boundary_pts = spline.evaluate(t)
    ax.plot(boundary_pts[:, 0], boundary_pts[:, 1], "r-", linewidth=2, label="Blob boundary")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Plot 2: AEP distribution from pooled layouts (top-right)
    ax = axes[0, 1]
    ax.set_title("Pooled Layout Performance")

    liberal_layouts = [l for l in best["all_layouts"] if l["strategy"] == "liberal"]
    conservative_layouts = [l for l in best["all_layouts"] if l["strategy"] == "conservative"]

    # Scatter plot of (AEP_absent, AEP_present) for all layouts
    lib_absent = [l["aep_absent"] for l in liberal_layouts]
    lib_present = [l["aep_present"] for l in liberal_layouts]
    con_absent = [l["aep_absent"] for l in conservative_layouts]
    con_present = [l["aep_present"] for l in conservative_layouts]

    ax.scatter(lib_absent, lib_present, c="blue", alpha=0.7, s=60, label="Liberal starts", marker="o")
    ax.scatter(con_absent, con_present, c="green", alpha=0.7, s=60, label="Conservative starts", marker="s")

    # Mark global bests
    ax.axvline(best["global_best_aep_absent"], color="gray", linestyle="--", alpha=0.5, label="Global best absent")
    ax.axhline(best["global_best_aep_present"], color="gray", linestyle=":", alpha=0.5, label="Global best present")

    ax.set_xlabel("AEP when neighbors absent [GWh]")
    ax.set_ylabel("AEP when neighbors present [GWh]")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 3: Regret distribution (bottom-left)
    ax = axes[1, 0]
    ax.set_title("Regret Distribution Across Pool")

    lib_regrets = [l["liberal_regret"] for l in best["all_layouts"]]
    con_regrets = [l["conservative_regret"] for l in best["all_layouts"]]

    ax.scatter(lib_regrets, con_regrets, c="steelblue", alpha=0.7, s=60)
    ax.axhline(0, color="gray", linestyle="-", linewidth=0.5)
    ax.axvline(0, color="gray", linestyle="-", linewidth=0.5)

    # Mark minimum regrets
    ax.axvline(best["min_liberal_regret"], color="coral", linestyle="--", label=f"Min lib regret: {best['min_liberal_regret']:.2f}")
    ax.axhline(best["min_conservative_regret"], color="skyblue", linestyle="--", label=f"Min con regret: {best['min_conservative_regret']:.2f}")

    ax.set_xlabel("Liberal Regret [GWh]")
    ax.set_ylabel("Conservative Regret [GWh]")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 4: Summary statistics (bottom-right)
    ax = axes[1, 1]
    ax.set_title("Pooled Regret Summary")

    labels = ["Min Liberal\nRegret", "Min Conservative\nRegret"]
    values = [best["min_liberal_regret"], best["min_conservative_regret"]]
    colors = ["coral", "skyblue"]
    bars = ax.bar(labels, values, color=colors, edgecolor="black")
    ax.set_ylabel("Minimum Regret [GWh]")
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{val:.2f}",
            ha="center",
            fontsize=11,
        )

    # Add explanation
    ax.text(
        0.5, -0.2,
        f"Pooled from {best['n_liberal_starts']} liberal + {best['n_conservative_starts']} conservative starts\n"
        f"Global best absent: {best['global_best_aep_absent']:.1f} GWh, present: {best['global_best_aep_present']:.1f} GWh",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        style="italic",
    )

    plt.tight_layout()
    plt.savefig(output_dir / f"discovery_seed{best['blob_seed']}.png", dpi=150)
    plt.close()

    blob_seed = best["blob_seed"]
    print(f"Plot saved to {output_dir / f'discovery_seed{blob_seed}.png'}")


def compute_pareto_and_regret(layouts: list[dict]) -> tuple[np.ndarray, float, int, int]:
    """Compute Pareto set and regret for a set of layouts.

    Returns: (pareto_mask, regret, liberal_opt_idx, conservative_opt_idx)
    """
    aep_absent = np.array([l["aep_absent"] for l in layouts])
    aep_present = np.array([l["aep_present"] for l in layouts])

    # Find Pareto-optimal points (maximizing both AEPs)
    pareto_mask = np.ones(len(layouts), dtype=bool)
    for i in range(len(layouts)):
        for j in range(len(layouts)):
            if i != j:
                if aep_absent[j] >= aep_absent[i] and aep_present[j] >= aep_present[i]:
                    if aep_absent[j] > aep_absent[i] or aep_present[j] > aep_present[i]:
                        pareto_mask[i] = False
                        break

    # Compute regret
    pareto_indices = np.where(pareto_mask)[0]
    if len(pareto_indices) <= 1:
        regret = 0.0
        lib_opt_idx = pareto_indices[0] if len(pareto_indices) == 1 else 0
        con_opt_idx = lib_opt_idx
    else:
        # Liberal-optimal: max AEP_absent among Pareto points
        pareto_absent = aep_absent[pareto_mask]
        pareto_present = aep_present[pareto_mask]
        lib_opt_local = np.argmax(pareto_absent)
        con_opt_local = np.argmax(pareto_present)
        lib_opt_idx = pareto_indices[lib_opt_local]
        con_opt_idx = pareto_indices[con_opt_local]
        regret = pareto_present[con_opt_local] - pareto_present[lib_opt_local]

    return pareto_mask, regret, lib_opt_idx, con_opt_idx


def plot_blob_result(
    result: dict,
    target_boundary: jnp.ndarray,
    neighbor_grid: jnp.ndarray,
    D: float,
    output_dir: Path,
    wd: jnp.ndarray | None = None,
    ws: jnp.ndarray | None = None,
    weights: jnp.ndarray | None = None,
):
    """Plot single blob result: Pareto frontier, liberal layout, conservative layout, wind rose.

    Args:
        result: Discovery result dictionary
        target_boundary: Target farm boundary vertices
        neighbor_grid: Neighbor grid positions
        D: Rotor diameter
        output_dir: Output directory for plots
        wd: Wind directions (degrees). If None, uses single direction.
        ws: Wind speeds (m/s). If None, uses 9.0.
        weights: Probability weights. If None, uses uniform.
    """
    from pixwake.optim.geometry import BSplineBoundary

    # Default to single direction if not provided
    if wd is None:
        wd = jnp.array([270.0])
        ws = jnp.array([9.0])
        weights = jnp.array([1.0])

    layouts = result["all_layouts"]
    blob_seed = result["blob_seed"]
    control_points = jnp.array(result["control_points"])

    # Compute Pareto set and regret
    pareto_mask, regret, lib_opt_idx, con_opt_idx = compute_pareto_and_regret(layouts)

    aep_absent = np.array([l["aep_absent"] for l in layouts])
    aep_present = np.array([l["aep_present"] for l in layouts])
    strategies = np.array([l["strategy"] for l in layouts])

    lib_layout = layouts[lib_opt_idx]
    con_layout = layouts[con_opt_idx]

    spline = BSplineBoundary(control_points)
    mask = spline.contains(neighbor_grid, temperature=10.0)
    t = jnp.linspace(0, 1, 100)
    boundary_pts = spline.evaluate(t)

    # Create figure with 4 panels
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Panel 1: Pareto frontier
    ax = axes[0]
    liberal_mask = strategies == "liberal"
    conservative_mask = strategies == "conservative"

    ax.scatter(aep_absent[liberal_mask], aep_present[liberal_mask], s=40, c="steelblue",
               alpha=0.6, label=f"Liberal ({liberal_mask.sum()})", zorder=2)
    ax.scatter(aep_absent[conservative_mask], aep_present[conservative_mask], s=40, c="forestgreen",
               alpha=0.6, label=f"Conservative ({conservative_mask.sum()})", zorder=2)

    # Pareto points with hollow circles
    pareto_absent = aep_absent[pareto_mask]
    pareto_present = aep_present[pareto_mask]
    ax.scatter(pareto_absent, pareto_present, s=120, facecolors='none', edgecolors='black',
               linewidths=2, label=f"Pareto ({pareto_mask.sum()})", zorder=4)

    # Connect Pareto points
    if len(pareto_absent) > 1:
        sort_idx = np.argsort(pareto_absent)
        ax.plot(pareto_absent[sort_idx], pareto_present[sort_idx], "k--", linewidth=1.5, alpha=0.7, zorder=3)

    ax.set_xlabel("AEP (absent) [GWh]", fontsize=9)
    ax.set_ylabel("AEP (present) [GWh]", fontsize=9)
    ax.set_title(f"Pareto: Regret={regret:.1f} GWh", fontsize=10)
    ax.legend(loc="lower right", fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 2: Liberal-optimal layout
    ax = axes[1]
    # Blob and neighbors
    ax.scatter(neighbor_grid[:, 0] / D, neighbor_grid[:, 1] / D, c=np.array(mask),
               cmap="Reds", s=20, alpha=0.5, vmin=0, vmax=1)
    ax.plot(boundary_pts[:, 0] / D, boundary_pts[:, 1] / D, "r-", linewidth=1.5)
    # Boundary
    boundary_closed = np.vstack([target_boundary, target_boundary[0]])
    ax.plot(boundary_closed[:, 0] / D, boundary_closed[:, 1] / D, 'k--', linewidth=0.5)
    # Layout
    ax.scatter(np.array(lib_layout["x"]) / D, np.array(lib_layout["y"]) / D,
               s=50, c="steelblue", edgecolors='black', linewidths=0.5, zorder=5)
    ax.set_xlim(-12, 18)
    ax.set_ylim(-2, 18)
    ax.set_aspect("equal")
    ax.set_xlabel("x [D]", fontsize=9)
    ax.set_title(f"Liberal-opt\nabsent={lib_layout['aep_absent']:.0f}, present={lib_layout['aep_present']:.0f}", fontsize=9)
    ax.annotate('', xy=(8, 8), xytext=(5, 8), arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

    # Panel 3: Conservative-optimal layout
    ax = axes[2]
    ax.scatter(neighbor_grid[:, 0] / D, neighbor_grid[:, 1] / D, c=np.array(mask),
               cmap="Reds", s=20, alpha=0.5, vmin=0, vmax=1)
    ax.plot(boundary_pts[:, 0] / D, boundary_pts[:, 1] / D, "r-", linewidth=1.5)
    ax.plot(boundary_closed[:, 0] / D, boundary_closed[:, 1] / D, 'k--', linewidth=0.5)
    ax.scatter(np.array(con_layout["x"]) / D, np.array(con_layout["y"]) / D,
               s=50, c="forestgreen", edgecolors='black', linewidths=0.5, zorder=5)
    ax.set_xlim(-12, 18)
    ax.set_ylim(-2, 18)
    ax.set_aspect("equal")
    ax.set_xlabel("x [D]", fontsize=9)
    ax.set_title(f"Conservative-opt\nabsent={con_layout['aep_absent']:.0f}, present={con_layout['aep_present']:.0f}", fontsize=9)
    ax.annotate('', xy=(8, 8), xytext=(5, 8), arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

    # Panel 4: Wind rose (polar plot)
    ax = axes[3]
    ax.remove()  # Remove rectangular axes
    ax = fig.add_subplot(1, 4, 4, projection="polar")

    # Plot wind rose
    # Meteorological: 0=N, 90=E, 180=S, 270=W (clockwise)
    # With set_theta_zero_location("N") and set_theta_direction(-1), no transform needed
    theta = jnp.deg2rad(wd)
    width = 2 * jnp.pi / max(len(wd), 1) * 0.85 if len(wd) > 1 else 0.3

    # Normalize weights for visualization
    weights_norm = np.array(weights) / np.max(weights) if len(weights) > 1 else np.array([1.0])

    bars = ax.bar(
        np.array(theta), weights_norm, width=width,
        color="steelblue", edgecolor="k", linewidth=0.3, alpha=0.7
    )
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)  # Clockwise
    ax.set_ylim(0, 1.2)
    ax.set_yticks([])

    # Add cardinal directions
    ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
    ax.set_xticklabels(["N", "E", "S", "W"], fontsize=8)
    ax.set_title(f"Wind Rose ({len(wd)} dir)", fontsize=9)

    plt.suptitle(f"Blob {blob_seed}", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f"blob_{blob_seed}.png", dpi=150, facecolor='white')
    plt.close()

    return regret, pareto_mask.sum()


def plot_pareto_frontier(results: list[dict], output_dir: Path):
    """Plot Pareto frontier for each blob and summary."""
    if not results:
        print("No results to plot")
        return

    # For now, just plot the first blob as example
    # In full analysis, would plot each blob separately
    result = results[0]
    layouts = result["all_layouts"]

    pareto_mask, regret, lib_opt_idx, con_opt_idx = compute_pareto_and_regret(layouts)

    aep_absent = np.array([l["aep_absent"] for l in layouts])
    aep_present = np.array([l["aep_present"] for l in layouts])
    strategies = np.array([l["strategy"] for l in layouts])

    fig, ax = plt.subplots(figsize=(8, 6))

    liberal_mask = strategies == "liberal"
    conservative_mask = strategies == "conservative"

    ax.scatter(aep_absent[liberal_mask], aep_present[liberal_mask], s=50, c="steelblue",
               alpha=0.6, label=f"Liberal ({liberal_mask.sum()})", zorder=2)
    ax.scatter(aep_absent[conservative_mask], aep_present[conservative_mask], s=50, c="forestgreen",
               alpha=0.6, label=f"Conservative ({conservative_mask.sum()})", zorder=2)

    # Pareto points with hollow circles
    pareto_absent = aep_absent[pareto_mask]
    pareto_present = aep_present[pareto_mask]
    ax.scatter(pareto_absent, pareto_present, s=150, facecolors='none', edgecolors='black',
               linewidths=2, label=f"Pareto ({pareto_mask.sum()})", zorder=4)

    # Connect Pareto points
    if len(pareto_absent) > 1:
        sort_idx = np.argsort(pareto_absent)
        ax.plot(pareto_absent[sort_idx], pareto_present[sort_idx], "k--", linewidth=1.5, alpha=0.7, zorder=3)

    ax.set_xlabel("AEP (neighbors absent) [GWh]", fontsize=11)
    ax.set_ylabel("AEP (neighbors present) [GWh]", fontsize=11)
    ax.set_title(f"Blob 0: {pareto_mask.sum()} Pareto points, Regret = {regret:.2f} GWh", fontsize=12)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "pareto_frontier.png", dpi=150, facecolor='white')
    plt.close()

    print(f"Pareto frontier saved to {output_dir / 'pareto_frontier.png'}")
    print(f"Pareto-optimal layouts: {pareto_mask.sum()}")
    print(f"Regret: {regret:.2f} GWh")


def run_wind_rose_comparison(
    n_blobs: int = 5,
    n_starts_per_strategy: int = 5,
    output_base_dir: str = "analysis/wind_rose_comparison",
) -> dict[str, list[dict]]:
    """Run discovery across multiple wind rose types to characterize effects.

    This function systematically compares how different wind rose configurations
    affect design regret. It runs the same blob configurations under each wind
    rose type to enable fair comparison.

    Args:
        n_blobs: Number of blob configurations per wind rose type.
        n_starts_per_strategy: Optimization starts per strategy.
        output_base_dir: Base directory for all outputs.

    Returns:
        Dictionary mapping wind rose type names to their results.
    """
    output_base = Path(output_base_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    # Define wind rose configurations to compare
    wind_rose_configs = [
        WindRoseConfig(rose_type="single", dominant_dir=270.0, mean_ws=9.0),
        WindRoseConfig(rose_type="uniform", n_directions=24, mean_ws=9.0),
        WindRoseConfig(rose_type="von_mises", n_directions=24, dominant_dir=270.0, concentration=1.0),
        WindRoseConfig(rose_type="von_mises", n_directions=24, dominant_dir=270.0, concentration=2.0),
        WindRoseConfig(rose_type="von_mises", n_directions=24, dominant_dir=270.0, concentration=4.0),
        WindRoseConfig(rose_type="bimodal", n_directions=24, dominant_dir=270.0, secondary_dir=90.0, concentration=2.0),
    ]

    all_results = {}

    for config in wind_rose_configs:
        config_name = str(config)
        print(f"\n{'#'*60}")
        print(f"# Wind Rose: {config_name}")
        print(f"{'#'*60}\n")

        output_dir = output_base / config_name
        results = run_multistart_pooled_discovery(
            n_blobs=n_blobs,
            n_starts_per_strategy=n_starts_per_strategy,
            output_dir=str(output_dir),
            wind_rose_config=config,
        )
        all_results[config_name] = results

    # Generate comparison summary
    _generate_wind_rose_comparison_summary(all_results, output_base)

    return all_results


def _generate_wind_rose_comparison_summary(
    all_results: dict[str, list[dict]],
    output_dir: Path,
):
    """Generate summary plots comparing regret across wind rose types."""
    if not all_results:
        print("No results to compare.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Collect data
    rose_types = []
    mean_lib_regrets = []
    mean_con_regrets = []
    max_lib_regrets = []
    max_con_regrets = []

    for rose_name, results in all_results.items():
        if not results:
            continue
        rose_types.append(rose_name.replace("_", "\n"))
        lib_regrets = [r["min_liberal_regret"] for r in results]
        con_regrets = [r["min_conservative_regret"] for r in results]
        mean_lib_regrets.append(np.mean(lib_regrets))
        mean_con_regrets.append(np.mean(con_regrets))
        max_lib_regrets.append(np.max(lib_regrets))
        max_con_regrets.append(np.max(con_regrets))

    x = np.arange(len(rose_types))
    width = 0.35

    # Plot 1: Mean regrets
    ax = axes[0]
    bars1 = ax.bar(x - width / 2, mean_lib_regrets, width, label="Liberal Regret", color="coral")
    bars2 = ax.bar(x + width / 2, mean_con_regrets, width, label="Conservative Regret", color="skyblue")
    ax.set_ylabel("Mean Minimum Regret [GWh]")
    ax.set_title("Mean Regret by Wind Rose Type")
    ax.set_xticks(x)
    ax.set_xticklabels(rose_types, fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Plot 2: Max regrets (worst-case)
    ax = axes[1]
    bars1 = ax.bar(x - width / 2, max_lib_regrets, width, label="Liberal Regret", color="coral")
    bars2 = ax.bar(x + width / 2, max_con_regrets, width, label="Conservative Regret", color="skyblue")
    ax.set_ylabel("Maximum Regret [GWh]")
    ax.set_title("Maximum Regret by Wind Rose Type")
    ax.set_xticks(x)
    ax.set_xticklabels(rose_types, fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_dir / "wind_rose_comparison.png", dpi=150, facecolor="white")
    plt.close()

    # Save summary as JSON
    summary = {
        "rose_types": list(all_results.keys()),
        "mean_liberal_regret": mean_lib_regrets,
        "mean_conservative_regret": mean_con_regrets,
        "max_liberal_regret": max_lib_regrets,
        "max_conservative_regret": max_con_regrets,
    }
    with open(output_dir / "comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nComparison summary saved to {output_dir / 'wind_rose_comparison.png'}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run pooled regret discovery with configurable wind roses.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--wind-rose", "-w",
        choices=["single", "uniform", "von_mises", "bimodal", "comparison"],
        default="single",
        help="Wind rose type (default: single). Use 'comparison' to run all types.",
    )
    parser.add_argument(
        "--n-directions", "-d",
        type=int,
        default=24,
        help="Number of wind directions (default: 24, i.e., 15-degree bins).",
    )
    parser.add_argument(
        "--dominant-dir",
        type=float,
        default=270.0,
        help="Dominant wind direction in degrees (default: 270, i.e., West).",
    )
    parser.add_argument(
        "--concentration", "-k",
        type=float,
        default=2.0,
        help="Von Mises concentration parameter kappa (default: 2.0).",
    )
    parser.add_argument(
        "--secondary-dir",
        type=float,
        default=90.0,
        help="Secondary direction for bimodal rose (default: 90, i.e., East).",
    )
    parser.add_argument(
        "--mean-ws",
        type=float,
        default=9.0,
        help="Mean wind speed in m/s (default: 9.0).",
    )
    parser.add_argument(
        "--n-blobs",
        type=int,
        default=10,
        help="Number of blob configurations to test (default: 10).",
    )
    parser.add_argument(
        "--n-starts",
        type=int,
        default=10,
        help="Number of optimization starts per strategy (default: 10).",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Output directory (default: auto-generated based on wind rose type).",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.wind_rose == "comparison":
        # Run comparison across all wind rose types
        run_wind_rose_comparison(
            n_blobs=args.n_blobs,
            n_starts_per_strategy=args.n_starts,
            output_base_dir=args.output_dir or "analysis/wind_rose_comparison",
        )
    else:
        # Run with single wind rose configuration
        config = WindRoseConfig(
            rose_type=args.wind_rose,
            n_directions=args.n_directions,
            dominant_dir=args.dominant_dir,
            concentration=args.concentration,
            secondary_dir=args.secondary_dir,
            mean_ws=args.mean_ws,
        )
        run_multistart_pooled_discovery(
            n_blobs=args.n_blobs,
            n_starts_per_strategy=args.n_starts,
            output_dir=args.output_dir,
            wind_rose_config=config,
        )

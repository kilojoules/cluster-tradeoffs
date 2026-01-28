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

Usage:
    pixi run python scripts/run_regret_discovery.py
"""

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

jax.config.update("jax_enable_x64", True)

from pixwake import Curve, Turbine, WakeSimulation
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
    output_dir: str = "analysis/blob_discovery",
):
    """Run pooled multi-start discovery across multiple blob configurations."""
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

    # Wind conditions - single direction for fastest computation
    wd = jnp.array([270.0])  # Wind from west
    ws = jnp.array([9.0])
    weights = jnp.array([1.0])

    # Discovery settings with proper pooled methodology
    settings = PooledBlobDiscoverySettings(
        n_starts=n_starts_per_strategy,
        sgd_settings=SGDSettings(max_iter=3000, learning_rate=D / 5),
        verbose=True,
    )

    print("=" * 60, flush=True)
    print("Pooled Multi-Start Blob Discovery", flush=True)
    print("=" * 60, flush=True)
    print(f"Target farm: {n_target} turbines in {target_size/D:.0f}D x {target_size/D:.0f}D", flush=True)
    print(f"Neighbor grid: {len(neighbor_grid)} potential positions", flush=True)
    print(f"Number of blob configurations: {n_blobs}", flush=True)
    print(f"Starts per strategy per blob: {n_starts_per_strategy}", flush=True)
    print(f"Total optimizations: {n_blobs * n_starts_per_strategy * 2}", flush=True)
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
        reg, n_pareto = plot_blob_result(result, target_boundary, neighbor_grid, D, output_dir)
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


def plot_blob_result(result: dict, target_boundary: jnp.ndarray, neighbor_grid: jnp.ndarray,
                     D: float, output_dir: Path):
    """Plot single blob result: Pareto frontier, liberal layout, conservative layout, wind rose."""
    from pixwake.optim.geometry import BSplineBoundary

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

    # Panel 4: Wind rose
    ax = axes[3]
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=0.5)
    ax.annotate('', xy=(1, 0), xytext=(0, 0), arrowprops=dict(arrowstyle='->', lw=2, color='steelblue'))
    ax.text(0, -0.3, "270° (West)", ha='center', fontsize=9)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Wind Rose", fontsize=10)

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


if __name__ == "__main__":
    run_multistart_pooled_discovery(n_blobs=10, n_starts_per_strategy=10)

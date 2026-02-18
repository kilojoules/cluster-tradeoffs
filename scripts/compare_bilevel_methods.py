"""Compare IFT bilevel results with blob discovery (random multi-start) results.

Loads IFT results from analysis/bilevel_ift/results.json and blob discovery
results from blob_discovery/results.json, produces comparison bar chart and
layout overlay.

Usage:
    pixi run python scripts/compare_bilevel_methods.py
"""

import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt


def load_ift_results(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_blob_results(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def main():
    output_dir = Path("analysis/bilevel_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    ift_path = Path("analysis/bilevel_ift/results.json")
    blob_path = Path("blob_discovery/results.json")

    if not ift_path.exists():
        print(f"IFT results not found at {ift_path}")
        print("Run scripts/run_bilevel_ift.py first.")
        return

    if not blob_path.exists():
        print(f"Blob discovery results not found at {blob_path}")
        return

    ift = load_ift_results(ift_path)
    blobs = load_blob_results(blob_path)

    print("=" * 60)
    print("Bilevel Method Comparison")
    print("=" * 60)

    # IFT results
    ift_regret = ift["regret"]
    ift_liberal = ift["liberal_aep"]
    ift_conservative = ift["conservative_aep"]
    ift_pct = ift.get("regret_pct", ift_regret / ift_liberal * 100 if ift_liberal > 0 else 0)

    print(f"\nIFT Bilevel Search:")
    print(f"  Liberal AEP:       {ift_liberal:.2f} GWh")
    print(f"  Conservative AEP:  {ift_conservative:.2f} GWh")
    print(f"  Regret:            {ift_regret:.2f} GWh ({ift_pct:.1f}%)")

    # Blob discovery: find maximum regret across all blobs
    blob_regrets = []
    for b in blobs:
        # Regret = best_absent - best_present (how much neighbors hurt)
        regret = b["global_best_aep_absent"] - b["global_best_aep_present"]
        blob_regrets.append({
            "seed": b["blob_seed"],
            "regret": regret,
            "aep_absent": b["global_best_aep_absent"],
            "aep_present": b["global_best_aep_present"],
            "same_best": b["same_best_layout"],
        })

    blob_regrets.sort(key=lambda x: x["regret"], reverse=True)
    best_blob = blob_regrets[0]

    print(f"\nBlob Discovery (best of {len(blobs)} blobs):")
    print(f"  Best blob seed:    {best_blob['seed']}")
    print(f"  AEP (no neighbor): {best_blob['aep_absent']:.2f} GWh")
    print(f"  AEP (neighbor):    {best_blob['aep_present']:.2f} GWh")
    print(f"  Regret:            {best_blob['regret']:.2f} GWh")
    print(f"  Same best layout:  {best_blob['same_best']}")

    # --- Plot 1: Bar chart comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Regret comparison bar chart
    ax = axes[0]
    methods = ["IFT Bilevel\n(gradient-based)", f"Blob Discovery\n(best of {len(blobs)})"]
    regrets = [ift_regret, best_blob["regret"]]
    colors = ["#2196F3", "#FF9800"]
    bars = ax.bar(methods, regrets, color=colors, width=0.5, edgecolor="black", linewidth=0.5)

    # Add value labels on bars
    for bar, val in zip(bars, regrets):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.2f} GWh", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("Regret (GWh)")
    ax.set_title("Maximum Regret by Method")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, max(regrets) * 1.3)

    # All blob regrets distribution
    ax = axes[1]
    all_regrets = [b["regret"] for b in blob_regrets]
    ax.hist(all_regrets, bins=max(5, len(all_regrets) // 2), color="#FF9800",
            alpha=0.7, edgecolor="black", linewidth=0.5, label="Blob configs")
    ax.axvline(ift_regret, color="#2196F3", linewidth=2, linestyle="--",
               label=f"IFT result ({ift_regret:.2f} GWh)")
    ax.axvline(best_blob["regret"], color="#FF9800", linewidth=2, linestyle="-",
               label=f"Best blob ({best_blob['regret']:.2f} GWh)")
    ax.set_xlabel("Regret (GWh)")
    ax.set_ylabel("Count")
    ax.set_title("Regret Distribution")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "method_comparison.png", dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {output_dir / 'method_comparison.png'}")
    plt.close()

    # --- Plot 2: Layout overlay ---
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    D = ift.get("D", 200.0)
    target_size = ift.get("target_size", 16 * D)

    # Target boundary
    bnd = np.array([
        [0, 0], [target_size, 0], [target_size, target_size], [0, target_size], [0, 0]
    ])
    ax.plot(bnd[:, 0], bnd[:, 1], "k-", linewidth=2, label="Target boundary")

    # IFT liberal layout
    ax.scatter(ift["liberal_x"], ift["liberal_y"], c="blue", marker="^", s=60,
               alpha=0.7, label="IFT liberal layout")

    # IFT conservative layout
    ax.scatter(ift["target_x"], ift["target_y"], c="blue", marker="s", s=60,
               alpha=0.7, label="IFT conservative layout")

    # IFT optimized neighbors
    ax.scatter(ift["neighbor_x"], ift["neighbor_y"], c="red", marker="D", s=100,
               zorder=5, label="IFT neighbors (optimized)")

    # Best blob: conservative layout
    best_blob_data = blobs[best_blob["seed"]]
    ax.scatter(best_blob_data["best_conservative_x"], best_blob_data["best_conservative_y"],
               c="orange", marker="o", s=60, alpha=0.7, label="Blob best conservative")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Layout Comparison: IFT vs Blob Discovery")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "layout_overlay.png", dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_dir / 'layout_overlay.png'}")
    plt.close()

    # Save comparison summary
    summary = {
        "ift": {
            "regret": ift_regret,
            "liberal_aep": ift_liberal,
            "conservative_aep": ift_conservative,
            "regret_pct": ift_pct,
        },
        "blob_discovery": {
            "best_seed": best_blob["seed"],
            "regret": best_blob["regret"],
            "aep_absent": best_blob["aep_absent"],
            "aep_present": best_blob["aep_present"],
            "n_blobs_tested": len(blobs),
            "all_regrets": all_regrets,
        },
    }
    with open(output_dir / "comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {output_dir / 'comparison_summary.json'}")


if __name__ == "__main__":
    main()

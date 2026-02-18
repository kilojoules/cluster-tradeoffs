"""Plot 3x5 grid of combined-case Pareto plots across the A-value sweep."""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_layouts_from_h5(h5_path):
    """Load all layouts from a single HDF5 file."""
    layouts = []
    with h5py.File(h5_path, "r") as f:
        for key in f.keys():
            if not key.startswith("layout_"):
                continue
            grp = f[key]
            layouts.append(
                {
                    "aep_absent": grp.attrs["aep_absent"],
                    "aep_present": grp.attrs["aep_present"],
                    "strategy": grp.attrs["strategy"],
                }
            )
    return layouts


def load_combined_layouts(analysis_dir):
    """Load layouts from main file + any shards."""
    layouts = []
    main = analysis_dir / "layouts_combined.h5"
    if main.exists():
        layouts.extend(load_layouts_from_h5(main))
    for shard in sorted(analysis_dir.glob("layouts_combined_s*.h5")):
        layouts.extend(load_layouts_from_h5(shard))
    return layouts


def load_combined_with_standard_liberal(analysis_dir):
    """Load conservative from combined file, liberal from re-evaluated standard set.

    Uses liberal_standard_combined.h5 (liberal layouts re-evaluated against all
    9 neighbors) when available. Falls back to combined file's own liberal solutions.
    """
    all_layouts = load_combined_layouts(analysis_dir)
    conservative = [l for l in all_layouts if l["strategy"] == "conservative"]

    liberal_h5 = analysis_dir / "liberal_standard_combined.h5"
    if liberal_h5.exists():
        liberal = load_layouts_from_h5(liberal_h5)
    else:
        liberal = [l for l in all_layouts if l["strategy"] == "liberal"]

    return liberal + conservative


def find_pareto(aep_absent, aep_present):
    """Find Pareto-optimal indices (maximising both objectives)."""
    n = len(aep_absent)
    pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not pareto[i]:
            continue
        for j in range(n):
            if i == j or not pareto[j]:
                continue
            if (aep_absent[j] >= aep_absent[i] and aep_present[j] > aep_present[i]) or (
                aep_absent[j] > aep_absent[i] and aep_present[j] >= aep_present[i]
            ):
                pareto[i] = False
                break
    return pareto


def main():
    A_values = [f"{0.02 * i:.2f}" for i in range(1, 16)]
    fig, axes = plt.subplots(3, 5, figsize=(22, 13))

    for idx, A in enumerate(A_values):
        ax = axes[idx // 5, idx % 5]
        analysis_dir = Path(f"analysis/dei_A{A}")

        layouts = load_combined_with_standard_liberal(analysis_dir)
        if not layouts:
            ax.text(0.5, 0.5, f"A={A}\nNo data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"A = {A}", fontsize=10)
            continue

        aep_absent = np.array([l["aep_absent"] for l in layouts])
        aep_present = np.array([l["aep_present"] for l in layouts])
        strategies = np.array([l["strategy"] for l in layouts])
        n_seeds = len(layouts) // 2

        pareto_mask = find_pareto(aep_absent, aep_present)
        n_pareto = pareto_mask.sum()

        # Liberal (blue circles)
        lib = strategies == "liberal"
        ax.scatter(
            aep_absent[lib & ~pareto_mask],
            aep_present[lib & ~pareto_mask],
            c="steelblue", alpha=0.5, s=20, label="Liberal",
        )
        ax.scatter(
            aep_absent[lib & pareto_mask],
            aep_present[lib & pareto_mask],
            c="steelblue", edgecolors="black", linewidths=1.5, s=50,
        )

        # Conservative (red squares)
        con = strategies == "conservative"
        ax.scatter(
            aep_absent[con & ~pareto_mask],
            aep_present[con & ~pareto_mask],
            c="indianred", alpha=0.5, s=20, marker="s", label="Conservative",
        )
        ax.scatter(
            aep_absent[con & pareto_mask],
            aep_present[con & pareto_mask],
            c="indianred", edgecolors="black", linewidths=1.5, s=50, marker="s",
        )

        # Compute regret from Pareto front
        pareto_absent = aep_absent[pareto_mask]
        pareto_present = aep_present[pareto_mask]
        if len(pareto_absent) > 0:
            best_absent = pareto_absent.max()
            best_present = pareto_present.max()
            # Regret = gap between the utopia point and the Pareto front
            regret = best_absent + best_present - (pareto_absent + pareto_present).max()
        else:
            regret = 0

        ax.set_title(f"A = {A}  ({n_seeds} seeds, {n_pareto} Pareto)\nRegret: {regret:.1f} GWh", fontsize=9)
        ax.set_xlabel("AEP w/o neighbors (GWh)", fontsize=8)
        ax.set_ylabel("AEP w/ neighbors (GWh)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

        if idx == 0:
            ax.legend(loc="lower right", fontsize=7)

    fig.suptitle(
        "DEI Combined Case — A-Value Sweep\nAll 594 neighbor turbines, TurboGaussian wake model",
        fontsize=14, y=1.01,
    )
    plt.tight_layout()

    out = Path("docs/figures/dei_combined_A_sweep.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

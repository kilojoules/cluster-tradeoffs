"""Plot regret vs A for the combined-case A-value sweep."""

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
    A_values = [0.02 * i for i in range(1, 16)]
    A_labels = [f"{a:.2f}" for a in A_values]

    regrets = []
    regrets_pct = []
    valid_A = []

    for A, label in zip(A_values, A_labels):
        analysis_dir = Path(f"analysis/dei_A{label}")
        layouts = load_combined_with_standard_liberal(analysis_dir)

        if len(layouts) < 100:  # skip A-values with too few seeds
            continue

        aep_absent = np.array([l["aep_absent"] for l in layouts])
        aep_present = np.array([l["aep_present"] for l in layouts])

        pareto_mask = find_pareto(aep_absent, aep_present)
        pareto_indices = np.where(pareto_mask)[0]

        if len(pareto_indices) == 0:
            continue

        # Regret: best conservative AEP_present minus best liberal AEP_present
        # on the Pareto front
        lib_opt_idx = pareto_indices[np.argmax(aep_absent[pareto_mask])]
        con_opt_idx = pareto_indices[np.argmax(aep_present[pareto_mask])]
        regret_gwh = aep_present[con_opt_idx] - aep_present[lib_opt_idx]
        regret_pct = regret_gwh / aep_present[con_opt_idx] * 100

        n_seeds = len(layouts) // 2
        print(f"A={label}: {n_seeds:>4d} seeds, "
              f"regret = {regret_gwh:6.1f} GWh ({regret_pct:.2f}%), "
              f"Pareto pts = {pareto_mask.sum()}")

        valid_A.append(A)
        regrets.append(regret_gwh)
        regrets_pct.append(regret_pct)

    valid_A = np.array(valid_A)
    regrets = np.array(regrets)
    regrets_pct = np.array(regrets_pct)

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: absolute regret
    ax1.plot(valid_A, regrets, "o-", color="steelblue", markersize=8, linewidth=2)
    ax1.set_xlabel("A (wake expansion coefficient)", fontsize=12)
    ax1.set_ylabel("Regret (GWh/yr)", fontsize=12)
    ax1.set_title("Absolute regret vs A", fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, valid_A.max() + 0.02)

    # Right: percentage regret
    ax2.plot(valid_A, regrets_pct, "s-", color="indianred", markersize=8, linewidth=2)
    ax2.set_xlabel("A (wake expansion coefficient)", fontsize=12)
    ax2.set_ylabel("Regret (% of conservative-optimal AEP)", fontsize=12)
    ax2.set_title("Relative regret vs A", fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, valid_A.max() + 0.02)

    fig.suptitle(
        "DEI Combined Case — Regret vs Wake Expansion (A)\n"
        "All 594 neighbor turbines, TurboGaussian wake model",
        fontsize=14,
    )
    plt.tight_layout()

    out = Path("docs/figures/regret_vs_A.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()

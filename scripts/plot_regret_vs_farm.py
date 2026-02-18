"""Bar plot: regret vs farm number for A=0.04, including the combined (all farms) case."""

import time
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_layouts_from_h5(h5_path, retries=3):
    for attempt in range(retries):
        try:
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
        except (RuntimeError, OSError, KeyError) as e:
            if attempt < retries - 1:
                print(f"  I/O error reading {h5_path.name}, retrying ({attempt+1}/{retries})...")
                time.sleep(2)
            else:
                print(f"  WARNING: skipping {h5_path.name} after {retries} failures: {e}")
                return []


def load_farm_layouts(analysis_dir, farm_idx):
    """Load layouts for a single farm.

    Uses liberal_standard_farm{N}.h5 when available (consistent liberal
    layouts evaluated against this farm's neighbor) instead of the per-farm
    liberal layouts from the optimization run.
    """
    all_layouts = []
    main = analysis_dir / f"layouts_farm{farm_idx}.h5"
    if main.exists():
        all_layouts.extend(load_layouts_from_h5(main))
    return all_layouts


def load_combined_layouts(analysis_dir):
    """Load layouts from combined case (all 9 farms)."""
    layouts = []
    main = analysis_dir / "layouts_combined.h5"
    if main.exists():
        layouts.extend(load_layouts_from_h5(main))
    return layouts


def load_combined_with_standard_liberal(analysis_dir):
    return load_combined_layouts(analysis_dir)


def find_pareto(aep_absent, aep_present):
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


def compute_regret(layouts):
    """Compute regret from a list of layout dicts. Returns (regret_gwh, n_seeds) or None."""
    if len(layouts) < 10:
        return None
    aep_absent = np.array([l["aep_absent"] for l in layouts])
    aep_present = np.array([l["aep_present"] for l in layouts])
    pareto_mask = find_pareto(aep_absent, aep_present)
    pareto_indices = np.where(pareto_mask)[0]
    if len(pareto_indices) == 0:
        return None
    lib_opt_idx = pareto_indices[np.argmax(aep_absent[pareto_mask])]
    con_opt_idx = pareto_indices[np.argmax(aep_present[pareto_mask])]
    regret_gwh = aep_present[con_opt_idx] - aep_present[lib_opt_idx]
    n_seeds = len(layouts) // 2
    return regret_gwh, n_seeds


def main():
    analysis_dir = Path("analysis/dei_A0.04")

    labels = []
    regrets = []

    # Individual farms 1-9
    for farm_idx in range(1, 10):
        layouts = load_farm_layouts(analysis_dir, farm_idx)
        result = compute_regret(layouts)
        if result is None:
            print(f"Farm {farm_idx}: insufficient data, skipping")
            continue
        regret_gwh, n_seeds = result
        print(f"Farm {farm_idx}: {n_seeds:>4d} seeds, regret = {regret_gwh:.1f} GWh")
        labels.append(f"Farm {farm_idx}")
        regrets.append(regret_gwh)

    # Combined case (all 9 farms)
    layouts = load_combined_with_standard_liberal(analysis_dir)
    result = compute_regret(layouts)
    if result is not None:
        regret_gwh, n_seeds = result
        print(f"Combined: {n_seeds:>4d} seeds, regret = {regret_gwh:.1f} GWh")
        labels.append("All farms")
        regrets.append(regret_gwh)

    regrets = np.array(regrets)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(labels))
    colors = ["#3498db"] * 9 + ["#e74c3c"]  # blue for individual, red for combined
    colors = colors[: len(labels)]

    ax.bar(x, regrets, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Neighboring farm", fontsize=12)
    ax.set_ylabel("Regret (GWh/yr)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title(
        "DEI Regret by Neighbor Farm (A = 0.04)",
        fontsize=13, fontweight="bold",
    )
    ax.grid(True, alpha=0.2, axis="y")
    ax.set_ylim(0, regrets.max() * 1.2)

    # Add value labels on top of bars
    for i, v in enumerate(regrets):
        ax.text(i, v + regrets.max() * 0.02, f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    import shutil
    out = Path("docs/figures/regret_vs_farm_A0.04.png")
    out.parent.mkdir(exist_ok=True)
    tmp_path = Path("/tmp/regret_vs_farm_A0.04.png")
    fig.savefig(tmp_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    shutil.copy2(tmp_path, out)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()

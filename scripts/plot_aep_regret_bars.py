"""Bar plot: best conservative AEP, best liberal AEP, and regret vs A.

AEPs share the left y-axis; regret uses the right y-axis.
"""

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
        except (RuntimeError, OSError) as e:
            if attempt < retries - 1:
                print(f"  I/O error reading {h5_path.name}, retrying ({attempt+1}/{retries})...")
                time.sleep(2)
            else:
                print(f"  WARNING: skipping {h5_path.name} after {retries} failures: {e}")
                return []


def load_combined_layouts(analysis_dir):
    layouts = []
    main = analysis_dir / "layouts_combined.h5"
    if main.exists():
        layouts.extend(load_layouts_from_h5(main))
    for shard in sorted(analysis_dir.glob("layouts_combined_s*.h5")):
        layouts.extend(load_layouts_from_h5(shard))
    return layouts


def load_combined_with_standard_liberal(analysis_dir):
    all_layouts = load_combined_layouts(analysis_dir)
    conservative = [l for l in all_layouts if l["strategy"] == "conservative"]
    liberal_h5 = analysis_dir / "liberal_standard_combined.h5"
    if liberal_h5.exists():
        liberal = load_layouts_from_h5(liberal_h5)
    else:
        liberal = [l for l in all_layouts if l["strategy"] == "liberal"]
    return liberal + conservative


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


def main():
    A_values = [0.02 * i for i in range(1, 16)]
    A_labels = [f"{a:.2f}" for a in A_values]

    valid_A = []
    valid_labels = []
    best_con_aep = []
    best_lib_aep = []
    regrets = []

    for A, label in zip(A_values, A_labels):
        analysis_dir = Path(f"analysis/dei_A{label}")
        layouts = load_combined_with_standard_liberal(analysis_dir)

        if len(layouts) < 100:
            continue

        aep_absent = np.array([l["aep_absent"] for l in layouts])
        aep_present = np.array([l["aep_present"] for l in layouts])

        pareto_mask = find_pareto(aep_absent, aep_present)
        pareto_indices = np.where(pareto_mask)[0]

        if len(pareto_indices) == 0:
            continue

        lib_opt_idx = pareto_indices[np.argmax(aep_absent[pareto_mask])]
        con_opt_idx = pareto_indices[np.argmax(aep_present[pareto_mask])]
        regret_gwh = aep_present[con_opt_idx] - aep_present[lib_opt_idx]

        n_seeds = len(layouts) // 2
        print(
            f"A={label}: {n_seeds:>4d} seeds, "
            f"con AEP={aep_present[con_opt_idx]:.1f}, "
            f"lib AEP={aep_present[lib_opt_idx]:.1f}, "
            f"regret={regret_gwh:.1f} GWh"
        )

        valid_A.append(A)
        valid_labels.append(label)
        best_con_aep.append(aep_present[con_opt_idx])
        best_lib_aep.append(aep_present[lib_opt_idx])
        regrets.append(regret_gwh)

    valid_A = np.array(valid_A)
    best_con_aep = np.array(best_con_aep)
    best_lib_aep = np.array(best_lib_aep)
    regrets = np.array(regrets)

    # --- Plot ---
    fig, ax1 = plt.subplots(figsize=(14, 6))

    x = np.arange(len(valid_A))
    bar_width = 0.25

    # AEP bars on left y-axis
    ax1.bar(
        x - bar_width, best_con_aep, bar_width,
        label="Best conservative AEP", color="#e74c3c", alpha=0.85,
    )
    ax1.bar(
        x, best_lib_aep, bar_width,
        label="Best liberal AEP", color="#3498db", alpha=0.85,
    )

    ax1.set_xlabel("A (wake expansion coefficient)", fontsize=12)
    ax1.set_ylabel("AEP with neighbors present (GWh/yr)", fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(valid_labels, rotation=45, ha="right")

    # Break y-axis to zoom into relevant range
    aep_min = min(best_con_aep.min(), best_lib_aep.min())
    aep_max = max(best_con_aep.max(), best_lib_aep.max())
    aep_range = aep_max - aep_min
    ax1.set_ylim(aep_min - aep_range * 0.3, aep_max + aep_range * 0.15)

    # Regret bars on right y-axis
    ax2 = ax1.twinx()
    ax2.bar(
        x + bar_width, regrets, bar_width,
        label="Regret", color="#2ecc71", alpha=0.85,
    )
    ax2.set_ylabel("Regret (GWh/yr)", fontsize=12, color="#2ecc71")
    ax2.tick_params(axis="y", labelcolor="#2ecc71")
    ax2.set_ylim(0, regrets.max() * 1.5)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)

    ax1.set_title(
        "DEI Combined Case — Best AEP and Regret vs Wake Expansion (A)",
        fontsize=13, fontweight="bold",
    )
    ax1.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    out = Path("docs/figures/aep_regret_bars.png")
    out.parent.mkdir(exist_ok=True)
    # Save to /tmp first to avoid filesystem I/O issues, then copy
    import shutil
    tmp_path = Path("/tmp/aep_regret_bars.png")
    fig.savefig(tmp_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    shutil.copy2(tmp_path, out)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()

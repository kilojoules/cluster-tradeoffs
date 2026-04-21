"""Regret vs bearing at fixed buffer distance for all wind roses.
Shows how the directional pattern changes with wind rose shape."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

D = 240.0

cases = [
    ("Single dir (E)", "unidir90", "black", "-", 3),
    ("$a$=0.9, $f$=1.0", "a0.9_f1.0", "tab:red", "-", 2),
    ("$a$=0.5, $f$=1.0", "a0.5_f1.0", "tab:orange", "-", 2),
    ("$a$=0.7, $f$=0.5", "a0.7_f0.5", "tab:green", "--", 2),
    ("$a$=0.9, $f$=0.0", "a0.9_f0.0", "tab:blue", "--", 2),
    ("DEI", "dei", "tab:purple", "-.", 2),
    ("$a$=0.5, $f$=0.0", "a0.5_f0.0", "tab:cyan", "--", 2),
]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, dist, title in [(axes[0], 2, "Buffer = 2$D$"),
                         (axes[1], 10, "Buffer = 10$D$")]:
    for label, case_dir, color, ls, lw in cases:
        p = Path(f"analysis/cross_section_fixed/{case_dir}_d{dist}/Nt50/results.json")
        if not p.exists():
            continue
        data = json.load(open(p))
        g = np.array(data["regret_grid_gwh"])
        lib_aep = data["liberal_aep_gwh"]
        bearings = np.array(data["bearings_deg"])
        regret_pct = 100 * g[0, :] / lib_aep

        ax.plot(bearings, regret_pct, ls, color=color, linewidth=lw,
                label=label, markersize=4)

    ax.set_xlabel("Neighbor bearing ($^\\circ$)", fontsize=12)
    ax.set_ylabel("Design regret (% of AEP)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 360)
    ax.set_xticks(np.arange(0, 361, 45))
    ax.set_ylim(bottom=0)

fig.suptitle("Regret vs. Neighbor Bearing at Fixed Buffer Distances\n"
             "Identical farm copy, $K_{lib}=K_{cons}=300$. "
             "Wind prevails from 270$^\\circ$ (elliptical) or 90$^\\circ$ (single dir).",
             fontsize=12)
plt.tight_layout()

out = Path("paper_v3/figures/bearing_sweep.png")
fig.savefig(str(out), dpi=200, bbox_inches="tight")
print(f"Saved: {out}")

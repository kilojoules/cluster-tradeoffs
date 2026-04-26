"""Regret vs bearing at fixed buffer distance.
a-sweep ablation (fixed f=1.0, a from 0.3 to 0.9) plus single-dir + DEI overlays.
Traces transition from diffuse to concentrated unidirectional, ending at degenerate single-sector."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

D = 240.0

a_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
            0.95, 0.98, 0.99, 1.0, 1.5, 2.0, 3.0, 5.0]


def load(path):
    if not path.exists():
        return None
    data = json.load(open(path))
    g = np.array(data["regret_grid_gwh"])
    lib = data["liberal_aep_gwh"]
    b = np.array(data["bearings_deg"])
    return b, 100 * g[0, :] / lib


def a_sweep_path(a, dist):
    # a=0.9,f=1.0 already exists in f-sweep; others in a_sweep
    if abs(a - 0.9) < 1e-9:
        return Path(f"analysis/ablation_f_sweep/a0.9_f1.0_d{dist}/Nt50/results.json")
    return Path(f"analysis/ablation_a_sweep/a{a}_f1.0_d{dist}/Nt50/results.json")


fig, axes = plt.subplots(1, 2, figsize=(16, 6))
cmap = plt.cm.viridis
a_colors = cmap(np.linspace(0.05, 0.95, len(a_values)))

for ax, dist, title in [(axes[0], 2, "Buffer = 2$D$"),
                         (axes[1], 10, "Buffer = 10$D$")]:
    # a-sweep curves (fixed f=1.0)
    for a, color in zip(a_values, a_colors):
        r = load(a_sweep_path(a, dist))
        if r is None:
            continue
        b, pct = r
        ax.plot(b, pct, "-", color=color, linewidth=1.8, alpha=0.9,
                label=f"$a$={a:.1f}, $f$=1.0")

    # DEI overlay (real-rose reference point)
    r = load(Path(f"analysis/cross_section_fixed/dei_d{dist}/Nt50/results.json"))
    if r is not None:
        b, pct = r
        ax.plot(b, pct, "-.", color="tab:purple", linewidth=2.2, zorder=10,
                label="DEI real rose")

    ax.set_xlabel("Neighbor bearing ($^\\circ$)", fontsize=12)
    ax.set_ylabel("Design regret (% of AEP)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 360)
    ax.set_xticks(np.arange(0, 361, 45))
    ax.set_ylim(bottom=0)

fig.suptitle("Regret vs. Neighbor Bearing — $a$-ablation at $f=1.0$\n"
             "$a<1/\\sqrt{\\pi}\\approx0.564$: ellipse perpendicular to prevailing (bimodal at $\\theta_\\text{prev}\\pm90°$). "
             "$a>0.564$: concentrated at prevailing. $K_{lib}=K_{cons}=300$.",
             fontsize=12)
plt.tight_layout()

out = Path("paper_v3/figures/bearing_sweep.png")
fig.savefig(str(out), dpi=200, bbox_inches="tight")
print(f"Saved: {out}")

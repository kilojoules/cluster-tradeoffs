"""Ablation line plot: regret vs bearing as f sweeps from bidir to unidir.
Fixed a=0.9, varying f. Two panels for buffer 2D and 10D.
Pure single-direction case (wind from 270) overlaid for reference."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

D = 240.0
f_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
cmap = plt.cm.viridis
colors = cmap(np.linspace(0.05, 0.95, len(f_values)))

for ax, dist, title in [(axes[0], 2, "Buffer = 2$D$"),
                         (axes[1], 10, "Buffer = 10$D$")]:
    for f, color in zip(f_values, colors):
        p = Path(f"analysis/ablation_f_sweep/a0.9_f{f}_d{dist}/Nt50/results.json")
        if not p.exists():
            continue
        data = json.load(open(p))
        g = np.array(data["regret_grid_gwh"])
        lib_aep = data["liberal_aep_gwh"]
        bearings = np.array(data["bearings_deg"])
        regret_pct = 100 * g[0, :] / lib_aep
        ax.plot(bearings, regret_pct, "-", color=color, linewidth=2,
                label=f"$f$={f:.1f}")

    # Pure unidir overlay (wind from 270)
    p_uni = Path(f"analysis/ablation_f_sweep/unidir270_d{dist}/Nt50/results.json")
    if p_uni.exists():
        data = json.load(open(p_uni))
        g = np.array(data["regret_grid_gwh"])
        lib_aep = data["liberal_aep_gwh"]
        bearings = np.array(data["bearings_deg"])
        regret_pct = 100 * g[0, :] / lib_aep
        ax.plot(bearings, regret_pct, "--", color="red", linewidth=2.5,
                label="Single dir (270°)", zorder=10)

    ax.axvline(270, color="steelblue", ls=":", lw=1.5, alpha=0.6)
    ax.text(270, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1,
            " wind from 270°", fontsize=9, color="steelblue",
            ha="left", va="top")

    ax.set_xlabel("Neighbor bearing ($^\\circ$)", fontsize=12)
    ax.set_ylabel("Design regret (% of AEP)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 360)
    ax.set_xticks(np.arange(0, 361, 45))
    ax.set_ylim(bottom=0)

fig.suptitle("Ablation: Regret vs. Bearing as Wind Rose Transitions from Bidirectional to Unidirectional\n"
             "Fixed $a$=0.9, $f$ sweeps 0.0 (bidir) → 1.0 (unidir). $K_{lib}=K_{cons}=300$.",
             fontsize=12)
plt.tight_layout()

out = Path("paper_v3/figures/ablation_f_sweep.png")
fig.savefig(str(out), dpi=200, bbox_inches="tight")
print(f"Saved: {out}")

# Summary
print("\n" + "="*70)
print(f"{'f':>6} {'d':>4} {'max_pct':>10} {'bearing_max':>12}")
print("-"*70)
for f in f_values:
    for dist in [2, 10]:
        p = Path(f"analysis/ablation_f_sweep/a0.9_f{f}_d{dist}/Nt50/results.json")
        if not p.exists():
            continue
        data = json.load(open(p))
        g = np.array(data["regret_grid_gwh"])
        lib_aep = data["liberal_aep_gwh"]
        bearings = np.array(data["bearings_deg"])
        regret_pct = 100 * g[0, :] / lib_aep
        imax = int(np.argmax(regret_pct))
        print(f"{f:>6.1f} {dist:>4d} {regret_pct[imax]:>9.3f}% {bearings[imax]:>11.0f}°")
for dist in [2, 10]:
    p = Path(f"analysis/ablation_f_sweep/unidir270_d{dist}/Nt50/results.json")
    if not p.exists():
        continue
    data = json.load(open(p))
    g = np.array(data["regret_grid_gwh"])
    lib_aep = data["liberal_aep_gwh"]
    bearings = np.array(data["bearings_deg"])
    regret_pct = 100 * g[0, :] / lib_aep
    imax = int(np.argmax(regret_pct))
    print(f"{'uni':>6} {dist:>4d} {regret_pct[imax]:>9.3f}% {bearings[imax]:>11.0f}°")

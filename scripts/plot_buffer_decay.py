"""Plot buffer distance decay curves from pilot results."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

configs = [
    ("$a$=0.9, $f$=1.0\n(concentrated, unidirectional)", "0.9", "1.0", "tab:red", "o"),
    ("$a$=0.7, $f$=0.5\n(mid-range)", "0.7", "0.5", "tab:orange", "s"),
    ("$a$=0.5, $f$=0.0\n(moderate, bidirectional)", "0.5", "0.0", "tab:blue", "D"),
]
buffers_D = [2, 10, 20, 40]
D = 240  # m

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

for label, a, f, color, marker in configs:
    regrets = []
    liberal_aeps = []
    for b in buffers_D:
        p = Path(f"analysis/buffer_sweep/a{a}_f{f}_buf{b}D/results.json")
        with open(p) as fh:
            d = json.load(fh)
        regrets.append(d["regret_gwh"])
        liberal_aeps.append(d["liberal_aep_gwh"])
    regrets = np.array(regrets)
    liberal_aep = liberal_aeps[0]
    buffers_km = np.array(buffers_D) * D / 1000

    # Absolute regret
    ax1.plot(buffers_km, regrets, f"{marker}-", color=color, label=label,
             linewidth=2, markersize=8)

    # Relative regret (% of AEP)
    ax2.plot(buffers_km, 100 * regrets / liberal_aep, f"{marker}-", color=color,
             label=label, linewidth=2, markersize=8)

# NYSERDA 4nm reference line
nyserda_km = 4 * 1.852  # 7.4 km
for ax in (ax1, ax2):
    ax.axvline(nyserda_km, color="gray", ls="--", lw=1.5, alpha=0.7)
    ax.text(nyserda_km + 0.15, ax.get_ylim()[1] * 0.92 if ax == ax1 else 2.3,
            "NYSERDA\n4 nm", fontsize=8, color="gray", va="top")
    ax.set_xlabel("Buffer distance (km)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")

# Secondary x-axis in D
for ax in (ax1, ax2):
    ax_top = ax.secondary_xaxis("top", functions=(lambda x: x * 1000 / D, lambda x: x * D / 1000))
    ax_top.set_xlabel("Buffer distance ($D$)")

ax1.set_ylabel("Design regret (GWh/yr)")
ax1.set_title("Absolute Regret vs. Buffer Distance")

ax2.set_ylabel("Design regret (% of liberal AEP)")
ax2.set_title("Relative Regret vs. Buffer Distance")

fig.suptitle("Buffer Distance Decay: Design Regret from 30 Greedy Neighbors\n"
             "Bastankhah $k$=0.04, DEI polygon, 50 IEA-15MW turbines, "
             "$\\theta_{prev}$=270°, K=5 multistart",
             fontsize=11)
plt.tight_layout()

out = Path("analysis/buffer_sweep/buffer_decay_curves.png")
fig.savefig(str(out), dpi=200, bbox_inches="tight")
print(f"Saved: {out}")

# Print summary
print("\nKey findings:")
print(f"  Concentrated unidir (a=0.9, f=1.0): {regrets[0]:.0f} → {regrets[-1]:.0f} GWh "
      f"(2D → 40D), {regrets[-1]/regrets[0]*100:.0f}% remaining")
for label, a, f, color, marker in configs:
    regrets = []
    for b in buffers_D:
        p = Path(f"analysis/buffer_sweep/a{a}_f{f}_buf{b}D/results.json")
        with open(p) as fh:
            d = json.load(fh)
        regrets.append(d["regret_gwh"])
    regrets = np.array(regrets)
    print(f"  {label.replace(chr(10), ' ')}: half-regret at ~{buffers_D[np.searchsorted(-regrets, -regrets[0]/2)]}D "
          f"({buffers_D[np.searchsorted(-regrets, -regrets[0]/2)] * D / 1000:.1f} km)"
          if regrets[-1] < regrets[0] / 2 else
          f"  {label.replace(chr(10), ' ')}: regret never halves within 40D")

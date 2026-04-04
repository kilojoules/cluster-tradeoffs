"""Plot 2D regret heatmap from edrose (a, f) sweep results."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Load results
a_vals = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
f_vals = [0.0, 0.25, 0.5, 0.75, 1.0]

regret = np.zeros((len(a_vals), len(f_vals)))
liberal_aep = np.zeros_like(regret)

for i, a in enumerate(a_vals):
    for j, f in enumerate(f_vals):
        p = Path(f"analysis/edrose_sweep/a{a}_f{f}/results.json")
        with open(p) as fh:
            d = json.load(fh)
        regret[i, j] = d["regret_gwh"]
        liberal_aep[i, j] = d["liberal_aep_gwh"]

# Relative regret (% of liberal AEP)
regret_pct = 100 * regret / liberal_aep

# --- Figure 1: Absolute regret heatmap ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Left: absolute regret
ax = axes[0]
im = ax.imshow(regret, origin="lower", aspect="auto",
               extent=[f_vals[0] - 0.125, f_vals[-1] + 0.125,
                       a_vals[0] - 0.05, a_vals[-1] + 0.05],
               cmap="YlOrRd", interpolation="bicubic")
cb = fig.colorbar(im, ax=ax, label="Design regret (GWh/yr)")
# Add text annotations
for i, a in enumerate(a_vals):
    for j, f in enumerate(f_vals):
        color = "white" if regret[i, j] > 70 else "black"
        ax.text(f, a, f"{regret[i, j]:.0f}", ha="center", va="center",
                fontsize=9, fontweight="bold", color=color)
ax.set_xlabel("Folding parameter $f$\n(0 = bidirectional, 1 = unidirectional)")
ax.set_ylabel("Shape parameter $a$\n(low = uniform, high = concentrated)")
ax.set_title("Design Regret (GWh/yr)")
ax.set_xticks(f_vals)
ax.set_yticks(a_vals)

# Mark uniform reference
a_uniform = 1 / np.sqrt(np.pi)  # ~0.564
ax.axhline(a_uniform, color="white", ls="--", lw=1, alpha=0.7)
ax.text(0.95, a_uniform + 0.02, "$a = 1/\\sqrt{\\pi}$ (uniform)",
        ha="right", va="bottom", fontsize=8, color="white", style="italic")

# DEI overlay (fitted: a=0.6365, f=0.3840)
ax.plot(0.384, 0.637, "s", color="cyan", markersize=12, markeredgecolor="black",
        markeredgewidth=1.5, zorder=10, label="DEI")
ax.text(0.384 + 0.04, 0.637, "DEI", fontsize=9, fontweight="bold",
        color="cyan", va="center")

# Right: relative regret
ax = axes[1]
im2 = ax.imshow(regret_pct, origin="lower", aspect="auto",
                extent=[f_vals[0] - 0.125, f_vals[-1] + 0.125,
                        a_vals[0] - 0.05, a_vals[-1] + 0.05],
                cmap="YlOrRd", interpolation="bicubic")
cb2 = fig.colorbar(im2, ax=ax, label="Design regret (% of liberal AEP)")
for i, a in enumerate(a_vals):
    for j, f in enumerate(f_vals):
        color = "white" if regret_pct[i, j] > 1.8 else "black"
        ax.text(f, a, f"{regret_pct[i, j]:.1f}%", ha="center", va="center",
                fontsize=8, fontweight="bold", color=color)
ax.set_xlabel("Folding parameter $f$\n(0 = bidirectional, 1 = unidirectional)")
ax.set_ylabel("Shape parameter $a$\n(low = uniform, high = concentrated)")
ax.set_title("Design Regret (% of AEP)")
ax.set_xticks(f_vals)
ax.set_yticks(a_vals)
ax.axhline(a_uniform, color="white", ls="--", lw=1, alpha=0.7)
ax.text(0.95, a_uniform + 0.02, "$a = 1/\\sqrt{\\pi}$ (uniform)",
        ha="right", va="bottom", fontsize=8, color="white", style="italic")
ax.plot(0.384, 0.637, "s", color="cyan", markersize=12, markeredgecolor="black",
        markeredgewidth=1.5, zorder=10)
ax.text(0.384 + 0.04, 0.637, "DEI", fontsize=9, fontweight="bold",
        color="cyan", va="center")

fig.suptitle("Wind Rose Shape Space: Design Regret from 30 Greedy Neighbors\n"
             "Bastankhah $k$=0.04, DEI polygon, 50 IEA-15MW turbines, "
             "$\\theta_{prev}$=270°, K=5 multistart",
             fontsize=11)
plt.tight_layout()

out = Path("analysis/edrose_sweep/regret_heatmap.png")
fig.savefig(str(out), dpi=200, bbox_inches="tight")
print(f"Saved: {out}")

# --- Figure 2: Regret vs a, sliced by f ---
fig2, ax2 = plt.subplots(figsize=(8, 5))
for j, f in enumerate(f_vals):
    label = f"$f$ = {f}" + (" (bidirectional)" if f == 0.0 else
                             " (unidirectional)" if f == 1.0 else "")
    ax2.plot(a_vals, regret[:, j], "o-", label=label, linewidth=2, markersize=6)
ax2.axvline(a_uniform, color="gray", ls="--", lw=1, alpha=0.7)
ax2.text(a_uniform + 0.01, ax2.get_ylim()[1] * 0.95, "uniform",
         fontsize=9, color="gray", va="top")
ax2.set_xlabel("Shape parameter $a$")
ax2.set_ylabel("Design regret (GWh/yr)")
ax2.set_title("Regret vs. Concentration, Sliced by Folding")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
plt.tight_layout()

out2 = Path("analysis/edrose_sweep/regret_vs_a.png")
fig2.savefig(str(out2), dpi=200, bbox_inches="tight")
print(f"Saved: {out2}")

# --- Figure 3: Regret vs f, sliced by a ---
fig3, ax3 = plt.subplots(figsize=(8, 5))
for i, a in enumerate(a_vals):
    ax3.plot(f_vals, regret[i, :], "o-", label=f"$a$ = {a}", linewidth=2, markersize=6)
ax3.set_xlabel("Folding parameter $f$")
ax3.set_ylabel("Design regret (GWh/yr)")
ax3.set_title("Regret vs. Folding (Bidirectional → Unidirectional)")
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)
plt.tight_layout()

out3 = Path("analysis/edrose_sweep/regret_vs_f.png")
fig3.savefig(str(out3), dpi=200, bbox_inches="tight")
print(f"Saved: {out3}")

# Print summary
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"Min regret: {regret.min():.1f} GWh at a={a_vals[np.unravel_index(regret.argmin(), regret.shape)[0]]}, "
      f"f={f_vals[np.unravel_index(regret.argmin(), regret.shape)[1]]}")
print(f"Max regret: {regret.max():.1f} GWh at a={a_vals[np.unravel_index(regret.argmax(), regret.shape)[0]]}, "
      f"f={f_vals[np.unravel_index(regret.argmax(), regret.shape)[1]]}")
print(f"Max/min ratio: {regret.max()/regret.min():.1f}x")
print(f"\nRegret range by f (bidirectional → unidirectional):")
for j, f in enumerate(f_vals):
    print(f"  f={f:4.2f}: {regret[:, j].min():.1f} - {regret[:, j].max():.1f} GWh "
          f"(mean {regret[:, j].mean():.1f})")
print(f"\nRegret range by a (uniform → concentrated):")
for i, a in enumerate(a_vals):
    print(f"  a={a:3.1f}: {regret[i, :].min():.1f} - {regret[i, :].max():.1f} GWh "
          f"(mean {regret[i, :].mean():.1f})")

"""Generate all figures for paper v2 (K=500 converged results)."""

import jax
jax.config.update("jax_enable_x64", True)

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

D = 240.0
OUT = Path("paper_v2/figures")
OUT.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Figure 1: Convergence study
# =============================================================================
print("Figure 1: Convergence study...")
with open("analysis/convergence_study/results.json") as f:
    conv = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

configs = [
    ("concentrated_unidir_close", "Conc. unidir, 5D"),
    ("concentrated_unidir_far", "Conc. unidir, 20D"),
    ("moderate_bidir", "Moderate bidir, 5D"),
]

for ax, (name, label) in zip(axes, configs):
    cfg = conv[name]
    cons = cfg["conservative_convergence"]
    K_vals = [c["K"] for c in cons]
    regrets = [c["regret_gwh"] for c in cons]
    ax.semilogx(K_vals, regrets, "o-", color="firebrick", linewidth=2, markersize=6)
    ax.axhline(regrets[-1], color="gray", ls="--", alpha=0.5, label=f"K=2000: {regrets[-1]:.1f}")
    # Mark K=5
    k5_r = [c["regret_gwh"] for c in cons if c["K"] == 5][0]
    ax.axhline(k5_r, color="steelblue", ls=":", alpha=0.5, label=f"K=5: {k5_r:.1f}")
    ax.set_xlabel("Number of starts $K$")
    ax.set_ylabel("Design regret (GWh/yr)")
    ax.set_title(label)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.8, 2500)

fig.suptitle("Convergence of Design Regret with Multistart Optimization Depth",
             fontsize=13)
plt.tight_layout()
fig.savefig(OUT / "convergence.png", dpi=200, bbox_inches="tight")
print(f"  Saved: {OUT / 'convergence.png'}")
plt.close()

# =============================================================================
# Figure 2: K=500 edrose heatmap
# =============================================================================
print("Figure 2: edrose heatmap (K=500)...")
a_vals = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
f_vals = [0.0, 0.25, 0.5, 0.75, 1.0]
regret = np.zeros((len(a_vals), len(f_vals)))
liberal_aep = np.zeros_like(regret)

for i, a in enumerate(a_vals):
    for j, f in enumerate(f_vals):
        d = json.load(open(f"analysis/edrose_sweep_k500/a{a}_f{f}/results.json"))
        regret[i, j] = d["regret_gwh"]
        liberal_aep[i, j] = d["liberal_aep_gwh"]

regret_pct = 100 * regret / liberal_aep

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

for ax, data, cblabel, title, fmt in [
    (axes[0], regret, "Design regret (GWh/yr)", "Absolute Regret (K=500)", "{:.0f}"),
    (axes[1], regret_pct, "Design regret (% of AEP)", "Relative Regret (K=500)", "{:.1f}%"),
]:
    im = ax.imshow(data, origin="lower", aspect="auto",
                   extent=[f_vals[0]-0.125, f_vals[-1]+0.125,
                           a_vals[0]-0.05, a_vals[-1]+0.05],
                   cmap="YlOrRd", interpolation="bicubic")
    fig.colorbar(im, ax=ax, label=cblabel)
    for i, a in enumerate(a_vals):
        for j, f in enumerate(f_vals):
            thresh = 70 if ax == axes[0] else 1.8
            color = "white" if data[i, j] > thresh else "black"
            txt = fmt.format(data[i, j])
            ax.text(f, a, txt, ha="center", va="center", fontsize=8,
                    fontweight="bold", color=color)
    ax.set_xlabel("Folding $f$ (0=bidirectional, 1=unidirectional)")
    ax.set_ylabel("Shape $a$ (low=uniform, high=concentrated)")
    ax.set_title(title)
    ax.set_xticks(f_vals)
    ax.set_yticks(a_vals)
    # Uniform reference
    a_unif = 1 / np.sqrt(np.pi)
    ax.axhline(a_unif, color="white", ls="--", lw=1, alpha=0.7)
    # DEI marker (fitted: a=0.64, f=0.38)
    ax.plot(0.384, 0.637, "s", color="cyan", markersize=12,
            markeredgecolor="black", markeredgewidth=1.5, zorder=10)
    ax.text(0.384 + 0.04, 0.637, "DEI", fontsize=9, fontweight="bold",
            color="cyan", va="center")

fig.suptitle("Design Regret Across Wind Rose Shape Space\n"
             "30 greedy neighbors, Bastankhah $k$=0.04, 50 IEA-15MW turbines, K=500 multistart",
             fontsize=11)
plt.tight_layout()
fig.savefig(OUT / "edrose_heatmap_k500.png", dpi=200, bbox_inches="tight")
print(f"  Saved: {OUT / 'edrose_heatmap_k500.png'}")
plt.close()

# =============================================================================
# Figure 3: Regret vs folding (line plot)
# =============================================================================
print("Figure 3: Regret vs folding...")
fig, ax = plt.subplots(figsize=(8, 5))
for i, a in enumerate(a_vals):
    ax.plot(f_vals, regret_pct[i, :], "o-", label=f"$a$ = {a}", linewidth=2, markersize=6)
ax.set_xlabel("Folding parameter $f$")
ax.set_ylabel("Design regret (% of AEP)")
ax.set_title("Regret vs. Folding (K=500)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(OUT / "regret_vs_f.png", dpi=200, bbox_inches="tight")
print(f"  Saved: {OUT / 'regret_vs_f.png'}")
plt.close()

# =============================================================================
# Figure 4: Buffer distance decay (K=500)
# =============================================================================
print("Figure 4: Buffer decay (K=500)...")
buffers_D = [2, 10, 20, 40]
buffers_km = np.array(buffers_D) * D / 1000

buf_configs = [
    ("$a$=0.9, $f$=1.0 (conc. unidir)", "0.9", "1.0", "tab:red", "o"),
    ("$a$=0.7, $f$=0.5 (mid-range)", "0.7", "0.5", "tab:orange", "s"),
    ("$a$=0.5, $f$=0.0 (mod. bidir)", "0.5", "0.0", "tab:blue", "D"),
]

fig, ax = plt.subplots(figsize=(8, 5.5))
for label, a, f, color, marker in buf_configs:
    regrets_buf = []
    lib_aep_buf = None
    for b in buffers_D:
        d = json.load(open(f"analysis/buffer_sweep_k500/a{a}_f{f}_buf{b}D/results.json"))
        regrets_buf.append(d["regret_gwh"])
        if lib_aep_buf is None:
            lib_aep_buf = d["liberal_aep_gwh"]
    regrets_buf = np.array(regrets_buf)
    ax.plot(buffers_km, 100 * regrets_buf / lib_aep_buf, f"{marker}-", color=color,
            label=label, linewidth=2, markersize=8)

nyserda_km = 4 * 1.852
ax.axvline(nyserda_km, color="gray", ls="--", lw=1.5, alpha=0.7)
ax.text(nyserda_km + 0.15, ax.get_ylim()[1] * 0.9,
        "NYSERDA\n4 nm", fontsize=8, color="gray", va="top")
ax.set_xlabel("Buffer distance (km)")
ax.set_ylabel("Design regret (% of liberal AEP)")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)
ax_top = ax.secondary_xaxis("top", functions=(lambda x: x*1000/D, lambda x: x*D/1000))
ax_top.set_xlabel("Buffer distance ($D$)")
fig.suptitle("Buffer Distance Decay of Design Regret (K=500 Multistart)", fontsize=12)
plt.tight_layout()
fig.savefig(OUT / "buffer_decay_k500.png", dpi=200, bbox_inches="tight")
print(f"  Saved: {OUT / 'buffer_decay_k500.png'}")
plt.close()

# =============================================================================
# Figure 5: K=5 vs K=500 comparison
# =============================================================================
print("Figure 5: K=5 vs K=500 comparison...")
regret_k5 = np.zeros_like(regret)
for i, a in enumerate(a_vals):
    for j, f in enumerate(f_vals):
        d = json.load(open(f"analysis/edrose_sweep/a{a}_f{f}/results.json"))
        regret_k5[i, j] = d["regret_gwh"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Scatter: K=5 vs K=500
ax1.scatter(regret.ravel(), regret_k5.ravel(), c="steelblue", s=60,
            edgecolors="navy", linewidths=0.5)
lims = [0, max(regret.max(), regret_k5.max()) * 1.1]
ax1.plot(lims, lims, "k--", alpha=0.5)
ax1.set_xlabel("K=500 regret (GWh/yr)")
ax1.set_ylabel("K=5 regret (GWh/yr)")
ax1.set_title("K=5 vs K=500: Individual Cases")
ax1.set_aspect("equal")
ax1.grid(True, alpha=0.3)

# Histogram of ratios
ratios = regret_k5.ravel() / np.maximum(regret.ravel(), 1)
ax2.hist(ratios * 100, bins=15, color="steelblue", edgecolor="navy", alpha=0.7)
ax2.axvline(100, color="k", ls="--", alpha=0.5, label="K=5 = K=500")
ax2.axvline(ratios.mean() * 100, color="red", ls="-", lw=2,
            label=f"Mean: {ratios.mean()*100:.0f}%")
ax2.set_xlabel("K=5 regret / K=500 regret (%)")
ax2.set_ylabel("Count")
ax2.set_title("Distribution of K=5/K=500 Ratios")
ax2.legend()
ax2.grid(True, alpha=0.3)

fig.suptitle("Effect of Multistart Depth on Greedy Grid Search Results", fontsize=12)
plt.tight_layout()
fig.savefig(OUT / "k5_vs_k500.png", dpi=200, bbox_inches="tight")
print(f"  Saved: {OUT / 'k5_vs_k500.png'}")
plt.close()

print("\nAll figures generated.")

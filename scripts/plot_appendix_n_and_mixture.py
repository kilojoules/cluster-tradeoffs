"""Build appendix figures: N_t sensitivity + mixture wind roses."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


# ---------- N_t sensitivity ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
N_VALS = [25, 50, 75, 100]
D_VALS = [2, 10, 20, 40]
markers = {25: "o", 50: "s", 75: "^", 100: "D"}
colors = {25: "tab:blue", 50: "tab:orange", 75: "tab:green", 100: "tab:red"}

for ax, case in zip(axes, ["dei", "a0.9_f1.0"]):
    for N in N_VALS:
        peaks = []
        for d in D_VALS:
            if N == 50:
                p = Path(f"analysis/cross_section_fixed/{case}_d{d}/Nt50/results.json")
            else:
                p = Path(f"analysis/n_target_sweep/{case}_N{N}_d{d}/Nt{N}/results.json")
            if not p.exists():
                peaks.append(np.nan); continue
            data = json.load(open(p))
            g = np.array(data["regret_grid_gwh"]); lib = data["liberal_aep_gwh"]
            peaks.append(100 * g[0, :].max() / lib)
        ax.plot(D_VALS, peaks, "-", marker=markers[N], color=colors[N],
                label=f"$N_t = {N}$", linewidth=2, markersize=8)
    ax.set_xlabel("Buffer distance ($D$)")
    ax.set_ylabel("Peak regret (\\% of AEP)")
    ax.set_title("DEI" if case == "dei" else "$a$=0.9, $f$=1.0", fontweight="bold")
    ax.set_xscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

fig.suptitle("Target farm size sensitivity. Neighbor remains identical-copy of target. "
             "$K_\\text{lib}=K_\\text{cons}=500$.", fontsize=11)
plt.tight_layout()
out = Path("paper_v3/figures/n_target_sensitivity.png")
fig.savefig(str(out), dpi=180, bbox_inches="tight")
print(f"Saved: {out}")


# ---------- Mixture configurations ----------
mix_configs = [
    ("bidir_270_90", "Bidir (W+E)"),
    ("ortho_270_180", "Orth (W+S)"),
    ("wind_strong_weak", "Strong+weak (W+E)"),
    ("tri_270_180_90", "Tri (W+S+E)"),
    ("broad_two_modes", "Broad two modes"),
    ("narrow_two_modes", "Narrow two modes"),
    ("asym_three_quart", "Asym 3/4 (W+E)"),
    ("north_sea_like", "North-Sea-like (SW+NE)"),
]

fig, ax = plt.subplots(1, 1, figsize=(10, 5.5))
labels = []
peaks_2D = []; peaks_20D = []
for tag, label in mix_configs:
    labels.append(label)
    p2 = Path(f"analysis/mixture_explore/{tag}_d2/Nt50/results.json")
    p20 = Path(f"analysis/mixture_explore/{tag}_d20/Nt50/results.json")
    d2 = json.load(open(p2)); d20 = json.load(open(p20))
    p2_pct = 100 * np.array(d2["regret_grid_gwh"])[0, :].max() / d2["liberal_aep_gwh"]
    p20_pct = 100 * np.array(d20["regret_grid_gwh"])[0, :].max() / d20["liberal_aep_gwh"]
    peaks_2D.append(p2_pct); peaks_20D.append(p20_pct)

x = np.arange(len(labels))
w = 0.4
ax.bar(x - w/2, peaks_2D, w, color="firebrick", label="$d = 2D$")
ax.bar(x + w/2, peaks_20D, w, color="steelblue", label="$d = 20D$")
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
ax.set_ylabel("Peak regret (\\% of AEP)")
ax.set_title("Mixture-of-ellipses wind roses: peak regret at $d=2D$ and $d=20D$. "
             "$K_\\text{lib}=K_\\text{cons}=500$", fontsize=11)
ax.legend()
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
out = Path("paper_v3/figures/mixture_explore.png")
fig.savefig(str(out), dpi=180, bbox_inches="tight")
print(f"Saved: {out}")

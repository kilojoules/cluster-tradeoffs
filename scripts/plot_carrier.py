"""Carrier figure for paper_v4 ('No Regrets').

Two-panel:
  Left: r(d) for 4 real sites + 3 Hart archetypes, log-y, with tau=0.2% line.
  Right: d*(a) at f=1, real sites as labeled dots.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator

D_M = 240.0
A_VALS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
F_VALS = [0.0, 0.25, 0.5, 0.75, 1.0]
D_VALS = [2, 5, 10, 15, 20, 30, 40]
TAU = 0.2

SITES = [
    ("Horns Rev 1", 0.649, 0.384, "tab:blue", "s"),
    ("Lillgrund", 0.621, 0.544, "tab:green", "D"),
    ("Borssele", 0.728, 0.291, "tab:orange", "^"),
    ("DEI", 0.637, 0.384, "tab:red", "P"),
]
ARCHETYPES = [
    ("Bidir (a=0.5, f=0)", 0.5, 0.0, "tab:purple", "-."),
    ("Mid (a=0.7, f=0.5)", 0.7, 0.5, "tab:brown", "--"),
    ("Concentrated (a=0.9, f=1)", 0.9, 1.0, "black", "-"),
]


def peak_array():
    peak = np.full((len(A_VALS), len(F_VALS), len(D_VALS)), np.nan)
    for i, a in enumerate(A_VALS):
        for j, f in enumerate(F_VALS):
            for k, d in enumerate(D_VALS):
                p = Path(f"analysis/buffer_table/a{a}_f{f}_d{d}/Nt50/results.json")
                if not p.exists(): continue
                data = json.load(open(p))
                g = np.array(data["regret_grid_gwh"])
                lib = data["liberal_aep_gwh"]
                peak[i, j, k] = 100 * g[0, :].max() / lib
    return peak

peak = peak_array()


def r_at(a, f):
    """Bilinear interp of r(d) at given (a, f)."""
    out = np.zeros(len(D_VALS))
    for k in range(len(D_VALS)):
        plane = peak[:, :, k]
        # use available cells; bilinear via RegularGridInterpolator
        if np.isnan(plane).all():
            out[k] = np.nan; continue
        # Replace remaining nans with column means as a fallback
        fill = np.nanmean(plane)
        plane_filled = np.where(np.isnan(plane), fill, plane)
        interp = RegularGridInterpolator(
            (np.array(A_VALS), np.array(F_VALS)), plane_filled,
            method="linear", bounds_error=False, fill_value=fill)
        out[k] = float(interp([[a, f]])[0])
    return out


# ---- LEFT panel: r(d) ----
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
ax = axes[0]
for name, a, f, color, marker in SITES:
    r = r_at(a, f)
    ax.plot(D_VALS, r, "o-", color=color, label=name, linewidth=2, markersize=6)
for name, a, f, color, ls in ARCHETYPES:
    r = r_at(a, f)
    ax.plot(D_VALS, r, ls, color=color, label=name, linewidth=2.5, alpha=0.85)
ax.axhline(TAU, color="gray", lw=1, ls=":", alpha=0.7)
ax.text(40, TAU, f"  $\\tau={TAU}\\%$", color="gray", va="center", fontsize=9)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Buffer distance ($D$)")
ax.set_ylabel("Peak regret (\\% of AEP)")
ax.set_title("Regret vs.\\ buffer distance")
ax.grid(True, alpha=0.3, which="both")
ax.legend(fontsize=8, loc="lower left")


# ---- RIGHT panel: d*(a) at f=1, real sites overlaid ----
ax = axes[1]


def invert_d_star(R, d_arr, tau):
    R = np.asarray(R, dtype=float); d = np.asarray(d_arr, dtype=float)
    if np.any(np.isnan(R)): return np.nan
    R_mono = np.maximum.accumulate(R[::-1])[::-1]
    if R_mono[0] < tau: return d[0]
    if R_mono[-1] > tau: return np.inf
    for i in range(len(d) - 1):
        if R_mono[i] >= tau >= R_mono[i + 1]:
            r0, r1 = R_mono[i], R_mono[i + 1]
            return d[i] + (tau - r0) * (d[i + 1] - d[i]) / (r1 - r0)
    return np.nan


# d*(a) at f=1.0
a_dense = np.linspace(0.3, 0.9, 31)
d_star_a = []
for a in a_dense:
    r = r_at(a, 1.0)
    d_star_a.append(invert_d_star(r, D_VALS, TAU))
d_star_a = np.array([60 if np.isinf(x) else x for x in d_star_a])
ax.plot(a_dense, d_star_a, "-", color="black", lw=2, label="$f$=1.0 (unidir)")

# Also at f=0
d_star_a_f0 = []
for a in a_dense:
    r = r_at(a, 0.0)
    d_star_a_f0.append(invert_d_star(r, D_VALS, TAU))
d_star_a_f0 = np.array([60 if np.isinf(x) else x for x in d_star_a_f0])
ax.plot(a_dense, d_star_a_f0, "--", color="tab:purple", lw=2, label="$f$=0.0 (bidir)")

# Real sites: plot at their actual f
for name, a, f, color, marker in SITES:
    r = r_at(a, f)
    d_star_site = invert_d_star(r, D_VALS, TAU)
    if np.isinf(d_star_site): d_star_site = 60
    ax.plot(a, d_star_site, marker, color=color, markersize=12,
            markeredgecolor="black", linewidth=0)
    ax.annotate(name, (a, d_star_site), textcoords="offset points",
                xytext=(8, 6), fontsize=9)

ax.set_xlabel("$a$ (concentration)")
ax.set_ylabel(f"Required buffer $d^*$ ($D$) for $\\tau={TAU}\\%$ AEP")
ax.set_title(f"Regulator-actionable buffer at $\\tau={TAU}\\%$")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9, loc="upper left")
ax.set_xlim(0.28, 0.92)

fig.suptitle("No Regrets: how much buffer to leave between adjacent wind farms", fontsize=13)
plt.tight_layout()
out = Path("paper_v4/figures/carrier.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(str(out), dpi=200, bbox_inches="tight")
print(f"Saved: {out}")

"""Geometric feasibility of multi-neighbor rings.

How many same-sized neighbor farms can physically surround a target at a given
buffer distance, without violating a minimum neighbor-to-neighbor boundary gap?

This is pure geometry - no wake simulation - and it bounds which regions of the
(n, buffer) parameter space a planner could ever encounter.

Output: paper_v3/figures/ring_feasibility.png
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_ring_regret as rr

D = rr.D
FIGDIR = Path("paper_v3/figures")
BUFFERS = list(range(2, 151, 2))   # fine grid: coarse sampling misplaced the steps
MGS = [(0.5, "C0", "--"), (2.0, "C3", "-"), (10.0, "C4", ":")]
N_MAX_SCAN = 14


def max_feasible_n(buffer_D, mg_D):
    """Largest n such that every ring size up to n also fits.

    Feasibility is not always monotone in n — for an elongated polygon a given
    bearing set can pack better than one with fewer farms — so the planner-
    relevant quantity is the contiguous run from n=1, not the outright maximum.
    """
    best = 0
    for n in range(1, N_MAX_SCAN + 1):
        try:
            *_, mult = rr.ring_offsets(n, 270.0, buffer_D * D, mg_D * D)
        except RuntimeError:
            break
        if abs(mult - 1.0) < 1e-9:
            best = n
        else:
            break
    return best


def site_requirements():
    """Buffer distances our own buffer tables demand at the fitted real sites."""
    out = {}
    for tag, path in [("bast", "analysis/buffer_table_funwake/d_star_summary.json"),
                      ("tp", "analysis/buffer_table_tp_funwake/d_star_summary.json")]:
        p = Path(path)
        if not p.exists():
            continue
        out[tag] = json.load(open(p))
    return out


fig = plt.figure(figsize=(14.5, 5.8))
gs = fig.add_gridspec(1, 3, width_ratios=[1.65, 1, 1], wspace=0.28)
ax = fig.add_subplot(gs[0, 0])

for mg, color, ls in MGS:
    ys = [max_feasible_n(b, mg) for b in BUFFERS]
    ax.step(BUFFERS, ys, ls, where="post", color=color, lw=2,
            label=f"neighbour--neighbour min {mg}$D$")

# analytic model: n ~ pi / arctan(R_farm / R_ring)
p = rr.boundary_np / D
R_farm = np.hypot(p[:, 0], p[:, 1]).mean()
bb = np.linspace(2, 150, 300)
ax.plot(bb, np.pi / np.arctan(R_farm / (R_farm + bb)), color="gray", lw=1.2,
        alpha=0.8, zorder=0,
        label=r"$\pi\,/\,\arctan(R_\mathrm{farm}/R_\mathrm{ring})$")

# reference buffers
ax.axvline(30.9, color="k", lw=1.0, ls="-.", alpha=0.7)
ax.annotate("NYSERDA 4 nmi", (29.5, 1.15), fontsize=7.5, ha="right",
            va="bottom", color="k")
ax.axvspan(7.6, 9.3, color="C0", alpha=0.16)
ax.annotate("Bast.\n$d^*$", (8.4, 0.35), fontsize=7.5, ha="center",
            va="bottom", color="C0")
ax.axvspan(41.9, 60.0, color="C3", alpha=0.16)
ax.annotate("TurboPark $d^*$ (real sites)", (51, 0.25), fontsize=7.5,
            ha="center", va="bottom", color="C3")
ax.axvspan(2, 40, color="gray", alpha=0.07, zorder=0)

ax.set_xlim(2, 150)
ax.set_ylim(0, 9.6)
ax.set_yticks(range(0, 10, 2))
ax.set_xlabel("Buffer distance to target (rotor diameters $D$)")
ax.set_ylabel("Neighbouring farms that physically fit")
ax.set_title("Ring feasibility is set by farm size, not by the buffer")
ax.grid(True, alpha=0.28)
ax.legend(fontsize=7.5, loc="upper left")
ax.annotate("steps are jagged because the footprint is elongated:\n"
            "equal boundary gaps put farms at very unequal radii",
            (150, 7.6), fontsize=7.5, ha="right", va="top", color="dimgray")

secax = ax.secondary_xaxis("top", functions=(lambda x: x * D / 1000,
                                             lambda x: x * 1000 / D))
secax.set_xlabel("km", fontsize=9)

# ---- schematic panels ----
for k, (buf, title) in enumerate([(2, "buffer 2$D$"), (80, "buffer 80$D$")]):
    axs = fig.add_subplot(gs[0, 1 + k])
    n = max_feasible_n(buf, 2.0)
    b, offs, dirs, gaps, mult = rr.ring_offsets(n, 270.0, buf * D, 2.0 * D)
    axs.add_patch(MplPolygon(rr.boundary_np / D, closed=True, fill=True,
                             facecolor="steelblue", alpha=0.30,
                             edgecolor="steelblue", lw=1.3))
    for o, dv in zip(offs, dirs):
        axs.add_patch(MplPolygon((rr.boundary_np + o * np.asarray(dv)) / D,
                                 closed=True, fill=True, facecolor="indianred",
                                 alpha=0.22, edgecolor="indianred", lw=1.0))
    lim = 1.08 * max(np.abs((rr.boundary_np + max(offs) * np.asarray(dirs[0])) / D).max(),
                     R_farm * 1.4)
    axs.set_xlim(-lim, lim)
    axs.set_ylim(-lim, lim)
    axs.set_aspect("equal")
    axs.set_title(f"{title}: {n} fit", fontsize=10)
    axs.set_xlabel("$x/D$")
    axs.grid(True, alpha=0.2)

fig.text(0.5, 0.015,
         "Lease geometry bounds the multi-neighbour problem. The neighbour farms are ~150$D$ across, so the ring radius is set mostly by "
         "their own size: the buffer contributes only ~3% of it at 2$D$.\nAt the 2$D$ neighbour-to-neighbour minimum used throughout this "
         "study, four is the largest ring that fits anywhere in the swept 2--40$D$ range; a fifth first fits at 42$D$ (10 km), by which distance "
         "regret has already collapsed.\nCurves give the largest $n$ for which every smaller ring also fits. Right: farms sit at equal "
         "boundary gap but unequal radius, which is why the steps are not monotone.",
         ha="center", va="bottom", fontsize=9)
fig.subplots_adjust(top=0.86, bottom=0.24)
out = FIGDIR / "ring_feasibility.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
print(f"Saved: {out}")

print("\nmax feasible n by buffer (neighbour--neighbour min 2D):")
for bdist in BUFFERS:
    print(f"  {bdist:>4}D ({bdist*D/1000:5.1f} km): {max_feasible_n(bdist, 2.0)}")

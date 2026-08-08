"""Polar plot of regret vs neighbor bearing at d=5D buffer for a
somewhat concentrated wind rose (a=0.7, f=0.5). Side-by-side Bastankhah vs TurboPark."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.projections.polar import PolarAxes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pathlib import Path

THETA_PREV = 270
N_BINS = 24


def get_rose(a, f):
    from edrose import EllipticalWindRose
    wr = EllipticalWindRose(a=a, f=f, theta_prev=THETA_PREV, n_sectors=N_BINS)
    return np.array(wr.wind_directions), np.array(wr.sector_frequencies)


a, f, d = 0.7, 0.5, 5
fig, axes = plt.subplots(1, 2, figsize=(12, 6),
                         subplot_kw={"projection": "polar"})

for ax, source, label in [(axes[0], "buffer_table",    "Bastankhah"),
                          (axes[1], "buffer_table_tp", "TurboPark")]:
    p = Path(f"analysis/{source}/a{a}_f{f}_d{d}/Nt50/results.json")
    data = json.load(open(p))
    g = np.array(data["regret_grid_gwh"])
    lib = data["liberal_aep_gwh"]
    bearings = np.array(data["bearings_deg"])
    pct = np.clip(100 * g[0, :] / lib, 0, None)

    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    theta = np.deg2rad(bearings)
    width = np.deg2rad(360 / len(bearings)) * 0.95
    bars = ax.bar(theta, pct, width=width,
                  color=plt.cm.YlOrRd(pct / max(pct.max(), 1e-6)),
                  edgecolor="gray", linewidth=0.3)
    ax.set_ylim(0, pct.max() * 1.05)
    ax.set_title(f"{label}: peak {pct.max():.3f}\\%", fontsize=11)

    # Rose inset
    wr_dirs, wr_freq = get_rose(a, f)
    ax_inset = inset_axes(ax, width="28%", height="28%", loc="lower right",
                          axes_class=PolarAxes)
    ax_inset.set_theta_zero_location("N"); ax_inset.set_theta_direction(-1)
    inset_width = np.deg2rad(360 / len(wr_dirs)) * 0.9
    ax_inset.bar(np.deg2rad(wr_dirs), wr_freq, width=inset_width,
                 color="steelblue", alpha=0.8, edgecolor="navy", linewidth=0.3)
    ax_inset.set_yticks([]); ax_inset.set_xticks([])
    ax_inset.patch.set_alpha(0.85)

fig.suptitle(f"Regret vs.\\ neighbor bearing at $d={d}D$, "
             f"$a={a}$, $f={f}$ (somewhat concentrated rose). "
             f"$K_\\text{{lib}}=K_\\text{{cons}}=500$.",
             fontsize=12)
plt.tight_layout()

out = Path(f"paper_v3/figures/polar_d{d}D_a{a}_f{f}.png")
fig.savefig(str(out), dpi=200, bbox_inches="tight")
print(f"Saved: {out}")

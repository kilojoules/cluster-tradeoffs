"""Matched wind-rose shape-space contour charts for both neighbour scenarios.

Two ways of putting neighbours next to a target farm, rendered identically so
they can be compared directly:

  GREEDY   unconstrained individual turbines placed anywhere on a grid around
           the target, chosen to maximise regret (adversarial ceiling).
           Source: analysis/edrose_sweep_k500 (Bastankhah), edrose_sweep_tp.
           Baseline SGD schedule, K = 500, 30 turbines placed.

  RADAR    an identical copy of the target farm at a fixed boundary gap,
           peak taken over 24 bearings (realistic scenario).
           Source: analysis/buffer_table{_tp}_funwake. FunWake, K = 2000.

Outputs
  paper_v3/figures/shape_space_greedy.png       greedy, both wake models
  paper_v3/figures/shape_space_radar.png        radar at 2D, both wake models
  paper_v3/figures/shape_space_compare.png      greedy vs radar, TurboPark
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGDIR = Path("paper_v3/figures")
A_VALS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
F_VALS = [0.0, 0.25, 0.5, 0.75, 1.0]

# name, a, f, label offset (a, f) to keep coincident markers legible
SITES = [
    ("Horns Rev 1",     0.649, 0.384, ( 0.012,  0.045)),
    ("Lillgrund",       0.621, 0.544, ( 0.012,  0.030)),
    ("Borssele",        0.728, 0.291, ( 0.012, -0.055)),
    ("Princess Amalia", 0.650, 0.182, (-0.012, -0.060)),
    ("DEI",             0.637, 0.384, (-0.014, -0.055)),
]


def load_greedy(base):
    z = np.full((len(A_VALS), len(F_VALS)), np.nan)
    for i, a in enumerate(A_VALS):
        for j, f in enumerate(F_VALS):
            p = Path(f"analysis/{base}/a{a}_f{f}/results.json")
            if not p.exists():
                continue
            d = json.load(open(p))
            z[i, j] = 100 * d["regret_gwh"] / d["liberal_aep_gwh"]
    return z


def load_radar(base, d_buf):
    z = np.full((len(A_VALS), len(F_VALS)), np.nan)
    for i, a in enumerate(A_VALS):
        for j, f in enumerate(F_VALS):
            p = Path(f"analysis/{base}/a{a}_f{f}_d{d_buf}/Nt50/results.json")
            if not p.exists():
                continue
            d = json.load(open(p))
            g = np.array(d["regret_grid_gwh"])
            z[i, j] = 100 * g[0, :].max() / d["liberal_aep_gwh"]
    return z


def panel(ax, Z, title, vmax, show_ylabel, show_sites=True):
    A_g, F_g = np.meshgrid(A_VALS, F_VALS, indexing="ij")
    levels = np.linspace(0, vmax, 11)
    im = ax.contourf(A_g, F_g, Z, levels=levels, cmap="YlOrRd", extend="max")
    cs = ax.contour(A_g, F_g, Z, levels=levels, colors="k",
                    linewidths=0.4, alpha=0.45)
    ax.clabel(cs, inline=True, fontsize=6.5, fmt="%.1f")
    if show_sites:
        for name, a, f, (da, df) in SITES:
            ax.plot(a, f, "o", mfc="white", mec="k", mew=1.3, ms=7, zorder=6)
            ax.annotate(name, (a, f), xytext=(a + da, f + df),
                        fontsize=7, fontweight="bold", zorder=7,
                        ha="left" if da > 0 else "right",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                  ec="none", alpha=0.75))
    ax.set_xlabel("$a$  (concentration)")
    if show_ylabel:
        ax.set_ylabel("$f$   (0 = bidirectional, 1 = unidirectional)")
    ax.set_title(title, fontsize=11)
    ax.set_xlim(0.3, 0.9)
    ax.set_ylim(0.0, 1.0)
    return im


def two_panel(z_left, z_right, t_left, t_right, suptitle, sub, out,
              shared_scale=True):
    vmax_l = np.nanmax(z_left)
    vmax_r = np.nanmax(z_right)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4))
    if shared_scale:
        v = max(vmax_l, vmax_r)
        im_l = panel(axes[0], z_left, t_left, v, True)
        im_r = panel(axes[1], z_right, t_right, v, False)
        cax = fig.add_axes([0.915, 0.14, 0.016, 0.68])
        cb = fig.colorbar(im_r, cax=cax)
        cb.set_label("Peak design regret (% of AEP)")
    else:
        im_l = panel(axes[0], z_left, t_left, vmax_l, True)
        fig.colorbar(im_l, ax=axes[0], shrink=0.86, pad=0.02,
                     label="Peak design regret (% of AEP)")
        im_r = panel(axes[1], z_right, t_right, vmax_r, False)
        fig.colorbar(im_r, ax=axes[1], shrink=0.86, pad=0.02,
                     label="Peak design regret (% of AEP)")
    fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=0.995)
    fig.text(0.5, 0.925, sub, fontsize=9.5, ha="center", color="dimgray")
    if not shared_scale:
        fig.tight_layout(rect=[0, 0, 1, 0.90])
    else:
        fig.subplots_adjust(top=0.855, bottom=0.14, left=0.07, right=0.895, wspace=0.16)
    fig.savefig(FIGDIR / out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}   left max {vmax_l:.2f}%   right max {vmax_r:.2f}%")


if __name__ == "__main__":
    g_bast = load_greedy("edrose_sweep_k500")
    g_tp = load_greedy("edrose_sweep_tp")
    r_bast = load_radar("buffer_table_funwake", 2)
    r_tp = load_radar("buffer_table_tp_funwake", 2)

    # 1. Greedy, both wake models (own scales - an order of magnitude apart)
    two_panel(g_bast, g_tp,
              "Bastankhah", "TurboPark",
              "Adversarial ceiling: greedy, boundary-free turbine placement",
              "30 individual turbines placed anywhere on a $5D$ grid around the target, each chosen to maximise regret. "
              "Baseline SGD schedule, $K$ = 500.",
              "shape_space_greedy.png", shared_scale=False)

    # 2. Radar, both wake models
    two_panel(r_bast, r_tp,
              "Bastankhah", "TurboPark",
              "Realistic scenario: identical neighbouring farm at a $2D$ buffer gap",
              "Peak over 24 bearings of an identical copy of the target farm. FunWake schedule, $K$ = 2000.",
              "shape_space_radar.png", shared_scale=False)

    # 3. The comparison that matters: same wake model, two scenarios
    two_panel(g_tp, r_tp,
              "Greedy, boundary-free turbines", "Identical neighbouring farm ($2D$)",
              "TurboPark: adversarial ceiling vs. realistic scenario",
              "Both are peak design regret over the same wind-rose shape space, on a shared colour scale.",
              "shape_space_compare.png", shared_scale=True)

    print("\nshape-space peaks (% of AEP)")
    for lbl, z in [("greedy bast", g_bast), ("greedy tp", g_tp),
                   ("radar bast 2D", r_bast), ("radar tp 2D", r_tp)]:
        print(f"  {lbl:15s} min {np.nanmin(z):5.2f}  max {np.nanmax(z):6.2f}  "
              f"at a={A_VALS[np.unravel_index(np.nanargmax(z), z.shape)[0]]}, "
              f"f={F_VALS[np.unravel_index(np.nanargmax(z), z.shape)[1]]}")

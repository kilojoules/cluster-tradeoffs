"""Mechanism figures for design regret: why regret is non-monotonic in the
number of neighbors, and why wind-rose symmetry and neighbor symmetry are the
same effect.

Core identity:      regret = AEP loss  x  recoverable fraction
Loss rises with encirclement; recoverable fraction falls; the product peaks.

Figures (all from data already on disk):
  paper_v3/figures/mech_phase_trajectory.png   -- loss-regret phase paths (Tier 1.1)
  paper_v3/figures/mech_two_paths.png          -- two routes to symmetry (Tier 1.2)
  paper_v3/figures/mech_decomposition.png      -- loss / irreducible / regret (Tier 1.3)
  paper_v3/figures/mech_escape_rose.png        -- blocked vs open bearings (Tier 2.4)
"""

import json
import glob
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

D = 240.0
FIGDIR = Path("paper_v3/figures")

ROSES = {
    "a0.9_f1.0": ("$a$=0.9, $f$=1.0 (conc. unidir)", "C3"),
    "dei": ("DEI real rose", "C0"),
    "a0.5_f0.0": ("$a$=0.5, $f$=0.0 (mod. bidir)", "C2"),
}
WAKES = [("Bastankhah", "analysis/ring_regret_funwake"),
         ("TurboPark", "analysis/ring_regret_tp_funwake")]


def load_rings(base):
    series = defaultdict(dict)
    for fp in sorted(glob.glob(f"{base}/*/results.json")):
        d = json.load(open(fp))
        key = fp.split("/")[-2].rsplit("_n", 1)[0]
        for r in d["rings"]:
            series[key][r["n_farms"]] = r
    return series


# =============================================================================
# 1. Loss-regret phase trajectory
# =============================================================================
def fig_phase_trajectory():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    for ax, (wake, base) in zip(axes, WAKES):
        series = load_rings(base)
        xmax = 0
        for rose, (label, color) in ROSES.items():
            key = f"{rose}_d2"
            if key not in series:
                continue
            rows = series[key]
            ns = sorted(rows)
            loss = np.array([rows[n]["aep_loss_pct"] for n in ns])
            reg = np.array([rows[n]["regret_pct"] for n in ns])
            feas = [rows[n]["separation_multiplier"] == 1.0 for n in ns]
            xmax = max(xmax, loss.max())
            ax.plot(loss, reg, "-", color=color, lw=1.6, alpha=0.85, label=label)
            for i, n in enumerate(ns):
                ax.plot(loss[i], reg[i], "o", color=color, ms=8, zorder=5,
                        markerfacecolor=color if feas[i] else "white",
                        markeredgecolor=color)
                ax.annotate(str(n), (loss[i], reg[i]), fontsize=7,
                            color="white" if feas[i] else color,
                            ha="center", va="center", zorder=6,
                            fontweight="bold")
            # arrow showing direction of increasing n along the path
            for i in range(len(ns) - 1):
                ax.annotate("", xy=(loss[i + 1], reg[i + 1]),
                            xytext=(loss[i], reg[i]),
                            arrowprops=dict(arrowstyle="-|>", color=color,
                                            alpha=0.5, lw=1.0,
                                            shrinkA=8, shrinkB=8))
        # iso-recoverable-fraction rays
        for frac in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
            ax.plot([0, xmax * 1.05], [0, frac * xmax * 1.05], ":",
                    color="gray", lw=0.7, alpha=0.6, zorder=0)
            ax.annotate(f"{frac:.1f}", (xmax * 1.05, frac * xmax * 1.05),
                        fontsize=7, color="gray", va="center", ha="left")
        ax.set_xlim(0, xmax * 1.16)
        ax.set_ylim(0, None)
        ax.set_xlabel("AEP loss (% of AEP)")
        ax.set_ylabel("Design regret (% of AEP)")
        ax.set_title(f"{wake}")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Loss--regret phase trajectory as neighbors are added ($n$ labels each point, $2D$ nominal gap).\n"
                 "Dotted rays: constant recoverable fraction. Paths climb a steep ray, "
                 "then hook back onto a shallower one --- damage keeps rising while recoverability collapses.",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out = FIGDIR / "mech_phase_trajectory.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# =============================================================================
# 2. Two paths to symmetry
# =============================================================================
def peak_cell_fraction(path):
    """Recoverable fraction + loss at the peak-regret bearing of a cross-section cell."""
    d = json.load(open(path))
    lib = d["liberal_aep_gwh"]
    ev = max(d["evaluations"], key=lambda e: e["regret_gwh"])
    loss = lib - ev["liberal_aep_present_gwh"]
    if loss <= 0:
        return None
    return ev["regret_gwh"] / loss, 100 * loss / lib, 100 * ev["regret_gwh"] / lib


def fig_two_paths():
    F_VALS = [0.0, 0.25, 0.5, 0.75, 1.0]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)
    ax_w, ax_n = axes

    # --- left: symmetry via the wind rose (single neighbor, vary folding f) ---
    for wake, base, color, mk in [("Bastankhah", "analysis/buffer_table_funwake", "C0", "o"),
                                  ("TurboPark", "analysis/buffer_table_tp_funwake", "C3", "s")]:
        for a, ls, alpha in [(0.9, "-", 1.0), (0.5, "--", 0.65)]:
            xs, ys = [], []
            for f in F_VALS:
                p = f"{base}/a{a}_f{f}_d2/Nt50/results.json"
                if not Path(p).exists():
                    continue
                res = peak_cell_fraction(p)
                if res:
                    xs.append(f)
                    ys.append(res[0])
            if xs:
                ax_w.plot(xs, ys, ls, marker=mk, color=color, lw=2, ms=6,
                          alpha=alpha, label=f"{wake}, $a$={a}")
    ax_w.set_xlabel("Folding $f$   (0 = bidirectional $\\rightarrow$ 1 = unidirectional)")
    ax_w.set_ylabel("Recoverable fraction (regret / AEP loss)")
    ax_w.set_title("Symmetry from the wind rose\n(one neighbor, vary directional symmetry)")
    ax_w.grid(True, alpha=0.3)
    ax_w.legend(fontsize=8)
    ax_w.annotate("more symmetric\n$\\rightarrow$ less recoverable", (0.03, 0.06),
                  xycoords="axes fraction", fontsize=8, color="gray")

    # --- right: symmetry via neighbors (fixed rose, vary ring size n) ---
    for wake, base, color, mk in [("Bastankhah", "analysis/ring_regret_funwake", "C0", "o"),
                                  ("TurboPark", "analysis/ring_regret_tp_funwake", "C3", "s")]:
        series = load_rings(base)
        for rose, ls, alpha in [("a0.9_f1.0", "-", 1.0), ("a0.5_f0.0", "--", 0.65)]:
            key = f"{rose}_d2"
            if key not in series:
                continue
            rows = series[key]
            ns = sorted(rows)
            rl = [rows[n]["regret_over_loss"] or np.nan for n in ns]
            feas = [rows[n]["separation_multiplier"] == 1.0 for n in ns]
            lbl = f"{wake}, {'conc. unidir' if rose.startswith('a0.9') else 'mod. bidir'}"
            ax_n.plot(ns, rl, ls, color=color, lw=2, alpha=alpha, label=lbl)
            for n, y, ok in zip(ns, rl, feas):
                ax_n.plot(n, y, mk, color=color, ms=6, alpha=alpha,
                          markerfacecolor=color if ok else "white",
                          markeredgecolor=color)
    ax_n.axvspan(4.5, 8.5, color="gray", alpha=0.08)
    ax_n.set_xlabel("Number of neighboring farms $n$")
    ax_n.set_title("Symmetry from the neighbors\n(fixed rose, vary how encircled the target is)")
    ax_n.set_xticks(range(1, 9))
    ax_n.grid(True, alpha=0.3)
    ax_n.legend(fontsize=8)
    ax_n.annotate("more encircled\n$\\rightarrow$ less recoverable", (0.55, 0.9),
                  xycoords="axes fraction", fontsize=8, color="gray")

    fig.suptitle("Two routes to the same place: symmetrizing the threat destroys the value of neighbor foresight.\n"
                 "Left, symmetry comes from the wind; right, from the neighbors. "
                 "The trend is resolved under TurboPark (large regret signal);\n"
                 "Bastankhah curves sit near the multistart noise floor and are shown for completeness only.",
                 fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.89])
    out = FIGDIR / "mech_two_paths.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# =============================================================================
# 3. Decomposition: recoverable vs irreducible damage
# =============================================================================
def fig_decomposition():
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), sharex=True)
    for row, (wake, base) in enumerate(WAKES):
        series = load_rings(base)
        for col, (rose, (label, color)) in enumerate(ROSES.items()):
            ax = axes[row][col]
            key = f"{rose}_d2"
            if key not in series:
                ax.set_visible(False)
                continue
            rows = series[key]
            ns = sorted(rows)
            loss = np.array([rows[n]["aep_loss_pct"] for n in ns])
            reg = np.array([rows[n]["regret_pct"] for n in ns])
            ax.fill_between(ns, 0, reg, color=color, alpha=0.75,
                            label="recoverable by re-design (regret)")
            ax.fill_between(ns, reg, loss, color="gray", alpha=0.30,
                            label="irreducible damage")
            ax.plot(ns, loss, "-", color="k", lw=1.4, label="total AEP loss")
            ax.axvspan(4.5, 8.5, color="gray", alpha=0.08)
            ax.set_xticks(range(1, 9))
            ax.grid(True, alpha=0.25)
            if row == 0:
                ax.set_title(label, fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{wake}\n% of AEP")
            if row == 1:
                ax.set_xlabel("Number of neighboring farms $n$")
            if row == 0 and col == 0:
                ax.legend(fontsize=7.5, loc="upper left")
    fig.suptitle("Decomposition: regret = AEP loss $\\times$ recoverable fraction ($2D$ nominal gap).\n"
                 "The grey band is damage no re-design can undo; it widens as the target is encircled.",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    out = FIGDIR / "mech_decomposition.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# =============================================================================
# 4. Escape-angle rose
# =============================================================================
def true_blocked_sectors(n, base_bearing=270.0, gap_D=2.0):
    """Angular sector each neighbor actually subtends at the target centroid.

    Recomputes the production ring geometry (deterministic, no simulation) and
    measures, for every neighbor polygon, the min/max bearing of its vertices
    as seen from the target centroid at the origin.
    """
    import run_ring_regret as rr
    bearings, offsets, dirs, gaps, mult = rr.ring_offsets(
        n, base_bearing, gap_D * D, 2.0 * D)
    sectors = []
    for off, dirn in zip(offsets, dirs):
        poly = rr.boundary_np + off * np.asarray(dirn)
        th = np.degrees(np.arctan2(poly[:, 0], poly[:, 1])) % 360.0  # bearing conv.
        centre = np.degrees(np.arctan2((off * np.asarray(dirn))[0],
                                       (off * np.asarray(dirn))[1])) % 360.0
        rel = (th - centre + 180.0) % 360.0 - 180.0
        sectors.append((centre, rel.min(), rel.max()))
    open_frac = max(0.0, 1.0 - sum(hi - lo for _, lo, hi in sectors) / 360.0)
    return sectors, open_frac, gaps


def fig_escape_rose():
    base = "analysis/ring_regret_tp_funwake"
    series = load_rings(base)
    rows = series.get("a0.9_f1.0_d2", {})
    show_n = [n for n in [1, 2, 4, 8] if n in rows]
    fig, axes = plt.subplots(1, len(show_n), figsize=(4.1 * len(show_n), 5.1),
                             subplot_kw={"projection": "polar"})
    if len(show_n) == 1:
        axes = [axes]
    for ax, n in zip(axes, show_n):
        r = rows[n]
        sectors, open_frac, gaps = true_blocked_sectors(n)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        for centre, lo, hi in sectors:
            ax.bar(np.radians(centre + 0.5 * (lo + hi)), 1.0,
                   width=np.radians(hi - lo), bottom=0.0,
                   color="firebrick", alpha=0.55, edgecolor="darkred", linewidth=0.6)
        # prevailing wind direction (270 deg = from west)
        ax.annotate("", xy=(np.radians(270), 0.98), xytext=(np.radians(270), 1.32),
                    arrowprops=dict(arrowstyle="-|>", color="navy", lw=2))
        ax.set_ylim(0, 1.35)
        ax.set_yticks([])
        ax.set_xticks(np.radians([0, 90, 180, 270]))
        ax.set_xticklabels(["N", "E", "S", "W"], fontsize=8)
        frac = r["regret_over_loss"] or np.nan
        ax.set_title(f"$n$={n}   gap {min(gaps)/D:.0f}$D$\n"
                     f"open bearings {100*open_frac:.0f}%   "
                     f"recoverable {frac:.2f}",
                     fontsize=9.5, pad=12)
    fig.suptitle("Escape geometry: red sectors are the bearings actually subtended by neighboring farms "
                 "(TurboPark, conc. unidirectional rose).\n"
                 "Blue arrow: prevailing wind. Beyond $n$=4 the packing constraint pushes farms outward, "
                 "so each subtends less --- but the target is\nsurrounded, and the recoverable share of the damage still falls.",
                 fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    out = FIGDIR / "mech_escape_rose.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig_phase_trajectory()
    fig_two_paths()
    fig_decomposition()
    fig_escape_rose()

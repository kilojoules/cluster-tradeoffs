"""Tier-3 mechanism figures (require the gbar `viz` campaign).

  paper_v3/figures/mech_displacement.png  -- liberal->conservative turbine
      displacement field vs number of neighbors (needs --save-layouts runs)
  paper_v3/figures/mech_angular_spread.png -- fixed total neighbor capacity and
      fixed buffer gap, varying angular spread (needs --split-neighbors runs)

Both skip gracefully if the corresponding runs are not yet on disk.
"""

import json
import glob
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = 240.0
FIGDIR = Path("paper_v3/figures")


def load_rings(pattern):
    series = defaultdict(dict)
    for fp in sorted(glob.glob(pattern)):
        d = json.load(open(fp))
        key = fp.split("/")[-2].rsplit("_", 1)[0]  # strip chunk suffix (_A1/_B2)
        for r in d["rings"]:
            r["_liberal_x"] = d.get("liberal_x")
            r["_liberal_y"] = d.get("liberal_y")
            r["_liberal_aep"] = d.get("liberal_aep_gwh")
            series[key][r["n_farms"]] = r
    return series


# =============================================================================
# Displacement field
# =============================================================================
def fig_displacement(rose="a0.9_f1.0"):
    """Use whichever wake model has the most complete layout series."""
    best, wake_tag, wake_name, rows = -1, None, None, {}
    for tag, name in [("_tp", "TurboPark"), ("", "Bastankhah")]:
        s = load_rings(f"analysis/ring_disp{tag}_funwake/*/results.json").get(rose, {})
        have = len([n for n in [1, 2, 4, 8] if n in s and "cons_x" in s[n]])
        if have > best:
            best, wake_tag, wake_name, rows = have, tag, name, s
    show = [n for n in [1, 2, 4, 8] if n in rows and "cons_x" in rows[n]]
    if not show:
        print(f"[displacement] no layout data yet for {rose} — skipping")
        return False
    print(f"[displacement] using {wake_name} ({len(show)} ring sizes)")
    _lx0 = np.array(rows[show[0]]["_liberal_x"]) / D
    _ly0 = np.array(rows[show[0]]["_liberal_y"]) / D
    span_D = 1.9 * max(np.abs(np.concatenate([_lx0, _ly0])).max(), 1.0)
    fig, axes = plt.subplots(1, len(show), figsize=(4.3 * len(show), 4.8))
    if len(show) == 1:
        axes = [axes]
    for ax, n in zip(axes, show):
        r = rows[n]
        lx = np.array(r["_liberal_x"]) / D
        ly = np.array(r["_liberal_y"]) / D
        cx = np.array(r["cons_x"]) / D
        cy = np.array(r["cons_y"]) / D
        nx = np.array(r["neighbor_x"]) / D
        ny = np.array(r["neighbor_y"]) / D
        # Liberal and conservative layouts are independent multistart optima, so
        # turbine index carries no correspondence. Pair them by the assignment
        # that minimises total travel — the physically meaningful "how far did
        # the layout have to move" measure.
        from scipy.optimize import linear_sum_assignment
        cost = np.hypot(lx[:, None] - cx[None, :], ly[:, None] - cy[None, :])
        ri, ci = linear_sum_assignment(cost)
        lx, ly = lx[ri], ly[ri]
        cx, cy = cx[ci], cy[ci]
        # Farm boundary polygons, so the buffer geometry is unambiguous.
        import run_ring_regret as rr
        from matplotlib.patches import Polygon as MplPolygon
        poly_t = rr.boundary_np / D
        ax.add_patch(MplPolygon(poly_t, closed=True, fill=False,
                                edgecolor="steelblue", lw=1.1, alpha=0.85, zorder=1))
        _b, _offs, _dirs, _gaps, _mult = rr.ring_offsets(n, 270.0, 2 * D, 2 * D)
        for _o, _dv in zip(_offs, _dirs):
            pn = (rr.boundary_np + _o * np.asarray(_dv)) / D
            ax.add_patch(MplPolygon(pn, closed=True, fill=False,
                                    edgecolor="indianred", lw=0.9, alpha=0.7,
                                    linestyle="--", zorder=1))
        gap_lbl = f"{min(_gaps)/D:.0f}$D$ gap"
        ax.plot(nx, ny, ".", color="lightcoral", ms=2.5, alpha=0.7,
                label="neighbor turbines" if n == show[0] else None)
        ax.plot(lx, ly, "o", mfc="none", mec="steelblue", ms=4.5, mew=0.9,
                label="liberal layout" if n == show[0] else None)
        disp = np.hypot(cx - lx, cy - ly)
        ax.quiver(lx, ly, cx - lx, cy - ly, angles="xy", scale_units="xy",
                  scale=1, width=0.005, color="k", alpha=0.85)
        # Common zoom on the target farm across every panel, so the arrows stay
        # comparable; distant neighbors simply clip out of view.
        ax.set_xlim(-span_D, span_D)
        ax.set_ylim(-span_D, span_D)
        ax.set_aspect("equal")
        ax.set_title(f"$n$={n}   {gap_lbl}   median shift {np.median(disp):.1f}$D$\n"
                     f"regret {r['regret_pct']:.2f}%   "
                     f"recoverable {r['regret_over_loss'] or float('nan'):.2f}",
                     fontsize=9.5)
        ax.set_xlabel("$x/D$")
        if n == show[0]:
            ax.set_ylabel("$y/D$")
            ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.2)
    fig.suptitle(f"Where re-design actually moves turbines ({wake_name}, conc. unidirectional rose): arrows run from the "
                 "liberal layout to the neighbor-aware layout, paired by minimum-travel assignment\n(the two are "
                 "independent multistart optima, so turbine index carries no correspondence). Solid outline: target "
                 "boundary; dashed: neighbors, at the realized buffer gap.\nHow far the layout travels tracks how much "
                 "regret is recovered — both peak at $n$=4 and collapse once packing forces the ring out to 123$D$.",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out = FIGDIR / "mech_displacement.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return True


# =============================================================================
# Angular spread at fixed capacity and fixed gap
# =============================================================================
def fig_angular_spread():
    have = False
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.9))
    styles = [("Bastankhah", "", "C0", "o"), ("TurboPark", "_tp", "C3", "s")]
    roses = [("a0.9_f1.0", "-", "conc. unidir"), ("a0.5_f0.0", "--", "mod. bidir")]
    for wake, tag, color, mk in styles:
        split = load_rings(f"analysis/ring_split{tag}_funwake/*/results.json")
        ident = load_rings(f"analysis/ring_regret{tag}_funwake/*/results.json")
        for rose, ls, rlabel in roses:
            rows = split.get(rose, {})
            if not rows:
                continue
            have = True
            ns = sorted(rows)
            # only points where the nominal gap is actually realized
            keep = [n for n in ns if rows[n]["separation_multiplier"] == 1.0]
            reg = [rows[n]["regret_pct"] for n in keep]
            loss = [rows[n]["aep_loss_pct"] for n in keep]
            rl = [rows[n]["regret_over_loss"] or np.nan for n in keep]
            lbl = f"{wake}, {rlabel}"
            axes[0].plot(keep, reg, ls, marker=mk, color=color, lw=2, ms=6, label=lbl)
            axes[1].plot(keep, loss, ls, marker=mk, color=color, lw=2, ms=6, label=lbl)
            axes[2].plot(keep, rl, ls, marker=mk, color=color, lw=2, ms=6, label=lbl)
            # identical-copy reference (same rose/wake) in faint grey
            irows = ident.get(f"{rose}_d2", {})
            ikeep = [n for n in sorted(irows)
                     if irows[n]["separation_multiplier"] == 1.0]
            if ikeep:
                axes[0].plot(ikeep, [irows[n]["regret_pct"] for n in ikeep], ":",
                             color=color, lw=1.2, alpha=0.45)
    if not have:
        print("[angular-spread] no split-neighbor data yet — skipping")
        plt.close(fig)
        return False
    for ax, ylab, title in zip(
            axes,
            ["Design regret (% of AEP)", "AEP loss (% of AEP)",
             "Recoverable fraction"],
            ["Regret", "AEP loss", "Recoverable fraction"]):
        ax.set_xlabel("Number of neighboring farms $n$\n(total capacity, area and buffer gap held fixed)")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.set_xticks(range(1, 6))
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=7.5)
    fig.suptitle("Dividing a fixed neighbor capacity: the same total turbines and area at the same $2D$ gap, split into "
                 "$n$ smaller farms around the target.\nLoss follows the wind — spreading raises it under a bidirectional "
                 "rose and lowers it under a unidirectional one — but the recoverable fraction rises in both,\nbecause every "
                 "individual wake corridor gets narrower. Concentration of wake mass, not angular symmetry, is what closes "
                 "escape routes. Dotted: identical-copy ring.", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.87])
    out = FIGDIR / "mech_angular_spread.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return True


if __name__ == "__main__":
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig_displacement("a0.9_f1.0")
    fig_angular_spread()

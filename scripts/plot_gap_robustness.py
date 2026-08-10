"""Robustness of the multi-neighbour result to the inter-neighbour spacing rule.

The ring construction requires a minimum boundary gap between neighbouring
farms.  That constant was chosen (2D) rather than derived, and it controls when
the ring is forced outward.  This sweeps it over 0.5-20D and asks whether any
conclusion moves.

Output: paper_v3/figures/gap_robustness.png
"""

import json
import glob
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGDIR = Path("paper_v3/figures")
MGS = [(0.5, "C0", "o", "-"), (2.0, "C3", "s", "-"),
       (10.0, "C2", "^", "--"), (20.0, "C4", "D", ":")]
ROSES = [("a0.9_f1.0", "conc. unidirectional ($a$=0.9, $f$=1.0)"),
         ("a0.5_f0.0", "mod. bidirectional ($a$=0.5, $f$=0.0)")]


def load():
    tab = defaultdict(dict)
    for fp in sorted(glob.glob("analysis/ring_gap_tp_funwake/*/results.json")):
        d = json.load(open(fp))
        rose = fp.split("/")[-2].split("_mg")[0]
        mg = d["min_neighbor_gap_D"]
        for r in d["rings"]:
            tab[(rose, r["n_farms"])][mg] = r
    # n = 1..3 never engage the constraint; take them from the production ring runs
    for fp in sorted(glob.glob("analysis/ring_regret_tp_funwake/*_d2_*/results.json")):
        d = json.load(open(fp))
        rose = fp.split("/")[-2].rsplit("_d2", 1)[0]
        for r in d["rings"]:
            for mg, *_ in MGS:
                tab[(rose, r["n_farms"])].setdefault(mg, r)
    return tab


tab = load()
fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)

for row, (rose, rose_label) in enumerate(ROSES):
    for mg, color, mk, ls in MGS:
        ns, reg, rl, gap = [], [], [], []
        for n in range(1, 9):
            r = tab.get((rose, n), {}).get(mg)
            if r is None:
                continue
            ns.append(n)
            reg.append(r["regret_pct"])
            rl.append(r["regret_over_loss"] or np.nan)
            gap.append(min(r["realized_target_gaps_D"]))
        if not ns:
            continue
        lab = f"min {mg}$D$"
        axes[row][0].plot(ns, reg, ls, marker=mk, color=color, lw=1.8, ms=5, label=lab)
        axes[row][1].plot(ns, rl, ls, marker=mk, color=color, lw=1.8, ms=5, label=lab)
        axes[row][2].plot(ns, gap, ls, marker=mk, color=color, lw=1.8, ms=5, label=lab)

    axes[row][0].set_ylabel(f"{rose_label}\n\nDesign regret (% of AEP)")
    axes[row][1].set_ylabel("Recoverable fraction")
    axes[row][2].set_ylabel("Realized buffer gap ($D$)")
    axes[row][2].set_yscale("log")
    for c in range(3):
        axes[row][c].axvspan(0.5, 3.5, color="gray", alpha=0.07)
        axes[row][c].grid(True, alpha=0.28)
        axes[row][c].set_xticks(range(1, 9))
    axes[row][0].legend(fontsize=7.5, title="neighbour--neighbour", title_fontsize=7.5)

for c, t in enumerate(["Regret", "Recoverable fraction", "Geometry the rule produces"]):
    axes[0][c].set_title(t)
for c in range(3):
    axes[1][c].set_xlabel("Number of neighbouring farms $n$")

axes[0][1].annotate("curves collapse: the mechanism\ndoes not depend on the rule",
                    (0.5, 0.97), xycoords="axes fraction", fontsize=8,
                    ha="center", va="top", color="dimgray")
axes[0][0].annotate("shaded: constraint\nis slack here", (0.03, 0.06),
                    xycoords="axes fraction", fontsize=7.5, va="bottom",
                    color="dimgray")

fig.suptitle("Does the inter-neighbour spacing rule drive the result?  TurboPark, FunWake schedule, $K$=500, requested buffer 2$D$.\n"
             "Varying the rule over a 40-fold range moves the geometry it produces by up to 4.5$\\times$ (right column) and shifts regret "
             "magnitudes modestly (left),\nbut the recoverable fraction — the quantity the mechanism is stated in — barely moves, and the "
             "peak at $n$=4 survives everywhere. For $n\\leq3$ the\nneighbours are naturally 50--70$D$ apart, so the rule never engages and "
             "all four settings are identical by construction.", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.91])
out = FIGDIR / "gap_robustness.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
print(f"Saved: {out}")

for rose, lbl in ROSES:
    print(f"\n[{rose}] recoverable fraction spread across the four spacing rules")
    for n in range(1, 9):
        vals = [tab.get((rose, n), {}).get(mg) for mg, *_ in MGS]
        vals = [v["regret_over_loss"] for v in vals if v and v["regret_over_loss"]]
        if len(vals) > 1:
            print(f"  n={n}: {min(vals):.2f}-{max(vals):.2f}  (spread {max(vals)-min(vals):.02f})")

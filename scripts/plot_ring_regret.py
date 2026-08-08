"""Ring-regret figure: regret, loss, and recoverable fraction vs number of
neighboring farms, both wake models.

Reads analysis/ring_regret_funwake/ and analysis/ring_regret_tp_funwake/
(chunked job dirs merged by rose_dist key).

Output: paper_v3/figures/ring_regret_fw.png
"""

import json
import glob
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROSES = {
    "dei": ("DEI real rose", "C0"),
    "a0.9_f1.0": ("$a$=0.9, $f$=1.0 (conc. unidir)", "C3"),
    "a0.5_f0.0": ("$a$=0.5, $f$=0.0 (mod. bidir)", "C2"),
}
DISTS = {2: "-", 10: "--"}
WAKES = [("Bastankhah", "analysis/ring_regret_funwake"),
         ("TurboPark", "analysis/ring_regret_tp_funwake")]


def load(base):
    series = defaultdict(dict)
    for fp in sorted(glob.glob(f"{base}/*/results.json")):
        d = json.load(open(fp))
        key = fp.split("/")[-2].rsplit("_n", 1)[0]
        for r in d["rings"]:
            series[key][r["n_farms"]] = r
    return series


fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True)
for row, (wake, base) in enumerate(WAKES):
    series = load(base)
    ax_r, ax_l, ax_f = axes[row]
    for rose, (label, color) in ROSES.items():
        for d, ls in DISTS.items():
            key = f"{rose}_d{d}"
            if key not in series:
                continue
            rows = series[key]
            ns = sorted(rows)
            reg = [rows[n]["regret_pct"] for n in ns]
            loss = [rows[n]["aep_loss_pct"] for n in ns]
            rl = [rows[n]["regret_over_loss"] or np.nan for n in ns]
            feas = [rows[n]["separation_multiplier"] == 1.0 for n in ns]
            lab = f"{label}, {d}$D$" if row == 0 else None
            for ax, ys in [(ax_r, reg), (ax_l, loss), (ax_f, rl)]:
                ax.plot(ns, ys, ls, color=color, lw=1.8, alpha=0.9, label=lab)
                lab = None
                # filled markers where the nominal gap is realized, open beyond
                for n, y, ok in zip(ns, ys, feas):
                    ax.plot(n, y, "o", color=color, markersize=6,
                            markerfacecolor=color if ok else "white",
                            markeredgecolor=color)
    for ax in (ax_r, ax_l, ax_f):
        ax.axvspan(4.5, 8.5, color="gray", alpha=0.08)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.5, 8.5)
        ax.set_xticks(range(1, 9))
    ax_r.set_ylabel(f"{wake}\nPooled regret (% of AEP)")
    ax_l.set_ylabel("AEP loss (% of AEP)")
    ax_f.set_ylabel("Recoverable fraction (regret/loss)")
    ax_f.set_ylim(0, 0.75)

for ax in axes[1]:
    ax.set_xlabel("Number of neighboring farms $n$")
axes[0][0].set_title("Design regret")
axes[0][1].set_title("AEP loss (no re-optimization)")
axes[0][2].set_title("Recoverable fraction")
axes[0][0].legend(fontsize=8, loc="upper left")

fig.suptitle("Ring sweep: $n$ identical-copy neighbors evenly spaced around the target "
             "(FunWake schedule, $K$=500).\n"
             "Filled markers: all farms at nominal gap. Open markers (shaded region): "
             "ring forced outward by the 2$D$ inter-neighbor packing constraint.",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.93])
out = "paper_v3/figures/ring_regret_fw.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
print(f"Saved: {out}")

# console summary for tex numbers
for wake, base in WAKES:
    series = load(base)
    print(f"\n[{wake}]")
    for key in sorted(series):
        rows = series[key]
        ns = sorted(rows)
        n1 = rows[1]["regret_pct"] if 1 in rows else np.nan
        feas_ns = [n for n in ns if rows[n]["separation_multiplier"] == 1.0]
        n4 = rows[max(feas_ns)]["regret_pct"]
        n8 = rows[max(ns)]["regret_pct"]
        print(f"  {key:18s} n1={n1:.2f}%  n{max(feas_ns)}={n4:.2f}% "
              f"({n4/n1 if n1 else np.nan:.2f}x)  n{max(ns)}={n8:.2f}%")

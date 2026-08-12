"""Optimizer effect on greedy adversarial regret, at fixed neighbour configuration.

Left  - controlled comparison: the same 30 placed neighbour turbines, the same
        multistart set, only the inner SGD schedule differs.
Right - reproducibility: the originally reported greedy value against this
        re-evaluation.  Most cells agree; the outliers show that a greedy
        configuration is partly co-adapted to the liberal layout it was
        optimised against, since that layout is not stored and cannot be reused.

Output: paper_v3/figures/greedy_reeval.png
"""

import json
import glob
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGDIR = Path("paper_v3/figures")
ORDER = ["a0.5_f0.0", "a0.9_f0.5", "a0.7_f1.0", "a0.9_f1.0"]
NICE = {"a0.5_f0.0": "$a$0.5\n$f$0.0", "a0.9_f0.5": "$a$0.9\n$f$0.5",
        "a0.7_f1.0": "$a$0.7\n$f$1.0", "a0.9_f1.0": "$a$0.9\n$f$1.0"}


def load():
    out = {}
    for fp in sorted(glob.glob("analysis/greedy_reeval*/*/results.json")):
        d = json.load(open(fp))
        tag = "TurboPark" if "_tp" in fp else "Bastankhah"
        out[(tag, fp.split("/")[-2])] = d
    return out


data = load()
fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.2))

# ---- left: baseline vs FunWake, same configuration and starts ----
ax = axes[0]
w = 0.36
for wi, (wake, color) in enumerate([("Bastankhah", "C0"), ("TurboPark", "C3")]):
    cells = [c for c in ORDER if (wake, c) in data]
    xs = np.arange(len(cells)) + wi * (len(ORDER) + 0.8)
    base = [data[(wake, c)]["by_schedule"]["sgd_baseline"]["regret_pct"] for c in cells]
    fw = [data[(wake, c)]["by_schedule"]["funwake_iter192"]["regret_pct"] for c in cells]
    ax.bar(xs - w / 2, base, w, color=color, alpha=0.42,
           label=f"{wake}, baseline schedule")
    ax.bar(xs + w / 2, fw, w, color=color, alpha=0.95,
           label=f"{wake}, FunWake schedule")
    for x, b, f in zip(xs, base, fw):
        ax.annotate(f"{f/b:.2f}$\\times$", (x, max(b, f)), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=8.5,
                    fontweight="bold", color=color)
    ax.set_xticks(list(ax.get_xticks()) + list(xs))
ticks, labels = [], []
for wi, wake in enumerate(["Bastankhah", "TurboPark"]):
    cells = [c for c in ORDER if (wake, c) in data]
    for k, c in enumerate(cells):
        ticks.append(k + wi * (len(ORDER) + 0.8))
        labels.append(NICE[c])
ax.set_xticks(ticks)
ax.set_xticklabels(labels, fontsize=8.5)
ax.set_ylabel("Design regret (% of AEP)")
ax.set_title("Same neighbours, same starts, different inner optimizer")
ax.grid(True, axis="y", alpha=0.3)
ax.legend(fontsize=8, loc="upper left")
ax.annotate("Bastankhah", (1.5, -0.19), xycoords=("data", "axes fraction"),
            ha="center", fontsize=9.5, fontweight="bold", color="C0")
ax.annotate("TurboPark", (1.5 + len(ORDER) + 0.8, -0.19),
            xycoords=("data", "axes fraction"), ha="center", fontsize=9.5,
            fontweight="bold", color="C3")

# ---- right: co-adaptation - reported value vs re-evaluation ----
ax = axes[1]
for wake, color, mk in [("Bastankhah", "C0", "o"), ("TurboPark", "C3", "s")]:
    cells = [c for c in ORDER if (wake, c) in data]
    src = [data[(wake, c)]["source_regret_pct"] for c in cells]
    fw = [data[(wake, c)]["by_schedule"]["funwake_iter192"]["regret_pct"] for c in cells]
    ax.plot(src, fw, mk, color=color, ms=10, label=wake)
    for s, f, c in zip(src, fw, cells):
        ax.annotate(c.replace("_", " "), (s, f), xytext=(6, -9),
                    textcoords="offset points", fontsize=7.5, color=color)
lim = 10.0
ax.plot([0, lim], [0, lim], "k--", lw=1.2, alpha=0.6)
ax.annotate("agreement", (8.8, 9.15), fontsize=8, color="dimgray",
            ha="center", rotation=45)
ax.fill_between([0, lim], [0, 0], [0, lim], color="gray", alpha=0.06)
ax.annotate("below the line: those neighbours are less\ndamaging to a different liberal optimum",
            (9.6, 0.5), fontsize=8, color="dimgray", ha="right")
ax.set_xlim(0, lim)
ax.set_ylim(0, lim)
ax.set_aspect("equal")
ax.set_xlabel("Regret reported by the greedy search (% of AEP)")
ax.set_ylabel("Regret re-evaluated here, FunWake (% of AEP)")
ax.set_title("Reproducibility against a different liberal optimum")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9, loc="upper left")

fig.suptitle("Greedy adversarial regret depends on the optimizer that produced it",
             fontsize=13, fontweight="bold", y=1.045)
fig.text(0.5, 0.965,
         "Left: holding the 30 placed neighbours and the multistart set fixed, a stronger inner schedule raises measured regret by "
         "1.1--2.1$\\times$ (median 1.4).\nRight: six of eight cells reproduce the reported value within 25%, but the two extremes do not --- "
         "including the headline adversarial peak (TurboPark $a$0.9 $f$1.0, 9.0% reported vs 5.3% here),\nbecause the greedy search "
         "co-adapts its neighbours to the particular liberal layout it optimised against.",
         fontsize=9.5, ha="center", color="dimgray")
fig.tight_layout(rect=[0, 0, 1, 0.93])
out = FIGDIR / "greedy_reeval.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved: {out}")

print("\nfw/base by cell:")
for (wake, c), d in sorted(data.items()):
    b = d["by_schedule"]["sgd_baseline"]["regret_pct"]
    f = d["by_schedule"]["funwake_iter192"]["regret_pct"]
    print(f"  {wake:11s} {c:10s} {b:5.2f}% -> {f:5.2f}%  ({f/b:.2f}x)   "
          f"source {d['source_regret_pct']:5.2f}%")

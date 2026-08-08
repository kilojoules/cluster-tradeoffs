"""Regret-vs-K convergence from stored per-start AEPs (all_cons_aeps_gwh).

Every FunWake buffer_table evaluation stores all 2000 conservative-start AEPs.
The expected regret at K starts is computed EXACTLY via order statistics of a
size-K subsample without replacement (no bootstrap noise):

    P(max = a_(i)) = C(i-1, K-1) / C(n, K)   for sorted a_(1) <= ... <= a_(n)

Regret uses the pooled estimator max(best conservative over K, liberal_present)
- liberal_present, matching the runner.

Outputs:
    paper_v3/figures/funwake_convergence.png
      panel 1: expected peak-bearing regret vs K, representative cells
      panel 2: capture fraction at K=500 across ALL grid cells (bast vs tp)
    analysis/funwake_convergence_summary.json
"""

import json
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.special import gammaln

K_GRID = np.unique(np.round(np.geomspace(1, 2000, 40)).astype(int))
K_REF = 500


def expected_max_subsample(vals, Ks):
    """E[max of a size-K uniform subsample w/o replacement], exact, all K at once."""
    a = np.sort(np.asarray(vals))
    n = len(a)
    i = np.arange(1, n + 1)
    out = np.empty(len(Ks))
    for m, K in enumerate(Ks):
        if K >= n:
            out[m] = a[-1]
            continue
        # log C(i-1, K-1) - log C(n, K), zero weight where i < K
        with np.errstate(divide="ignore"):
            logw = (gammaln(i) - gammaln(K) - gammaln(i - K + 1)
                    - (gammaln(n + 1) - gammaln(K + 1) - gammaln(n - K + 1)))
        w = np.where(i >= K, np.exp(logw), 0.0)
        out[m] = float(np.dot(w, a))
    return out


def cell_curves(fp, Ks):
    """Expected regret vs K at the cell's peak bearing (peak at full K)."""
    d = json.load(open(fp))
    best_regret, best_curve = -np.inf, None
    for ev in d["evaluations"]:
        aeps = np.asarray(ev["all_cons_aeps_gwh"])
        lib = ev["liberal_aep_present_gwh"]
        full = max(aeps.max(), lib) - lib
        if full > best_regret:
            best_regret = full
            best_curve = np.maximum(expected_max_subsample(aeps, Ks), lib) - lib
    return best_curve, best_regret, d["liberal_aep_gwh"]


# ---- panel 1: representative cells ----
REPS = [
    ("bast a0.9 f1.0 d2",  "analysis/buffer_table_funwake/a0.9_f1.0_d2/Nt50/results.json",  "C0", "-"),
    ("bast a0.5 f0.0 d2",  "analysis/buffer_table_funwake/a0.5_f0.0_d2/Nt50/results.json",  "C0", "--"),
    ("tp a0.9 f1.0 d2",    "analysis/buffer_table_tp_funwake/a0.9_f1.0_d2/Nt50/results.json", "C3", "-"),
    ("tp a0.5 f0.0 d2",    "analysis/buffer_table_tp_funwake/a0.5_f0.0_d2/Nt50/results.json", "C3", "--"),
]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
rep_out = {}
for label, fp, color, ls in REPS:
    curve, full, lib = cell_curves(fp, K_GRID)
    pct = 100 * curve / lib
    ax.plot(K_GRID, pct, color=color, ls=ls, lw=2, label=label)
    rep_out[label] = {"K": K_GRID.tolist(), "regret_pct": pct.tolist(),
                      "full_regret_gwh": full}
ax.axvline(K_REF, color="gray", ls=":", lw=1)
ax.text(K_REF, 0.02, f" $K{{=}}{K_REF}$", fontsize=8, color="gray",
        va="bottom", transform=ax.get_xaxis_transform())
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Number of conservative starts $K$")
ax.set_ylabel("Expected peak-bearing regret (% of AEP)")
ax.set_title("Regret vs.\\ $K$ (exact subsample expectation)")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)

# ---- panel 2: capture at K=500 across the full grids ----
ax = axes[1]
summary = {}
for key, pat, color in [
    ("bast", "analysis/buffer_table_funwake/*/Nt50/results.json", "C0"),
    ("tp",   "analysis/buffer_table_tp_funwake/*/Nt50/results.json", "C3"),
]:
    captures, fulls = [], []
    for fp in sorted(glob.glob(pat)):
        curve, full, lib = cell_curves(fp, np.array([K_REF, 2000]))
        if full <= 0:
            continue
        captures.append(curve[0] / full)
        fulls.append(full)
    captures = np.asarray(captures)
    ax.hist(100 * captures, bins=np.arange(30, 101, 2.5), alpha=0.55,
            color=color, label=f"{key} (n={len(captures)})")
    q = np.percentile(captures, [5, 25, 50, 75, 95])
    summary[key] = {
        "n_cells": len(captures),
        "capture_at_K500_quantiles_5_25_50_75_95": (100 * q).round(1).tolist(),
        "capture_min_pct": round(100 * captures.min(), 1),
        "peak_regret_gwh_max": float(np.max(fulls)),
    }
    print(f"[{key}] capture@K={K_REF}: median {100*q[2]:.1f}%  "
          f"[5-95%: {100*q[0]:.1f}-{100*q[4]:.1f}]  min {100*captures.min():.1f}%  "
          f"(n={len(captures)})")
ax.set_xlabel(f"Captured fraction of $K{{=}}2000$ regret at $K{{=}}{K_REF}$ (%)")
ax.set_ylabel("Grid cells")
ax.set_title(f"Capture at $K{{=}}{K_REF}$, all $(a,f,d)$ cells")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)

fig.suptitle("FunWake schedule: multistart convergence from stored per-start AEPs")
out = "paper_v3/figures/funwake_convergence.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
print(f"Saved: {out}")

with open("analysis/funwake_convergence_summary.json", "w") as f:
    json.dump({"K_ref": K_REF, "representative": rep_out, "capture": summary}, f, indent=2)
print("Saved: analysis/funwake_convergence_summary.json")

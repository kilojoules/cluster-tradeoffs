"""Fit regression model for regret as a function of edrose parameters (a, f)."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

a_vals = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
f_vals = [0.0, 0.25, 0.5, 0.75, 1.0]

# Load all 35 results
data = []
for a in a_vals:
    for f in f_vals:
        p = Path(f"analysis/edrose_sweep/a{a}_f{f}/results.json")
        with open(p) as fh:
            d = json.load(fh)
        regret_pct = 100 * d["regret_gwh"] / d["liberal_aep_gwh"]
        data.append({"a": a, "f": f, "regret_gwh": d["regret_gwh"],
                     "liberal_aep_gwh": d["liberal_aep_gwh"],
                     "regret_pct": regret_pct})

a_arr = np.array([d["a"] for d in data])
f_arr = np.array([d["f"] for d in data])
regret_pct = np.array([d["regret_pct"] for d in data])
regret_gwh = np.array([d["regret_gwh"] for d in data])

# --- Model 1: Multivariate  regret_pct = b0 + b1*f + b2*a + b3*a*f ---
X_multi = np.column_stack([np.ones(len(data)), f_arr, a_arr, a_arr * f_arr])
beta_multi, residuals, rank, sv = np.linalg.lstsq(X_multi, regret_pct, rcond=None)
pred_multi = X_multi @ beta_multi
ss_res = np.sum((regret_pct - pred_multi) ** 2)
ss_tot = np.sum((regret_pct - regret_pct.mean()) ** 2)
r2_multi = 1 - ss_res / ss_tot

print("=" * 60)
print("MODEL 1: regret_pct = b0 + b1*f + b2*a + b3*a*f")
print(f"  b0 = {beta_multi[0]:.4f}")
print(f"  b1 = {beta_multi[1]:.4f}  (f coefficient)")
print(f"  b2 = {beta_multi[2]:.4f}  (a coefficient)")
print(f"  b3 = {beta_multi[3]:.4f}  (a*f interaction)")
print(f"  R² = {r2_multi:.4f}")
print(f"\n  Developer formula:")
print(f"  regret(%) ≈ {beta_multi[0]:.2f} + {beta_multi[1]:.2f}·f + {beta_multi[2]:.2f}·a + {beta_multi[3]:.2f}·a·f")

# --- Model 2: Univariate  regret_pct = b0 + b1*f ---
X_uni = np.column_stack([np.ones(len(data)), f_arr])
beta_uni, _, _, _ = np.linalg.lstsq(X_uni, regret_pct, rcond=None)
pred_uni = X_uni @ beta_uni
ss_res_uni = np.sum((regret_pct - pred_uni) ** 2)
r2_uni = 1 - ss_res_uni / ss_tot

print(f"\n{'='*60}")
print("MODEL 2: regret_pct = b0 + b1*f")
print(f"  b0 = {beta_uni[0]:.4f}")
print(f"  b1 = {beta_uni[1]:.4f}")
print(f"  R² = {r2_uni:.4f}")
print(f"\n  Simple formula: regret(%) ≈ {beta_uni[0]:.2f} + {beta_uni[1]:.2f}·f")

# --- Model 3: Quadratic  regret_pct = b0 + b1*f + b2*f^2 + b3*a + b4*a^2 + b5*a*f ---
X_quad = np.column_stack([np.ones(len(data)), f_arr, f_arr**2, a_arr, a_arr**2, a_arr * f_arr])
beta_quad, _, _, _ = np.linalg.lstsq(X_quad, regret_pct, rcond=None)
pred_quad = X_quad @ beta_quad
ss_res_quad = np.sum((regret_pct - pred_quad) ** 2)
r2_quad = 1 - ss_res_quad / ss_tot

print(f"\n{'='*60}")
print("MODEL 3: regret_pct = b0 + b1*f + b2*f² + b3*a + b4*a² + b5*a*f")
print(f"  R² = {r2_quad:.4f}")

# --- Figure: Predicted vs Actual ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, pred, r2, title in [
    (axes[0], pred_uni, r2_uni, f"Univariate (f only)\n$R^2$ = {r2_uni:.3f}"),
    (axes[1], pred_multi, r2_multi, f"Linear + interaction\n$R^2$ = {r2_multi:.3f}"),
    (axes[2], pred_quad, r2_quad, f"Quadratic\n$R^2$ = {r2_quad:.3f}"),
]:
    ax.scatter(regret_pct, pred, c=f_arr, cmap="coolwarm", s=60, edgecolors="k", linewidths=0.5)
    lims = [min(regret_pct.min(), pred.min()) - 0.1,
            max(regret_pct.max(), pred.max()) + 0.1]
    ax.plot(lims, lims, "k--", alpha=0.5)
    ax.set_xlabel("Actual regret (% of AEP)")
    ax.set_ylabel("Predicted regret (% of AEP)")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

fig.suptitle("Regression Models for Design Regret vs. Wind Rose Parameters",
             fontsize=12)
plt.tight_layout()

out = Path("analysis/regret_regression.png")
fig.savefig(str(out), dpi=200, bbox_inches="tight")
print(f"\nSaved: {out}")

# --- Decision rule thresholds ---
print(f"\n{'='*60}")
print("DECISION RULE ANALYSIS")
print(f"{'='*60}")
for f_thresh in [0.2, 0.25, 0.3, 0.35]:
    mask = f_arr <= f_thresh
    if mask.sum() > 0:
        max_r = regret_pct[mask].max()
        mean_r = regret_pct[mask].mean()
        print(f"  f ≤ {f_thresh}: max regret = {max_r:.2f}%, mean = {mean_r:.2f}% ({mask.sum()} cases)")

print()
for f_thresh in [0.7, 0.75, 0.8]:
    for a_thresh in [0.6, 0.7, 0.8]:
        mask = (f_arr >= f_thresh) & (a_arr >= a_thresh)
        if mask.sum() > 0:
            min_r = regret_pct[mask].min()
            mean_r = regret_pct[mask].mean()
            print(f"  f ≥ {f_thresh} & a ≥ {a_thresh}: min regret = {min_r:.2f}%, "
                  f"mean = {mean_r:.2f}% ({mask.sum()} cases)")

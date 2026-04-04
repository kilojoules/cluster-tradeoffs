"""Plot regret saturation curves: regret vs number of placed neighbors."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

cases = [
    ("Bastankhah, unidir (270°)", "analysis/sweep_pad50D_200t/bastankhah_unidir", "tab:red", "-"),
    ("Bastankhah, uniform", "analysis/sweep_pad50D_200t/bastankhah_uniform", "tab:blue", "-"),
    ("TurboPark, unidir (270°)", "analysis/sweep_pad50D_200t/turbopark_unidir", "tab:red", "--"),
]

# Also check for TurboPark uniform if available
tp_uniform = Path("analysis/sweep_pad50D_200t/turbopark_uniform/results.json")
if tp_uniform.exists():
    cases.append(("TurboPark, uniform", "analysis/sweep_pad50D_200t/turbopark_uniform", "tab:blue", "--"))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

for label, path, color, ls in cases:
    p = Path(path) / "results.json"
    if not p.exists():
        print(f"  Skipping {label} (not found)")
        continue
    with open(p) as f:
        d = json.load(f)
    rh = d["regret_history"]
    lib_aep = d["liberal_aep_gwh"]
    steps = np.arange(1, len(rh) + 1)

    ax1.plot(steps, rh, color=color, ls=ls, linewidth=2, label=label)
    ax2.plot(steps, 100 * np.array(rh) / lib_aep, color=color, ls=ls, linewidth=2, label=label)
    print(f"  {label}: {len(rh)} steps, final regret = {rh[-1]:.1f} GWh ({100*rh[-1]/lib_aep:.2f}%)")

ax1.set_xlabel("Number of Neighbors Placed")
ax1.set_ylabel("Design Regret (GWh/yr)")
ax1.set_title("Absolute Regret Saturation")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

ax2.set_xlabel("Number of Neighbors Placed")
ax2.set_ylabel("Design Regret (% of liberal AEP)")
ax2.set_title("Relative Regret Saturation")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

fig.suptitle("Regret Saturation: How Many Neighbors Until Regret Plateaus?\n"
             "Greedy adversarial placement, K=5 multistart, 50D grid pad",
             fontsize=12)
plt.tight_layout()

out = Path("analysis/saturation_curves.png")
fig.savefig(str(out), dpi=200, bbox_inches="tight")
print(f"Saved: {out}")

"""Merge per-distance DEI cross-section results into one results.json."""

import json
import numpy as np
from pathlib import Path

distances_D = [5, 10, 15, 20, 30, 40, 60]
n_bearings = 24

# Load first to get structure
d0 = json.load(open(f"analysis/cross_section_k500v/dei_d{distances_D[0]}/results.json"))
bearings = d0["bearings_deg"]
liberal_aep = d0["liberal_aep_gwh"]

regret_grid = np.full((len(distances_D), n_bearings), np.nan)
all_evals = []

for di, dist in enumerate(distances_D):
    p = Path(f"analysis/cross_section_k500v/dei_d{dist}/results.json")
    if not p.exists():
        print(f"  MISSING: {p}")
        continue
    d = json.load(open(p))
    g = np.array(d["regret_grid_gwh"])
    # Each sub-result has shape (1, n_bearings) since it's one distance
    regret_grid[di, :] = g[0, :]
    all_evals.extend(d["evaluations"])
    print(f"  dist={dist}D: max regret = {g.max():.1f} GWh")

regret_pct = 100 * regret_grid / liberal_aep

merged = {
    "n_target": d0["n_target"],
    "ref_rows": d0["ref_rows"],
    "ref_cols": d0["ref_cols"],
    "ref_spacing_D": d0["ref_spacing_D"],
    "n_ref_turbines": d0["n_ref_turbines"],
    "liberal_aep_gwh": liberal_aep,
    "bearings_deg": bearings,
    "distances_D": distances_D,
    "regret_grid_gwh": regret_grid.tolist(),
    "regret_grid_pct": regret_pct.tolist(),
    "evaluations": all_evals,
    "config": d0["config"],
}

out = Path("analysis/cross_section_k500v/dei/results.json")
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    json.dump(merged, f, indent=2)
print(f"\nMerged: {out}")
print(f"Max regret: {np.nanmax(regret_grid):.1f} GWh ({np.nanmax(regret_pct):.2f}%)")

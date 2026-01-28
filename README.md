# Cluster Tradeoffs

Discovering worst-case wind farm cluster configurations that maximize design regret.

## Research Question

What is the maximum regret we can create for a wind farm by controlling neighboring wind farm characteristics?

## Key Finding

For certain neighbor configurations (blob geometries), choosing a "liberal" design strategy (ignoring neighbors) can cost up to **64 GWh/year** compared to a "conservative" design that accounts for potential neighbors.

## Results

### Blob 3: Highest Regret Configuration (64.1 GWh)

![Blob 3 Results](blob_discovery/blob_3.png)

**Panels (left to right):**
1. **Pareto frontier**: AEP with neighbors absent vs present. Hollow circles mark Pareto-optimal layouts.
2. **Liberal-optimal layout**: Maximizes AEP when alone (1168 GWh), but drops to 993 GWh when neighbors appear.
3. **Conservative-optimal layout**: Shifted downwind to avoid neighbor wakes. Achieves 1117 GWh alone, 1057 GWh with neighbors.
4. **Wind rose**: Single direction from west (270°).

**Regret = 1057 - 993 = 64 GWh** — the cost of choosing liberal when neighbors appear.

### Summary Across All Blob Configurations

| Blob | Regret (GWh) | Pareto Points |
|------|-------------|---------------|
| 3    | **64.1**    | 7             |
| 7    | 47.3        | 4             |
| 9    | 42.0        | 4             |
| 0    | 38.9        | 3             |
| 5    | 30.7        | 3             |
| 1    | 22.1        | 3             |
| 2    | 12.6        | 2             |
| 8    | 10.0        | 3             |
| 4    | 8.3         | 2             |
| 6    | **2.3**     | 2             |

## Methodology

### Inner Optimization (per blob configuration)

1. Run 10 multi-start SGD optimizations assuming **no neighbors** (liberal strategy)
2. Run 10 multi-start SGD optimizations assuming **neighbors present** (conservative strategy)
3. Pool all 20 layouts
4. Evaluate each layout under both scenarios (neighbors absent / present)
5. Compute Pareto frontier and regret

### Regret Definition

- **Pareto frontier**: Layouts where no other layout dominates in both AEP_absent and AEP_present
- **Liberal-optimal**: Pareto point with max AEP_absent
- **Conservative-optimal**: Pareto point with max AEP_present
- **Regret** = AEP_present(conservative-opt) - AEP_present(liberal-opt)

If regret > 0, there is a fundamental tradeoff between optimizing for isolated vs. neighbor scenarios.

## Replicating Results

### Prerequisites

```bash
# Clone the repository
git clone git@github.com:kilojoules/cluster-tradeoffs.git
cd cluster-tradeoffs

# Install dependencies with pixi
pixi install
```

### Run the Analysis

```bash
# Full analysis (10 blobs × 20 optimizations = 200 total)
pixi run python scripts/run_regret_discovery.py
```

This will:
1. Generate 10 random blob configurations
2. For each blob, run pooled multi-start optimization
3. Save results to `blob_discovery/results.json`
4. Generate plots: `blob_discovery/blob_*.png` and `blob_discovery/pareto_frontier.png`

### Configuration

Edit `scripts/run_regret_discovery.py` to modify:

```python
run_multistart_pooled_discovery(
    n_blobs=10,              # Number of blob configurations to test
    n_starts_per_strategy=10  # Random starts per strategy (liberal/conservative)
)
```

SGD settings (line ~170):
```python
SGDSettings(
    max_iter=3000,           # SGD iterations per optimization
    learning_rate=D / 5      # Learning rate (D = rotor diameter)
)
```

## Dependencies

Currently includes `pixwake` source in `src/`. Future work will disentangle this into a proper dependency.

Key modules used:
- `pixwake.optim.adversarial.PooledBlobDiscovery` — pooled multi-start optimization
- `pixwake.optim.sgd.topfarm_sgd_solve` — constrained SGD optimizer
- `pixwake.optim.geometry.BSplineBoundary` — blob geometry representation

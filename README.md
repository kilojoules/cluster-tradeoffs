# Cluster Tradeoffs

Discovering worst-case wind farm cluster configurations that maximize design regret.

## Research Question

What is the maximum regret we can create for a wind farm by controlling neighboring wind farm characteristics?

## Key Finding

For certain neighbor configurations (blob geometries), choosing a "liberal" design strategy (ignoring neighbors) can cost up to **64 GWh/year** compared to a "conservative" design that accounts for potential neighbors.

## Methodology

1. **Inner optimization (per blob config):**
   - Run N multi-start optimizations assuming no neighbors (liberal)
   - Run N multi-start optimizations assuming neighbors present (conservative)
   - Pool all layouts, evaluate under both scenarios
   - Compute Pareto frontier and regret

2. **Outer optimization:**
   - Search over blob geometry to maximize regret
   - Identify "danger zone" configurations

## Results

See `blob_discovery/` for results across 10 blob configurations:
- Max regret: 64.1 GWh (blob 3)
- Min regret: 2.3 GWh (blob 6)

## Dependencies

Currently depends on `pixwake` (included in `src/`). Will be disentangled in future work.

## Usage

```bash
pixi install
pixi run python scripts/run_regret_discovery.py
```

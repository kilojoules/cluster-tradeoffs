# Replication Guide

## Prerequisites

### System Requirements

- Python 3.11+
- ~8 GB RAM (for JAX compilation)
- ~2 GB disk space (for results)

### Installation

```bash
# Clone the repository
git clone https://github.com/kilojoules/cluster-tradeoffs.git
cd cluster-tradeoffs

# Install dependencies with pixi
curl -fsSL https://pixi.sh/install.sh | bash
pixi install
```

## Quick Start

### Run a Single Configuration

```bash
# Single direction, 5 blobs, 5 starts (fastest, ~10 minutes)
pixi run python scripts/run_regret_discovery.py \
    --wind-rose=single \
    --n-blobs=5 \
    --n-starts=5
```

### Run Wind Rose Comparison

```bash
# Compare all wind rose types (5 blobs each, ~1 hour)
pixi run python scripts/run_regret_discovery.py \
    --wind-rose=comparison \
    --n-blobs=5 \
    --n-starts=5
```

### Full Analysis (Paper Results)

```bash
# Full analysis (20 blobs, 20 starts, ~4-6 hours)
pixi run python scripts/run_regret_discovery.py \
    --wind-rose=comparison \
    --n-blobs=20 \
    --n-starts=20 \
    --output-dir=analysis/wind_rose_comparison_full
```

## Command Reference

### Main Script: `run_regret_discovery.py`

```
usage: run_regret_discovery.py [-h] [--wind-rose {single,uniform,von_mises,bimodal,comparison}]
                               [--n-directions N] [--dominant-dir DEG] [--concentration K]
                               [--secondary-dir DEG] [--mean-ws M/S] [--n-blobs N]
                               [--n-starts N] [--output-dir DIR]

Options:
  --wind-rose, -w       Wind rose type (default: single)
                        - single: unidirectional at dominant-dir
                        - uniform: omnidirectional with n-directions
                        - von_mises: Von Mises distribution
                        - bimodal: two peaks
                        - comparison: run all types

  --n-directions, -d    Number of wind directions (default: 24)
  --dominant-dir        Primary wind direction in degrees (default: 270)
  --concentration, -k   Von Mises kappa parameter (default: 2.0)
  --secondary-dir       Secondary direction for bimodal (default: 90)
  --mean-ws             Mean wind speed in m/s (default: 9.0)
  --n-blobs             Number of blob configurations (default: 10)
  --n-starts            Optimization starts per strategy (default: 10)
  --output-dir, -o      Output directory (auto-generated if not specified)
```

### Convergence Study: `run_convergence_study.py`

```bash
# Verify convergence of regret estimates
pixi run python scripts/run_convergence_study.py --n-starts-max=40
```

## Output Structure

```
analysis/
└── wind_rose_comparison_full/
    ├── comparison_summary.json      # Summary statistics
    ├── pareto_comparison_all.png    # Combined Pareto plot
    ├── wind_rose_comparison.png     # Bar chart comparison
    │
    ├── single_270deg/
    │   ├── results.json             # Full optimization results
    │   ├── wind_rose_config.json    # Wind rose parameters
    │   ├── pareto_frontier.png      # Pareto frontier for blob 0
    │   ├── blob_0.png               # Individual blob result
    │   ├── blob_1.png
    │   └── ...
    │
    ├── uniform_24dir/
    │   └── ...
    │
    └── von_mises_270deg_k2.0/
        └── ...
```

### Results JSON Format

```json
{
  "blob_seed": 3,
  "control_points": [[x1, y1], [x2, y2], ...],
  "all_layouts": [
    {
      "strategy": "liberal",
      "x": [x1, x2, ...],
      "y": [y1, y2, ...],
      "aep_absent": 1168.0,
      "aep_present": 1011.0
    },
    ...
  ],
  "global_best_aep_absent": 1168.0,
  "global_best_aep_present": 1072.0,
  "min_liberal_regret": 0.0,
  "min_conservative_regret": 0.0
}
```

## Customization

### Changing Farm Size

Edit `scripts/run_regret_discovery.py`:

```python
D = 200.0           # Rotor diameter [m]
target_size = 16*D  # Farm size [m]
min_spacing = 4*D   # Minimum turbine spacing [m]
n_target = 16       # Number of turbines
```

### Changing Turbine

Modify the `create_turbine()` function:

```python
def create_turbine(rotor_diameter=200.0):
    ws = jnp.array([0.0, 4.0, 10.0, 15.0, 25.0])
    power = jnp.array([0.0, 0.0, 10000.0, 10000.0, 0.0])  # kW
    ct = jnp.array([0.0, 0.8, 0.8, 0.4, 0.0])
    return Turbine(
        rotor_diameter=rotor_diameter,
        hub_height=120.0,
        power_curve=Curve(ws=ws, values=power),
        ct_curve=Curve(ws=ws, values=ct),
    )
```

### Adding New Wind Rose Types

Add to `scripts/run_regret_discovery.py`:

```python
def _generate_custom_rose(config):
    """Custom wind rose implementation."""
    wd = jnp.linspace(0, 360, config.n_directions, endpoint=False)
    # Custom weight calculation
    weights = ...
    ws = jnp.full_like(wd, config.mean_ws)
    return wd, ws, weights
```

## Troubleshooting

### JAX Compilation Warnings

```
UserWarning: Error reading persistent compilation cache entry
```

These are harmless cache corruption warnings. JAX will recompile as needed.

### Memory Issues

If you encounter OOM errors:

1. Reduce `n_directions` (e.g., 12 instead of 24)
2. Reduce `n_starts` (e.g., 10 instead of 20)
3. Run wind rose types sequentially instead of comparison mode

### Slow Performance

- First run is slow due to JAX compilation (~5 min)
- Subsequent runs use cached compilation
- GPU acceleration available if JAX GPU is installed

## Citation

If you use this code, please cite:

```bibtex
@software{cluster_tradeoffs,
  author = {Quick, Julian},
  title = {Wind Farm Cluster Tradeoffs},
  year = {2025},
  url = {https://github.com/kilojoules/cluster-tradeoffs}
}
```

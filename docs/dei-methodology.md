# DEI Analysis Methodology

This document specifies the analysis protocol for the Danish Energy Island (DEI) design regret study.

## Goals

1. **Quantify design regret** for each neighboring wind farm individually
2. **Compare individual vs combined** regret (all 9 neighbors present simultaneously)
3. **Test sensitivity** to wake model configuration
4. **Enable verification** by saving layouts for cross-validation with PyWake

## Analysis Protocol

### Wind Resource

- **Data source**: 10-year daily averaged wind data (2012-2021) from `OMAE_neighbors/energy_island_10y_daily_av_wind.csv`
- **Evaluation method**: **Full time series** (3653 samples)
  - NOT binned wind rose (which introduces P(E[v]) vs E[P(v)] error)
  - Binned wind rose may be used internally during optimization for speed, but all reported AEP values must come from full time series evaluation

### Optimization

- **Method**: Gradient-based (Adam) with multi-start
- **Starts**: 50 random initial layouts per strategy
- **Iterations**: 2000 per start
- **Strategies**:
  - **Liberal**: Optimize target layout ignoring neighbors
  - **Conservative**: Optimize target layout considering neighbors

### Evaluation Cases

For each wake model configuration, run:

| Case | Neighbors in Optimization | Neighbors in Evaluation | Purpose |
|------|--------------------------|------------------------|---------|
| Farm 1 | Farm 1 only | Farm 1 only | Individual regret |
| Farm 2 | Farm 2 only | Farm 2 only | Individual regret |
| ... | ... | ... | ... |
| Farm 9 | Farm 9 only | Farm 9 only | Individual regret |
| **Combined** | **All 594 turbines** | **All 594 turbines** | Ring geometry effect |

### Output Requirements

For each case, save:

1. **Layouts** (HDF5): All 100 optimized layouts (50 liberal + 50 conservative)
   - x, y coordinates
   - AEP with neighbor absent
   - AEP with neighbor present
   - Strategy label

2. **Results** (JSON): Summary statistics
   - Pareto point count
   - Regret (GWh and %)
   - Liberal-optimal and conservative-optimal AEP values

3. **Figures**: Pareto frontier plots

### PyWake Verification

Saved layouts enable verification by:
1. Loading layouts from HDF5
2. Evaluating in PyWake with matching wake model configuration
3. Comparing AEP values (should match within ~0.2%)

## Wake Model Configurations

### 1. Nygaard_2022 (PyWake Literature Default)

| Parameter | Value |
|-----------|-------|
| Model | TurboGaussianDeficit |
| A | 0.04 |
| ct2a | ct2a_mom1d |
| ctlim | 0.96 |
| superposition | SquaredSum |
| use_effective_ws | False |
| use_effective_ti | False |
| Turbulence | None |
| Ambient TI | 0.06 |

### 2. Bastankhah Gaussian

| Parameter | Value |
|-----------|-------|
| Model | BastankhahGaussianDeficit |
| k | 0.04 |
| superposition | SquaredSum |
| Turbulence | None |

### 3. OMAE TurboPark (if needed for comparison)

| Parameter | Value |
|-----------|-------|
| Model | TurboGaussianDeficit |
| A | 0.02 |
| ct2a | ct2a_mom1d |
| ctlim | 0.96 |
| superposition | LinearSum |
| use_effective_ws | True |
| use_effective_ti | True |
| Turbulence | CrespoHernandez |
| Ambient TI | 0.06 |

## Turbine Specification

All analyses use the same turbine:

| Parameter | Value |
|-----------|-------|
| Rotor diameter | 240 m |
| Hub height | 150 m |
| Rated power | 15 MW |
| Power/CT curves | PyWake GenericWindTurbine(diameter=240, hub_height=150, power_norm=15000) |

## Expected Outputs

For each wake model:
- `analysis/dei_{model}/layouts_farm{1-9}.h5` - Individual neighbor layouts
- `analysis/dei_{model}/layouts_combined.h5` - Combined case layouts
- `analysis/dei_{model}/dei_results.json` - All results
- `analysis/dei_{model}/dei_pareto_farm{1-9}.png` - Individual Pareto plots
- `analysis/dei_{model}/dei_pareto_combined.png` - Combined Pareto plot
- `analysis/dei_{model}/dei_polar_summary.png` - Regret by direction

## Checklist

- [ ] Nygaard_2022: All 9 farms + combined
- [ ] Bastankhah: All 9 farms + combined
- [ ] OMAE TurboPark: All 9 farms + combined (optional)
- [ ] PyWake verification for each model
- [ ] Documentation updated with results

# Wind Farm Cluster Tradeoffs

Exploring how neighboring wind farm configurations affect design tradeoffs through Monte Carlo sampling.

## Key Question

**How much regret can a wind farm developer face when neighboring farms are uncertain?**

When designing a wind farm layout, developers must decide whether to:
- **Liberal strategy**: Optimize assuming no neighbors will be built (maximize standalone performance)
- **Conservative strategy**: Optimize assuming neighbors will appear (sacrifice standalone performance for robustness)

**Regret** measures the cost of choosing the wrong strategy.

## Case Study: Danish Energy Island

We analyzed the real-world Danish Energy Island (DEI) cluster with 10 years of site wind data. **Key finding: A single southern neighbor causes 101 GWh regret despite off-axis position.**

| Neighbor Direction | Regret |
|-------------------|--------|
| Western (262°) - dominant wind | 0 GWh |
| **Southern (163°)** - secondary wind | **101 GWh** |
| All 9 neighbors together | 101 GWh |

This demonstrates the **"ambush effect"**: neighbors off-axis from the dominant wind can cause more regret than on-axis neighbors because the liberal layout doesn't account for them.

[Full DEI Case Study →](dei-case-study.md)

## Main Findings

### 1. Sampled Regret Can Exceed 60 GWh/year

Among the randomly sampled blob configurations, the highest regret found was **61 GWh/year** for a 16-turbine farm under single-direction wind conditions. This represents choosing the liberal strategy when neighbors appear.

![Single Direction High Regret Case](figures/single_blob3.png)
*Blob 3 under single-direction wind (270°): The liberal-optimal layout achieves 1168 GWh alone but drops to 1011 GWh with neighbors. The conservative-optimal layout achieves 1133 GWh alone and 1072 GWh with neighbors. Regret = 61 GWh.*

### 2. Wind Rose Type Dramatically Affects Regret

| Wind Rose | Max Regret (GWh) | Mean Regret (GWh) |
|-----------|------------------|-------------------|
| Single (270°) | **60.99** | 20.2 |
| Von Mises κ=1 | 35.74 | 10.3 |
| Von Mises κ=4 | 31.76 | 13.9 |
| Uniform | 25.74 | 11.9 |
| Bimodal | 19.66 | 7.4 |
| Von Mises κ=2 | **16.13** | **4.4** |

### 3. Non-Monotonic Relationship with Directional Spread

Regret doesn't simply decrease with more wind directions. There's a **sweet spot at moderate concentration (κ≈2)**:

```
Single → κ=1 → κ=4 → Uniform → Bimodal → κ=2
  61      36     32      26       20       16   (max regret, GWh)
```

**Physical interpretation:**
- **Too concentrated** (single direction): Narrow but intense wake corridor creates sharp tradeoffs
- **Too diffuse** (uniform): Neighbors affect you from all directions—no "safe" layout exists
- **Moderate** (κ≈2): Directional preference allows layout adaptation without extreme penalties

![Pareto Comparison](figures/pareto_comparison.png)
*Pareto frontiers across wind rose types. Steeper, longer frontiers indicate higher regret.*

## Methodology

### Random Blob Sampling + Pooled Multi-Start Optimization

We randomly sample 20 neighbor "blob" configurations per wind rose type. For each blob:

1. **Sample** a random blob shape (B-spline with 4 control points)
2. Run 20 multi-start SGD optimizations on **target layout** with **liberal** assumptions (ignoring neighbors)
3. Run 20 multi-start SGD optimizations on **target layout** with **conservative** assumptions (accounting for neighbors)
4. Pool all 40 target layouts
5. Evaluate each layout under both scenarios
6. Compute Pareto frontier and regret

**Note**: The blob shapes are randomly sampled, not optimized. This Monte Carlo approach explores the distribution of regret across neighbor geometries but does not find guaranteed worst-case configurations.

### Regret Definition

- **Pareto frontier**: Layouts where no other layout dominates in both AEP_absent and AEP_present
- **Liberal-optimal**: Pareto point maximizing AEP when neighbors are absent
- **Conservative-optimal**: Pareto point maximizing AEP when neighbors are present
- **Regret** = AEP_present(conservative) − AEP_present(liberal)

### Convergence Verification

Regret values stabilize by n=20 starts per strategy:

| Configuration | n=5 | n=10 | n=20 | n=40 |
|--------------|-----|------|------|------|
| Single direction | 53.70 | 38.62 | 41.15 | 38.62 |
| Uniform | 24.27 | 24.27 | 20.29 | 20.29 |
| Von Mises κ=4 | 16.69 | 17.75 | 9.77 | 9.77 |

![Convergence](figures/convergence.png)

## Setup

### Configuration

- **Target farm**: 16 turbines in 16D × 16D area (D = 200m rotor diameter)
- **Minimum spacing**: 4D between turbines
- **Neighbor representation**: Randomly sampled "blob" shapes using B-spline boundaries (20 samples per wind rose type)
- **Neighbor grid**: 25 potential turbine positions on a 5×5 grid, masked by blob boundary
- **Wake model**: Bastankhah Gaussian deficit (k=0.04)
- **Turbine**: 10 MW class (200m rotor, 120m hub height)

### Wind Rose Types

| Type | Description |
|------|-------------|
| **Single** | Unidirectional, 270° (West) |
| **Uniform** | 24 directions, equal probability |
| **Von Mises** | Circular normal distribution centered at 270° |
| **Bimodal** | Two peaks at 270° (70%) and 90° (30%) |

The Von Mises concentration parameter κ controls spread:
- κ = 0: Uniform
- κ = 2: Moderate (typical offshore)
- κ → ∞: Single direction

## Replication

### Prerequisites

```bash
git clone https://github.com/kilojoules/cluster-tradeoffs.git
cd cluster-tradeoffs
pixi install
```

### Run Full Analysis

```bash
# Full wind rose comparison (20 blobs × 20 starts × 6 types ≈ 4-6 hours)
pixi run python scripts/run_regret_discovery.py \
    --wind-rose=comparison \
    --n-blobs=20 \
    --n-starts=20

# Single wind rose type
pixi run python scripts/run_regret_discovery.py \
    --wind-rose=von_mises \
    --concentration=2.0 \
    --n-blobs=10 \
    --n-starts=20

# Convergence study
pixi run python scripts/run_convergence_study.py
```

### Command-Line Options

```
--wind-rose, -w     Type: single, uniform, von_mises, bimodal, comparison
--n-directions, -d  Number of wind directions (default: 24)
--dominant-dir      Dominant direction in degrees (default: 270)
--concentration, -k Von Mises kappa parameter (default: 2.0)
--n-blobs           Number of blob configurations (default: 10)
--n-starts          Optimization starts per strategy (default: 10)
--output-dir, -o    Output directory
```

## Results by Wind Rose Type

Detailed analysis for each wind rose configuration:

- [Single Direction (270°)](results/#single-direction-270)
- [Von Mises κ=1 (Broad Spread)](results/#von-mises-1-broad-spread)
- [Von Mises κ=2 (Optimal)](results/#von-mises-2-optimal)
- [Von Mises κ=4 (Concentrated)](results/#von-mises-4-concentrated)
- [Uniform Distribution](results/#uniform-distribution)
- [Bimodal Distribution](results/#bimodal-distribution)

## Real-World Case Study

- [Danish Energy Island (DEI)](dei-case-study.md) - Analysis of the 10-farm North Sea cluster with actual wind data

## References

- Wake model: Bastankhah & Porté-Agel (2014)
- Optimization: JAX-based gradient descent with soft boundary constraints
- Wind rose statistics: Von Mises distribution for circular data

---

*Generated with [pixwake](https://github.com/kilojoules/pixwake) - JAX-based wind farm simulation*

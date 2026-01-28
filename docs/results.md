# Results

## Summary Table

Full analysis: 20 blobs × 20 starts per strategy × 6 wind rose types = **4,800 optimizations**

| Wind Rose | Max Regret (GWh) | Mean Regret (GWh) | Blobs with Tradeoff |
|-----------|------------------|-------------------|---------------------|
| Single (270°) | **60.99** | 20.23 | 18/20 |
| Von Mises κ=1 | 35.74 | 10.26 | 13/20 |
| Von Mises κ=4 | 31.76 | 13.92 | 18/20 |
| Uniform | 25.74 | 11.93 | 17/20 |
| Bimodal | 19.66 | 7.43 | 10/20 |
| Von Mises κ=2 | **16.13** | **4.37** | 10/20 |

## Wind Rose Types Analyzed

![Wind Rose Comparison](figures/wind_rose_comparison.png)

Six wind rose types were analyzed, ranging from highly concentrated (single direction) to fully uniform.

## Pareto Frontiers by Wind Rose Type

![Pareto Comparison](figures/pareto_comparison.png)

Each subplot shows Pareto frontiers for blob configurations with regret > 5 GWh. Key observations:

- **Single direction**: Wide spread, steep frontiers → high regret
- **Von Mises κ=2**: Tight clustering → low regret
- **Uniform**: Moderate spread despite omnidirectionality

## Detailed Analysis by Wind Rose

### Single Direction (270°)

**Maximum regret: 60.99 GWh** (Blob 3)

![Single Pareto Frontier](figures/single_pareto.png)

![Single Blob 3](figures/single_blob3.png)

The single-direction case shows the most dramatic tradeoffs:

- Liberal-optimal layout packs turbines efficiently for standalone operation
- Conservative-optimal layout shifts turbines to avoid the wake corridor
- Difference in AEP_present: 1072 - 1011 = 61 GWh

**Distribution of regret across blobs:**

| Regret Range | Count |
|--------------|-------|
| 0 GWh | 2 |
| 1-10 GWh | 5 |
| 10-30 GWh | 6 |
| 30-50 GWh | 4 |
| 50+ GWh | 3 |

### Von Mises κ=1 (Broad Spread)

**Maximum regret: 35.74 GWh** (Blob 3)

![Von Mises κ=1 Pareto Frontier](figures/von_mises_k1_pareto.png)

![Von Mises κ=1 Blob 3](figures/von_mises_k1_blob3.png)

The broad spread of κ=1 provides intermediate results:

- **Moderate directional preference**: Layouts can still adapt to the dominant direction
- **Significant spreading**: Wake effects are somewhat averaged
- **Intermediate regret**: Falls between single-direction and more uniform cases

### Von Mises κ=2 (Optimal)

**Maximum regret: 16.13 GWh** (Blob 18)

![Von Mises κ=2 Pareto Frontier](figures/von_mises_k2_pareto.png)

![Von Mises κ=2 Blob 18](figures/von_mises_k2_blob18.png)

This configuration minimizes regret because:

1. **Directional preference exists**: Layouts can adapt to the dominant direction
2. **Spread is sufficient**: Wakes are partially "smeared" across directions
3. **No extreme penalties**: Neither scenario dominates

**Distribution of regret:**

| Regret Range | Count |
|--------------|-------|
| 0 GWh | 10 |
| 1-10 GWh | 6 |
| 10-20 GWh | 4 |

### Von Mises κ=4 (Concentrated)

**Maximum regret: 31.76 GWh** (Blob 5)

![Von Mises κ=4 Pareto Frontier](figures/von_mises_k4_pareto.png)

![Von Mises κ=4 Blob 5](figures/von_mises_k4_blob5.png)

The more concentrated κ=4 case shows:

- **Strong directional preference**: Similar to single-direction but with some spreading
- **Higher regret than κ=2**: Concentration increases vulnerability to neighbor placement
- **Sharp wake corridors**: Still has well-defined danger zones

### Uniform Distribution

**Maximum regret: 25.74 GWh** (Blob 3)

![Uniform Pareto Frontier](figures/uniform_pareto.png)

![Uniform Blob 3](figures/uniform_blob3.png)

Surprisingly, uniform wind doesn't minimize regret:

- Neighbors affect the target farm from **all** directions
- No layout can be "safe" from all possible wake angles
- Tradeoff: optimize for average vs. worst-case directions

### Bimodal Distribution

**Maximum regret: 19.66 GWh** (Blob 5)

![Bimodal Pareto Frontier](figures/bimodal_pareto.png)

![Bimodal Blob 5](figures/bimodal_blob5.png)

Two dominant directions (270° and 90°) create:

- Two separate "danger zones" for neighbor placement
- Intermediate regret between single-direction and uniform
- Layout must balance exposure from both directions

## Impact of Blob Configuration

While wind rose type determines the overall magnitude of regret, the **blob configuration** (neighbor position, size, and shape) determines which specific scenarios produce high or low regret.

### Per-Blob Regret Across Wind Rose Types

| Blob | Single | Uniform | κ=1 | κ=2 | κ=4 | Bimodal | Avg |
|-----:|-------:|--------:|----:|----:|----:|--------:|----:|
| 3 | **61.0** | 25.7 | **35.7** | 11.2 | 31.6 | 16.3 | **30.3** |
| 7 | 59.0 | 25.7 | 17.4 | 13.3 | 18.4 | 18.8 | 25.4 |
| 18 | 40.6 | 24.0 | 25.8 | **16.1** | 27.3 | 17.9 | 25.3 |
| 5 | 31.2 | 18.9 | 14.0 | 10.2 | **31.8** | **19.7** | 21.0 |
| 17 | 41.6 | 22.2 | 25.8 | 0.7 | 19.0 | 14.5 | 20.6 |
| 9 | 47.6 | 15.8 | 19.7 | 6.6 | 14.1 | 14.3 | 19.7 |
| 4 | 0.0 | 1.9 | 0.0 | 0.7 | 9.2 | 0.0 | 2.0 |
| 11 | 0.0 | 6.9 | 2.7 | 0.0 | 6.7 | 0.0 | 2.7 |

*Selected blobs showing highest and lowest average regret. Bold indicates maximum for that wind rose.*

**Key observations:**
- **Blob 3** produces the highest regret for 3 of 6 wind rose types
- **Blob 4 and 11** consistently produce near-zero regret
- Some blobs (e.g., Blob 5) produce high regret only for specific wind roses

### What Makes a Blob High-Regret?

Analysis of blob characteristics reveals **size is the dominant factor**:

| Characteristic | Correlation with Regret |
|----------------|------------------------|
| Size (max radius) | **r = +0.80** |
| Y extent | r = +0.72 |
| X extent | r = +0.59 |
| Eccentricity | r = +0.45 |
| Centroid X | r = +0.21 |
| Centroid Y | r = -0.17 |

**High-regret blobs** (>30 GWh, n=6):
- Average size: 2,571 m (12.9D)
- Average eccentricity: 1.44

**Low-regret blobs** (<10 GWh, n=8):
- Average size: 1,636 m (8.2D)
- Average eccentricity: 1.28

### Physical Explanation

Larger blobs create more design tradeoff because:

1. **Greater wake coverage**: A large neighbor can shadow a larger portion of the target farm area
2. **Fewer escape routes**: The conservative strategy has less room to shift turbines away from wakes
3. **Amplified divergence**: Liberal and conservative optimal layouts must differ more dramatically

Blob position (centroid) has weak correlation because:
- All blobs are sampled from the upwind region
- Wake effects decay with distance, so very distant blobs have low impact regardless of size
- The critical factor is how much of the target area the blob can shadow, not where its center is

## Physical Interpretation

### Why Single Direction Has Highest Regret

With wind always from 270° (West):

1. **Wake alignment is deterministic**: A neighbor directly upwind creates maximum losses
2. **Sharp danger zone**: Critical positions form a narrow wedge
3. **Layouts diverge**: Liberal packs tight; conservative shifts east

### Why κ=2 Minimizes Regret

Moderate concentration balances two effects:

1. **Enough directionality**: Layouts can adapt to the primary wind
2. **Enough spread**: Wake effects are partially averaged out

### Why Uniform Doesn't Minimize Regret

With equal probability from all directions:

1. **No escape**: Neighbors affect you regardless of their position
2. **Conflicting objectives**: Can't optimize for all directions simultaneously
3. **Averaging penalty**: Must compromise across all scenarios

## Key Insights

!!! success "Main Finding"
    Moderate wind rose concentration (κ≈2) minimizes design regret by balancing directional preference with wake spreading.

!!! warning "Single-Direction Risk"
    Sites with highly directional wind resources face up to 4× higher regret than sites with moderate spread.

!!! info "Design Recommendation"
    For uncertain neighbor scenarios, conservative designs are most valuable at sites with concentrated wind roses.

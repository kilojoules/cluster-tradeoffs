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

## Pareto Frontiers by Wind Rose Type

![Pareto Comparison](figures/pareto_comparison.png)

Each subplot shows Pareto frontiers for blob configurations with regret > 5 GWh. Key observations:

- **Single direction**: Wide spread, steep frontiers → high regret
- **Von Mises κ=2**: Tight clustering → low regret
- **Uniform**: Moderate spread despite omnidirectionality

## Detailed Analysis by Wind Rose

### Single Direction (270°)

**Maximum regret: 60.99 GWh** (Blob 3)

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

### Von Mises κ=2 (Optimal)

**Maximum regret: 16.13 GWh** (Blob 18)

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

### Uniform Distribution

**Maximum regret: 25.74 GWh** (Blobs 3, 7)

Surprisingly, uniform wind doesn't minimize regret:

- Neighbors affect the target farm from **all** directions
- No layout can be "safe" from all possible wake angles
- Tradeoff: optimize for average vs. worst-case directions

### Bimodal Distribution

**Maximum regret: 19.66 GWh** (Blob 5)

Two dominant directions (270° and 90°) create:

- Two separate "danger zones" for neighbor placement
- Intermediate regret between single-direction and uniform
- Layout must balance exposure from both directions

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

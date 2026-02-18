# Cluster Tradeoffs — Development Guide

## Project Overview

JAX-based wind farm layout optimization studying "design regret" — the AEP (Annual Energy Production) loss from designing a wind farm ignoring potential neighbors vs. accounting for them. The core question: *what neighbor configurations maximize this regret?*

## Environment

```bash
pixi install          # install all dependencies
pixi run python ...   # run scripts in the project environment
```

**Critical:** Always set `jax.config.update("jax_enable_x64", True)` BEFORE importing pixwake. The package `__init__.py` sets x64 to False, but float64 is required for finite-difference gradient accuracy in the IFT backward pass.

## Architecture

### Source: `src/pixwake/`

- **`core.py`** — `Curve`, `Turbine`, `WakeSimulation`, fixed-point wake iteration with `custom_vjp`
- **`deficit/`** — Wake deficit models (BastankhahGaussian, TurboGaussian, NOJ)
- **`optim/sgd.py`** — `topfarm_sgd_solve` (constrained SGD), `sgd_solve_implicit` (SGD with IFT `custom_vjp` for bilevel optimization)
- **`optim/adversarial.py`** — `GradientAdversarialSearch` (IFT bilevel), `BlobAdversarialDiscovery` (FD bilevel over blob shapes), `PooledBlobDiscovery` (multi-start pooling), `CMAAdversarialSearch` (CMA-ES outer loop)
- **`optim/geometry.py`** — `BSplineBoundary`, `phi_to_control_points`, `sample_random_blob`
- **`optim/soft_packing.py`** — `create_reference_grid`, soft neighbor farm representation
- **`definitions/v80.py`** — Vestas V80 turbine definition (good for tests)

### Scripts: `scripts/`

- `run_regret_discovery.py` — Main analysis: pooled multi-start across random blobs and wind roses
- `run_dei_single_neighbor.py` — Danish Energy Island case study
- `run_convergence_study.py` — Convergence verification

### Results: `blob_discovery/`

- `results.json` — Precomputed results from pooled multi-start methodology
- `blob_*.png` — Per-blob Pareto/layout plots

## Current Task: Get IFT Bilevel Search Running End-to-End

### Background

`sgd_solve_implicit` (`sgd.py:537`) implements IFT-based implicit differentiation through the inner SGD optimizer via JAX `custom_vjp`. `GradientAdversarialSearch` (`adversarial.py:87`) uses this to do gradient ascent on neighbor positions to maximize regret. **Neither has ever been run or tested.** All existing results come from `PooledBlobDiscovery` (random blobs + multi-start) and `CMAAdversarialSearch` (CMA-ES).

The goal: run `GradientAdversarialSearch` end-to-end, validate gradients, and show it finds higher-regret configurations than derivative-free methods.

### Step 1: Fix FD epsilon in IFT backward pass

**File:** `src/pixwake/optim/sgd.py`, `_sgd_solve_implicit_bwd` function

The cross-derivative Jacobian computation (line ~687) uses `eps = 1e-5`. Neighbor positions are ~1000m scale, so a 0.01mm perturbation produces no measurable wake change. The gradient will be floating-point noise.

**Fix:** Replace `eps = 1e-5` with adaptive epsilon:
```python
eps_base = 1e-4
# Scale per-parameter: eps_i = eps_base * max(1.0, |param_i|)
```

This gives ~0.1m perturbations for positions at ~1000m — large enough for the wake sim to respond, small enough for accurate central differences in float64.

### Step 2: Replace linear solver with CG + Tikhonov damping

**File:** `src/pixwake/optim/sgd.py`, `solve_linear_system` inside `_sgd_solve_implicit_bwd` (lines ~739-769)

Current: damped fixed-point iteration with hardcoded `damping=0.1`, `max_iter=50`. May not converge for ill-conditioned Hessians.

**Fix:** Replace with Conjugate Gradient solver. Add Tikhonov regularization (`damping=0.01`) to handle near-singular Hessians. Use `jax.lax.while_loop` for JIT compatibility. Increase max_iter to 100.

### Step 3: Create gradient verification tests

**New file:** `tests/test_ift_gradients.py`

Four tests:

1. **Quadratic IFT**: `min_x (x-p)^2` — verify `custom_vjp` plumbing returns correct dx*/dp
2. **Inner solver convergence**: 4 V80 turbines, NOJDeficit, single wind dir — verify `topfarm_sgd_solve` converges
3. **Wake sim IFT gradient**: 4 targets + 1 neighbor — compare `jax.grad` through `sgd_solve_implicit` vs outer FD (1m perturbation). Allow generous tolerance (~50% relative error is OK for validating direction/sign)
4. **GradientAdversarialSearch smoke test**: 4 targets + 2 neighbors, 10 outer iterations — verify regret increases, no NaN

Use V80 turbine (`definitions/v80.py`) with `NOJDeficit(k=0.1)` for speed. Always set x64 before imports.

### Step 4: Add ADAM + diagnostics to outer loop

**File:** `src/pixwake/optim/adversarial.py`, `GradientAdversarialSearch.search()` (lines ~260-286)

Current: vanilla SGD with fixed learning rate. Add:
- ADAM momentum (beta1=0.9, beta2=0.999) for smoother convergence
- Gradient norm logging (when verbose=True)
- NaN guard: break early if gradient is NaN, return best result so far

### Step 5: End-to-end script + comparison

**New file:** `scripts/run_bilevel_ift.py`

Setup matching `run_regret_discovery.py`:
- 16 target turbines in 16D×16D area, D=200m, 4D min spacing
- BastankhahGaussianDeficit(k=0.04), single wind direction 270°, 9 m/s
- 4 initial neighbors placed upwind (-3D to -6D in x)
- Run `GradientAdversarialSearch` with ~50 outer iterations
- Save JSON results (same schema as `blob_discovery/results.json`)
- Plot convergence curve and layout comparison

**New file:** `scripts/compare_bilevel_methods.py`

Load IFT results + `blob_discovery/results.json`, produce comparison bar chart and layout overlay.

### Key Technical Details

**How `sgd_solve_implicit` works:**
- Forward: runs `topfarm_sgd_solve` normally, stores `(opt_x, opt_y, params)` as residuals
- Backward (IFT): At optimality, `grad L(x*, params) = 0`. Differentiating w.r.t. params gives `H @ dx*/dp + d²L/dxdp = 0`. Solves `H @ v = g` (adjoint), then computes `grad_params = v^T @ (d²L/dxdp)`
- Cross-derivatives `d²L/dxdp` computed via **finite differences** of `jax.grad` (avoids nested custom_vjp issues with wake sim's fixed-point iteration)
- HVP `H@v` also via finite differences of `jax.grad`
- The `vmap(compute_jac_col)` at line ~706 parallelizes the per-parameter FD columns

**Nested custom_vjp safety:** The backward pass calls `jax.grad(total_obj)` which goes through `WakeSimulation`'s `custom_vjp` on `fixed_point`. This is first-order AD through a custom_vjp, which JAX handles correctly. The FD approach specifically avoids second-order AD (no `custom_jvp` needed).

**Why the sign is `-grad_params` (line ~782):** IFT gives `dx*/dp = -H^{-1} @ d²L/dxdp`. The VJP needs `g^T @ dx*/dp`. We solve `H @ v = g`, then `g^T @ dx*/dp = -v^T @ d²L/dxdp = -grad_params`. Correct.

**`GradientAdversarialSearch.search()` flow:**
1. Compute liberal layout (no neighbors) via `topfarm_sgd_solve` — fixed baseline
2. Define `compute_regret(neighbor_params)`: calls `sgd_solve_implicit` to get conservative layout, computes `liberal_aep - conservative_aep`
3. `value_and_grad(compute_regret)` gives regret + gradient w.r.t. neighbor positions
4. Gradient ascent on neighbor positions (maximize regret)
5. Optional: clip neighbor positions to boundary box (post-step projection — does not affect gradients)

### Expected Failure Modes

1. **FD epsilon too small** → gradient is noise → outer loop wanders randomly. **Fix in Step 1.**
2. **Linear solver doesn't converge** → wrong gradient direction. **Fix in Step 2.** Diagnose via `jax.debug.print` of residual norms.
3. **Inner SGD not at true optimum** → IFT assumption `grad=0` violated → inaccurate implicit gradient. **Fix:** use tighter tolerance (`tol=1e-8`) and more iterations (`max_iter=3000`) for inner SGD when used with IFT.
4. **NaN from division by zero** in HVP when `v_norm ≈ 0`. Already handled by `eps_scaled = eps * max(1.0, v_norm)`.

### Run Commands

```bash
# Install
pixi install

# Run tests
pixi run python -m pytest tests/test_ift_gradients.py -v

# Run bilevel IFT search
pixi run python scripts/run_bilevel_ift.py

# Generate comparison plots
pixi run python scripts/compare_bilevel_methods.py

# Animated prototype (generates MP4s)
pixi run python scripts/prototype_bilevel_animated.py
pixi run python scripts/animate_inner_loop.py
```

## Gradient Flow, Multistarts, and Performance

> Full cost tradeoff analysis with figures: [`docs/cost_tradeoffs.md`](docs/cost_tradeoffs.md)
> Reproduce benchmarks: `pixi run python scripts/explore_cost_tradeoffs.py`

### How Gradients Flow

The bilevel problem has two nested levels of optimization:

```
OUTER: max_{neighbors}  regret(neighbors)
         where regret = AEP_liberal - AEP_conservative(neighbors)

INNER: AEP_conservative(neighbors) = AEP(x*(neighbors), neighbors)
         where x* = argmin_x  -AEP(x, neighbors)   [SGD with constraints]
```

For one call to `value_and_grad(compute_regret)(neighbor_params)`:

**Forward pass:**
1. `sgd_solve_implicit` runs `topfarm_sgd_solve` (up to 3000 SGD steps via `jax.lax.while_loop`)
2. Each SGD step evaluates `jax.grad(objective)` and `jax.grad(penalty)` inside the compiled XLA loop
3. Returns `(opt_x, opt_y)` and stores residuals for backward pass
4. Evaluates `AEP(opt_x, opt_y, neighbors)` at the converged point

**Backward pass (IFT via `_sgd_solve_implicit_bwd`):**

The backward pass does NOT re-run the inner SGD. It evaluates derivatives only at the converged point `(opt_x, opt_y)`:

1. **Cross-derivative Jacobian** `d^2L/d(x,y)d(params)` via finite differences:
   - `2 x n_params` calls to `jax.grad(total_obj)` (central differences, parallelized via `vmap`)
   - Each grad goes through the wake sim's `fixed_point` custom_vjp

2. **CG solver** for `(H + lambda*I) @ v = g` (adjoint equation):
   - Up to 100 CG iterations (via `while_loop`)
   - Each iteration: 1 HVP via finite differences = 2 grad evaluations

3. **Matrix-vector product** `v^T @ Jacobian` (no additional sim calls)

### Measured Cost (4 targets, 2 neighbors, V80+NOJ, 1 wind dir)

| Component | Measured | Share |
|-----------|---------|-------|
| Forward pass (SGD + AEP eval) | 0.55 s | 36% |
| Backward pass (Jacobian FD + CG) | 0.99 s | 64% |
| **Total per outer step** | **1.54 s** | 100% |

**The backward pass dominates.** The forward SGD is fast because `jax.lax.while_loop` compiles the entire loop into a single XLA program (~357x faster than naive Python-level iteration). The backward pass is more expensive because each Jacobian FD column and each CG HVP triggers individual `jax.grad` calls through the wake sim.

### Nested custom_vjp Safety

The gradient chain passes through two custom_vjp boundaries:

```
jax.grad(regret)
  -> sgd_solve_implicit.custom_vjp     [IFT: solves adjoint at optimum]
    -> jax.grad(total_obj)             [1st-order AD, called by FD in backward]
      -> WakeSimulation.__call__
        -> fixed_point.custom_vjp      [adjoint fixed-point iteration]
```

This is safe: each level only requires first-order AD through the level below. No second-order AD through custom_vjp occurs. The FD approach in `_sgd_solve_implicit_bwd` specifically avoids this by using finite differences where `jax.jacfwd` or `jax.jacrev` would create second-order AD.

### XLA Compilation and Performance

`topfarm_sgd_solve` uses `jax.lax.while_loop`, which XLA compiles into a fused loop. This gives ~357x speedup over naive Python-level iteration (0.44s measured vs 195s estimated for 500 inner steps). There is no need for an explicit `@jax.jit` wrapper on the forward pass.

**Remaining speedup opportunities:**
- **Warm-starting:** Initialize inner SGD from previous outer iteration's solution. Reduces inner iterations needed.
- **Reduce inner iterations:** For this small problem, the inner SGD converges in <50 iterations. All `max_iter` values from 50 to 3000 produce identical regret and gradient (cosine similarity = 1.0).
- **Neighbor scaling is mild:** Going from 2 to 12 parameters increases backward cost by only ~16% (0.92s to 1.07s), because `vmap` parallelizes the Jacobian FD.

### The Multistart Problem

IFT assumes a unique, smooth mapping `params -> x*(params)`. With K random starts, the inner optimization finds K local optima, and the true regret uses the best:

```
true_regret(p) = AEP_liberal - max_k AEP(x*_k(p), p)
```

The `max` over starts is non-differentiable at start-switching boundaries.

### Multistart Strategies

See [`docs/cost_tradeoffs.md`](docs/cost_tradeoffs.md) for full theory (including the envelope theorem) and projected costs. Summary:

**1. Stochastic single-start:** Randomize inner initial layout each outer iteration. Gradient is unbiased in expectation. Cost: 1x per step, needs ~2.5x more iterations.

**2. LogSumExp soft selection:** Run K starts, aggregate via `(1/tau) * logsumexp(tau * regrets_k)`. Fully differentiable. Cost: Kx per step.

**3. Envelope theorem (recommended):** Run K starts forward-only, IFT backward through winner only. The envelope theorem guarantees correctness when the winning start is locally stable:
```
d/dp max_k regret_k(p) = d/dp regret_{k*}(p)
```
Cost: K x forward + 1 x backward. At K=5: 3.1 min for 50 outer iterations (only 2.4x more than single-start).

**4. CMA-ES (derivative-free baseline):** No gradients needed, but 100 gens x 10 pop x K starts = 45 min at K=5.

### Projected Costs (50 outer iterations)

| Strategy | K=1 | K=5 | K=10 | K=20 |
|----------|-----|-----|------|------|
| Single-start IFT | 1.3 min | - | - | - |
| Envelope theorem | 1.3 min | 3.1 min | 5.4 min | 9.9 min |
| LogSumExp | 1.3 min | 6.4 min | 12.8 min | 25.6 min |
| CMA-ES | 9.1 min | 45.5 min | 91.1 min | 182.1 min |

### Recommended Path Forward

**Phase 1 (current):** Single-start IFT with fixed initial layout. Validates the gradient machinery.

**Phase 2:** Stochastic single-start. Randomize `init_x, init_y` each outer iteration.

**Phase 3:** Envelope theorem with K=5. Best cost/robustness tradeoff for production use.

**Phase 4 (if needed):** LogSumExp for fully differentiable multi-start.

"""Debug IFT gradient: compare direct, IFT, and FD gradients."""
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from pathlib import Path

from pixwake import Curve, Turbine, WakeSimulation
from pixwake.deficit import BastankhahGaussianDeficit
from pixwake.optim.sgd import SGDSettings, sgd_solve_implicit, topfarm_sgd_solve

# ── DEI setup (from convergence study) ───────────────────────────────
D = 240.0

def load_target_boundary():
    from scipy.spatial import ConvexHull
    raw = np.array([
        706694.3923283464, 6224158.532895836,
        703972.0844905999, 6226906.597455995,
        702624.6334635273, 6253853.5386425415,
        712771.6248419734, 6257704.934445341,
        715639.3355871611, 6260664.6846508905,
        721593.2420745814, 6257906.998015941,
    ]).reshape((-1, 2))
    hull = ConvexHull(raw)
    return raw[hull.vertices]

def create_dei_turbine():
    ws = jnp.array([0.,1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.,
                    13.,14.,15.,16.,17.,18.,19.,20.,21.,22.,23.,24.,25.])
    power = jnp.array([
        0., 0., 2.3986, 209.2581, 689.1977, 1480.6085,
        2661.2377, 4308.9290, 6501.0566, 9260.5163, 12081.4039, 13937.2966,
        14705.0160, 14931.0392, 14985.2085, 14996.9062, 14999.3433, 14999.8550,
        14999.9662, 14999.9916, 14999.9978, 14999.9994, 14999.9998, 14999.9999,
        15000.0000, 15000.0000,
    ])
    ct = jnp.array([
        0.8889, 0.8889, 0.8889, 0.8003, 0.8000, 0.8000,
        0.8000, 0.8000, 0.7999, 0.7930, 0.7354, 0.6100,
        0.4764, 0.3698, 0.2915, 0.2341, 0.1910, 0.1581,
        0.1325, 0.1122, 0.0958, 0.0826, 0.0717, 0.0626,
        0.0550, 0.0486,
    ])
    return Turbine(
        rotor_diameter=D, hub_height=150.,
        power_curve=Curve(ws=ws, values=power),
        ct_curve=Curve(ws=ws, values=ct),
    )

def load_wind_data():
    import pandas as pd
    csv_path = Path(__file__).parent.parent / "energy_island_10y_daily_av_wind.csv"
    df = pd.read_csv(csv_path, sep=';')
    wd_ts, ws_ts = df['WD_150'].values, df['WS_150'].values
    n_bins = 24
    bin_edges = np.linspace(0, 360, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    weights_arr = np.zeros(n_bins)
    mean_speeds = np.zeros(n_bins)
    for i in range(n_bins):
        mask = ((wd_ts >= bin_edges[i]) | (wd_ts < bin_edges[0])) if i == n_bins - 1 \
            else ((wd_ts >= bin_edges[i]) & (wd_ts < bin_edges[i + 1]))
        weights_arr[i] = mask.sum()
        mean_speeds[i] = ws_ts[mask].mean() if mask.sum() > 0 else ws_ts.mean()
    weights_arr /= weights_arr.sum()
    return jnp.array(bin_centers), jnp.array(mean_speeds), jnp.array(weights_arr)

def generate_initial_layout(boundary, n_turbines, seed=42):
    from matplotlib.path import Path as MplPath
    poly_path = MplPath(boundary)
    x_min, x_max = boundary[:, 0].min(), boundary[:, 0].max()
    y_min, y_max = boundary[:, 1].min(), boundary[:, 1].max()
    rng = np.random.default_rng(seed)
    pts = []
    while len(pts) < n_turbines:
        cands = rng.uniform([x_min, y_min], [x_max, y_max], size=(n_turbines * 5, 2))
        pts.extend(cands[poly_path.contains_points(cands)].tolist())
    pts = np.array(pts[:n_turbines])
    return jnp.array(pts[:, 0]), jnp.array(pts[:, 1])

def place_initial_neighbors(boundary, n_neighbors, seed=123):
    cy = boundary[:, 1].mean()
    x_offset = boundary[:, 0].min() - 5 * D
    rng = np.random.default_rng(seed)
    y_spread = (boundary[:, 1].max() - boundary[:, 1].min()) * 0.5
    nb_x = np.full(n_neighbors, x_offset) + rng.uniform(-2 * D, 0, n_neighbors)
    nb_y = cy + rng.uniform(-y_spread, y_spread, n_neighbors)
    return jnp.array(nb_x), jnp.array(nb_y)

# ── Setup ─────────────────────────────────────────────────────────────
N_TARGET = 4
N_NEIGHBOR = 2
min_spacing = 4.0 * D
seed = 42

boundary_np = load_target_boundary()
boundary = jnp.array(boundary_np)
turbine = create_dei_turbine()
wd, ws, weights = load_wind_data()
sim = WakeSimulation(turbine, BastankhahGaussianDeficit(k=0.04))

init_x, init_y = generate_initial_layout(boundary_np, N_TARGET, seed=seed)
init_nb_x, init_nb_y = place_initial_neighbors(boundary_np, N_NEIGHBOR, seed=seed+1)
nb_params = jnp.concatenate([init_nb_x, init_nb_y])

print(f"Targets: {N_TARGET}, Neighbors: {N_NEIGHBOR}")
print(f"Neighbor positions: x={np.array(init_nb_x)}, y={np.array(init_nb_y)}")
print(f"Min target-neighbor distance: {float(jnp.min(jnp.sqrt((init_x[:,None]-init_nb_x[None,:])**2 + (init_y[:,None]-init_nb_y[None,:])**2))):.0f}m = {float(jnp.min(jnp.sqrt((init_x[:,None]-init_nb_x[None,:])**2 + (init_y[:,None]-init_nb_y[None,:])**2)))/D:.1f}D")

def objective_with_neighbors(x, y, p):
    n_nb = p.shape[0] // 2
    nb_x, nb_y = p[:n_nb], p[n_nb:]
    x_all = jnp.concatenate([x, nb_x])
    y_all = jnp.concatenate([y, nb_y])
    result = sim(x_all, y_all, ws_amb=ws, wd_amb=wd)
    power = result.power()[:, :N_TARGET]
    return -jnp.sum(power * weights[:, None]) * 8760 / 1e6

settings = SGDSettings(learning_rate=50.0, max_iter=500, tol=1e-10)

# Liberal layout
def liberal_obj(x, y):
    result = sim(x, y, ws_amb=ws, wd_amb=wd)
    power = result.power()[:, :N_TARGET]
    return -jnp.sum(power * weights[:, None]) * 8760 / 1e6

liberal_x, liberal_y = topfarm_sgd_solve(liberal_obj, init_x, init_y, boundary, min_spacing, settings)
liberal_aep = float(-liberal_obj(liberal_x, liberal_y))
print(f"\nLiberal AEP: {liberal_aep:.4f} GWh")

# Conservative solve
def cons_obj(x, y):
    return objective_with_neighbors(x, y, nb_params)
cons_x, cons_y = topfarm_sgd_solve(cons_obj, liberal_x, liberal_y, boundary, min_spacing, settings)
cons_aep_val = float(-cons_obj(cons_x, cons_y))
# Liberal layout (optimized in isolation) evaluated WITH neighbors
x_lib_all = jnp.concatenate([liberal_x, init_nb_x])
y_lib_all = jnp.concatenate([liberal_y, init_nb_y])
result_lib = sim(x_lib_all, y_lib_all, ws_amb=ws, wd_amb=wd)
power_lib = result_lib.power()[:, :N_TARGET]
liberal_aep_present = float(jnp.sum(power_lib * weights[:, None]) * 8760 / 1e6)
regret = cons_aep_val - liberal_aep_present
print(f"Conservative AEP: {cons_aep_val:.4f} GWh")
print(f"Liberal AEP (with neighbors): {liberal_aep_present:.4f} GWh")
print(f"Regret: {regret:.6f} GWh")

# ── 1. Direct gradient: d(regret)/d(neighbors) holding targets fixed ─
print("\n" + "=" * 60)
print("1. DIRECT GRADIENT (targets fixed at conservative optimum)")
print("=" * 60)
def direct_conservative_aep(p):
    n_nb = p.shape[0] // 2
    nb_x, nb_y = p[:n_nb], p[n_nb:]
    x_all = jnp.concatenate([cons_x, nb_x])
    y_all = jnp.concatenate([cons_y, nb_y])
    result = sim(x_all, y_all, ws_amb=ws, wd_amb=wd)
    power = result.power()[:, :N_TARGET]
    return jnp.sum(power * weights[:, None]) * 8760 / 1e6

def direct_liberal_aep(p):
    """Liberal layout (optimized in isolation) evaluated WITH neighbors."""
    n_nb = p.shape[0] // 2
    nb_x, nb_y = p[:n_nb], p[n_nb:]
    x_all = jnp.concatenate([liberal_x, nb_x])
    y_all = jnp.concatenate([liberal_y, nb_y])
    result = sim(x_all, y_all, ws_amb=ws, wd_amb=wd)
    power = result.power()[:, :N_TARGET]
    return jnp.sum(power * weights[:, None]) * 8760 / 1e6

direct_cons_grad = jax.grad(direct_conservative_aep)(nb_params)
direct_lib_grad = jax.grad(direct_liberal_aep)(nb_params)
direct_grad = direct_cons_grad - direct_lib_grad
print(f"  direct_cons_grad = {np.array(direct_cons_grad)}")
print(f"  direct_lib_grad  = {np.array(direct_lib_grad)}")
print(f"  direct_grad (cons - lib) = {np.array(direct_grad)}")
print(f"  |direct_grad| = {float(jnp.linalg.norm(direct_grad)):.6e}")

# ── 2. Upstream gradient: d(AEP)/d(target positions) ─────────────────
print("\n" + "=" * 60)
print("2. UPSTREAM GRADIENT (d AEP / d target positions at optimum)")
print("=" * 60)
def aep_wrt_targets(x, y):
    x_all = jnp.concatenate([x, init_nb_x])
    y_all = jnp.concatenate([y, init_nb_y])
    result = sim(x_all, y_all, ws_amb=ws, wd_amb=wd)
    power = result.power()[:, :N_TARGET]
    return jnp.sum(power * weights[:, None]) * 8760 / 1e6

gx, gy = jax.grad(aep_wrt_targets, argnums=(0, 1))(cons_x, cons_y)
print(f"  g_x = {np.array(gx)}")
print(f"  g_y = {np.array(gy)}")
print(f"  |g| = {float(jnp.sqrt(jnp.sum(gx**2)+jnp.sum(gy**2))):.6e}")

# ── 3. Full IFT gradient via value_and_grad ───────────────────────────
print("\n" + "=" * 60)
print("3. FULL IFT GRADIENT (value_and_grad through sgd_solve_implicit)")
print("=" * 60)
def regret_fn(p):
    opt_x, opt_y = sgd_solve_implicit(
        objective_with_neighbors, liberal_x, liberal_y,
        boundary, min_spacing, settings, p,
    )
    n_nb = p.shape[0] // 2
    nb_x, nb_y = p[:n_nb], p[n_nb:]
    x_all = jnp.concatenate([opt_x, nb_x])
    y_all = jnp.concatenate([opt_y, nb_y])
    result = sim(x_all, y_all, ws_amb=ws, wd_amb=wd)
    power = result.power()[:, :N_TARGET]
    conservative_aep = jnp.sum(power * weights[:, None]) * 8760 / 1e6
    # Liberal layout (optimized in isolation) evaluated WITH neighbors
    x_lib_all = jnp.concatenate([liberal_x, nb_x])
    y_lib_all = jnp.concatenate([liberal_y, nb_y])
    result_lib = sim(x_lib_all, y_lib_all, ws_amb=ws, wd_amb=wd)
    power_lib = result_lib.power()[:, :N_TARGET]
    liberal_aep_present = jnp.sum(power_lib * weights[:, None]) * 8760 / 1e6
    return conservative_aep - liberal_aep_present

regret_val, ift_grad = jax.value_and_grad(regret_fn)(nb_params)
print(f"  regret = {float(regret_val):.6f}")
print(f"  ift_grad = {np.array(ift_grad)}")
print(f"  |ift_grad| = {float(jnp.linalg.norm(ift_grad)):.6e}")

# ── 4. Decompose: IFT indirect = full - direct ───────────────────────
print("\n" + "=" * 60)
print("4. DECOMPOSITION")
print("=" * 60)
# regret = conservative_aep - liberal_aep_present
# d(regret)/dp = d(cons)/dp - d(lib_present)/dp
# Direct part (targets fixed): d(cons)/dp|fixed - d(lib_present)/dp = direct_grad
# Indirect part (IFT): d(cons)/d(targets) * d(targets)/dp  (from IFT)
# Total = direct + indirect
indirect_grad = ift_grad - direct_grad
print(f"  direct (targets fixed) = {np.array(direct_grad)}")
print(f"  indirect (IFT)         = {np.array(indirect_grad)}")
print(f"  total (IFT full)       = {np.array(ift_grad)}")
print(f"  |direct|   = {float(jnp.linalg.norm(direct_grad)):.6e}")
print(f"  |indirect| = {float(jnp.linalg.norm(indirect_grad)):.6e}")
print(f"  |total|    = {float(jnp.linalg.norm(ift_grad)):.6e}")

# ── 5. FD ground truth ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. FD GROUND TRUTH")
print("=" * 60)
fd_step = 1.0
n_params = nb_params.shape[0]
grad_fd = np.zeros(n_params)
def fd_regret(nb_p):
    """Compute regret via forward solve for FD ground truth."""
    n_nb = nb_p.shape[0] // 2
    nb_x, nb_y = nb_p[:n_nb], nb_p[n_nb:]
    def obj_fn(x, y): return objective_with_neighbors(x, y, nb_p)
    ox, oy = topfarm_sgd_solve(obj_fn, liberal_x, liberal_y, boundary, min_spacing, settings)
    # Conservative AEP (with neighbors)
    x_all = jnp.concatenate([ox, nb_x])
    y_all = jnp.concatenate([oy, nb_y])
    result = sim(x_all, y_all, ws_amb=ws, wd_amb=wd)
    power = result.power()[:, :N_TARGET]
    cons_aep = float(jnp.sum(power * weights[:, None]) * 8760 / 1e6)
    # Liberal layout (optimized in isolation) evaluated WITH neighbors
    x_lib_all = jnp.concatenate([liberal_x, nb_x])
    y_lib_all = jnp.concatenate([liberal_y, nb_y])
    result_lib = sim(x_lib_all, y_lib_all, ws_amb=ws, wd_amb=wd)
    power_lib = result_lib.power()[:, :N_TARGET]
    lib_aep_present = float(jnp.sum(power_lib * weights[:, None]) * 8760 / 1e6)
    return cons_aep - lib_aep_present

for i in range(n_params):
    e = jnp.zeros(n_params).at[i].set(fd_step)
    r_plus = fd_regret(nb_params + e)
    r_minus = fd_regret(nb_params - e)

    grad_fd[i] = (r_plus - r_minus) / (2 * fd_step)
    label = f"nb_x[{i}]" if i < N_NEIGHBOR else f"nb_y[{i-N_NEIGHBOR}]"
    print(f"  param {i} ({label}): FD = {grad_fd[i]:.6e}  (r+={r_plus:.6f}, r-={r_minus:.6f})")

print(f"\n  FD grad = {grad_fd}")
print(f"  |FD grad| = {np.linalg.norm(grad_fd):.6e}")

# Compare
cos_sim = float(np.dot(np.array(ift_grad), grad_fd) /
               (np.linalg.norm(np.array(ift_grad)) * np.linalg.norm(grad_fd) + 1e-30))
print(f"\n  cos_sim(IFT, FD)   = {cos_sim:.4f}")
print(f"  ratio |IFT|/|FD|  = {float(jnp.linalg.norm(ift_grad))/np.linalg.norm(grad_fd):.6e}")

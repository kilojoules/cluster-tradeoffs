"""Diagnose IFT internals: check Jacobian, HVP, and CG at 50x50 scale."""
import jax
jax.config.update("jax_enable_x64", True)

import time
import jax.numpy as jnp
import numpy as np
from pathlib import Path
import pandas as pd

from pixwake import Curve, Turbine, WakeSimulation
from pixwake.deficit import BastankhahGaussianDeficit
from pixwake.optim.sgd import (
    SGDSettings, sgd_solve_implicit, topfarm_sgd_solve,
    boundary_penalty, spacing_penalty,
)

# ── Same setup as animate script ──
D = 240.0
N_TARGET = 50
N_NEIGHBOR = 50

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
boundary_np = raw[hull.vertices]
boundary = jnp.array(boundary_np)
min_spacing = 4.0 * D

ws_t = jnp.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.,
                   13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25.])
power_curve = jnp.array([
    0., 0., 2.3986, 209.2581, 689.1977, 1480.6085,
    2661.2377, 4308.9290, 6501.0566, 9260.5163, 12081.4039, 13937.2966,
    14705.0160, 14931.0392, 14985.2085, 14996.9062, 14999.3433, 14999.8550,
    14999.9662, 14999.9916, 14999.9978, 14999.9994, 14999.9998, 14999.9999,
    15000.0000, 15000.0000,
])
ct_curve = jnp.array([
    0.8889, 0.8889, 0.8889, 0.8003, 0.8000, 0.8000,
    0.8000, 0.8000, 0.7999, 0.7930, 0.7354, 0.6100,
    0.4764, 0.3698, 0.2915, 0.2341, 0.1910, 0.1581,
    0.1325, 0.1122, 0.0958, 0.0826, 0.0717, 0.0626,
    0.0550, 0.0486,
])
turbine = Turbine(
    rotor_diameter=D, hub_height=150.0,
    power_curve=Curve(ws=ws_t, values=power_curve),
    ct_curve=Curve(ws=ws_t, values=ct_curve),
)
sim = WakeSimulation(turbine, BastankhahGaussianDeficit(k=0.04))

# 51 subsampled timeseries
csv_path = Path(__file__).parent.parent / "energy_island_10y_daily_av_wind.csv"
df = pd.read_csv(csv_path, sep=';')
SUBSAMPLE = 73
wd = jnp.array(df['WD_150'].values[::SUBSAMPLE])
ws_wind = jnp.array(df['WS_150'].values[::SUBSAMPLE])
N_SAMPLES = len(wd)
HOURS_PER_SAMPLE = 24.0 * SUBSAMPLE
N_YEARS = 10.0

# Initial layouts (same seeds as animate)
from matplotlib.path import Path as MplPath
poly_path = MplPath(boundary_np)
rng = np.random.default_rng(42)
pts = []
while len(pts) < N_TARGET:
    cands = rng.uniform([boundary_np[:, 0].min(), boundary_np[:, 1].min()],
                        [boundary_np[:, 0].max(), boundary_np[:, 1].max()],
                        size=(N_TARGET * 5, 2))
    pts.extend(cands[poly_path.contains_points(cands)].tolist())
pts = np.array(pts[:N_TARGET])
init_x, init_y = jnp.array(pts[:, 0]), jnp.array(pts[:, 1])

# Neighbors
buffer = 3.0 * D
n_verts = len(boundary_np)
centroid = boundary_np.mean(axis=0)
edges_list = []
for j in range(n_verts):
    p0, p1 = boundary_np[j], boundary_np[(j + 1) % n_verts]
    edge = p1 - p0
    normal = np.array([-edge[1], edge[0]])
    normal /= np.linalg.norm(normal)
    if np.dot(normal, p0 - centroid) < 0:
        normal = -normal
    edges_list.append((p0 + normal * buffer, p1 + normal * buffer))
offset_verts = []
for j in range(n_verts):
    a0, a1 = edges_list[j]
    b0, b1 = edges_list[(j + 1) % n_verts]
    da, db = a1 - a0, b1 - b0
    denom = da[0] * db[1] - da[1] * db[0]
    if abs(denom) < 1e-12:
        offset_verts.append(a1)
    else:
        t = ((b0[0] - a0[0]) * db[1] - (b0[1] - a0[1]) * db[0]) / denom
        offset_verts.append(a0 + t * da)
exclusion = np.array(offset_verts)
exclusion_path = MplPath(exclusion)
pad = 30.0 * D
rng2 = np.random.default_rng(123)
nb_pts = []
while len(nb_pts) < N_NEIGHBOR:
    cands = rng2.uniform([boundary_np[:, 0].min() - pad, boundary_np[:, 1].min() - pad],
                         [boundary_np[:, 0].max() + pad, boundary_np[:, 1].max() + pad],
                         size=(N_NEIGHBOR * 5, 2))
    nb_pts.extend(cands[~exclusion_path.contains_points(cands)].tolist())
nb_pts = np.array(nb_pts[:N_NEIGHBOR])
init_nb_x, init_nb_y = jnp.array(nb_pts[:, 0]), jnp.array(nb_pts[:, 1])

sgd_settings = SGDSettings(
    learning_rate=D / 5, max_iter=1000, tol=1e-6,
    ift_cg_damping=0.01, ift_polish_max_iter=20, ift_polish_tol=1e-8,
)
params = jnp.concatenate([init_nb_x, init_nb_y])

def compute_aep(target_x, target_y, nb_x=None, nb_y=None):
    if nb_x is not None:
        x_all = jnp.concatenate([target_x, nb_x])
        y_all = jnp.concatenate([target_y, nb_y])
    else:
        x_all = target_x
        y_all = target_y
    result = sim(x_all, y_all, ws_amb=ws_wind, wd_amb=wd)
    pw = result.power()[:, :N_TARGET]
    return jnp.sum(pw) * HOURS_PER_SAMPLE / N_YEARS / 1e6

def objective_with_neighbors(x, y, neighbor_params):
    n_nb = neighbor_params.shape[0] // 2
    nb_x, nb_y = neighbor_params[:n_nb], neighbor_params[n_nb:]
    return -compute_aep(x, y, nb_x, nb_y)

def liberal_objective(x, y):
    result = sim(x, y, ws_amb=ws_wind, wd_amb=wd)
    pw = result.power()[:, :N_TARGET]
    return -jnp.sum(pw) * HOURS_PER_SAMPLE / N_YEARS / 1e6

# Liberal baseline
print("Computing liberal layout...", flush=True)
t0 = time.time()
liberal_x, liberal_y = topfarm_sgd_solve(
    liberal_objective, init_x, init_y, boundary, min_spacing, sgd_settings,
)
print(f"Liberal AEP: {float(compute_aep(liberal_x, liberal_y)):.2f} GWh ({time.time()-t0:.1f}s)", flush=True)

# Run inner optimization WITH polishing to get (opt_x, opt_y)
from pixwake.optim.sgd import _sgd_and_polish
print("\nRunning inner optimization (SGD + Newton polish)...", flush=True)
t0 = time.time()
opt_x, opt_y = _sgd_and_polish(
    objective_with_neighbors, init_x, init_y, boundary, min_spacing, sgd_settings, params,
)
print(f"Inner solve: {time.time()-t0:.1f}s", flush=True)

# Check inner convergence (gradient at optimum should be ~0)
rho = sgd_settings.ks_rho
def total_obj(x, y, p):
    obj = objective_with_neighbors(x, y, p)
    pen_b = sgd_settings.boundary_weight * boundary_penalty(x, y, boundary, rho)
    pen_s = sgd_settings.spacing_weight * spacing_penalty(x, y, min_spacing, rho)
    return obj + pen_b + pen_s

grad_at_opt_x, grad_at_opt_y = jax.grad(lambda x, y: total_obj(x, y, params), argnums=(0, 1))(opt_x, opt_y)
print(f"\nInner grad at optimum:")
print(f"  |grad_x| = {float(jnp.linalg.norm(grad_at_opt_x)):.6e}")
print(f"  |grad_y| = {float(jnp.linalg.norm(grad_at_opt_y)):.6e}")
print(f"  max|grad_x| = {float(jnp.max(jnp.abs(grad_at_opt_x))):.6e}")
print(f"  max|grad_y| = {float(jnp.max(jnp.abs(grad_at_opt_y))):.6e}")

# Check cross-Jacobian d(grad_xy)/d(params) via jacfwd
print("\nComputing cross-Jacobian via jacfwd...", flush=True)
t0 = time.time()
def grad_xy_fn(p):
    return jax.grad(lambda x, y: total_obj(x, y, p), argnums=(0, 1))(opt_x, opt_y)

jac = jax.jacfwd(grad_xy_fn)(params)
jac_x = jac[0]  # (n_turbines, n_params)
jac_y = jac[1]  # (n_turbines, n_params)
print(f"Jacobian time: {time.time()-t0:.1f}s")
print(f"  jac_x shape: {jac_x.shape}, |jac_x| = {float(jnp.linalg.norm(jac_x)):.6e}, max = {float(jnp.max(jnp.abs(jac_x))):.6e}")
print(f"  jac_y shape: {jac_y.shape}, |jac_y| = {float(jnp.linalg.norm(jac_y)):.6e}, max = {float(jnp.max(jnp.abs(jac_y))):.6e}")

# Check HVP
print("\nChecking HVP with random vector...", flush=True)
key = jax.random.PRNGKey(0)
vx = jax.random.normal(key, shape=opt_x.shape)
vy = jax.random.normal(jax.random.PRNGKey(1), shape=opt_y.shape)

def grad_obj(x, y):
    return jax.grad(lambda xx, yy: total_obj(xx, yy, params), argnums=(0, 1))(x, y)

_, (hvp_x, hvp_y) = jax.jvp(grad_obj, (opt_x, opt_y), (vx, vy))
print(f"  |hvp_x| = {float(jnp.linalg.norm(hvp_x)):.6e}")
print(f"  |hvp_y| = {float(jnp.linalg.norm(hvp_y)):.6e}")

# Test CG manually with the upstream g vector
print("\nTesting CG with g = (1, 1, ..., 1)...", flush=True)
g_x = jnp.ones_like(opt_x)
g_y = jnp.ones_like(opt_y)

damping = 0.01
def damped_hvp(vx, vy):
    _, (hx, hy) = jax.jvp(grad_obj, (opt_x, opt_y), (vx, vy))
    return hx + damping * vx, hy + damping * vy

# Run 5 CG iterations manually to see convergence
from jax.lax import while_loop

def dot_xy(ax, ay, bx, by):
    return jnp.sum(ax * bx) + jnp.sum(ay * by)

v_x, v_y = jnp.zeros_like(g_x), jnp.zeros_like(g_y)
r_x, r_y = g_x, g_y
p_x, p_y = g_x, g_y
rs_old = dot_xy(r_x, r_y, r_x, r_y)
print(f"  CG iter 0: |r|^2 = {float(rs_old):.6e}")

for it in range(10):
    ap_x, ap_y = damped_hvp(p_x, p_y)
    pap = dot_xy(p_x, p_y, ap_x, ap_y)

    if float(pap) <= 0:
        print(f"  CG iter {it+1}: NEGATIVE CURVATURE (pAp = {float(pap):.6e}), stopping")
        break

    alpha_cg = float(rs_old) / max(float(pap), 1e-30)
    v_x = v_x + alpha_cg * p_x
    v_y = v_y + alpha_cg * p_y
    r_x = r_x - alpha_cg * ap_x
    r_y = r_y - alpha_cg * ap_y
    rs_new = dot_xy(r_x, r_y, r_x, r_y)
    print(f"  CG iter {it+1}: |r|^2 = {float(rs_new):.6e}, pAp = {float(pap):.6e}, alpha = {alpha_cg:.6e}")

    beta_cg = float(rs_new) / max(float(rs_old), 1e-30)
    p_x = r_x + beta_cg * p_x
    p_y = r_y + beta_cg * p_y
    rs_old = rs_new

print(f"\n  CG solution: |v_x| = {float(jnp.linalg.norm(v_x)):.6e}, |v_y| = {float(jnp.linalg.norm(v_y)):.6e}")

# Final: compute what the IFT gradient would be
# grad_params = adj_x^T @ jac_x + adj_y^T @ jac_y  (using g = upstream grad)
# But for actual regret, g comes from the chain rule
print("\nDone!", flush=True)

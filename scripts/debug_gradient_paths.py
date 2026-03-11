"""Decompose regret gradient into IFT path vs direct paths."""
import jax
jax.config.update("jax_enable_x64", True)

import time
import jax.numpy as jnp
import numpy as np
from pathlib import Path
import pandas as pd

from pixwake import Curve, Turbine, WakeSimulation
from pixwake.deficit import BastankhahGaussianDeficit
from pixwake.optim.sgd import SGDSettings, sgd_solve_implicit, topfarm_sgd_solve

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
power_curve = jnp.array([0., 0., 2.3986, 209.2581, 689.1977, 1480.6085,
    2661.2377, 4308.9290, 6501.0566, 9260.5163, 12081.4039, 13937.2966,
    14705.0160, 14931.0392, 14985.2085, 14996.9062, 14999.3433, 14999.8550,
    14999.9662, 14999.9916, 14999.9978, 14999.9994, 14999.9998, 14999.9999,
    15000.0000, 15000.0000])
ct_curve = jnp.array([0.8889, 0.8889, 0.8889, 0.8003, 0.8000, 0.8000,
    0.8000, 0.8000, 0.7999, 0.7930, 0.7354, 0.6100,
    0.4764, 0.3698, 0.2915, 0.2341, 0.1910, 0.1581,
    0.1325, 0.1122, 0.0958, 0.0826, 0.0717, 0.0626, 0.0550, 0.0486])
turbine = Turbine(rotor_diameter=D, hub_height=150.0,
    power_curve=Curve(ws=ws_t, values=power_curve),
    ct_curve=Curve(ws=ws_t, values=ct_curve))
sim = WakeSimulation(turbine, BastankhahGaussianDeficit(k=0.04))

csv_path = Path(__file__).parent.parent / "energy_island_10y_daily_av_wind.csv"
df = pd.read_csv(csv_path, sep=';')
SUBSAMPLE = 73
wd = jnp.array(df['WD_150'].values[::SUBSAMPLE])
ws_wind = jnp.array(df['WS_150'].values[::SUBSAMPLE])
HOURS_PER_SAMPLE = 24.0 * SUBSAMPLE
N_YEARS = 10.0

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

sgd_settings = SGDSettings(learning_rate=D / 5, max_iter=1000, tol=1e-6, ift_cg_damping=0.01)
params = jnp.concatenate([init_nb_x, init_nb_y])

def compute_aep(target_x, target_y, nb_x, nb_y):
    x_all = jnp.concatenate([target_x, nb_x])
    y_all = jnp.concatenate([target_y, nb_y])
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

print("Computing liberal layout...", flush=True)
liberal_x, liberal_y = topfarm_sgd_solve(
    liberal_objective, init_x, init_y, boundary, min_spacing, sgd_settings)
print(f"Liberal AEP: {float(compute_aep(liberal_x, liberal_y, jnp.zeros(1), jnp.zeros(1))):.2f} (placeholder)")

# Solve inner problem once (fixed layout)
print("\nSolving inner problem...", flush=True)
def obj_fn(x, y):
    return objective_with_neighbors(x, y, params)
opt_x, opt_y = topfarm_sgd_solve(obj_fn, init_x, init_y, boundary, min_spacing, sgd_settings)

# ── Path 1: DIRECT-ONLY gradient (treat opt_x, opt_y as fixed) ──
print("\nComputing DIRECT gradient (opt_xy fixed, no IFT)...", flush=True)
def regret_direct_only(neighbor_params):
    """Regret treating inner layout as FIXED (no IFT path)."""
    n_nb = neighbor_params.shape[0] // 2
    nb_x, nb_y = neighbor_params[:n_nb], neighbor_params[n_nb:]
    conservative_aep = compute_aep(opt_x, opt_y, nb_x, nb_y)
    liberal_aep_present = compute_aep(liberal_x, liberal_y, nb_x, nb_y)
    return conservative_aep - liberal_aep_present

t0 = time.time()
regret_val, direct_grad = jax.value_and_grad(regret_direct_only)(params)
print(f"Regret = {float(regret_val):.4f} GWh, |direct_grad| = {float(jnp.linalg.norm(direct_grad)):.6f} ({time.time()-t0:.1f}s)")

# ── Path 2: FULL gradient (includes IFT) ──
print("\nComputing FULL gradient (includes IFT through sgd_solve_implicit)...", flush=True)
def regret_full(neighbor_params):
    n_nb = neighbor_params.shape[0] // 2
    nb_x, nb_y = neighbor_params[:n_nb], neighbor_params[n_nb:]
    opt_x_ift, opt_y_ift = sgd_solve_implicit(
        objective_with_neighbors, init_x, init_y,
        boundary, min_spacing, sgd_settings, neighbor_params)
    conservative_aep = compute_aep(opt_x_ift, opt_y_ift, nb_x, nb_y)
    liberal_aep_present = compute_aep(liberal_x, liberal_y, nb_x, nb_y)
    return conservative_aep - liberal_aep_present

t0 = time.time()
regret_val2, full_grad = jax.value_and_grad(regret_full)(params)
print(f"Regret = {float(regret_val2):.4f} GWh, |full_grad| = {float(jnp.linalg.norm(full_grad)):.6f} ({time.time()-t0:.1f}s)")

# ── Path 3: Outer FD (ground truth) ──
print("\nComputing outer FD gradient (first 5 params)...", flush=True)
eps = 10.0

# Compare all three for first 5 params
print(f"\n{'param':>8}  {'Direct':>12}  {'Full(IFT)':>12}  {'OuterFD':>12}  {'Dir/FD':>8}  {'IFT/FD':>8}")
print("-" * 76)
for i in range(5):
    p_plus = params.at[i].set(params[i] + eps)
    p_minus = params.at[i].set(params[i] - eps)
    r_plus = regret_full(p_plus)
    r_minus = regret_full(p_minus)
    fd_grad_i = float(r_plus - r_minus) / (2 * eps)
    direct_i = float(direct_grad[i])
    full_i = float(full_grad[i])

    dir_ratio = direct_i / fd_grad_i if abs(fd_grad_i) > 1e-12 else float('inf')
    ift_ratio = full_i / fd_grad_i if abs(fd_grad_i) > 1e-12 else float('inf')

    ptype = "nb_x" if i < N_NEIGHBOR else "nb_y"
    idx = i if i < N_NEIGHBOR else i - N_NEIGHBOR
    print(f"  {ptype}[{idx:2d}]  {direct_i:+12.6e}  {full_i:+12.6e}  {fd_grad_i:+12.6e}  {dir_ratio:+8.2f}  {ift_ratio:+8.2f}")

print("\nDone!", flush=True)

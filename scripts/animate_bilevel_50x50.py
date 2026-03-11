"""Animated bilevel CMA-ES search: 50 targets x 4 neighbors, DEI polygon boundary.

Uses CMA-ES (derivative-free) to optimize neighbor positions that maximize regret.
Single dominant wind direction (278 deg, 9.5 m/s) for fast inner solves, with
full timeseries post-evaluation of the best result.

Produces three MP4s:
  1. analysis/bilevel_dei_50x50/outer_loop.mp4   — outer loop (neighbor evolution)
  2. analysis/bilevel_dei_50x50/inner_first.mp4  — inner SGD at first outer iteration
  3. analysis/bilevel_dei_50x50/inner_last.mp4   — inner SGD at last outer iteration

Usage:
    pixi run python scripts/animate_bilevel_50x50.py
"""

import jax

jax.config.update("jax_enable_x64", True)

import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon as MplPolygon

import cma

from pixwake import Curve, Turbine, WakeSimulation
from pixwake.deficit import BastankhahGaussianDeficit
from pixwake.optim.sgd import (
    SGDSettings,
    _compute_mid_bisection,
    _init_sgd_state,
    _sgd_step,
    boundary_penalty,
    spacing_penalty,
    topfarm_sgd_solve,
)

# ── DEI Configuration ──────────────────────────────────────────────────
D = 240.0  # Rotor diameter (DEI 15 MW turbine)
N_TARGET = 50
N_NEIGHBOR = 15  # 30 outer dims
INNER_MAX_ITER = 1000
SNAPSHOT_EVERY = 10  # inner SGD: capture every N steps
MIN_SPACING_D = 4.0
BUFFER_D = 3.0  # exclusion buffer around target boundary (in D)
NEIGHBOR_PAD_D = 30.0  # neighbor bounding box padding (in D)

# CMA-ES settings
CMA_MAXITER = 50
CMA_POPSIZE = 20  # larger population for 30D
CMA_SIGMA0 = 5.0 * D  # initial step size ~1200m

output_dir = Path("analysis/bilevel_dei_50x50")
output_dir.mkdir(parents=True, exist_ok=True)


# ── DEI Setup Functions ────────────────────────────────────────────────

def load_target_boundary():
    """Load DEI target farm boundary — convex hull, CCW order."""
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
    """Create 15 MW DEI turbine with exact PyWake power/CT curves."""
    ws_t = jnp.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.,
                       13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25.])
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
        rotor_diameter=D, hub_height=150.0,
        power_curve=Curve(ws=ws_t, values=power),
        ct_curve=Curve(ws=ws_t, values=ct),
    )


def load_wind_data():
    """Load DEI wind data -> binned 24-sector wind rose."""
    import pandas as pd
    csv_path = Path(__file__).parent.parent / "energy_island_10y_daily_av_wind.csv"
    df = pd.read_csv(csv_path, sep=';')
    wd_ts, ws_ts = df['WD_150'].values, df['WS_150'].values
    n_bins = 24
    bin_edges = np.linspace(0, 360, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    weights = np.zeros(n_bins)
    mean_speeds = np.zeros(n_bins)
    for i in range(n_bins):
        mask = ((wd_ts >= bin_edges[i]) | (wd_ts < bin_edges[0])) if i == n_bins - 1 \
            else ((wd_ts >= bin_edges[i]) & (wd_ts < bin_edges[i + 1]))
        weights[i] = mask.sum()
        mean_speeds[i] = ws_ts[mask].mean() if mask.sum() > 0 else ws_ts.mean()
    weights /= weights.sum()
    return jnp.array(bin_centers), jnp.array(mean_speeds), jnp.array(weights)


def generate_initial_layout(boundary, n_turbines, seed=42):
    """Random layout inside polygon via rejection sampling."""
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


def place_initial_neighbors(boundary, n_neighbors, buffer, seed=123):
    """Scatter neighbors randomly outside the buffered boundary."""
    from matplotlib.path import Path as MplPath
    n_verts = len(boundary)
    centroid = boundary.mean(axis=0)
    edges_list = []
    for j in range(n_verts):
        p0 = boundary[j]
        p1 = boundary[(j + 1) % n_verts]
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

    pad = NEIGHBOR_PAD_D * D
    x_min = boundary[:, 0].min() - pad
    x_max = boundary[:, 0].max() + pad
    y_min = boundary[:, 1].min() - pad
    y_max = boundary[:, 1].max() + pad

    rng = np.random.default_rng(seed)
    pts = []
    while len(pts) < n_neighbors:
        cands = rng.uniform([x_min, y_min], [x_max, y_max], size=(n_neighbors * 5, 2))
        outside = ~exclusion_path.contains_points(cands)
        pts.extend(cands[outside].tolist())
    pts = np.array(pts[:n_neighbors])
    return jnp.array(pts[:, 0]), jnp.array(pts[:, 1])


# ── Load DEI data ──────────────────────────────────────────────────────
boundary_np = load_target_boundary()
boundary = jnp.array(boundary_np)
min_spacing = MIN_SPACING_D * D
buffer = BUFFER_D * D

turbine = create_dei_turbine()
sim = WakeSimulation(turbine, BastankhahGaussianDeficit(k=0.04))

# 4 wind directions covering all quadrants — so every neighbor matters
# Weighted by approximate DEI frequency (W dominant, E weakest)
wd = jnp.array([278.0, 45.0, 150.0, 200.0])
ws = jnp.array([9.5, 8.0, 7.5, 8.5])
wd_weights = jnp.array([0.35, 0.20, 0.20, 0.25])  # roughly matches DEI rose
N_SAMPLES = len(wd)
HOURS_PER_SAMPLE = 8760.0  # will be scaled by weights
N_YEARS = 1.0

# Coordinate transform: center on polygon centroid, display in km
cx_center = boundary_np[:, 0].mean()
cy_center = boundary_np[:, 1].mean()


def to_km(x, y):
    return (np.asarray(x) - cx_center) / 1000., (np.asarray(y) - cy_center) / 1000.


# Neighbor bounding box (for clipping)
pad = NEIGHBOR_PAD_D * D
nb_clip_x_lo = float(boundary_np[:, 0].min() - pad)
nb_clip_x_hi = float(boundary_np[:, 0].max() + pad)
nb_clip_y_lo = float(boundary_np[:, 1].min() - pad)
nb_clip_y_hi = float(boundary_np[:, 1].max() + pad)

# Initial layouts
init_x, init_y = generate_initial_layout(boundary_np, N_TARGET, seed=42)
init_nb_x, init_nb_y = place_initial_neighbors(boundary_np, N_NEIGHBOR, buffer, seed=123)

sgd_settings = SGDSettings(
    learning_rate=D / 5, max_iter=INNER_MAX_ITER, tol=1e-6,
)

print(f"DEI polygon: {boundary_np.shape[0]} vertices")
print(f"Wind: {N_SAMPLES} directions (weighted), covering all quadrants")
print(f"Turbine: D={D:.0f}m, 15 MW")
print(f"Target turbines: {N_TARGET}, neighbors: {N_NEIGHBOR}")
print(f"Min spacing: {MIN_SPACING_D:.0f}D = {min_spacing:.0f}m")
print(f"Buffer: {BUFFER_D:.0f}D = {buffer:.0f}m")
print(f"CMA-ES: popsize={CMA_POPSIZE}, maxiter={CMA_MAXITER}, sigma0={CMA_SIGMA0:.0f}m", flush=True)


# ── Objectives ──────────────────────────────────────────────────────────

def compute_aep(target_x, target_y, nb_x=None, nb_y=None):
    """Compute AEP for target turbines, weighted by wind direction frequency."""
    if nb_x is not None:
        x_all = jnp.concatenate([target_x, nb_x])
        y_all = jnp.concatenate([target_y, nb_y])
    else:
        x_all = target_x
        y_all = target_y
    result = sim(x_all, y_all, ws_amb=ws, wd_amb=wd)
    power = result.power()[:, :N_TARGET]  # (N_SAMPLES, n_target) in kW
    # Weight each direction by its frequency, sum over targets, convert to GWh/yr
    weighted_power = jnp.sum(power * wd_weights[:, None])  # broadcast weights over turbines
    return weighted_power * HOURS_PER_SAMPLE / N_YEARS / 1e6


def liberal_objective(x, y):
    return -compute_aep(x, y)


# ── Liberal baseline ────────────────────────────────────────────────────
print("\nComputing liberal layout (no neighbors)...", flush=True)
t0 = time.time()
liberal_x, liberal_y = topfarm_sgd_solve(
    liberal_objective, init_x, init_y, boundary, min_spacing, sgd_settings,
)
liberal_aep = float(compute_aep(liberal_x, liberal_y))
print(f"Liberal AEP: {liberal_aep:.2f} GWh ({time.time()-t0:.1f}s)", flush=True)


# ── Project neighbors outside boundary + buffer ──────────────────────
def _project_outside_boundary(nb_x, nb_y, bnd, buffer_dist):
    """Push each neighbor outside the convex boundary + buffer."""
    n_verts = bnd.shape[0]
    px, py = nb_x, nb_y
    for e in range(n_verts):
        x1, y1 = bnd[e]
        x2, y2 = bnd[(e + 1) % n_verts]
        ex, ey = x2 - x1, y2 - y1
        edge_len = jnp.sqrt(ex**2 + ey**2) + 1e-10
        nx, ny = -ey / edge_len, ex / edge_len  # inward normal (CCW)
        dist = (px - x1) * nx + (py - y1) * ny
        push = jnp.maximum(dist + buffer_dist, 0.0)
        px = px - push * nx
        py = py - push * ny
    return px, py


# ── Regret function (CMA-ES black-box) ────────────────────────────────

def compute_regret(params_np):
    """Compute regret for given neighbor positions (numpy in, float out)."""
    params = jnp.array(params_np)
    n_nb = params.shape[0] // 2
    nb_x, nb_y = params[:n_nb], params[n_nb:]

    # Enforce: neighbors must stay outside boundary + buffer
    nb_x, nb_y = _project_outside_boundary(nb_x, nb_y, boundary, buffer)

    def obj_fn(x, y):
        return -compute_aep(x, y, nb_x, nb_y)

    opt_x, opt_y = topfarm_sgd_solve(
        obj_fn, init_x, init_y, boundary, min_spacing, sgd_settings,
    )
    conservative_aep = float(compute_aep(opt_x, opt_y, nb_x, nb_y))
    liberal_aep_present = float(compute_aep(liberal_x, liberal_y, nb_x, nb_y))
    return conservative_aep - liberal_aep_present


# ── Run CMA-ES outer loop ─────────────────────────────────────────────
x0 = np.concatenate([np.array(init_nb_x), np.array(init_nb_y)])

lower = np.array([nb_clip_x_lo] * N_NEIGHBOR + [nb_clip_y_lo] * N_NEIGHBOR)
upper = np.array([nb_clip_x_hi] * N_NEIGHBOR + [nb_clip_y_hi] * N_NEIGHBOR)

es = cma.CMAEvolutionStrategy(x0, CMA_SIGMA0, {
    'bounds': [lower.tolist(), upper.tolist()],
    'maxiter': CMA_MAXITER,
    'popsize': CMA_POPSIZE,
    'seed': 42,
    'verbose': 1,
    'tolfun': 0,           # don't stop on flat fitness — need exploration
    'tolfunhist': 0,       # same for fitness history
    'tolflatfitness': 0,   # don't stop when all fitnesses are equal
    'tolstagnation': 200,  # generous stagnation tolerance
})

history_regret = []  # best-ever regret at each generation (for convergence plot)
history_sigma = []
history_nb = []      # per-generation best neighbor positions (for animation)
best_ever_regret = -np.inf
best_ever_params = None
gen = 0

print(f"\nRunning CMA-ES (popsize={CMA_POPSIZE}, maxiter={CMA_MAXITER})...", flush=True)
print(f"{'Gen':>4}  {'Gen Best':>10}  {'Best Ever':>10}  {'Sigma':>10}  {'Time':>8}")
print("-" * 55, flush=True)

while not es.stop():
    t0 = time.time()
    solutions = es.ask()
    fitnesses = []
    for s in solutions:
        regret = compute_regret(s)
        fitnesses.append(-regret)  # CMA-ES minimizes
    es.tell(solutions, fitnesses)

    # Track best of this generation AND best-ever
    best_idx = int(np.argmin(fitnesses))
    gen_best_regret = -fitnesses[best_idx]
    gen_best_params = solutions[best_idx]
    if gen_best_regret > best_ever_regret:
        best_ever_regret = gen_best_regret
        best_ever_params = gen_best_params.copy()
    # Project per-gen best params for animation (shows exploration)
    proj_x, proj_y = _project_outside_boundary(
        jnp.array(gen_best_params[:N_NEIGHBOR]),
        jnp.array(gen_best_params[N_NEIGHBOR:]),
        boundary, buffer,
    )
    history_regret.append(best_ever_regret)
    history_sigma.append(es.sigma)
    history_nb.append((np.array(proj_x), np.array(proj_y)))
    gen += 1
    elapsed = time.time() - t0
    print(f"{gen:4d}  {gen_best_regret:10.4f}  {best_ever_regret:10.4f}  {es.sigma:10.1f}  {elapsed:7.1f}s",
          flush=True)

es.result_pretty()
best_params = best_ever_params
# Project to get actual positions
proj_best_x, proj_best_y = _project_outside_boundary(
    jnp.array(best_params[:N_NEIGHBOR]),
    jnp.array(best_params[N_NEIGHBOR:]),
    boundary, buffer,
)
neighbor_params = jnp.concatenate([proj_best_x, proj_best_y])

n_total = len(history_regret)
print(f"\nCMA-ES done. {n_total} generations.", flush=True)
if n_total > 0:
    print(f"Initial best regret: {history_regret[0]:.3f} GWh")
    print(f"Final best regret:   {history_regret[-1]:.3f} GWh")
    print(f"Overall best regret: {max(history_regret):.3f} GWh", flush=True)

if n_total == 0:
    print("No generations completed — cannot generate animations.")
    raise SystemExit(1)

# ── Full timeseries post-evaluation ───────────────────────────────────
print("\nEvaluating best result under full timeseries...", flush=True)
import pandas as pd
SUBSAMPLE_FULL = 73  # ~50 samples
csv_path = Path(__file__).parent.parent / "energy_island_10y_daily_av_wind.csv"
df = pd.read_csv(csv_path, sep=';')
wd_full = jnp.array(df['WD_150'].values[::SUBSAMPLE_FULL])
ws_full = jnp.array(df['WS_150'].values[::SUBSAMPLE_FULL])
hours_full = 24.0 * SUBSAMPLE_FULL
n_years_full = 10.0

# Temporarily swap wind data for full evaluation
wd_save, ws_save, weights_save = wd, ws, wd_weights
HOURS_save, NYEARS_save = HOURS_PER_SAMPLE, N_YEARS

best_nb_x, best_nb_y = _project_outside_boundary(
    jnp.array(best_params[:N_NEIGHBOR]),
    jnp.array(best_params[N_NEIGHBOR:]),
    boundary, buffer,
)

# Evaluate with full timeseries (patch globals temporarily)
n_full = len(wd_full)
wd, ws = wd_full, ws_full
wd_weights = jnp.ones(n_full)  # each sample is a time period, not a probability
HOURS_PER_SAMPLE, N_YEARS = hours_full, n_years_full

t0 = time.time()
full_liberal_aep = float(compute_aep(liberal_x, liberal_y))
full_liberal_aep_present = float(compute_aep(liberal_x, liberal_y, best_nb_x, best_nb_y))

def full_obj(x, y):
    return -compute_aep(x, y, best_nb_x, best_nb_y)
full_opt_x, full_opt_y = topfarm_sgd_solve(
    full_obj, init_x, init_y, boundary, min_spacing, sgd_settings,
)
full_conservative_aep = float(compute_aep(full_opt_x, full_opt_y, best_nb_x, best_nb_y))
full_regret = full_conservative_aep - full_liberal_aep_present
print(f"  Full timeseries ({len(wd_full)} samples):")
print(f"    Liberal AEP (no neighbors): {full_liberal_aep:.2f} GWh/yr")
print(f"    Liberal AEP (with neighbors): {full_liberal_aep_present:.2f} GWh/yr")
print(f"    Conservative AEP: {full_conservative_aep:.2f} GWh/yr")
print(f"    Regret: {full_regret:.4f} GWh/yr  ({time.time()-t0:.1f}s)", flush=True)

# Restore 4-WD globals for animation frame recomputation
wd, ws, wd_weights = wd_save, ws_save, weights_save
HOURS_PER_SAMPLE, N_YEARS = HOURS_save, NYEARS_save

first_nb_params = jnp.concatenate([jnp.array(history_nb[0][0]), jnp.array(history_nb[0][1])])
last_nb_params = neighbor_params

# Subsample for animation frames — always include first and last
ANIM_FRAMES = 250
FRAME_STRIDE = max(1, n_total // ANIM_FRAMES)
frame_indices = sorted(set(
    list(range(0, n_total, FRAME_STRIDE)) + [n_total - 1]
))
n_frames = len(frame_indices)

# Compute conservative layouts only for animation frames
print(f"\nComputing conservative layouts for {n_frames} animation frames...", flush=True)
history_target = {}
for count, fi in enumerate(frame_indices):
    nb_x_fi = jnp.array(history_nb[fi][0])
    nb_y_fi = jnp.array(history_nb[fi][1])

    def frame_obj(x, y):
        return -compute_aep(x, y, nb_x_fi, nb_y_fi)

    opt_x, opt_y = topfarm_sgd_solve(
        frame_obj, init_x, init_y, boundary, min_spacing, sgd_settings,
    )
    history_target[fi] = (np.array(opt_x), np.array(opt_y))
    if count % 10 == 0:
        print(f"  {count}/{n_frames} frames computed", flush=True)
print(f"  Done ({n_frames} layouts computed)", flush=True)

# ══════════════════════════════════════════════════════════════════════════
# RENDER 1: Outer loop animation
# ══════════════════════════════════════════════════════════════════════════
print(f"\nRendering outer loop animation ({n_frames} frames)...", flush=True)

bnd_km = np.column_stack(to_km(boundary_np[:, 0], boundary_np[:, 1]))
liberal_x_np = np.array(liberal_x)
liberal_y_np = np.array(liberal_y)
lib_km_x, lib_km_y = to_km(liberal_x_np, liberal_y_np)

# Axis limits (km)
all_nb_x = np.concatenate([nb[0] for nb in history_nb])
all_nb_y = np.concatenate([nb[1] for nb in history_nb])
all_nb_km_x, all_nb_km_y = to_km(all_nb_x, all_nb_y)
pad_km = 3.0
layout_x_lo = min(bnd_km[:, 0].min(), all_nb_km_x.min()) - pad_km
layout_x_hi = max(bnd_km[:, 0].max(), all_nb_km_x.max()) + pad_km
layout_y_lo = min(bnd_km[:, 1].min(), all_nb_km_y.min()) - pad_km
layout_y_hi = max(bnd_km[:, 1].max(), all_nb_km_y.max()) + pad_km

regret_lo = min(history_regret) - abs(min(history_regret)) * 0.1
regret_hi = max(history_regret) * 1.1
sigma_lo = min(history_sigma) * 0.5
sigma_hi = max(history_sigma) * 1.2

fig_outer, axes_outer = plt.subplots(1, 3, figsize=(20, 7))


def draw_outer(anim_idx):
    for ax in axes_outer:
        ax.clear()

    i = frame_indices[anim_idx]
    nb_x_cur, nb_y_cur = history_nb[i]
    opt_x, opt_y = history_target[i]

    ax = axes_outer[0]
    # Draw polygon boundary (closed)
    poly_patch = MplPolygon(bnd_km, closed=True, fill=True,
                            facecolor='lightyellow', edgecolor='black', lw=2)
    ax.add_patch(poly_patch)

    # Liberal layout (fixed)
    ax.scatter(lib_km_x, lib_km_y, c="royalblue", marker="^",
               s=25, alpha=0.4, label="Liberal", zorder=4)
    # Conservative layout
    opt_km_x, opt_km_y = to_km(opt_x, opt_y)
    ax.scatter(opt_km_x, opt_km_y, c="seagreen", marker="s",
               s=25, alpha=0.6, label="Conservative", zorder=5)

    # Neighbor trail
    trail_step = max(1, i // 10)
    for j in range(0, i + 1, trail_step):
        nbx, nby = history_nb[j]
        nbkx, nbky = to_km(nbx, nby)
        alpha_val = 0.05 + 0.4 * (j / max(i, 1))
        ax.scatter(nbkx, nbky, c="red", marker=".", s=8,
                   alpha=alpha_val, zorder=3)
    # Current neighbors
    nb_km_x, nb_km_y = to_km(nb_x_cur, nb_y_cur)
    ax.scatter(nb_km_x, nb_km_y, c="red", marker="D", s=40,
               edgecolors="black", linewidths=0.5,
               label="Neighbors", zorder=6)

    ax.set_xlim(layout_x_lo, layout_x_hi)
    ax.set_ylim(layout_y_lo, layout_y_hi)
    ax.set_aspect("equal")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_title("Layout", fontsize=11)
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(True, alpha=0.2)

    # Regret convergence
    ax = axes_outer[1]
    ax.plot(history_regret[:i + 1], "-", color="purple", lw=1.5, alpha=0.7)
    ax.set_xlim(-0.5, n_total - 0.5)
    ax.set_ylim(regret_lo, regret_hi)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Regret (GWh)")
    ax.set_title(f"Regret = {history_regret[i]:.2f} GWh", fontsize=11)
    ax.grid(True, alpha=0.3)

    # CMA-ES sigma (step size)
    ax = axes_outer[2]
    ax.plot(history_sigma[:i + 1], "-", color="darkorange", lw=1.5, alpha=0.7)
    ax.set_xlim(-0.5, n_total - 0.5)
    ax.set_ylim(sigma_lo, sigma_hi)
    ax.set_xlabel("Generation")
    ax.set_ylabel("CMA-ES sigma (m)")
    ax.set_title(f"sigma = {history_sigma[i]:.0f} m", fontsize=11)
    ax.grid(True, alpha=0.3)

    fig_outer.suptitle(
        f"DEI 50x{N_NEIGHBOR} CMA-ES Bilevel — Generation {i}/{n_total - 1}",
        fontsize=13,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])


anim_outer = FuncAnimation(fig_outer, draw_outer, frames=n_frames, interval=150, repeat=True)
anim_outer.save(str(output_dir / "outer_loop.mp4"), writer="ffmpeg", fps=12, dpi=150)
plt.close(fig_outer)
print(f"Saved -> {output_dir / 'outer_loop.mp4'}", flush=True)

# ══════════════════════════════════════════════════════════════════════════
# Inner SGD capture (unrolled Python loop)
# ══════════════════════════════════════════════════════════════════════════


def sgd_solve_with_history(objective_fn, init_x, init_y, boundary, min_spacing,
                           settings, snapshot_every=SNAPSHOT_EVERY):
    """Run SGD in Python loop, capturing snapshots."""
    if settings.mid is None:
        gamma_min = settings.gamma_min_factor
        computed_mid = _compute_mid_bisection(
            settings.learning_rate, gamma_min, settings.max_iter,
            settings.bisect_lower, settings.bisect_upper,
        )
        from dataclasses import replace as _dc_replace
        settings = _dc_replace(settings, mid=computed_mid)

    rho = settings.ks_rho
    grad_obj_fn = jax.grad(objective_fn, argnums=(0, 1))

    def constraint_pen(x, y):
        return (settings.boundary_weight * boundary_penalty(x, y, boundary, rho)
                + settings.spacing_weight * spacing_penalty(x, y, min_spacing, rho))

    grad_con_fn = jax.grad(constraint_pen, argnums=(0, 1))

    x, y = init_x, init_y
    grad_obj_x, grad_obj_y = grad_obj_fn(x, y)
    state = _init_sgd_state(x, y, grad_obj_x, grad_obj_y, settings)

    snapshots = []
    prev_x, prev_y = x - 1.0, y - 1.0

    for step in range(settings.max_iter):
        change = float(jnp.max(jnp.abs(x - prev_x)) + jnp.max(jnp.abs(y - prev_y)))
        if step > 0 and change < settings.tol:
            break

        if step % snapshot_every == 0:
            obj_val = float(objective_fn(x, y))
            pen_val = float(constraint_pen(x, y))
            snapshots.append({
                "step": step,
                "x": np.array(x),
                "y": np.array(y),
                "obj": obj_val,
                "penalty": pen_val,
            })

        prev_x, prev_y = x, y
        grad_obj_x, grad_obj_y = grad_obj_fn(x, y)
        grad_con_x, grad_con_y = grad_con_fn(x, y)
        x, y, state = _sgd_step(
            x, y, state, grad_obj_x, grad_obj_y, grad_con_x, grad_con_y, settings,
        )

    snapshots.append({
        "step": step,
        "x": np.array(x),
        "y": np.array(y),
        "obj": float(objective_fn(x, y)),
        "penalty": float(constraint_pen(x, y)),
    })

    return np.array(x), np.array(y), snapshots


def make_inner_objective(nb_params):
    n_nb = nb_params.shape[0] // 2
    nb_x, nb_y = nb_params[:n_nb], nb_params[n_nb:]

    def obj(x, y):
        return -compute_aep(x, y, nb_x, nb_y)

    return obj


print("\nCapturing inner SGD at FIRST outer iteration...", flush=True)
t0 = time.time()
first_obj = make_inner_objective(first_nb_params)
_, _, snaps_first = sgd_solve_with_history(
    first_obj, init_x, init_y, boundary, min_spacing, sgd_settings,
)
print(f"  {len(snaps_first)} snapshots, {snaps_first[-1]['step']} steps ({time.time()-t0:.1f}s)", flush=True)

print("Capturing inner SGD at LAST outer iteration...", flush=True)
t0 = time.time()
last_obj = make_inner_objective(last_nb_params)
_, _, snaps_last = sgd_solve_with_history(
    last_obj, init_x, init_y, boundary, min_spacing, sgd_settings,
)
print(f"  {len(snaps_last)} snapshots, {snaps_last[-1]['step']} steps ({time.time()-t0:.1f}s)", flush=True)

# ══════════════════════════════════════════════════════════════════════════
# RENDER 2 & 3: Inner loop animations
# ══════════════════════════════════════════════════════════════════════════


def render_inner_animation(snapshots, nb_params, label, output_path):
    """Render inner SGD animation to MP4 (km coordinates)."""
    nb_x_raw = np.array(nb_params[:N_NEIGHBOR])
    nb_y_raw = np.array(nb_params[N_NEIGHBOR:])
    nb_km_x, nb_km_y = to_km(nb_x_raw, nb_y_raw)
    n_inner_frames = len(snapshots)

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    ax_layout, ax_obj, ax_pen = axes

    obj_vals = [s["obj"] for s in snapshots]
    pen_vals = [s["penalty"] for s in snapshots]
    obj_lo, obj_hi = min(obj_vals) * 1.02, max(obj_vals) * 0.98
    pen_hi = max(max(pen_vals) * 1.1, 0.01)

    all_x = np.concatenate([s["x"] for s in snapshots])
    all_y = np.concatenate([s["y"] for s in snapshots])
    all_km_x, all_km_y = to_km(all_x, all_y)
    pad_inner = 3.0
    x_lo = min(all_km_x.min(), nb_km_x.min(), bnd_km[:, 0].min()) - pad_inner
    x_hi = max(all_km_x.max(), nb_km_x.max(), bnd_km[:, 0].max()) + pad_inner
    y_lo = min(all_km_y.min(), nb_km_y.min(), bnd_km[:, 1].min()) - pad_inner
    y_hi = max(all_km_y.max(), nb_km_y.max(), bnd_km[:, 1].max()) + pad_inner

    def draw(frame_idx):
        for ax in axes:
            ax.clear()

        s = snapshots[frame_idx]
        tx_km, ty_km = to_km(s["x"], s["y"])

        # Layout with polygon
        poly_patch = MplPolygon(bnd_km, closed=True, fill=True,
                                facecolor='lightyellow', edgecolor='black', lw=2)
        ax_layout.add_patch(poly_patch)

        # Turbine trail
        for j in range(0, frame_idx + 1, max(1, frame_idx // 12)):
            sp = snapshots[j]
            skx, sky = to_km(sp["x"], sp["y"])
            alpha_val = 0.05 + 0.3 * (j / max(frame_idx, 1))
            ax_layout.scatter(skx, sky, c="seagreen", marker=".",
                              s=6, alpha=alpha_val, zorder=3)

        ax_layout.scatter(tx_km, ty_km, c="seagreen", marker="s", s=30,
                          edgecolors="black", linewidths=0.5,
                          label="Targets", zorder=6)
        init_km_x, init_km_y = to_km(snapshots[0]["x"], snapshots[0]["y"])
        ax_layout.scatter(init_km_x, init_km_y, c="gray", marker="o", s=20,
                          alpha=0.3, label="Initial", zorder=4)
        ax_layout.scatter(nb_km_x, nb_km_y, c="red", marker="D", s=30,
                          edgecolors="black", linewidths=0.5,
                          label="Neighbors (fixed)", zorder=6)

        ax_layout.set_xlim(x_lo, x_hi)
        ax_layout.set_ylim(y_lo, y_hi)
        ax_layout.set_aspect("equal")
        ax_layout.set_xlabel("x (km)")
        ax_layout.set_ylabel("y (km)")
        ax_layout.set_title("Inner SGD — Layout", fontsize=11)
        ax_layout.legend(fontsize=7, loc="lower left")
        ax_layout.grid(True, alpha=0.2)

        # Objective
        ax_obj.plot(
            [sn["step"] for sn in snapshots[:frame_idx + 1]],
            [sn["obj"] for sn in snapshots[:frame_idx + 1]],
            "o-", color="purple", markersize=2, lw=1.5,
        )
        ax_obj.set_xlim(-5, snapshots[-1]["step"] + 5)
        ax_obj.set_ylim(obj_lo, obj_hi)
        ax_obj.set_xlabel("SGD step")
        ax_obj.set_ylabel("Objective (neg AEP)")
        ax_obj.set_title(f"Obj = {s['obj']:.2f}", fontsize=11)
        ax_obj.grid(True, alpha=0.3)

        # Penalty
        ax_pen.plot(
            [sn["step"] for sn in snapshots[:frame_idx + 1]],
            [sn["penalty"] for sn in snapshots[:frame_idx + 1]],
            "s-", color="orangered", markersize=2, lw=1.5,
        )
        ax_pen.set_xlim(-5, snapshots[-1]["step"] + 5)
        ax_pen.set_ylim(-pen_hi * 0.05, pen_hi)
        ax_pen.set_xlabel("SGD step")
        ax_pen.set_ylabel("Constraint penalty")
        ax_pen.set_title(f"Penalty = {s['penalty']:.4f}", fontsize=11)
        ax_pen.grid(True, alpha=0.3)

        fig.suptitle(
            f"{label} — SGD step {s['step']}/{snapshots[-1]['step']}",
            fontsize=13,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.94])

    anim = FuncAnimation(fig, draw, frames=n_inner_frames, interval=25, repeat=True)
    anim.save(str(output_path), writer="ffmpeg", fps=40, dpi=150)
    plt.close(fig)
    print(f"Saved -> {output_path}", flush=True)


print("\nRendering inner loop animations...", flush=True)
render_inner_animation(
    snaps_first, first_nb_params,
    "First outer iteration (initial neighbors)",
    output_dir / "inner_first.mp4",
)
render_inner_animation(
    snaps_last, last_nb_params,
    "Last outer iteration (optimized neighbors)",
    output_dir / "inner_last.mp4",
)

print("\nAll done!")
print(f"  {output_dir / 'outer_loop.mp4'}")
print(f"  {output_dir / 'inner_first.mp4'}")
print(f"  {output_dir / 'inner_last.mp4'}")

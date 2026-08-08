"""Greedy N-farm regret search.

Places up to N identical copies of the target farm at controlled positions
chosen to maximise design regret on the target. Constraint: every pair of
farms (target included) maintains a minimum boundary-gap distance.

For each step k = 1..N:
    1. Generate candidate positions on a (bearing, distance) grid.
    2. Filter by inter-farm boundary-gap constraint vs already-placed farms.
    3. Two-stage screen-then-evaluate (cheap AEP-loss screen, then full K-start
       conservative re-optimization for the top candidates).
    4. Place the farm with maximum incremental regret; record regret_k.

Outputs:
    results.json with regret_k for k=1..N, placed farm centroids,
    and the sequence of (bearing, distance) selections.
"""

import jax
jax.config.update("jax_enable_x64", True)

import argparse
import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import sys
sys.path.insert(0, "scripts")
from run_regret_cross_section import (
    boundary_np, boundary, D, create_dei_turbine,
    centroid_offset_for_gap, compute_boundary_gap,
    sample_polygon_boundary, generate_target_grid,
    _polygon_path,
)
from pixwake import WakeSimulation
from pixwake.deficit import BastankhahGaussianDeficit
from pixwake.deficit.gaussian import TurboGaussianDeficit
from pixwake.optim.sgd import SGDSettings, topfarm_sgd_solve


def boundary_gap_between(bnd_a, bnd_b, n_per_edge=30):
    return compute_boundary_gap(bnd_a, bnd_b, n_per_edge)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-target", type=int, default=50)
    p.add_argument("--n-max", type=int, default=4, help="Max number of neighboring farms")
    p.add_argument("--bearings", type=int, default=24)
    p.add_argument("--distances-D", type=str, default="2,5,10,15,20,30",
                   help="Comma list of buffer distances (D units) between any farm pair")
    p.add_argument("--inter-farm-gap-D", type=float, default=2.0,
                   help="Min boundary-gap between any two farms (D)")
    p.add_argument("--screen-top-k", type=int, default=8)
    p.add_argument("--n-inner-starts", type=int, default=300)
    p.add_argument("--inner-max-iter", type=int, default=2000)
    p.add_argument("--k-liberal", type=int, default=300)
    p.add_argument("--chunk-size", type=int, default=50)
    p.add_argument("--deficit", type=str, default="bastankhah",
                   choices=["bastankhah", "turbopark"])
    p.add_argument("--ti", type=float, default=0.06)
    p.add_argument("--wind-rose", type=str, default="dei",
                   choices=["dei", "elliptical", "unidirectional"])
    p.add_argument("--ed-a", type=float, default=0.9)
    p.add_argument("--ed-f", type=float, default=1.0)
    p.add_argument("--wind-dir", type=float, default=270.0)
    p.add_argument("--wind-speed", type=float, default=9.0)
    p.add_argument("--n-bins", type=int, default=24)
    p.add_argument("--output-dir", type=str, required=True)
    args = p.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    distances_D = [float(x) for x in args.distances_D.split(",")]
    distances_m = [d * D for d in distances_D]
    bearings = np.linspace(0, 360, args.bearings, endpoint=False)
    inter_gap_m = args.inter_farm_gap_D * D

    # ---- wind rose ----
    if args.wind_rose == "dei":
        import pandas as pd
        df = pd.read_csv("energy_island_10y_daily_av_wind.csv", sep=";")
        wd_ts, ws_ts = df["WD_150"].values, df["WS_150"].values
        edges = np.linspace(0, 360, args.n_bins + 1)
        centers = (edges[:-1] + edges[1:]) / 2
        w = np.zeros(args.n_bins); means = np.zeros(args.n_bins)
        for i in range(args.n_bins):
            if i == args.n_bins - 1:
                mask = (wd_ts >= edges[i]) | (wd_ts < edges[0])
            else:
                mask = (wd_ts >= edges[i]) & (wd_ts < edges[i+1])
            w[i] = mask.sum()
            means[i] = ws_ts[mask].mean() if mask.sum() > 0 else ws_ts.mean()
        w /= w.sum()
        wd_arr, ws_arr, weights = jnp.array(centers), jnp.array(means), jnp.array(w)
    elif args.wind_rose == "elliptical":
        from edrose import EllipticalWindRose
        wr = EllipticalWindRose(a=args.ed_a, f=args.ed_f,
                                theta_prev=args.wind_dir, n_sectors=args.n_bins)
        wd_arr = jnp.array(wr.wind_directions)
        ws_arr = jnp.full((args.n_bins,), args.wind_speed)
        weights = jnp.array(wr.sector_frequencies)
    else:
        wd_arr = jnp.array([args.wind_dir])
        ws_arr = jnp.array([args.wind_speed])
        weights = jnp.array([1.0])

    turbine = create_dei_turbine()
    if args.deficit == "bastankhah":
        deficit = BastankhahGaussianDeficit(k=0.04)
    else:
        deficit = TurboGaussianDeficit(A=0.04)
    sim = WakeSimulation(turbine, deficit)
    ti_amb = args.ti if args.deficit == "turbopark" else None

    # ---- generate liberal layout ----
    print("Generating initial target grid...")
    init_x, init_y = generate_target_grid(boundary_np, args.n_target, spacing=4 * D)
    init_x_j, init_y_j = jnp.array(init_x), jnp.array(init_y)

    sgd_settings = SGDSettings(
        learning_rate=2.0,
        max_iter=args.inner_max_iter,
        additional_constant_lr_iterations=args.inner_max_iter,
        tol=1e-6,
    )
    from dataclasses import replace as _dc_replace
    from pixwake.optim.sgd import _compute_mid_bisection
    if sgd_settings.mid is None:
        _mid = _compute_mid_bisection(
            learning_rate=sgd_settings.learning_rate,
            gamma_min=sgd_settings.gamma_min_factor,
            max_iter=sgd_settings.max_iter,
            lower=sgd_settings.bisect_lower,
            upper=sgd_settings.bisect_upper,
        )
        sgd_settings = _dc_replace(sgd_settings, mid=_mid)

    def aep_only(x_all, y_all, n_target):
        res = sim(x_all, y_all, ws_amb=ws_arr, wd_amb=wd_arr, ti_amb=ti_amb)
        p_kw = res.power()
        tgt_p = jnp.sum(p_kw[:, :n_target] * weights[:, None], axis=0)
        return 8760.0 * jnp.sum(tgt_p) / 1e6

    def liberal_objective(x, y):
        return -aep_only(x, y, args.n_target)

    print(f"Liberal multistart with K={args.k_liberal}...")
    best_lib_aep = -np.inf
    best_lib_x, best_lib_y = init_x_j, init_y_j
    for k in range(args.k_liberal):
        key = jax.random.PRNGKey(k * 7919 + 42)
        if k == 0:
            sx, sy = init_x_j, init_y_j
        else:
            sx = jax.random.uniform(key, (args.n_target,),
                                     minval=boundary_np[:, 0].min(),
                                     maxval=boundary_np[:, 0].max())
            sy = jax.random.uniform(jax.random.PRNGKey(k * 7919 + 43),
                                     (args.n_target,),
                                     minval=boundary_np[:, 1].min(),
                                     maxval=boundary_np[:, 1].max())
        lx, ly = topfarm_sgd_solve(liberal_objective, sx, sy, boundary, D * 4, sgd_settings)
        lib_aep = float(aep_only(lx, ly, args.n_target))
        if lib_aep > best_lib_aep:
            best_lib_aep = lib_aep
            best_lib_x, best_lib_y = lx, ly
    liberal_x, liberal_y = best_lib_x, best_lib_y
    print(f"Liberal AEP = {best_lib_aep:.2f} GWh (best of {args.k_liberal})")

    # ---- pre-compute boundary-gap offsets for every (bearing, distance) ----
    print("Pre-computing offsets...")
    offset_table = {}
    for di, gap_m in enumerate(distances_m):
        for bi, bear in enumerate(bearings):
            off, dirn, _ = centroid_offset_for_gap(boundary_np, bear, gap_m)
            offset_table[(di, bi)] = (off, dirn)

    # placed farms = list of (centroid_offset_x, centroid_offset_y, ref_x, ref_y)
    placed = []  # each entry: dict(off_x, off_y, ref_x, ref_y, ref_bnd)
    history = []

    target_bnd = boundary_np.copy()  # target sits at origin (centred)

    def farm_pair_gap(off_a, off_b):
        bnd_a = boundary_np + np.array(off_a)
        bnd_b = boundary_np + np.array(off_b)
        return boundary_gap_between(bnd_a, bnd_b)

    def baseline_with_placed():
        if not placed:
            return best_lib_aep
        nx_all = []; ny_all = []
        for f in placed:
            nx_all.append(np.array(f["ref_x"]))
            ny_all.append(np.array(f["ref_y"]))
        nx = jnp.array(np.concatenate(nx_all))
        ny = jnp.array(np.concatenate(ny_all))
        x_all = jnp.concatenate([liberal_x, nx])
        y_all = jnp.concatenate([liberal_y, ny])
        return float(aep_only(x_all, y_all, args.n_target))

    # ---- greedy loop ----
    for step in range(args.n_max):
        baseline = baseline_with_placed()
        print(f"\n=== Step {step+1}/{args.n_max} (baseline AEP w/ {step} placed = {baseline:.2f} GWh) ===")

        # Build candidate positions, filter by inter-farm gap
        cand_positions = []
        for di, gap_m in enumerate(distances_m):
            for bi, bear in enumerate(bearings):
                off, dirn = offset_table[(di, bi)]
                cand_off = np.array([off * dirn[0], off * dirn[1]])
                ok = True
                for f in placed:
                    other_off = np.array([f["off_x"], f["off_y"]])
                    if farm_pair_gap(cand_off, other_off) < inter_gap_m:
                        ok = False; break
                if ok:
                    cand_positions.append((di, bi, distances_D[di], bear, cand_off))
        print(f"  {len(cand_positions)} candidates pass inter-farm gap")
        if not cand_positions:
            print("  No valid candidates; stopping.")
            break

        # Stage 1: screen by AEP loss (fast forward sim, no re-opt)
        existing_x = jnp.array(np.concatenate([np.array(f["ref_x"]) for f in placed])) if placed else jnp.array([])
        existing_y = jnp.array(np.concatenate([np.array(f["ref_y"]) for f in placed])) if placed else jnp.array([])

        def screen_one(off_x, off_y):
            new_x = jnp.array(np.array(liberal_x)) + 0.0  # not used; just want template
            ref_x = liberal_x + off_x
            ref_y = liberal_y + off_y
            nb_x = jnp.concatenate([existing_x, ref_x]) if placed else ref_x
            nb_y = jnp.concatenate([existing_y, ref_y]) if placed else ref_y
            x_all = jnp.concatenate([liberal_x, nb_x])
            y_all = jnp.concatenate([liberal_y, nb_y])
            return baseline - aep_only(x_all, y_all, args.n_target)

        screen_batch = jax.jit(jax.vmap(screen_one))
        offs = jnp.array([c[4] for c in cand_positions])
        losses = np.zeros(len(cand_positions))
        CHUNK = args.chunk_size
        for i0 in range(0, len(cand_positions), CHUNK):
            i1 = min(i0 + CHUNK, len(cand_positions))
            losses[i0:i1] = np.array(screen_batch(offs[i0:i1, 0], offs[i0:i1, 1]))
        top_k = np.argsort(-losses)[:args.screen_top_k]
        print(f"  screen top loss = {losses[top_k[0]]:.2f} GWh")

        # Stage 2: full conservative re-opt for each top candidate
        def conservative_objective(x, y, neighbor_x, neighbor_y):
            x_all = jnp.concatenate([x, neighbor_x])
            y_all = jnp.concatenate([y, neighbor_y])
            return -aep_only(x_all, y_all, args.n_target)

        best_regret = -np.inf
        best_choice = None
        for ci in top_k:
            di, bi, dval, bear, cand_off = cand_positions[ci]
            ref_x_new = liberal_x + cand_off[0]
            ref_y_new = liberal_y + cand_off[1]
            full_nb_x = jnp.concatenate([existing_x, ref_x_new]) if placed else ref_x_new
            full_nb_y = jnp.concatenate([existing_y, ref_y_new]) if placed else ref_y_new
            lib_with_neighbors = float(aep_only(
                jnp.concatenate([liberal_x, full_nb_x]),
                jnp.concatenate([liberal_y, full_nb_y]),
                args.n_target,
            ))
            # Multistart conservative: K starts
            best_cons_aep = lib_with_neighbors
            best_cx, best_cy = liberal_x, liberal_y
            K = args.n_inner_starts
            for k in range(K):
                key = jax.random.PRNGKey(k * 7919 + 42)
                if k == 0:
                    sx, sy = init_x_j, init_y_j
                elif k == 1:
                    sx, sy = liberal_x, liberal_y
                else:
                    sx = jax.random.uniform(key, (args.n_target,),
                                             minval=boundary_np[:, 0].min(),
                                             maxval=boundary_np[:, 0].max())
                    sy = jax.random.uniform(jax.random.PRNGKey(k * 7919 + 43),
                                             (args.n_target,),
                                             minval=boundary_np[:, 1].min(),
                                             maxval=boundary_np[:, 1].max())
                cx, cy = topfarm_sgd_solve(
                    lambda x, y: conservative_objective(x, y, full_nb_x, full_nb_y),
                    sx, sy, boundary, D * 4, sgd_settings)
                cons_aep = float(aep_only(
                    jnp.concatenate([cx, full_nb_x]),
                    jnp.concatenate([cy, full_nb_y]),
                    args.n_target))
                if cons_aep > best_cons_aep:
                    best_cons_aep = cons_aep
                    best_cx, best_cy = cx, cy
            regret = max(best_cons_aep, lib_with_neighbors) - lib_with_neighbors
            cumulative_regret = max(best_cons_aep, lib_with_neighbors) - best_lib_aep
            if regret > best_regret:
                best_regret = regret
                best_choice = (di, bi, dval, bear, cand_off, ref_x_new, ref_y_new,
                               regret, cumulative_regret)

        di, bi, dval, bear, cand_off, ref_x_new, ref_y_new, reg, cum_reg = best_choice
        print(f"  PLACE farm {step+1} at bearing {bear:.0f}°, distance {dval:.0f}D, "
              f"incremental regret = {reg:.2f} GWh, cumulative regret vs liberal-no-nbr = {cum_reg:.2f} GWh")
        placed.append({
            "off_x": float(cand_off[0]), "off_y": float(cand_off[1]),
            "bearing_deg": float(bear), "distance_D": float(dval),
            "ref_x": [float(v) for v in np.array(ref_x_new)],
            "ref_y": [float(v) for v in np.array(ref_y_new)],
        })
        history.append({
            "step": step + 1,
            "incremental_regret_gwh": float(reg),
            "cumulative_regret_vs_no_nbr_gwh": float(cum_reg),
            "bearing_deg": float(bear),
            "distance_D": float(dval),
        })

    out = {
        "n_target": args.n_target,
        "n_placed": len(placed),
        "liberal_aep_gwh": best_lib_aep,
        "liberal_x": [float(v) for v in np.array(liberal_x)],
        "liberal_y": [float(v) for v in np.array(liberal_y)],
        "placed_farms": placed,
        "history": history,
        "config": vars(args),
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_dir/'results.json'}")


if __name__ == "__main__":
    main()

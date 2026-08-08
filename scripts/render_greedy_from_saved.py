"""Render greedy-grid animation from saved regret_maps.npz + results.json.
No AEP recomputation; just plays back the saved heatmaps."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.path import Path as MplPath
from scipy.spatial import ConvexHull

D = 240.0

_dk0w_raw = np.array([
    706694.3923283464, 6224158.532895836,
    703972.0844905999, 6226906.597455995,
    702624.6334635273, 6253853.5386425415,
    712771.6248419734, 6257704.934445341,
    715639.3355871611, 6260664.6846508905,
    721593.2420745814, 6257906.998015941,
]).reshape((-1, 2))
CENT_X, CENT_Y = _dk0w_raw[:, 0].mean(), _dk0w_raw[:, 1].mean()
_hull = ConvexHull(_dk0w_raw - np.array([CENT_X, CENT_Y]))
boundary_np = (_dk0w_raw - np.array([CENT_X, CENT_Y]))[_hull.vertices]


def build_grid(spacing, pad, buffer):
    from shapely.geometry import Polygon
    x_lo = boundary_np[:, 0].min() - pad
    x_hi = boundary_np[:, 0].max() + pad
    y_lo = boundary_np[:, 1].min() - pad
    y_hi = boundary_np[:, 1].max() + pad
    gx = np.arange(x_lo, x_hi, spacing)
    gy = np.arange(y_lo, y_hi, spacing)
    gx2, gy2 = np.meshgrid(gx, gy)
    cand = np.column_stack([gx2.ravel(), gy2.ravel()])
    poly = Polygon(boundary_np)
    excl = MplPath(np.column_stack(poly.buffer(buffer).exterior.xy))
    outside = ~excl.contains_points(cand)
    return cand[outside], gx, gy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True)
    p.add_argument("--grid-pad-D", type=float, default=200.0)
    p.add_argument("--buffer-D", type=float, default=2.0,
                   help="Grid construction buffer (must match the run that produced regret_maps)")
    p.add_argument("--show-buffer-D", type=float, default=30.0,
                   help="Visual exclusion: only color candidates outside this distance from boundary")
    p.add_argument("--out", type=str, default=None)
    # Wind rose for inset
    p.add_argument("--wind-rose", type=str, default="dei",
                   choices=["dei", "elliptical", "mixture", "unidirectional", "uniform"])
    p.add_argument("--ed-a", type=float, default=0.9)
    p.add_argument("--ed-f", type=float, default=1.0)
    p.add_argument("--wind-dir", type=float, default=270.0)
    p.add_argument("--n-bins", type=int, default=24)
    args = p.parse_args()
    rdir = Path(args.results_dir)
    res = json.load(open(rdir / "results.json"))
    rm = np.load(rdir / "regret_maps.npz")

    grid, gx_1d, gy_1d = build_grid(5 * D, args.grid_pad_D * D, args.buffer_D * D)

    # Visual exclusion: build mask of candidates outside the show-buffer.
    from shapely.geometry import Polygon
    show_excl = Polygon(boundary_np).buffer(args.show_buffer_D * D)
    show_excl_pts = np.column_stack(show_excl.exterior.xy)
    show_path = MplPath(show_excl_pts)
    visible_mask = ~show_path.contains_points(grid)

    placement_order = res["placement_order"]
    nbr_x = np.array(res["neighbor_x"])
    nbr_y = np.array(res["neighbor_y"])
    lib_aep = res["liberal_aep_gwh"]
    n_steps = len(placement_order)

    # Compute regret_history for display
    regret_hist = res.get("regret_history", [])

    # Liberal layout (target) — saved? Try config
    # If not in results, use a target grid heuristic
    lib_x = nbr_x  # placeholder; we'll just plot neighbors

    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_axes([0.05, 0.08, 0.55, 0.85])
    ax_rose = fig.add_axes([0.66, 0.55, 0.18, 0.35], projection="polar")
    ax_curve = fig.add_axes([0.66, 0.10, 0.32, 0.35])
    ax_info = fig.add_axes([0.85, 0.55, 0.13, 0.35])
    ax_info.axis("off")

    # Wind rose inset
    def get_rose():
        if args.wind_rose == "dei":
            import pandas as pd
            df = pd.read_csv("energy_island_10y_daily_av_wind.csv", sep=";")
            wd_ts = df["WD_150"].values
            n = args.n_bins
            edges = np.linspace(0, 360, n + 1)
            centers = (edges[:-1] + edges[1:]) / 2
            w = np.zeros(n)
            for i in range(n):
                if i == n - 1:
                    mask = (wd_ts >= edges[i]) | (wd_ts < edges[0])
                else:
                    mask = (wd_ts >= edges[i]) & (wd_ts < edges[i + 1])
                w[i] = mask.sum()
            return centers, w / w.sum()
        elif args.wind_rose == "elliptical":
            from edrose import EllipticalWindRose
            wr = EllipticalWindRose(a=args.ed_a, f=args.ed_f,
                                    theta_prev=args.wind_dir, n_sectors=args.n_bins)
            return np.array(wr.wind_directions), np.array(wr.sector_frequencies)
        elif args.wind_rose == "unidirectional":
            return np.array([args.wind_dir]), np.array([1.0])
        else:
            return np.linspace(0, 360, args.n_bins, endpoint=False), np.full(args.n_bins, 1.0/args.n_bins)

    rose_dirs, rose_freq = get_rose()
    ax_rose.set_theta_zero_location("N"); ax_rose.set_theta_direction(-1)
    width = np.deg2rad(min(360 / max(len(rose_dirs), 1), 15))
    ax_rose.bar(np.deg2rad(rose_dirs), rose_freq, width=width,
                color="steelblue", alpha=0.85, edgecolor="navy", linewidth=0.3)
    ax_rose.set_yticks([]); ax_rose.set_xticks([])
    ax_rose.set_title("Wind rose", fontsize=9, pad=4)

    # Domain
    ax.set_aspect("equal")
    poly = MplPolygon(boundary_np / 1000, fill=False, edgecolor="black", lw=1.5, zorder=5)
    ax.add_patch(poly)
    ax.add_patch(MplPolygon(show_excl_pts / 1000, fill=False, edgecolor="gray",
                            linestyle="--", lw=0.8, zorder=4,
                            label=f"{args.show_buffer_D:.0f}$D$ exclusion"))
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")

    # Determine grid extent
    gx_km = gx_1d / 1000
    gy_km = gy_1d / 1000
    ax.set_xlim(gx_km[0], gx_km[-1])
    ax.set_ylim(gy_km[0], gy_km[-1])

    # Per-frame: subtract per-step baseline so variation across candidates is visible.
    # Each step's regret_map = AEP loss vs liberal-no-neighbor (total including all placed
    # so far). We subtract per-frame minimum to show INCREMENTAL loss per candidate.
    # Initial scatter (visible candidates only)
    vis_grid = grid[visible_mask]
    sc = ax.scatter(vis_grid[:, 0] / 1000, vis_grid[:, 1] / 1000,
                    c=np.zeros(len(vis_grid)),
                    cmap="YlOrRd", s=25, marker="s", edgecolor="none", zorder=3)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02,
                        label="Incremental AEP loss if placed here (GWh)")
    placed_scatter = ax.scatter([], [], s=100, c="crimson", marker="X",
                                  edgecolor="black", linewidth=1.0, zorder=7)
    title = ax.set_title("")

    # Curve (regret as % of liberal AEP)
    regret_pct = [100 * r / lib_aep for r in regret_hist] if regret_hist else []
    ax_curve.set_xlabel("Step")
    ax_curve.set_ylabel("Cumulative regret (\\% of liberal AEP)")
    ax_curve.set_xlim(0, n_steps)
    ax_curve.set_ylim(0, max(regret_pct) * 1.1 if regret_pct else 1)
    curve, = ax_curve.plot([], [], "o-", color="firebrick", lw=2, ms=5)
    ax_curve.grid(True, alpha=0.3)

    def draw(step):
        m = rm[f"step_{step}"]
        m_vis = m[visible_mask]
        # Subtract minimum (= baseline w/ placed) for per-candidate incremental loss
        valid = ~np.isnan(m_vis)
        if valid.any():
            base = float(np.nanmin(m_vis[valid]))
            m_disp = np.where(valid, m_vis - base, 0)
        else:
            m_disp = np.zeros_like(m_vis)
        sc.set_array(m_disp)
        sc.set_clim(vmin=0, vmax=max(float(m_disp.max()), 1e-3))
        # Placed up to step
        if step > 0:
            px = nbr_x[:step] / 1000
            py = nbr_y[:step] / 1000
        else:
            px, py = [], []
        placed_scatter.set_offsets(np.column_stack([px, py]) if len(px) else np.empty((0, 2)))
        title.set_text(f"Greedy step {step+1}/{n_steps}")
        if regret_pct and step < len(regret_pct):
            curve.set_data(range(1, step + 2), regret_pct[:step + 1])
        ax_info.clear(); ax_info.axis("off")
        ax_info.text(0.0, 0.95, f"Liberal AEP: {lib_aep:.1f} GWh", fontsize=10)
        ax_info.text(0.0, 0.85, f"Step: {step+1}/{n_steps}", fontsize=10)
        if regret_pct and step < len(regret_pct):
            ax_info.text(0.0, 0.75, f"Cumulative regret: {regret_pct[step]:.3f}\\% of AEP", fontsize=10)
        ax_info.text(0.0, 0.65, f"Neighbors placed: {step+1}", fontsize=10)

    anim = FuncAnimation(fig, draw, frames=n_steps, interval=400, repeat=True)
    out = args.out or str(rdir / "dei_greedy_grid.mp4")
    anim.save(out, writer="ffmpeg", fps=3, dpi=130)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

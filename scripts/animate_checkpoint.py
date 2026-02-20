"""Render an animation from the latest bilevel optimization checkpoint.

Reads analysis/dei_ift_bilevel/checkpoint.npz (written every 10 iterations
by run_dei_ift_bilevel.py) and produces an MP4 showing the optimization
progress so far.

Can be run while the optimization is still running.

Usage:
    pixi run python scripts/animate_checkpoint.py
    pixi run python scripts/animate_checkpoint.py --fps=10 --subsample=2
    pixi run python scripts/animate_checkpoint.py --input=analysis/dei_ift_bilevel/checkpoint.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Polygon as MplPolygon

sys.stdout.reconfigure(line_buffering=True)


def main():
    parser = argparse.ArgumentParser(
        description="Animate bilevel optimization from checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=str,
                        default="analysis/dei_ift_bilevel/checkpoint.npz")
    parser.add_argument("--output", type=str, default=None,
                        help="Output MP4 path (default: next to checkpoint)")
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--subsample", type=int, default=1,
                        help="Use every Nth iteration as a frame")
    parser.add_argument("--dpi", type=int, default=120)
    args = parser.parse_args()

    # Load checkpoint
    ckpt_path = Path(args.input)
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        print("Is run_dei_ift_bilevel.py running?")
        return

    ckpt = np.load(ckpt_path, allow_pickle=True)
    regret = ckpt["regret"]
    grad_norm = ckpt["grad_norm"]
    nb_x = ckpt["nb_x"]       # (n_iters, n_neighbor)
    nb_y = ckpt["nb_y"]
    tgt_x = ckpt["tgt_x"]     # (n_iters, n_target)
    tgt_y = ckpt["tgt_y"]
    liberal_x = ckpt["liberal_x"]
    liberal_y = ckpt["liberal_y"]
    liberal_aep = float(ckpt["liberal_aep"])
    boundary = ckpt["boundary"]
    n_target = int(ckpt["n_target"])
    n_neighbor = int(ckpt["n_neighbor"])

    # New fields (with fallbacks for old checkpoints)
    con_pen = ckpt["con_pen"] if "con_pen" in ckpt else None
    objective = ckpt["objective"] if "objective" in ckpt else None
    min_spacing = float(ckpt["min_spacing"]) if "min_spacing" in ckpt else 960.0
    half_spacing_km = (min_spacing / 2.0) / 1000.0  # radius in km
    buffer = float(ckpt["buffer"]) if "buffer" in ckpt else 3.0 * 240.0

    n_iters = len(regret)
    print(f"Loaded checkpoint: {n_iters} iterations")
    print(f"  Regret range: {regret.min():.2f} — {regret.max():.2f} GWh")
    print(f"  Best regret: {regret.max():.2f} GWh ({regret.max()/liberal_aep*100:.2f}%)")
    print(f"  {n_target} targets, {n_neighbor} neighbors")
    print(f"  Min spacing: {min_spacing:.0f} m (circle radius: {min_spacing/2:.0f} m)")
    has_penalty = con_pen is not None and objective is not None
    if has_penalty:
        print(f"  Constraint penalty range: {con_pen.min():.2f} — {con_pen.max():.2f}")

    # Subsample frames
    frame_indices = np.arange(0, n_iters, args.subsample)
    if frame_indices[-1] != n_iters - 1:
        frame_indices = np.append(frame_indices, n_iters - 1)
    n_frames = len(frame_indices)
    print(f"  Rendering {n_frames} frames (subsample={args.subsample})")

    # Coordinate transform to km relative to boundary centroid
    cx = boundary[:, 0].mean()
    cy = boundary[:, 1].mean()
    def to_km(x, y):
        return (np.asarray(x) - cx) / 1000., (np.asarray(y) - cy) / 1000.

    bnd_km = np.column_stack(to_km(boundary[:, 0], boundary[:, 1]))
    lib_km_x, lib_km_y = to_km(liberal_x, liberal_y)

    # Compute buffer polygon (offset boundary outward by buffer distance)
    def offset_convex_polygon(verts, dist):
        """Offset a convex polygon outward by dist. Returns new vertices."""
        n = len(verts)
        centroid = verts.mean(axis=0)
        offset_edges = []
        for j in range(n):
            p0 = verts[j]
            p1 = verts[(j + 1) % n]
            edge = p1 - p0
            normal = np.array([-edge[1], edge[0]])
            normal /= np.linalg.norm(normal)
            if np.dot(normal, p0 - centroid) < 0:
                normal = -normal
            offset_edges.append((p0 + normal * dist, p1 + normal * dist))
        new_verts = []
        for j in range(n):
            # Intersect edge j with edge (j+1)
            a0, a1 = offset_edges[j]
            b0, b1 = offset_edges[(j + 1) % n]
            da = a1 - a0
            db = b1 - b0
            denom = da[0] * db[1] - da[1] * db[0]
            if abs(denom) < 1e-12:
                new_verts.append(a1)  # parallel edges, use endpoint
            else:
                t = ((b0[0] - a0[0]) * db[1] - (b0[1] - a0[1]) * db[0]) / denom
                new_verts.append(a0 + t * da)
        return np.array(new_verts)

    buffer_poly = offset_convex_polygon(boundary, buffer)
    buffer_km = np.column_stack(to_km(buffer_poly[:, 0], buffer_poly[:, 1]))

    # Compute axis limits from all data
    all_nb_km_x, all_nb_km_y = to_km(nb_x.ravel(), nb_y.ravel())
    pad = 3.0
    x_lo = min(bnd_km[:, 0].min(), all_nb_km_x.min()) - pad
    x_hi = max(bnd_km[:, 0].max(), all_nb_km_x.max()) + pad
    y_lo = min(bnd_km[:, 1].min(), all_nb_km_y.min()) - pad
    y_hi = max(bnd_km[:, 1].max(), all_nb_km_y.max()) + pad

    # Plot limits
    regret_lo = regret.min() * 0.9
    regret_hi = regret.max() * 1.1
    grad_lo = max(grad_norm.min() * 0.5, 1e-8)
    grad_hi = grad_norm.max() * 2.0
    cons_aeps = liberal_aep - regret
    aep_lo = cons_aeps.min() * 0.999
    aep_hi = liberal_aep * 1.001

    if has_penalty:
        pen_lo = 0
        pen_hi = con_pen.max() * 1.1 if con_pen.max() > 0 else 1.0
        obj_lo = objective.min() - abs(objective.min()) * 0.1
        obj_hi = objective.max() + abs(objective.max()) * 0.1

    # Precompute neighbor trail in km
    nb_trail_km_x = (nb_x - cx) / 1000.
    nb_trail_km_y = (nb_y - cy) / 1000.

    # --- Build animation ---
    # 3x2 grid: layout (tall, spans 2 rows), then 5 metric panels
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.2, 1], hspace=0.35, wspace=0.25)

    # Layout panel spans all 3 rows on the left
    ax_layout = fig.add_subplot(gs[:, 0])
    # Right column: 3 stacked panels
    ax_regret = fig.add_subplot(gs[0, 1])
    ax_pen = fig.add_subplot(gs[1, 1])
    ax_obj = fig.add_subplot(gs[2, 1])

    right_axes = [ax_regret, ax_pen, ax_obj]

    def draw_frame(fi):
        ax_layout.clear()
        for ax in right_axes:
            ax.clear()

        i = frame_indices[fi]

        # ---- Layout panel ----
        poly = MplPolygon(bnd_km, closed=True, fill=True,
                          facecolor='lightyellow', edgecolor='black', lw=2)
        ax_layout.add_patch(poly)

        # Buffer zone (dashed outline)
        buf_poly = MplPolygon(buffer_km, closed=True, fill=False,
                              edgecolor='black', lw=1.5, ls='--', alpha=0.5,
                              label=f"Buffer ({buffer/240:.0f}D)", zorder=1)
        ax_layout.add_patch(buf_poly)

        # Liberal turbines with spacing circles
        ax_layout.scatter(lib_km_x, lib_km_y, c="royalblue", marker="^",
                          s=60, label="Liberal", zorder=5, alpha=0.4)
        for k in range(n_target):
            circ = Circle((lib_km_x[k], lib_km_y[k]), half_spacing_km,
                          fill=False, edgecolor="royalblue", lw=0.5,
                          alpha=0.2, ls="--", zorder=2)
            ax_layout.add_patch(circ)

        # Conservative (optimized) turbines with spacing circles
        opt_km_x, opt_km_y = to_km(tgt_x[i], tgt_y[i])
        ax_layout.scatter(opt_km_x, opt_km_y, c="seagreen", marker="s", s=60,
                          label="Conservative", zorder=5)
        for k in range(n_target):
            circ = Circle((opt_km_x[k], opt_km_y[k]), half_spacing_km,
                          fill=False, edgecolor="seagreen", lw=0.6,
                          alpha=0.35, zorder=4)
            ax_layout.add_patch(circ)

        # Displacement arrows
        for k in range(n_target):
            dxk = opt_km_x[k] - lib_km_x[k]
            dyk = opt_km_y[k] - lib_km_y[k]
            if np.sqrt(dxk**2 + dyk**2) > 0.01:
                ax_layout.annotate(
                    "", xy=(lib_km_x[k]+dxk*3, lib_km_y[k]+dyk*3),
                    xytext=(lib_km_x[k], lib_km_y[k]),
                    arrowprops=dict(arrowstyle="-|>", color="seagreen",
                                    lw=1.5, alpha=0.7), zorder=4)

        # Neighbor trail
        trail_step = max(1, i // 30)
        for j in range(0, i + 1, trail_step):
            alpha_val = 0.05 + 0.6 * (j / max(i, 1))
            ax_layout.scatter(nb_trail_km_x[j], nb_trail_km_y[j],
                              c="red", marker="x", s=12, alpha=alpha_val, zorder=3)

        # Current neighbors with spacing circles
        cur_nb_km_x, cur_nb_km_y = to_km(nb_x[i], nb_y[i])
        ax_layout.scatter(cur_nb_km_x, cur_nb_km_y, c="red", marker="D", s=80,
                          edgecolors="black", linewidths=0.8, label="Neighbors", zorder=6)
        for k in range(n_neighbor):
            circ = Circle((cur_nb_km_x[k], cur_nb_km_y[k]), half_spacing_km,
                          fill=False, edgecolor="red", lw=0.6,
                          alpha=0.35, zorder=4)
            ax_layout.add_patch(circ)

        ax_layout.set_xlim(x_lo, x_hi)
        ax_layout.set_ylim(y_lo, y_hi)
        ax_layout.set_xlabel("x (km)", fontsize=11)
        ax_layout.set_ylabel("y (km)", fontsize=11)
        ax_layout.set_title(f"Layout — iter {i}  |  "
                            f"Regret = {regret[i]:.2f} GWh ({regret[i]/liberal_aep*100:.2f}%)",
                            fontsize=12, fontweight="bold")
        ax_layout.legend(loc="upper right", fontsize=8)
        ax_layout.set_aspect("equal")

        # ---- Regret panel ----
        ax_regret.plot(range(i+1), regret[:i+1], "b-", lw=1.5)
        ax_regret.set_xlim(-0.5, n_iters - 0.5)
        ax_regret.set_ylim(regret_lo, regret_hi)
        ax_regret.set_ylabel("Regret (GWh)", fontsize=10)
        ax_regret.set_title("Design Regret", fontsize=11, fontweight="bold")
        ax_regret.grid(True, alpha=0.3)
        ax_regret.tick_params(labelbottom=False)

        # ---- Constraint penalty panel ----
        if has_penalty:
            ax_pen.semilogy(range(i+1), np.maximum(con_pen[:i+1], 1e-6),
                            "darkorange", lw=1.5)
            ax_pen.set_xlim(-0.5, n_iters - 0.5)
            ax_pen.set_ylabel("Constraint Penalty", fontsize=10)
            ax_pen.set_title("Outer Constraint Penalty (spacing + boundary)",
                             fontsize=11, fontweight="bold")
            ax_pen.grid(True, alpha=0.3)
            ax_pen.tick_params(labelbottom=False)
        else:
            ax_pen.text(0.5, 0.5, "No penalty data\n(old checkpoint)",
                        ha="center", va="center", transform=ax_pen.transAxes,
                        fontsize=12, color="gray")
            ax_pen.tick_params(labelbottom=False)

        # ---- Objective panel ----
        if has_penalty:
            ax_obj.plot(range(i+1), objective[:i+1], "purple", lw=1.5)
            ax_obj.set_xlim(-0.5, n_iters - 0.5)
            ax_obj.set_xlabel("Outer Iteration", fontsize=10)
            ax_obj.set_ylabel("Objective", fontsize=10)
            ax_obj.set_title(r"Outer Objective  $-\mathrm{regret} + \alpha \cdot \mathrm{penalty}$",
                             fontsize=11, fontweight="bold")
            ax_obj.grid(True, alpha=0.3)
        else:
            ax_obj.text(0.5, 0.5, "No objective data\n(old checkpoint)",
                        ha="center", va="center", transform=ax_obj.transAxes,
                        fontsize=12, color="gray")
            ax_obj.set_xlabel("Outer Iteration", fontsize=10)

        fig.suptitle(
            f"IFT Bilevel Optimization: {n_target} targets, {n_neighbor} neighbors  |  "
            f"Iter {i}/{n_iters-1}",
            fontsize=14, fontweight="bold", y=0.98,
        )

    # Determine output path
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = ckpt_path.parent / "progress.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Rendering {n_frames} frames to {out_path}...")
    anim = FuncAnimation(fig, draw_frame, frames=n_frames, interval=200, repeat=True)
    anim.save(str(out_path), writer="ffmpeg", fps=args.fps, dpi=args.dpi)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

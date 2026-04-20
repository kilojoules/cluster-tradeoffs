"""Schematic figure: regret cross-section methodology with boundary-gap distance.

Shows the target farm polygon, identical reference farm copies placed at
different bearings using the boundary-gap distance metric, and annotated
distance measurement between polygon edges.
"""

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, FancyArrowPatch
from matplotlib.projections.polar import PolarAxes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pathlib import Path
from scipy.spatial.distance import cdist
import json

D = 240.0

# DEI polygon
_dk0w_raw = np.array([
    706694.3923283464, 6224158.532895836,
    703972.0844905999, 6226906.597455995,
    702624.6334635273, 6253853.5386425415,
    712771.6248419734, 6257704.934445341,
    715639.3355871611, 6260664.6846508905,
    721593.2420745814, 6257906.998015941,
]).reshape((-1, 2))
cx, cy = _dk0w_raw[:, 0].mean(), _dk0w_raw[:, 1].mean()
boundary_np = _dk0w_raw - np.array([cx, cy])

def to_km(x, y):
    return np.asarray(x) / 1000, np.asarray(y) / 1000

bnd_km = np.column_stack(to_km(boundary_np[:, 0], boundary_np[:, 1]))

# Target turbine positions (simple grid for illustration)
from matplotlib.path import Path as MplPath
poly_path = MplPath(bnd_km)
tx = np.arange(bnd_km[:, 0].min() + 1, bnd_km[:, 0].max() - 1, 4 * D / 1000)
ty = np.arange(bnd_km[:, 1].min() + 1, bnd_km[:, 1].max() - 1, 4 * D / 1000)
tgx, tgy = np.meshgrid(tx, ty)
cands = np.column_stack([tgx.ravel(), tgy.ravel()])
inside = poly_path.contains_points(cands)
target_x, target_y = cands[inside, 0], cands[inside, 1]
if len(target_x) > 50:
    idx = np.round(np.linspace(0, len(target_x) - 1, 50)).astype(int)
    target_x, target_y = target_x[idx], target_y[idx]


def sample_polygon_boundary(vertices, n_per_edge=50):
    pts = []
    n = len(vertices)
    for i in range(n):
        j = (i + 1) % n
        t = np.linspace(0, 1, n_per_edge, endpoint=False)
        edge_pts = vertices[i] + t[:, None] * (vertices[j] - vertices[i])
        pts.append(edge_pts)
    return np.vstack(pts)


def centroid_offset_for_gap(boundary_km, bearing_deg, gap_km, n_per_edge=50):
    """Compute centroid offset to achieve desired boundary-to-boundary gap."""
    bearing_rad = np.radians(bearing_deg)
    direction = np.array([np.sin(bearing_rad), np.cos(bearing_rad)])

    # Initial estimate
    projections = boundary_km @ direction
    extent_fwd = projections.max()
    extent_back = -projections.min()
    offset = extent_fwd + extent_back + gap_km

    # Iterative refinement
    for _ in range(30):
        ref_bnd = boundary_km + offset * direction
        pts1 = sample_polygon_boundary(boundary_km, n_per_edge)
        pts2 = sample_polygon_boundary(ref_bnd, n_per_edge)
        actual_gap = cdist(pts1, pts2).min()
        error = actual_gap - gap_km
        if abs(error) < 0.01:  # 10m tolerance
            break
        offset -= error

    # Find closest boundary points for annotation
    ref_bnd = boundary_km + offset * direction
    pts1 = sample_polygon_boundary(boundary_km, n_per_edge)
    pts2 = sample_polygon_boundary(ref_bnd, n_per_edge)
    dists = cdist(pts1, pts2)
    min_idx = np.unravel_index(dists.argmin(), dists.shape)
    closest_target = pts1[min_idx[0]]
    closest_ref = pts2[min_idx[1]]

    return offset, direction, actual_gap, closest_target, closest_ref


# Reference farm placements: bearing, gap in km, color, label
placements = [
    (270, 5 * D / 1000, "tab:red", "5$D$ buffer\n(upwind)"),
    (135, 10 * D / 1000, "tab:orange", "10$D$ buffer\n(oblique)"),
    (0, 15 * D / 1000, "tab:blue", "15$D$ buffer\n(crosswind)"),
]

# =========================================================================
# Figure
# =========================================================================
fig = plt.figure(figsize=(14, 10))
ax = fig.add_axes([0.05, 0.05, 0.9, 0.85])

# Target polygon
poly = MplPolygon(bnd_km, closed=True, fill=True,
                  facecolor="#e0e0f0", edgecolor="black", lw=2.5, zorder=2,
                  label="Target farm boundary")
ax.add_patch(poly)

# Target turbines
ax.scatter(target_x, target_y, c="navy", marker="^", s=25, zorder=3,
            label="Target turbines (50)")

# Place identical reference farms
for bearing, gap_km, color, label in placements:
    offset, direction, actual_gap, pt_target, pt_ref = centroid_offset_for_gap(
        bnd_km, bearing, gap_km)

    # Reference polygon (identical shape, translated)
    ref_bnd = bnd_km + offset * direction
    ref_poly = MplPolygon(ref_bnd, closed=True, fill=True,
                          facecolor=color, edgecolor=color, lw=2, zorder=2,
                          alpha=0.2)
    ax.add_patch(ref_poly)
    # Reference boundary outline
    ref_outline = MplPolygon(ref_bnd, closed=True, fill=False,
                             edgecolor=color, lw=2, ls="-", zorder=4)
    ax.add_patch(ref_outline)

    # Reference turbines (same relative positions, translated)
    ref_tx = target_x + offset * direction[0]
    ref_ty = target_y + offset * direction[1]
    ax.scatter(ref_tx, ref_ty, c=color, marker="^", s=15, alpha=0.6, zorder=3)

    # Distance annotation: line between closest boundary points
    ax.annotate("", xy=pt_ref, xytext=pt_target,
                arrowprops=dict(arrowstyle="<->", color=color, lw=2,
                                connectionstyle="arc3,rad=0"))

    # Distance label
    mid = (pt_target + pt_ref) / 2
    gap_D = actual_gap * 1000 / D
    # Offset label perpendicular to the line
    perp = np.array([-direction[1], direction[0]])
    label_pos = mid + 1.5 * perp
    ax.text(label_pos[0], label_pos[1], f"{gap_D:.0f}$D$ ({actual_gap:.1f} km)",
            fontsize=10, color=color, fontweight="bold", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=color, alpha=0.9))

    # Farm label
    ref_center = bnd_km.mean(axis=0) + offset * direction
    ax.text(ref_center[0], ref_center[1] + 2, label,
            fontsize=9, color=color, ha="center", va="bottom",
            fontweight="bold")

# Compass rose
for angle, lbl in [(0, "N"), (90, "E"), (180, "S"), (270, "W")]:
    rad = np.radians(angle)
    r = 28
    ax.text(r * np.sin(rad), r * np.cos(rad), lbl,
            fontsize=14, ha="center", va="center", fontweight="bold",
            color="gray")

# Wind arrow
ax.annotate("", xy=(-20, 0), xytext=(-25, 0),
            arrowprops=dict(arrowstyle="-|>", color="steelblue", lw=3))
ax.text(-22.5, 1.5, "Prevailing\nwind (270$^\\circ$)", fontsize=9,
        color="steelblue", ha="center", va="bottom", fontweight="bold")

# Set limits
all_x = [bnd_km[:, 0].min()]
all_y = [bnd_km[:, 1].min()]
for bearing, gap_km, _, _ in placements:
    offset, direction, _, _, _ = centroid_offset_for_gap(bnd_km, bearing, gap_km)
    ref_bnd = bnd_km + offset * direction
    all_x.extend([ref_bnd[:, 0].min(), ref_bnd[:, 0].max()])
    all_y.extend([ref_bnd[:, 1].min(), ref_bnd[:, 1].max()])
all_x.extend([bnd_km[:, 0].max()])
all_y.extend([bnd_km[:, 1].max()])
pad = 4
ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
ax.set_ylim(min(all_y) - pad, max(all_y) + pad)
ax.set_aspect("equal")
ax.set_xlabel("x (km)", fontsize=12)
ax.set_ylabel("y (km)", fontsize=12)
ax.legend(loc="lower left", fontsize=10)

fig.suptitle("Regret Cross-Section Methodology\n"
             "An identical copy of the target farm is placed at varying bearings and "
             "boundary-gap distances.\nDistance is measured between the closest polygon "
             "boundary points (not centroids).",
             fontsize=12, y=0.98)

out = Path("paper_v3/figures/radar_schematic.png")
fig.savefig(str(out), dpi=200, bbox_inches="tight")
print(f"Saved: {out}")

"""Schematic figure explaining the regret cross-section methodology.

Shows the target farm and identical copies (reference farms) placed at different
bearings and buffer distances, with distance measured as the minimum gap between
farm boundary polygons.
"""

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Circle
from matplotlib.projections.polar import PolarAxes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from pathlib import Path
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
cx = _dk0w_raw[:, 0].mean()
cy = _dk0w_raw[:, 1].mean()
_hull = ConvexHull(_dk0w_raw - np.array([cx, cy]))
boundary_np = (_dk0w_raw - np.array([cx, cy]))[_hull.vertices]

from matplotlib.path import Path as MplPath
_polygon_path = MplPath(boundary_np)

def to_km(x, y):
    return np.asarray(x) / 1000, np.asarray(y) / 1000

bnd_km = np.column_stack(to_km(boundary_np[:, 0], boundary_np[:, 1]))


def sample_polygon_boundary(vertices, n_per_edge=50):
    pts = []
    n = len(vertices)
    for i in range(n):
        j = (i + 1) % n
        t = np.linspace(0, 1, n_per_edge, endpoint=False)
        edge_pts = vertices[i] + t[:, None] * (vertices[j] - vertices[i])
        pts.append(edge_pts)
    return np.vstack(pts)


def compute_boundary_gap(bnd1, bnd2, n_per_edge=50):
    pts1 = sample_polygon_boundary(bnd1, n_per_edge)
    pts2 = sample_polygon_boundary(bnd2, n_per_edge)
    dists = cdist(pts1, pts2)
    min_idx = np.unravel_index(dists.argmin(), dists.shape)
    return float(dists.min()), pts1[min_idx[0]], pts2[min_idx[1]]


def centroid_offset_for_gap(boundary, bearing_deg, gap_m, n_per_edge=50):
    bearing_rad = np.radians(bearing_deg)
    direction = np.array([np.sin(bearing_rad), np.cos(bearing_rad)])
    projections = boundary @ direction
    offset = projections.max() - projections.min() + gap_m
    for _ in range(30):
        ref_bnd = boundary + offset * direction
        actual_gap, _, _ = compute_boundary_gap(boundary, ref_bnd, n_per_edge)
        error = actual_gap - gap_m
        if abs(error) < 0.5:
            break
        offset -= error
    return offset, direction


# Target turbines (grid within polygon, for illustration)
tx_arr = np.arange(boundary_np[:, 0].min() + 2*D, boundary_np[:, 0].max() - 2*D, 4*D)
ty_arr = np.arange(boundary_np[:, 1].min() + 2*D, boundary_np[:, 1].max() - 2*D, 4*D)
tgx, tgy = np.meshgrid(tx_arr, ty_arr)
cands = np.column_stack([tgx.ravel(), tgy.ravel()])
inside = _polygon_path.contains_points(cands)
target_x_m, target_y_m = cands[inside, 0], cands[inside, 1]
if len(target_x_m) > 50:
    idx = np.round(np.linspace(0, len(target_x_m) - 1, 50)).astype(int)
    target_x_m, target_y_m = target_x_m[idx], target_y_m[idx]
target_x_km, target_y_km = to_km(target_x_m, target_y_m)

# Reference placements: (bearing_deg, buffer_D, color, label)
ref_placements = [
    (270, 10, "tab:red",    "High regret\n(upwind, $10D$ buffer)"),
    (135, 15, "tab:orange", "Ambush\n(oblique, $15D$ buffer)"),
    (  0, 20, "tab:green",  "Low regret\n(crosswind, $20D$ buffer)"),
]

# =========================================================================
# Figure: Two-panel schematic
# =========================================================================
fig = plt.figure(figsize=(18, 8))

# LEFT: Spatial layout
ax1 = fig.add_axes([0.02, 0.05, 0.52, 0.85])

# Target polygon
poly_closed = np.vstack([bnd_km, bnd_km[0:1]])
poly = MplPolygon(bnd_km, closed=True, fill=True,
                  facecolor="#dde0f0", edgecolor="navy", lw=2.5, zorder=2,
                  label="Target farm boundary")
ax1.add_patch(poly)
ax1.scatter(target_x_km, target_y_km, c="navy", marker="^", s=25, zorder=3,
            label=f"Target turbines ({len(target_x_km)})")

# Place identical copies
all_pts_x = [bnd_km[:, 0].min(), bnd_km[:, 0].max()]
all_pts_y = [bnd_km[:, 1].min(), bnd_km[:, 1].max()]

for bearing, buffer_D, color, label in ref_placements:
    gap_m = buffer_D * D
    offset, direction = centroid_offset_for_gap(boundary_np, bearing, gap_m)

    # Reference farm boundary (translated)
    ref_bnd_m = boundary_np + offset * direction
    ref_bnd_km = np.column_stack(to_km(ref_bnd_m[:, 0], ref_bnd_m[:, 1]))

    # Reference turbines (translated)
    ref_tx_m = target_x_m + offset * direction[0]
    ref_ty_m = target_y_m + offset * direction[1]
    ref_tx_km, ref_ty_km = to_km(ref_tx_m, ref_ty_m)

    # Draw reference polygon
    ref_poly = MplPolygon(ref_bnd_km, closed=True, fill=True,
                          facecolor=matplotlib.colors.to_rgba(color, 0.15),
                          edgecolor=color, lw=2, ls="--", zorder=2)
    ax1.add_patch(ref_poly)

    # Draw reference turbines
    ax1.scatter(ref_tx_km, ref_ty_km, c=color, marker="^", s=15, alpha=0.6, zorder=3)

    # Find closest boundary points and draw distance annotation
    _, pt_target, pt_ref = compute_boundary_gap(boundary_np, ref_bnd_m)
    pt_t_km = pt_target / 1000
    pt_r_km = pt_ref / 1000
    ax1.plot([pt_t_km[0], pt_r_km[0]], [pt_t_km[1], pt_r_km[1]],
             color=color, lw=2, ls="-", zorder=6)
    ax1.plot(pt_t_km[0], pt_t_km[1], "o", color=color, ms=5, zorder=7)
    ax1.plot(pt_r_km[0], pt_r_km[1], "o", color=color, ms=5, zorder=7)

    # Midpoint label
    mid_km = (pt_t_km + pt_r_km) / 2
    buf_km = gap_m / 1000
    ax1.text(mid_km[0], mid_km[1], f"  {buffer_D}$D$ ({buf_km:.1f} km)",
             fontsize=8, color=color, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                       edgecolor=color, alpha=0.9),
             zorder=8)

    # Label for the farm
    ref_cx_km = ref_bnd_km[:, 0].mean()
    ref_cy_km = ref_bnd_km[:, 1].mean()
    label_offset_x = 1.5 * np.sin(np.radians(bearing))
    label_offset_y = 1.5 * np.cos(np.radians(bearing))
    ax1.text(ref_cx_km + label_offset_x, ref_cy_km + label_offset_y, label,
             fontsize=8, color=color, ha="center", va="center",
             fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor=color, alpha=0.9),
             zorder=9)

    # Track extents
    all_pts_x.extend([ref_bnd_km[:, 0].min() - 2, ref_bnd_km[:, 0].max() + 2])
    all_pts_y.extend([ref_bnd_km[:, 1].min() - 2, ref_bnd_km[:, 1].max() + 2])

# Distance rings
for r_km in [10, 20, 30, 40]:
    circle = Circle((0, 0), r_km, fill=False, edgecolor="gray", ls=":",
                     lw=0.6, alpha=0.4, zorder=1)
    ax1.add_patch(circle)
    ax1.text(r_km * 0.71, r_km * 0.71, f"{r_km} km", fontsize=7,
             color="gray", ha="center", va="center")

# Compass rose
pad = 4
x_lo, x_hi = min(all_pts_x) - pad, max(all_pts_x) + pad
y_lo, y_hi = min(all_pts_y) - pad, max(all_pts_y) + pad
ax1.set_xlim(x_lo, x_hi)
ax1.set_ylim(y_lo, y_hi)
r_compass = min(abs(x_lo), abs(x_hi), abs(y_lo), abs(y_hi)) * 0.95
for angle, lbl in [(0, "N"), (90, "E"), (180, "S"), (270, "W")]:
    rad = np.radians(angle)
    ax1.text(r_compass * np.sin(rad), r_compass * np.cos(rad), lbl,
             fontsize=13, ha="center", va="center", fontweight="bold",
             color="gray")

ax1.set_aspect("equal")
ax1.set_xlabel("x (km)", fontsize=12)
ax1.set_ylabel("y (km)", fontsize=12)
ax1.set_title("Step 1: Place identical copy at $(\\theta, d_{\\mathrm{buffer}})$\n"
              "and re-optimize target layout",
              fontsize=13, fontweight="bold")
ax1.legend(loc="lower left", fontsize=9)

# RIGHT: Polar heatmap (use existing data if available)
ax2 = fig.add_axes([0.58, 0.1, 0.38, 0.75], projection="polar")
ax2.set_theta_zero_location("N")
ax2.set_theta_direction(-1)

xsec_path = Path("analysis/cross_section/a0.9_f1.0/results.json")
if xsec_path.exists():
    d = json.load(open(xsec_path))
    bearings_data = np.array(d["bearings_deg"])
    distances_D_arr = np.array(d["distances_D"])
    regret_pct = np.array(d["regret_grid_pct"])

    dbear = bearings_data[1] - bearings_data[0]
    bear_edges = np.radians(np.concatenate([bearings_data - dbear/2,
                                             [bearings_data[-1] + dbear/2]]))
    dist_edges = [distances_D_arr[0] - (distances_D_arr[1] - distances_D_arr[0]) / 2]
    for i in range(len(distances_D_arr) - 1):
        dist_edges.append((distances_D_arr[i] + distances_D_arr[i+1]) / 2)
    dist_edges.append(distances_D_arr[-1] + (distances_D_arr[-1] - distances_D_arr[-2]) / 2)
    dist_edges = np.maximum(np.array(dist_edges), 0)

    theta_grid, r_grid = np.meshgrid(bear_edges, dist_edges)
    im = ax2.pcolormesh(theta_grid, r_grid, np.clip(regret_pct, 0, None),
                         cmap="YlOrRd", shading="flat")

    idx = np.unravel_index(np.nanargmax(regret_pct), regret_pct.shape)
    ax2.plot(np.radians(bearings_data[idx[1]]), distances_D_arr[idx[0]],
             "k*", markersize=16, zorder=10)

    cbar = fig.colorbar(im, ax=ax2, pad=0.12, shrink=0.8)
    cbar.set_label("Design regret (% of AEP)", fontsize=10)

ax2.set_title("Step 2: Regret Cross-Section\n"
              "(buffer distance = boundary gap)",
              fontsize=13, fontweight="bold", pad=20)
ax2.set_rlabel_position(135)
ax2.set_yticks([10, 20, 30, 40, 60])
ax2.set_yticklabels(["10$D$", "20$D$", "30$D$", "40$D$", "60$D$"], fontsize=8)

# Wind rose inset
try:
    from edrose import EllipticalWindRose
    wr = EllipticalWindRose(a=0.9, f=1.0, theta_prev=270, n_sectors=24)
    ax_inset = inset_axes(ax2, width="30%", height="30%", loc="lower right",
                          axes_class=PolarAxes)
    ax_inset.set_theta_zero_location("N")
    ax_inset.set_theta_direction(-1)
    wr_width = np.radians(360 / 24)
    ax_inset.bar(np.radians(wr.wind_directions), wr.sector_frequencies,
                 width=wr_width, color="steelblue", alpha=0.7,
                 edgecolor="navy", linewidth=0.3)
    ax_inset.set_yticks([])
    ax_inset.set_xticks([])
    ax_inset.patch.set_alpha(0.8)
except ImportError:
    pass

fig.suptitle("Regret Cross-Section Methodology: Identical-Copy Neighbor Placement",
             fontsize=15, fontweight="bold", y=0.98)

out = Path("paper_v3/figures/radar_schematic.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(str(out), dpi=200, bbox_inches="tight")
print(f"Saved: {out}")

# Also save to paper_v2 for backwards compat
out2 = Path("paper_v2/figures/radar_schematic.png")
if out2.parent.exists():
    fig.savefig(str(out2), dpi=200, bbox_inches="tight")
    print(f"Saved: {out2}")

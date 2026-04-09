"""Schematic figure explaining the regret cross-section ('radar') methodology.

Shows the target farm, a few example reference farm placements at different
bearings and distances, and the resulting polar regret map concept.
"""

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, FancyArrowPatch, Circle
from matplotlib.projections.polar import PolarAxes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
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
boundary_np = _dk0w_raw - np.array([cx, cy])

def to_km(x, y):
    return np.asarray(x) / 1000, np.asarray(y) / 1000

bnd_km = np.column_stack(to_km(boundary_np[:, 0], boundary_np[:, 1]))

# Compute polygon extent in each direction to place farms outside
def polygon_extent_km(bearing_deg, bnd_km):
    """Max projection of polygon onto bearing direction (km from centroid)."""
    rad = np.radians(bearing_deg)
    projections = bnd_km[:, 0] * np.sin(rad) + bnd_km[:, 1] * np.cos(rad)
    return projections.max()

ref_farm_half_km = 2 * 7 * D / 1000  # half-width of 5x5 reference farm

# Reference farm positions: bearing, buffer_km_from_edge, color, label
ref_placements = [
    (270, 3.0, "tab:red", "High regret\n(upwind)"),
    (135, 5.0, "tab:orange", "Ambush\n(oblique)"),
    (0, 8.0, "tab:green", "Low regret\n(crosswind)"),
    (45, 15.0, "tab:blue", "Minimal\n(far)"),
]

# Convert to absolute distance from centroid
ref_positions = []
for bearing, buffer_km, color, label in ref_placements:
    edge_km = polygon_extent_km(bearing, bnd_km)
    dist_km = edge_km + ref_farm_half_km + buffer_km
    ref_positions.append((bearing, dist_km, color, label))

# Build a tiny 5x5 grid for the reference farm
ref_spacing = 7 * D
xs = np.arange(5) * ref_spacing - 2 * ref_spacing
ys = np.arange(5) * ref_spacing - 2 * ref_spacing
gx, gy = np.meshgrid(xs, ys)
ref_x_local, ref_y_local = gx.ravel() / 1000, gy.ravel() / 1000  # in km

# Target turbine positions (simple grid for illustration)
from matplotlib.path import Path as MplPath
poly_path = MplPath(bnd_km)
tx = np.arange(bnd_km[:, 0].min() + 1, bnd_km[:, 0].max() - 1, 4 * D / 1000)
ty = np.arange(bnd_km[:, 1].min() + 1, bnd_km[:, 1].max() - 1, 4 * D / 1000)
tgx, tgy = np.meshgrid(tx, ty)
cands = np.column_stack([tgx.ravel(), tgy.ravel()])
inside = poly_path.contains_points(cands)
target_x, target_y = cands[inside, 0], cands[inside, 1]
# Subsample to ~50
if len(target_x) > 50:
    idx = np.round(np.linspace(0, len(target_x) - 1, 50)).astype(int)
    target_x, target_y = target_x[idx], target_y[idx]

# =========================================================================
# Figure: Two-panel schematic
# =========================================================================
fig = plt.figure(figsize=(18, 8))

# LEFT: Spatial layout showing the concept
ax1 = fig.add_axes([0.02, 0.05, 0.52, 0.85])

# Target polygon
poly = MplPolygon(bnd_km, closed=True, fill=True,
                  facecolor="#e8e8f0", edgecolor="black", lw=2.5, zorder=2)
ax1.add_patch(poly)

# Target turbines
ax1.scatter(target_x, target_y, c="navy", marker="^", s=30, zorder=3,
            label="Target farm (50 turbines)")

# Distance rings (from centroid)
for r_km in [5, 10, 15, 20, 25]:
    circle = Circle((0, 0), r_km, fill=False, edgecolor="gray", ls=":",
                     lw=0.8, alpha=0.5, zorder=1)
    ax1.add_patch(circle)
    ax1.text(r_km * 0.71, r_km * 0.71, f"{r_km} km", fontsize=7,
             color="gray", ha="center", va="center")

# Reference farm placements
for bearing, dist_km, color, label in ref_positions:
    bearing_rad = np.radians(bearing)
    nx_km = dist_km * np.sin(bearing_rad)
    ny_km = dist_km * np.cos(bearing_rad)

    # Place reference farm
    rx = ref_x_local + nx_km
    ry = ref_y_local + ny_km
    ax1.scatter(rx, ry, c=color, marker="s", s=15, alpha=0.7, zorder=4)

    # Arrow from centroid to farm center
    ax1.annotate("", xy=(nx_km, ny_km), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5,
                                 connectionstyle="arc3,rad=0.1"),
                 zorder=5)

    # Label
    offset_x = 1.5 * np.sin(bearing_rad)
    offset_y = 1.5 * np.cos(bearing_rad)
    ax1.text(nx_km + offset_x, ny_km + offset_y, label,
             fontsize=9, color=color, ha="center", va="center",
             fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor=color, alpha=0.9))

# Compass rose at edge of plot
xlim = ax1.get_xlim()
ylim = ax1.get_ylim()
r_compass = min(abs(xlim[0]), abs(xlim[1]), abs(ylim[0]), abs(ylim[1])) * 0.92
for angle, lbl in [(0, "N"), (90, "E"), (180, "S"), (270, "W")]:
    rad = np.radians(angle)
    ax1.text(r_compass * np.sin(rad), r_compass * np.cos(rad), lbl,
             fontsize=13, ha="center", va="center", fontweight="bold",
             color="gray")

# Auto-scale to fit all reference farms
all_pts_x = [bnd_km[:, 0].min(), bnd_km[:, 0].max()]
all_pts_y = [bnd_km[:, 1].min(), bnd_km[:, 1].max()]
for bearing, dist_km, color, label in ref_positions:
    rad = np.radians(bearing)
    cx_ref = dist_km * np.sin(rad)
    cy_ref = dist_km * np.cos(rad)
    all_pts_x.extend([cx_ref - ref_farm_half_km - 2, cx_ref + ref_farm_half_km + 2])
    all_pts_y.extend([cy_ref - ref_farm_half_km - 2, cy_ref + ref_farm_half_km + 2])
pad = 3
ax1.set_xlim(min(all_pts_x) - pad, max(all_pts_x) + pad)
ax1.set_ylim(min(all_pts_y) - pad, max(all_pts_y) + pad)
ax1.set_aspect("equal")
ax1.set_xlabel("x (km)", fontsize=12)
ax1.set_ylabel("y (km)", fontsize=12)
ax1.set_title("Step 1: Place reference farm at $(\\theta, r)$\n"
              "and re-optimize target layout",
              fontsize=13, fontweight="bold")
ax1.legend(loc="lower left", fontsize=9)

# RIGHT: Polar heatmap result (use K=5 data if available)
ax2 = fig.add_axes([0.58, 0.1, 0.38, 0.75], projection="polar")
ax2.set_theta_zero_location("N")
ax2.set_theta_direction(-1)

# Try to load actual cross-section data
xsec_path = Path("analysis/cross_section/a0.9_f1.0/results.json")
if xsec_path.exists():
    d = json.load(open(xsec_path))
    bearings = np.array(d["bearings_deg"])
    distances_D_arr = np.array(d["distances_D"])
    regret_pct = np.array(d["regret_grid_pct"])

    dbear = bearings[1] - bearings[0]
    bear_edges = np.radians(np.concatenate([bearings - dbear/2,
                                             [bearings[-1] + dbear/2]]))
    dist_edges = [distances_D_arr[0] - (distances_D_arr[1] - distances_D_arr[0]) / 2]
    for i in range(len(distances_D_arr) - 1):
        dist_edges.append((distances_D_arr[i] + distances_D_arr[i+1]) / 2)
    dist_edges.append(distances_D_arr[-1] + (distances_D_arr[-1] - distances_D_arr[-2]) / 2)
    dist_edges = np.maximum(np.array(dist_edges), 0)

    theta_grid, r_grid = np.meshgrid(bear_edges, dist_edges)
    im = ax2.pcolormesh(theta_grid, r_grid, np.clip(regret_pct, 0, None),
                         cmap="YlOrRd", shading="flat")

    # Mark max
    idx = np.unravel_index(np.nanargmax(regret_pct), regret_pct.shape)
    ax2.plot(np.radians(bearings[idx[1]]), distances_D_arr[idx[0]],
             "k*", markersize=16, zorder=10)

    cbar = fig.colorbar(im, ax=ax2, pad=0.12, shrink=0.8)
    cbar.set_label("Design regret (% of AEP)", fontsize=10)

ax2.set_title("Step 2: Regret Cross-Section\n"
              "($a$=0.9, $f$=1.0, K=5 preliminary)",
              fontsize=13, fontweight="bold", pad=20)
ax2.set_rlabel_position(135)
ax2.set_yticks([10, 20, 30, 40, 60])
ax2.set_yticklabels(["10$D$", "20$D$", "30$D$", "40$D$", "60$D$"], fontsize=8)

# Wind rose inset
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

fig.suptitle("Regret Cross-Section Methodology: Mapping Directional Vulnerability",
             fontsize=15, fontweight="bold", y=0.98)

out = Path("paper_v2/figures/radar_schematic.png")
fig.savefig(str(out), dpi=200, bbox_inches="tight")
print(f"Saved: {out}")

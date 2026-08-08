"""Single polar pcolormesh: radial = buffer distance, theta = neighbor bearing.
Bastankhah, a=0.7, f=0.5 (somewhat concentrated rose)."""

import io
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.projections.polar import PolarAxes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pathlib import Path

a, f = 0.7, 0.5
distances_D = [2, 5, 10, 15, 20, 30, 40]
THETA_PREV, N_BINS = 270, 24


def get_rose(a, f):
    from edrose import EllipticalWindRose
    wr = EllipticalWindRose(a=a, f=f, theta_prev=THETA_PREV, n_sectors=N_BINS)
    return np.array(wr.wind_directions), np.array(wr.sector_frequencies)


def render_rose_to_image(dirs, freq, size_in=0.8):
    fr = plt.figure(figsize=(size_in, size_in), dpi=200)
    ar = fr.add_subplot(111, projection="polar")
    ar.set_theta_zero_location("N"); ar.set_theta_direction(-1)
    width = np.deg2rad(360 / N_BINS) * 0.95
    ar.bar(np.deg2rad(dirs), freq, width=width,
           color="steelblue", edgecolor="navy", alpha=0.9, linewidth=0.4)
    ar.set_yticks([]); ar.set_xticks([])
    ar.set_facecolor("none")
    fr.patch.set_alpha(0)
    buf = io.BytesIO()
    fr.savefig(buf, format="png", dpi=200, transparent=True,
               bbox_inches="tight", pad_inches=0.02)
    plt.close(fr); buf.seek(0)
    return mpimg.imread(buf)


# Load per-distance Bastankhah results
regret_rows = []
lib = None; bearings = None
for d in distances_D:
    p = Path(f"analysis/buffer_table_tp/a{a}_f{f}_d{d}/Nt50/results.json")
    data = json.load(open(p))
    regret_rows.append(np.array(data["regret_grid_gwh"])[0, :])
    if lib is None:
        lib = data["liberal_aep_gwh"]
        bearings = np.array(data["bearings_deg"])

regret_grid = np.array(regret_rows)
pct = np.clip(100 * regret_grid / lib, 0, None)

fig, ax = plt.subplots(1, 1, figsize=(7, 6.5), subplot_kw={"projection": "polar"})
ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)

dbear = bearings[1] - bearings[0]
bear_edges = np.radians(np.concatenate([bearings - dbear/2, [bearings[-1] + dbear/2]]))
dists = np.array(distances_D)
dist_edges = [dists[0] - (dists[1] - dists[0]) / 2]
for i in range(len(dists) - 1):
    dist_edges.append((dists[i] + dists[i+1]) / 2)
dist_edges.append(dists[-1] + (dists[-1] - dists[-2]) / 2)
dist_edges = np.maximum(np.array(dist_edges), 0)

theta_grid, r_grid = np.meshgrid(bear_edges, dist_edges)
im = ax.pcolormesh(theta_grid, r_grid, pct, cmap="YlOrRd", vmin=0,
                   vmax=pct.max(), shading="flat")
fig.colorbar(im, ax=ax, label="Design regret (\\% of AEP)", shrink=0.8, pad=0.12)

# Mark the 5D ring with a circle for emphasis
theta_circle = np.linspace(0, 2*np.pi, 360)
ax.plot(theta_circle, np.full_like(theta_circle, 5), "k--", lw=1.2, alpha=0.7)
ax.text(np.deg2rad(45), 5, "$5D$", fontsize=10, fontweight="bold",
        ha="center", va="bottom")

ax.set_yticks([2, 5, 10, 20, 30, 40])
ax.set_yticklabels(["2D", "5D", "10D", "20D", "30D", "40D"], fontsize=8)
ax.set_rlabel_position(135)

wr_dirs, wr_freq = get_rose(a, f)
rose_img = render_rose_to_image(wr_dirs, wr_freq, size_in=1.0)

# Place rose image in the title area
fig.suptitle("TurboPark cross-section. Dashed ring: $5D$ buffer. $K=500$. Rose:",
             fontsize=11, x=0.42, y=0.97)
img_box = OffsetImage(rose_img, zoom=0.4)
ab = AnnotationBbox(img_box, (0.82, 0.95), xycoords="figure fraction",
                    frameon=False, box_alignment=(0, 0.5))
fig.add_artist(ab)
plt.tight_layout()

out = Path(f"paper_v3/figures/polar_radial_buffer_a{a}_f{f}_tp.png")
fig.savefig(str(out), dpi=200, bbox_inches="tight")
print(f"Saved: {out}")
print(f"\nRegret at 5D: max = {pct[distances_D.index(5), :].max():.3f}% at "
      f"bearing {bearings[pct[distances_D.index(5), :].argmax()]:.0f}°")

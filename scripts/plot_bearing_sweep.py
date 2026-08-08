"""Regret vs bearing at fixed buffer distance.
Legend has miniature wind-rose icons next to each a/f label."""

import io
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, HPacker, VPacker, TextArea, AnchoredOffsetbox
from pathlib import Path

D = 240.0
THETA_PREV = 270
N_BINS = 24
a_curves = [0.01, 0.05, 0.2, 0.564, 0.9, 2.0, 5.0]


def load(path):
    if not path.exists(): return None
    data = json.load(open(path))
    g = np.array(data["regret_grid_gwh"]); lib = data["liberal_aep_gwh"]
    return np.array(data["bearings_deg"]), 100 * g[0, :] / lib


def a_path(a, dist):
    if abs(a - 0.9) < 1e-9:
        return Path(f"analysis/ablation_f_sweep/a0.9_f1.0_d{dist}/Nt50/results.json")
    return Path(f"analysis/ablation_a_sweep/a{a}_f1.0_d{dist}/Nt50/results.json")


def dei_freqs():
    import pandas as pd
    df = pd.read_csv("energy_island_10y_daily_av_wind.csv", sep=";")
    wd_ts = df["WD_150"].values
    edges = np.linspace(0, 360, N_BINS + 1); centers = (edges[:-1] + edges[1:]) / 2
    w = np.zeros(N_BINS)
    for i in range(N_BINS):
        if i == N_BINS - 1: mask = (wd_ts >= edges[i]) | (wd_ts < edges[0])
        else: mask = (wd_ts >= edges[i]) & (wd_ts < edges[i+1])
        w[i] = mask.sum()
    return centers, w / w.sum()


def ell_freqs(a, f=1.0):
    from edrose import EllipticalWindRose
    wr = EllipticalWindRose(a=a, f=f, theta_prev=THETA_PREV, n_sectors=N_BINS)
    return np.array(wr.wind_directions), np.array(wr.sector_frequencies)


def rose_image(dirs, freq, color, size=0.6):
    """Render a small wind-rose to a numpy RGBA array (transparent bg)."""
    fig_r = plt.figure(figsize=(size, size), dpi=150)
    ax_r = fig_r.add_subplot(111, projection="polar")
    ax_r.set_theta_zero_location("N"); ax_r.set_theta_direction(-1)
    width = np.deg2rad(360 / N_BINS) * 0.95
    ax_r.bar(np.deg2rad(dirs), freq, width=width, color=color,
             edgecolor=color, alpha=1.0, linewidth=0.0)
    ax_r.set_yticks([]); ax_r.set_xticks([])
    ax_r.set_facecolor("none")
    fig_r.patch.set_alpha(0)
    buf = io.BytesIO()
    fig_r.savefig(buf, format="png", dpi=150, transparent=True,
                  bbox_inches="tight", pad_inches=0.02)
    plt.close(fig_r)
    buf.seek(0)
    import matplotlib.image as mpimg
    return mpimg.imread(buf)


cmap = plt.cm.viridis
a_colors = cmap(np.linspace(0.05, 0.95, len(a_curves)))

# Pre-render rose images
rose_imgs = []
for a, color in zip(a_curves, a_colors):
    dirs, freq = ell_freqs(a, 1.0)
    rose_imgs.append(rose_image(dirs, freq, color))
d_dirs, d_freq = dei_freqs()
dei_img = rose_image(d_dirs, d_freq, (0.5, 0.0, 0.5))  # tab:purple-ish

fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
for ax, dist, title in [(axes[0], 2, "Buffer = 2$D$"),
                         (axes[1], 10, "Buffer = 10$D$")]:
    line_handles = []; line_labels = []
    for a, color in zip(a_curves, a_colors):
        r = load(a_path(a, dist))
        if r is None: continue
        b, pct = r
        h, = ax.plot(b, pct, "-", color=color, linewidth=2, alpha=0.95)
        line_handles.append((h, color))
    r = load(Path(f"analysis/cross_section_fixed/dei_d{dist}/Nt50/results.json"))
    if r is not None:
        b, pct = r
        h, = ax.plot(b, pct, "-.", color="tab:purple", linewidth=2.2, zorder=10)
        line_handles.append((h, "tab:purple"))
    ax.set_xlabel("Neighbor bearing ($^\\circ$)")
    ax.set_ylabel("Design regret (\\% of AEP)")
    ax.set_title(title, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 360); ax.set_xticks(np.arange(0, 361, 45))
    ax.set_ylim(bottom=0)

    # Build a custom legend: rows of [rose icon | text]
    rows = []
    for (a, color), img in zip(zip(a_curves, a_colors), rose_imgs):
        icon = OffsetImage(img, zoom=0.32)
        txt = TextArea(f" $a$={a:g}, $f$=1", textprops={"fontsize": 9})
        rows.append(HPacker(children=[icon, txt], pad=0, sep=2, align="center"))
    icon_dei = OffsetImage(dei_img, zoom=0.32)
    txt_dei = TextArea(" DEI real rose", textprops={"fontsize": 9, "color": "tab:purple"})
    rows.append(HPacker(children=[icon_dei, txt_dei], pad=0, sep=2, align="center"))
    legend_box = VPacker(children=rows, pad=4, sep=2, align="left")
    anchored = AnchoredOffsetbox(loc="upper right", child=legend_box,
                                  pad=0.4, borderpad=0.3, frameon=True)
    anchored.patch.set_facecolor("white")
    anchored.patch.set_alpha(0.92)
    ax.add_artist(anchored)

fig.suptitle("Regret vs.\\ neighbor bearing at $f=1$ (unidirectional in probability)\n"
             "Legend icons show each curve's wind rose ($\\theta_\\text{prev}=270^\\circ$). "
             "$K_\\text{lib}=K_\\text{cons}=500$.",
             fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.93])

out = Path("paper_v3/figures/bearing_sweep.png")
fig.savefig(str(out), dpi=200, bbox_inches="tight")
print(f"Saved: {out}")

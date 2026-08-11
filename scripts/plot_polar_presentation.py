"""Large single-case polar regret cross-sections for presentation slides.

One big polar per (wind rose, wake model): radial axis is buffer distance,
angular axis is neighbour bearing, colour is design regret.  Wind-rose inset,
NYSERDA 4 nmi reference ring, peak marked and annotated.

Reads the converged FunWake results, drawing each cell from the first source
that has it:
  1. analysis/buffer_table{_tp}_funwake/{case}_d{d}/            (K=2000)
  2. analysis/cross_section_fixed{_tp}_funwake/{case}_d{d}/     (K=2000)
  3. analysis/cross_section_fixed{_tp}_funwake_k500_packed/...  (K=500)

Output: paper_v3/figures/slide_polar_{case}_{wake}.png
"""

import io
import json
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from pathlib import Path

D = 240.0
DISTANCES = [2, 5, 10, 15, 20, 30, 40]
N_BINS = 24
THETA_PREV = 270
NYSERDA_D = 30.9  # 4 nautical miles in rotor diameters

CASES = [
    ("a0.9_f1.0", "Concentrated unidirectional  ($a$=0.9, $f$=1.0)"),
    ("a0.7_f0.5", "Mid-range  ($a$=0.7, $f$=0.5)"),
    ("a0.5_f0.0", "Moderately bidirectional  ($a$=0.5, $f$=0.0)"),
    ("dei", "Danish Energy Island  (measured rose)"),
]
WAKES = [("", "Bastankhah"), ("_tp", "TurboPark")]


def get_rose(case):
    if case == "dei":
        import pandas as pd
        df = pd.read_csv(Path("energy_island_10y_daily_av_wind.csv"), sep=";")
        wd = df["WD_150"].values
        edges = np.linspace(0, 360, N_BINS + 1)
        centers = (edges[:-1] + edges[1:]) / 2
        w = np.zeros(N_BINS)
        for i in range(N_BINS):
            if i == N_BINS - 1:
                m = (wd >= edges[i]) | (wd < edges[0])
            else:
                m = (wd >= edges[i]) & (wd < edges[i + 1])
            w[i] = m.sum()
        return centers, w / w.sum()
    a, f = (float(x) for x in case.replace("a", "").replace("f", "").split("_"))
    from edrose import EllipticalWindRose
    wr = EllipticalWindRose(a=a, f=f, theta_prev=THETA_PREV, n_sectors=N_BINS)
    return np.array(wr.wind_directions), np.array(wr.sector_frequencies)


def render_rose(dirs, freq, size_in=1.0):
    fr = plt.figure(figsize=(size_in, size_in), dpi=200)
    ar = fr.add_subplot(111, projection="polar")
    ar.set_theta_zero_location("N")
    ar.set_theta_direction(-1)
    ar.bar(np.deg2rad(dirs), freq, width=np.deg2rad(360 / N_BINS) * 0.95,
           color="steelblue", edgecolor="navy", alpha=0.9, linewidth=0.4)
    ar.set_yticks([])
    ar.set_xticks([])
    ar.set_facecolor("none")
    fr.patch.set_alpha(0)
    buf = io.BytesIO()
    fr.savefig(buf, format="png", dpi=200, transparent=True,
               bbox_inches="tight", pad_inches=0.02)
    plt.close(fr)
    buf.seek(0)
    return mpimg.imread(buf)


def load_row(case, d, sfx):
    """(regret_row_gwh, liberal_aep, bearings, K_tag) or None."""
    for base, k in [
        (f"analysis/buffer_table{sfx}_funwake/{case}_d{d}/Nt50/results.json", 2000),
        (f"analysis/cross_section_fixed{sfx}_funwake/{case}_d{d}/Nt50/results.json", 2000),
    ]:
        p = Path(base)
        if p.exists():
            dat = json.load(open(p))
            g = np.array(dat["regret_grid_gwh"])
            return g[0, :], dat["liberal_aep_gwh"], np.array(dat["bearings_deg"]), k
    for p in sorted(glob.glob(
            f"analysis/cross_section_fixed{sfx}_funwake_k500_packed/{case}*/Nt50/results.json")):
        dat = json.load(open(p))
        ds = [float(x) for x in np.atleast_1d(dat["distances_D"])]
        if float(d) in ds:
            g = np.array(dat["regret_grid_gwh"])
            return (g[ds.index(float(d)), :], dat["liberal_aep_gwh"],
                    np.array(dat["bearings_deg"]), 500)
    return None


def load_case(case, sfx):
    rows, lib, bearings, ks = [], None, None, set()
    for d in DISTANCES:
        hit = load_row(case, d, sfx)
        if hit is None:
            rows.append(np.full(N_BINS, np.nan))
            continue
        row, lib_aep, b, k = hit
        rows.append(row)
        ks.add(k)
        if lib is None:
            lib, bearings = lib_aep, b
    if bearings is None:
        return None
    return np.array(rows), lib, bearings, sorted(ks)


def make_figure(case, case_label, sfx, wake_label):
    got = load_case(case, sfx)
    if got is None:
        print(f"  [skip] {case}{sfx}: no data")
        return
    grid, lib, bearings, ks = got
    pct = np.clip(100 * grid / lib, 0, None)
    if np.all(np.isnan(pct)):
        print(f"  [skip] {case}{sfx}: all-NaN")
        return

    fig, ax = plt.subplots(1, 1, figsize=(8.2, 7.4),
                           subplot_kw={"projection": "polar"})
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    dbear = bearings[1] - bearings[0]
    bear_edges = np.radians(np.concatenate([bearings - dbear / 2,
                                            [bearings[-1] + dbear / 2]]))
    dists = np.array(DISTANCES, dtype=float)
    dist_edges = [dists[0] - (dists[1] - dists[0]) / 2]
    for i in range(len(dists) - 1):
        dist_edges.append((dists[i] + dists[i + 1]) / 2)
    dist_edges.append(dists[-1] + (dists[-1] - dists[-2]) / 2)
    dist_edges = np.maximum(np.array(dist_edges), 0)

    theta_grid, r_grid = np.meshgrid(bear_edges, dist_edges)
    im = ax.pcolormesh(theta_grid, r_grid, pct, cmap="YlOrRd",
                       vmin=0, vmax=np.nanmax(pct), shading="flat")
    cb = fig.colorbar(im, ax=ax, label="Design regret (% of AEP)",
                      shrink=0.78, pad=0.11)
    cb.ax.tick_params(labelsize=9)

    # NYSERDA 4 nmi reference ring
    tc = np.linspace(0, 2 * np.pi, 361)
    ax.plot(tc, np.full_like(tc, NYSERDA_D), "k--", lw=1.3, alpha=0.75)
    ax.text(np.deg2rad(58), NYSERDA_D, "NYSERDA 4 nmi", fontsize=8.5,
            fontweight="bold", ha="center", va="bottom", color="k")

    # peak marker
    idx = np.unravel_index(np.nanargmax(pct), pct.shape)
    pk_bear, pk_dist, pk_val = bearings[idx[1]], DISTANCES[idx[0]], pct[idx]
    ax.plot(np.radians(pk_bear), pk_dist, "*", color="k", ms=20, zorder=10)
    ax.annotate(f"peak {pk_val:.2f}% of AEP\n{pk_bear:.0f}$^\\circ$, {pk_dist}$D$",
                xy=(np.radians(pk_bear), pk_dist), xytext=(0.02, 0.02),
                textcoords="figure fraction", fontsize=10, fontweight="bold",
                ha="left", va="bottom",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="k", alpha=0.9))

    ax.set_yticks([10, 20, 30, 40])
    ax.set_yticklabels(["10D", "20D", "30D", "40D"], fontsize=8.5)
    ax.set_rlabel_position(112)
    ax.tick_params(axis="x", labelsize=10)

    ktxt = "K=2000" if ks == [2000] else ("K=500" if ks == [500] else "K=500-2000")
    fig.suptitle(f"{wake_label}   |   {case_label}", fontsize=13.5,
                 fontweight="bold", x=0.42, y=0.99)
    windtxt = "" if case == "dei" else f"   Wind from {THETA_PREV}$^\\circ$."
    fig.text(0.42, 0.944, "Identical neighbouring farm at each bearing and buffer gap.",
             fontsize=9.5, ha="center", color="dimgray")
    fig.text(0.42, 0.916, f"FunWake schedule, {ktxt}.{windtxt}",
             fontsize=9.5, ha="center", color="dimgray")

    rimg = render_rose(*get_rose(case))
    ab = AnnotationBbox(OffsetImage(rimg, zoom=0.33), (0.875, 0.945),
                        xycoords="figure fraction", frameon=False,
                        box_alignment=(0, 0.5))
    fig.add_artist(ab)
    fig.text(0.878, 0.882, "wind rose", fontsize=8.5, color="dimgray", ha="left")

    tag = "tp" if sfx else "bast"
    out = Path(f"paper_v3/figures/slide_polar_{case}_{tag}.png")
    fig.savefig(str(out), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out.name:38s} peak {pk_val:6.2f}%  at {pk_bear:3.0f}deg {pk_dist:2d}D  [{ktxt}]")


if __name__ == "__main__":
    for sfx, wake_label in WAKES:
        print(f"[{wake_label}]")
        for case, case_label in CASES:
            make_figure(case, case_label, sfx, wake_label)

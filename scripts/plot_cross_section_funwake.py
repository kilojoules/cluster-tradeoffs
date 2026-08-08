"""FunWake-schedule cross-section polar plots, both wake models.

Data sources, per (case, distance), first hit wins:
  1. analysis/buffer_table{_tp}_funwake/{case}_d{d}/Nt50/results.json
     (elliptical cases ARE buffer-table grid points; K=2000)
  2. analysis/cross_section_fixed{_tp}_funwake/{case}_d{d}/Nt50/results.json
     (LUMI-era converged cells; K=2000)
  3. analysis/cross_section_fixed{_tp}_funwake_k500_packed/{case}*/Nt50/results.json
     (gbar K=500 packed chunks; each JSON holds several distances)

Missing cells render blank (NaN) and are listed in the coverage report.

Outputs:
  paper_v3/figures/cross_section_fixed_fw.png       (bast, 6 multidir panels)
  paper_v3/figures/cross_section_unidir_fw.png      (bast, unidir90)
  paper_v3/figures/cross_section_fixed_tp_fw.png    (turbopark, 6 panels)
  paper_v3/figures/cross_section_unidir_tp_fw.png   (turbopark, unidir90)
"""

import json
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.projections.polar import PolarAxes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pathlib import Path

CASES = [
    ("DEI real\nwind rose", "dei"),
    ("$a$=0.9, $f$=1.0\n(conc. unidir)", "a0.9_f1.0"),
    ("$a$=0.5, $f$=1.0\n(mod. unidir)", "a0.5_f1.0"),
    ("$a$=0.7, $f$=0.5\n(mid-range)", "a0.7_f0.5"),
    ("$a$=0.9, $f$=0.0\n(conc. bidir)", "a0.9_f0.0"),
    ("$a$=0.5, $f$=0.0\n(mod. bidir)", "a0.5_f0.0"),
]
UNIDIR = ("Single direction\n(wind from E)", "unidir90")
DISTANCES = [2, 5, 10, 15, 20, 30, 40]
N_BEARINGS = 24


def get_wind_rose(case_dir):
    if case_dir == "unidir90":
        return np.array([90.0]), np.array([1.0])
    if case_dir == "dei":
        import pandas as pd
        df = pd.read_csv(Path("energy_island_10y_daily_av_wind.csv"), sep=";")
        wd_ts = df["WD_150"].values
        edges = np.linspace(0, 360, N_BEARINGS + 1)
        centers = (edges[:-1] + edges[1:]) / 2
        w = np.zeros(N_BEARINGS)
        for i in range(N_BEARINGS):
            if i == N_BEARINGS - 1:
                mask = (wd_ts >= edges[i]) | (wd_ts < edges[0])
            else:
                mask = (wd_ts >= edges[i]) & (wd_ts < edges[i + 1])
            w[i] = mask.sum()
        return centers, w / w.sum()
    a_val, f_val = (float(x) for x in
                    case_dir.replace("a", "").replace("f", "").split("_"))
    from edrose import EllipticalWindRose
    wr = EllipticalWindRose(a=a_val, f=f_val, theta_prev=270, n_sectors=24)
    return np.array(wr.wind_directions), np.array(wr.sector_frequencies)


def load_row(case, d, sfx):
    """Return (regret_row_gwh, lib_aep, bearings, tag) or None."""
    for base, tag in [
        (f"analysis/buffer_table{sfx}_funwake/{case}_d{d}/Nt50/results.json", "bt_k2000"),
        (f"analysis/cross_section_fixed{sfx}_funwake/{case}_d{d}/Nt50/results.json", "cs_k2000"),
    ]:
        p = Path(base)
        if p.exists():
            data = json.load(open(p))
            g = np.array(data["regret_grid_gwh"])
            return g[0, :], data["liberal_aep_gwh"], np.array(data["bearings_deg"]), tag
    for p in sorted(glob.glob(
            f"analysis/cross_section_fixed{sfx}_funwake_k500_packed/{case}*/Nt50/results.json")):
        data = json.load(open(p))
        ds = [float(x) for x in np.atleast_1d(data["distances_D"])]
        if float(d) in ds:
            k = ds.index(float(d))
            g = np.array(data["regret_grid_gwh"])
            return g[k, :], data["liberal_aep_gwh"], np.array(data["bearings_deg"]), "packed_k500"
    return None


def load_case(case, sfx):
    rows, lib, bearings, tags = [], None, None, []
    for d in DISTANCES:
        hit = load_row(case, d, sfx)
        if hit is None:
            rows.append(np.full(N_BEARINGS, np.nan))
            tags.append("MISSING")
            continue
        row, lib_aep, b, tag = hit
        rows.append(row)
        tags.append(tag)
        if lib is None:
            lib, bearings = lib_aep, b
    if bearings is None:
        return None
    grid = np.array(rows)
    return {"regret_grid_gwh": grid,
            "regret_grid_pct": 100 * grid / lib,
            "bearings_deg": bearings,
            "liberal_aep_gwh": lib,
            "tags": tags}


def polar_panel(ax, data, label, vmax):
    bearings = data["bearings_deg"]
    dists = np.array(DISTANCES, dtype=float)
    pct = data["regret_grid_pct"]
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    dbear = bearings[1] - bearings[0]
    bear_edges = np.radians(np.concatenate([bearings - dbear / 2,
                                            [bearings[-1] + dbear / 2]]))
    dist_edges = [dists[0] - (dists[1] - dists[0]) / 2]
    for i in range(len(dists) - 1):
        dist_edges.append((dists[i] + dists[i + 1]) / 2)
    dist_edges.append(dists[-1] + (dists[-1] - dists[-2]) / 2)
    dist_edges = np.maximum(np.array(dist_edges), 0)
    theta_grid, r_grid = np.meshgrid(bear_edges, dist_edges)
    im = ax.pcolormesh(theta_grid, r_grid, np.clip(pct, 0, None),
                       cmap="YlOrRd", vmin=0, vmax=vmax, shading="flat")
    if np.any(~np.isnan(pct)):
        idx = np.unravel_index(np.nanargmax(pct), pct.shape)
        ax.plot(np.radians(bearings[idx[1]]), dists[idx[0]],
                "k*", markersize=14, zorder=10)
    ax.set_title(label, fontsize=10, pad=15)
    ax.set_rlabel_position(135)
    ax.set_yticks([5, 10, 20, 30, 40])
    ax.set_yticklabels(["5D", "10D", "20D", "30D", "40D"], fontsize=7)
    ax.tick_params(axis="x", labelsize=8)
    return im


def add_rose_inset(ax, case_dir, size="35%"):
    wr_dirs, wr_freq = get_wind_rose(case_dir)
    ai = inset_axes(ax, width=size, height=size, loc="lower right",
                    axes_class=PolarAxes)
    ai.set_theta_zero_location("N")
    ai.set_theta_direction(-1)
    width = np.radians(min(360 / max(len(wr_dirs), 1), 15))
    ai.bar(np.radians(wr_dirs), wr_freq, width=width,
           color="steelblue", alpha=0.7, edgecolor="navy", linewidth=0.3)
    ai.set_yticks([])
    ai.set_xticks([])
    ai.patch.set_alpha(0.8)


for sfx, wake_label in [("", "Bastankhah"), ("_tp", "TurboPark")]:
    all_data = {}
    for label, case in CASES + [UNIDIR]:
        d = load_case(case, sfx)
        if d is not None:
            all_data[case] = d

    print(f"\n[{wake_label}] coverage:")
    for label, case in CASES + [UNIDIR]:
        if case not in all_data:
            print(f"  {case:<12} NO DATA")
            continue
        tags = all_data[case]["tags"]
        n_ok = sum(t != "MISSING" for t in tags)
        detail = ", ".join(f"d{d}:{t}" for d, t in zip(DISTANCES, tags))
        print(f"  {case:<12} {n_ok}/{len(DISTANCES)}  [{detail}]")

    have = [c for _, c in CASES if c in all_data]
    if have:
        vmax = max(np.nanmax(all_data[c]["regret_grid_pct"]) for c in have)
        fig, axes = plt.subplots(1, 6, figsize=(30, 5.5),
                                 subplot_kw={"projection": "polar"})
        im = None
        for ax, (label, case) in zip(axes, CASES):
            if case not in all_data:
                ax.set_title(label + "\n(no data)", fontsize=10, pad=15)
                continue
            im = polar_panel(ax, all_data[case], label, vmax)
            add_rose_inset(ax, case)
        fig.suptitle(f"Regret cross-sections, {wake_label} (FunWake iter192 schedule)\n"
                     "Identical reference farm at boundary-gap distances 2--40$D$",
                     fontsize=12, y=1.02)
        if im is not None:
            cbar_ax = fig.add_axes([0.92, 0.15, 0.012, 0.7])
            fig.colorbar(im, cax=cbar_ax, label="Design Regret (% of AEP)")
        plt.tight_layout(rect=[0, 0, 0.91, 0.95])
        out = Path(f"paper_v3/figures/cross_section_fixed{sfx}_fw.png")
        fig.savefig(str(out), dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")

    if UNIDIR[1] in all_data:
        label, case = UNIDIR
        data = all_data[case]
        fig2, ax2 = plt.subplots(1, 1, figsize=(7, 6.5),
                                 subplot_kw={"projection": "polar"})
        im2 = polar_panel(ax2, data, label,
                          float(np.nanmax(data["regret_grid_pct"])))
        add_rose_inset(ax2, case, size="28%")
        fig2.suptitle(f"Cross-section: single wind direction, {wake_label} "
                      "(FunWake iter192)", fontsize=11)
        fig2.colorbar(im2, ax=ax2, shrink=0.75,
                      label="Design Regret (% of AEP)", pad=0.12)
        plt.tight_layout()
        out2 = Path(f"paper_v3/figures/cross_section_unidir{sfx}_fw.png")
        fig2.savefig(str(out2), dpi=200, bbox_inches="tight")
        plt.close(fig2)
        print(f"Saved: {out2}")

    # summary
    print(f"[{wake_label}] peak regret per case:")
    for label, case in CASES + [UNIDIR]:
        if case not in all_data:
            continue
        g = all_data[case]["regret_grid_gwh"]
        pct = all_data[case]["regret_grid_pct"]
        if np.all(np.isnan(g)):
            continue
        idx = np.unravel_index(np.nanargmax(g), g.shape)
        print(f"  {case:<12} {g[idx]:>8.2f} GWh  {pct[idx]:>6.3f}%  "
              f"bearing {all_data[case]['bearings_deg'][idx[1]]:>4.0f}deg  "
              f"d={DISTANCES[idx[0]]}D")

"""Audit which FunWake cells are missing locally.

Lists every (study, config) cell that launch_all_funwake.sh would submit,
and marks DONE/MISSING based on analysis/<study>_funwake/.../results.json.
Prints by-study totals + a structured CSV of missing cells.
"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
ANALYSIS = ROOT / "analysis"


def buffer_table():
    """7 a × 5 f × 7 d = 245 cells per wake model."""
    out = []
    for a in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        for f in [0.0, 0.25, 0.5, 0.75, 1.0]:
            for d in [2, 5, 10, 15, 20, 30, 40]:
                for wm, dirsfx in [("bast", "funwake"), ("tp", "tp_funwake")]:
                    name = f"a{a}_f{f}_d{d}"
                    path = ANALYSIS / f"buffer_table_{dirsfx}" / name / "Nt50/results.json"
                    out.append(("buffer_table", wm, name, path))
    return out


def cross_section_fixed():
    """7 cases × 7 d = 49 per wake model."""
    cases = ["dei", "a0.5_f0.0", "a0.5_f1.0", "a0.7_f0.5", "a0.9_f0.0", "a0.9_f1.0", "unidir90"]
    out = []
    for c in cases:
        for d in [2, 5, 10, 15, 20, 30, 40]:
            for wm, dirsfx in [("bast", "funwake"), ("tp", "tp_funwake")]:
                name = f"{c}_d{d}"
                path = ANALYSIS / f"cross_section_fixed_{dirsfx}" / name / "Nt50/results.json"
                out.append(("cross_section_fixed", wm, name, path))
    return out


def ablation_a_sweep():
    """24 a × 2 d = 48 per wake model."""
    a_vals = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3,
              0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 1.0,
              1.5, 2.0, 3.0, 5.0]
    out = []
    for a in a_vals:
        for d in [2, 10]:
            for wm, dirsfx in [("bast", "funwake"), ("tp", "tp_funwake")]:
                name = f"a{a}_f1.0_d{d}"
                path = ANALYSIS / f"ablation_a_sweep_{dirsfx}" / name / "Nt50/results.json"
                out.append(("ablation_a_sweep", wm, name, path))
    return out


def ablation_f_sweep():
    """24 f × 1 d = 24 per wake model."""
    f_vals = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
              0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,
              1.0, 1.1, 1.25, 1.5]
    out = []
    for f in f_vals:
        for d in [2]:
            for wm, dirsfx in [("bast", "funwake"), ("tp", "tp_funwake")]:
                name = f"a0.9_f{f}_d{d}"
                path = ANALYSIS / f"ablation_f_sweep_{dirsfx}" / name / "Nt50/results.json"
                out.append(("ablation_f_sweep", wm, name, path))
    return out


def mixture_explore():
    """8 configs × 2 d = 16 per wake model."""
    configs = ["bidir_270_90", "ortho_270_180", "wind_strong_weak", "tri_270_180_90",
               "broad_two_modes", "narrow_two_modes", "asym_three_quart", "north_sea_like"]
    out = []
    for c in configs:
        for d in [2, 20]:
            for wm, dirsfx in [("bast", "funwake"), ("tp", "tp_funwake")]:
                name = f"{c}_d{d}"
                path = ANALYSIS / f"mixture_explore_{dirsfx}" / name / "Nt50/results.json"
                out.append(("mixture_explore", wm, name, path))
    return out


def n_target_sweep():
    """2 cases × 4 N × 4 d = 32 per wake model."""
    cases = ["dei", "a0.9_f1.0"]
    out = []
    for c in cases:
        for N in [25, 50, 75, 100]:
            for d in [2, 10, 20, 40]:
                for wm, dirsfx in [("bast", "funwake"), ("tp", "tp_funwake")]:
                    name = f"{c}_N{N}_d{d}"
                    path = ANALYSIS / f"n_target_sweep_{dirsfx}" / name / f"Nt{N}/results.json"
                    out.append(("n_target_sweep", wm, name, path))
    return out


def main():
    all_cells = (
        buffer_table()
        + cross_section_fixed()
        + ablation_a_sweep()
        + ablation_f_sweep()
        + mixture_explore()
        + n_target_sweep()
    )

    done_by = {}
    missing_by = {}
    missing_rows = []
    for study, wm, name, path in all_cells:
        key = (study, wm)
        if path.exists():
            done_by[key] = done_by.get(key, 0) + 1
        else:
            missing_by[key] = missing_by.get(key, 0) + 1
            missing_rows.append((study, wm, name, str(path)))

    print(f"{'study':<22s} {'wake':<6s} {'done':>6s} {'miss':>6s}")
    studies = sorted({(s, w) for s, w, _, _ in all_cells})
    total_done = 0
    total_miss = 0
    for s, w in studies:
        d = done_by.get((s, w), 0)
        m = missing_by.get((s, w), 0)
        total_done += d
        total_miss += m
        print(f"{s:<22s} {w:<6s} {d:>6d} {m:>6d}")
    print(f"{'TOTAL':<22s} {'':<6s} {total_done:>6d} {total_miss:>6d}")
    print(f"\nTotal cells in scope: {len(all_cells)}")

    out_csv = ROOT / "scripts/funwake_missing_cells.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["study", "wake", "name", "expected_path"])
        w.writerows(missing_rows)
    print(f"\nWrote {out_csv}  ({len(missing_rows)} rows)")


if __name__ == "__main__":
    main()

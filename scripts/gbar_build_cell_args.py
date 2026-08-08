"""Build per-cell python-args CSV for gbar LSF array job.

Mirrors launch_all_funwake.sh logic but emits CSV rows for ONLY the missing
cells (those listed in funwake_missing_cells.csv).

Output columns: idx, study, wake, name, outdir, python_args
"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

CS_FW_BAST = (
    "scripts/run_regret_cross_section.py --n-bearings 24 --n-inner-starts 2000 "
    "--inner-max-iter 5000 --k-liberal 2000 --deficit bastankhah "
    "--schedule funwake_iter192 --chunk-size 25 --wind-speed 9.0 --n-bins 24"
)
CS_FW_TP = (
    "scripts/run_regret_cross_section.py --n-bearings 24 --n-inner-starts 2000 "
    "--inner-max-iter 5000 --k-liberal 2000 --deficit turbopark --ti 0.06 "
    "--schedule funwake_iter192 --chunk-size 25 --wind-speed 9.0 --n-bins 24"
)


def base(wm):
    return CS_FW_BAST if wm == "bast" else CS_FW_TP


def dirsfx(wm):
    return "funwake" if wm == "bast" else "tp_funwake"


def emit_all_cells():
    """Yield (study, wake, name, outdir, args, walltime_hr) for every launcher cell."""
    # buffer_table
    for a in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        for f in [0.0, 0.25, 0.5, 0.75, 1.0]:
            for d in [2, 5, 10, 15, 20, 30, 40]:
                for wm in ["bast", "tp"]:
                    name = f"a{a}_f{f}_d{d}"
                    outdir = f"analysis/buffer_table_{dirsfx(wm)}/{name}"
                    args = f"{base(wm)} --distances-D {d} --wind-rose elliptical --ed-a {a} --ed-f {f} --wind-dir 270"
                    yield ("buffer_table", wm, name, outdir, args, 48)
    # cross_section_fixed
    cases = [
        ("dei", "--wind-rose dei"),
        ("a0.5_f0.0", "--wind-rose elliptical --ed-a 0.5 --ed-f 0.0 --wind-dir 270"),
        ("a0.5_f1.0", "--wind-rose elliptical --ed-a 0.5 --ed-f 1.0 --wind-dir 270"),
        ("a0.7_f0.5", "--wind-rose elliptical --ed-a 0.7 --ed-f 0.5 --wind-dir 270"),
        ("a0.9_f0.0", "--wind-rose elliptical --ed-a 0.9 --ed-f 0.0 --wind-dir 270"),
        ("a0.9_f1.0", "--wind-rose elliptical --ed-a 0.9 --ed-f 1.0 --wind-dir 270"),
        ("unidir90", "--wind-rose unidirectional --wind-dir 90"),
    ]
    for cname, rose in cases:
        for d in [2, 5, 10, 15, 20, 30, 40]:
            for wm in ["bast", "tp"]:
                name = f"{cname}_d{d}"
                outdir = f"analysis/cross_section_fixed_{dirsfx(wm)}/{name}"
                args = f"{base(wm)} --distances-D {d} {rose}"
                yield ("cross_section_fixed", wm, name, outdir, args, 48)
    # ablation_a_sweep
    a_vals = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3,
              0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 1.0,
              1.5, 2.0, 3.0, 5.0]
    for a in a_vals:
        for d in [2, 10]:
            for wm in ["bast", "tp"]:
                name = f"a{a}_f1.0_d{d}"
                outdir = f"analysis/ablation_a_sweep_{dirsfx(wm)}/{name}"
                args = f"{base(wm)} --distances-D {d} --wind-rose elliptical --ed-a {a} --ed-f 1.0 --wind-dir 270"
                yield ("ablation_a_sweep", wm, name, outdir, args, 48)
    # ablation_f_sweep
    f_vals = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
              0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,
              1.0, 1.1, 1.25, 1.5]
    for fv in f_vals:
        for d in [2]:
            for wm in ["bast", "tp"]:
                name = f"a0.9_f{fv}_d{d}"
                outdir = f"analysis/ablation_f_sweep_{dirsfx(wm)}/{name}"
                args = f"{base(wm)} --distances-D {d} --wind-rose elliptical --ed-a 0.9 --ed-f {fv} --wind-dir 270"
                yield ("ablation_f_sweep", wm, name, outdir, args, 48)
    # mixture_explore
    configs = [
        ("bidir_270_90",     "0.9 1.0 270", "0.9 1.0 90",  "0.5"),
        ("ortho_270_180",    "0.9 1.0 270", "0.9 1.0 180", "0.5"),
        ("wind_strong_weak", "0.9 1.0 270", "0.5 0.5 90",  "0.7"),
        ("tri_270_180_90",   "0.9 1.0 270", "0.7 0.5 180", "0.6"),
        ("broad_two_modes",  "0.5 0.5 270", "0.5 0.5 90",  "0.5"),
        ("narrow_two_modes", "1.5 1.0 270", "1.5 1.0 90",  "0.5"),
        ("asym_three_quart", "0.9 1.0 270", "0.7 0.5 90",  "0.75"),
        ("north_sea_like",   "0.7 0.4 240", "0.7 0.4 60",  "0.7"),
    ]
    for cname, p1, p2, w in configs:
        a1, f1, d1 = p1.split()
        a2, f2, d2 = p2.split()
        for d in [2, 20]:
            for wm in ["bast", "tp"]:
                name = f"{cname}_d{d}"
                outdir = f"analysis/mixture_explore_{dirsfx(wm)}/{name}"
                args = (f"{base(wm)} --distances-D {d} --wind-rose mixture "
                        f"--ed-a {a1} --ed-f {f1} --wind-dir {d1} "
                        f"--ed-a2 {a2} --ed-f2 {f2} --wind-dir2 {d2} "
                        f"--mixture-weight {w}")
                yield ("mixture_explore", wm, name, outdir, args, 48)
    # n_target_sweep
    nts_cases = [
        ("dei", "--wind-rose dei"),
        ("a0.9_f1.0", "--wind-rose elliptical --ed-a 0.9 --ed-f 1.0 --wind-dir 270"),
    ]
    for cname, rose in nts_cases:
        for N in [25, 50, 75, 100]:
            for d in [2, 10, 20, 40]:
                for wm in ["bast", "tp"]:
                    name = f"{cname}_N{N}_d{d}"
                    outdir = f"analysis/n_target_sweep_{dirsfx(wm)}/{name}"
                    args = f"{base(wm)} --n-target {N} --distances-D {d} {rose}"
                    yield ("n_target_sweep", wm, name, outdir, args, 72)


def main():
    miss = ROOT / "scripts/funwake_missing_cells.csv"
    missing = set()
    with open(miss) as f:
        r = csv.DictReader(f)
        for row in r:
            missing.add((row["study"], row["wake"], row["name"]))

    rows = []
    for study, wm, name, outdir, args, wt in emit_all_cells():
        if (study, wm, name) in missing:
            rows.append((study, wm, name, outdir, args, wt))

    out = ROOT / "scripts/gbar_funwake_jobs.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "study", "wake", "name", "outdir", "walltime_hr", "python_args"])
        for i, (study, wm, name, outdir, args, wt) in enumerate(rows, start=1):
            w.writerow([i, study, wm, name, outdir, wt, args])
    print(f"Wrote {out}  ({len(rows)} rows, expected ~301)")


if __name__ == "__main__":
    main()

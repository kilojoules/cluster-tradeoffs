"""Build packed-cell CSV for gbar array.

Packs distances into single runs: one liberal + N inner sweeps.
Skips cells whose results.json already exists locally.
Groups by (study, wake, name_key) where name_key is the cell name
minus the distance suffix.

Output CSV columns: idx, study, wake, name_key, outdir_pattern, walltime_hr,
                    n_distances, python_args
where python_args uses `--distances-D d1,d2,...` and `--output-dir <pattern>`
that includes `_d{d}` templated via runner's default behavior.

BUT: runner writes to `--output-dir` and puts Nt{N}/results.json under
it, one merged file across all distances. That's a semantic change.

Actually re-read: results_grid is (n_distances, n_bearings). The results.json
has "distances_D" as a list. So one packed call → one JSON per (rose, wake, N)
containing all distances.

For downstream analysis compatibility, we need separate directories per
distance OR downstream must adapt.

Approach: keep one results.json per packed call. Downstream code that reads
`analysis/buffer_table_funwake/a{a}_f{f}_d{d}/Nt50/results.json` needs to
read `analysis/buffer_table_funwake_packed/a{a}_f{f}/Nt50/results.json`
and slice by distance.

For now: emit packed jobs. Downstream loader change is separate.
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
ANALYSIS = ROOT / "analysis"

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
    """Yield (study, wake, name_key, distance, rose_args, n_target)."""
    # buffer_table: name = a{a}_f{f}_d{d}, key = a{a}_f{f}
    for a in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        for f in [0.0, 0.25, 0.5, 0.75, 1.0]:
            for d in [2, 5, 10, 15, 20, 30, 40]:
                for wm in ["bast", "tp"]:
                    key = f"a{a}_f{f}"
                    rose = f"--wind-rose elliptical --ed-a {a} --ed-f {f} --wind-dir 270"
                    yield ("buffer_table", wm, key, d, rose, 50)
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
                yield ("cross_section_fixed", wm, cname, d, rose, 50)
    # ablation_a_sweep
    a_vals = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3,
              0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 1.0,
              1.5, 2.0, 3.0, 5.0]
    for a in a_vals:
        for d in [2, 10]:
            for wm in ["bast", "tp"]:
                key = f"a{a}_f1.0"
                rose = f"--wind-rose elliptical --ed-a {a} --ed-f 1.0 --wind-dir 270"
                yield ("ablation_a_sweep", wm, key, d, rose, 50)
    # ablation_f_sweep — single distance, no packing benefit but emit anyway
    f_vals = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
              0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,
              1.0, 1.1, 1.25, 1.5]
    for fv in f_vals:
        for wm in ["bast", "tp"]:
            key = f"a0.9_f{fv}"
            rose = f"--wind-rose elliptical --ed-a 0.9 --ed-f {fv} --wind-dir 270"
            yield ("ablation_f_sweep", wm, key, 2, rose, 50)
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
        rose = (f"--wind-rose mixture --ed-a {a1} --ed-f {f1} --wind-dir {d1} "
                f"--ed-a2 {a2} --ed-f2 {f2} --wind-dir2 {d2} --mixture-weight {w}")
        for d in [2, 20]:
            for wm in ["bast", "tp"]:
                yield ("mixture_explore", wm, cname, d, rose, 50)
    # n_target_sweep — key includes N since results dir differs
    nts_cases = [
        ("dei", "--wind-rose dei"),
        ("a0.9_f1.0", "--wind-rose elliptical --ed-a 0.9 --ed-f 1.0 --wind-dir 270"),
    ]
    for cname, rose in nts_cases:
        for N in [25, 50, 75, 100]:
            for d in [2, 10, 20, 40]:
                for wm in ["bast", "tp"]:
                    key = f"{cname}_N{N}"
                    yield ("n_target_sweep", wm, key, d, rose, N)


def cell_result_path(study, wm, key, d, N):
    """Path to unpacked results.json — matches old launcher schema."""
    if study == "n_target_sweep":
        cname = key.rsplit("_N", 1)[0]
        return ANALYSIS / f"{study}_{dirsfx(wm)}" / f"{cname}_N{N}_d{d}" / f"Nt{N}/results.json"
    else:
        return ANALYSIS / f"{study}_{dirsfx(wm)}" / f"{key}_d{d}" / f"Nt{N}/results.json"


def packed_result_path(study, wm, key, N):
    """Path packed job writes to — one JSON per (study, wake, key, N)."""
    if study == "n_target_sweep":
        cname = key.rsplit("_N", 1)[0]
        return ANALYSIS / f"{study}_{dirsfx(wm)}_packed" / f"{cname}_N{N}" / f"Nt{N}/results.json"
    else:
        return ANALYSIS / f"{study}_{dirsfx(wm)}_packed" / key / f"Nt{N}/results.json"


def packed_outdir(study, wm, key, N):
    """--output-dir arg."""
    if study == "n_target_sweep":
        cname = key.rsplit("_N", 1)[0]
        return f"analysis/{study}_{dirsfx(wm)}_packed/{cname}_N{N}"
    else:
        return f"analysis/{study}_{dirsfx(wm)}_packed/{key}"


def main():
    # Group cells by (study, wake, key, N, rose_args) → set of distances
    groups = defaultdict(lambda: {"distances": set(), "rose": None, "n_target": None})
    for study, wm, key, d, rose, N in emit_all_cells():
        g = groups[(study, wm, key)]
        g["distances"].add(d)
        g["rose"] = rose
        g["n_target"] = N

    # For each group, drop distances whose unpacked result.json already exists locally.
    # If the packed result.json exists, skip the whole group.
    rows = []
    stats = defaultdict(int)
    for (study, wm, key), g in sorted(groups.items()):
        N = g["n_target"]
        if packed_result_path(study, wm, key, N).exists():
            stats["packed_done"] += 1
            continue

        # Remaining distances = distances whose unpacked JSON is missing.
        remaining = sorted(d for d in g["distances"]
                           if not cell_result_path(study, wm, key, d, N).exists())
        if not remaining:
            stats["all_unpacked_done"] += 1
            continue

        distances_str = ",".join(str(int(d)) if d == int(d) else str(d) for d in remaining)
        outdir = packed_outdir(study, wm, key, N)
        args = f"{base(wm)} --distances-D {distances_str} {g['rose']}"
        if N != 50:
            args += f" --n-target {N}"

        # Walltime estimate: 4hr liberal + n_distances * 8hr inner, with 30% buffer.
        est = int(4 + len(remaining) * 8)
        wallhr = min(72, max(24, int(est * 1.3)))

        rows.append({
            "study": study, "wake": wm, "name_key": key,
            "n_target": N, "n_distances": len(remaining),
            "outdir": outdir, "walltime_hr": wallhr,
            "distances": distances_str, "args": args,
        })
        stats["packed_to_run"] += 1
        stats[f"packed_{study}"] += 1

    out = ROOT / "scripts/gbar_packed_jobs.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "study", "wake", "name_key", "n_target",
                    "n_distances", "outdir", "walltime_hr",
                    "distances", "python_args"])
        for i, r in enumerate(rows, start=1):
            w.writerow([i, r["study"], r["wake"], r["name_key"], r["n_target"],
                        r["n_distances"], r["outdir"], r["walltime_hr"],
                        r["distances"], r["args"]])
    print(f"Wrote {out}  ({len(rows)} rows)")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

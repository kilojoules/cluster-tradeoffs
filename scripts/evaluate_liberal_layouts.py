"""Evaluate standard liberal layouts against all neighbor configurations.

The liberal optimizer ignores neighbors, so liberal layouts from any farm case are
equivalent (same x,y coordinates for the same seed). However, aep_present values
differ per neighbor configuration because wake losses depend on which neighbors exist.

This script takes liberal layout coordinates from a source farm (default: farm 1)
and evaluates their AEP against a specified neighbor case using the full timeseries
wake model.

Usage:
    # Evaluate layouts 0-19 against farm 8 (for batch processing):
    pixi run python scripts/evaluate_liberal_layouts.py -i analysis/dei_A0.04 --case farm8 --A 0.04 --start 0 --count 20

    # Evaluate all against farm 8 (will OOM on login node — use SLURM):
    pixi run python scripts/evaluate_liberal_layouts.py -i analysis/dei_A0.04 --case farm8 --A 0.04
"""

import argparse
import time
from pathlib import Path

import h5py
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)

from run_dei_single_neighbor import (
    N_NEIGHBOR_FARMS,
    create_dei_turbine,
    create_full_timeseries_aep_evaluator,
    load_neighbor_layout,
    load_wind_data,
    TurboGaussianDeficit,
    SquaredSum,
    WakeSimulation,
    ct2a_mom1d,
)


def load_liberal_coordinates(analysis_dir, source="farm1"):
    """Load liberal layout coordinates (x, y) and seeds from a source HDF5 file.

    Returns list of dicts with keys: x, y, seed, aep_absent_original.
    Sorted by seed number.
    """
    if source == "combined":
        h5_path = analysis_dir / "layouts_combined.h5"
    else:
        farm_idx = int(source.replace("farm", ""))
        h5_path = analysis_dir / f"layouts_farm{farm_idx}.h5"

    if not h5_path.exists():
        raise FileNotFoundError(f"Source file not found: {h5_path}")

    layouts = []
    with h5py.File(h5_path, "r") as f:
        for key in f.keys():
            if not key.startswith("layout_"):
                continue
            grp = f[key]
            if grp.attrs["strategy"] != "liberal":
                continue
            layouts.append({
                "x": grp["x"][:],
                "y": grp["y"][:],
                "seed": int(grp.attrs["seed"]),
                "aep_absent_original": float(grp.attrs["aep_absent"]),
            })

    # Sort by seed for deterministic ordering
    layouts.sort(key=lambda l: l["seed"])
    print(f"Loaded {len(layouts)} liberal layouts from {h5_path}")
    return layouts


def load_neighbor_positions(case):
    """Load neighbor positions for a given case.

    Args:
        case: 'farm1'..'farm9' or 'combined'

    Returns:
        x_neighbor, y_neighbor arrays
    """
    if case == "combined":
        x_all, y_all = [], []
        for farm_idx in range(1, N_NEIGHBOR_FARMS + 1):
            x_n, y_n = load_neighbor_layout(farm_idx)
            if x_n is not None:
                x_all.extend(x_n)
                y_all.extend(y_n)
        return np.array(x_all), np.array(y_all)
    else:
        farm_idx = int(case.replace("farm", ""))
        x_n, y_n = load_neighbor_layout(farm_idx)
        if x_n is None:
            raise ValueError(f"No layout found for {case}")
        return x_n, y_n


def get_completed_seeds(output_path):
    """Get set of seeds already evaluated in the output file."""
    if not output_path.exists():
        return set()
    with h5py.File(output_path, "r") as f:
        return {int(f[k].attrs["seed"]) for k in f.keys() if k.startswith("layout_")}


def save_results_incremental(results, output_path):
    """Append evaluated layouts to HDF5, preserving existing entries."""
    with h5py.File(output_path, "a") as f:
        existing_count = len([k for k in f.keys() if k.startswith("layout_")])
        for layout in results:
            grp = f.create_group(f"layout_{existing_count}")
            grp.create_dataset("x", data=layout["x"])
            grp.create_dataset("y", data=layout["y"])
            grp.attrs["aep_absent"] = layout["aep_absent"]
            grp.attrs["aep_present"] = layout["aep_present"]
            grp.attrs["seed"] = layout["seed"]
            grp.attrs["strategy"] = layout["strategy"]
            existing_count += 1
        f.attrs["n_layouts"] = existing_count

    print(f"Saved {len(results)} layouts to {output_path} (total: {existing_count})")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate standard liberal layouts against neighbor configurations"
    )
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Analysis directory (e.g., analysis/dei_A0.04)")
    parser.add_argument("--source", type=str, default="farm1",
                        help="Source of liberal layouts (default: farm1)")
    parser.add_argument("--case", type=str, required=True,
                        help="Neighbor case to evaluate: farm1..farm9 or combined")
    parser.add_argument("--A", type=float, default=0.04,
                        help="Wake expansion coefficient A")
    parser.add_argument("--ti", type=float, default=0.06,
                        help="Ambient turbulence intensity")
    parser.add_argument("--start", type=int, default=0,
                        help="Index of first layout to evaluate (for batch processing)")
    parser.add_argument("--count", type=int, default=0,
                        help="Number of layouts to evaluate (0 = all remaining)")
    parser.add_argument("--output-prefix", type=str, default="liberal_standard",
                        help="Prefix for output files")

    args = parser.parse_args()
    analysis_dir = Path(args.input)

    # Load liberal layouts from source
    all_layouts = load_liberal_coordinates(analysis_dir, source=args.source)
    if not all_layouts:
        print("No liberal layouts found!")
        return

    # Determine output path
    if args.case == "combined":
        output_path = analysis_dir / f"{args.output_prefix}_combined.h5"
    else:
        output_path = analysis_dir / f"{args.output_prefix}_{args.case}.h5"

    # Skip already-completed seeds
    completed_seeds = get_completed_seeds(output_path)
    if completed_seeds:
        print(f"Found {len(completed_seeds)} already-completed seeds in {output_path}")

    # Select the subset to evaluate
    end = args.start + args.count if args.count > 0 else len(all_layouts)
    end = min(end, len(all_layouts))
    layouts = all_layouts[args.start:end]

    # Filter out already-completed seeds
    layouts = [l for l in layouts if l["seed"] not in completed_seeds]
    if not layouts:
        print(f"All {end - args.start} layouts in range already completed!")
        return

    print(f"Will evaluate {len(layouts)} layouts (indices {args.start}-{end-1}, "
          f"skipped {(end - args.start) - len(layouts)} already done)")

    # Set up wake model (TurboGaussian, matching run_dei_single_neighbor.py)
    print(f"\nSetting up wake model (TurboGaussian, A={args.A})...")
    turbine = create_dei_turbine()
    deficit = TurboGaussianDeficit(
        A=args.A,
        ct2a=ct2a_mom1d,
        ctlim=0.96,
        use_effective_ws=False,
        use_effective_ti=False,
        superposition=SquaredSum(),
    )
    sim = WakeSimulation(turbine, deficit)

    # Load wind data and create evaluator
    wd_ts, ws_ts = load_wind_data()
    print(f"Creating full timeseries AEP evaluator ({len(wd_ts)} samples)...")
    compute_aep = create_full_timeseries_aep_evaluator(
        sim, wd_ts, ws_ts, ti_amb=args.ti, batch_size=500
    )

    # Load neighbor positions
    x_neighbor, y_neighbor = load_neighbor_positions(args.case)
    print(f"Neighbor turbines: {len(x_neighbor)}")

    import jax.numpy as jnp

    # Evaluate layouts
    print(f"\nEvaluating {len(layouts)} layouts against {args.case}...")
    results = []
    t0 = time.time()

    for i, layout in enumerate(layouts):
        x = jnp.array(layout["x"])
        y = jnp.array(layout["y"])

        aep_absent = compute_aep(x, y, None, None)
        aep_present = compute_aep(x, y, x_neighbor, y_neighbor)

        results.append({
            "x": layout["x"],
            "y": layout["y"],
            "seed": layout["seed"],
            "aep_absent": float(aep_absent),
            "aep_present": float(aep_present),
            "strategy": "liberal",
        })

        if (i + 1) % 5 == 0 or i == len(layouts) - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(layouts) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{len(layouts)}] seed={layout['seed']}, "
                  f"absent={aep_absent:.1f}, present={aep_present:.1f} GWh, "
                  f"ETA {eta:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/len(results):.1f}s per layout)")

    # Save incrementally
    save_results_incremental(results, output_path)

    # Quick sanity check: aep_absent should match source
    aep_absent_orig = np.array([l["aep_absent_original"] for l in layouts])
    aep_absent_new = np.array([r["aep_absent"] for r in results])
    max_diff = np.max(np.abs(aep_absent_orig - aep_absent_new))
    print(f"aep_absent verification: max diff = {max_diff:.6f} GWh "
          f"({'OK' if max_diff < 0.1 else 'WARNING: large difference!'})")


if __name__ == "__main__":
    main()

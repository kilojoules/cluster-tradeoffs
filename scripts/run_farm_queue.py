#!/usr/bin/env python
"""Queue manager: keeps MAX_PARALLEL optimization jobs running at all times."""

import argparse
import subprocess
import sys
import time
import h5py
from pathlib import Path


def count_starts(h5_path):
    """Count completed starts in an HDF5 file."""
    if not h5_path.exists():
        return 0
    with h5py.File(h5_path, 'r') as f:
        return len([k for k in f.keys() if k.startswith('layout_')]) // 2


def main():
    parser = argparse.ArgumentParser(description='Queue manager for farm optimizations')
    parser.add_argument('--max-parallel', type=int, default=2)
    parser.add_argument('--n-starts', type=int, default=500)
    parser.add_argument('--max-iter', type=int, default=2000)
    parser.add_argument('--A', type=float, default=0.02)
    parser.add_argument('--farms', type=int, nargs='+', default=list(range(1, 10)))
    args = parser.parse_args()

    output_dir = Path(f'analysis/dei_A{args.A}')
    log_dir = output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    farms_todo = list(args.farms)
    running = {}  # farm -> subprocess.Popen

    print(f"=== Farm Queue Manager ===")
    print(f"max_parallel={args.max_parallel}, n_starts={args.n_starts}, "
          f"max_iter={args.max_iter}, A={args.A}")
    print(f"Farms: {farms_todo}")
    print()

    while farms_todo or running:
        # Check for finished jobs
        for farm in list(running.keys()):
            proc = running[farm]
            if proc.poll() is not None:
                h5_path = output_dir / f'layouts_farm{farm}.h5'
                n = count_starts(h5_path)
                print(f"[{time.strftime('%H:%M:%S')}] Farm {farm} finished ({n} starts)")
                del running[farm]

        # Launch new jobs if capacity available
        while len(running) < args.max_parallel and farms_todo:
            farm = farms_todo.pop(0)
            h5_path = output_dir / f'layouts_farm{farm}.h5'
            n = count_starts(h5_path)

            if n >= args.n_starts:
                print(f"[{time.strftime('%H:%M:%S')}] Farm {farm}: already has {n}/{args.n_starts} starts, skipping")
                continue

            log_file = log_dir / f'farm{farm}_auto.log'
            print(f"[{time.strftime('%H:%M:%S')}] Starting farm {farm} ({n}/{args.n_starts} starts done)")

            proc = subprocess.Popen(
                ['pixi', 'run', 'python', 'scripts/run_dei_single_neighbor.py',
                 f'--n-starts={args.n_starts}', f'--max-iter={args.max_iter}',
                 f'--A={args.A}', f'--farm={farm}', '--skip-combined',
                 '--seed-offset=0', f'-o={output_dir}'],
                stdout=open(log_file, 'w'),
                stderr=subprocess.STDOUT,
            )
            running[farm] = proc

        if not running and not farms_todo:
            break

        # Status
        running_farms = ', '.join(str(f) for f in sorted(running.keys()))
        counts = []
        for f in sorted(running.keys()):
            h5_path = output_dir / f'layouts_farm{f}.h5'
            counts.append(f"F{f}={count_starts(h5_path)}")
        print(f"[{time.strftime('%H:%M:%S')}] Running: [{running_farms}] | {' '.join(counts)}")

        time.sleep(60)

    print()
    print("=== All farms complete! ===")
    for farm in args.farms:
        h5_path = output_dir / f'layouts_farm{farm}.h5'
        print(f"Farm {farm}: {count_starts(h5_path)} starts")


if __name__ == '__main__':
    main()

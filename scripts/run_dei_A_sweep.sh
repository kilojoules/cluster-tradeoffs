#!/bin/bash
# DEI Analysis A Parameter Sweep
# Runs all 9 farms + combined for A values from 0.02 to 0.20

set -e

N_STARTS=50
MAX_ITER=2000

for A in 0.02 0.04 0.06 0.08 0.10 0.12 0.14 0.16 0.18 0.20; do
    echo "========================================"
    echo "Running A=${A}"
    echo "========================================"

    OUTPUT_DIR="analysis/dei_A${A}"

    pixi run python scripts/run_dei_single_neighbor.py \
        --wake-model=turbopark \
        --A=${A} \
        --n-starts=${N_STARTS} \
        --max-iter=${MAX_ITER} \
        --output-dir=${OUTPUT_DIR}

    echo "Completed A=${A}"
    echo ""
done

echo "========================================"
echo "A PARAMETER SWEEP COMPLETE"
echo "========================================"

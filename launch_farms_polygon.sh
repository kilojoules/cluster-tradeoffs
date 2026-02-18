#!/bin/bash
# Launch 27 job arrays: 3 A values × 9 farms, 20 array tasks each (100 seeds total)
# Each array task runs 5 seeds with local-env caching.

set -e

WORKDIR=/work/users/juqu/cluster-tradeoffs
cd "${WORKDIR}"

A_VALUES=(0.04 0.08 0.12)
FARMS=(1 2 3 4 5 6 7 8 9)

# Clear old invalid (bounding-box) individual farm results
echo "Clearing old individual farm HDF5 files..."
for A in "${A_VALUES[@]}"; do
    DIR="analysis/dei_A${A}"
    if [ -d "${DIR}" ]; then
        rm -f "${DIR}"/layouts_farm*.h5
        echo "  Cleared ${DIR}/layouts_farm*.h5"
    fi
done

# Submit one job array per (A, farm) pair
echo ""
echo "Submitting 27 job arrays..."
for A in "${A_VALUES[@]}"; do
    for FARM in "${FARMS[@]}"; do
        JOB_ID=$(sbatch --array=0-19 \
            --export=ALL,SWEEP_A=${A},SWEEP_FARM=${FARM} \
            --job-name="f${FARM}_A${A}" \
            --output="analysis/dei_A${A}/farm${FARM}_%a.out" \
            run_farms_polygon.sh | awk '{print $4}')
        echo "  A=${A} Farm=${FARM} -> Job ${JOB_ID} (20 tasks)"
    done
done

echo ""
echo "Done. 27 jobs × 20 tasks = 540 array tasks submitted."
echo "Each task runs 5 seeds => 100 seeds per (A, farm) pair."
echo "Monitor: squeue -u $(whoami) | head -30"

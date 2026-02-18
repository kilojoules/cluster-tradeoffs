#!/bin/bash
# Launch combined case (all 9 neighbors) for A=0.04, 0.08, 0.12
# 3 job arrays × 20 tasks = 60 array tasks, 100 seeds per A value.

set -e

WORKDIR=/work/users/juqu/cluster-tradeoffs
cd "${WORKDIR}"

A_VALUES=(0.04 0.08 0.12)

# Clear old invalid combined results
echo "Clearing old combined HDF5 files..."
for A in "${A_VALUES[@]}"; do
    DIR="analysis/dei_A${A}"
    if [ -d "${DIR}" ]; then
        rm -f "${DIR}"/layouts_combined*.h5
        echo "  Cleared ${DIR}/layouts_combined*.h5"
    fi
done

echo ""
echo "Submitting combined case job arrays..."
for A in "${A_VALUES[@]}"; do
    JOB_ID=$(sbatch --array=0-19 \
        --export=ALL,SWEEP_A=${A} \
        --job-name="cA${A}" \
        --output="analysis/dei_A${A}/combined_%a.out" \
        run_combined_polygon.sh | awk '{print $4}')
    echo "  A=${A} -> Job ${JOB_ID} (20 tasks, 100 seeds)"
done

echo ""
echo "Done. 3 jobs × 20 tasks = 60 array tasks submitted."

#!/bin/bash
# Launch all polygon-constrained jobs:
#   - Individual farms 1-9, A=0.04 only (9 job arrays)
#   - Combined case, A=0.04/0.08/0.12 (3 job arrays)
# Total: 12 job arrays × 20 tasks = 240 tasks
# Staggered launches (10s between jobs) to avoid BeeGFS tarball read storms.

set -e

WORKDIR=/work/users/juqu/cluster-tradeoffs
cd "${WORKDIR}"

echo "=== Individual farms (A=0.04) ==="
for FARM in 1 2 3 4 5 6 7 8 9; do
    JOB_ID=$(sbatch --array=0-19 \
        --export=ALL,SWEEP_A=0.04,SWEEP_FARM=${FARM} \
        --job-name="f${FARM}_A0.04" \
        --output="analysis/dei_A0.04/farm${FARM}_%a.out" \
        run_farms_polygon.sh | awk '{print $4}')
    echo "  Farm ${FARM} -> Job ${JOB_ID}"
    sleep 10
done

echo ""
echo "=== Combined case ==="
for A in 0.04 0.08 0.12; do
    JOB_ID=$(sbatch --array=0-19 \
        --export=ALL,SWEEP_A=${A} \
        --job-name="cA${A}" \
        --output="analysis/dei_A${A}/combined_%a.out" \
        run_combined_polygon.sh | awk '{print $4}')
    echo "  A=${A} -> Job ${JOB_ID}"
    sleep 10
done

echo ""
echo "Done. 12 jobs × 20 tasks = 240 tasks (100 seeds each)."
echo "Monitor: squeue -u $(whoami)"

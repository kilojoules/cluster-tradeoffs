#!/bin/bash
#SBATCH --partition=workq
#SBATCH --job-name="farms04_2k"
#SBATCH --output=analysis/dei_A0.04/logs/farm_%a_2k.out
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --array=1-9

# Individual farm optimization for A=0.04 — 2000 seeds target
# One farm per array task, with batch loop to avoid XLA memory leak.
# Launch: sbatch run_farms_A0.04_2k.sh
#
# Current state (farms 1-8 have 500 seeds, farm 9 has 119).
# Existing seeds are automatically skipped.

. ~/.bashrc

export JAX_ENABLE_X64=True
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

FARM=$SLURM_ARRAY_TASK_ID

echo "=== Farm ${FARM}, A=0.04, target=2000 seeds ==="
echo "Host: $(hostname)"
echo "Start: $(date)"

mkdir -p analysis/dei_A0.04/logs

TOTAL_SEEDS=2000
BATCH_SIZE=5

for (( OFFSET=0; OFFSET<TOTAL_SEEDS; OFFSET+=BATCH_SIZE )); do
    echo "--- Batch: seeds ${OFFSET}-$((OFFSET+BATCH_SIZE-1)) [$(date)] ---"
    pixi run python -u scripts/run_dei_single_neighbor.py \
        --skip-combined \
        --farm=${FARM} \
        --n-starts=${BATCH_SIZE} \
        --max-iter=2000 \
        --A=0.04 \
        --seed-offset=${OFFSET} \
        -o analysis/dei_A0.04
    RC=$?
    if [ $RC -ne 0 ]; then
        echo "WARNING: Batch at offset ${OFFSET} exited with code ${RC}, continuing..."
    fi
done

echo "End: $(date)"

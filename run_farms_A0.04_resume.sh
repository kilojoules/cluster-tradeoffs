#!/bin/bash
#SBATCH --partition=workq
#SBATCH --job-name="farms04_r"
#SBATCH --output=analysis/dei_A0.04/logs/farm_%a_resume.out
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --array=1-9

# Resume individual farm optimization for A=0.04
# Picks up from seed 900 (farm 9 stopped there; farms 1-8 at ~1250).
# Farms 1-8 will redo ~350 seeds but that's much better than 1250.

. ~/.bashrc

export JAX_ENABLE_X64=True
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

FARM=$SLURM_ARRAY_TASK_ID

echo "=== Farm ${FARM}, A=0.04, RESUME from seed 900 ==="
echo "Host: $(hostname)"
echo "Start: $(date)"

mkdir -p analysis/dei_A0.04/logs

START_OFFSET=900
TOTAL_SEEDS=2000
BATCH_SIZE=5

for (( OFFSET=START_OFFSET; OFFSET<TOTAL_SEEDS; OFFSET+=BATCH_SIZE )); do
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

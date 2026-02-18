#!/bin/bash
#SBATCH --partition=windq
#SBATCH --job-name="sweep_comb"
#SBATCH --output=analysis/sweep_combined_%a.out
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --exclusive

# A-value sweep: combined case (all 9 neighbors) only
# Array index -> A value mapping:
#   1=0.02  2=0.04  3=0.06  4=0.08  5=0.10  6=0.12  7=0.14  8=0.16
#   9=0.18 10=0.20 11=0.22 12=0.24 13=0.26 14=0.28 15=0.30
#
# Launch: sbatch --array=1,3-15%10 run_sweep_combined.sh
# (skip index 2 = A=0.04, already running)

. ~/.bashrc

# Thread-limiting env vars to avoid oversubscription
export JAX_ENABLE_X64=True
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

# Map array task ID to A value (index 0 unused)
A_VALUES=(0.00 0.02 0.04 0.06 0.08 0.10 0.12 0.14 0.16 0.18 0.20 0.22 0.24 0.26 0.28 0.30)
A=${A_VALUES[$SLURM_ARRAY_TASK_ID]}

echo "=== Task ${SLURM_ARRAY_TASK_ID}: A=${A} ==="
echo "Host: $(hostname)"
echo "Start: $(date)"

# Create output directory
mkdir -p "analysis/dei_A${A}"

# Run in small batches to avoid XLA/LLVM memory accumulation.
# Each batch is a fresh Python process; the script skips already-completed seeds.
TOTAL_SEEDS=500
BATCH_SIZE=5

for (( OFFSET=0; OFFSET<TOTAL_SEEDS; OFFSET+=BATCH_SIZE )); do
    echo "--- Batch: seeds ${OFFSET}-$((OFFSET+BATCH_SIZE-1)) [$(date)] ---"
    pixi run python -u scripts/run_dei_single_neighbor.py \
        --only-combined \
        --n-starts=${BATCH_SIZE} \
        --max-iter=2000 \
        --A="${A}" \
        --seed-offset=${OFFSET} \
        -o "analysis/dei_A${A}"
    RC=$?
    if [ $RC -ne 0 ]; then
        echo "WARNING: Batch at offset ${OFFSET} exited with code ${RC}, continuing..."
    fi
done

echo "End: $(date)"

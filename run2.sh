#!/bin/bash
#SBATCH --partition=windfatq
#SBATCH --job-name="IFI2"
#SBATCH --output=run2.out
#SBATCH --time=2-00:00:00
#SBATCH --ntasks-per-core 1
#SBATCH --ntasks-per-node 10
#SBATCH --nodes=1
#SBATCH --exclusive

. ~/.bashrc
#conEnv
#eval "$(pixi shell-hook)"

export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

# Run in small batches to avoid XLA/LLVM memory accumulation.
TOTAL_SEEDS=500
BATCH_SIZE=5

for (( OFFSET=0; OFFSET<TOTAL_SEEDS; OFFSET+=BATCH_SIZE )); do
    echo "--- Batch: seeds ${OFFSET}-$((OFFSET+BATCH_SIZE-1)) [$(date)] ---"
    pixi run python -u scripts/run_dei_single_neighbor.py \
        --only-combined \
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

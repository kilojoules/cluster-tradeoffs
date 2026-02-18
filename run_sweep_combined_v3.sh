#!/bin/bash
#SBATCH --partition=windq,workq
#SBATCH --job-name="sweep_v3"
#SBATCH --output=analysis/sweep_v3_%a.out
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --exclusive

# V3: Copies pixi env to node-local /tmp to eliminate BeeGFS I/O errors.
# Otherwise identical to run_sweep_combined_parallel_2.sh.
#
# 15 A values × 50 seed batches (40 seeds each) = 750 array tasks.
# Uses different file-tag (s{OFFSET}_v3) to avoid conflicts with running v2 jobs.
# Merge with: pixi run python bin/merge_combined.py
#
# Launch: sbatch --array=0-749%100 run_sweep_combined_v3.sh

. ~/.bashrc

export JAX_ENABLE_X64=True
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

WORKDIR=/work/users/juqu/cluster-tradeoffs
cd "${WORKDIR}"

# --- Copy pixi env to node-local tmpfs ---
LOCAL_ENV=/tmp/pixi_env_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
echo "Copying pixi env to local /tmp ($(hostname))..."
cp -a "${WORKDIR}/.pixi/envs/default" "${LOCAL_ENV}"
echo "Copy done ($(date))"

PYTHON="${LOCAL_ENV}/bin/python"
export PYTHONHOME="${LOCAL_ENV}"

# Verify local python works
${PYTHON} -c "import jax; import h5py; print('Local Python OK')" || {
    echo "ERROR: Local Python failed, aborting"
    rm -rf "${LOCAL_ENV}"
    exit 1
}

# Cleanup on exit (normal or error)
trap "rm -rf ${LOCAL_ENV}" EXIT

# 15 A values: 0.02, 0.04, ..., 0.30
A_VALUES=(0.02 0.04 0.06 0.08 0.10 0.12 0.14 0.16 0.18 0.20 0.22 0.24 0.26 0.28 0.30)

SEEDS_PER_TASK=40
BATCH_SIZE=5  # fresh Python process every 5 seeds (XLA memory leak workaround)
BATCHES_PER_A=50

A_IDX=$(( SLURM_ARRAY_TASK_ID / BATCHES_PER_A ))
BATCH_IDX=$(( SLURM_ARRAY_TASK_ID % BATCHES_PER_A ))
OFFSET=$(( BATCH_IDX * SEEDS_PER_TASK ))
A=${A_VALUES[$A_IDX]}

echo "=== Task ${SLURM_ARRAY_TASK_ID}: A=${A}, seeds ${OFFSET}-$((OFFSET+SEEDS_PER_TASK-1)) ==="
echo "Host: $(hostname)"
echo "Start: $(date)"

mkdir -p "analysis/dei_A${A}"

# Run 40 seeds in 8 sub-batches of 5 (fresh Python each to avoid XLA leak)
for (( S=OFFSET; S<OFFSET+SEEDS_PER_TASK; S+=BATCH_SIZE )); do
    echo "--- Sub-batch: seeds ${S}-$((S+BATCH_SIZE-1)) [$(date)] ---"
    ${PYTHON} -u scripts/run_dei_single_neighbor.py \
        --only-combined \
        --n-starts=${BATCH_SIZE} \
        --max-iter=2000 \
        --A="${A}" \
        --seed-offset=${S} \
        --file-tag="s${OFFSET}_v3" \
        -o "analysis/dei_A${A}"
    RC=$?
    if [ $RC -ne 0 ]; then
        echo "WARNING: Sub-batch at seed ${S} exited with code ${RC}, continuing..."
    fi
done

echo "End: $(date)"

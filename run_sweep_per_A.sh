#!/bin/bash
#SBATCH --partition=windq,workq
#SBATCH --job-name="sw_AVAL"
#SBATCH --output=analysis/sweep_v4_A%x_%a.out
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --exclude=sn412,sn445

# Per-A-value sweep script. Each job handles one A value with 50 array tasks.
# The A value is passed via --export=A=<value> or --job-name suffix.
#
# Each array task: 40 seeds in 8 sub-batches of 5 (fresh Python each).
# File-tag: s${OFFSET}_v4 to avoid conflicts with running v3 tasks.
#
# Launch all A values:
#   for A in 0.02 0.04 0.06 0.08 0.10 0.12 0.14 0.16 0.18 0.20 0.22 0.24 0.26 0.28 0.30; do
#     sbatch --array=0-49%7 --export=ALL,A_VALUE=$A --job-name="sw_${A}" run_sweep_per_A.sh
#   done

. ~/.bashrc

export JAX_ENABLE_X64=True
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

WORKDIR=/work/users/juqu/cluster-tradeoffs
cd "${WORKDIR}"

# A_VALUE is passed via --export
A=${A_VALUE:?ERROR: A_VALUE not set. Use --export=ALL,A_VALUE=0.02}

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

# Cleanup on exit
trap "rm -rf ${LOCAL_ENV}" EXIT

SEEDS_PER_TASK=40
BATCH_SIZE=5
OFFSET=$(( SLURM_ARRAY_TASK_ID * SEEDS_PER_TASK ))

echo "=== A=${A}, task ${SLURM_ARRAY_TASK_ID}, seeds ${OFFSET}-$((OFFSET+SEEDS_PER_TASK-1)) ==="
echo "Host: $(hostname)"
echo "Start: $(date)"

mkdir -p "analysis/dei_A${A}"

for (( S=OFFSET; S<OFFSET+SEEDS_PER_TASK; S+=BATCH_SIZE )); do
    echo "--- Sub-batch: seeds ${S}-$((S+BATCH_SIZE-1)) [$(date)] ---"
    ${PYTHON} -u scripts/run_dei_single_neighbor.py \
        --only-combined \
        --n-starts=${BATCH_SIZE} \
        --max-iter=2000 \
        --A="${A}" \
        --seed-offset=${S} \
        --file-tag="s${OFFSET}_v4" \
        -o "analysis/dei_A${A}"
    RC=$?
    if [ $RC -ne 0 ]; then
        echo "WARNING: Sub-batch at seed ${S} exited with code ${RC}, continuing..."
    fi
done

echo "End: $(date)"

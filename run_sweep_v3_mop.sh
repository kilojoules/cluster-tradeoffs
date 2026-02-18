#!/bin/bash
#SBATCH --partition=windq,workq
#SBATCH --job-name="v3_mop"
#SBATCH --output=analysis/sweep_v3_mop_%a.out
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --exclude=sn412,sn445

# Mop-up for v3 sweep tasks that failed with Bus errors on bad nodes.
# 8 tasks total — re-runs exact same A/seed combos with _v3mop file-tag.
#
# Launch: sbatch --array=0-7 run_sweep_v3_mop.sh

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

# Failed tasks (original v3 task_id -> A, seed offset):
#   Task   6: A=0.02, offset=240
#   Task  17: A=0.02, offset=680
#   Task 101: A=0.06, offset=40
#   Task 106: A=0.06, offset=240
#   Task 152: A=0.08, offset=80
#   Task 154: A=0.08, offset=160
#   Task 155: A=0.08, offset=200
#   Task 156: A=0.08, offset=240

A_LIST=(0.02 0.02 0.06 0.06 0.08 0.08 0.08 0.08)
OFFSET_LIST=(240 680 40 240 80 160 200 240)

A=${A_LIST[$SLURM_ARRAY_TASK_ID]}
OFFSET=${OFFSET_LIST[$SLURM_ARRAY_TASK_ID]}

SEEDS_PER_TASK=40
BATCH_SIZE=5

echo "=== Mop-up task ${SLURM_ARRAY_TASK_ID}: A=${A}, seeds ${OFFSET}-$((OFFSET+SEEDS_PER_TASK-1)) ==="
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
        --file-tag="s${OFFSET}_v3mop" \
        -o "analysis/dei_A${A}"
    RC=$?
    if [ $RC -ne 0 ]; then
        echo "WARNING: Sub-batch at seed ${S} exited with code ${RC}, continuing..."
    fi
done

echo "End: $(date)"

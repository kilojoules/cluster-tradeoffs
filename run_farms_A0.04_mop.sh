#!/bin/bash
#SBATCH --partition=windq,workq
#SBATCH --job-name="farms04m"
#SBATCH --output=analysis/dei_A0.04/logs/farm_%a_mop.out
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --exclusive

# Mop-up for individual farms (A=0.04) that finished short due to I/O errors.
# Copies pixi env to node-local /tmp. Skips already-completed seeds.
#
# Launch for specific farms: sbatch --array=1,3 run_farms_A0.04_mop.sh
# Launch for all farms:      sbatch --array=1-9 run_farms_A0.04_mop.sh

. ~/.bashrc

export JAX_ENABLE_X64=True
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

WORKDIR=/work/users/juqu/cluster-tradeoffs
cd "${WORKDIR}"

FARM=$SLURM_ARRAY_TASK_ID

# --- Copy pixi env to node-local tmpfs ---
LOCAL_ENV=/tmp/pixi_env_${SLURM_JOB_ID}_${FARM}
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

echo "=== Farm ${FARM} mop-up, A=0.04 ==="
echo "Host: $(hostname)"
echo "Start: $(date)"

mkdir -p analysis/dei_A0.04/logs

TOTAL_SEEDS=2000
BATCH_SIZE=5

for (( OFFSET=0; OFFSET<TOTAL_SEEDS; OFFSET+=BATCH_SIZE )); do
    echo "--- Batch: seeds ${OFFSET}-$((OFFSET+BATCH_SIZE-1)) [$(date)] ---"
    ${PYTHON} -u scripts/run_dei_single_neighbor.py \
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

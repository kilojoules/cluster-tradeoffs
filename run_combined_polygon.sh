#!/bin/bash
#SBATCH --partition=windq,workq
#SBATCH --job-name="comb_poly"
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --exclusive

# Polygon-constrained combined case (all 9 neighbors).
# Uses tarball for fast env staging.

. ~/.bashrc

export JAX_ENABLE_X64=True
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

WORKDIR=/work/users/juqu/cluster-tradeoffs
cd "${WORKDIR}"

# --- Stage pixi env from tarball (single file copy) ---
LOCAL_ENV=/tmp/pixi_env
if [ ! -d "${LOCAL_ENV}" ]; then
    echo "Staging pixi env from tarball ($(hostname))..."
    cp "${WORKDIR}/pixi_env.tar" /tmp/pixi_env.tar
    tar xf /tmp/pixi_env.tar -C /tmp/
    mv /tmp/default "${LOCAL_ENV}"
    rm -f /tmp/pixi_env.tar
    echo "Staging done ($(date))"
else
    echo "Pixi env already staged on $(hostname)"
fi

PYTHON="${LOCAL_ENV}/bin/python"
export PYTHONHOME="${LOCAL_ENV}"

${PYTHON} -c "import jax; import h5py; print('Local Python OK')" || {
    echo "ERROR: Local Python failed, aborting"
    exit 1
}

A="${SWEEP_A}"
SEEDS_PER_TASK=5
OFFSET=$(( SLURM_ARRAY_TASK_ID * SEEDS_PER_TASK ))

echo "=== Combined, A=${A}, seeds ${OFFSET}-$((OFFSET+SEEDS_PER_TASK-1)) ==="
echo "Host: $(hostname), Start: $(date)"

mkdir -p "analysis/dei_A${A}"

${PYTHON} -u scripts/run_dei_single_neighbor.py \
    --only-combined \
    --n-starts=${SEEDS_PER_TASK} \
    --max-iter=2000 \
    --A="${A}" \
    --seed-offset=${OFFSET} \
    -o "analysis/dei_A${A}"

echo "Exit code: $?, End: $(date)"

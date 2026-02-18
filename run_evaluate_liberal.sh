#!/bin/bash
#SBATCH --partition=windq
#SBATCH --job-name="lib_eval"
#SBATCH --output=analysis/lib_eval_%a.out
#SBATCH --time=0-04:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --exclusive

# Evaluate standard liberal layouts (from farm 1) against all neighbor cases.
# Array indices 0-8 = farm1..farm9, index 9 = combined.
#
# Uses batch wrapper (20 layouts per fresh Python process) to avoid XLA memory leak.
#
# Usage (single A value):
#   sbatch --array=0-9 run_evaluate_liberal.sh 0.04
#
# Usage (all 15 A values — 150 tasks, 30 concurrent):
#   sbatch --array=0-149%30 run_evaluate_liberal.sh all

. ~/.bashrc

export JAX_ENABLE_X64=True
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

A_VALUES=(0.02 0.04 0.06 0.08 0.10 0.12 0.14 0.16 0.18 0.20 0.22 0.24 0.26 0.28 0.30)
CASES=(farm1 farm2 farm3 farm4 farm5 farm6 farm7 farm8 farm9 combined)

MODE="${1:-0.04}"  # first argument: specific A value or "all"
BATCH_SIZE=20      # layouts per fresh Python process (XLA memory leak workaround)
TOTAL_LAYOUTS=500  # total liberal layouts to evaluate

if [ "$MODE" = "all" ]; then
    # 150 tasks: index = A_idx * 10 + case_idx
    A_IDX=$(( SLURM_ARRAY_TASK_ID / 10 ))
    CASE_IDX=$(( SLURM_ARRAY_TASK_ID % 10 ))
    A=${A_VALUES[$A_IDX]}
else
    # 10 tasks: one per case
    A="$MODE"
    CASE_IDX=$SLURM_ARRAY_TASK_ID
fi

CASE=${CASES[$CASE_IDX]}

echo "=== Task ${SLURM_ARRAY_TASK_ID}: A=${A}, case=${CASE} ==="
echo "Host: $(hostname)"
echo "Start: $(date)"

ANALYSIS_DIR="analysis/dei_A${A}"

if [ ! -d "$ANALYSIS_DIR" ]; then
    echo "ERROR: ${ANALYSIS_DIR} does not exist"
    exit 1
fi

if [ ! -f "${ANALYSIS_DIR}/layouts_farm1.h5" ]; then
    echo "ERROR: ${ANALYSIS_DIR}/layouts_farm1.h5 not found (source for liberal layouts)"
    exit 1
fi

# Run in batches of BATCH_SIZE with fresh Python each time (XLA memory leak workaround)
for (( S=0; S<TOTAL_LAYOUTS; S+=BATCH_SIZE )); do
    echo "--- Batch: layouts ${S}-$((S+BATCH_SIZE-1)) [$(date)] ---"
    pixi run python -u scripts/evaluate_liberal_layouts.py \
        -i "${ANALYSIS_DIR}" \
        --source farm1 \
        --case "${CASE}" \
        --A "${A}" \
        --start ${S} \
        --count ${BATCH_SIZE}
    RC=$?
    if [ $RC -ne 0 ]; then
        echo "WARNING: Batch at start=${S} exited with code ${RC}, continuing..."
    fi
done

echo "End: $(date)"

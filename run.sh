#!/bin/bash
#SBATCH --partition=windfatq
#SBATCH --job-name="IFI"
#SBATCH --output=run.out
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --exclusive

. ~/.bashrc

export JAX_ENABLE_X64=True
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

pixi run python -u scripts/run_farm_queue.py --max-parallel=2 --n-starts=500 --max-iter=2000 --A=0.04

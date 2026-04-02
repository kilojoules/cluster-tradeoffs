#!/bin/bash
#SBATCH --job-name=gs-bast-unif
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=16:00:00
#SBATCH --output=logs/sweep_bastankhah_uniform_%j.out
#SBATCH --error=logs/sweep_bastankhah_uniform_%j.err

export PATH="$HOME/.pixi/bin:$PATH"
module load rocm/6.0.3
export JAX_PLATFORMS=rocm
export XLA_FLAGS="--xla_gpu_enable_triton_softmax_fusion=false"

cd /scratch/project_465002609/julian_clusters
mkdir -p logs

PIXI="pixi run -e rocm --manifest-path pyproject.toml"

$PIXI python scripts/run_dei_greedy_grid.py \
    --n-place 200 \
    --n-inner-starts 5 --inner-max-iter 5000 --screen-top-k 10 \
    --eval-parallel --grid-pad-D 50 \
    --deficit bastankhah \
    --wind-rose uniform --wind-speed 9.0 \
    --output-dir analysis/sweep_pad50D_200t/bastankhah_uniform

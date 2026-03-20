#!/bin/bash
#SBATCH --job-name=greedy-tp
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --output=logs/greedy_tp_%j.out
#SBATCH --error=logs/greedy_tp_%j.err

# ---- Environment ----
export PATH="$HOME/.pixi/bin:$PATH"
module load rocm/6.0.3

export JAX_PLATFORMS=rocm
export XLA_FLAGS="--xla_gpu_enable_triton_softmax_fusion=false"

cd /scratch/project_465002609/julian_clusters
mkdir -p logs

PIXI="pixi run -e rocm --manifest-path pyproject.toml"

# ---- Run: TurbOPark, DEI wind rose, 25D grid pad ----
echo "=== TurbOPark + DEI rose + 25D pad ==="
$PIXI python scripts/run_dei_greedy_grid.py \
    --n-inner-starts 5 \
    --inner-max-iter 5000 \
    --screen-top-k 10 \
    --eval-parallel \
    --deficit turbopark \
    --grid-pad-D 25 \
    --output-dir analysis/dei_greedy_grid_turbopark_pad25D

echo "=== Done ==="

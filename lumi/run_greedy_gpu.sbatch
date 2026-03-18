#!/bin/bash
#SBATCH --job-name=greedy-K5
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --output=logs/greedy_K5_%j.out
#SBATCH --error=logs/greedy_K5_%j.err

# ---- Environment ----
export PATH="$HOME/.pixi/bin:$PATH"
module load rocm/6.0.3  # LUMI ROCm module

# JAX ROCm settings
export JAX_PLATFORMS=rocm
export XLA_FLAGS="--xla_gpu_enable_triton_softmax_fusion=false"

# Move to repo
cd /scratch/project_465002609/julian_clusters
mkdir -p logs

PIXI="pixi run -e rocm --manifest-path pyproject.toml"

# ---- Smoke test ----
echo "=== GPU smoke test ==="
$PIXI python -c "
import jax
jax.config.update('jax_enable_x64', True)
print('JAX version:', jax.__version__)
print('Devices:', jax.devices())
print('Backend:', jax.default_backend())
"
echo ""

# ---- Run ----
echo "=== Starting greedy grid search K=5, 5000 iter ==="
$PIXI python scripts/run_dei_greedy_grid.py \
    --n-inner-starts 5 \
    --inner-max-iter 5000 \
    --screen-top-k 10 \
    --eval-parallel \
    --output-dir analysis/dei_greedy_grid_lumi_K5_5000iter

echo "=== Done ==="

#!/bin/bash
#SBATCH --job-name=flat-mobj
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --output=logs/flat_mobj_%j.out
#SBATCH --error=logs/flat_mobj_%j.err

# ---- Environment ----
export PATH="$HOME/.pixi/bin:$PATH"
module load rocm/6.0.3
export JAX_PLATFORMS=rocm
export XLA_FLAGS="--xla_gpu_enable_triton_softmax_fusion=false"

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

# ---- Run: weight sweep ----
echo "=== Flat multi-objective: weight sweep ==="
$PIXI python scripts/run_dei_flat_multiobjective.py \
    --sweep \
    --flat-max-iter 2000 \
    --flat-lr 20.0 \
    --n-neighbors 30 \
    --penalty-weight 100.0 \
    --output-dir analysis/dei_flat_sweep

echo "=== Done ==="

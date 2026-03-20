#!/bin/bash
#SBATCH --job-name=greedy-sweep
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --output=logs/greedy_sweep_%j.out
#SBATCH --error=logs/greedy_sweep_%j.err

# ---- Environment ----
export PATH="$HOME/.pixi/bin:$PATH"
module load rocm/6.0.3

export JAX_PLATFORMS=rocm
export XLA_FLAGS="--xla_gpu_enable_triton_softmax_fusion=false"

cd /scratch/project_465002609/julian_clusters
mkdir -p logs

PIXI="pixi run -e rocm --manifest-path pyproject.toml"
COMMON="--n-inner-starts 5 --inner-max-iter 5000 --screen-top-k 10 --eval-parallel --grid-pad-D 50"

echo "=== Unidirectional 270 deg ==="
$PIXI python scripts/run_dei_greedy_grid.py $COMMON \
    --wind-rose unidirectional --wind-dir 270 --wind-speed 9.0 \
    --output-dir analysis/sweep_pad50D/unidirectional_270

echo ""
echo "=== Uniform 24-sector ==="
$PIXI python scripts/run_dei_greedy_grid.py $COMMON \
    --wind-rose uniform --wind-speed 9.0 \
    --output-dir analysis/sweep_pad50D/uniform

echo ""
echo "=== DEI wind rose ==="
$PIXI python scripts/run_dei_greedy_grid.py $COMMON \
    --wind-rose dei \
    --output-dir analysis/sweep_pad50D/dei

echo "=== All done ==="

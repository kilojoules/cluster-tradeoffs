#!/bin/bash
#SBATCH --job-name=buf-pilot
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=04:00:00

# Parameters passed via environment: ED_A, ED_F, BUFFER_D
export PATH="$HOME/.pixi/bin:$PATH"
module load rocm/6.0.3
export JAX_PLATFORMS=rocm
export XLA_FLAGS="--xla_gpu_enable_triton_softmax_fusion=false"

cd /scratch/project_465002609/julian_clusters
mkdir -p logs

PIXI="pixi run -e rocm --manifest-path pyproject.toml"

OUTDIR="analysis/buffer_sweep/a${ED_A}_f${ED_F}_buf${BUFFER_D}D"

$PIXI python scripts/run_dei_greedy_grid.py \
    --n-place 30 \
    --n-inner-starts 5 --inner-max-iter 5000 --screen-top-k 10 \
    --eval-parallel --grid-pad-D 50 \
    --deficit bastankhah \
    --wind-rose elliptical --ed-a ${ED_A} --ed-f ${ED_F} \
    --wind-dir 270 --wind-speed 9.0 --n-bins 24 \
    --buffer-D ${BUFFER_D} \
    --output-dir ${OUTDIR}

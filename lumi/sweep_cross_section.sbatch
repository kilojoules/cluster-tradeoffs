#!/bin/bash
#SBATCH --job-name=xsec
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=12:00:00

# Parameters via environment: ED_A, ED_F, WIND_ROSE
export PATH="$HOME/.pixi/bin:$PATH"
module load rocm/6.0.3
export JAX_PLATFORMS=rocm
export XLA_FLAGS="--xla_gpu_enable_triton_softmax_fusion=false"

cd /scratch/project_465002609/julian_clusters
mkdir -p logs

PIXI="pixi run -e rocm --manifest-path pyproject.toml"

if [ "$WIND_ROSE" = "dei" ]; then
    OUTDIR="analysis/cross_section/dei"
    ROSE_ARGS="--wind-rose dei"
else
    OUTDIR="analysis/cross_section/a${ED_A}_f${ED_F}"
    ROSE_ARGS="--wind-rose elliptical --ed-a ${ED_A} --ed-f ${ED_F} --wind-dir 270"
fi

$PIXI python scripts/run_regret_cross_section.py \
    --n-bearings 24 --distances-D 5,10,15,20,30,40,60 \
    --ref-rows 5 --ref-cols 5 --ref-spacing-D 7 \
    --n-inner-starts 5 --inner-max-iter 5000 \
    --deficit bastankhah \
    ${ROSE_ARGS} --wind-speed 9.0 --n-bins 24 \
    --output-dir ${OUTDIR}

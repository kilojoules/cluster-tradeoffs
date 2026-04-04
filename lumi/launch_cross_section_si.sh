#!/bin/bash
# Launch self-interested neighbor cross-section for a=0.9, f=1.0
cd /scratch/project_465002609/julian_clusters
mkdir -p logs

export PATH="$HOME/.pixi/bin:$PATH"
PIXI="pixi run -e rocm --manifest-path pyproject.toml"

sbatch --job-name="xsec-si" \
       --account=project_465002609 \
       --partition=small-g \
       --gpus-per-node=1 \
       --ntasks=1 \
       --cpus-per-task=7 \
       --mem=60G \
       --time=24:00:00 \
       --output="logs/xsec_si_a0.9_f1.0_%j.out" \
       --error="logs/xsec_si_a0.9_f1.0_%j.err" \
       --wrap="export PATH=\"\$HOME/.pixi/bin:\$PATH\" && module load rocm/6.0.3 && export JAX_PLATFORMS=rocm && export XLA_FLAGS=\"--xla_gpu_enable_triton_softmax_fusion=false\" && cd /scratch/project_465002609/julian_clusters && pixi run -e rocm --manifest-path pyproject.toml python scripts/run_regret_cross_section_si.py --n-bearings 24 --distances-D 5,10,15,20,30,40,60 --ref-rows 5 --ref-cols 5 --ref-spacing-D 7 --n-inner-starts 5 --inner-max-iter 5000 --deficit bastankhah --wind-rose elliptical --ed-a 0.9 --ed-f 1.0 --wind-dir 270 --wind-speed 9.0 --n-bins 24 --output-dir analysis/cross_section_si/a0.9_f1.0"

echo "Self-interested cross-section job submitted."

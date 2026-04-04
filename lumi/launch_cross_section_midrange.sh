#!/bin/bash
# Launch cross-section at mid-range (a=0.7, f=0.5)
cd /scratch/project_465002609/julian_clusters
mkdir -p logs

sbatch --job-name="xsec-mid" \
       --account=project_465002609 \
       --partition=small-g \
       --gpus-per-node=1 \
       --ntasks=1 \
       --cpus-per-task=7 \
       --mem=60G \
       --time=12:00:00 \
       --output="logs/xsec_a0.7_f0.5_%j.out" \
       --error="logs/xsec_a0.7_f0.5_%j.err" \
       --wrap="export PATH=\"\$HOME/.pixi/bin:\$PATH\" && module load rocm/6.0.3 && export JAX_PLATFORMS=rocm && export XLA_FLAGS=\"--xla_gpu_enable_triton_softmax_fusion=false\" && cd /scratch/project_465002609/julian_clusters && pixi run -e rocm --manifest-path pyproject.toml python scripts/run_regret_cross_section.py --n-bearings 24 --distances-D 5,10,15,20,30,40,60 --ref-rows 5 --ref-cols 5 --ref-spacing-D 7 --n-inner-starts 5 --inner-max-iter 5000 --deficit bastankhah --wind-rose elliptical --ed-a 0.7 --ed-f 0.5 --wind-dir 270 --wind-speed 9.0 --n-bins 24 --output-dir analysis/cross_section/a0.7_f0.5"

echo "Mid-range cross-section job submitted."

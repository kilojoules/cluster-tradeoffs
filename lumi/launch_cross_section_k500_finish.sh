#!/bin/bash
# Re-run the 2 K=500 cross-sections that didn't finish:
# - DEI (timed out at 95.2% on 3-day walltime)
# - a=0.5, f=0.0 (failed with pixi env error)
cd /scratch/project_465002609/julian_clusters
mkdir -p logs

# DEI real wind rose
sbatch --job-name="xv5-dei2" \
       --account=project_465002609 --partition=small-g \
       --gpus-per-node=1 --ntasks=1 --cpus-per-task=7 --mem=60G \
       --time=3-00:00:00 \
       --output="logs/xsec_k500v2_dei_%j.out" \
       --error="logs/xsec_k500v2_dei_%j.err" \
       --wrap="export PATH=\"\$HOME/.pixi/bin:\$PATH\" && module load rocm/6.0.3 && export JAX_PLATFORMS=rocm && export XLA_FLAGS=\"--xla_gpu_enable_triton_softmax_fusion=false\" && cd /scratch/project_465002609/julian_clusters && pixi run -e rocm --manifest-path pyproject.toml python scripts/run_regret_cross_section.py --n-bearings 24 --distances-D 5,10,15,20,30,40,60 --ref-rows 5 --ref-cols 5 --ref-spacing-D 7 --n-inner-starts 500 --inner-max-iter 5000 --deficit bastankhah --wind-rose dei --wind-speed 9.0 --n-bins 24 --output-dir analysis/cross_section_k500v/dei"

# a=0.5, f=0.0
sbatch --job-name="xv5-0.5-0.0" \
       --account=project_465002609 --partition=small-g \
       --gpus-per-node=1 --ntasks=1 --cpus-per-task=7 --mem=60G \
       --time=3-00:00:00 \
       --output="logs/xsec_k500v2_a0.5_f0.0_%j.out" \
       --error="logs/xsec_k500v2_a0.5_f0.0_%j.err" \
       --wrap="export PATH=\"\$HOME/.pixi/bin:\$PATH\" && module load rocm/6.0.3 && export JAX_PLATFORMS=rocm && export XLA_FLAGS=\"--xla_gpu_enable_triton_softmax_fusion=false\" && cd /scratch/project_465002609/julian_clusters && pixi run -e rocm --manifest-path pyproject.toml python scripts/run_regret_cross_section.py --n-bearings 24 --distances-D 5,10,15,20,30,40,60 --ref-rows 5 --ref-cols 5 --ref-spacing-D 7 --n-inner-starts 500 --inner-max-iter 5000 --deficit bastankhah --wind-rose elliptical --ed-a 0.5 --ed-f 0.0 --wind-dir 270 --wind-speed 9.0 --n-bins 24 --output-dir analysis/cross_section_k500v/a0.5_f0.0"

echo "2 cross-section K=500 re-run jobs submitted."

#!/bin/bash
# Single wind direction case: wind FROM east (90 degrees)
# This should show zero regret for neighbors to the west (downwind)
# and maximum regret for neighbors to the east (upwind)
cd /scratch/project_465002609/julian_clusters
mkdir -p logs

count=0
for dist in 2 5 10 15 20 30 40; do
    sbatch --job-name="xf-uni90-d${dist}" \
           --account=project_465002609 --partition=small-g \
           --gpus-per-node=1 --ntasks=1 --cpus-per-task=7 --mem=60G \
           --time=1-00:00:00 \
           --output="logs/xsec_fixed_unidir90_d${dist}_%j.out" \
           --error="logs/xsec_fixed_unidir90_d${dist}_%j.err" \
           --wrap="export PATH=\"\$HOME/.pixi/bin:\$PATH\" && module load rocm/6.0.3 && export JAX_PLATFORMS=rocm && export XLA_FLAGS=\"--xla_gpu_enable_triton_softmax_fusion=false\" && cd /scratch/project_465002609/julian_clusters && pixi run -e rocm --manifest-path pyproject.toml python scripts/run_regret_cross_section.py --n-bearings 24 --distances-D ${dist} --n-inner-starts 300 --inner-max-iter 2000 --k-liberal 300 --deficit bastankhah --chunk-size 50 --wind-rose unidirectional --wind-dir 90 --wind-speed 9.0 --output-dir analysis/cross_section_fixed/unidir90_d${dist}"
    count=$((count + 1))
done

echo "${count} unidirectional (east) cross-section jobs submitted."

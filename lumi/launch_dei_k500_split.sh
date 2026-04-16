#!/bin/bash
# Split DEI K=500 cross-section across 6 GPU jobs, each handling ~28 positions
# Total: 168 positions / 6 = 28 positions per job
# Each job runs positions for 1 of the 7 distances (5,10,15,20,30,40,60D)
# But since 24 bearings × 1 distance = 24 positions → fits easily in walltime
cd /scratch/project_465002609/julian_clusters
mkdir -p logs

count=0
for dist in 5 10 15 20 30 40 60; do
    sbatch --job-name="dei-d${dist}" \
           --account=project_465002609 --partition=small-g \
           --gpus-per-node=1 --ntasks=1 --cpus-per-task=7 --mem=60G \
           --time=1-12:00:00 \
           --output="logs/dei_k500_d${dist}_%j.out" \
           --error="logs/dei_k500_d${dist}_%j.err" \
           --wrap="export PATH=\"\$HOME/.pixi/bin:\$PATH\" && module load rocm/6.0.3 && export JAX_PLATFORMS=rocm && export XLA_FLAGS=\"--xla_gpu_enable_triton_softmax_fusion=false\" && cd /scratch/project_465002609/julian_clusters && pixi run -e rocm --manifest-path pyproject.toml python scripts/run_regret_cross_section.py --n-bearings 24 --distances-D ${dist} --ref-rows 5 --ref-cols 5 --ref-spacing-D 7 --n-inner-starts 500 --inner-max-iter 5000 --deficit bastankhah --wind-rose dei --wind-speed 9.0 --n-bins 24 --output-dir analysis/cross_section_k500v/dei_d${dist}"
    count=$((count + 1))
done

echo "${count} DEI split jobs submitted (one per distance)."
echo "After completion, merge results with: python scripts/merge_dei_cross_section.py"

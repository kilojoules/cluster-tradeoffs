#!/bin/bash
# Production rerun: cross-sections with K=500
# 6 cases: DEI + 4 edrose corners + mid-range
cd /scratch/project_465002609/julian_clusters
mkdir -p logs

count=0

# DEI real wind rose
sbatch --job-name="xk5-dei" \
       --account=project_465002609 --partition=small-g \
       --gpus-per-node=1 --ntasks=1 --cpus-per-task=7 --mem=60G \
       --time=08:00:00 \
       --output="logs/xsec_k500_dei_%j.out" \
       --error="logs/xsec_k500_dei_%j.err" \
       --wrap="export PATH=\"\$HOME/.pixi/bin:\$PATH\" && module load rocm/6.0.3 && export JAX_PLATFORMS=rocm && export XLA_FLAGS=\"--xla_gpu_enable_triton_softmax_fusion=false\" && cd /scratch/project_465002609/julian_clusters && pixi run -e rocm --manifest-path pyproject.toml python scripts/run_regret_cross_section.py --n-bearings 24 --distances-D 5,10,15,20,30,40,60 --ref-rows 5 --ref-cols 5 --ref-spacing-D 7 --n-inner-starts 500 --inner-max-iter 5000 --deficit bastankhah --wind-rose dei --wind-speed 9.0 --n-bins 24 --output-dir analysis/cross_section_k500/dei"
count=$((count + 1))

# 5 edrose cases
for params in "0.5 0.0" "0.5 1.0" "0.7 0.5" "0.9 0.0" "0.9 1.0"; do
    a=$(echo $params | cut -d' ' -f1)
    f=$(echo $params | cut -d' ' -f2)
    sbatch --job-name="xk5-${a}-${f}" \
           --account=project_465002609 --partition=small-g \
           --gpus-per-node=1 --ntasks=1 --cpus-per-task=7 --mem=60G \
           --time=08:00:00 \
           --output="logs/xsec_k500_a${a}_f${f}_%j.out" \
           --error="logs/xsec_k500_a${a}_f${f}_%j.err" \
           --wrap="export PATH=\"\$HOME/.pixi/bin:\$PATH\" && module load rocm/6.0.3 && export JAX_PLATFORMS=rocm && export XLA_FLAGS=\"--xla_gpu_enable_triton_softmax_fusion=false\" && cd /scratch/project_465002609/julian_clusters && pixi run -e rocm --manifest-path pyproject.toml python scripts/run_regret_cross_section.py --n-bearings 24 --distances-D 5,10,15,20,30,40,60 --ref-rows 5 --ref-cols 5 --ref-spacing-D 7 --n-inner-starts 500 --inner-max-iter 5000 --deficit bastankhah --wind-rose elliptical --ed-a ${a} --ed-f ${f} --wind-dir 270 --wind-speed 9.0 --n-bins 24 --output-dir analysis/cross_section_k500/a${a}_f${f}"
    count=$((count + 1))
done

echo "${count} cross-section K=500 jobs submitted."

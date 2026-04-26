#!/bin/bash
# Extend a-sweep above 0.9 at f=1.0: concentrates along prevailing direction
# a -> infinity approaches single-sector behavior
cd /scratch/project_465002609/julian_clusters
mkdir -p logs

count=0
for a in 0.95 0.98 0.99 1.0 1.5 2.0 3.0 5.0; do
    for dist in 2 10; do
        sbatch --job-name="abl-a${a}-d${dist}" \
               --account=project_465002609 --partition=small-g \
               --gpus-per-node=1 --ntasks=1 --cpus-per-task=7 --mem=60G \
               --time=1-00:00:00 \
               --output="logs/ablation_a${a}_d${dist}_%j.out" \
               --error="logs/ablation_a${a}_d${dist}_%j.err" \
               --wrap="export PATH=\"\$HOME/.pixi/bin:\$PATH\" && module load rocm/6.0.3 && export JAX_PLATFORMS=rocm && export XLA_FLAGS=\"--xla_gpu_enable_triton_softmax_fusion=false\" && cd /scratch/project_465002609/julian_clusters && pixi run -e rocm --manifest-path pyproject.toml python scripts/run_regret_cross_section.py --n-bearings 24 --distances-D ${dist} --n-inner-starts 300 --inner-max-iter 2000 --k-liberal 300 --deficit bastankhah --chunk-size 50 --wind-rose elliptical --ed-a ${a} --ed-f 1.0 --wind-dir 270 --wind-speed 9.0 --n-bins 24 --output-dir analysis/ablation_a_sweep/a${a}_f1.0_d${dist}"
        count=$((count + 1))
    done
done

echo "${count} high-a ablation jobs submitted (8 a-values x 2 distances = 16)."

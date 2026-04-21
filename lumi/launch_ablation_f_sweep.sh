#!/bin/bash
# Ablation: fix a=0.9, sweep f from 0.0 to 1.0 in steps of 0.1
# At buffer distances 2D and 10D
# Shows how directional regret pattern transitions from bidir to unidir
cd /scratch/project_465002609/julian_clusters
mkdir -p logs

count=0
for f in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
    for dist in 2 10; do
        sbatch --job-name="abl-f${f}-d${dist}" \
               --account=project_465002609 --partition=small-g \
               --gpus-per-node=1 --ntasks=1 --cpus-per-task=7 --mem=60G \
               --time=1-00:00:00 \
               --output="logs/ablation_f${f}_d${dist}_%j.out" \
               --error="logs/ablation_f${f}_d${dist}_%j.err" \
               --wrap="export PATH=\"\$HOME/.pixi/bin:\$PATH\" && module load rocm/6.0.3 && export JAX_PLATFORMS=rocm && export XLA_FLAGS=\"--xla_gpu_enable_triton_softmax_fusion=false\" && cd /scratch/project_465002609/julian_clusters && pixi run -e rocm --manifest-path pyproject.toml python scripts/run_regret_cross_section.py --n-bearings 24 --distances-D ${dist} --n-inner-starts 300 --inner-max-iter 2000 --k-liberal 300 --deficit bastankhah --chunk-size 50 --wind-rose elliptical --ed-a 0.9 --ed-f ${f} --wind-dir 270 --wind-speed 9.0 --n-bins 24 --output-dir analysis/ablation_f_sweep/a0.9_f${f}_d${dist}"
        count=$((count + 1))
    done
done

# Also add the pure unidirectional case (single direction from 270)
for dist in 2 10; do
    sbatch --job-name="abl-uni-d${dist}" \
           --account=project_465002609 --partition=small-g \
           --gpus-per-node=1 --ntasks=1 --cpus-per-task=7 --mem=60G \
           --time=1-00:00:00 \
           --output="logs/ablation_unidir270_d${dist}_%j.out" \
           --error="logs/ablation_unidir270_d${dist}_%j.err" \
           --wrap="export PATH=\"\$HOME/.pixi/bin:\$PATH\" && module load rocm/6.0.3 && export JAX_PLATFORMS=rocm && export XLA_FLAGS=\"--xla_gpu_enable_triton_softmax_fusion=false\" && cd /scratch/project_465002609/julian_clusters && pixi run -e rocm --manifest-path pyproject.toml python scripts/run_regret_cross_section.py --n-bearings 24 --distances-D ${dist} --n-inner-starts 300 --inner-max-iter 2000 --k-liberal 300 --deficit bastankhah --chunk-size 50 --wind-rose unidirectional --wind-dir 270 --wind-speed 9.0 --output-dir analysis/ablation_f_sweep/unidir270_d${dist}"
    count=$((count + 1))
done

echo "${count} ablation jobs submitted (11 f-values × 2 distances + 2 pure unidir = 24)."

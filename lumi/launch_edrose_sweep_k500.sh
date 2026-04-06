#!/bin/bash
# Production rerun: edrose (a,f) sweep with K=500
# 7 values of a × 5 values of f = 35 jobs
cd /scratch/project_465002609/julian_clusters
mkdir -p logs

count=0
for a in 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
    for f in 0.0 0.25 0.5 0.75 1.0; do
        export ED_A=$a ED_F=$f
        sbatch --export=ED_A,ED_F \
               --job-name="k500-${a}-${f}" \
               --account=project_465002609 \
               --partition=small-g \
               --gpus-per-node=1 --ntasks=1 --cpus-per-task=7 --mem=60G \
               --time=12:00:00 \
               --output="logs/edrose_k500_a${a}_f${f}_%j.out" \
               --error="logs/edrose_k500_a${a}_f${f}_%j.err" \
               --wrap="export PATH=\"\$HOME/.pixi/bin:\$PATH\" && module load rocm/6.0.3 && export JAX_PLATFORMS=rocm && export XLA_FLAGS=\"--xla_gpu_enable_triton_softmax_fusion=false\" && cd /scratch/project_465002609/julian_clusters && pixi run -e rocm --manifest-path pyproject.toml python scripts/run_dei_greedy_grid.py --n-place 30 --n-inner-starts 500 --inner-max-iter 5000 --screen-top-k 10 --eval-parallel --grid-pad-D 50 --deficit bastankhah --wind-rose elliptical --ed-a \${ED_A} --ed-f \${ED_F} --wind-dir 270 --wind-speed 9.0 --n-bins 24 --output-dir analysis/edrose_sweep_k500/a\${ED_A}_f\${ED_F}"
        count=$((count + 1))
    done
done

echo "${count} edrose K=500 sweep jobs submitted."

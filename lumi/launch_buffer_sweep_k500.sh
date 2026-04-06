#!/bin/bash
# Production rerun: buffer sweep with K=500
# 3 wind roses × 4 buffer distances = 12 jobs
cd /scratch/project_465002609/julian_clusters
mkdir -p logs

count=0
for buffer in 2 10 20 40; do
    for params in "0.9 1.0" "0.5 0.0" "0.7 0.5"; do
        a=$(echo $params | cut -d' ' -f1)
        f=$(echo $params | cut -d' ' -f2)
        sbatch --job-name="bk5-${a}-${f}-${buffer}" \
               --account=project_465002609 --partition=small-g \
               --gpus-per-node=1 --ntasks=1 --cpus-per-task=7 --mem=60G \
               --time=12:00:00 \
               --output="logs/buffer_k500_a${a}_f${f}_buf${buffer}D_%j.out" \
               --error="logs/buffer_k500_a${a}_f${f}_buf${buffer}D_%j.err" \
               --wrap="export PATH=\"\$HOME/.pixi/bin:\$PATH\" && module load rocm/6.0.3 && export JAX_PLATFORMS=rocm && export XLA_FLAGS=\"--xla_gpu_enable_triton_softmax_fusion=false\" && cd /scratch/project_465002609/julian_clusters && pixi run -e rocm --manifest-path pyproject.toml python scripts/run_dei_greedy_grid.py --n-place 30 --n-inner-starts 500 --inner-max-iter 5000 --screen-top-k 10 --eval-parallel --grid-pad-D 50 --deficit bastankhah --wind-rose elliptical --ed-a ${a} --ed-f ${f} --wind-dir 270 --wind-speed 9.0 --n-bins 24 --buffer-D ${buffer} --output-dir analysis/buffer_sweep_k500/a${a}_f${f}_buf${buffer}D"
        count=$((count + 1))
    done
done

echo "${count} buffer K=500 sweep jobs submitted."

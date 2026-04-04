#!/bin/bash
# K-sensitivity: 3 wind roses × K={3, 10} = 6 runs (K=5 already exists)
cd /scratch/project_465002609/julian_clusters
mkdir -p logs

count=0
for K in 3 10; do
    for params in "0.9 1.0" "0.5 0.0" "0.7 0.5"; do
        a=$(echo $params | cut -d' ' -f1)
        f=$(echo $params | cut -d' ' -f2)

        sbatch --job-name="ksens-${a}-${f}-K${K}" \
               --account=project_465002609 \
               --partition=small-g \
               --gpus-per-node=1 \
               --ntasks=1 \
               --cpus-per-task=7 \
               --mem=60G \
               --time=04:00:00 \
               --output="logs/ksens_a${a}_f${f}_K${K}_%j.out" \
               --error="logs/ksens_a${a}_f${f}_K${K}_%j.err" \
               --wrap="export PATH=\"\$HOME/.pixi/bin:\$PATH\" && module load rocm/6.0.3 && export JAX_PLATFORMS=rocm && export XLA_FLAGS=\"--xla_gpu_enable_triton_softmax_fusion=false\" && cd /scratch/project_465002609/julian_clusters && pixi run -e rocm --manifest-path pyproject.toml python scripts/run_dei_greedy_grid.py --n-place 30 --n-inner-starts ${K} --inner-max-iter 5000 --screen-top-k 10 --eval-parallel --grid-pad-D 50 --deficit bastankhah --wind-rose elliptical --ed-a ${a} --ed-f ${f} --wind-dir 270 --wind-speed 9.0 --n-bins 24 --output-dir analysis/k_sensitivity/a${a}_f${f}_K${K}"
        count=$((count + 1))
    done
done

echo "${count} K-sensitivity jobs submitted."

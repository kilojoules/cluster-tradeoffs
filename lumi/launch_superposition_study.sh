#!/bin/bash
# Superposition study: compare LinearSum vs SquaredSum
# Run 4 cases: {Bastankhah, TurboPark} × {LinearSum, SquaredSum} for unidirectional wind
# Also run Bastankhah LinearSum cross-section for ambush effect comparison
cd /scratch/project_465002609/julian_clusters
mkdir -p logs

count=0

# Greedy grid search: 30 neighbors, unidirectional, comparing superposition
for sup in linearsum squaredsum; do
    for deficit in bastankhah turbopark; do
        # Skip SquaredSum cases we already have
        if [ "$sup" = "squaredsum" ]; then
            continue
        fi
        sbatch --job-name="sup-${deficit:0:4}-${sup:0:3}" \
               --account=project_465002609 \
               --partition=small-g \
               --gpus-per-node=1 --ntasks=1 --cpus-per-task=7 --mem=60G \
               --time=04:00:00 \
               --output="logs/sup_${deficit}_${sup}_unidir_%j.out" \
               --error="logs/sup_${deficit}_${sup}_unidir_%j.err" \
               --wrap="export PATH=\"\$HOME/.pixi/bin:\$PATH\" && module load rocm/6.0.3 && export JAX_PLATFORMS=rocm && export XLA_FLAGS=\"--xla_gpu_enable_triton_softmax_fusion=false\" && cd /scratch/project_465002609/julian_clusters && pixi run -e rocm --manifest-path pyproject.toml python scripts/run_dei_greedy_grid.py --n-place 30 --n-inner-starts 5 --inner-max-iter 5000 --screen-top-k 10 --eval-parallel --grid-pad-D 50 --deficit ${deficit} --superposition ${sup} --wind-rose unidirectional --wind-dir 270 --wind-speed 9.0 --output-dir analysis/superposition/${deficit}_${sup}_unidir"
        count=$((count + 1))
    done
done

# Also run LinearSum for uniform wind to complete the 2x2 table
for deficit in bastankhah turbopark; do
    sbatch --job-name="sup-${deficit:0:4}-lin-uni" \
           --account=project_465002609 \
           --partition=small-g \
           --gpus-per-node=1 --ntasks=1 --cpus-per-task=7 --mem=60G \
           --time=04:00:00 \
           --output="logs/sup_${deficit}_linearsum_uniform_%j.out" \
           --error="logs/sup_${deficit}_linearsum_uniform_%j.err" \
           --wrap="export PATH=\"\$HOME/.pixi/bin:\$PATH\" && module load rocm/6.0.3 && export JAX_PLATFORMS=rocm && export XLA_FLAGS=\"--xla_gpu_enable_triton_softmax_fusion=false\" && cd /scratch/project_465002609/julian_clusters && pixi run -e rocm --manifest-path pyproject.toml python scripts/run_dei_greedy_grid.py --n-place 30 --n-inner-starts 5 --inner-max-iter 5000 --screen-top-k 10 --eval-parallel --grid-pad-D 50 --deficit ${deficit} --superposition linearsum --wind-rose uniform --wind-speed 9.0 --output-dir analysis/superposition/${deficit}_linearsum_uniform"
    count=$((count + 1))
done

# Cross-section with Bastankhah LinearSum for ambush effect comparison
sbatch --job-name="xsec-lin" \
       --account=project_465002609 \
       --partition=small-g \
       --gpus-per-node=1 --ntasks=1 --cpus-per-task=7 --mem=60G \
       --time=12:00:00 \
       --output="logs/xsec_bastankhah_linearsum_a0.9_f1.0_%j.out" \
       --error="logs/xsec_bastankhah_linearsum_a0.9_f1.0_%j.err" \
       --wrap="export PATH=\"\$HOME/.pixi/bin:\$PATH\" && module load rocm/6.0.3 && export JAX_PLATFORMS=rocm && export XLA_FLAGS=\"--xla_gpu_enable_triton_softmax_fusion=false\" && cd /scratch/project_465002609/julian_clusters && pixi run -e rocm --manifest-path pyproject.toml python scripts/run_regret_cross_section.py --n-bearings 24 --distances-D 5,10,15,20,30,40,60 --ref-rows 5 --ref-cols 5 --ref-spacing-D 7 --n-inner-starts 5 --inner-max-iter 5000 --deficit bastankhah --superposition linearsum --wind-rose elliptical --ed-a 0.9 --ed-f 1.0 --wind-dir 270 --wind-speed 9.0 --n-bins 24 --output-dir analysis/cross_section_linearsum/a0.9_f1.0"
count=$((count + 1))

echo "${count} superposition study jobs submitted."

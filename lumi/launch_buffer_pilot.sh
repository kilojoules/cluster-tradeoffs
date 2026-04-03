#!/bin/bash
# Phase 1 pilot: 3 wind roses × 4 buffer levels = 12 runs
# Wind roses: max-regret (0.9, 1.0), min-regret (0.5, 0.0), mid-range (0.7, 0.5)
# Buffers: 2D, 10D, 20D, 40D
cd /scratch/project_465002609/julian_clusters
mkdir -p logs

count=0
for buffer in 2 10 20 40; do
    for params in "0.9 1.0" "0.5 0.0" "0.7 0.5"; do
        a=$(echo $params | cut -d' ' -f1)
        f=$(echo $params | cut -d' ' -f2)
        export ED_A=$a ED_F=$f BUFFER_D=$buffer
        sbatch --export=ED_A,ED_F,BUFFER_D \
               --job-name="buf-${a}-${f}-${buffer}D" \
               --output="logs/buffer_a${a}_f${f}_buf${buffer}D_%j.out" \
               --error="logs/buffer_a${a}_f${f}_buf${buffer}D_%j.err" \
               lumi/sweep_buffer_pilot.sbatch
        count=$((count + 1))
    done
done

echo "${count} buffer pilot jobs submitted. Monitor with: squeue -u \$USER"

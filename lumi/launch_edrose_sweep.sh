#!/bin/bash
# Launch full (a, f) wind rose sweep using edrose elliptical parameterization
# 7 values of a × 5 values of f = 35 jobs
cd /scratch/project_465002609/julian_clusters
mkdir -p logs

count=0
for a in 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
    for f in 0.0 0.25 0.5 0.75 1.0; do
        export ED_A=$a
        export ED_F=$f
        sbatch --export=ED_A,ED_F \
               --job-name="ed-${a}-${f}" \
               --output="logs/edrose_a${a}_f${f}_%j.out" \
               --error="logs/edrose_a${a}_f${f}_%j.err" \
               lumi/sweep_edrose.sbatch
        count=$((count + 1))
    done
done

echo "${count} edrose sweep jobs submitted. Monitor with: squeue -u \$USER"

#!/bin/bash
# Launch regret cross-section for 5 wind roses:
# DEI real rose + 4 edrose corners of the (a,f) space
cd /scratch/project_465002609/julian_clusters
mkdir -p logs

count=0

# DEI real wind rose
export WIND_ROSE=dei ED_A=0 ED_F=0
sbatch --export=WIND_ROSE,ED_A,ED_F \
       --job-name="xsec-dei" \
       --output="logs/xsec_dei_%j.out" \
       --error="logs/xsec_dei_%j.err" \
       lumi/sweep_cross_section.sbatch
count=$((count + 1))

# 4 edrose cases spanning the shape space
for params in "0.5 0.0" "0.5 1.0" "0.9 0.0" "0.9 1.0"; do
    a=$(echo $params | cut -d' ' -f1)
    f=$(echo $params | cut -d' ' -f2)
    export WIND_ROSE=elliptical ED_A=$a ED_F=$f
    sbatch --export=WIND_ROSE,ED_A,ED_F \
           --job-name="xsec-${a}-${f}" \
           --output="logs/xsec_a${a}_f${f}_%j.out" \
           --error="logs/xsec_a${a}_f${f}_%j.err" \
           lumi/sweep_cross_section.sbatch
    count=$((count + 1))
done

echo "${count} cross-section jobs submitted. Monitor with: squeue -u \$USER"

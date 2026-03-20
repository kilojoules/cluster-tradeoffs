#!/bin/bash
# Launch all 4 sweep cases in parallel
cd /scratch/project_465002609/julian_clusters
mkdir -p logs

sbatch lumi/sweep_bastankhah_unidir.sbatch
sbatch lumi/sweep_bastankhah_uniform.sbatch
sbatch lumi/sweep_turbopark_unidir.sbatch
sbatch lumi/sweep_turbopark_uniform.sbatch

echo "All 4 jobs submitted. Monitor with: squeue -u $USER"

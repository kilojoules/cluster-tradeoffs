#!/bin/bash
# Comprehensive convergence study:
# - 3 test configurations (2 wind roses × 2 distances)
# - Liberal convergence: K_lib up to 200
# - Conservative convergence: K up to 500
# - SGD iteration sweep: 100-10000 at K=200
# - Bootstrap stability analysis: 1000 reshuffles
# Expected runtime: ~12-24 hours on GPU
cd /scratch/project_465002609/julian_clusters
mkdir -p logs

sbatch --job-name="conv-study" \
       --account=project_465002609 \
       --partition=small-g \
       --gpus-per-node=1 --ntasks=1 --cpus-per-task=7 --mem=60G \
       --time=3-00:00:00 \
       --output="logs/convergence_study_%j.out" \
       --error="logs/convergence_study_%j.err" \
       --wrap="export PATH=\"\$HOME/.pixi/bin:\$PATH\" && module load rocm/6.0.3 && export JAX_PLATFORMS=rocm && export XLA_FLAGS=\"--xla_gpu_enable_triton_softmax_fusion=false\" && cd /scratch/project_465002609/julian_clusters && pixi run -e rocm --manifest-path pyproject.toml python scripts/run_regret_convergence.py --output-dir analysis/convergence_study"

echo "Convergence study submitted (3-day walltime)."

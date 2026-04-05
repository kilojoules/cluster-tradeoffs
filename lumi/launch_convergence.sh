#!/bin/bash
# Convergence study: sweep K (1-200) and max_iter (100-10000)
# Run at the high-regret configuration (bearing=105, dist=5D, a=0.9, f=1.0)
cd /scratch/project_465002609/julian_clusters
mkdir -p logs

sbatch --job-name="conv-study" \
       --account=project_465002609 \
       --partition=small-g \
       --gpus-per-node=1 --ntasks=1 --cpus-per-task=7 --mem=60G \
       --time=2-00:00:00 \
       --output="logs/convergence_study_%j.out" \
       --error="logs/convergence_study_%j.err" \
       --wrap="export PATH=\"\$HOME/.pixi/bin:\$PATH\" && module load rocm/6.0.3 && export JAX_PLATFORMS=rocm && export XLA_FLAGS=\"--xla_gpu_enable_triton_softmax_fusion=false\" && cd /scratch/project_465002609/julian_clusters && pixi run -e rocm --manifest-path pyproject.toml python scripts/run_regret_convergence.py --bearing 105 --distance-D 5 --ed-a 0.9 --ed-f 1.0 --output-dir analysis/convergence_study"

echo "Convergence study submitted."

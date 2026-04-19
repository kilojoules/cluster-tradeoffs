#!/bin/bash
# Cases that OOM'd on Sophia even on fatq (256GB)
# a=0.7 d10-d40 and all a=0.9 cases
cd /scratch/project_465002609/julian_clusters
mkdir -p logs

count=0

for case_args in \
    "a0.7_f0.5 --wind-rose elliptical --ed-a 0.7 --ed-f 0.5 --wind-dir 270 --wind-speed 9.0 --n-bins 24" \
    "a0.9_f0.0 --wind-rose elliptical --ed-a 0.9 --ed-f 0.0 --wind-dir 270 --wind-speed 9.0 --n-bins 24" \
    "a0.9_f1.0 --wind-rose elliptical --ed-a 0.9 --ed-f 1.0 --wind-dir 270 --wind-speed 9.0 --n-bins 24"
do
    case_name=$(echo $case_args | cut -d' ' -f1)
    rose_args=$(echo $case_args | cut -d' ' -f2-)

    for dist in 2 5 10 15 20 30 40; do
        sbatch --job-name="xf-${case_name:0:6}-d${dist}" \
               --account=project_465002609 --partition=small-g \
               --gpus-per-node=1 --ntasks=1 --cpus-per-task=7 --mem=60G \
               --time=1-12:00:00 \
               --output="logs/xsec_fixed_${case_name}_d${dist}_%j.out" \
               --error="logs/xsec_fixed_${case_name}_d${dist}_%j.err" \
               --wrap="export PATH=\"\$HOME/.pixi/bin:\$PATH\" && module load rocm/6.0.3 && export JAX_PLATFORMS=rocm && export XLA_FLAGS=\"--xla_gpu_enable_triton_softmax_fusion=false\" && cd /scratch/project_465002609/julian_clusters && pixi run -e rocm --manifest-path pyproject.toml python scripts/run_regret_cross_section.py --n-bearings 24 --distances-D ${dist} --n-inner-starts 300 --inner-max-iter 2000 --k-liberal 300 --deficit bastankhah --chunk-size 50 ${rose_args} --output-dir analysis/cross_section_fixed/${case_name}_d${dist}"
        count=$((count + 1))
    done
done

echo "${count} remaining cross-section jobs submitted on LUMI."

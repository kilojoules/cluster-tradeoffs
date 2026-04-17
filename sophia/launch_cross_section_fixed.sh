#!/bin/bash
# Corrected cross-sections on Sophia (CPU, 32 cores per node)
# 6 wind roses × 7 distances = 42 jobs
# Each job: 24 bearings × K=500 = 12,000 SGD starts on 32 cores
cd /work/users/juqu/cluster-tradeoffs

count=0

for case_args in \
    "dei --wind-rose dei --wind-speed 9.0 --n-bins 24" \
    "a0.5_f0.0 --wind-rose elliptical --ed-a 0.5 --ed-f 0.0 --wind-dir 270 --wind-speed 9.0 --n-bins 24" \
    "a0.5_f1.0 --wind-rose elliptical --ed-a 0.5 --ed-f 1.0 --wind-dir 270 --wind-speed 9.0 --n-bins 24" \
    "a0.7_f0.5 --wind-rose elliptical --ed-a 0.7 --ed-f 0.5 --wind-dir 270 --wind-speed 9.0 --n-bins 24" \
    "a0.9_f0.0 --wind-rose elliptical --ed-a 0.9 --ed-f 0.0 --wind-dir 270 --wind-speed 9.0 --n-bins 24" \
    "a0.9_f1.0 --wind-rose elliptical --ed-a 0.9 --ed-f 1.0 --wind-dir 270 --wind-speed 9.0 --n-bins 24"
do
    case_name=$(echo $case_args | cut -d' ' -f1)
    rose_args=$(echo $case_args | cut -d' ' -f2-)

    for dist in 2 5 10 15 20 30 40; do
        sbatch --job-name="xf-${case_name:0:6}-d${dist}" \
               --partition=windq,workq \
               --ntasks=1 --cpus-per-task=32 \
               --time=2-00:00:00 \
               --output="logs/xsec_fixed_${case_name}_d${dist}_%j.out" \
               --error="logs/xsec_fixed_${case_name}_d${dist}_%j.err" \
               --wrap="cd /work/users/juqu/cluster-tradeoffs && export JAX_PLATFORMS=cpu && pixi run python scripts/run_regret_cross_section.py --n-bearings 24 --distances-D ${dist} --n-inner-starts 500 --inner-max-iter 5000 --deficit bastankhah ${rose_args} --output-dir analysis/cross_section_fixed/${case_name}_d${dist}"
        count=$((count + 1))
    done
done

echo "${count} cross-section jobs submitted on Sophia."

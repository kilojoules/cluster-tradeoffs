#!/bin/bash
#BSUB -J fwsanity
#BSUB -q gpua10
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 02:00
#BSUB -u /dev/null
#BSUB -o /zhome/5c/f/180689/clusters_logs/fwsanity.%J.out
#BSUB -e /zhome/5c/f/180689/clusters_logs/fwsanity.%J.err

set -euo pipefail

WORK=/tmp/$LSB_JOBID
mkdir -p $WORK/pixi_cache $WORK/pip_cache $WORK/rattler $WORK/hf
export PIXI_CACHE_DIR=$WORK/pixi_cache
export RATTLER_CACHE_DIR=$WORK/rattler
export PIP_CACHE_DIR=$WORK/pip_cache
export HF_HOME=$WORK/hf
export TRANSFORMERS_CACHE=$WORK/hf
export PATH="$HOME/bin:$HOME/.pixi/bin:$PATH"

echo "[sanity] node=$(hostname)  work=$WORK  date=$(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

cd $WORK
git clone --depth 1 -b gbar-funwake https://github.com/kilojoules/cluster-tradeoffs.git repo
cd repo

export PIXI_NO_PROGRESS=true
mkdir -p $WORK/jax_cache $WORK/jax_autotune $WORK/tmp
export TMPDIR=$WORK/tmp
export JAX_COMPILATION_CACHE_DIR=$WORK/jax_cache
export XLA_FLAGS="--xla_gpu_per_fusion_autotune_cache_dir=$WORK/jax_autotune --xla_gpu_autotune_level=0"
pixi install -e cuda --manifest-path pyproject.toml
echo "[sanity] XLA_FLAGS=$XLA_FLAGS"
pixi run -e cuda --manifest-path pyproject.toml bash -c 'echo "[sanity] inside pixi run XLA_FLAGS=$XLA_FLAGS"; echo "[sanity] inside pixi run TMPDIR=$TMPDIR"; python -c "import jax; print(\"jax\", jax.__version__, \"devices\", jax.devices())"'

OUTDIR=analysis/sanity_funwake/buffer_table_a0.9_f1.0_d2_K50
pixi run -e cuda --manifest-path pyproject.toml python scripts/run_regret_cross_section.py \
    --n-bearings 4 --n-inner-starts 50 --inner-max-iter 5000 --k-liberal 50 \
    --deficit bastankhah --schedule funwake_iter192 --chunk-size 10 \
    --wind-speed 9.0 --n-bins 24 \
    --distances-D 2 --wind-rose elliptical --ed-a 0.9 --ed-f 1.0 --wind-dir 270 \
    --output-dir $OUTDIR
RC=$?

DEST=$HOME/clusters_results/$OUTDIR
mkdir -p $DEST
rsync -av $OUTDIR/ $DEST/

echo "[sanity] DONE  rc=$RC  result -> $DEST  date=$(date)"
exit $RC

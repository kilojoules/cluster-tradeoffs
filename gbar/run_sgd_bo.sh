#!/bin/bash
#BSUB -J sgdbo
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00
#BSUB -u /dev/null
#BSUB -o /zhome/5c/f/180689/clusters_logs/sgdbo.%J.out
#BSUB -e /zhome/5c/f/180689/clusters_logs/sgdbo.%J.err

set -euo pipefail

WORK=/tmp/$LSB_JOBID.sgdbo
mkdir -p $WORK/pixi_cache $WORK/pip_cache $WORK/rattler
export PIXI_CACHE_DIR=$WORK/pixi_cache
export RATTLER_CACHE_DIR=$WORK/rattler
export PIP_CACHE_DIR=$WORK/pip_cache
export PATH="$HOME/bin:$HOME/.pixi/bin:$PATH"

echo "[sgdbo] node=$(hostname) work=$WORK date=$(date)"
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
pixi run -e cuda --manifest-path pyproject.toml python -m ensurepip --upgrade || true
pixi run -e cuda --manifest-path pyproject.toml python -m pip install -q optuna
pixi run -e cuda --manifest-path pyproject.toml python -c "import jax, optuna; print('jax', jax.__version__, 'optuna', optuna.__version__, jax.devices())"

OUTDIR=analysis/sgd_bo_tuning/a0.9_f1.0
pixi run -e cuda --manifest-path pyproject.toml python scripts/run_sgd_bo_tuning.py \
    --wind-rose elliptical --ed-a 0.9 --ed-f 1.0 --wind-dir 270 \
    --deficit bastankhah --n-trials 80 --total-iter 5000 --n-holdout 8 \
    --output-dir $OUTDIR
RC=$?

DEST=$HOME/clusters_results/$OUTDIR
mkdir -p $DEST
rsync -av $OUTDIR/ $DEST/

echo "[sgdbo] DONE rc=$RC result -> $DEST date=$(date)"
exit $RC

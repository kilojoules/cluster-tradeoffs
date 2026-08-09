#!/bin/bash
#BSUB -J ring[1-30]%4
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00
#BSUB -u /dev/null
#BSUB -o /zhome/5c/f/180689/clusters_logs/ring.%J.%I.out
#BSUB -e /zhome/5c/f/180689/clusters_logs/ring.%J.%I.err

set -euo pipefail

WORK=/tmp/$LSB_JOBID.$LSB_JOBINDEX
mkdir -p $WORK/pixi_cache $WORK/pip_cache $WORK/rattler $WORK/hf
export PIXI_CACHE_DIR=$WORK/pixi_cache
export RATTLER_CACHE_DIR=$WORK/rattler
export PIP_CACHE_DIR=$WORK/pip_cache
export HF_HOME=$WORK/hf
export TRANSFORMERS_CACHE=$WORK/hf
export PATH="$HOME/bin:$HOME/.pixi/bin:$PATH"

CSV=$HOME/clusters_jobs/gbar_ring_jobs.csv
LINE=$((LSB_JOBINDEX + 1))
ROW=$(sed -n "${LINE}p" "$CSV" | tr -d '\r')
if [ -z "$ROW" ]; then
    echo "[ring] empty row at idx=$LSB_JOBINDEX" >&2
    exit 1
fi

STUDY=$(echo "$ROW"  | cut -d'|' -f2)
WAKE=$(echo  "$ROW"  | cut -d'|' -f3)
NAME=$(echo  "$ROW"  | cut -d'|' -f4)
OUTDIR=$(echo "$ROW" | cut -d'|' -f7)
PYARGS=$(echo "$ROW" | cut -d'|' -f10-)

echo "[ring] idx=$LSB_JOBINDEX study=$STUDY wake=$WAKE key=$NAME"
echo "[ring] outdir=$OUTDIR"
echo "[ring] args=$PYARGS"
echo "[ring] node=$(hostname) work=$WORK date=$(date)"
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
pixi run -e cuda --manifest-path pyproject.toml python -c "import jax; print('jax', jax.__version__, 'devices', jax.devices())"

pixi run -e cuda --manifest-path pyproject.toml python $PYARGS --output-dir $OUTDIR
RC=$?

DEST=$HOME/clusters_results/$OUTDIR
mkdir -p $DEST
rsync -av $OUTDIR/ $DEST/

echo "[ring] DONE rc=$RC result -> $DEST date=$(date)"
exit $RC

#!/bin/bash
# Runs optimization for all farms, keeping at most MAX_PARALLEL jobs at a time.
# Auto-resumes: the Python script skips already-completed seeds.

MAX_PARALLEL=${1:-2}
N_STARTS=${2:-500}
MAX_ITER=${3:-2000}
A=${4:-0.02}
OUTPUT_DIR="analysis/dei_A${A}"
LOG_DIR="${OUTPUT_DIR}/logs"

mkdir -p "$LOG_DIR"

# Farms still needing work
FARMS=(1 2 3 4 5 6 7 8 9)

declare -A RUNNING_PIDS  # farm -> pid

check_done() {
    # Remove finished PIDs from RUNNING_PIDS
    for farm in "${!RUNNING_PIDS[@]}"; do
        pid=${RUNNING_PIDS[$farm]}
        if ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid" 2>/dev/null
            echo "[$(date +%H:%M:%S)] Farm $farm (PID $pid) finished"
            unset RUNNING_PIDS[$farm]
        fi
    done
}

launch_farm() {
    local farm=$1
    # Check if farm already has enough starts
    local current
    current=$(pixi run python -c "
import h5py
with h5py.File('${OUTPUT_DIR}/layouts_farm${farm}.h5', 'r') as f:
    print(len([k for k in f.keys() if k.startswith('layout_')]) // 2)
" 2>/dev/null || echo "0")

    if [ "$current" -ge "$N_STARTS" ]; then
        echo "[$(date +%H:%M:%S)] Farm $farm already has $current/$N_STARTS starts, skipping"
        return 1
    fi

    echo "[$(date +%H:%M:%S)] Starting farm $farm ($current/$N_STARTS starts done)..."
    nohup pixi run python scripts/run_dei_single_neighbor.py \
        --n-starts=$N_STARTS --max-iter=$MAX_ITER --A=$A \
        --farm=$farm --skip-combined --seed-offset=0 \
        -o "$OUTPUT_DIR" \
        > "${LOG_DIR}/farm${farm}_auto.log" 2>&1 &
    RUNNING_PIDS[$farm]=$!
    return 0
}

echo "=== Farm Queue Manager ==="
echo "MAX_PARALLEL=$MAX_PARALLEL, N_STARTS=$N_STARTS, MAX_ITER=$MAX_ITER, A=$A"
echo ""

FARM_IDX=0

# Main loop
while true; do
    check_done

    # Launch new jobs if we have capacity
    while [ ${#RUNNING_PIDS[@]} -lt $MAX_PARALLEL ] && [ $FARM_IDX -lt ${#FARMS[@]} ]; do
        farm=${FARMS[$FARM_IDX]}
        FARM_IDX=$((FARM_IDX + 1))
        launch_farm "$farm" || true
    done

    # Exit if nothing running and nothing left to launch
    if [ ${#RUNNING_PIDS[@]} -eq 0 ] && [ $FARM_IDX -ge ${#FARMS[@]} ]; then
        echo ""
        echo "[$(date +%H:%M:%S)] All farms complete!"
        break
    fi

    # Status update
    running_farms=$(echo "${!RUNNING_PIDS[@]}" | tr ' ' ',')
    echo "[$(date +%H:%M:%S)] Running: farms {$running_farms} (${#RUNNING_PIDS[@]}/$MAX_PARALLEL slots)"

    sleep 60
done

# Final summary
echo ""
echo "=== Final Layout Counts ==="
pixi run python -c "
import h5py
for f in range(1,10):
    path = '${OUTPUT_DIR}/layouts_farm' + str(f) + '.h5'
    with h5py.File(path, 'r') as hf:
        n = len([k for k in hf.keys() if k.startswith('layout_')])
        print(f'Farm {f}: {n//2} starts')
"

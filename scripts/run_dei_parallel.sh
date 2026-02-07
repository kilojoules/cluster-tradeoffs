#!/bin/bash
# Run DEI individual farm analysis in parallel batches
# 50 starts, 2000 iterations per farm
# 3 farms per batch to stay within memory limits

OUTPUT_DIR="analysis/dei_50starts_2000iter"
N_STARTS=50
MAX_ITER=2000

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "DEI Individual Farm Analysis"
echo "N_STARTS=$N_STARTS, MAX_ITER=$MAX_ITER"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

run_batch() {
    local farms=("$@")
    echo ""
    echo "--- Starting batch: farms ${farms[*]} ---"

    pids=()
    for farm in "${farms[@]}"; do
        echo "Launching farm $farm..."
        pixi run python scripts/run_dei_single_neighbor.py \
            --n-starts=$N_STARTS \
            --max-iter=$MAX_ITER \
            --farm=$farm \
            --skip-combined \
            -o "$OUTPUT_DIR" \
            > "$OUTPUT_DIR/farm${farm}.log" 2>&1 &
        pids+=($!)
    done

    echo "Waiting for batch to complete (PIDs: ${pids[*]})..."
    for pid in "${pids[@]}"; do
        wait $pid
        echo "  PID $pid completed"
    done
    echo "--- Batch complete ---"
}

# Run in 3 batches of 3 farms each
START_TIME=$(date +%s)

run_batch 1 2 3
run_batch 4 5 6
run_batch 7 8 9

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "=========================================="
echo "All individual farms complete!"
echo "Total time: $((ELAPSED / 60)) minutes"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "NOTE: 'All neighbors combined' analysis still needs to be run separately."
echo "This will take ~3.4 hours. Run with:"
echo "  pixi run python scripts/run_dei_single_neighbor.py --n-starts=50 --max-iter=2000"
echo "=========================================="

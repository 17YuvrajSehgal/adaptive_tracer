#!/bin/bash
# lmat_sync_run.sh — LTTng tracing ON + LMAT running synchronously (worst case overhead)
#
# Starts three processes in parallel:
#   1. collect_trace.sh (LTTng kernel + UST)
#   2. load_generator.py (200 users)
#   3. online_inference.py --mode sync (LMAT inference blocks event loop)
#
# Usage: ./lmat_sync_run.sh <run_id> [duration_seconds]
#   e.g. ./lmat_sync_run.sh run01 300
set -e

RUN_ID=${1:-run01}
DURATION=${2:-300}
EXPERIMENT_DIR=~/experiments/lmat_sync/$RUN_ID
FRONTEND_HOST=${FRONTEND_HOST:-http://localhost:80}
LOAD_USERS=${LOAD_USERS:-200}

# ── Edit these paths before running on the GCP VM ────────────────────────────
MODEL_PATH=${MODEL_PATH:-~/adaptive_tracer/checkpoints/model_best_lstm.pt}
VOCAB_PATH=${VOCAB_PATH:-~/adaptive_tracer/micro-service-trace-data/preprocessed/vocab.pkl}
DELAY_PATH=${DELAY_PATH:-~/adaptive_tracer/micro-service-trace-data/preprocessed/delay_spans.pkl}
MODEL_TYPE=${MODEL_TYPE:-lstm}
# LSTM trained config (adjust if using transformer):
N_HIDDEN=${N_HIDDEN:-1024}; N_LAYER=${N_LAYER:-6}; N_HEAD=${N_HEAD:-8}
DIM_SYS=${DIM_SYS:-48};     DIM_ENTRY=${DIM_ENTRY:-12}; DIM_RET=${DIM_RET:-12}
DIM_PROC=${DIM_PROC:-48};   DIM_PID=${DIM_PID:-12};     DIM_TID=${DIM_TID:-12}
DIM_ORDER=${DIM_ORDER:-12}; DIM_TIME=${DIM_TIME:-12}
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p "$EXPERIMENT_DIR"/load_logs

echo "🚀 LMAT SYNC: $RUN_ID (${DURATION}s, ${LOAD_USERS} users)"

TRACE_DIR=~/traces/lmat_sync/$RUN_ID
RUN_START_EPOCH=$(date -u +%s)

# Process 1: LTTng collection
(cd ~ && ./collect_trace.sh lmat_sync "$RUN_ID" "$DURATION") &
TRACE_PID=$!

# Process 2: Load generator
python3 ~/load_generator.py \
    --host "$FRONTEND_HOST" \
    --users "$LOAD_USERS" \
    --duration "$DURATION" \
    --think-min 0.2 \
    --think-max 1.0 \
    --log-level WARNING \
    --output "$EXPERIMENT_DIR/load_results.csv" &
LOAD_PID=$!

# Give LTTng 3s to start writing before LMAT starts reading
sleep 10

# Process: periodic LTTng ring-buffer flush so babeltrace2 can read new data
# (without this, babeltrace2 exits at EOF immediately and gets no events)
(while true; do
    sudo lttng flush sockshop-kernel 2>/dev/null || true
    sleep 2
done) &
FLUSH_PID=$!

# Process 3: LMAT synchronous inference (reads live CTF as it grows)
python3 ~/adaptive_tracer/microservice-lttng-data-collection-scripts/online_inference.py \
    --model_path       "$MODEL_PATH" \
    --vocab_path       "$VOCAB_PATH" \
    --delay_spans_path "$DELAY_PATH" \
    --trace_dir        "$TRACE_DIR/kernel" \
    --model_type       "$MODEL_TYPE" \
    --n_hidden "$N_HIDDEN" --n_layer "$N_LAYER" --n_head "$N_HEAD" \
    --dim_sys "$DIM_SYS" --dim_entry "$DIM_ENTRY" --dim_ret "$DIM_RET" \
    --dim_proc "$DIM_PROC" --dim_pid "$DIM_PID" --dim_tid "$DIM_TID" \
    --dim_order "$DIM_ORDER" --dim_time "$DIM_TIME" \
    --mode sync \
    --window_ms 100 \
    --log_file "$EXPERIMENT_DIR/inference.log" &
INFER_PID=$!

# Wait for load and tracing to finish; then stop inference and flush loop
wait "$TRACE_PID" "$LOAD_PID"
kill "$INFER_PID"  2>/dev/null && wait "$INFER_PID"  2>/dev/null || true
kill "$FLUSH_PID" 2>/dev/null && wait "$FLUSH_PID" 2>/dev/null || true

RUN_END_EPOCH=$(date -u +%s)
sudo chown -R "$(whoami)" "$TRACE_DIR" 2>/dev/null || true


REQ_COUNT=$(tail -n +2 "$EXPERIMENT_DIR/load_results.csv" 2>/dev/null | wc -l || echo 0)
ELAPSED=$((RUN_END_EPOCH - RUN_START_EPOCH))

cat <<EOF

✅ LMAT SYNC $RUN_ID COMPLETE
📊 Requests   : $REQ_COUNT  (in ${ELAPSED}s)
📈 Throughput : $(echo "scale=1; $REQ_COUNT / $ELAPSED" | bc) req/s (approx)
🧠 Inference  : $EXPERIMENT_DIR/inference.log
📁 Output     : $EXPERIMENT_DIR

EOF

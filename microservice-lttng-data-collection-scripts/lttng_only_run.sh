#!/bin/bash
# lttng_only_run.sh — LTTng tracing ON, no LMAT model
# This is the fair comparison point: all the tracing cost, none of the inference cost.
#
# Usage: ./lttng_only_run.sh <run_id> [duration_seconds]
#   e.g. ./lttng_only_run.sh run01 300
set -e

RUN_ID=${1:-run01}
DURATION=${2:-300}
EXPERIMENT_DIR=~/experiments/lttng_only/$RUN_ID
FRONTEND_HOST=${FRONTEND_HOST:-http://localhost:80}
LOAD_USERS=${LOAD_USERS:-200}
QUIET_FLAG=${3:-}    # pass --quiet to silence the OTel span printing

mkdir -p "$EXPERIMENT_DIR"/load_logs

echo "🚀 LTTng ONLY (no LMAT): $RUN_ID (${DURATION}s, ${LOAD_USERS} users)"

RUN_START_EPOCH=$(date -u +%s)

# Fix traces dir permissions BEFORE collect_trace.sh tries to create it
TRACE_DIR=~/traces/lttng_only/$RUN_ID
sudo mkdir -p "$TRACE_DIR"/{kernel,ust} 2>/dev/null || true
sudo chown -R "$(whoami)" ~/traces/lttng_only 2>/dev/null || true

# ── Tracing (kernel + UST, same as normal runs) ──────────────────────────────
(~/adaptive_tracer/microservice-lttng-data-collection-scripts/collect_trace.sh lttng_only "$RUN_ID" "$DURATION" $QUIET_FLAG) &
TRACE_PID=$!

# ── Load generator ───────────────────────────────────────────────────────────
python3 ~/load_generator.py \
    --host "$FRONTEND_HOST" \
    --users "$LOAD_USERS" \
    --duration "$DURATION" \
    --think-min 0.2 \
    --think-max 1.0 \
    --log-level WARNING \
    --output "$EXPERIMENT_DIR/load_results.csv" &
LOAD_PID=$!

wait "$TRACE_PID" "$LOAD_PID"

RUN_END_EPOCH=$(date -u +%s)

# Fix perms on kernel subdirectory (written by sudo lttng)
sudo chown -R "$(whoami)" "$TRACE_DIR" 2>/dev/null || true


REQ_COUNT=$(tail -n +2 "$EXPERIMENT_DIR/load_results.csv" 2>/dev/null | wc -l || echo 0)
ELAPSED=$((RUN_END_EPOCH - RUN_START_EPOCH))

cat <<EOF

✅ LTTng-ONLY $RUN_ID COMPLETE
📊 Requests    : $REQ_COUNT  (in ${ELAPSED}s)
📈 Throughput  : $(echo "scale=1; $REQ_COUNT / $ELAPSED" | bc) req/s (approx)
📁 Output      : $EXPERIMENT_DIR

EOF

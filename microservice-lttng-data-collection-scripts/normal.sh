#!/bin/bash
set -e

RUN_ID=${1:-run_01}
DURATION=${2:-300}
EXPERIMENT_DIR=~/experiments/normal/$RUN_ID
FRONTEND_HOST=${FRONTEND_HOST:-http://localhost:80}
LOAD_USERS=${LOAD_USERS:-200}

mkdir -p "$EXPERIMENT_DIR"/{metrics,load_logs}

echo "🚀 Normal: $RUN_ID (${DURATION}s, ${LOAD_USERS} users)"

RUN_START_EPOCH=$(date -u +%s)

# Tracing
(cd ~ && ./collect_trace.sh normal "$RUN_ID" "$DURATION") &
TRACE_PID=$!

# Load
python3 ~/load_generator.py \
  --host "$FRONTEND_HOST" \
  --users "$LOAD_USERS" \
  --duration "$DURATION" \
  --think-min 0.2 \
  --think-max 1.0 \
  --log-level INFO \
  --output "$EXPERIMENT_DIR/load_results.csv" &
LOAD_PID=$!

wait "$TRACE_PID" "$LOAD_PID"

RUN_END_EPOCH=$(date -u +%s)

# Fix perms
TRACE_DIR=~/traces/normal/"$RUN_ID"
sudo chown -R "$(whoami)" "$TRACE_DIR" 2>/dev/null || true

# 30s pause
echo "⏸️  Prometheus flush (30s)..."
sleep 30

# Metrics (with buffer)
START_ISO=$(date -u -d "@$((RUN_START_EPOCH-30))" '+%Y-%m-%dT%H:%M:%SZ')
END_ISO=$(date -u -d "@$((RUN_END_EPOCH+30))" '+%Y-%m-%dT%H:%M:%SZ')
./download_metrics.sh "$START_ISO" "$END_ISO" "$EXPERIMENT_DIR/metrics"

# Summary
REQ_COUNT=$(tail -n +2 "$EXPERIMENT_DIR/load_results.csv" 2>/dev/null | wc -l || echo 0)
OTEL_SPANS=$(babeltrace "$TRACE_DIR" 2>/dev/null | grep -c "otel.spans" || echo 0)
BUSINESS_SPANS=$(babeltrace "$TRACE_DIR" 2>/dev/null | grep -c -i "cart\|orders" || echo 0)

cat << EOF

✅ $RUN_ID COMPLETE
📊 Requests: $REQ_COUNT
🔍 Spans: $OTEL_SPANS ($BUSINESS_SPANS business)
📈 Metrics: $(find "$EXPERIMENT_DIR/metrics" -type f | wc -l) files
💾 $(du -sh "$EXPERIMENT_DIR" 2>/dev/null | cut -f1)
💾 $(du -sh "$TRACE_DIR" 2>/dev/null | cut -f1)

EOF
